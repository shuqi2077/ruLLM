use crate::{BatchSamplingState, CausalModelLimits, ContinuousBatchScheduler, DeviceBatchModel,
    GenerationError, KvPageId, PagedKvAppendReservation, PagedKvCacheConfig, PrefixBatchModel,
    RequestId, ScheduledBatch, ScheduledBatchKind, TokenSampler};
use ruda_tensor::api::{backend::Backend, FloatDType, Tensor};
use std::{collections::BTreeMap, marker::PhantomData, sync::Arc};
mod models;
pub use models::DenseCausalBatchModel;
mod hybrid;
pub use hybrid::{CausalBatchModel,CausalBatchCache,CausalBatchPrefix,CausalBatchDelta};

/// A block forward plus exact token-wise semantic-state commits. Hybrid models
/// must include convolution/recurrent state in Delta, not just attention KV.
pub trait SpeculativeBatchModel<B: Backend>: PrefixBatchModel<B> {
    type Delta: Clone;
    fn begin_batch_speculation(&self, _cache: &mut Self::Cache, _batch: &ScheduledBatch) -> Result<(), GenerationError> { Ok(()) }
    fn forward_batch_block(&self, batch: &ScheduledBatch, cache: &mut Self::Cache,
        device: &B::Device) -> Result<Tensor<B, 3>, GenerationError>;
    fn capture_batch_delta(&self, cache: &Self::Cache, id: RequestId, start: usize,
        end: usize) -> Result<Self::Delta, GenerationError>;
    fn truncate_batch_cache(&self, cache: &mut Self::Cache, id: RequestId,
        position: usize) -> Result<(), GenerationError>;
    fn append_batch_delta(&self, cache: &mut Self::Cache, id: RequestId, table: &[KvPageId],
        delta: &Self::Delta, offset: usize) -> Result<(), GenerationError>;
}

#[derive(Clone, Debug)]
pub struct CausalDraftBatchConfig {
    pub draft_tokens: usize,
    pub draft_model_id: String,
    /// Exact identities of token-ID mappings. Vocabulary size is not sufficient.
    pub target_tokenizer_id: String,
    pub draft_tokenizer_id: String,
}

pub struct CausalDraftBatch<B: Backend, T: SpeculativeBatchModel<B>, D: SpeculativeBatchModel<B>> {
    target: T,
    draft: D,
    config: CausalDraftBatchConfig,
    _backend: PhantomData<fn() -> B>,
}

#[derive(Clone)]
struct Verified<B: Backend, TD: Clone, DD: Clone> {
    candidates: Arc<[i32]>,
    probabilities: Arc<[Vec<f64>]>,
    logits: Tensor<B, 3>,
    target: TD,
    draft: DD,
    cursor: usize,
    accepted: bool,
}

pub struct CausalDraftBatchCache<B: Backend, T: SpeculativeBatchModel<B>, D: SpeculativeBatchModel<B>> {
    target: T::Cache,
    draft: D::Cache,
    verified: BTreeMap<RequestId, Verified<B, T::Delta, D::Delta>>,
    config: PagedKvCacheConfig,
}

#[derive(Clone)]
pub struct CausalDraftBatchPrefix<T: Clone, D: Clone> { target: T, draft: D }

fn error(message: impl ToString) -> GenerationError { GenerationError(message.to_string()) }
fn greedy(values: &[f32]) -> Result<i32, GenerationError> {
    let mut best = None;
    for (token, &value) in values.iter().enumerate() {
        if !value.is_nan() && best.is_none_or(|(_, maximum)| value > maximum) { best = Some((token, value)); }
    }
    i32::try_from(best.ok_or_else(|| error("all draft/target logits are NaN"))?.0).map_err(error)
}
fn read<B: Backend>(logits: Tensor<B, 3>, batch: usize, steps: usize, vocabulary: usize)
    -> Result<Vec<f32>, GenerationError> {
    if logits.dims() != [batch, steps, vocabulary] { return Err(error("causal draft logit geometry differs")); }
    logits.cast(FloatDType::F32).try_into_data().map_err(error)?.to_vec::<f32>().map_err(error)
}

impl<B: Backend, T: SpeculativeBatchModel<B>, D: SpeculativeBatchModel<B>> CausalDraftBatch<B, T, D> {
    pub fn new(target: T, draft: D, config: CausalDraftBatchConfig) -> Result<Self, GenerationError> {
        let t = target.batch_limits(); let d = draft.batch_limits();
        if config.draft_tokens == 0 || config.draft_model_id.is_empty() || config.target_tokenizer_id.is_empty()
            || config.target_tokenizer_id != config.draft_tokenizer_id || t.vocab_size == 0
            || t.vocab_size != d.vocab_size || t.vocab_size > i32::MAX as usize
            || t.max_sequence_length == 0 || d.max_sequence_length == 0 {
            return Err(error("causal draft requires positive limits, matching token-ID identities/vocabularies and an exact draft identity"));
        }
        Ok(Self { target, draft, config, _backend: PhantomData })
    }

    fn ordinary(&self, batch: &ScheduledBatch, cache: &mut CausalDraftBatchCache<B, T, D>, device: &B::Device)
        -> Result<Tensor<B, 3>, GenerationError> {
        let logits = self.target.forward_batch(batch, &mut cache.target, device)?;
        let draft = self.draft.forward_batch(batch, &mut cache.draft, device)?;
        let expected = [batch.batch_size(), 1, self.batch_limits().vocab_size];
        if logits.dims() != expected || draft.dims() != expected { return Err(error("causal draft ordinary geometry differs")); }
        Ok(logits)
    }

    fn validate_prefix(&self, prefix: &CausalDraftBatchPrefix<T::Prefix, D::Prefix>) -> Result<(), GenerationError> {
        let position = self.target.prefix_position(&prefix.target);
        if position == 0 || position != self.draft.prefix_position(&prefix.draft)
            || position > self.batch_limits().max_sequence_length { return Err(error("causal draft prefix positions differ")); }
        Ok(())
    }
}

impl<B: Backend, T: SpeculativeBatchModel<B>, D: SpeculativeBatchModel<B>> DeviceBatchModel<B> for CausalDraftBatch<B, T, D> {
    type Cache = CausalDraftBatchCache<B, T, D>;
    fn batch_limits(&self) -> CausalModelLimits {
        let t = self.target.batch_limits(); let d = self.draft.batch_limits();
        CausalModelLimits { vocab_size: t.vocab_size, max_sequence_length: t.max_sequence_length.min(d.max_sequence_length) }
    }
    fn new_batch_cache(&self, config: PagedKvCacheConfig) -> Self::Cache {
        Self::Cache { target: self.target.new_batch_cache(config), draft: self.draft.new_batch_cache(config),
            verified: BTreeMap::new(), config }
    }
    fn fork_batch_cache(&self, cache: &Self::Cache) -> Self::Cache {
        Self::Cache { target: self.target.fork_batch_cache(&cache.target), draft: self.draft.fork_batch_cache(&cache.draft),
            verified: cache.verified.clone(), config: cache.config }
    }
    fn retain_batch_cache(&self, cache: &mut Self::Cache, scheduler: &ContinuousBatchScheduler) {
        self.target.retain_batch_cache(&mut cache.target, scheduler);
        self.draft.retain_batch_cache(&mut cache.draft, scheduler);
        cache.verified.retain(|id, _| scheduler.kv_cache().contains_sequence(*id));
    }
    fn sync_batch(&self, device: &B::Device) -> Result<(), GenerationError> {
        let target = self.target.sync_batch(device); let draft = self.draft.sync_batch(device);
        target.and(draft)
    }
    fn lookahead_tokens(&self) -> usize { self.config.draft_tokens }
    fn forward_batch(&self, batch: &ScheduledBatch, cache: &mut Self::Cache, device: &B::Device)
        -> Result<Tensor<B, 3>, GenerationError> {
        let mut sampling = batch.sequences.iter().map(|row| (row.request_id, None)).collect();
        self.forward_batch_sampling(batch, &BTreeMap::new(), cache, device, &mut sampling)
    }
    fn forward_batch_sampling(&self, batch: &ScheduledBatch, lookahead: &BTreeMap<RequestId, PagedKvAppendReservation>,
        cache: &mut Self::Cache, device: &B::Device, sampling: &mut BatchSamplingState) -> Result<Tensor<B, 3>, GenerationError> {
        if batch.kind == ScheduledBatchKind::Prefill {
            for row in &batch.sequences { cache.verified.remove(&row.request_id); }
            return self.ordinary(batch, cache, device);
        }
        if batch.next_n() != 1 || batch.sequences.iter().any(|r| r.token_ids.len() != 1) {
            return Err(error("causal draft decode needs one committed anchor per row"));
        }
        let vocabulary = self.batch_limits().vocab_size;
        let mut output = (0..batch.batch_size()).map(|_| None).collect::<Vec<_>>();
        let mut groups = BTreeMap::<usize, Vec<usize>>::new();
        for (index, row) in batch.sequences.iter().enumerate() {
            if let Some(mut verified) = cache.verified.remove(&row.request_id) {
                if verified.accepted && verified.candidates.get(verified.cursor) == row.token_ids.first() {
                    self.target.append_batch_delta(&mut cache.target, row.request_id, &row.block_table, &verified.target, verified.cursor)?;
                    self.draft.append_batch_delta(&mut cache.draft, row.request_id, &row.block_table, &verified.draft, verified.cursor)?;
                    verified.cursor += 1; verified.accepted = false;
                    output[index] = Some(verified.logits.clone().slice([0..1, verified.cursor..verified.cursor + 1, 0..vocabulary]));
                    if verified.cursor < verified.candidates.len() { cache.verified.insert(row.request_id, verified); }
                    continue;
                }
            }
            let count = lookahead.get(&row.request_id).map_or(0, |r| r.token_count.saturating_sub(1))
                .min(self.config.draft_tokens).min(self.batch_limits().max_sequence_length.saturating_sub(row.context_length));
            groups.entry(count).or_default().push(index);
        }
        for (count, indices) in groups {
            let work = ScheduledBatch { id: batch.id, kind: batch.kind,
                sequences: indices.iter().map(|&i| batch.sequences[i].clone()).collect() };
            if count == 0 {
                let logits = self.ordinary(&work, cache, device)?;
                for (row, &index) in indices.iter().enumerate() {
                    output[index] = Some(logits.clone().slice([row..row + 1, 0..1, 0..vocabulary]));
                }
                continue;
            }
            self.draft.begin_batch_speculation(&mut cache.draft,&work)?;
            let mut draft_logits = self.draft.forward_batch_block(&work, &mut cache.draft, device)?;
            let mut candidates = vec![Vec::with_capacity(count); indices.len()];
            let mut probabilities = vec![Vec::with_capacity(count); indices.len()];
            for step in 0..count {
                let values = read(draft_logits, indices.len(), 1, vocabulary)?;
                let mut next = work.clone();
                for (row, sequence) in next.sequences.iter_mut().enumerate() {
                    let logits = &values[row * vocabulary..(row + 1) * vocabulary];
                    let sampler = sampling.get_mut(&sequence.request_id).ok_or_else(|| error("missing draft sampler transaction"))?;
                    let (token, distribution) = match sampler {
                        Some(sampler) => sampler.propose(logits)?,
                        None => (greedy(logits)?, Vec::new()),
                    };
                    candidates[row].push(token); probabilities[row].push(distribution);
                    sequence.start_position = work.sequences[row].context_length + step;
                    sequence.context_length = sequence.start_position + 1; sequence.token_ids = vec![token];
                    sequence.block_table = lookahead[&sequence.request_id].block_table[..sequence.context_length.div_ceil(cache.config.block_size)].to_vec();
                }
                draft_logits = self.draft.forward_batch_block(&next, &mut cache.draft, device)?;
            }
            let mut verify = work.clone();
            for (row, sequence) in verify.sequences.iter_mut().enumerate() {
                sequence.token_ids.extend_from_slice(&candidates[row]); sequence.context_length += count;
                sequence.block_table = lookahead[&sequence.request_id].block_table[..sequence.context_length.div_ceil(cache.config.block_size)].to_vec();
            }
            self.target.begin_batch_speculation(&mut cache.target,&verify)?;
            let logits = self.target.forward_batch_block(&verify, &mut cache.target, device)?;
            if logits.dims() != [indices.len(), count + 1, vocabulary] { return Err(error("causal draft target omitted block logits")); }
            for (row, ((tokens, probabilities), &index)) in candidates.into_iter().zip(probabilities).zip(&indices).enumerate() {
                let sequence = &verify.sequences[row]; let start = sequence.start_position + 1;
                let target = self.target.capture_batch_delta(&cache.target, sequence.request_id, start, sequence.context_length)?;
                let draft = self.draft.capture_batch_delta(&cache.draft, sequence.request_id, start, sequence.context_length)?;
                self.target.truncate_batch_cache(&mut cache.target, sequence.request_id, start)?;
                self.draft.truncate_batch_cache(&mut cache.draft, sequence.request_id, start)?;
                let row_logits = logits.clone().slice([row..row + 1, 0..count + 1, 0..vocabulary]);
                output[index] = Some(row_logits.clone().slice([0..1, 0..1, 0..vocabulary]));
                cache.verified.insert(sequence.request_id, Verified { candidates: tokens.into(), probabilities: probabilities.into(),
                    logits: row_logits, target, draft, cursor: 0, accepted: false });
            }
        }
        Ok(Tensor::cat(output.into_iter().map(|row| row.ok_or_else(|| error("missing causal draft output row")))
            .collect::<Result<Vec<_>, _>>()?, 0))
    }
    fn select_batch_token(&self, cache: &mut Self::Cache, id: RequestId, values: &[f32], sampler: &mut Option<TokenSampler>)
        -> Result<i32, GenerationError> {
        if let Some(verified) = cache.verified.get_mut(&id) {
            let candidate = verified.candidates[verified.cursor]; let q = &verified.probabilities[verified.cursor];
            let (token, accepted) = match sampler {
                Some(sampler) => sampler.verify_draft(candidate, q, values)?,
                None if q.is_empty() => { let token = greedy(values)?; (token, token == candidate) },
                None => return Err(error("causal draft sampling mode changed within a request")),
            };
            verified.accepted = accepted;
            if !accepted { cache.verified.remove(&id); }
            Ok(token)
        } else { match sampler { Some(sampler) => sampler.sample(values), None => greedy(values) } }
    }
}

impl<B: Backend, T: SpeculativeBatchModel<B>, D: SpeculativeBatchModel<B>> PrefixBatchModel<B> for CausalDraftBatch<B, T, D> {
    fn prefix_tensor_transfer(&self)->bool {self.target.prefix_tensor_transfer() && self.draft.prefix_tensor_transfer()}
    fn export_prefix_tensors(&self,prefix:&Self::Prefix,shard:crate::PrefixShard)->Result<crate::PrefixTensors<B>,GenerationError> {
        self.validate_prefix(prefix)?;
        let identity=serde_json::to_string(&("causal-draft-v1",&self.config.draft_model_id,&self.config.target_tokenizer_id)).map_err(error)?;
        let mut tensors=crate::PrefixTensors::new(identity,self.prefix_position(prefix));
        tensors.add_child("target",self.target.export_prefix_tensors(&prefix.target,shard)?)?;
        tensors.add_child("draft",self.draft.export_prefix_tensors(&prefix.draft,Default::default())?)?;
        Ok(tensors)
    }
    fn import_prefix_tensors(&self,tensors:&crate::PrefixTensors<B>,shard:crate::PrefixShard,device:&B::Device)->Result<Self::Prefix,GenerationError> {
        let identity=serde_json::to_string(&("causal-draft-v1",&self.config.draft_model_id,&self.config.target_tokenizer_id)).map_err(error)?;
        tensors.validate(&identity,self.batch_limits().max_sequence_length,0)?;
        if tensors.children()!=2 {return Err(error("causal draft tensor child count differs"));}
        let prefix=Self::Prefix{target:self.target.import_prefix_tensors(tensors.child("target")?,shard,device)?,
            draft:self.draft.import_prefix_tensors(tensors.child("draft")?,Default::default(),device)?};
        self.validate_prefix(&prefix)?;
        if self.prefix_position(&prefix)!=tensors.position {return Err(error("causal draft transfer position differs"));}
        Ok(prefix)
    }
    type Prefix = CausalDraftBatchPrefix<T::Prefix, D::Prefix>;
    fn capture_prefix(&self, cache: &Self::Cache, id: RequestId) -> Result<Self::Prefix, GenerationError> {
        let prefix = Self::Prefix { target: self.target.capture_prefix(&cache.target, id)?, draft: self.draft.capture_prefix(&cache.draft, id)? };
        self.validate_prefix(&prefix)?; Ok(prefix)
    }
    fn prefix_position(&self, prefix: &Self::Prefix) -> usize { self.target.prefix_position(&prefix.target) }
    fn prefix_bytes(&self, prefix: &Self::Prefix) -> usize {
        self.target.prefix_bytes(&prefix.target).saturating_add(self.draft.prefix_bytes(&prefix.draft))
    }
    fn restore_prefix(&self, cache: &mut Self::Cache, id: RequestId, table: &[KvPageId], prefix: &Self::Prefix) -> Result<(), GenerationError> {
        self.validate_prefix(prefix)?;
        self.target.restore_prefix(&mut cache.target, id, table, &prefix.target)?;
        self.draft.restore_prefix(&mut cache.draft, id, table, &prefix.draft)?;
        cache.verified.remove(&id); Ok(())
    }
    fn encode_prefix(&self, prefix: &Self::Prefix, model_id: &str) -> Result<Vec<u8>, GenerationError> {
        self.validate_prefix(prefix)?;
        let target = self.target.encode_prefix(&prefix.target, model_id)?;
        let draft = self.draft.encode_prefix(&prefix.draft, &self.config.draft_model_id)?;
        rmp_serde::to_vec(&(1u32, model_id, &self.config.draft_model_id, &self.config.target_tokenizer_id, target, draft)).map_err(error)
    }
    fn decode_prefix(&self, bytes: &[u8], model_id: &str, device: &B::Device) -> Result<Self::Prefix, GenerationError> {
        let (version, target_id, draft_id, tokenizer, target, draft): (u32, String, String, String, Vec<u8>, Vec<u8>) = rmp_serde::from_slice(bytes).map_err(error)?;
        if version != 1 || target_id != model_id || draft_id != self.config.draft_model_id || tokenizer != self.config.target_tokenizer_id {
            return Err(error("causal draft prefix model/tokenizer identity differs"));
        }
        let prefix = Self::Prefix { target: self.target.decode_prefix(&target, model_id, device)?,
            draft: self.draft.decode_prefix(&draft, &self.config.draft_model_id, device)? };
        self.validate_prefix(&prefix)?; Ok(prefix)
    }
}
