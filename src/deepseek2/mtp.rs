use super::*;
use crate::GenerationCacheContinuation;
use ruda_model::record::{PrecisionSettings, Record, RecorderError};
mod batching;
pub use batching::{DeepseekMtpBatchCache,DeepseekMtpBatchPrefix};
#[cfg(feature = "tensor-parallel")]
pub mod parallel;

struct Predictor<B: Backend> {
    embedding_norm: Tensor<B, 1>,
    hidden_norm: Tensor<B, 1>,
    projection: Projection<B>,
    decoder: Layer<B>,
    output_norm: Tensor<B, 1>,
}

pub struct DeepseekMtp<B: Backend> {
    pub target: DeepseekV2Model<B>,
    predictors: Vec<Predictor<B>>,
}

#[derive(Clone, Debug)]
pub struct DeepseekMtpCache<B: Backend> {
    layers: Vec<DeepseekV2Cache<B>>,
}

pub struct DeepseekMtpOutput<B: Backend> {
    pub logits: Tensor<B, 3>,
    pub hidden_states: Tensor<B, 3>,
}

pub struct LoadedDeepseekMtp<B: Backend> {
    pub model: DeepseekMtp<B>,
    pub eos_token_ids: Vec<i32>,
    pub report: HuggingFaceLoadReport,
}

pub fn load_huggingface_deepseek_mtp<B: Backend>(directory: impl AsRef<Path>, device: &B::Device)
    -> Result<LoadedDeepseekMtp<B>, HuggingFaceLoadError> {
    load(directory.as_ref(), device, None, None, None)
}

pub fn load_huggingface_mla_mtp<B: MlaFp8Backend>(directory: impl AsRef<Path>, device: &B::Device)
    -> Result<LoadedDeepseekMtp<B>, HuggingFaceLoadError> {
    load(directory.as_ref(), device, Some(weights::native_projection::<B>), None, None)
}

fn load<B: Backend>(directory: &Path, device: &B::Device, packed: Option<weights::Loader<B>>, shard:Option<ExpertShard>,
    reduction:Option<Arc<dyn Fn(Tensor<B,3>)->Tensor<B,3>+Send+Sync>>)
    -> Result<LoadedDeepseekMtp<B>, HuggingFaceLoadError> {
    let source: serde_json::Value = serde_json::from_slice(&std::fs::read(directory.join("config.json"))
        .map_err(|e| HuggingFaceLoadError(e.to_string()))?).map_err(|e| HuggingFaceLoadError(e.to_string()))?;
    let count = source.get("num_nextn_predict_layers").and_then(|v| v.as_u64())
        .and_then(|n| usize::try_from(n).ok()).filter(|&n| n > 0)
        .ok_or_else(|| HuggingFaceLoadError("DeepSeek MTP requires num_nextn_predict_layers > 0".into()))?;
    let target_shard=shard.as_ref().map(|s|ExpertShard{rank:s.rank,world:s.world});
    let loaded = load_rank_with(directory, device, target_shard, reduction, packed)?;
    let target = loaded.model;
    let config = &target.config;
    let dtype = target.embedding.dtype();
    let mut checkpoint = weights::Weights::open(directory, config, packed)?;
    let mut predictors = Vec::with_capacity(count);
    let mut excluded=BTreeSet::new();
    for depth in 0..count {
        let index = config.num_hidden_layers.checked_add(depth)
            .ok_or_else(|| HuggingFaceLoadError("MTP layer index overflow".into()))?;
        let prefix = format!("model.layers.{index}");
        let embedding_norm = checkpoint.tensor(&format!("{prefix}.enorm.weight"), [config.hidden_size], device)?;
        let hidden_norm = checkpoint.tensor(&format!("{prefix}.hnorm.weight"), [config.hidden_size], device)?;
        let output_norm = checkpoint.tensor(&format!("{prefix}.shared_head.norm.weight"), [config.hidden_size], device)?;
        if [embedding_norm.dtype(), hidden_norm.dtype(), output_norm.dtype()].iter().any(|&kind| kind != dtype) {
            return Err(HuggingFaceLoadError("MTP normalization storage differs from the target".into()));
        }
        let input = config.hidden_size.checked_mul(2).ok_or_else(|| HuggingFaceLoadError("MTP projection overflow".into()))?;
        let projection = projection(&mut checkpoint, &format!("{prefix}.eh_proj"), input, config.hidden_size, false, device, dtype)?;
        let decoder = load_layer(&mut checkpoint, config, index, shard.as_ref(), &mut excluded, device, dtype)?;
        checkpoint.confirm_shared_tensor(&format!("{prefix}.embed_tokens.weight"), "model.embed_tokens.weight")?;
        checkpoint.confirm_shared_tensor(&format!("{prefix}.shared_head.head.weight"), "lm_head.weight")?;
        checkpoint.confirm_shared_tensor(&format!("{prefix}.shared_head.head.weight_scale_inv"), "lm_head.weight_scale_inv")?;
        predictors.push(Predictor {embedding_norm, hidden_norm, projection, decoder, output_norm});
    }
    let unused = loaded.auxiliary_tensors.iter().filter(|name| !checkpoint.consumed.contains(*name) && !excluded.contains(*name)).cloned().collect::<Vec<_>>();
    if !unused.is_empty() {return Err(HuggingFaceLoadError(format!("unrepresented MTP tensors: {}", unused.join(", "))));}
    let report = HuggingFaceLoadReport {applied_tensors: loaded.report.applied_tensors + checkpoint.consumed.len(), ..loaded.report};
    Ok(LoadedDeepseekMtp {model: DeepseekMtp {target, predictors}, eos_token_ids: loaded.eos_token_ids, report})
}

impl<B: Backend> DeepseekMtp<B> {
    pub fn num_predictors(&self) -> usize {self.predictors.len()}
    pub fn with_attention_mode(mut self,mode:DeepseekV2AttentionMode)->Self {self.target.attention_mode=mode;self}

    pub fn new_mtp_cache(&self) -> DeepseekMtpCache<B> {
        let c = &self.target.config;
        DeepseekMtpCache {layers: (0..self.predictors.len()).map(|_| DeepseekV2Cache::new(1,
            c.max_position_embeddings, c.kv_lora_rank + c.qk_rope_head_dim)).collect()}
    }

    /// previous_hidden_states is aligned with the supplied token positions:
    /// target-final-normalized (depth 0), or preceding MTP-final-normalized (depth > 0).
    pub fn forward_mtp(&self, depth: usize, tokens: Tensor<B, 2, Int>, previous_hidden_states: Tensor<B, 3>,
        cache: &mut DeepseekMtpCache<B>) -> Result<DeepseekMtpOutput<B>, GenerationError> {
        self.forward_aligned(depth, tokens, previous_hidden_states, cache, 0, true)
    }

    fn forward_aligned(&self, depth: usize, tokens: Tensor<B, 2, Int>, previous_hidden_states: Tensor<B, 3>,
        cache: &mut DeepseekMtpCache<B>, position_offset: usize, zero_initial_embedding: bool)
        -> Result<DeepseekMtpOutput<B>, GenerationError> {
        let hidden_states=self.forward_hidden_aligned(depth,tokens,previous_hidden_states,cache,position_offset,zero_initial_embedding)?;
        Ok(DeepseekMtpOutput {logits:self.target.head.forward(hidden_states.clone()),hidden_states})
    }

    fn forward_hidden_aligned(&self, depth: usize, tokens: Tensor<B, 2, Int>, previous_hidden_states: Tensor<B, 3>,
        cache: &mut DeepseekMtpCache<B>, position_offset: usize, zero_initial_embedding: bool)
        -> Result<Tensor<B,3>, GenerationError> {
        let c = &self.target.config;
        let predictor = self.predictors.get(depth).ok_or_else(|| GenerationError("MTP predictor index out of range".into()))?;
        if cache.layers.len() != self.predictors.len() {return Err(GenerationError("MTP cache layer count differs".into()));}
        let state = &mut cache.layers[depth];
        let [batch, sequence] = tokens.dims();
        let start = state.position();
        let end = start.checked_add(sequence).filter(|&n| n <= c.max_position_embeddings)
            .ok_or_else(|| GenerationError("MTP context limit exceeded".into()))?;
        let rotary_start = start.checked_add(position_offset).filter(|&n| n.checked_add(sequence)
            .is_some_and(|end| end <= c.max_position_embeddings))
            .ok_or_else(|| GenerationError("MTP shifted positions exceed context".into()))?;
        let device = self.target.embedding.device();
        let dtype = self.target.embedding.dtype();
        if batch != 1 || sequence == 0 || tokens.device() != device || previous_hidden_states.device() != device
            || previous_hidden_states.dtype() != dtype || previous_hidden_states.dims() != [1, sequence, c.hidden_size]
            || !state.compatible(1, c.max_position_embeddings, c.kv_lora_rank + c.qk_rope_head_dim, &device, dtype) {
            return Err(GenerationError("MTP token/hidden/cache layout differs".into()));
        }
        let mut embedding = self.target.embedding.clone().select(0, tokens.reshape([sequence])).reshape([1, sequence, c.hidden_size]);
        if zero_initial_embedding && rotary_start == 0 {
            embedding = embedding.slice_assign([0..1, 0..1, 0..c.hidden_size], Tensor::zeros([1, 1, c.hidden_size], (&device, dtype)));
        }
        let merged = Tensor::cat(vec![norm(embedding, &predictor.embedding_norm, c.rms_norm_eps),
            norm(previous_hidden_states, &predictor.hidden_norm, c.rms_norm_eps)], 2);
        let input = predictor.projection.forward(merged);
        let (cos, sin, scale) = self.target.rotary(rotary_start, sequence, dtype, &device);
        let output = self.target.layer_forward(&predictor.decoder, input, &mut state.layers[0], start, cos, sin, scale)?;
        state.position = end;
        Ok(norm(output, &predictor.output_norm, c.rms_norm_eps))
    }

    /// Each depth consumes the next true token and predicts one further token.
    /// Outputs contain only rows with real labels; no fabricated leading KV rows.
    pub fn teacher_forced(&self, tokens: Tensor<B, 2, Int>, target_hidden: Tensor<B, 3>)
        -> Result<Vec<(DeepseekMtpOutput<B>, Tensor<B, 2, Int>)>, GenerationError> {
        let [batch, sequence] = tokens.dims();
        let hidden = self.target.config.hidden_size;
        if batch != 1 || sequence == 0 || target_hidden.dims() != [1, sequence, hidden] {
            return Err(GenerationError("MTP teacher inputs must share a nonempty target prefix".into()));
        }
        let mut cache = self.new_mtp_cache();
        let mut upstream = target_hidden;
        let mut outputs = Vec::with_capacity(self.predictors.len());
        for depth in 0..self.predictors.len() {
            if depth + 2 >= sequence { break; }
            let count = sequence - depth - 1;
            let input = tokens.clone().slice([0..1, depth + 1..sequence]);
            let previous = upstream.slice([0..1, 0..count, 0..hidden]);
            let output = self.forward_aligned(depth, input, previous, &mut cache, depth + 1, false)?;
            upstream = output.hidden_states.clone();
            outputs.push((DeepseekMtpOutput {
                logits: output.logits.slice([0..1, 0..count - 1, 0..self.target.config.vocab_size]),
                hidden_states: output.hidden_states.slice([0..1, 0..count - 1, 0..hidden]),
            }, tokens.clone().slice([0..1, depth + 2..sequence])));
        }
        Ok(outputs)
    }
}

impl<B: Backend> DeepseekMtpCache<B> {
    pub fn positions(&self) -> Vec<usize> {self.layers.iter().map(DeepseekV2Cache::position).collect()}
    pub fn fork(&self) -> Self {self.clone()}
}

#[derive(Clone, Debug)]
struct Verified<B: Backend> {
    tokens: Vec<i32>,
    logits: Tensor<B, 3>,
    hidden: Tensor<B, 3>,
    next: usize,
}

#[derive(Clone, Debug)]
pub struct DeepseekMtpGenerationCache<B: Backend> {
    target: DeepseekV2Cache<B>,
    mtp: DeepseekMtpCache<B>,
    position: usize,
    tails: Vec<Option<Tensor<B, 3>>>,
    verified: Option<Verified<B>>,
    hidden_width: usize,
}

impl<B: Backend> DeepseekMtpGenerationCache<B> {
    pub fn position(&self) -> usize { self.position }
    pub fn fork(&self) -> Self { self.clone() }
}

impl<B: Backend> DeepseekMtp<B> {
    fn consume_confirmed(&self, tokens: Tensor<B, 2, Int>, hidden: Tensor<B, 3>,
        cache: &mut DeepseekMtpGenerationCache<B>) -> Result<(), GenerationError> {
        let start = cache.position;
        let end = start + tokens.dims()[1];
        let width = self.target.config.hidden_size;
        let mut upstream_start = start;
        let mut upstream_last = cache.tails[0].take();
        cache.tails[0] = Some(hidden.clone().slice([0..1, end - start - 1..end - start, 0..width]));
        let mut upstream = hidden;
        for depth in 0..self.predictors.len() {
            let first = start.max(depth + 1);
            if first >= end { break; }
            let count = end - first;
            let previous = if first == start {
                let previous = upstream_last.ok_or_else(|| GenerationError("missing previous MTP hidden state".into()))?;
                if count == 1 { previous } else {
                    Tensor::cat(vec![previous, upstream.clone().slice([0..1, 0..count - 1, 0..width])], 1)
                }
            } else {
                upstream.clone().slice([0..1, first - 1 - upstream_start..end - 1 - upstream_start, 0..width])
            };
            if cache.mtp.layers[depth].position() != first - depth - 1 {
                return Err(GenerationError("MTP confirmed prefix positions differ".into()));
            }
            let output = self.forward_hidden_aligned(depth, tokens.clone().slice([0..1, first - start..end - start]),
                previous, &mut cache.mtp, depth + 1, false)?;
            upstream_last = cache.tails[depth + 1].take();
            cache.tails[depth + 1] = Some(output.clone().slice([0..1, count - 1..count, 0..width]));
            upstream = output;
            upstream_start = first;
        }
        cache.position = end;
        Ok(())
    }

    fn propose(&self, anchor: i32, cache: &DeepseekMtpGenerationCache<B>, device: &B::Device)
        -> Result<Vec<i32>, GenerationError> {
        let mut states = cache.mtp.fork();
        let mut tails = cache.tails.clone();
        let mut span = vec![anchor];
        let available = (self.target.config.max_position_embeddings - cache.position - 1).min(self.predictors.len());
        for step in 0..available {
            let previous = tails.clone();
            let mut selected = None;
            for depth in step..self.predictors.len() {
                let hidden = previous[depth].clone().ok_or_else(|| GenerationError("MTP draft prefix not primed".into()))?;
                let output = self.forward_hidden_aligned(depth, Tensor::from_data([[span[step]]], device), hidden,
                    &mut states, depth + 1, false)?;
                if depth == step { selected = Some(self.target.greedy_token(self.target.head.forward(output.clone()))?); }
                tails[depth + 1] = Some(output);
            }
            span.push(selected.ok_or_else(|| GenerationError("MTP proposal produced no token".into()))?);
        }
        Ok(span)
    }
}

impl<B: Backend> CausalModel<B> for DeepseekMtp<B> {
    type Cache = DeepseekMtpGenerationCache<B>;
    fn new_cache(&self) -> Self::Cache {
        DeepseekMtpGenerationCache { target: self.target.new_cache(), mtp: self.new_mtp_cache(), position: 0,
            tails: (0..=self.predictors.len()).map(|_| None).collect(), verified: None, hidden_width: self.target.config.hidden_size }
    }
    fn forward_cached_last(&self, tokens: Tensor<B, 2, Int>, cache: &mut Self::Cache) -> Tensor<B, 3> {
        self.try_forward_cached_last(tokens, cache).expect("validated MTP generation input")
    }
    fn try_forward_cached_last(&self, tokens: Tensor<B, 2, Int>, cache: &mut Self::Cache)
        -> Result<Tensor<B, 3>, GenerationError> {
        let [batch, count] = tokens.dims();
        let c = &self.target.config;
        if batch != 1 || count == 0 || tokens.device() != self.target.embedding.device()
            || cache.tails.len() != self.predictors.len() + 1 || cache.mtp.layers.len() != self.predictors.len() || cache.hidden_width != c.hidden_size
            || cache.position.checked_add(count).is_none_or(|end| end > c.max_position_embeddings) {
            return Err(GenerationError("incompatible MTP generation request".into()));
        }
        let input = tokens.clone().try_into_data().map_err(|e| GenerationError(e.to_string()))?
            .iter::<i32>().collect::<Vec<_>>();
        if input.iter().any(|&token| token < 0 || token as usize >= c.vocab_size) {
            return Err(GenerationError("invalid MTP input token".into()));
        }
        if count == 1 {
            if let Some(mut verified) = cache.verified.take() {
                if verified.tokens.get(verified.next) == input.first() {
                    let row = verified.next;
                    let hidden = verified.hidden.clone().slice([0..1, row..row + 1, 0..c.hidden_size]);
                    let logits = verified.logits.clone().slice([0..1, row..row + 1, 0..c.vocab_size]);
                    self.consume_confirmed(tokens, hidden, cache)?;
                    verified.next += 1;
                    if verified.next < verified.tokens.len() { cache.verified = Some(verified); }
                    return Ok(logits);
                }
            }
        }
        cache.verified = None;
        cache.target.truncate(cache.position)?;
        let ready = count == 1 && cache.tails.iter().all(Option::is_some)
            && cache.position + 1 < c.max_position_embeddings;
        if ready {
            let span = self.propose(input[0], cache, &tokens.device())?;
            let block = Tensor::from_data(TensorData::new(span.clone(), [1, span.len()]), &tokens.device());
            let (logits, hidden) = self.target.forward_features(block, &mut cache.target, false)?;
            let first = logits.clone().slice([0..1, 0..1, 0..c.vocab_size]);
            self.consume_confirmed(tokens, hidden.clone().slice([0..1, 0..1, 0..c.hidden_size]), cache)?;
            if span.len() > 1 { cache.verified = Some(Verified { tokens: span, logits, hidden, next: 1 }); }
            Ok(first)
        } else {
            let (logits, hidden) = self.target.forward_features_impl(tokens.clone(), &mut cache.target, true, true)?;
            self.consume_confirmed(tokens, hidden, cache)?;
            Ok(logits)
        }
    }
}

pub struct DeepseekMtpGenerationCacheRecord<B: Backend> {
    version: u32,
    model_id: String,
    target: DeepseekV2CacheRecord<B>,
    mtp: Vec<DeepseekV2CacheRecord<B>>,
    position: usize,
    hidden_width: usize,
    tails: Vec<Option<TensorData>>,
}

impl<B: Backend> Record<B> for DeepseekMtpGenerationCacheRecord<B> {
    type Item<S: PrecisionSettings> = (u32, String, <DeepseekV2CacheRecord<B> as Record<B>>::Item<S>,
        Vec<<DeepseekV2CacheRecord<B> as Record<B>>::Item<S>>, usize, usize, Vec<Option<TensorData>>);
    fn into_item<S: PrecisionSettings>(self) -> Self::Item<S> {
        (self.version, self.model_id, self.target.into_item::<S>(), self.mtp.into_iter().map(|cache| cache.into_item::<S>()).collect(),
            self.position, self.hidden_width, self.tails)
    }
    fn from_item<S: PrecisionSettings>(item: Self::Item<S>, device: &B::Device) -> Self {
        Self { version: item.0, model_id: item.1, target: DeepseekV2CacheRecord::from_item::<S>(item.2, device),
            mtp: item.3.into_iter().map(|cache| DeepseekV2CacheRecord::from_item::<S>(cache, device)).collect(),
            position: item.4, hidden_width: item.5, tails: item.6 }
    }
}

fn invalid(error: impl std::fmt::Display) -> RecorderError { RecorderError::Unknown(format!("invalid MTP continuation: {error}")) }

fn check_tails(tails: &[Option<TensorData>], position: usize, width: usize) -> Result<(), RecorderError> {
    let mut dtype = None;
    if width == 0 { return Err(invalid("empty hidden width")); }
    for (depth, value) in tails.iter().enumerate() {
        match value {
            Some(data) if position > depth => {
                if data.shape.as_slice() != [1, 1, width] || !matches!(data.dtype, DType::F32 | DType::F16 | DType::BF16)
                    || data.as_bytes().len() != width * data.dtype.size() || dtype.is_some_and(|kind| kind != data.dtype) {
                    return Err(invalid("hidden tail geometry or dtype differs"));
                }
                dtype = Some(data.dtype);
            }
            None if position <= depth => (),
            _ => return Err(invalid("hidden tail does not match committed position")),
        }
    }
    Ok(())
}

impl<B: Backend> GenerationCacheContinuation<B> for DeepseekMtpGenerationCache<B> {
    type Record = DeepseekMtpGenerationCacheRecord<B>;
    fn generation_position(&self) -> usize { self.position }
    fn capture_generation(&self, model_id: &str, device: &B::Device) -> Result<Self::Record, RecorderError> {
        if model_id.is_empty() || self.mtp.layers.is_empty() || self.tails.len() != self.mtp.layers.len() + 1 {
            return Err(invalid("missing model identity or predictor state"));
        }
        let mut target = self.target.clone();
        target.truncate(self.position).map_err(invalid)?;
        let target = target.capture_generation(model_id, device)?;
        let mut mtp = Vec::with_capacity(self.mtp.layers.len());
        for (depth, layer) in self.mtp.layers.iter().enumerate() {
            if layer.position() != self.position.saturating_sub(depth + 1) { return Err(invalid("predictor prefix position differs")); }
            mtp.push(layer.capture_generation(&format!("{model_id}:mtp:{depth}"), device)?);
        }
        let tails = self.tails.iter().map(|value| value.as_ref().map(|tensor| {
            if tensor.device() != *device { return Err(invalid("hidden tail device differs")); }
            tensor.clone().try_into_data().map_err(invalid)
        }).transpose()).collect::<Result<Vec<_>, _>>()?;
        check_tails(&tails, self.position, self.hidden_width)?;
        Ok(DeepseekMtpGenerationCacheRecord { version: 1, model_id: model_id.into(), target, mtp,
            position: self.position, hidden_width: self.hidden_width, tails })
    }
    fn restore_generation(record: Self::Record, model_id: &str, template: &Self, position: usize, device: &B::Device)
        -> Result<Self, RecorderError> {
        if record.version != 1 || model_id.is_empty() || record.model_id != model_id || record.position != position
            || record.hidden_width != template.hidden_width || record.mtp.len() != template.mtp.layers.len()
            || record.tails.len() != record.mtp.len() + 1 || record.mtp.is_empty() { return Err(invalid("identity, position or predictor geometry differs")); }
        check_tails(&record.tails, position, record.hidden_width)?;
        let target = DeepseekV2Cache::restore_generation(record.target, model_id, &template.target, position, device)?;
        let layers = record.mtp.into_iter().zip(&template.mtp.layers).enumerate().map(|(depth, (cache, template))| {
            DeepseekV2Cache::restore_generation(cache, &format!("{model_id}:mtp:{depth}"), template, position.saturating_sub(depth + 1), device)
        }).collect::<Result<Vec<_>, _>>()?;
        let tails = record.tails.into_iter().map(|data| data.map(|data| {
            let dtype = data.dtype; Tensor::from_data(data, (device, dtype))
        })).collect();
        Ok(Self { target, mtp: DeepseekMtpCache { layers }, position, tails, verified: None, hidden_width: record.hidden_width })
    }
}
