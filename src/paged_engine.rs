//! Device execution for [`ContinuousBatchScheduler`]: physical paged KV pools,
//! per-batch metadata upload, packed Llama/Qwen2 forward and token commit.

use crate::continuous_batch::ScheduledBatchKind;
use crate::llama::PackedLlamaGeometry;
use crate::paged_attention::PagedAttentionBatch;
use crate::{
    ContinuousBatchConfig, ContinuousBatchError, ContinuousBatchScheduler,
    ContinuousBatchSnapshot, FinishedGeneration, GenerationError, GreedyGenerationConfig,
    PackedLlamaForCausalLm, PagedKvCacheConfig, RequestId, SamplingGenerationConfig,
    ScheduledBatch, SpeculativeConfig, SamplingOptions,
};
use ruda::runtime::server::ComputeServer;
use ruda_kernel::tensor::RudaTensor;
use ruda_tensor::DeviceOps;
use ruda_tensor::api::{DType, Int, Tensor, TensorData};
use ruda_tensor_device::{BoolElement, DeviceBackend, DeviceRuntime, FloatElement, IntElement};

fn scheduler_error(error: ContinuousBatchError) -> GenerationError {
    GenerationError(error.0)
}

/// Physical K/V storage for every layer of one packed model.
pub struct PagedLlamaEngine<R: DeviceRuntime> {
    pools: Vec<(RudaTensor<R>, RudaTensor<R>)>,
    kv: PagedKvCacheConfig,
    geometry: PackedLlamaGeometry,
}

impl<R: DeviceRuntime> core::fmt::Debug for PagedLlamaEngine<R> {
    fn fmt(&self, formatter: &mut core::fmt::Formatter<'_>) -> core::fmt::Result {
        formatter
            .debug_struct("PagedLlamaEngine")
            .field("kv", &self.kv)
            .field("geometry", &self.geometry)
            .finish_non_exhaustive()
    }
}

impl<R> PagedLlamaEngine<R>
where
    R: DeviceRuntime,
    R::Server: ComputeServer,
    R::Device: DeviceOps,
{
    /// Allocate `[num_pages, kv_heads, block_size, head_dim]` K and V pools per
    /// layer in the model's dtype.
    pub fn new<F, I, BT>(
        model: &PackedLlamaForCausalLm<DeviceBackend<R, F, I, BT>>,
        kv: PagedKvCacheConfig,
        device: &R::Device,
    ) -> Result<Self, GenerationError>
    where
        F: FloatElement,
        I: IntElement,
        BT: BoolElement,
    {
        kv.validate()
            .map_err(|error| GenerationError(error.to_string()))?;
        let geometry = model.geometry();
        if !geometry.paged_rope {
            return Err(GenerationError("paged inference requires static full-head RoPE".into()));
        }
        if kv.max_sequence_length > geometry.max_sequence_length {
            return Err(GenerationError(format!(
                "paged KV max_sequence_length {} exceeds model capacity {}",
                kv.max_sequence_length, geometry.max_sequence_length
            )));
        }
        if !matches!(geometry.dtype, DType::F32 | DType::F16 | DType::BF16) {
            return Err(GenerationError(format!(
                "paged inference does not support model dtype {:?}",
                geometry.dtype
            )));
        }
        kv.num_pages
            .checked_mul(geometry.num_kv_heads)
            .and_then(|value| value.checked_mul(kv.block_size))
            .and_then(|value| value.checked_mul(geometry.head_dimension))
            .filter(|&value| value <= u32::MAX as usize)
            .ok_or_else(|| GenerationError("paged KV pool exceeds u32 indexing".into()))?;
        let shape = [
            kv.num_pages,
            geometry.num_kv_heads,
            kv.block_size,
            geometry.head_dimension,
        ];
        let pools = (0..geometry.num_layers)
            .map(|_| {
                let allocate = || {
                    Tensor::<DeviceBackend<R, F, I, BT>, 4>::empty(shape, (device, geometry.dtype))
                        .into_primitive()
                        .tensor()
                };
                (allocate(), allocate())
            })
            .collect();
        Ok(Self {
            pools,
            kv,
            geometry,
        })
    }

    pub const fn kv_config(&self) -> PagedKvCacheConfig {
        self.kv
    }

    /// Bytes held by all K/V pools.
    pub fn pool_bytes(&self) -> usize {
        let element = match self.geometry.dtype {
            DType::F32 => 4,
            _ => 2,
        };
        2 * self.geometry.num_layers
            * self.kv.num_pages
            * self.geometry.num_kv_heads
            * self.kv.block_size
            * self.geometry.head_dimension
            * element
    }

    /// Run one scheduled batch. Returns logits `[rows, next_n, vocab]` when
    /// `all_positions` (speculative verification), else `[rows, 1, vocab]`.
    pub fn execute_positions<F, I, BT>(
        &self,
        model: &PackedLlamaForCausalLm<DeviceBackend<R, F, I, BT>>,
        batch: &ScheduledBatch,
        all_positions: bool,
        device: &R::Device,
    ) -> Result<Tensor<DeviceBackend<R, F, I, BT>, 3>, GenerationError>
    where
        F: FloatElement,
        I: IntElement,
        BT: BoolElement,
    {
        let rows = batch.batch_size();
        let sequence = batch.next_n();
        if rows == 0 || sequence == 0 {
            return Err(GenerationError("scheduled batch is empty".into()));
        }
        let mut starts = Vec::with_capacity(rows);
        let mut contexts = Vec::with_capacity(rows);
        for row in &batch.sequences {
            if row.token_ids.len() != sequence {
                return Err(GenerationError(
                    "every row of a paged batch must append the same token count".into(),
                ));
            }
            if let Some(&token) = row
                .token_ids
                .iter()
                .find(|&&token| token < 0 || token as usize >= self.geometry.vocab_size)
            {
                return Err(GenerationError(format!(
                    "token {token} is outside vocabulary [0, {})",
                    self.geometry.vocab_size
                )));
            }
            let to_u32 = |value: usize| {
                u32::try_from(value)
                    .map_err(|_| GenerationError("paged position exceeds u32".into()))
            };
            starts.push(to_u32(row.start_position)?);
            contexts.push(to_u32(row.context_length)?);
        }
        let (table, stride) = batch.flattened_block_table().map_err(scheduler_error)?;
        if table.iter().any(|&page| page as usize >= self.kv.num_pages) {
            return Err(GenerationError("block table references a missing page".into()));
        }
        let (key_pool, _) = &self.pools[0];
        let metadata = PagedAttentionBatch::upload(
            &key_pool.client,
            &key_pool.device,
            &table,
            stride,
            &starts,
            &contexts,
            self.kv.block_size,
            sequence,
        )
        .map_err(GenerationError)?;
        let tokens = Tensor::<DeviceBackend<R, F, I, BT>, 2, Int>::from_data(
            TensorData::new(batch.token_matrix(), [rows, sequence]),
            device,
        );
        Ok(model.forward_paged_ruda(tokens, &self.pools, &metadata, all_positions))
    }

    /// Run one scheduled batch and return last-token logits `[rows, vocab]`.
    /// K/V for the batch's new tokens are written into their reserved slots.
    pub fn execute<F, I, BT>(
        &self,
        model: &PackedLlamaForCausalLm<DeviceBackend<R, F, I, BT>>,
        batch: &ScheduledBatch,
        device: &R::Device,
    ) -> Result<Tensor<DeviceBackend<R, F, I, BT>, 2>, GenerationError>
    where
        F: FloatElement,
        I: IntElement,
        BT: BoolElement,
    {
        let logits = self.execute_positions(model, batch, false, device)?;
        let [rows, _, vocabulary] = logits.dims();
        Ok(logits.reshape([rows, vocabulary]))
    }
}

/// Outcome of one scheduler step.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct BatchStepReport {
    pub batch_id: u64,
    pub kind: ScheduledBatchKind,
    pub rows: usize,
    /// Tokens appended to the KV cache by this step (rows × next_n).
    pub processed_tokens: usize,
    /// Tokens appended to each request, in scheduled row order: one per
    /// token-producing row (non-final prefill chunks produce none), or the
    /// accepted run of tokens for a speculative `Verify` row.
    pub selected: Vec<(RequestId, i32)>,
}

/// Continuous batching over one packed model: schedule → paged forward →
/// token selection → commit, with rollback of page reservations on failure.
#[derive(Debug)]
pub struct ContinuousBatchRunner<R: DeviceRuntime> {
    scheduler: ContinuousBatchScheduler,
    engine: PagedLlamaEngine<R>,
}

impl<R> ContinuousBatchRunner<R>
where
    R: DeviceRuntime,
    R::Server: ComputeServer,
    R::Device: DeviceOps,
{
    pub fn new<F, I, BT>(
        model: &PackedLlamaForCausalLm<DeviceBackend<R, F, I, BT>>,
        batching: ContinuousBatchConfig,
        kv: PagedKvCacheConfig,
        device: &R::Device,
    ) -> Result<Self, GenerationError>
    where
        F: FloatElement,
        I: IntElement,
        BT: BoolElement,
    {
        let engine = PagedLlamaEngine::new(model, kv, device)?;
        let scheduler = ContinuousBatchScheduler::new_paged(batching, kv).map_err(scheduler_error)?;
        Ok(Self { scheduler, engine })
    }

    pub fn submit(
        &mut self,
        prompt_token_ids: Vec<i32>,
        generation: GreedyGenerationConfig,
    ) -> Result<RequestId, GenerationError> {
        self.validate_prompt(&prompt_token_ids, &generation.eos_token_ids)?;
        self.scheduler
            .submit(prompt_token_ids, generation)
            .map_err(scheduler_error)
    }

    pub fn submit_sampled(
        &mut self,
        prompt_token_ids: Vec<i32>,
        generation: SamplingGenerationConfig,
    ) -> Result<RequestId, GenerationError> {
        self.validate_prompt(&prompt_token_ids, &generation.eos_token_ids)?;
        self.scheduler
            .submit_sampled(prompt_token_ids, generation)
            .map_err(scheduler_error)
    }

    pub fn submit_sampled_with_options(
        &mut self,
        prompt_token_ids: Vec<i32>,
        generation: SamplingGenerationConfig,
        options: SamplingOptions,
    ) -> Result<RequestId, GenerationError> {
        self.validate_prompt(&prompt_token_ids, &generation.eos_token_ids)?;
        self.scheduler.submit_sampled_with_options(prompt_token_ids, generation, options)
            .map_err(scheduler_error)
    }

    fn validate_prompt(&self, prompt: &[i32], eos: &[i32]) -> Result<(), GenerationError> {
        let vocabulary = self.engine.geometry.vocab_size;
        if let Some(token) = prompt
            .iter()
            .chain(eos)
            .find(|&&token| token < 0 || token as usize >= vocabulary)
        {
            return Err(GenerationError(format!(
                "token {token} is outside vocabulary [0, {vocabulary})"
            )));
        }
        Ok(())
    }

    pub fn scheduler(&self) -> &ContinuousBatchScheduler {
        &self.scheduler
    }

    /// Reuse KV pages for identical full-block prompt prefixes.
    pub fn set_prefix_caching(&mut self, enabled: bool) {
        self.scheduler.set_prefix_caching(enabled);
    }

    /// Prompt-lookup speculative decoding for greedy requests (`None` disables).
    pub fn set_speculative_decoding(
        &mut self,
        config: Option<SpeculativeConfig>,
    ) -> Result<(), GenerationError> {
        self.scheduler
            .set_speculative_decoding(config)
            .map_err(scheduler_error)
    }

    /// Drop a request that is not part of the in-flight batch.
    pub fn cancel(&mut self, request_id: RequestId) -> Result<Vec<i32>, GenerationError> {
        self.scheduler.cancel(request_id)
            .map(|request| request.map_or_else(Vec::new, |request| request.generated_token_ids))
            .map_err(scheduler_error)
    }

    pub fn engine(&self) -> &PagedLlamaEngine<R> {
        &self.engine
    }

    pub fn snapshot(&self) -> ContinuousBatchSnapshot {
        self.scheduler.snapshot()
    }

    pub fn pop_finished(&mut self) -> Option<FinishedGeneration> {
        self.scheduler.pop_finished()
    }

    /// Execute one scheduled batch. Returns `Ok(None)` when nothing is
    /// runnable. A failed device batch is rolled back before the error is
    /// returned, so the same requests can be scheduled again.
    pub fn step<F, I, BT>(
        &mut self,
        model: &PackedLlamaForCausalLm<DeviceBackend<R, F, I, BT>>,
        device: &R::Device,
    ) -> Result<Option<BatchStepReport>, GenerationError>
    where
        F: FloatElement,
        I: IntElement,
        BT: BoolElement,
    {
        let Some(batch) = self.scheduler.schedule().map_err(scheduler_error)? else {
            return Ok(None);
        };
        match self.run_batch(model, &batch, device) {
            Ok(selected) => {
                Ok(Some(BatchStepReport {
                    batch_id: batch.id,
                    kind: batch.kind,
                    rows: batch.batch_size(),
                    processed_tokens: batch.batch_size() * batch.next_n(),
                    selected,
                }))
            }
            Err(error) => {
                self.scheduler
                    .fail_batch(batch.id)
                    .map_err(scheduler_error)?;
                Err(error)
            }
        }
    }

    fn run_batch<F, I, BT>(
        &mut self,
        model: &PackedLlamaForCausalLm<DeviceBackend<R, F, I, BT>>,
        batch: &ScheduledBatch,
        device: &R::Device,
    ) -> Result<Vec<(RequestId, i32)>, GenerationError>
    where
        F: FloatElement,
        I: IntElement,
        BT: BoolElement,
    {
        let pair_rows = |tokens: Vec<i32>| -> Vec<(RequestId, i32)> {
            batch
                .sequences
                .iter()
                .zip(tokens)
                .filter(|(row, _)| row.samples_token)
                .map(|(row, token)| (row.request_id, token))
                .collect()
        };
        let read_indices = |indices: Tensor<DeviceBackend<R, F, I, BT>, 2, Int>| {
            indices
                .try_into_data()
                .map_err(|error| GenerationError(format!("cannot read paged argmax: {error}")))?
                .iter::<i64>()
                .map(|index| {
                    i32::try_from(index)
                        .map_err(|_| GenerationError("argmax index exceeds i32".into()))
                })
                .collect::<Result<Vec<_>, _>>()
        };
        if batch.kind == ScheduledBatchKind::Verify {
            let logits = self.engine.execute_positions(model, batch, true, device)?;
            let [rows, width, vocabulary] = logits.dims();
            if rows != batch.batch_size() || width != batch.next_n() {
                return Err(GenerationError("paged verify returned the wrong shape".into()));
            }
            let argmax = read_indices(logits.reshape([rows * width, vocabulary]).argmax(1))?;
            let appended = self
                .scheduler
                .complete_verify_batch(batch.id, &argmax)
                .map_err(scheduler_error)?;
            return Ok(appended
                .into_iter()
                .flat_map(|(id, tokens)| tokens.into_iter().map(move |token| (id, token)))
                .collect());
        }
        let logits = self.engine.execute(model, batch, device)?;
        let [rows, vocabulary] = logits.dims();
        if rows != batch.batch_size() {
            return Err(GenerationError("paged forward returned the wrong row count".into()));
        }
        if !self
            .scheduler
            .batch_requires_logits(batch.id)
            .map_err(scheduler_error)?
        {
            // All token-producing rows are greedy: read back one index per row
            // instead of `rows * vocabulary` floats.
            let argmax = read_indices(logits.argmax(1))?;
            return self
                .scheduler
                .complete_greedy_batch(batch.id, &argmax)
                .map(pair_rows)
                .map_err(scheduler_error);
        }
        let values = logits
            .cast(DType::F32)
            .try_into_data()
            .map_err(|error| GenerationError(format!("cannot read paged logits: {error}")))?
            .to_vec::<f32>()
            .map_err(|error| GenerationError(format!("cannot decode paged logits: {error}")))?;
        let selection = self
            .scheduler
            .select_batch_tokens(batch.id, &values, vocabulary)
            .map_err(scheduler_error)?;
        self.scheduler
            .complete_selected_batch(selection)
            .map(pair_rows)
            .map_err(scheduler_error)
    }

    /// Step until every submitted request has finished. Fails instead of
    /// spinning when requests remain but none can be scheduled (KV pages
    /// exhausted without preemption).
    pub fn run_until_idle<F, I, BT>(
        &mut self,
        model: &PackedLlamaForCausalLm<DeviceBackend<R, F, I, BT>>,
        device: &R::Device,
    ) -> Result<Vec<FinishedGeneration>, GenerationError>
    where
        F: FloatElement,
        I: IntElement,
        BT: BoolElement,
    {
        let mut finished = Vec::new();
        loop {
            let progressed = self.step(model, device)?.is_some();
            while let Some(done) = self.scheduler.pop_finished() {
                finished.push(done);
            }
            if progressed {
                continue;
            }
            let snapshot = self.scheduler.snapshot();
            if snapshot.pending_requests == 0 && snapshot.active_requests == 0 {
                return Ok(finished);
            }
            return Err(GenerationError(format!(
                "continuous batch stalled: {} pending and {} active requests, {} of {} KV pages free",
                snapshot.pending_requests,
                snapshot.active_requests,
                snapshot.free_kv_pages,
                snapshot.total_kv_pages
            )));
        }
    }
}

#[cfg(all(test, feature = "nvidia"))]
mod tests {
    use super::*;
    use crate::{LlamaConfig, LlamaForCausalLm, LlamaKvCache, generate_greedy_packed_ruda};
    use half::bf16;
    use ruda_tensor_device::cuda::{Cuda, CudaDevice};

    fn tiny_config() -> LlamaConfig {
        LlamaConfig {
            vocab_size: 64,
            d_model: 128,
            d_ff: 192,
            num_hidden_layers: 2,
            num_query_heads: 4,
            num_kv_heads: 2,
            max_sequence_length: 64,
            rms_norm_epsilon: 1.0e-6,
            rope_theta: 10_000.0,
        }
    }

    fn prompts() -> Vec<Vec<i32>> {
        vec![
            vec![3, 17, 9, 41, 5],
            vec![8, 8, 2, 60, 33],
            vec![1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19],
        ]
    }

    #[test]
    fn continuous_batch_runner_matches_single_sequence_greedy_f32() {
        type B = Cuda<f32, i32>;
        let device = CudaDevice::default();
        let config = tiny_config();
        let model = LlamaForCausalLm::<B>::init_qwen2(&config, &device)
            .unwrap()
            .into_packed_inference();
        let generation = GreedyGenerationConfig {
            max_new_tokens: 6,
            eos_token_ids: Vec::new(),
        };
        let mut all_prompts = prompts();
        // Shares the first 16 tokens (four full blocks) with the third prompt.
        all_prompts.push((1..=16).chain([40, 41, 42]).collect());
        let references: Vec<Vec<i32>> = all_prompts
            .iter()
            .map(|prompt| {
                generate_greedy_packed_ruda(&model, &config, prompt, &generation, &device)
                    .unwrap()
                    .generated_token_ids
            })
            .collect();
        // (max batch tokens, prefix caching, pages): plain, chunked prefill
        // with prefix reuse, and a pool small enough to force preemption.
        // The preemption case uses one-token pages: the first prefill wave
        // takes 29 of 30 pages, so after one decode step no row can grow and
        // the scheduler must preempt (decode otherwise just skips rows that
        // cannot reserve a page).
        for (max_batch_tokens, prefix_caching, num_pages, speculative, block_size) in [
            (32, false, 32, false, 4),
            (8, true, 32, false, 4),
            (32, false, 30, false, 1),
            (32, true, 32, true, 4),
        ] {
            let kv = PagedKvCacheConfig {
                block_size,
                num_pages,
                max_sequence_length: 32,
            };
            let batching = ContinuousBatchConfig {
                max_active_sequences: all_prompts.len(),
                max_batch_tokens,
            };
            let mut runner = ContinuousBatchRunner::new(&model, batching, kv, &device).unwrap();
            runner.set_prefix_caching(prefix_caching);
            if speculative {
                runner
                    .set_speculative_decoding(Some(SpeculativeConfig {
                        draft_tokens: 3,
                        ngram: 2,
                    }))
                    .unwrap();
            }
            let mut ids = Vec::new();
            for prompt in &all_prompts {
                ids.push(runner.submit(prompt.clone(), generation.clone()).unwrap());
            }
            let finished = runner.run_until_idle(&model, &device).unwrap();
            assert_eq!(finished.len(), all_prompts.len());
            let snapshot = runner.snapshot();
            assert_eq!(snapshot.free_kv_pages, snapshot.total_kv_pages, "all pages returned");
            eprintln!(
                "runner tokens={max_batch_tokens} prefix={prefix_caching} pages={num_pages} \
                 speculative={speculative}: preemptions={} prefix_cached_tokens={} \
                 drafted={} accepted={}",
                snapshot.preemptions,
                snapshot.prefix_cached_tokens,
                snapshot.drafted_tokens,
                snapshot.accepted_draft_tokens
            );
            if block_size == 1 {
                assert!(snapshot.preemptions > 0, "small pool must preempt");
            }
            for ((prompt, id), reference) in all_prompts.iter().zip(ids).zip(&references) {
                let done = finished.iter().find(|done| done.request_id == id).unwrap();
                assert_eq!(&done.generated_token_ids, reference, "{prompt:?}");
            }
        }
    }

    #[test]
    fn paged_prefill_logits_match_dense_bf16() {
        type B = Cuda<bf16, i32>;
        let device = CudaDevice::default();
        let config = tiny_config();
        let model = LlamaForCausalLm::<B>::init_qwen2(&config, &device)
            .unwrap()
            .into_packed_inference();
        let kv = PagedKvCacheConfig {
            block_size: 16,
            num_pages: 8,
            max_sequence_length: 48,
        };
        let batching = ContinuousBatchConfig {
            max_active_sequences: 2,
            max_batch_tokens: 64,
        };
        let engine = PagedLlamaEngine::new(&model, kv, &device).unwrap();
        let mut scheduler = ContinuousBatchScheduler::new(batching, kv).unwrap();
        let prompts = [vec![5, 9, 13, 2, 44, 7, 7, 1, 30, 31, 32, 3, 9, 50, 61, 4, 22, 8],
            vec![60, 1, 18, 18, 2, 40, 9, 11, 13, 17, 19, 23, 29, 31, 37, 41, 43, 47]];
        for prompt in &prompts {
            scheduler
                .submit(prompt.clone(), GreedyGenerationConfig { max_new_tokens: 2, eos_token_ids: vec![] })
                .unwrap();
        }
        let batch = scheduler.schedule().unwrap().unwrap();
        assert_eq!(batch.batch_size(), 2);
        let paged = engine
            .execute(&model, &batch, &device)
            .unwrap()
            .cast(DType::F32)
            .into_data()
            .to_vec::<f32>()
            .unwrap();
        let vocabulary = config.vocab_size;
        for (row, prompt) in prompts.iter().enumerate() {
            let mut cache: LlamaKvCache<B> = model.new_cache();
            let tokens = Tensor::<B, 2, Int>::from_data(
                TensorData::new(prompt.clone(), [1, prompt.len()]),
                &device,
            );
            let dense = model
                .forward_cached_last_ruda(tokens, &mut cache)
                .cast(DType::F32)
                .into_data()
                .to_vec::<f32>()
                .unwrap();
            let actual = &paged[row * vocabulary..(row + 1) * vocabulary];
            let error: f32 = actual.iter().zip(&dense).map(|(a, b)| (a - b).powi(2)).sum::<f32>().sqrt();
            let norm: f32 = dense.iter().map(|b| b * b).sum::<f32>().sqrt();
            eprintln!("paged vs dense bf16 prefill row {row}: relative L2 {:e}", error / norm);
            assert!(error / norm < 2.0e-2, "row {row}: relative L2 {}", error / norm);
        }
    }
}
