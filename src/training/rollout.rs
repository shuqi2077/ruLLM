//! Synchronous policy-version batches over the actual paged KV device engine.
use super::{PostTrainingState, Response, PagedResponse, ResponseRecord, SftTrainer, TrainError, TrainingBackend, PpoTrainer, GrpoTrainer};
use crate::{ContinuousBatchConfig, ContinuousBatchRunner, PagedKvCacheConfig,
    SamplingConfig, SamplingGenerationConfig, RequestId};
use ruda::runtime::server::ComputeServer;
use ruda_model::module::AutodiffModule;
use ruda_tensor::{DeviceOps, api::backend::AutodiffBackend};
use ruda_tensor_device::{BoolElement, DeviceBackend, DeviceRuntime, FloatElement, IntElement};
use std::collections::BTreeMap;

fn failure(error: impl core::fmt::Display) -> TrainError { TrainError(error.to_string()) }

/// One seed per request. It is committed to PostTrainingState only after the
/// complete batch succeeds, so a restored pre-rollout checkpoint retries the
/// same request-local categorical streams rather than advancing them twice.
fn request_seed(cursor: &mut u64) -> u64 {
    *cursor = cursor.wrapping_add(0x9E3779B97F4A7C15);
    let mut value = *cursor;
    value = (value ^ (value >> 30)).wrapping_mul(0xBF58476D1CE4E5B9);
    value = (value ^ (value >> 27)).wrapping_mul(0x94D049BB133111EB);
    value ^ (value >> 31)
}

macro_rules! paged_trainer_sampler {
    ($trainer:ident) => {
        impl<B: AutodiffBackend + TrainingBackend> $trainer<B> {
            pub fn sample_paged<R, F, I, BT>(&mut self, prompts: &[Vec<i32>], max_new_tokens: usize,
                eos: &[i32], batching: ContinuousBatchConfig, kv: PagedKvCacheConfig)
                -> Result<Vec<PagedResponse>, TrainError>
            where B: AutodiffBackend<InnerBackend = DeviceBackend<R, F, I, BT>, Device = R::RudaDevice>,
                R: DeviceRuntime, R::Server: ComputeServer, R::Device: DeviceOps,
                F: FloatElement, I: IntElement, BT: BoolElement,
            {
                self.actor.sample_paged(prompts, max_new_tokens, eos, batching, kv, &mut self.state)
            }
        }
    };
}
paged_trainer_sampler!(PpoTrainer);
paged_trainer_sampler!(GrpoTrainer);

impl<B: AutodiffBackend + TrainingBackend> SftTrainer<B> {
    /// Capture current full/LoRA weights and decode a ragged prompt batch using
    /// continuous batching and physical paged KV. The snapshot lives for this
    /// call only; the next call captures the newly updated actor, never stale
    /// weights or KV. Caller-controlled rewards and grouping remain unchanged.
    pub fn sample_paged<R, F, I, BT>(&self, prompts: &[Vec<i32>], max_new_tokens: usize,
        eos: &[i32], batching: ContinuousBatchConfig, kv: PagedKvCacheConfig,
        state: &mut PostTrainingState) -> Result<Vec<PagedResponse>, TrainError>
    where
        B: AutodiffBackend<InnerBackend = DeviceBackend<R, F, I, BT>, Device = R::RudaDevice>,
        R: DeviceRuntime, R::Server: ComputeServer, R::Device: DeviceOps,
        F: FloatElement, I: IntElement, BT: BoolElement,
    {
        if prompts.is_empty() || max_new_tokens == 0 {
            return Err(failure("paged rollout needs nonempty prompts and a positive token budget"));
        }
        kv.validate().map_err(failure)?;
        batching.validate().map_err(failure)?;
        let geometry = self.model().geometry();
        for prompt in prompts {
            if prompt.is_empty() || prompt.len().checked_add(max_new_tokens)
                .is_none_or(|length| length > kv.max_sequence_length || length > geometry.max_sequence_length)
                || prompt.iter().chain(eos).any(|&token| token < 0 || token as usize >= geometry.vocab_size) {
                return Err(failure("invalid paged rollout prompt, context capacity or EOS token"));
            }
        }
        let model = self.model().valid();
        let model = match self.adapters() {
            Some(adapters) => model.merge_lora(&adapters.valid()),
            None => model,
        };
        let mut runner = ContinuousBatchRunner::new(&model, batching, kv, self.device()).map_err(failure)?;
        let mut cursor = state.sampling_seed;
        let mut order = Vec::with_capacity(prompts.len());
        let mut histories: BTreeMap<RequestId, Vec<usize>> = BTreeMap::new();
        let mut probabilities: BTreeMap<RequestId, Vec<f32>> = BTreeMap::new();
        for prompt in prompts {
            let id = runner.submit_sampled(prompt.clone(), SamplingGenerationConfig {
                max_new_tokens, eos_token_ids: eos.to_vec(),
                sampling: SamplingConfig { temperature: 1.0, top_k: 0, top_p: 1.0,
                    seed: Some(request_seed(&mut cursor)) },
            }).map_err(failure)?;
            order.push(id);
            histories.insert(id, Vec::new());
            probabilities.insert(id, Vec::new());
        }
        let mut finished = BTreeMap::new();
        while finished.len() < prompts.len() {
            let report = runner.step(&model, self.device()).map_err(failure)?;
            if report.is_some() {
                for &(id, logp) in runner.selected_log_probs() {
                    probabilities.get_mut(&id).ok_or_else(|| failure("unknown rollout probability"))?.push(logp);
                }
                for &(id, start, end) in runner.appended_ranges() {
                    let history = histories.get_mut(&id).ok_or_else(|| failure("unknown rollout request"))?;
                    history.resize(history.len().max(end), 0);
                    for context in &mut history[start..end] {
                        // Preemption replays already committed contexts rather
                        // than replacing them with the longer prefill's end.
                        if *context == 0 { *context = end; }
                    }
                }
            }
            while let Some(done) = runner.pop_finished() {
                let length = done.prompt_token_ids.len() + done.generated_token_ids.len();
                let mut contexts = histories.remove(&done.request_id).ok_or_else(|| failure("missing rollout history"))?;
                contexts.resize(length, length);
                if contexts.contains(&0) { return Err(failure("incomplete rollout rotary history")); }
                let response = PagedResponse {
                    response: Response { prompt: done.prompt_token_ids, completion: done.generated_token_ids,
                        terminal: done.stopped_on_eos },
                    rotary_context: contexts,
                    behavior_log_probs: probabilities.remove(&done.request_id)
                        .ok_or_else(|| failure("missing rollout probabilities"))? };
                response.validate(self.model())?;
                finished.insert(done.request_id, response);
            }
            if report.is_none() && finished.len() != prompts.len() {
                let status = runner.snapshot();
                return Err(failure(format!("paged rollout stalled: {} pending, {} active, {}/{} KV pages free",
                    status.pending_requests, status.active_requests, status.free_kv_pages, status.total_kv_pages)));
            }
        }
        state.sampling_seed = cursor;
        order.into_iter().map(|id| finished.remove(&id).ok_or_else(|| failure("missing finished rollout"))).collect()
    }
}
