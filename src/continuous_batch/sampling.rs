use super::{ContinuousBatchError, ContinuousBatchScheduler, RequestPhase};
use crate::{GreedyGenerationConfig, RequestId, SamplingGenerationConfig, SamplingOptions, TokenSampler};
use std::collections::BTreeMap;

pub type BatchSamplingState = BTreeMap<RequestId, Option<TokenSampler>>;

/// Selection state is committed only together with the corresponding batch.
pub(crate) struct BatchTokenSelection {
    batch_id: u64,
    tokens: Vec<i32>,
    samplers: Vec<(RequestId, TokenSampler)>,
}

/// Placeholder for rows that do not produce a token (non-final prefill chunk).
pub(crate) const NO_TOKEN: i32 = -1;

impl ContinuousBatchScheduler {
    /// Submit a sampled request with its own RNG, independent of batch membership.
    pub fn submit_sampled(
        &mut self,
        prompt_token_ids: Vec<i32>,
        generation: SamplingGenerationConfig,
    ) -> Result<RequestId, ContinuousBatchError> {
        self.submit_sampled_with_options(prompt_token_ids, generation, SamplingOptions::default())
    }

    pub fn submit_sampled_with_options(
        &mut self,
        prompt_token_ids: Vec<i32>,
        generation: SamplingGenerationConfig,
        options: SamplingOptions,
    ) -> Result<RequestId, ContinuousBatchError> {
        let sampler = TokenSampler::new_with_options(generation.sampling, options)
            .map_err(|error| ContinuousBatchError(error.to_string()))?;
        self.submit_with_sampler(
            prompt_token_ids,
            GreedyGenerationConfig {
                max_new_tokens: generation.max_new_tokens,
                eos_token_ids: generation.eos_token_ids,
            },
            Some(sampler),
        )
    }

    /// True when some token-producing row of the in-flight batch samples from
    /// its full distribution; otherwise device-side argmax is sufficient.
    pub(crate) fn batch_requires_logits(&self, batch_id: u64) -> Result<bool, ContinuousBatchError> {
        let batch = self.in_flight_batch(batch_id)?;
        Ok(batch.sequences.iter().any(|row| {
            row.samples_token
                && self
                    .active
                    .get(&row.request_id)
                    .is_some_and(|request| request.sampler.is_some())
        }))
    }

    /// Commit a batch whose rows are all greedy, from device-side argmax.
    pub(crate) fn complete_greedy_batch(
        &mut self,
        batch_id: u64,
        argmax: &[i32],
    ) -> Result<Vec<i32>, ContinuousBatchError> {
        let batch = self.in_flight_batch(batch_id)?;
        if argmax.len() != batch.sequences.len() {
            return Err(ContinuousBatchError(
                "argmax rows do not match the scheduled batch".into(),
            ));
        }
        let tokens: Vec<i32> = batch
            .sequences
            .iter()
            .zip(argmax)
            .map(|(row, &token)| if row.samples_token { token } else { NO_TOKEN })
            .collect();
        if self.batch_requires_logits(batch_id)? {
            return Err(ContinuousBatchError(
                "batch has sampled rows and needs full logits".into(),
            ));
        }
        self.complete_batch(batch_id, &tokens)?;
        Ok(tokens)
    }

    fn in_flight_batch(&self, batch_id: u64) -> Result<&super::ScheduledBatch, ContinuousBatchError> {
        self.in_flight
            .as_ref()
            .filter(|batch| batch.id == batch_id)
            .ok_or_else(|| ContinuousBatchError(format!("batch {batch_id} is not in flight")))
    }

    pub(crate) fn select_batch_tokens(
        &self,
        batch_id: u64,
        logits: &[f32],
        vocabulary: usize,
    ) -> Result<BatchTokenSelection, ContinuousBatchError> {
        self.select_batch_tokens_with(batch_id, logits, vocabulary, self.batch_sampling_state(batch_id)?,
            |id, values, sampler| match sampler {
                Some(sampler) => {
                    let request = &self.active[&id];
                    sampler.sample_with_history(values, &request.prompt_token_ids, &request.generated_token_ids)
                        .map_err(|e| ContinuousBatchError(e.to_string()))
                },
                None => greedy_token(values, 0),
            })
    }

    pub(crate) fn batch_sampling_state(&self, batch_id: u64) -> Result<BatchSamplingState, ContinuousBatchError> {
        let batch = self.in_flight.as_ref().filter(|b| b.id == batch_id)
            .ok_or_else(|| ContinuousBatchError(format!("batch {batch_id} is not in flight")))?;
        batch.sequences.iter().map(|row| self.active.get(&row.request_id)
            .map(|request| (row.request_id, request.sampler.clone()))
            .ok_or_else(|| ContinuousBatchError("sampling references missing request".into()))).collect()
    }

    pub(crate) fn select_batch_tokens_with(
        &self, batch_id: u64, logits: &[f32], vocabulary: usize, mut state: BatchSamplingState,
        mut select: impl FnMut(RequestId, &[f32], &mut Option<TokenSampler>) -> Result<i32, ContinuousBatchError>,
    ) -> Result<BatchTokenSelection, ContinuousBatchError> {
        let batch = self
            .in_flight
            .as_ref()
            .filter(|batch| batch.id == batch_id)
            .ok_or_else(|| ContinuousBatchError(format!("batch {batch_id} is not in flight")))?;
        if vocabulary == 0 || batch.sequences.len().checked_mul(vocabulary) != Some(logits.len()) {
            return Err(ContinuousBatchError(
                "logit rows do not match the scheduled batch".into(),
            ));
        }
        let mut tokens = Vec::with_capacity(batch.sequences.len());
        let mut samplers = Vec::new();
        for (row, sequence) in batch.sequences.iter().enumerate() {
            let request = self.active.get(&sequence.request_id).ok_or_else(|| {
                ContinuousBatchError(format!(
                    "batch references missing request {}",
                    sequence.request_id.0
                ))
            })?;
            if request.phase != RequestPhase::InFlight(batch_id) {
                return Err(ContinuousBatchError(format!(
                    "request {} is not owned by batch {batch_id}",
                    sequence.request_id.0
                )));
            }
            if !sequence.samples_token {
                tokens.push(NO_TOKEN);
                continue;
            }
            let values = &logits[row * vocabulary..(row + 1) * vocabulary];
            let mut next = state.remove(&sequence.request_id)
                .ok_or_else(|| ContinuousBatchError("missing transactional batch sampler".into()))?;
            tokens.push(select(sequence.request_id, values, &mut next)?);
            if let Some(mut next) = next {
                // Batch previews retain RNG checkpoints, not one O(vocabulary)
                // buffer per request. Streaming single requests still reuse
                // their TokenSampler workspace across every decode step.
                next.clear_workspace();
                samplers.push((sequence.request_id, next));
            }
        }
        Ok(BatchTokenSelection {
            batch_id,
            tokens,
            samplers,
        })
    }

    pub(crate) fn complete_selected_batch(
        &mut self,
        selection: BatchTokenSelection,
    ) -> Result<Vec<i32>, ContinuousBatchError> {
        self.complete_batch(selection.batch_id, &selection.tokens)?;
        for (request_id, sampler) in selection.samplers {
            if let Some(request) = self.active.get_mut(&request_id) {
                request.sampler = Some(sampler);
            }
        }
        Ok(selection.tokens)
    }
}

pub(super) fn greedy_token(row_values: &[f32], row: usize) -> Result<i32, ContinuousBatchError> {
    let mut best = None;
    for (index, &value) in row_values.iter().enumerate() {
        if value.is_nan() {
            continue;
        }
        if best.is_none_or(|(_, best_value)| value > best_value) {
            best = Some((index, value));
        }
    }
    let (index, _) = best.ok_or_else(|| {
        ContinuousBatchError(format!("all vocabulary logits are NaN for batch row {row}"))
    })?;
    i32::try_from(index).map_err(|_| ContinuousBatchError("vocabulary index exceeds i32".into()))
}

#[cfg(test)]
mod tests;
