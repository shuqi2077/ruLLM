use crate::{
    GreedyGenerationConfig, KvPageId, KvReservationId, PagedKvAppendReservation,
    PagedKvCacheConfig, PagedKvCacheManager, PagedKvError, RequestId, TokenSampler,
};
use std::collections::{BTreeMap, BTreeSet, VecDeque};
use std::error::Error;
use std::fmt::{Display, Formatter};

mod sampling;
mod device;
pub use device::{DeviceBatchModel, DeviceBatchExecutor, ExecutedBatch};
mod prefix_device;
pub use prefix_device::{PrefixBatchModel,PrefixBatchExecutor,PrefixBatchStep,ActiveKvSwapConfig,ActiveKvSwapStats};
mod fenced;
pub use fenced::{BatchFence, BatchLaunchFailure, FencedBatchScheduler};
mod lifecycle;
pub use lifecycle::{CancelledGeneration, ContinuousBatchOptions, KvAdmissionPolicy};
pub(crate) use sampling::BatchTokenSelection;
pub use sampling::BatchSamplingState;

/// `max_batch_tokens` bounds the tokens of one device batch. Prompts longer
/// than that are prefilled in chunks of at most `max_batch_tokens` tokens.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct ContinuousBatchConfig {
    pub max_active_sequences: usize,
    pub max_batch_tokens: usize,
}

impl ContinuousBatchConfig {
    pub fn validate(self) -> Result<(), ContinuousBatchError> {
        if self.max_active_sequences == 0 || self.max_batch_tokens == 0 {
            return Err(ContinuousBatchError(
                "continuous batch limits must be non-zero".into(),
            ));
        }
        Ok(())
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ScheduledBatchKind {
    Prefill,
    Decode,
    /// Speculative decode: each row is `[last token, draft...]`; logits are
    /// needed at every position and only the verified prefix is committed.
    Verify,
}

/// Prompt-lookup (n-gram) speculative decoding for greedy requests.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct SpeculativeConfig {
    /// Draft tokens proposed per verification step.
    pub draft_tokens: usize,
    /// Length of the suffix matched against earlier context.
    pub ngram: usize,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct ScheduledSequence {
    pub request_id: RequestId,
    pub token_ids: Vec<i32>,
    pub start_position: usize,
    pub context_length: usize,
    pub block_table: Vec<KvPageId>,
    pub reservation_id: KvReservationId,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct ScheduledBatch {
    pub id: u64,
    pub kind: ScheduledBatchKind,
    pub sequences: Vec<ScheduledSequence>,
}

impl ScheduledBatch {
    pub fn batch_size(&self) -> usize {
        self.sequences.len()
    }

    pub fn next_n(&self) -> usize {
        self.sequences.first().map_or(0, |row| row.token_ids.len())
    }

    /// Row-major `[batch, next_n]` input expected by the Llama embedding path.
    pub fn token_matrix(&self) -> Vec<i32> {
        self.sequences
            .iter()
            .flat_map(|sequence| sequence.token_ids.iter().copied())
            .collect()
    }

    /// One causal length per query token, in the exact order required by
    /// `Fp8Fp4PagedAttention::context_lens`.
    pub fn context_lengths(&self) -> Result<Vec<u32>, ContinuousBatchError> {
        let mut lengths = Vec::with_capacity(self.batch_size().saturating_mul(self.next_n()));
        for sequence in &self.sequences {
            for offset in 0..sequence.token_ids.len() {
                let length = sequence
                    .start_position
                    .checked_add(offset)
                    .and_then(|value| value.checked_add(1))
                    .ok_or_else(|| ContinuousBatchError("context length overflow".into()))?;
                lengths.push(u32::try_from(length).map_err(|_| {
                    ContinuousBatchError("context length exceeds DeepGEMM u32 ABI".into())
                })?);
            }
        }
        Ok(lengths)
    }

    /// Return the padded row-major physical page table and its row stride.
    pub fn flattened_block_table(&self) -> Result<(Vec<u32>, usize), ContinuousBatchError> {
        let stride = self
            .sequences
            .iter()
            .map(|sequence| sequence.block_table.len())
            .max()
            .unwrap_or(0);
        if stride == 0 && !self.sequences.is_empty() {
            return Err(ContinuousBatchError(
                "scheduled sequence has no physical KV page".into(),
            ));
        }
        let length = self.batch_size().checked_mul(stride)
            .ok_or_else(|| ContinuousBatchError("block table size overflow".into()))?;
        let mut table = Vec::new();
        table.try_reserve_exact(length)
            .map_err(|error| ContinuousBatchError(format!("cannot allocate block table: {error}")))?;
        table.resize(length, 0_u32);
        for (row, sequence) in self.sequences.iter().enumerate() {
            for (column, page) in sequence.block_table.iter().enumerate() {
                table[row * stride + column] = page.0;
            }
        }
        Ok((table, stride))
    }
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct FinishedGeneration {
    pub request_id: RequestId,
    pub prompt_token_ids: Vec<i32>,
    pub generated_token_ids: Vec<i32>,
    pub stopped_on_eos: bool,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct ContinuousBatchSnapshot {
    pub pending_requests: usize,
    pub active_requests: usize,
    pub has_in_flight_batch: bool,
    pub finished_requests: usize,
    pub free_kv_pages: usize,
    pub total_kv_pages: usize,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct PagedBatchSnapshot {
    pub pending_requests: usize,
    pub active_requests: usize,
    pub has_in_flight_batch: bool,
    pub finished_requests: usize,
    pub free_kv_pages: usize,
    pub total_kv_pages: usize,
    /// Requests whose KV was dropped to make room and later recomputed.
    pub preemptions: u64,
    /// Prompt tokens mapped from the prefix cache instead of recomputed.
    pub prefix_cached_tokens: u64,
    /// Speculative draft tokens verified, and how many were accepted.
    pub drafted_tokens: u64,
    pub accepted_draft_tokens: u64,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct ContinuousBatchError(pub String);

impl Display for ContinuousBatchError {
    fn fmt(&self, formatter: &mut Formatter<'_>) -> std::fmt::Result {
        formatter.write_str(&self.0)
    }
}

impl Error for ContinuousBatchError {}

impl From<PagedKvError> for ContinuousBatchError {
    fn from(value: PagedKvError) -> Self {
        Self(value.0)
    }
}

#[derive(Debug)]
struct PendingRequest {
    id: RequestId,
    prompt_token_ids: Vec<i32>,
    generation: GreedyGenerationConfig,
    sampler: Option<TokenSampler>,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum RequestPhase {
    /// `history[committed..]` still has to be written to KV.
    NeedsPrefill,
    AwaitingDecode(i32),
    InFlight(u64),
}

#[derive(Debug)]
struct ActiveRequest {
    prompt_token_ids: Vec<i32>,
    generation: GreedyGenerationConfig,
    sampler: Option<TokenSampler>,
    eos: BTreeSet<i32>,
    generated_token_ids: Vec<i32>,
    /// Prompt followed by every generated token. KV holds a prefix of it;
    /// after preemption the whole history is prefilled again.
    history: Vec<i32>,
    /// Admission order; the newest holder of pages is preempted first.
    admitted: u64,
    phase: RequestPhase,
}

/// Metadata scheduler for paged Llama inference. It allows prefill requests to
/// enter while older requests are decoding, batches all ready one-token decode
/// rows, and commits KV positions only after the execution result succeeds.
///
/// Long prompts are prefilled in chunks of at most `max_batch_tokens`. When
/// no batch fits in the remaining KV pages, the most recently admitted request
/// holding pages is preempted: its pages are released and its prompt plus
/// generated tokens are recomputed later, so its output is unchanged.
#[derive(Debug)]
pub struct ContinuousBatchScheduler {
    config: ContinuousBatchConfig,
    options: ContinuousBatchOptions,
    kv_cache: PagedKvCacheManager,
    pending: VecDeque<PendingRequest>,
    deferred_prefill: BTreeSet<RequestId>,
    active: BTreeMap<RequestId, ActiveRequest>,
    finished: VecDeque<FinishedGeneration>,
    in_flight: Option<ScheduledBatch>,
    prefer_prefill: bool,
    next_request: u64,
    next_batch: u64,
    prefill_chunk_tokens:Option<usize>,
    admission_ceiling:Option<RequestId>,
    recompute_preemption: bool,
    next_admission: u64,
    preemptions: u64,
    speculative: Option<SpeculativeConfig>,
    speculative_turn: bool,
    drafted_tokens: u64,
    accepted_draft_tokens: u64,
}

/// Most recent earlier occurrence of the last `ngram` tokens of `history`,
/// followed by up to `draft_tokens` continuation tokens.
fn ngram_draft(history: &[i32], ngram: usize, draft_tokens: usize) -> Vec<i32> {
    for n in (1..=ngram.min(history.len().saturating_sub(1))).rev() {
        let pattern = &history[history.len() - n..];
        for start in (0..history.len() - n).rev() {
            if &history[start..start + n] == pattern {
                let from = start + n;
                let to = (from + draft_tokens).min(history.len());
                if from < to {
                    return history[from..to].to_vec();
                }
            }
        }
    }
    Vec::new()
}

impl ContinuousBatchScheduler {
    pub fn new(
        config: ContinuousBatchConfig,
        kv_config: PagedKvCacheConfig,
    ) -> Result<Self, ContinuousBatchError> {
        Self::with_options(config, kv_config, ContinuousBatchOptions::default())
    }

    /// Configure bounded queuing and optional full-sequence KV admission.
    pub fn with_options(
        config: ContinuousBatchConfig,
        kv_config: PagedKvCacheConfig,
        options: ContinuousBatchOptions,
    ) -> Result<Self, ContinuousBatchError> {
        config.validate()?;
        options.validate()?;
        Ok(Self {
            config,
            options,
            kv_cache: PagedKvCacheManager::new(kv_config)?,
            pending: VecDeque::new(),
            deferred_prefill:BTreeSet::new(),
            active: BTreeMap::new(),
            finished: VecDeque::new(),
            in_flight: None,
            prefer_prefill: true,
            next_request: 1,
            next_batch: 1,
            prefill_chunk_tokens:None,
            admission_ceiling:None,
            recompute_preemption: false,
            next_admission: 0,
            preemptions: 0,
            speculative: None,
            speculative_turn: false,
            drafted_tokens: 0,
            accepted_draft_tokens: 0,
        })
    }

    pub fn new_paged(
        config: ContinuousBatchConfig,
        kv_config: PagedKvCacheConfig,
    ) -> Result<Self, ContinuousBatchError> {
        let mut scheduler = Self::new(config, kv_config)?;
        scheduler.prefill_chunk_tokens = Some(config.max_batch_tokens);
        scheduler.recompute_preemption = true;
        Ok(scheduler)
    }

    /// Reuse KV pages of identical full-block prompt prefixes across requests.
    pub fn set_prefix_caching(&mut self, enabled: bool) {
        self.kv_cache.set_prefix_caching(enabled);
    }

    /// Enable prompt-lookup speculative decoding for greedy requests, or
    /// disable it with `None`. Outputs are identical to plain greedy decoding;
    /// only the number of device steps changes.
    pub fn set_speculative_decoding(
        &mut self,
        config: Option<SpeculativeConfig>,
    ) -> Result<(), ContinuousBatchError> {
        if let Some(config) = config
            && (config.draft_tokens == 0 || config.ngram == 0)
        {
            return Err(ContinuousBatchError(
                "speculative draft_tokens and ngram must be non-zero".into(),
            ));
        }
        self.speculative = config;
        Ok(())
    }

    pub fn submit(
        &mut self,
        prompt_token_ids: Vec<i32>,
        generation: GreedyGenerationConfig,
    ) -> Result<RequestId, ContinuousBatchError> {
        self.submit_with_sampler(prompt_token_ids, generation, None)
    }

    fn submit_with_sampler(
        &mut self,
        prompt_token_ids: Vec<i32>,
        generation: GreedyGenerationConfig,
        sampler: Option<TokenSampler>,
    ) -> Result<RequestId, ContinuousBatchError> {
        if prompt_token_ids.is_empty() {
            return Err(ContinuousBatchError(
                "continuous generation prompt must not be empty".into(),
            ));
        }
        if self.prefill_chunk_tokens.is_none() && prompt_token_ids.len() > self.config.max_batch_tokens {
            return Err(ContinuousBatchError(format!(
                "prompt has {} tokens, above max_batch_tokens {}",
                prompt_token_ids.len(),
                self.config.max_batch_tokens
            )));
        }
        let total_length = prompt_token_ids
            .len()
            .checked_add(generation.max_new_tokens)
            .ok_or_else(|| ContinuousBatchError("request sequence length overflow".into()))?;
        let kv = self.kv_cache.config();
        if total_length > kv.max_sequence_length {
            return Err(ContinuousBatchError(format!(
                "request length {total_length} exceeds paged KV capacity {}",
                kv.max_sequence_length
            )));
        }
        if total_length.div_ceil(kv.block_size) > kv.num_pages {
            return Err(ContinuousBatchError(format!(
                "request length {total_length} needs more than the {} KV pages in the pool",
                kv.num_pages
            )));
        }
        if prompt_token_ids.iter().chain(&generation.eos_token_ids).any(|&id| id < 0) {
            return Err(ContinuousBatchError("prompt and EOS token IDs must be non-negative".into()));
        }
        if generation.max_new_tokens > 0 {
            let kv = self.kv_cache.config();
            let required_pages = match self.options.kv_admission {
                KvAdmissionPolicy::OnDemand => prompt_token_ids.len().div_ceil(kv.block_size),
                KvAdmissionPolicy::ReserveSequenceCapacity =>
                    Self::request_page_budget(prompt_token_ids.len(), &generation, kv.block_size),
            };
            if required_pages > kv.num_pages {
                return Err(ContinuousBatchError(format!(
                    "request needs {required_pages} KV pages under the admission policy, but the pool has {}",
                    kv.num_pages
                )));
            }
            if self.options.max_pending_requests.is_some_and(|limit| self.pending.len() >= limit) {
                return Err(ContinuousBatchError("pending request queue is full".into()));
            }
        }
        let id = RequestId(self.next_request);
        self.next_request = self
            .next_request
            .checked_add(1)
            .ok_or_else(|| ContinuousBatchError("request id overflow".into()))?;
        if generation.max_new_tokens == 0 {
            self.finished.push_back(FinishedGeneration {
                request_id: id,
                prompt_token_ids,
                generated_token_ids: Vec::new(),
                stopped_on_eos: false,
            });
        } else {
            self.pending.push_back(PendingRequest {
                id,
                prompt_token_ids,
                generation,
                sampler,
            });
        }
        Ok(id)
    }

    /// Build the next device batch. At most one batch is in flight so page
    /// reservations have an unambiguous commit/rollback owner.
    pub fn schedule(&mut self) -> Result<Option<ScheduledBatch>, ContinuousBatchError> {
        if let Some(batch) = &self.in_flight {
            return Err(ContinuousBatchError(format!(
                "batch {} is still in flight",
                batch.id
            )));
        }
        // Check before acquiring reservations: an exhausted batch ID must not
        // leak pages or strand requests in a partially scheduled batch.
        if self.next_batch == u64::MAX {
            return Err(ContinuousBatchError("batch id overflow".into()));
        }
        self.promote_pending()?;
        loop {
            if let Some(batch) = self.try_schedule()? {
                self.in_flight = Some(batch.clone());
                return Ok(Some(batch));
            }
            if !self.recompute_preemption || !self.preempt_newest()? {
                return Ok(None);
            }
        }
    }

    fn try_schedule(&mut self) -> Result<Option<ScheduledBatch>, ContinuousBatchError> {
        if self.prefer_prefill
            && let Some(batch) = self.schedule_prefill()?
        {
            self.prefer_prefill = false;
            return Ok(Some(batch));
        }
        // Alternate speculative and plain decode turns so sampled requests
        // (which are never drafted) keep progressing.
        self.speculative_turn = !self.speculative_turn;
        if self.speculative_turn
            && let Some(batch) = self.schedule_verify()?
        {
            self.prefer_prefill = true;
            return Ok(Some(batch));
        }
        if let Some(batch) = self.schedule_decode()? {
            self.prefer_prefill = true;
            return Ok(Some(batch));
        }
        if let Some(batch) = self.schedule_prefill()? {
            self.prefer_prefill = false;
            return Ok(Some(batch));
        }
        Ok(None)
    }

    /// Release the KV pages of the newest page-holding request when at least
    /// one other request also holds pages. Returns false if nothing can be
    /// gained (no runnable work, or a single request exceeds the free pool).
    fn preempt_newest(&mut self) -> Result<bool, ContinuousBatchError> {
        let mut holders = Vec::new();
        for (&id, request) in &self.active {
            if !matches!(request.phase, RequestPhase::InFlight(_))
                && !self.kv_cache.block_table(id)?.is_empty()
            {
                holders.push((request.admitted, id));
            }
        }
        if holders.len() <= 1 {
            return Ok(false);
        }
        let (_, victim) = *holders.iter().max().expect("at least two holders");
        // Plain recreation: re-mapping cached prefix pages here would take
        // back the pages that the preemption is meant to release.
        self.kv_cache.remove_sequence(victim)?;
        self.kv_cache.create_sequence(victim)?;
        self.active
            .get_mut(&victim)
            .expect("victim came from the active map")
            .phase = RequestPhase::NeedsPrefill;
        self.preemptions += 1;
        // Let older requests decode into the released pages before the
        // victim is prefilled again.
        self.prefer_prefill = false;
        Ok(true)
    }

    /// Commit every page reservation and apply one sampled/selected token per
    /// token-producing row. Entries for non-final prefill chunks are ignored.
    /// The caller must preserve the scheduled row order.
    pub fn complete_batch(
        &mut self,
        batch_id: u64,
        generated_token_ids: &[i32],
    ) -> Result<(), ContinuousBatchError> {
        let batch = self.take_batch(batch_id)?;
        if generated_token_ids.len() != batch.sequences.len() {
            self.in_flight = Some(batch);
            return Err(ContinuousBatchError(format!(
                "batch {batch_id} has {} rows but {} generated tokens were supplied",
                self.in_flight.as_ref().unwrap().sequences.len(),
                generated_token_ids.len()
            )));
        }
        if batch.sequences.iter().zip(generated_token_ids).any(|(row, &token)| self.samples_token(row) && token < 0) {
            self.in_flight = Some(batch);
            return Err(ContinuousBatchError("generated token IDs must be non-negative".into()));
        }
        let ownership_error = batch.sequences.iter().find_map(|sequence| {
            let Some(request) = self.active.get(&sequence.request_id) else {
                return Some(ContinuousBatchError(format!(
                    "batch references missing request {}",
                    sequence.request_id.0
                )));
            };
            (request.phase != RequestPhase::InFlight(batch_id)).then(|| {
                ContinuousBatchError(format!(
                    "request {} is not owned by batch {batch_id}",
                    sequence.request_id.0
                ))
            })
        });
        if let Some(error) = ownership_error {
            self.in_flight = Some(batch);
            return Err(error);
        }
        let reservations = batch.sequences.iter().map(|row| (row.reservation_id,row.token_ids.len())).collect::<Vec<_>>();
        if let Err(error) = self.kv_cache.commit_append_prefixes(&reservations) {
            self.in_flight = Some(batch);
            return Err(error.into());
        }
        for sequence in &batch.sequences {
            let request = &self.active[&sequence.request_id];
            self.kv_cache
                .register_prefix(sequence.request_id, &request.history)?;
        }
        for (sequence, &token) in batch.sequences.iter().zip(generated_token_ids) {
            let request = self
                .active
                .get_mut(&sequence.request_id)
                .expect("batch ownership was validated before KV commit");
            if sequence.context_length < request.history.len() {
                request.phase = RequestPhase::NeedsPrefill;
                continue;
            }
            request.generated_token_ids.push(token);
            request.history.push(token);
            let stopped_on_eos = request.eos.contains(&token);
            let finished = stopped_on_eos
                || request.generated_token_ids.len() >= request.generation.max_new_tokens;
            if finished {
                let request = self
                    .active
                    .remove(&sequence.request_id)
                    .expect("active request was checked before removal");
                self.kv_cache.remove_sequence(sequence.request_id)?;
                self.finished.push_back(FinishedGeneration {
                    request_id: sequence.request_id,
                    prompt_token_ids: request.prompt_token_ids,
                    generated_token_ids: request.generated_token_ids,
                    stopped_on_eos,
                });
            } else {
                request.phase = RequestPhase::AwaitingDecode(token);
            }
        }
        Ok(())
    }

    /// Roll back all page allocations from a failed device batch and restore
    /// requests to the exact state from which that batch was scheduled.
    pub fn fail_batch(&mut self, batch_id: u64) -> Result<(), ContinuousBatchError> {
        let batch = self.take_batch(batch_id)?;
        if batch.sequences.iter().any(|row| {
            self.active.get(&row.request_id)
                .is_none_or(|request| request.phase != RequestPhase::InFlight(batch_id))
        }) {
            self.in_flight = Some(batch);
            return Err(ContinuousBatchError("failed batch does not own every request".into()));
        }
        let reservations = batch.sequences.iter().map(|row| row.reservation_id).collect::<Vec<_>>();
        if let Err(error) = self.kv_cache.cancel_appends(&reservations) {
            self.in_flight = Some(batch);
            return Err(error.into());
        }
        for sequence in &batch.sequences {
            let request = self.active.get_mut(&sequence.request_id)
                .expect("batch ownership was validated before cancellation");
            request.phase = match batch.kind {
                ScheduledBatchKind::Prefill => RequestPhase::NeedsPrefill,
                ScheduledBatchKind::Decode | ScheduledBatchKind::Verify => {
                    RequestPhase::AwaitingDecode(sequence.token_ids[0])
                }
            };
        }
        Ok(())
    }

    pub fn pop_finished(&mut self) -> Option<FinishedGeneration> {
        self.finished.pop_front()
    }

    pub fn snapshot(&self) -> ContinuousBatchSnapshot {
        ContinuousBatchSnapshot {
            pending_requests: self.pending.len(),
            active_requests: self.active.len(),
            has_in_flight_batch: self.in_flight.is_some(),
            finished_requests: self.finished.len(),
            free_kv_pages: self.kv_cache.free_page_count(),
            total_kv_pages: self.kv_cache.config().num_pages,
        }
    }

    pub fn paged_snapshot(&self) -> PagedBatchSnapshot {
        let snapshot = self.snapshot();
        PagedBatchSnapshot {
            pending_requests: snapshot.pending_requests,
            active_requests: snapshot.active_requests,
            has_in_flight_batch: snapshot.has_in_flight_batch,
            finished_requests: snapshot.finished_requests,
            free_kv_pages: snapshot.free_kv_pages,
            total_kv_pages: snapshot.total_kv_pages,
            preemptions: self.preemptions,
            prefix_cached_tokens: self.kv_cache.reused_tokens(),
            drafted_tokens: self.drafted_tokens,
            accepted_draft_tokens: self.accepted_draft_tokens,
        }
    }

    pub fn samples_token(&self, sequence: &ScheduledSequence) -> bool {
        self.active.get(&sequence.request_id)
            .is_some_and(|request| sequence.context_length >= request.history.len())
    }

    /// Greedy rows with an n-gram draft, each appending `[last, draft...]`.
    fn schedule_verify(&mut self) -> Result<Option<ScheduledBatch>, ContinuousBatchError> {
        let Some(config) = self.speculative else {
            return Ok(None);
        };
        let width = config.draft_tokens + 1;
        let max_rows = self.config.max_batch_tokens / width;
        if max_rows == 0 {
            return Ok(None);
        }
        let mut candidates = Vec::new();
        for (&id, request) in &self.active {
            let RequestPhase::AwaitingDecode(token) = request.phase else {
                continue;
            };
            let remaining = request.generation.max_new_tokens - request.generated_token_ids.len();
            if request.sampler.is_some() || remaining < 2 {
                continue;
            }
            let draft = ngram_draft(&request.history, config.ngram, config.draft_tokens);
            if draft.is_empty() || !self.kv_cache.can_append(id, width)? {
                continue;
            }
            candidates.push((id, token, draft));
        }
        let mut rows = Vec::new();
        for (id, token, mut draft) in candidates {
            if rows.len() >= max_rows {
                break;
            }
            if !self.kv_cache.can_append(id, width)? {
                continue;
            }
            // Uniform row width: pad with the last draft token; padding is
            // simply rejected during verification if it is wrong.
            let pad = *draft.last().expect("draft is non-empty");
            draft.resize(config.draft_tokens, pad);
            let mut tokens = Vec::with_capacity(width);
            tokens.push(token);
            tokens.extend(draft);
            let reservation = self.kv_cache.begin_append(id, width)?;
            rows.push(Self::scheduled_sequence(reservation, tokens));
        }
        self.finish_schedule(ScheduledBatchKind::Verify, rows)
    }

    /// Commit a verification batch from per-position argmax `[rows, width]`.
    /// Returns the tokens appended for every row, in row order.
    pub fn complete_verify_batch(
        &mut self,
        batch_id: u64,
        argmax: &[i32],
    ) -> Result<Vec<(RequestId, Vec<i32>)>, ContinuousBatchError> {
        let batch = self.take_batch(batch_id)?;
        let width = batch.next_n();
        if batch.kind != ScheduledBatchKind::Verify || argmax.len() != batch.batch_size() * width {
            self.in_flight = Some(batch);
            return Err(ContinuousBatchError(
                "verification argmax does not match the in-flight batch".into(),
            ));
        }
        let mut appended = Vec::with_capacity(batch.batch_size());
        for (row, sequence) in batch.sequences.iter().enumerate() {
            let predictions = &argmax[row * width..(row + 1) * width];
            let request = self.active.get(&sequence.request_id).ok_or_else(|| {
                ContinuousBatchError(format!(
                    "batch references missing request {}",
                    sequence.request_id.0
                ))
            })?;
            let remaining = request.generation.max_new_tokens - request.generated_token_ids.len();
            let mut produced = Vec::with_capacity(width);
            for (position, &prediction) in predictions.iter().enumerate() {
                produced.push(prediction);
                if produced.len() == remaining || request.eos.contains(&prediction) {
                    break;
                }
                // Continue only while the draft at the next input position
                // equals what the model just predicted.
                if position + 1 >= width || sequence.token_ids[position + 1] != prediction {
                    break;
                }
            }
            self.drafted_tokens += (width - 1) as u64;
            self.accepted_draft_tokens += (produced.len() - 1) as u64;
            // Inputs t0 and the first produced.len() - 1 drafts are valid.
            self.kv_cache
                .commit_append_partial(sequence.reservation_id, produced.len())?;
            let request = self
                .active
                .get_mut(&sequence.request_id)
                .expect("checked above");
            request.history.extend(produced.iter().copied());
            request.generated_token_ids.extend(produced.iter().copied());
            let history = request.history.clone();
            let last = *produced.last().expect("at least one token is produced");
            let stopped_on_eos = request.eos.contains(&last);
            let finished = stopped_on_eos
                || request.generated_token_ids.len() >= request.generation.max_new_tokens;
            self.kv_cache
                .register_prefix(sequence.request_id, &history)?;
            if finished {
                let request = self
                    .active
                    .remove(&sequence.request_id)
                    .expect("checked above");
                self.kv_cache.remove_sequence(sequence.request_id)?;
                self.finished.push_back(FinishedGeneration {
                    request_id: sequence.request_id,
                    prompt_token_ids: request.prompt_token_ids,
                    generated_token_ids: request.generated_token_ids,
                    stopped_on_eos,
                });
            } else {
                request.phase = RequestPhase::AwaitingDecode(last);
            }
            appended.push((sequence.request_id, produced));
        }
        Ok(appended)
    }

    pub fn kv_cache(&self) -> &PagedKvCacheManager {
        &self.kv_cache
    }

    pub(super) fn reserve_lookahead(&mut self,batch:&ScheduledBatch,wanted:usize)
        ->Result<BTreeMap<RequestId,PagedKvAppendReservation>,ContinuousBatchError> {
        let mut reserved=BTreeMap::new();
        if wanted==0 || batch.kind!=ScheduledBatchKind::Decode {return Ok(reserved);}
        let mut budget=self.config.max_batch_tokens.saturating_sub(batch.batch_size());
        let block=self.kv_cache.config().block_size;
        for row in &batch.sequences {
            let request=&self.active[&row.request_id];
            let remaining=request.generation.max_new_tokens.saturating_sub(request.generated_token_ids.len());
            let extra=wanted.min(budget).min(remaining.saturating_sub(1))
                .min(self.kv_cache.config().max_sequence_length-row.context_length);
            let pages=(row.context_length+extra).div_ceil(block).saturating_sub(row.block_table.len());
            if extra>0 && pages<=self.kv_cache.free_page_count() {
                let reservation=self.kv_cache.extend_append(row.reservation_id,extra)?;
                budget-=extra;reserved.insert(row.request_id,reservation);
            }
        }
        Ok(reserved)
    }

    /// Generated tokens of an active request (empty for pending requests).
    pub fn generated_tokens(&self, request_id: RequestId) -> Option<&[i32]> {
        self.active
            .get(&request_id)
            .map(|request| request.generated_token_ids.as_slice())
    }

    fn promote_pending(&mut self) -> Result<(), ContinuousBatchError> {
        let mut available_pages = self.kv_cache.config().num_pages;
        if self.options.kv_admission == KvAdmissionPolicy::ReserveSequenceCapacity {
            for request in self.active.values() {
                available_pages = available_pages.checked_sub(Self::request_page_budget(
                    request.prompt_token_ids.len(), &request.generation, self.kv_cache.config().block_size,
                )).ok_or_else(|| ContinuousBatchError("active KV page budgets exceed the pool".into()))?;
            }
        }
        while self.active.len() < self.config.max_active_sequences {
            let Some(position)=self.pending.iter().position(|r|!self.deferred_prefill.contains(&r.id)
                && self.admission_ceiling.is_none_or(|ceiling|r.id<ceiling)) else {break;};
            if self.options.kv_admission == KvAdmissionPolicy::ReserveSequenceCapacity {
                let pending=&self.pending[position];
                let pages = Self::request_page_budget(
                    pending.prompt_token_ids.len(), &pending.generation, self.kv_cache.config().block_size,
                );
                // FIFO among ready requests prevents short arrivals from
                // indefinitely overtaking an older, larger request.
                if pages > available_pages { break; }
                available_pages -= pages;
            }
            let pending=self.pending.remove(position).expect("ready pending position");
            self.kv_cache
                .create_sequence_with_prefix(pending.id, &pending.prompt_token_ids)?;
            let admitted = self.next_admission;
            self.next_admission += 1;
            self.active.insert(
                pending.id,
                ActiveRequest {
                    eos: pending.generation.eos_token_ids.iter().copied().collect(),
                    history: pending.prompt_token_ids.clone(),
                    prompt_token_ids: pending.prompt_token_ids,
                    generation: pending.generation,
                    sampler: pending.sampler,
                    generated_token_ids: Vec::new(),
                    admitted,
                    phase: RequestPhase::NeedsPrefill,
                },
            );
        }
        Ok(())
    }

    fn schedule_decode(&mut self) -> Result<Option<ScheduledBatch>, ContinuousBatchError> {
        let candidates = self
            .active
            .iter()
            .filter_map(|(&id, request)| match request.phase {
                RequestPhase::AwaitingDecode(token) => Some((id, token)),
                _ => None,
            })
            .collect::<Vec<_>>();
        let mut rows = Vec::new();
        for (request_id, token) in candidates {
            if rows.len() >= self.config.max_batch_tokens {
                break;
            }
            if !self.kv_cache.can_append(request_id, 1)? {
                continue;
            }
            let reservation = match self.kv_cache.begin_append(request_id, 1) {
                Ok(reservation) => reservation,
                Err(error) => {
                    self.rollback_scheduled_rows(&rows)?;
                    return Err(error.into());
                }
            };
            rows.push(Self::scheduled_sequence(reservation, vec![token]));
        }
        self.finish_schedule(ScheduledBatchKind::Decode, rows)
    }

    /// Next prefill chunk of a request: `(start, length, final)`.
    fn prefill_chunk(&self, request_id: RequestId, request: &ActiveRequest) -> Result<(usize, usize, bool), ContinuousBatchError> {
        let committed = self.kv_cache.committed_tokens(request_id)?;
        let remaining = request.history.len().saturating_sub(committed);
        let length = remaining.min(self.prefill_chunk_tokens.unwrap_or(remaining)).min(self.config.max_batch_tokens);
        Ok((committed, length, committed + length == request.history.len()))
    }

    fn schedule_prefill(&mut self) -> Result<Option<ScheduledBatch>, ContinuousBatchError> {
        let mut buckets = BTreeMap::<(usize,bool), Vec<RequestId>>::new();
        for (&request_id, request) in &self.active {
            let (position, next, _) = self.prefill_chunk(request_id, request)?;
            if request.phase == RequestPhase::NeedsPrefill
                && !self.deferred_prefill.contains(&request_id)
                && next>0
                && self
                    .kv_cache
                    .can_append(request_id, next)?
            {
                buckets
                    .entry((next,position!=0))
                    .or_default()
                    .push(request_id);
            }
        }
        let Some((&(next_n,_), candidates)) =
            buckets
                .iter()
                .max_by(|(left_len, left), (right_len, right)| {
                    let left_tokens =
                        left.len().min(self.config.max_batch_tokens / left_len.0) * left_len.0;
                    let right_tokens =
                        right.len().min(self.config.max_batch_tokens / right_len.0) * right_len.0;
                    left_tokens
                        .cmp(&right_tokens)
                        // For equally full batches, keep the oldest request first.
                        .then_with(|| right[0].cmp(&left[0]))
                })
        else {
            return Ok(None);
        };
        let candidates = candidates.clone();
        let max_rows = self.config.max_batch_tokens / next_n;
        let mut rows = Vec::new();
        for request_id in candidates {
            if rows.len() >= max_rows {
                break;
            }
            if !self.kv_cache.can_append(request_id, next_n)? {
                continue;
            }
            let request = &self.active[&request_id];
            let (start, length, _) = self.prefill_chunk(request_id, request)?;
            debug_assert_eq!(length, next_n);
            let tokens = request.history[start..start + length].to_vec();
            let reservation = match self.kv_cache.begin_append(request_id, next_n) {
                Ok(reservation) => reservation,
                Err(error) => {
                    self.rollback_scheduled_rows(&rows)?;
                    return Err(error.into());
                }
            };
            rows.push(Self::scheduled_sequence(reservation, tokens));
        }
        self.finish_schedule(ScheduledBatchKind::Prefill, rows)
    }

    fn rollback_scheduled_rows(&mut self, rows: &[ScheduledSequence]) -> Result<(), ContinuousBatchError> {
        let ids = rows.iter().map(|row| row.reservation_id).collect::<Vec<_>>();
        self.kv_cache.cancel_appends(&ids)?;
        Ok(())
    }

    fn finish_schedule(
        &mut self,
        kind: ScheduledBatchKind,
        rows: Vec<ScheduledSequence>,
    ) -> Result<Option<ScheduledBatch>, ContinuousBatchError> {
        if rows.is_empty() {
            return Ok(None);
        }
        let id = self.next_batch;
        let Some(next_batch) = self.next_batch.checked_add(1) else {
            self.rollback_scheduled_rows(&rows)?;
            return Err(ContinuousBatchError("batch id overflow".into()));
        };
        if rows.iter().any(|row| !self.active.contains_key(&row.request_id)) {
            self.rollback_scheduled_rows(&rows)?;
            return Err(ContinuousBatchError("scheduled request disappeared".into()));
        }
        self.next_batch = next_batch;
        for row in &rows {
            let request = self.active.get_mut(&row.request_id).ok_or_else(|| {
                ContinuousBatchError(format!(
                    "scheduled request {} disappeared",
                    row.request_id.0
                ))
            })?;
            request.phase = RequestPhase::InFlight(id);
        }
        Ok(Some(ScheduledBatch {
            id,
            kind,
            sequences: rows,
        }))
    }

    fn scheduled_sequence(
        reservation: PagedKvAppendReservation,
        token_ids: Vec<i32>,
    ) -> ScheduledSequence {
        ScheduledSequence {
            request_id: reservation.request_id,
            token_ids,
            start_position: reservation.start_position,
            context_length: reservation.context_length,
            block_table: reservation.block_table,
            reservation_id: reservation.id,
        }
    }

    fn take_batch(&mut self, batch_id: u64) -> Result<ScheduledBatch, ContinuousBatchError> {
        let batch = self
            .in_flight
            .take()
            .ok_or_else(|| ContinuousBatchError(format!("batch {batch_id} is not in flight")))?;
        if batch.id != batch_id {
            let actual = batch.id;
            self.in_flight = Some(batch);
            return Err(ContinuousBatchError(format!(
                "batch {batch_id} cannot complete while batch {actual} is in flight"
            )));
        }
        Ok(batch)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn generation() -> GreedyGenerationConfig {
        GreedyGenerationConfig {
            max_new_tokens: 2,
            eos_token_ids: Vec::new(),
        }
    }

    fn scheduler(max_active_sequences: usize, max_batch_tokens: usize) -> ContinuousBatchScheduler {
        ContinuousBatchScheduler::new(
            ContinuousBatchConfig {
                max_active_sequences,
                max_batch_tokens,
            },
            PagedKvCacheConfig {
                block_size: 8,
                num_pages: 32,
                max_sequence_length: 32,
            },
        )
        .unwrap()
    }

    /// Drive the scheduler with a fixed token per row; returns finished requests.
    fn drain(scheduler: &mut ContinuousBatchScheduler, token: i32) -> Vec<FinishedGeneration> {
        let mut finished = Vec::new();
        while let Some(batch) = scheduler.schedule().unwrap() {
            let tokens = vec![token; batch.batch_size()];
            scheduler.complete_batch(batch.id, &tokens).unwrap();
            while let Some(done) = scheduler.pop_finished() {
                finished.push(done);
            }
        }
        finished
    }

    #[test]
    fn prefill_selects_the_shape_bucket_with_the_most_useful_tokens() {
        let mut scheduler = scheduler(3, 8);
        let oldest = scheduler.submit(vec![1, 2], generation()).unwrap();
        let first_full = scheduler.submit(vec![3, 4, 5], generation()).unwrap();
        let second_full = scheduler.submit(vec![6, 7, 8], generation()).unwrap();

        let batch = scheduler.schedule().unwrap().unwrap();

        assert_eq!(batch.kind, ScheduledBatchKind::Prefill);
        assert_eq!(batch.next_n(), 3);
        assert_eq!(
            batch
                .sequences
                .iter()
                .map(|row| row.request_id)
                .collect::<Vec<_>>(),
            vec![first_full, second_full]
        );
        assert!(!batch.sequences.iter().any(|row| row.request_id == oldest));
    }

    #[test]
    fn decode_skips_unreservable_rows_without_underfilling_the_batch() {
        let mut scheduler = ContinuousBatchScheduler::new(
            ContinuousBatchConfig {
                max_active_sequences: 2,
                max_batch_tokens: 1,
            },
            PagedKvCacheConfig {
                block_size: 2,
                num_pages: 2,
                max_sequence_length: 4,
            },
        )
        .unwrap();
        let blocked = RequestId(1);
        let runnable = RequestId(2);
        for (admitted, request_id) in [blocked, runnable].into_iter().enumerate() {
            scheduler.kv_cache.create_sequence(request_id).unwrap();
            scheduler.active.insert(
                request_id,
                ActiveRequest {
                    prompt_token_ids: vec![1],
                    generation: generation(),
                    sampler: None,
                    eos: BTreeSet::new(),
                    generated_token_ids: Vec::new(),
                    history: vec![1, 9],
                    admitted: admitted as u64,
                    phase: RequestPhase::AwaitingDecode(9),
                },
            );
        }
        let blocked_prefill = scheduler.kv_cache.begin_append(blocked, 2).unwrap();
        scheduler
            .kv_cache
            .commit_append(blocked_prefill.id)
            .unwrap();
        let runnable_prefill = scheduler.kv_cache.begin_append(runnable, 1).unwrap();
        scheduler
            .kv_cache
            .commit_append(runnable_prefill.id)
            .unwrap();

        let batch = scheduler.schedule_decode().unwrap().unwrap();

        assert_eq!(batch.batch_size(), 1);
        assert_eq!(batch.sequences[0].request_id, runnable);
    }

    #[test]
    fn long_prompt_is_prefilled_in_chunks_and_samples_only_the_last() {
        let mut scheduler = scheduler(1, 4);
        scheduler.prefill_chunk_tokens = Some(4);
        let id = scheduler.submit((1..=10).collect(), generation()).unwrap();
        let mut chunks = Vec::new();
        loop {
            let batch = scheduler.schedule().unwrap().unwrap();
            assert_eq!(batch.kind, ScheduledBatchKind::Prefill);
            let row = &batch.sequences[0];
            let last = scheduler.samples_token(row);
            chunks.push((row.start_position, row.token_ids.clone(), last));
            scheduler.complete_batch(batch.id, &[42]).unwrap();
            if last {
                break;
            }
        }
        assert_eq!(
            chunks,
            vec![
                (0, vec![1, 2, 3, 4], false),
                (4, vec![5, 6, 7, 8], false),
                (8, vec![9, 10], true),
            ]
        );
        assert_eq!(scheduler.generated_tokens(id), Some(&[42][..]));
    }

    #[test]
    fn preemption_recomputes_history_and_preserves_outputs() {
        // Two requests of 6 tokens + 4 new tokens need 5 pages each of size 2,
        // but only 6 pages exist, so the newer request is preempted.
        let mut scheduler = ContinuousBatchScheduler::new_paged(
            ContinuousBatchConfig {
                max_active_sequences: 2,
                max_batch_tokens: 16,
            },
            PagedKvCacheConfig {
                block_size: 2,
                num_pages: 6,
                max_sequence_length: 10,
            },
        )
        .unwrap();
        let generation = GreedyGenerationConfig {
            max_new_tokens: 4,
            eos_token_ids: Vec::new(),
        };
        let first = scheduler.submit(vec![1, 2, 3, 4, 5, 6], generation.clone()).unwrap();
        let second = scheduler.submit(vec![7, 8, 9, 10, 11, 12], generation).unwrap();
        let finished = drain(&mut scheduler, 3);
        assert_eq!(finished.len(), 2);
        for id in [first, second] {
            let done = finished.iter().find(|done| done.request_id == id).unwrap();
            assert_eq!(done.generated_token_ids, vec![3, 3, 3, 3]);
        }
        let snapshot = scheduler.paged_snapshot();
        assert!(snapshot.preemptions > 0);
        assert_eq!(snapshot.free_kv_pages, 6);
    }

    #[test]
    fn ngram_draft_uses_the_most_recent_match() {
        assert_eq!(ngram_draft(&[1, 2, 3, 9, 1, 2, 4, 1, 2], 2, 3), vec![4, 1, 2]);
        assert_eq!(ngram_draft(&[1, 2, 3], 2, 3), Vec::<i32>::new());
        assert_eq!(ngram_draft(&[7, 8, 7], 3, 2), vec![8, 7]);
    }

    #[test]
    fn verification_commits_only_the_accepted_draft_prefix() {
        let mut scheduler = scheduler(1, 16);
        scheduler
            .set_speculative_decoding(Some(SpeculativeConfig {
                draft_tokens: 3,
                ngram: 2,
            }))
            .unwrap();
        let id = scheduler
            .submit(
                vec![5, 6, 7, 8, 5, 6],
                GreedyGenerationConfig {
                    max_new_tokens: 8,
                    eos_token_ids: Vec::new(),
                },
            )
            .unwrap();
        let prefill = scheduler.schedule().unwrap().unwrap();
        assert_eq!(prefill.kind, ScheduledBatchKind::Prefill);
        scheduler.complete_batch(prefill.id, &[7]).unwrap();
        let verify = scheduler.schedule().unwrap().unwrap();
        assert_eq!(verify.kind, ScheduledBatchKind::Verify);
        assert_eq!(verify.sequences[0].token_ids, vec![7, 8, 5, 6]);
        let appended = scheduler
            .complete_verify_batch(verify.id, &[8, 5, 9, 1])
            .unwrap();
        assert_eq!(appended, vec![(id, vec![8, 5, 9])]);
        assert_eq!(scheduler.kv_cache.committed_tokens(id).unwrap(), 9);
        assert_eq!(scheduler.generated_tokens(id), Some(&[7, 8, 5, 9][..]));
        let snapshot = scheduler.paged_snapshot();
        assert_eq!((snapshot.drafted_tokens, snapshot.accepted_draft_tokens), (3, 2));
        let next = scheduler.schedule().unwrap().unwrap();
        assert_eq!(next.kind, ScheduledBatchKind::Decode);
        assert_eq!(next.sequences[0].token_ids, vec![9]);
    }

    #[test]
    fn prefix_cache_skips_shared_full_blocks() {
        let mut scheduler = scheduler(1, 32);
        scheduler.set_prefix_caching(true);
        let prompt: Vec<i32> = (1..=20).collect();
        scheduler.submit(prompt.clone(), generation()).unwrap();
        drain(&mut scheduler, 5);
        let mut second = prompt.clone();
        second.push(99);
        scheduler.submit(second, generation()).unwrap();
        let batch = scheduler.schedule().unwrap().unwrap();
        // 21 tokens, block 8: two full cached blocks are reused.
        assert_eq!(batch.sequences[0].start_position, 16);
        assert_eq!(batch.sequences[0].token_ids, vec![17, 18, 19, 20, 99]);
        assert_eq!(scheduler.paged_snapshot().prefix_cached_tokens, 16);
    }
}
