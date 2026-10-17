use crate::llama::rope::RotaryState;
use crate::{GenerationError, KvPageId, RequestId, ScheduledBatch};
use ruda_tensor::Backend;
use ruda_tensor::api::{DType, Tensor};
use std::collections::BTreeMap;

struct PendingRow {
    id: RequestId,
    start: usize,
    pages: Vec<KvPageId>,
    contexts: Vec<Vec<usize>>,
}

/// Rotary contexts follow physical pages for shared prefixes, and request
/// history for recomputation. Speculative contexts are published only for
/// the accepted prefix, including the original verification context length.
pub(super) struct PagedRotaryCache {
    block_size: usize,
    slots: Vec<Vec<usize>>,
    requests: BTreeMap<RequestId, Vec<Vec<usize>>>,
    pending: BTreeMap<u64, Vec<PendingRow>>,
}

impl PagedRotaryCache {
    pub(super) fn new(states: &[Option<RotaryState>], pages: usize, block_size: usize) -> Self {
        Self {
            block_size,
            slots: states
                .iter()
                .map(|state| {
                    if state.is_some() {
                        vec![0; pages * block_size]
                    } else {
                        Vec::new()
                    }
                })
                .collect(),
            requests: BTreeMap::new(),
            pending: BTreeMap::new(),
        }
    }

    pub(super) fn prepare<B: Backend>(
        &mut self,
        states: &[Option<RotaryState>],
        batch: &ScheduledBatch,
        dtype: DType,
        device: &B::Device,
    ) -> Result<Vec<Option<Tensor<B, 3>>>, GenerationError> {
        let mut tables = (0..states.len()).map(|_| Vec::new()).collect::<Vec<_>>();
        let mut pending = Vec::with_capacity(batch.sequences.len());
        for row in &batch.sequences {
            let end = row.start_position + row.token_ids.len();
            let mut layers = Vec::with_capacity(states.len());
            for (layer, state) in states.iter().enumerate() {
                let Some(state) = state else {
                    layers.push(Vec::new());
                    continue;
                };
                let mut history = self
                    .requests
                    .get(&row.request_id)
                    .and_then(|layers| layers.get(layer))
                    .cloned()
                    .unwrap_or_default();
                while history.len() < row.start_position {
                    let position = history.len();
                    let page = row
                        .block_table
                        .get(position / self.block_size)
                        .ok_or_else(|| {
                            GenerationError("rotary prefix has no physical page".into())
                        })?
                        .0 as usize;
                    let context =
                        self.slots[layer][page * self.block_size + position % self.block_size];
                    if context == 0 {
                        return Err(GenerationError(
                            "shared prefix has no committed rotary history".into(),
                        ));
                    }
                    history.push(context);
                }
                if history.len() < end {
                    let context = state.paged_context(history.last().copied(), end);
                    history.resize(end, context);
                }
                tables[layer].push(state.paged_table(
                    row.start_position,
                    &history[row.start_position..end],
                    dtype,
                    device,
                ));
                layers.push(history);
            }
            pending.push(PendingRow {
                id: row.request_id,
                start: row.start_position,
                pages: row.block_table.clone(),
                contexts: layers,
            });
        }
        if states.iter().any(Option::is_some) {
            self.pending.insert(batch.id, pending);
        }
        Ok(tables
            .into_iter()
            .map(|tables| {
                if tables.is_empty() {
                    None
                } else {
                    Some(Tensor::cat(tables, 0))
                }
            })
            .collect())
    }

    pub(super) fn commit(
        &mut self,
        batch: &ScheduledBatch,
        counts: &[usize],
    ) -> Result<(), GenerationError> {
        if counts.len() != batch.batch_size()
            || counts
                .iter()
                .zip(&batch.sequences)
                .any(|(&count, row)| count > row.token_ids.len())
        {
            return Err(GenerationError(
                "rotary commit counts differ from the executed batch".into(),
            ));
        }
        let Some(pending) = self.pending.remove(&batch.id) else {
            return Ok(());
        };
        for (row, &count) in pending.into_iter().zip(counts) {
            let end = row.start + count;
            let histories = self
                .requests
                .entry(row.id)
                .or_insert_with(|| vec![Vec::new(); self.slots.len()]);
            for (layer, contexts) in row.contexts.iter().enumerate() {
                if contexts.is_empty() {
                    continue;
                }
                if histories[layer].len() < end {
                    histories[layer].resize(end, 0);
                }
                histories[layer][..end].copy_from_slice(&contexts[..end]);
                for position in row.start..end {
                    let slot = row.pages[position / self.block_size].0 as usize * self.block_size
                        + position % self.block_size;
                    self.slots[layer][slot] = contexts[position];
                }
            }
        }
        Ok(())
    }

    pub(super) fn rollback(&mut self, batch: u64) {
        self.pending.remove(&batch);
    }
    pub(super) fn forget(&mut self, request: RequestId) {
        self.requests.remove(&request);
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::llama::rope::{RopeParameters, RopeScaling};
    use crate::{KvReservationId, ScheduledBatchKind, ScheduledSequence};
    use ruda_tensor_host::Host;

    fn batch(id: u64, request: u64, start: usize, count: usize) -> ScheduledBatch {
        ScheduledBatch {
            id,
            kind: ScheduledBatchKind::Verify,
            sequences: vec![ScheduledSequence {
                request_id: RequestId(request),
                token_ids: vec![1; count],
                start_position: start,
                context_length: start + count,
                block_table: vec![KvPageId(0), KvPageId(1), KvPageId(2)],
                reservation_id: KvReservationId(id),
            }],
        }
    }

    #[test]
    fn dynamic_prefix_partial_acceptance_rollback_and_recompute_preserve_contexts() {
        let state = RopeParameters {
            theta: 10000.0,
            inverse_frequencies: vec![1.0, 0.01],
            attention_factor: 1.0,
            scaling: RopeScaling::Dynamic {
                factor: 2.0,
                original_length: 4,
            },
            per_layer: vec![],
        }
        .state();
        let states = vec![Some(state)];
        let device = Default::default();
        let mut cache = PagedRotaryCache::new(&states, 3, 4);
        let first = batch(1, 1, 0, 3);
        cache
            .prepare::<Host>(&states, &first, DType::F32, &device)
            .unwrap();
        cache.commit(&first, &[3]).unwrap();
        let speculative = batch(2, 1, 3, 5);
        cache
            .prepare::<Host>(&states, &speculative, DType::F32, &device)
            .unwrap();
        cache.commit(&speculative, &[2]).unwrap();
        assert_eq!(cache.requests[&RequestId(1)][0], [4, 4, 4, 8, 8]);
        let failed = batch(3, 1, 5, 4);
        cache
            .prepare::<Host>(&states, &failed, DType::F32, &device)
            .unwrap();
        cache.rollback(3);
        assert_eq!(cache.requests[&RequestId(1)][0], [4, 4, 4, 8, 8]);
        let shared = batch(4, 2, 4, 1);
        cache
            .prepare::<Host>(&states, &shared, DType::F32, &device)
            .unwrap();
        cache.commit(&shared, &[1]).unwrap();
        assert_eq!(cache.requests[&RequestId(2)][0], [4, 4, 4, 8, 8]);
        // Recompute must retain the original contexts beyond this chunk.
        let recompute = batch(5, 1, 0, 2);
        cache
            .prepare::<Host>(&states, &recompute, DType::F32, &device)
            .unwrap();
        cache.commit(&recompute, &[2]).unwrap();
        assert_eq!(cache.requests[&RequestId(1)][0], [4, 4, 4, 8, 8]);
        let continuation = batch(6, 1, 5, 1);
        cache
            .prepare::<Host>(&states, &continuation, DType::F32, &device)
            .unwrap();
        cache.commit(&continuation, &[1]).unwrap();
        assert_eq!(cache.requests[&RequestId(1)][0], [4, 4, 4, 8, 8, 8]);
        cache.forget(RequestId(1));
        assert!(!cache.requests.contains_key(&RequestId(1)));
    }
}
