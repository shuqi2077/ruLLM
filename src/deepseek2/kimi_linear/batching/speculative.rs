use super::*;

#[derive(Clone)]
pub(super) struct Trajectory<B: Backend> {
    pub start: usize, pub end: usize, pub initial: Vec<kda::State<B>>, pub steps: Vec<Vec<kda::State<B>>>,
}
#[derive(Clone)]
pub struct KimiLinearBatchDelta<B: Backend> {start: usize, dense: DenseBatchDelta<B>, steps: Vec<Vec<kda::State<B>>>}

impl<R, F, I, BT> SpeculativeBatchModel<DeviceBackend<R, F, I, BT>> for KimiLinearModel<DeviceBackend<R, F, I, BT>>
where R: DeviceRuntime, R::Server: ComputeServer, R::Device: DeviceOps, F: FloatElement, I: IntElement, BT: BoolElement {
    type Delta = KimiLinearBatchDelta<DeviceBackend<R, F, I, BT>>;
    fn begin_batch_speculation(&self, cache: &mut Self::Cache, batch: &ScheduledBatch) -> Result<(), GenerationError> {
        for row in &batch.sequences { cache.trajectories.remove(&row.request_id); } Ok(())
    }
    fn forward_batch_block(&self, batch: &ScheduledBatch, cache: &mut Self::Cache, device: &R::Device)
        -> Result<Tensor<DeviceBackend<R, F, I, BT>, 3>, GenerationError> { self.forward_batch_impl(batch, cache, device, true) }
    fn capture_batch_delta(&self, cache: &Self::Cache, id: RequestId, start: usize, end: usize) -> Result<Self::Delta, GenerationError> {
        let history = cache.trajectories.get(&id).ok_or_else(|| error("missing Kimi Linear speculative history"))?;
        if start >= end || start < history.start || end > history.end || history.steps.len() != history.end - history.start
            || cache.sequence_length(id) != Some(history.end) { return Err(error("Kimi Linear speculative span differs")); }
        let steps = history.steps[start - history.start..end - history.start].to_vec();
        for step in &steps { self.validate_recurrent(step, true)?; }
        Ok(Self::Delta {start, dense: cache.dense.capture_delta(id, start, end)?, steps})
    }
    fn truncate_batch_cache(&self, cache: &mut Self::Cache, id: RequestId, position: usize) -> Result<(), GenerationError> {
        let history = cache.trajectories.get(&id).ok_or_else(|| error("missing Kimi Linear truncate history"))?;
        if position < history.start || position > history.end || cache.sequence_length(id) != Some(history.end) {
            return Err(error("Kimi Linear truncate position differs"));
        }
        let state = if position == history.start { history.initial.clone() }
            else { history.steps.get(position - history.start - 1).ok_or_else(|| error("incomplete Kimi Linear truncate history"))?.clone() };
        self.validate_recurrent(&state, position > 0)?;
        cache.dense.truncate(id, position)?; cache.recurrent.insert(id, state); cache.trajectories.remove(&id); Ok(())
    }
    fn append_batch_delta(&self, cache: &mut Self::Cache, id: RequestId, table: &[KvPageId], delta: &Self::Delta, offset: usize)
        -> Result<(), GenerationError> {
        let step = delta.steps.get(offset).ok_or_else(|| error("Kimi Linear delta offset exceeds history"))?;
        self.validate_recurrent(step, true)?;
        if delta.start.checked_add(offset) != cache.sequence_length(id) || !cache.recurrent.contains_key(&id) {
            return Err(error("Kimi Linear delta destination differs"));
        }
        cache.dense.append_delta(id, table, &delta.dense, offset)?;
        cache.recurrent.insert(id, step.clone()); cache.trajectories.remove(&id); Ok(())
    }
}
