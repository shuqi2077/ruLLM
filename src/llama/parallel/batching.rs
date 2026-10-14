use super::*;
use crate::{ContinuousBatchScheduler, DenseBatchCache, PagedKvCacheConfig, ScheduledBatch};

crate::batch_kv::dense_prefix_model!(LlamaTensorParallelRank, |rank| rank.model.layers.iter().map(|layer| {
    let attention = &layer.self_attn;
    (attention.num_kv_heads, attention.head_dimension, attention.head_dimension,
        rank.model.embed_tokens.weight.val().dtype())
}).collect::<Vec<_>>());
crate::batch_kv::dense_speculative_model!(LlamaTensorParallelRank);
crate::parallel_batch::parallel_batch_model!(LlamaTensorParallel, LlamaTensorParallelRank,|group|
    crate::PrefixShard{kv_heads:Some(group.config.num_kv_heads),..Default::default()});
crate::parallel_batch::parallel_speculative_model!(LlamaTensorParallel, LlamaTensorParallelRank);

impl<R, F, I, BT> DeviceBatchModel<DeviceBackend<R, F, I, BT>> for LlamaTensorParallelRank<DeviceBackend<R, F, I, BT>>
where R: DeviceRuntime, R::Server: ComputeServer, R::Device: DeviceOps,
    F: FloatElement, I: IntElement, BT: BoolElement {
    type Cache = DenseBatchCache<DeviceBackend<R, F, I, BT>>;
    fn batch_limits(&self) -> CausalModelLimits { self.model.batch_limits() }
    fn new_batch_cache(&self, config: PagedKvCacheConfig) -> Self::Cache { self.model.new_batch_cache(config) }
    fn fork_batch_cache(&self, cache: &Self::Cache) -> Self::Cache { cache.clone() }
    fn retain_batch_cache(&self, cache: &mut Self::Cache, scheduler: &ContinuousBatchScheduler) {
        self.model.retain_batch_cache(cache, scheduler);
    }
    fn forward_batch(&self, batch: &ScheduledBatch, cache: &mut Self::Cache, device: &R::Device)
        -> Result<Tensor<DeviceBackend<R, F, I, BT>, 3>, GenerationError> {
        self.forward_batch_logits(batch, cache, device, true)
    }
}

impl<R, F, I, BT> LlamaTensorParallelRank<DeviceBackend<R, F, I, BT>>
where R: DeviceRuntime, R::Server: ComputeServer, R::Device: DeviceOps,
    F: FloatElement, I: IntElement, BT: BoolElement {
    fn forward_batch_logits(&self, batch: &ScheduledBatch, cache: &mut DenseBatchCache<DeviceBackend<R, F, I, BT>>,
        device: &R::Device, last_only: bool) -> Result<Tensor<DeviceBackend<R, F, I, BT>, 3>, GenerationError> {
        self.model.forward_batch_logits_with_reduction(batch, cache, device, last_only, Some(self.reduction.as_ref()))
    }
}
