use super::*;
use crate::{DeviceBatchModel, PrefixBatchModel, GenerationCacheContinuation};
use ruda_model::record::{PrecisionSettings, Record, RecorderError};
use ruda_tensor::{device::Device, distributed::{DistributedBackend, ReduceOperation}};

pub struct KimiLinearExpertParallel<B: DistributedBackend> {
    pub config: KimiLinearConfig, ranks: Vec<KimiLinearModel<B>>, devices: Vec<B::Device>,
}
#[derive(Clone)]
pub struct KimiLinearExpertParallelCache<B: DistributedBackend> {ranks: Vec<KimiLinearCache<B>>, devices: Vec<B::Device>}
pub struct KimiLinearExpertParallelCacheRecord<B: DistributedBackend> {ranks: Vec<KimiLinearCacheRecord<B>>}
pub struct LoadedKimiLinearExpertParallel<B: DistributedBackend> {
    pub model: KimiLinearExpertParallel<B>, pub eos_token_ids: Vec<i32>, pub reports: Vec<HuggingFaceLoadReport>,
}
pub fn load_huggingface_kimi_linear_expert_parallel<B: DistributedBackend + super::super::super::MlaFp8Backend>(
    directory: impl AsRef<Path>, devices: Vec<B::Device>) -> Result<LoadedKimiLinearExpertParallel<B>, HuggingFaceLoadError> {
    let ids = devices.iter().map(Device::to_id).collect::<Vec<_>>();
    if ids.is_empty() || ids.iter().enumerate().any(|(i, id)| ids[..i].contains(id)) {
        return Err(HuggingFaceLoadError("Kimi Linear EP requires distinct nonempty devices".into()));
    }
    let mut ranks = Vec::with_capacity(devices.len()); let mut reports = Vec::with_capacity(devices.len()); let mut eos = None;
    for (rank, device) in devices.iter().enumerate() {
        let group = ids.clone();
        let reduction = Arc::new(move |value: Tensor<B, 3>| {
            if group.len() == 1 { return value; }
            Tensor::from_primitive(TensorPrimitive::Float(B::all_reduce(value.into_primitive().tensor(), ReduceOperation::Sum, group.clone()).resolve()))
        });
        let loaded = loading::load_rank(directory.as_ref(), device, Some((rank, devices.len())), Some(reduction))?;
        if eos.as_ref().is_some_and(|ids| *ids != loaded.eos_token_ids) { return Err(HuggingFaceLoadError("Kimi Linear EOS changed during EP loading".into())); }
        eos = Some(loaded.eos_token_ids); ranks.push(loaded.model); reports.push(loaded.report);
    }
    Ok(LoadedKimiLinearExpertParallel {model: KimiLinearExpertParallel {config: ranks[0].config.clone(), ranks, devices},
        eos_token_ids: eos.unwrap_or_default(), reports})
}
impl<B: DistributedBackend> KimiLinearExpertParallel<B> {
    pub fn devices(&self) -> &[B::Device] { &self.devices }
    pub fn limits(&self) -> CausalModelLimits { self.config.limits() }
}
impl<B: DistributedBackend> KimiLinearExpertParallel<B>
where KimiLinearModel<B>: CausalModel<B, Cache = KimiLinearCache<B>> + SpeculativeModel<B> + Sync {
    fn forward_block(&self, tokens: Tensor<B, 2, Int>, cache: &mut KimiLinearExpertParallelCache<B>, last: bool)
        -> Result<Tensor<B, 3>, GenerationError> {
        let [b, s] = tokens.dims(); let position = cache.generation_position();
        if b == 0 || s == 0 || self.devices.first() != Some(&tokens.device()) || cache.devices != self.devices
            || cache.ranks.len() != self.ranks.len() || position.checked_add(s).is_none_or(|n| n > self.config.model_max_length) {
            return Err(error("Kimi Linear EP topology/input/cache differs"));
        }
        for (state, model) in cache.ranks.iter().zip(&self.ranks) {
            state.validate(&model.config, model.embedding.dtype())?;
            if state.position != position || state.batch.is_some_and(|old| old != b) { return Err(error("Kimi Linear EP rank positions differ")); }
        }
        let data = tokens.try_into_data().map_err(error)?;
        std::thread::scope(|scope| {
            let mut handles = Vec::with_capacity(self.ranks.len());
            for ((model, state), device) in self.ranks.iter().zip(&mut cache.ranks).zip(&self.devices) {
                let data = data.clone();
                handles.push(scope.spawn(move || {
                    let tokens = Tensor::from_data(data, device);
                    let result = if last { model.try_forward_cached_last(tokens, state) } else { model.try_forward_cached_block(tokens, state) };
                    B::sync(device).map_err(error)?; result
                }));
            }
            let mut root = None; let mut failure = None;
            for (rank, handle) in handles.into_iter().enumerate() {
                match handle.join() {
                    Ok(Ok(output)) if rank == 0 => root = Some(output), Ok(Ok(_)) => (),
                    Ok(Err(e)) => failure = Some(e), Err(_) => failure = Some(error("Kimi Linear EP rank panicked")),
                }
            }
            if let Some(e) = failure { Err(e) } else { root.ok_or_else(|| error("Kimi Linear EP root missing")) }
        })
    }
}
impl<B: DistributedBackend> CausalModel<B> for KimiLinearExpertParallel<B>
where KimiLinearModel<B>: CausalModel<B, Cache = KimiLinearCache<B>> + SpeculativeModel<B> + Sync {
    type Cache = KimiLinearExpertParallelCache<B>;
    fn new_cache(&self) -> Self::Cache { Self::Cache {ranks: self.ranks.iter().map(CausalModel::new_cache).collect(), devices: self.devices.clone()} }
    fn forward_cached_last(&self, tokens: Tensor<B, 2, Int>, cache: &mut Self::Cache) -> Tensor<B, 3> {
        self.forward_block(tokens, cache, true).expect("Kimi Linear EP forward")
    }
    fn try_forward_cached_last(&self, tokens: Tensor<B, 2, Int>, cache: &mut Self::Cache) -> Result<Tensor<B, 3>, GenerationError> {
        self.forward_block(tokens, cache, true)
    }
}
impl<B: DistributedBackend> SpeculativeModel<B> for KimiLinearExpertParallel<B>
where KimiLinearModel<B>: CausalModel<B, Cache = KimiLinearCache<B>> + SpeculativeModel<B> + Sync {
    fn fork_cache(&self, cache: &Self::Cache) -> Self::Cache { cache.clone() }
    fn try_forward_cached_block(&self, tokens: Tensor<B, 2, Int>, cache: &mut Self::Cache) -> Result<Tensor<B, 3>, GenerationError> {
        self.forward_block(tokens, cache, false)
    }
}
impl<B: DistributedBackend> Record<B> for KimiLinearExpertParallelCacheRecord<B> {
    type Item<S: PrecisionSettings> = Vec<<KimiLinearCacheRecord<B> as Record<B>>::Item<S>>;
    fn into_item<S: PrecisionSettings>(self) -> Self::Item<S> { self.ranks.into_iter().map(|r| r.into_item::<S>()).collect() }
    fn from_item<S: PrecisionSettings>(item: Self::Item<S>, device: &B::Device) -> Self {
        Self {ranks: item.into_iter().map(|r| KimiLinearCacheRecord::from_item::<S>(r, device)).collect()}
    }
}
impl<B: DistributedBackend> GenerationCacheContinuation<B> for KimiLinearExpertParallelCache<B> {
    type Record = KimiLinearExpertParallelCacheRecord<B>;
    fn generation_position(&self) -> usize { self.ranks.first().map_or(0, |s| s.position) }
    fn capture_generation(&self, model_id: &str, device: &B::Device) -> Result<Self::Record, RecorderError> {
        if model_id.is_empty() || self.devices.first() != Some(device) || self.devices.len() != self.ranks.len()
            || self.ranks.iter().any(|s| s.position != self.generation_position()) {
            return Err(RecorderError::Unknown("Kimi Linear EP continuation topology differs".into()));
        }
        let world = self.ranks.len();
        let ranks = self.ranks.iter().enumerate().map(|(rank, state)| state.record(&format!("{model_id}:ep-{world}-rank-{rank}")))
            .collect::<Result<Vec<_>, _>>()?;
        Ok(Self::Record {ranks})
    }
    fn restore_generation(record: Self::Record, model_id: &str, template: &Self, position: usize, device: &B::Device) -> Result<Self, RecorderError> {
        let world = template.ranks.len();
        if model_id.is_empty() || world == 0 || record.ranks.len() != world || template.devices.len() != world || template.devices.first() != Some(device) {
            return Err(RecorderError::Unknown("Kimi Linear EP continuation world differs".into()));
        }
        let ranks = record.ranks.into_iter().zip(&template.ranks).zip(&template.devices).enumerate().map(|(rank, ((record, state), device))|
            KimiLinearCache::restore_generation(record, &format!("{model_id}:ep-{world}-rank-{rank}"), state, position, device))
            .collect::<Result<Vec<_>, _>>()?;
        Ok(Self {ranks, devices: template.devices.clone()})
    }
}
crate::parallel_batch::parallel_batch_model!(KimiLinearExpertParallel, KimiLinearModel);
crate::parallel_batch::parallel_speculative_model!(KimiLinearExpertParallel, KimiLinearModel);
