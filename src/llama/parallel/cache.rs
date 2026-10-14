use super::*;
use ruda_model::record::{PrecisionSettings, Record, RecorderError};

#[derive(Clone, Debug)]
pub struct LlamaTensorParallelCache<B: DistributedBackend> {
    pub(super) ranks: Vec<LlamaKvCache<B>>,
    pub(super) devices: Vec<B::Device>,
}

pub struct LlamaTensorParallelCacheRecord<B: DistributedBackend> {
    ranks: Vec<LlamaKvCacheRecord<B>>,
}

impl<B: DistributedBackend> Record<B> for LlamaTensorParallelCacheRecord<B> {
    type Item<S: PrecisionSettings> = Vec<<LlamaKvCacheRecord<B> as Record<B>>::Item<S>>;
    fn into_item<S: PrecisionSettings>(self) -> Self::Item<S> {
        self.ranks.into_iter().map(|rank| rank.into_item::<S>()).collect()
    }
    fn from_item<S: PrecisionSettings>(item: Self::Item<S>, device: &B::Device) -> Self {
        Self { ranks: item.into_iter().map(|rank| LlamaKvCacheRecord::from_item::<S>(rank, device)).collect() }
    }
}

fn rank_identity(model_id: &str, rank: usize, world: usize) -> String {
    format!("{model_id}:llama-qwen2-tp:{world}:{rank}")
}

impl<B: DistributedBackend> GenerationCacheContinuation<B> for LlamaTensorParallelCache<B> {
    type Record = LlamaTensorParallelCacheRecord<B>;
    fn generation_position(&self) -> usize { self.ranks.first().map_or(0, LlamaKvCache::position) }
    fn capture_generation(&self, model_id: &str, device: &B::Device) -> Result<Self::Record, RecorderError> {
        let position = self.generation_position();
        if model_id.is_empty() || self.ranks.is_empty() || self.ranks.len() != self.devices.len()
            || self.devices.first() != Some(device) || self.ranks.iter().any(|rank| rank.position() != position) {
            return Err(RecorderError::Unknown("incomplete Llama/Qwen2 TP cache group".into()));
        }
        let ranks = self.ranks.iter().zip(&self.devices).enumerate().map(|(rank, (cache, device))| {
            B::sync(device).map_err(|e| RecorderError::Unknown(e.to_string()))?;
            cache.capture_generation(&rank_identity(model_id, rank, self.ranks.len()), device)
        }).collect::<Result<Vec<_>, _>>()?;
        Ok(LlamaTensorParallelCacheRecord { ranks })
    }
    fn restore_generation(record: Self::Record, model_id: &str, template: &Self, position: usize, device: &B::Device)
        -> Result<Self, RecorderError> {
        let world = template.ranks.len();
        if model_id.is_empty() || world == 0 || record.ranks.len() != world
            || template.devices.len() != world || template.devices.first() != Some(device) {
            return Err(RecorderError::Unknown("Llama/Qwen2 TP record topology differs".into()));
        }
        let ranks = record.ranks.into_iter().zip(&template.ranks).zip(&template.devices).enumerate()
            .map(|(rank, ((record, cache), device))| {
                LlamaKvCache::restore_generation(record, &rank_identity(model_id, rank, world), cache, position, device)
            }).collect::<Result<Vec<_>, _>>()?;
        Ok(Self { ranks, devices: template.devices.clone() })
    }
}
