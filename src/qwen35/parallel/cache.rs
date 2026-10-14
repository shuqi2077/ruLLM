use super::*;
use ruda_model::record::{PrecisionSettings, Record, RecorderError};

#[derive(Clone)]
pub struct Qwen35TensorParallelCache<B: DistributedBackend> {
    pub(super) ranks: Vec<Qwen35Cache<B>>,
    pub(super) devices: Vec<B::Device>,
}

pub struct Qwen35TensorParallelCacheRecord<B: DistributedBackend> {
    ranks: Vec<Qwen35CacheRecord<B>>,
}

impl<B: DistributedBackend> Record<B> for Qwen35TensorParallelCacheRecord<B> {
    type Item<S: PrecisionSettings> = Vec<<Qwen35CacheRecord<B> as Record<B>>::Item<S>>;
    fn into_item<S: PrecisionSettings>(self) -> Self::Item<S> {
        self.ranks.into_iter().map(|rank| rank.into_item::<S>()).collect()
    }
    fn from_item<S: PrecisionSettings>(item: Self::Item<S>, device: &B::Device) -> Self {
        Self { ranks: item.into_iter().map(|rank| Qwen35CacheRecord::from_item::<S>(rank, device)).collect() }
    }
}

fn rank_identity(model_id: &str, rank: usize, world: usize) -> String {
    format!("{model_id}:qwen35-tp:{world}:{rank}")
}

impl<B: DistributedBackend> GenerationCacheContinuation<B> for Qwen35TensorParallelCache<B> {
    type Record = Qwen35TensorParallelCacheRecord<B>;
    fn generation_position(&self) -> usize { self.ranks.first().map_or(0, Qwen35Cache::sequence_length) }
    fn capture_generation(&self, model_id: &str, device: &B::Device) -> Result<Self::Record, RecorderError> {
        let position = self.generation_position();
        if model_id.is_empty() || self.ranks.is_empty() || self.ranks.len() != self.devices.len()
            || self.devices.first() != Some(device) || self.ranks.iter().any(|rank| rank.position != position || rank.batch != self.ranks[0].batch) {
            return Err(RecorderError::Unknown("incomplete Qwen3.5 TP cache group".into()));
        }
        let ranks = self.ranks.iter().zip(&self.devices).enumerate().map(|(rank, (cache, device))| {
            B::sync(device).map_err(|e| RecorderError::Unknown(e.to_string()))?;
            cache.capture_generation(&rank_identity(model_id, rank, self.ranks.len()), device)
        }).collect::<Result<Vec<_>, _>>()?;
        Ok(Qwen35TensorParallelCacheRecord { ranks })
    }
    fn restore_generation(record: Self::Record, model_id: &str, template: &Self, position: usize, device: &B::Device)
        -> Result<Self, RecorderError> {
        let world = template.ranks.len();
        if model_id.is_empty() || world == 0 || record.ranks.len() != world
            || template.devices.len() != world || template.devices.first() != Some(device) {
            return Err(RecorderError::Unknown("Qwen3.5 TP record topology differs".into()));
        }
        let ranks = record.ranks.into_iter().zip(&template.ranks).zip(&template.devices).enumerate()
            .map(|(rank, ((record, cache), device))| Qwen35Cache::restore_generation(
                record, &rank_identity(model_id, rank, world), cache, position, device,
            )).collect::<Result<Vec<_>, _>>()?;
        if ranks.iter().any(|rank| rank.batch != ranks[0].batch) {
            return Err(RecorderError::Unknown("Qwen3.5 TP record batch sizes differ".into()));
        }
        Ok(Self { ranks, devices: template.devices.clone() })
    }
}
