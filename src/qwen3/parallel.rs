use super::{Qwen3Config, Qwen3Model, Shard, load_rank};
use crate::{
    CausalModel, GenerationCacheContinuation, GenerationError, HuggingFaceLoadError,
    HuggingFaceLoadReport, LlamaKvCache, LlamaKvCacheRecord, SpeculativeModel,
};
use ruda_model::record::{PrecisionSettings, Record, RecorderError};
use ruda_tensor::api::{Int, Tensor, TensorPrimitive};
use ruda_tensor::device::Device;
use ruda_tensor::distributed::{DistributedBackend, ReduceOperation};
use std::{path::Path, sync::Arc};

pub struct Qwen3TensorParallel<B: DistributedBackend> {
    pub config: Qwen3Config,
    ranks: Vec<Qwen3Model<B>>,
    devices: Vec<B::Device>,
}

#[derive(Clone, Debug)]
pub struct Qwen3TensorParallelCache<B: DistributedBackend> {
    ranks: Vec<LlamaKvCache<B>>,
    devices: Vec<B::Device>,
}

pub struct Qwen3TensorParallelCacheRecord<B: DistributedBackend> {
    ranks: Vec<LlamaKvCacheRecord<B>>,
}

pub struct LoadedQwen3TensorParallel<B: DistributedBackend> {
    pub model: Qwen3TensorParallel<B>,
    pub eos_token_ids: Vec<i32>,
    pub reports: Vec<HuggingFaceLoadReport>,
}

pub fn load_huggingface_qwen3_tensor_parallel<B: DistributedBackend>(
    directory: impl AsRef<Path>,
    devices: Vec<B::Device>,
) -> Result<LoadedQwen3TensorParallel<B>, HuggingFaceLoadError> {
    let ids = devices.iter().map(Device::to_id).collect::<Vec<_>>();
    if devices.is_empty() || ids.iter().enumerate().any(|(i, id)| ids[..i].contains(id)) {
        return Err(HuggingFaceLoadError(
            "TP requires a nonempty group of distinct native devices".into(),
        ));
    }
    let config: Qwen3Config = serde_json::from_slice(
        &std::fs::read(directory.as_ref().join("config.json"))
            .map_err(|e| HuggingFaceLoadError(e.to_string()))?,
    )
    .map_err(|e| HuggingFaceLoadError(e.to_string()))?;
    config.validate()?;
    let mut ranks = Vec::with_capacity(devices.len());
    let mut reports = Vec::with_capacity(devices.len());
    let mut eos_token_ids = None;
    for (rank, device) in devices.iter().enumerate() {
        let group = ids.clone();
        let reduction = Arc::new(move |value: Tensor<B, 3>| {
            let result = B::all_reduce(
                value.into_primitive().tensor(),
                ReduceOperation::Sum,
                group.clone(),
            )
            .resolve();
            Tensor::from_primitive(TensorPrimitive::Float(result))
        });
        let loaded = load_rank(
            directory.as_ref(),
            device,
            Some(Shard {
                rank,
                world: devices.len(),
            }),
            Some(reduction),
        )?;
        if eos_token_ids
            .as_ref()
            .is_some_and(|ids| *ids != loaded.eos_token_ids)
        {
            return Err(HuggingFaceLoadError(
                "Qwen3 EOS configuration changed during TP loading".into(),
            ));
        }
        eos_token_ids = Some(loaded.eos_token_ids);
        ranks.push(loaded.model);
        reports.push(loaded.report);
    }
    Ok(LoadedQwen3TensorParallel {
        model: Qwen3TensorParallel {
            config,
            ranks,
            devices,
        },
        eos_token_ids: eos_token_ids.unwrap_or_default(),
        reports,
    })
}

impl<B: DistributedBackend> Qwen3TensorParallel<B> {
    pub fn devices(&self) -> &[B::Device] {
        &self.devices
    }

    fn forward_block(
        &self,
        tokens: Tensor<B, 2, Int>,
        cache: &mut Qwen3TensorParallelCache<B>,
        last_only: bool,
    ) -> Result<Tensor<B, 3>, GenerationError> {
        let [batch, sequence] = tokens.dims();
        let position = cache.generation_position();
        if cache.ranks.len() != self.ranks.len()
            || cache.devices != self.devices
            || tokens.device() != self.devices[0]
            || batch != 1
            || sequence == 0
            || position
                .checked_add(sequence)
                .is_none_or(|end| end > self.config.max_position_embeddings)
            || cache.ranks.iter().any(|state| {
                state.position() != position
                    || state.capacity() != self.config.max_position_embeddings
                    || state.layers.len() != self.config.num_hidden_layers
                    || state
                        .layers
                        .iter()
                        .any(|layer| layer.sequence_length() != position)
            })
        {
            return Err(GenerationError(
                "incompatible TP input, devices, cache or context length".into(),
            ));
        }
        let data = tokens
            .try_into_data()
            .map_err(|e| GenerationError(e.to_string()))?;
        std::thread::scope(|scope| {
            let mut handles = Vec::with_capacity(self.ranks.len());
            for ((model, state), device) in
                self.ranks.iter().zip(&mut cache.ranks).zip(&self.devices)
            {
                let data = data.clone();
                handles.push(scope.spawn(move || {
                    let input = Tensor::<B, 2, Int>::from_data(data, device);
                    let output = model.forward_block(input, state, last_only)?;
                    B::sync(device).map_err(|e| GenerationError(e.to_string()))?;
                    Ok::<_, GenerationError>(output)
                }));
            }
            let mut first = None;
            let mut error = None;
            for (rank, handle) in handles.into_iter().enumerate() {
                match handle.join() {
                    Ok(Ok(output)) if rank == 0 => first = Some(output),
                    Ok(Ok(_)) => (),
                    Ok(Err(e)) => error = Some(e),
                    Err(_) => {
                        error = Some(GenerationError(
                            "native tensor-parallel rank panicked".into(),
                        ))
                    }
                }
            }
            if let Some(error) = error {
                Err(error)
            } else {
                first.ok_or_else(|| GenerationError("TP group returned no root logits".into()))
            }
        })
    }
}

impl<B: DistributedBackend> CausalModel<B> for Qwen3TensorParallel<B> {
    type Cache = Qwen3TensorParallelCache<B>;
    fn new_cache(&self) -> Self::Cache {
        Qwen3TensorParallelCache {
            ranks: self.ranks.iter().map(CausalModel::new_cache).collect(),
            devices: self.devices.clone(),
        }
    }
    fn forward_cached_last(
        &self,
        tokens: Tensor<B, 2, Int>,
        cache: &mut Self::Cache,
    ) -> Tensor<B, 3> {
        self.forward_block(tokens, cache, true)
            .expect("validated Qwen3 TP forward")
    }
    fn try_forward_cached_last(
        &self,
        tokens: Tensor<B, 2, Int>,
        cache: &mut Self::Cache,
    ) -> Result<Tensor<B, 3>, GenerationError> {
        self.forward_block(tokens, cache, true)
    }
}

impl<B: DistributedBackend> SpeculativeModel<B> for Qwen3TensorParallel<B> {
    fn fork_cache(&self, cache: &Self::Cache) -> Self::Cache {
        cache.clone()
    }
    fn try_forward_cached_block(
        &self,
        tokens: Tensor<B, 2, Int>,
        cache: &mut Self::Cache,
    ) -> Result<Tensor<B, 3>, GenerationError> {
        self.forward_block(tokens, cache, false)
    }
}

impl<B: DistributedBackend> Record<B> for Qwen3TensorParallelCacheRecord<B> {
    type Item<S: PrecisionSettings> = Vec<<LlamaKvCacheRecord<B> as Record<B>>::Item<S>>;
    fn into_item<S: PrecisionSettings>(self) -> Self::Item<S> {
        self.ranks
            .into_iter()
            .map(|rank| rank.into_item::<S>())
            .collect()
    }
    fn from_item<S: PrecisionSettings>(item: Self::Item<S>, device: &B::Device) -> Self {
        Self {
            ranks: item
                .into_iter()
                .map(|rank| LlamaKvCacheRecord::from_item::<S>(rank, device))
                .collect(),
        }
    }
}

fn rank_identity(model_id: &str, rank: usize, world: usize) -> String {
    format!("{model_id}:qwen3-tp:{world}:{rank}")
}

impl<B: DistributedBackend> GenerationCacheContinuation<B> for Qwen3TensorParallelCache<B> {
    type Record = Qwen3TensorParallelCacheRecord<B>;
    fn generation_position(&self) -> usize {
        self.ranks.first().map_or(0, LlamaKvCache::position)
    }
    fn capture_generation(
        &self,
        model_id: &str,
        _device: &B::Device,
    ) -> Result<Self::Record, RecorderError> {
        let position = self.generation_position();
        if model_id.is_empty()
            || self.ranks.is_empty()
            || self.ranks.len() != self.devices.len()
            || self.ranks.iter().any(|rank| rank.position() != position)
        {
            return Err(RecorderError::Unknown("incomplete TP cache group".into()));
        }
        let ranks = self
            .ranks
            .iter()
            .zip(&self.devices)
            .enumerate()
            .map(|(rank, (cache, device))| {
                B::sync(device).map_err(|e| RecorderError::Unknown(e.to_string()))?;
                cache.capture_generation(&rank_identity(model_id, rank, self.ranks.len()), device)
            })
            .collect::<Result<Vec<_>, _>>()?;
        Ok(Qwen3TensorParallelCacheRecord { ranks })
    }
    fn restore_generation(
        record: Self::Record,
        model_id: &str,
        template: &Self,
        position: usize,
        _device: &B::Device,
    ) -> Result<Self, RecorderError> {
        let world = template.ranks.len();
        if model_id.is_empty()
            || world == 0
            || record.ranks.len() != world
            || template.devices.len() != world
        {
            return Err(RecorderError::Unknown("TP record topology differs".into()));
        }
        let ranks = record
            .ranks
            .into_iter()
            .zip(&template.ranks)
            .zip(&template.devices)
            .enumerate()
            .map(|(rank, ((record, cache), device))| {
                LlamaKvCache::restore_generation(
                    record,
                    &rank_identity(model_id, rank, world),
                    cache,
                    position,
                    device,
                )
            })
            .collect::<Result<Vec<_>, _>>()?;
        Ok(Self {
            ranks,
            devices: template.devices.clone(),
        })
    }
}
