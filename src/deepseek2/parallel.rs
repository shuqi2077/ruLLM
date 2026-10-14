use super::{
    DeepseekV2AttentionMode, DeepseekV2Cache, DeepseekV2CacheRecord, DeepseekV2Config,
    DeepseekV2Model, ExpertShard, load_rank_with, MlaFp8Backend, weights,
};
use crate::{
    CausalModel, GenerationCacheContinuation, GenerationError, HuggingFaceLoadError,
    HuggingFaceLoadReport, SpeculativeModel,
};
use ruda_model::record::{PrecisionSettings, Record, RecorderError};
use ruda_tensor::api::{Int, Tensor, TensorPrimitive};
use ruda_tensor::device::Device;
use ruda_tensor::distributed::{DistributedBackend, ReduceOperation};
use std::{path::Path, sync::Arc};
use crate::{DeviceBatchModel,PrefixBatchModel};
crate::parallel_batch::parallel_batch_model!(DeepseekV2ExpertParallel,DeepseekV2Model);
crate::parallel_batch::parallel_speculative_model!(DeepseekV2ExpertParallel,DeepseekV2Model);

pub struct DeepseekV2ExpertParallel<B: DistributedBackend> {
    pub config: DeepseekV2Config,
    ranks: Vec<DeepseekV2Model<B>>,
    devices: Vec<B::Device>,
}
#[derive(Clone, Debug)]
pub struct DeepseekV2ExpertParallelCache<B: DistributedBackend> {
    ranks: Vec<DeepseekV2Cache<B>>,
    devices: Vec<B::Device>,
}
pub struct DeepseekV2ExpertParallelCacheRecord<B: DistributedBackend> {
    ranks: Vec<DeepseekV2CacheRecord<B>>,
}
pub struct LoadedDeepseekV2ExpertParallel<B: DistributedBackend> {
    pub model: DeepseekV2ExpertParallel<B>,
    pub eos_token_ids: Vec<i32>,
    pub reports: Vec<HuggingFaceLoadReport>,
    pub auxiliary_tensors: Vec<Vec<String>>,
}

pub fn load_huggingface_deepseek_v2_expert_parallel<B: DistributedBackend>(
    directory: impl AsRef<Path>,
    devices: Vec<B::Device>,
) -> Result<LoadedDeepseekV2ExpertParallel<B>, HuggingFaceLoadError> {
    load_parallel(directory.as_ref(), devices, None)
}

pub fn load_huggingface_mla_expert_parallel<B: DistributedBackend + MlaFp8Backend>(directory: impl AsRef<Path>, devices: Vec<B::Device>)
    -> Result<LoadedDeepseekV2ExpertParallel<B>, HuggingFaceLoadError> {
    load_parallel(directory.as_ref(), devices, Some(weights::native_projection::<B>))
}

fn load_parallel<B: DistributedBackend>(directory: &Path, devices: Vec<B::Device>, packed: Option<weights::Loader<B>>)
    -> Result<LoadedDeepseekV2ExpertParallel<B>, HuggingFaceLoadError> {
    let ids = devices.iter().map(Device::to_id).collect::<Vec<_>>();
    if devices.is_empty() || ids.iter().enumerate().any(|(i, id)| ids[..i].contains(id)) {
        return Err(HuggingFaceLoadError(
            "EP requires distinct native devices".into(),
        ));
    }
    let config: DeepseekV2Config = serde_json::from_slice(
        &std::fs::read(directory.join("config.json"))
            .map_err(|e| HuggingFaceLoadError(e.to_string()))?,
    )
    .map_err(|e| HuggingFaceLoadError(e.to_string()))?;
    config.validate()?;
    if !config.n_routed_experts.is_multiple_of(devices.len()) {
        return Err(HuggingFaceLoadError(
            "EP group must evenly partition the checkpoint's routed experts".into(),
        ));
    }
    let mut ranks = Vec::with_capacity(devices.len());
    let mut reports = Vec::with_capacity(devices.len());
    let mut auxiliary_tensors = Vec::with_capacity(devices.len());
    let mut eos_token_ids = None;
    for (rank, device) in devices.iter().enumerate() {
        let group = ids.clone();
        let reduce = Arc::new(move |value: Tensor<B, 3>| {
            Tensor::from_primitive(TensorPrimitive::Float(
                B::all_reduce(
                    value.into_primitive().tensor(),
                    ReduceOperation::Sum,
                    group.clone(),
                )
                .resolve(),
            ))
        });
        let loaded = load_rank_with(
            directory.as_ref(),
            device,
            Some(ExpertShard {
                rank,
                world: devices.len(),
            }),
            Some(reduce),
            packed,
        )?;
        if eos_token_ids
            .as_ref()
            .is_some_and(|ids| *ids != loaded.eos_token_ids)
        {
            return Err(HuggingFaceLoadError(
                "DeepSeek-V2 EOS configuration changed during EP loading".into(),
            ));
        }
        eos_token_ids = Some(loaded.eos_token_ids);
        ranks.push(loaded.model);
        reports.push(loaded.report);
        auxiliary_tensors.push(loaded.auxiliary_tensors);
    }
    Ok(LoadedDeepseekV2ExpertParallel {
        model: DeepseekV2ExpertParallel {
            config,
            ranks,
            devices,
        },
        eos_token_ids: eos_token_ids.unwrap_or_default(),
        reports,
        auxiliary_tensors,
    })
}

impl<B: DistributedBackend> DeepseekV2ExpertParallel<B> {
    pub fn devices(&self) -> &[B::Device] {
        &self.devices
    }
    pub fn with_attention_mode(mut self, mode: DeepseekV2AttentionMode) -> Self {
        for model in &mut self.ranks {
            model.attention_mode = mode;
        }
        self
    }
    fn forward_block(
        &self,
        tokens: Tensor<B, 2, Int>,
        cache: &mut DeepseekV2ExpertParallelCache<B>,
        last_only: bool,
    ) -> Result<Tensor<B, 3>, GenerationError> {
        let [batch, sequence] = tokens.dims();
        let position = cache.generation_position();
        let c = &self.config;
        if batch != 1
            || sequence == 0
            || tokens.device() != self.devices[0]
            || cache.ranks.len() != self.ranks.len()
            || cache.devices != self.devices
            || position
                .checked_add(sequence)
                .is_none_or(|end| end > c.max_position_embeddings)
            || self.ranks.iter().zip(&cache.ranks).zip(&self.devices).any(
                |((model, state), device)| {
                    state.position() != position
                        || !state.compatible(
                            c.num_hidden_layers,
                            c.max_position_embeddings,
                            c.kv_lora_rank + c.qk_rope_head_dim,
                            device,
                            model.embedding.dtype(),
                        )
                },
            )
        {
            return Err(GenerationError(
                "incompatible expert-parallel input, topology or MLA history".into(),
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
                    let result = model.forward_block(
                        Tensor::<B, 2, Int>::from_data(data, device),
                        state,
                        last_only,
                    )?;
                    B::sync(device).map_err(|e| GenerationError(e.to_string()))?;
                    Ok::<_, GenerationError>(result)
                }));
            }
            let mut root = None;
            let mut failure = None;
            for (rank, handle) in handles.into_iter().enumerate() {
                match handle.join() {
                    Ok(Ok(value)) if rank == 0 => root = Some(value),
                    Ok(Ok(_)) => (),
                    Ok(Err(e)) => failure = Some(e),
                    Err(_) => {
                        failure = Some(GenerationError(
                            "native expert-parallel rank panicked".into(),
                        ))
                    }
                }
            }
            if let Some(failure) = failure {
                Err(failure)
            } else {
                root.ok_or_else(|| GenerationError("EP group returned no root logits".into()))
            }
        })
    }
}

impl<B: DistributedBackend> CausalModel<B> for DeepseekV2ExpertParallel<B> {
    type Cache = DeepseekV2ExpertParallelCache<B>;
    fn new_cache(&self) -> Self::Cache {
        DeepseekV2ExpertParallelCache {
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
            .expect("validated DeepSeek-V2 EP forward")
    }
    fn try_forward_cached_last(
        &self,
        tokens: Tensor<B, 2, Int>,
        cache: &mut Self::Cache,
    ) -> Result<Tensor<B, 3>, GenerationError> {
        self.forward_block(tokens, cache, true)
    }
}
impl<B: DistributedBackend> SpeculativeModel<B> for DeepseekV2ExpertParallel<B> {
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
impl<B: DistributedBackend> Record<B> for DeepseekV2ExpertParallelCacheRecord<B> {
    type Item<S: PrecisionSettings> = Vec<<DeepseekV2CacheRecord<B> as Record<B>>::Item<S>>;
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
                .map(|rank| DeepseekV2CacheRecord::from_item::<S>(rank, device))
                .collect(),
        }
    }
}
fn rank_identity(model_id: &str, rank: usize, world: usize) -> String {
    format!("{model_id}:deepseek-v2-ep:{world}:{rank}")
}
impl<B: DistributedBackend> GenerationCacheContinuation<B> for DeepseekV2ExpertParallelCache<B> {
    type Record = DeepseekV2ExpertParallelCacheRecord<B>;
    fn generation_position(&self) -> usize {
        self.ranks.first().map_or(0, DeepseekV2Cache::position)
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
            return Err(RecorderError::Unknown("incomplete EP cache group".into()));
        }
        let ranks = self
            .ranks
            .iter()
            .zip(&self.devices)
            .enumerate()
            .map(|(rank, (cache, device))| {
                cache.capture_generation(&rank_identity(model_id, rank, self.ranks.len()), device)
            })
            .collect::<Result<Vec<_>, _>>()?;
        Ok(DeepseekV2ExpertParallelCacheRecord { ranks })
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
            return Err(RecorderError::Unknown(
                "EP continuation topology differs".into(),
            ));
        }
        let ranks = record
            .ranks
            .into_iter()
            .zip(&template.ranks)
            .zip(&template.devices)
            .enumerate()
            .map(|(rank, ((record, cache), device))| {
                DeepseekV2Cache::restore_generation(
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
