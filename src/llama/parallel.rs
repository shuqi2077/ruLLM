use super::*;
use crate::{CausalModel, CausalModelLimits, DeviceBatchModel, GenerationCacheContinuation,
    GenerationError, HuggingFaceLoadError, HuggingFaceLoadReport, PrefixBatchModel, SpeculativeModel};
use ruda_tensor::api::TensorPrimitive;
use ruda_tensor::device::Device;
use ruda_tensor::distributed::{DistributedBackend, ReduceOperation};
use std::{path::Path, sync::Arc};

mod loading;
mod cache;
mod batching;
pub use cache::{LlamaTensorParallelCache, LlamaTensorParallelCacheRecord};

type Reduction<B> = Arc<dyn Fn(Tensor<B, 3>) -> Tensor<B, 3> + Send + Sync>;

pub struct LlamaTensorParallelRank<B: Backend> {
    model: LlamaForCausalLm<B>,
    reduction: Reduction<B>,
}

pub struct LlamaTensorParallel<B: DistributedBackend> {
    pub config: LlamaConfig,
    ranks: Vec<LlamaTensorParallelRank<B>>,
    devices: Vec<B::Device>,
}

pub struct LoadedLlamaTensorParallel<B: DistributedBackend> {
    pub model: LlamaTensorParallel<B>,
    pub eos_token_ids: Vec<i32>,
    pub reports: Vec<HuggingFaceLoadReport>,
}

pub fn load_huggingface_llama_tensor_parallel<B: DistributedBackend>(
    directory: impl AsRef<Path>, devices: Vec<B::Device>,
) -> Result<LoadedLlamaTensorParallel<B>, HuggingFaceLoadError> {
    load_parallel(directory.as_ref(), devices, false)
}

pub fn load_huggingface_qwen2_tensor_parallel<B: DistributedBackend>(
    directory: impl AsRef<Path>, devices: Vec<B::Device>,
) -> Result<LoadedLlamaTensorParallel<B>, HuggingFaceLoadError> {
    load_parallel(directory.as_ref(), devices, true)
}

fn load_parallel<B: DistributedBackend>(
    directory: &Path, devices: Vec<B::Device>, qwen2: bool,
) -> Result<LoadedLlamaTensorParallel<B>, HuggingFaceLoadError> {
    let ids = devices.iter().map(Device::to_id).collect::<Vec<_>>();
    let world = devices.len();
    if world == 0 || ids.iter().enumerate().any(|(i, id)| ids[..i].contains(id)) {
        return Err(HuggingFaceLoadError("TP requires distinct nonempty native devices".into()));
    }
    let directory = directory.canonicalize().map_err(|e| HuggingFaceLoadError(e.to_string()))?;
    let (config, tied, eos_token_ids, rotary, window) = crate::huggingface::dense_parallel_config(&directory, qwen2)?;
    if !config.num_query_heads.is_multiple_of(world) || !config.d_ff.is_multiple_of(world)
        || !(config.num_kv_heads.is_multiple_of(world) || world.is_multiple_of(config.num_kv_heads)) {
        return Err(HuggingFaceLoadError("Llama/Qwen2 TP requires divisible query/FFN shards and compatible KV-head groups".into()));
    }
    let mut ranks = Vec::with_capacity(world);
    let mut reports = Vec::with_capacity(world);
    for (rank, device) in devices.iter().enumerate() {
        let group = ids.clone();
        let reduction = Arc::new(move |value: Tensor<B, 3>| {
            if group.len() == 1 { return value; }
            Tensor::from_primitive(TensorPrimitive::Float(B::all_reduce(
                value.into_primitive().tensor(), ReduceOperation::Sum, group.clone(),
            ).resolve()))
        });
        let (mut model, report) = loading::load_rank(&directory, &config, &rotary, tied, qwen2, rank, world, device, reduction)?;
        if let Some((window, first_layer)) = window {
            model.model.set_sliding_window(window, first_layer);
        }
        ranks.push(model);
        reports.push(report);
    }
    Ok(LoadedLlamaTensorParallel {
        model: LlamaTensorParallel { config, ranks, devices }, eos_token_ids, reports,
    })
}

impl<B: Backend> LlamaTensorParallelRank<B> {
    fn forward_block(&self, tokens: Tensor<B, 2, Int>, cache: &mut LlamaKvCache<B>, last_only: bool)
        -> Tensor<B, 3> {
        let sequence = tokens.dims()[1];
        let mut hidden = self.model.embed_tokens.forward(tokens);
        for (layer, state) in self.model.layers.iter().zip(&mut cache.layers) {
            let attention = layer.self_attn.forward_cached(layer.input_layernorm.forward(hidden.clone()), state, cache.position);
            hidden = hidden + (self.reduction)(attention);
            let mlp = layer.mlp.forward(layer.post_attention_layernorm.forward(hidden.clone()));
            hidden = hidden + (self.reduction)(mlp);
        }
        cache.position += sequence;
        if last_only {
            let [batch, sequence, width] = hidden.dims();
            hidden = hidden.slice([0..batch, sequence - 1..sequence, 0..width]);
        }
        self.model.lm_head.forward(self.model.norm.forward(hidden))
    }
}

impl<B: DistributedBackend> LlamaTensorParallel<B> {
    pub fn devices(&self) -> &[B::Device] { &self.devices }

    pub fn limits(&self) -> CausalModelLimits {
        CausalModelLimits { vocab_size: self.config.vocab_size, max_sequence_length: self.config.max_sequence_length }
    }

    fn forward_block(&self, tokens: Tensor<B, 2, Int>, cache: &mut LlamaTensorParallelCache<B>, last_only: bool)
        -> Result<Tensor<B, 3>, GenerationError> {
        let [batch, sequence] = tokens.dims();
        let position = cache.generation_position();
        if cache.ranks.len() != self.ranks.len() || cache.devices != self.devices
            || tokens.device() != self.devices[0] || batch == 0 || sequence == 0
            || position.checked_add(sequence).is_none_or(|end| end > self.config.max_sequence_length)
            || cache.ranks.iter().zip(&self.ranks).zip(&self.devices).any(|((state, rank), device)| {
                state.position() != position || state.capacity() != self.config.max_sequence_length
                    || state.layers.len() != self.config.num_hidden_layers
                    || state.layers.iter().zip(&rank.model.layers).any(|(state, layer)| {
                        if state.sequence_length() != position { return true; }
                        match (&state.key, &state.value) {
                            (None, None) => position != 0,
                            (Some(k), Some(v)) => {
                                let [kb, heads, capacity, width] = k.dims();
                                kb != batch || heads != layer.self_attn.num_kv_heads
                                    || width != layer.self_attn.head_dimension || capacity < position
                                    || v.dims() != k.dims() || k.device() != *device || v.device() != *device
                                    || k.dtype() != layer.self_attn.k_proj.weight.val().dtype() || v.dtype() != k.dtype()
                            },
                            _ => true,
                        }
                    })
            }) {
            return Err(GenerationError("Llama/Qwen2 TP input, cache geometry, topology or context differs".into()));
        }
        let data = tokens.try_into_data().map_err(|e| GenerationError(e.to_string()))?;
        std::thread::scope(|scope| {
            let mut handles = Vec::with_capacity(self.ranks.len());
            for ((model, state), device) in self.ranks.iter().zip(&mut cache.ranks).zip(&self.devices) {
                let data = data.clone();
                handles.push(scope.spawn(move || {
                    let output = model.forward_block(Tensor::from_data(data, device), state, last_only);
                    B::sync(device).map_err(|e| GenerationError(e.to_string()))?;
                    Ok::<_, GenerationError>(output)
                }));
            }
            let mut root = None;
            let mut failure = None;
            for (rank, handle) in handles.into_iter().enumerate() {
                match handle.join() {
                    Ok(Ok(output)) if rank == 0 => root = Some(output),
                    Ok(Ok(_)) => (),
                    Ok(Err(error)) => failure = Some(error),
                    Err(_) => failure = Some(GenerationError("Llama/Qwen2 TP rank panicked".into())),
                }
            }
            if let Some(error) = failure { Err(error) }
            else { root.ok_or_else(|| GenerationError("Llama/Qwen2 TP root logits absent".into())) }
        })
    }
}

impl<B: DistributedBackend> CausalModel<B> for LlamaTensorParallel<B> {
    type Cache = LlamaTensorParallelCache<B>;
    fn new_cache(&self) -> Self::Cache {
        LlamaTensorParallelCache { ranks: self.ranks.iter().map(|rank| rank.model.new_cache()).collect(), devices: self.devices.clone() }
    }
    fn forward_cached_last(&self, tokens: Tensor<B, 2, Int>, cache: &mut Self::Cache) -> Tensor<B, 3> {
        self.forward_block(tokens, cache, true).expect("validated Llama/Qwen2 TP forward")
    }
    fn try_forward_cached_last(&self, tokens: Tensor<B, 2, Int>, cache: &mut Self::Cache) -> Result<Tensor<B, 3>, GenerationError> {
        self.forward_block(tokens, cache, true)
    }
}

impl<B: DistributedBackend> SpeculativeModel<B> for LlamaTensorParallel<B> {
    fn fork_cache(&self, cache: &Self::Cache) -> Self::Cache { cache.clone() }
    fn try_forward_cached_block(&self, tokens: Tensor<B, 2, Int>, cache: &mut Self::Cache) -> Result<Tensor<B, 3>, GenerationError> {
        self.forward_block(tokens, cache, false)
    }
}
