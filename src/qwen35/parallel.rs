use super::*;
use crate::{CausalModelLimits, DeviceBatchModel, GenerationCacheContinuation, HuggingFaceLoadError,
    HuggingFaceLoadReport, PrefixBatchModel, SpeculativeModel};
use ruda_tensor::device::Device;
use ruda_tensor::distributed::{DistributedBackend, ReduceOperation};
use std::path::Path;

mod loading;
mod weights;
mod awq;
mod cache;
pub use cache::{Qwen35TensorParallelCache, Qwen35TensorParallelCacheRecord};

pub struct Qwen35TensorParallel<B: DistributedBackend> {
    pub config: Qwen35TextConfig,
    ranks: Vec<Qwen35TextModel<B>>,
    devices: Vec<B::Device>,
}

pub struct LoadedQwen35TensorParallel<B: DistributedBackend> {
    pub model: Qwen35TensorParallel<B>,
    pub reports: Vec<HuggingFaceLoadReport>,
    pub unloaded_tensors: Vec<Vec<String>>,
    pub dequantized_awq_linears: Vec<usize>,
    pub packed_awq_linears: Vec<usize>,
    pub packed_fp8_linears: Vec<usize>,
}

pub fn load_huggingface_qwen35_tensor_parallel<B: DistributedBackend>(directory: impl AsRef<Path>, devices: Vec<B::Device>)
    -> Result<LoadedQwen35TensorParallel<B>, HuggingFaceLoadError> {
    load_parallel(directory.as_ref(), devices, None, None)
}

pub fn load_huggingface_qwen35_tensor_parallel_packed<R, F, I, BT>(directory: impl AsRef<Path>, devices: Vec<R::Device>)
    -> Result<LoadedQwen35TensorParallel<DeviceBackend<R, F, I, BT>>, HuggingFaceLoadError>
where R: DeviceRuntime, R::Server: ComputeServer, R::Device: DeviceOps,
    F: FloatElement, I: IntElement, BT: BoolElement, DeviceBackend<R, F, I, BT>: DistributedBackend<Device = R::Device> {
    load_parallel(directory.as_ref(), devices, Some(awq::packed::<R, F, I, BT>), Some(super::fp8_loading::native::<R,F,I,BT>))
}

fn load_parallel<B: DistributedBackend>(directory: &Path, devices: Vec<B::Device>, packed: Option<weights::PackedPartLoader<B>>,
    fp8_loader: Option<super::fp8_loading::Loader<B>>)
    -> Result<LoadedQwen35TensorParallel<B>, HuggingFaceLoadError> {
    let ids = devices.iter().map(Device::to_id).collect::<Vec<_>>();
    if devices.is_empty() || ids.iter().enumerate().any(|(rank, id)| ids[..rank].contains(id)) {
        return Err(HuggingFaceLoadError("Qwen3.5 TP requires distinct nonempty native devices".into()));
    }
    let directory = directory.canonicalize().map_err(|e| HuggingFaceLoadError(e.to_string()))?;
    let (config, quantization, multimodal) = super::loading::text_configuration(&directory)?;
    loading::validate(&config, devices.len())?;
    let mut ranks = Vec::with_capacity(devices.len());
    let mut reports = Vec::with_capacity(devices.len());
    let mut unloaded_tensors = Vec::with_capacity(devices.len());
    let mut dequantized_awq_linears = Vec::with_capacity(devices.len());
    let mut packed_awq_linears = Vec::with_capacity(devices.len());
    let mut packed_fp8_linears = Vec::with_capacity(devices.len());
    for (rank, device) in devices.iter().enumerate() {
        let group = ids.clone();
        let reduction = std::sync::Arc::new(move |value: Tensor<B, 3>| {
            if group.len() == 1 { return value; }
            Tensor::from_primitive(TensorPrimitive::Float(B::all_reduce(
                value.into_primitive().tensor(), ReduceOperation::Sum, group.clone(),
            ).resolve()))
        });
        let loaded = loading::load_rank(&directory, &config, quantization.clone(), multimodal,
            rank, devices.len(), device, reduction, packed, fp8_loader)?;
        ranks.push(loaded.model);
        reports.push(loaded.report);
        unloaded_tensors.push(loaded.unloaded_tensors);
        dequantized_awq_linears.push(loaded.dequantized_awq_linears);
        packed_awq_linears.push(loaded.packed_awq_linears);
        packed_fp8_linears.push(loaded.packed_fp8_linears);
    }
    Ok(LoadedQwen35TensorParallel { model: Qwen35TensorParallel { config, ranks, devices },
        reports, unloaded_tensors, dequantized_awq_linears, packed_awq_linears, packed_fp8_linears })
}

impl<B: DistributedBackend> Qwen35TensorParallel<B> {
    pub fn set_context_limit(&mut self, length: usize) -> Result<(), GenerationError> {
        if length == 0 || self.ranks.iter().any(|rank| length > rank.config.max_position_embeddings
            && matches!(rank.rotary.scaling, crate::llama::rope::RopeScaling::Static)) {
            return Err(GenerationError("context extension requires runtime-scaled Qwen3.5 RoPE on every TP rank".into()));
        }
        for rank in &mut self.ranks { rank.set_context_limit(length)?; }
        self.config.max_position_embeddings = length;
        Ok(())
    }
    pub fn devices(&self) -> &[B::Device] { &self.devices }
    pub fn limits(&self) -> CausalModelLimits {
        CausalModelLimits { vocab_size: self.config.vocab_size, max_sequence_length: self.config.max_position_embeddings }
    }
    fn validate_cache(&self, cache: &Qwen35TensorParallelCache<B>, batch: usize, sequence: usize, device: &B::Device)
        -> Result<(), GenerationError> {
        let position = cache.generation_position();
        let mut valid = self.devices.first() == Some(device) && cache.devices == self.devices
            && cache.ranks.len() == self.ranks.len() && batch > 0 && sequence > 0
            && position.checked_add(sequence).is_some_and(|end| end <= self.config.max_position_embeddings);
        for ((state, model), device) in cache.ranks.iter().zip(&self.ranks).zip(&self.devices) {
            let c = &model.config;
            valid &= state.position == position && state.layers.len() == model.layers.len()
                && model.rotary.state_at(state.rotary_length).is_some()
                && cache.ranks.first().is_some_and(|first| first.rotary_length == state.rotary_length)
                && (state.batch == Some(batch) || (position == 0 && state.batch.is_none()));
            for (layer, kind) in state.layers.iter().zip(&c.layer_types) {
                valid &= match (layer, kind) {
                    (LayerCache::Full { key: None, value: None }, LayerType::FullAttention)
                    | (LayerCache::Delta { convolution: None, state: None }, LayerType::LinearAttention) => position == 0,
                    (LayerCache::Full { key: Some(k), value: Some(v) }, LayerType::FullAttention) =>
                        k.dims() == [batch, c.num_key_value_heads, position, c.head_dim]
                            && v.dims() == k.dims() && v.dtype() == k.dtype() && k.device() == *device && v.device() == *device,
                    (LayerCache::Delta { convolution: Some(conv), state: Some(s) }, LayerType::LinearAttention) =>
                        conv.dims() == [batch, 2 * c.linear_num_key_heads * c.linear_key_head_dim
                            + c.linear_num_value_heads * c.linear_value_head_dim, c.linear_conv_kernel_dim]
                            && s.dims() == [batch, c.linear_num_value_heads, c.linear_key_head_dim, c.linear_value_head_dim]
                            && s.dtype() == DType::F32 && s.device() == *device && conv.device() == *device,
                    _ => false,
                };
            }
        }
        if valid { Ok(()) } else { Err(GenerationError("Qwen3.5 TP input, topology, full/recurrent cache or context differs".into())) }
    }
}

impl<B: DistributedBackend> Qwen35TensorParallel<B>
where Qwen35TextModel<B>: CausalModel<B, Cache = Qwen35Cache<B>> + SpeculativeModel<B> + Sync {
    fn forward_block(&self, tokens: Tensor<B, 2, Int>, cache: &mut Qwen35TensorParallelCache<B>, last_only: bool)
        -> Result<Tensor<B, 3>, GenerationError> {
        let [batch, sequence] = tokens.dims();
        self.validate_cache(cache, batch, sequence, &tokens.device())?;
        let data = tokens.try_into_data().map_err(|e| GenerationError(e.to_string()))?;
        std::thread::scope(|scope| {
            let mut handles = Vec::with_capacity(self.ranks.len());
            for ((model, state), device) in self.ranks.iter().zip(&mut cache.ranks).zip(&self.devices) {
                let data = data.clone();
                handles.push(scope.spawn(move || {
                    let tokens = Tensor::from_data(data, device);
                    let output = if last_only { model.try_forward_cached_last(tokens, state) }
                        else { model.try_forward_cached_block(tokens, state) };
                    B::sync(device).map_err(|e| GenerationError(e.to_string()))?;
                    output
                }));
            }
            let mut root = None;
            let mut failure = None;
            for (rank, handle) in handles.into_iter().enumerate() {
                match handle.join() {
                    Ok(Ok(output)) if rank == 0 => root = Some(output), Ok(Ok(_)) => (),
                    Ok(Err(error)) => failure = Some(error),
                    Err(_) => failure = Some(GenerationError("Qwen3.5 TP rank panicked".into())),
                }
            }
            if let Some(error) = failure { Err(error) }
            else { root.ok_or_else(|| GenerationError("Qwen3.5 TP root logits absent".into())) }
        })
    }
}

impl<B: DistributedBackend> CausalModel<B> for Qwen35TensorParallel<B>
where Qwen35TextModel<B>: CausalModel<B, Cache = Qwen35Cache<B>> + SpeculativeModel<B> + Sync {
    type Cache = Qwen35TensorParallelCache<B>;
    fn new_cache(&self) -> Self::Cache {
        Qwen35TensorParallelCache { ranks: self.ranks.iter().map(Qwen35TextModel::new_cache).collect(), devices: self.devices.clone() }
    }
    fn forward_cached_last(&self, tokens: Tensor<B, 2, Int>, cache: &mut Self::Cache) -> Tensor<B, 3> {
        self.forward_block(tokens, cache, true).expect("validated Qwen3.5 TP forward")
    }
    fn try_forward_cached_last(&self, tokens: Tensor<B, 2, Int>, cache: &mut Self::Cache) -> Result<Tensor<B, 3>, GenerationError> {
        self.forward_block(tokens, cache, true)
    }
}

impl<B: DistributedBackend> SpeculativeModel<B> for Qwen35TensorParallel<B>
where Qwen35TextModel<B>: CausalModel<B, Cache = Qwen35Cache<B>> + SpeculativeModel<B> + Sync {
    fn fork_cache(&self, cache: &Self::Cache) -> Self::Cache { cache.clone() }
    fn try_forward_cached_block(&self, tokens: Tensor<B, 2, Int>, cache: &mut Self::Cache) -> Result<Tensor<B, 3>, GenerationError> {
        self.forward_block(tokens, cache, false)
    }
}

crate::parallel_batch::parallel_batch_model!(Qwen35TensorParallel, Qwen35TextModel,|group|
    crate::PrefixShard{kv_heads:Some(group.config.num_key_value_heads),linear_key_heads:Some(group.config.linear_num_key_heads),
        linear_value_heads:Some(group.config.linear_num_value_heads),..Default::default()});
crate::parallel_batch::parallel_speculative_model!(Qwen35TensorParallel, Qwen35TextModel);
