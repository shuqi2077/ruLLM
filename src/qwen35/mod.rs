mod attention;
mod batching;
pub use batching::{Qwen35BatchCache,Qwen35BatchDelta};
mod cache_record;
pub use cache_record::{Qwen35CacheRecord,Qwen35LayerCacheState};
mod config;
mod delta;
mod layer;
mod moe;
use moe::FeedForward;
#[cfg(feature = "libtorch")]
mod libtorch;
#[cfg(feature = "libtorch")]
mod libtorch_delta;
mod loading;
mod quantized_loading;
mod fp8_loading;
mod projection;
mod training;
use projection::Projection;
#[cfg(feature = "tensor-parallel")]
pub(crate) mod parallel;
mod vision;
mod multimodal;
pub use multimodal::{Qwen35MultimodalCache, Qwen35MultimodalModel, load_huggingface_qwen35_multimodal};
pub use multimodal::{Qwen35MultimodalCacheRecord,Qwen35ImageConditioned,Qwen35CachedMultimodalDecoder};
pub use vision::{LoadedQwen35Vision, Qwen35VisionConfig, Qwen35VisionModel, Qwen35VisionOutput, load_huggingface_qwen35_vision};
pub use vision::{Qwen35ImageProcessor, Qwen35PreparedImages, Qwen35RgbImage};
#[cfg(all(test, feature = "nvidia"))]
mod tests;

use crate::{CausalModel, GenerationError};
use config::LayerType;
pub use config::{LayerType as Qwen35LayerType, Qwen35MoeConfig, Qwen35TextConfig, RopeConfig as Qwen35RopeConfig};
pub use loading::{LoadedQwen35Text, load_huggingface_qwen35_text, load_huggingface_qwen35_text_packed};
use ruda::runtime::server::ComputeServer;
use ruda_nn::{Embedding, Linear};
use ruda_tensor::api::{DType, Int, Tensor, TensorData, activation::silu, backend::Backend};
use ruda_tensor::{DeviceOps, TensorPrimitive};
use ruda_tensor_device::{BoolElement, DeviceBackend, DeviceRuntime, FloatElement, IntElement};

struct Norm<B: Backend> {
    weight: Tensor<B, 1>,
    epsilon: f64,
}
impl<R, F, I, BT> Norm<DeviceBackend<R, F, I, BT>>
where
    R: DeviceRuntime, R::Server: ComputeServer, R::Device: DeviceOps,
    F: FloatElement, I: IntElement, BT: BoolElement,
{
    fn forward<const D: usize>(&self, input: Tensor<DeviceBackend<R, F, I, BT>, D>)
        -> Result<Tensor<DeviceBackend<R, F, I, BT>, D>, GenerationError>
    {
        let output = rudnn::normalization::rms_norm(
            input.into_primitive().tensor(),
            (self.weight.clone().cast(DType::F32) + 1.0).into_primitive().tensor(),
            self.epsilon as f32,
        ).map_err(|error| GenerationError(error.to_string()))?;
        Ok(Tensor::from_primitive(TensorPrimitive::Float(output)))
    }
}

struct Mlp<B: Backend> {
    gate: Projection<B>,
    up: Projection<B>,
    down: Projection<B>,
}
impl<R, F, I, BT> Mlp<DeviceBackend<R, F, I, BT>>
where
    R: DeviceRuntime, R::Server: ComputeServer, R::Device: DeviceOps,
    F: FloatElement, I: IntElement, BT: BoolElement,
{
    fn forward(&self, input: Tensor<DeviceBackend<R, F, I, BT>, 3>)
        -> Result<Tensor<DeviceBackend<R, F, I, BT>, 3>, GenerationError>
    {
        let gate = self.gate.forward(input.clone());
        let up = self.up.forward(input);
        let activated = ruda_tensor::api::activation::swiglu_native(gate, up)
            .map_err(|error| GenerationError(error.to_string()))?;
        Ok(self.down.forward(activated))
    }
}
enum Mixer<B: Backend> {
    Full(attention::Attention<B>),
    Delta(delta::Delta<B>),
}
struct Layer<B: Backend> {
    mixer: Mixer<B>,
    input_norm: Norm<B>,
    post_norm: Norm<B>,
    mlp: FeedForward<B>,
}

pub struct Qwen35TextModel<B: Backend> {
    pub config: Qwen35TextConfig,
    weight_prefix: String,
    rotary: crate::llama::rope::RopeParameters,
    embedding: Embedding<B>,
    layers: Vec<Layer<B>>,
    norm: Norm<B>,
    head: Projection<B>,
    reduction: Option<Reduction<B>>,
}

type Reduction<B: Backend> = std::sync::Arc<dyn Fn(Tensor<B, 3>) -> Tensor<B, 3> + Send + Sync>;

#[derive(Clone)]
enum LayerCache<B: Backend> {
    Full {
        key: Option<Tensor<B, 4>>,
        value: Option<Tensor<B, 4>>,
    },
    Delta {
        convolution: Option<Tensor<B, 3>>,
        state: Option<Tensor<B, 4>>,
    },
}

#[derive(Clone)]
pub struct Qwen35Cache<B: Backend> {
    layers: Vec<LayerCache<B>>,
    position: usize,
    batch: Option<usize>,
    rotary_length: usize,
}

impl<B: Backend> Qwen35Cache<B> {
    pub fn sequence_length(&self) -> usize {
        self.position
    }
}

impl<B: Backend> Qwen35TextModel<B> {
    pub fn set_context_limit(&mut self, length: usize) -> Result<(), GenerationError> {
        if length == 0 || (length > self.config.max_position_embeddings
            && matches!(self.rotary.scaling, crate::llama::rope::RopeScaling::Static)) {
            return Err(GenerationError("context extension requires runtime-scaled Qwen3.5 RoPE".into()));
        }
        self.config.max_position_embeddings = length;
        Ok(())
    }
    fn rotary_for(&self, cache: &Qwen35Cache<B>) -> Result<crate::llama::rope::RotaryState, GenerationError> {
        self.rotary.state_at(cache.rotary_length)
            .ok_or_else(|| GenerationError("Qwen3.5 cache rotary state differs from the model".into()))
    }
    fn reduce(&self, value: Tensor<B, 3>) -> Tensor<B, 3> {
        match &self.reduction { Some(reduce) => reduce(value), None => value }
    }
    pub fn new_cache(&self) -> Qwen35Cache<B> {
        Qwen35Cache {
            layers: self
                .config
                .layer_types
                .iter()
                .map(|kind| match kind {
                    LayerType::FullAttention => LayerCache::Full {
                        key: None,
                        value: None,
                    },
                    LayerType::LinearAttention => LayerCache::Delta {
                        convolution: None,
                        state: None,
                    },
                })
                .collect(),
            position: 0,
            batch: None,
            rotary_length: self.rotary.initial_cached_length(),
        }
    }
}

impl<R, F, I, BT> Qwen35TextModel<DeviceBackend<R, F, I, BT>>
where
    R: DeviceRuntime,
    R::Server: ComputeServer,
    R::Device: DeviceOps,
    F: FloatElement,
    I: IntElement,
    BT: BoolElement,
{
    pub fn forward_cached(
        &self,
        tokens: Tensor<DeviceBackend<R, F, I, BT>, 2, Int>,
        cache: &mut Qwen35Cache<DeviceBackend<R, F, I, BT>>,
    ) -> Result<Tensor<DeviceBackend<R, F, I, BT>, 3>, GenerationError> {
        let hidden = self.forward_cached_hidden(tokens, cache)?;
        Ok(self.head.forward(self.norm.forward(hidden)?))
    }

    pub fn forward_cached_last(
        &self,
        tokens: Tensor<DeviceBackend<R, F, I, BT>, 2, Int>,
        cache: &mut Qwen35Cache<DeviceBackend<R, F, I, BT>>,
    ) -> Result<Tensor<DeviceBackend<R, F, I, BT>, 3>, GenerationError> {
        let hidden = self.forward_cached_hidden(tokens, cache)?;
        let [batch, sequence, width] = hidden.dims();
        Ok(self.head.forward(self.norm.forward(hidden.slice([
            0..batch,
            sequence - 1..sequence,
            0..width,
        ]))?))
    }

    fn forward_cached_hidden(
        &self,
        tokens: Tensor<DeviceBackend<R, F, I, BT>, 2, Int>,
        cache: &mut Qwen35Cache<DeviceBackend<R, F, I, BT>>,
    ) -> Result<Tensor<DeviceBackend<R, F, I, BT>, 3>, GenerationError> {
        let [batch, sequence] = tokens.dims();
        if batch == 0
            || sequence == 0
            || cache.batch.is_some_and(|n| n != batch)
            || cache.layers.len() != self.layers.len()
            || cache
                .position
                .checked_add(sequence)
                .is_none_or(|n| n > self.config.max_position_embeddings)
        {
            return Err(GenerationError(
                "invalid Qwen3.5 batch, sequence or cache capacity".into(),
            ));
        }
        let hidden = self.embedding.forward(tokens);
        let rotary = self.rotary_for(cache)?;
        let (cos, sin) = attention::text_rope(
            &rotary,
            cache.position,
            sequence,
            hidden.dtype(),
            &hidden.device(),
        );
        let hidden = self.run_layers(hidden, cos, sin, cache)?;
        cache.rotary_length = rotary.cached_length();
        Ok(hidden)
    }

    fn forward_embeddings_last(
        &self,
        hidden: Tensor<DeviceBackend<R, F, I, BT>, 3>,
        coordinates: &[[usize; 3]],
        cache: &mut Qwen35Cache<DeviceBackend<R, F, I, BT>>,
    ) -> Result<Tensor<DeviceBackend<R, F, I, BT>, 3>, GenerationError> {
        let [batch, sequence, width] = hidden.dims();
        if batch == 0 || sequence == 0 || width != self.config.hidden_size
            || cache.batch.is_some_and(|n| n != batch)
            || cache.layers.len() != self.layers.len()
            || cache.position.checked_add(sequence).is_none_or(|n| n > self.config.max_position_embeddings)
        {
            return Err(GenerationError("invalid Qwen3.5 embedded input or cache capacity".into()));
        }
        let rotary = self.rotary_for(cache)?;
        let (cos, sin) = attention::multimodal_rope_with_state(
            &self.config, &rotary, coordinates, batch, sequence, hidden.dtype(), &hidden.device(),
        )?;
        let hidden = self.run_layers(hidden, cos, sin, cache)?;
        cache.rotary_length = rotary.cached_length();
        Ok(self.head.forward(self.norm.forward(hidden.slice([0..batch, sequence-1..sequence, 0..width]))?))
    }

    fn run_layers(
        &self,
        hidden: Tensor<DeviceBackend<R, F, I, BT>, 3>,
        cos: Tensor<DeviceBackend<R, F, I, BT>, 4>,
        sin: Tensor<DeviceBackend<R, F, I, BT>, 4>,
        cache: &mut Qwen35Cache<DeviceBackend<R, F, I, BT>>,
    ) -> Result<Tensor<DeviceBackend<R, F, I, BT>, 3>, GenerationError> {
        self.run_layers_observed(hidden, cos, sin, cache, |_, _| {})
    }

    fn run_layers_observed(
        &self,
        mut hidden: Tensor<DeviceBackend<R, F, I, BT>, 3>,
        cos: Tensor<DeviceBackend<R, F, I, BT>, 4>,
        sin: Tensor<DeviceBackend<R, F, I, BT>, 4>,
        cache: &mut Qwen35Cache<DeviceBackend<R, F, I, BT>>,
        mut observe: impl FnMut(usize, &Tensor<DeviceBackend<R, F, I, BT>, 3>),
    ) -> Result<Tensor<DeviceBackend<R, F, I, BT>, 3>, GenerationError> {
        let [batch, sequence, _] = hidden.dims();
        for (index, (layer, state)) in self.layers.iter().zip(cache.layers.iter_mut()).enumerate() {
            hidden = layer.forward_reduced(hidden, &self.config, cos.clone(), sin.clone(), state, cache.position, self.reduction.as_ref())?;
            observe(index, &hidden);
        }
        cache.position += sequence;
        cache.batch = Some(batch);
        Ok(hidden)
    }
}

impl<R, F, I, BT> crate::SpeculativeModel<DeviceBackend<R, F, I, BT>>
    for Qwen35TextModel<DeviceBackend<R, F, I, BT>>
where
    R: DeviceRuntime, R::Server: ComputeServer, R::Device: DeviceOps,
    F: FloatElement, I: IntElement, BT: BoolElement,
{
    fn fork_cache(&self, cache: &Self::Cache) -> Self::Cache { cache.clone() }

    fn try_forward_cached_block(
        &self, tokens: Tensor<DeviceBackend<R, F, I, BT>, 2, Int>, cache: &mut Self::Cache,
    ) -> Result<Tensor<DeviceBackend<R, F, I, BT>, 3>, GenerationError> {
        self.forward_cached(tokens, cache)
    }
}

impl<R, F, I, BT> CausalModel<DeviceBackend<R, F, I, BT>>
    for Qwen35TextModel<DeviceBackend<R, F, I, BT>>
where
    R: DeviceRuntime,
    R::Server: ComputeServer,
    R::Device: DeviceOps,
    F: FloatElement,
    I: IntElement,
    BT: BoolElement,
{
    type Cache = Qwen35Cache<DeviceBackend<R, F, I, BT>>;
    fn new_cache(&self) -> Self::Cache {
        self.new_cache()
    }
    fn forward_cached_last(
        &self,
        tokens: Tensor<DeviceBackend<R, F, I, BT>, 2, Int>,
        cache: &mut Self::Cache,
    ) -> Tensor<DeviceBackend<R, F, I, BT>, 3> {
        self.forward_cached_last(tokens, cache)
            .expect("validated Qwen3.5 cached generation")
    }

    fn try_forward_cached_last(
        &self,
        tokens: Tensor<DeviceBackend<R, F, I, BT>, 2, Int>,
        cache: &mut Self::Cache,
    ) -> Result<Tensor<DeviceBackend<R, F, I, BT>, 3>, GenerationError> {
        self.forward_cached_last(tokens, cache)
    }
}
