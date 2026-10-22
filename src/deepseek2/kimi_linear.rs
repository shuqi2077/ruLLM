use super::{DeepseekV2Config, FeedForward, Mlp, Projection, load_mlp, norm, projection, weights};
use crate::{CausalModel, CausalModelLimits, GenerationError, HuggingFaceLoadError, HuggingFaceLoadReport, SpeculativeModel};
use ruda::runtime::server::ComputeServer;
use ruda_tensor::api::{backend::Backend, DType, Int, Tensor, TensorData};
use ruda_tensor::{DeviceOps, TensorPrimitive};
use ruda_tensor_device::{BoolElement, DeviceBackend, DeviceRuntime, FloatElement, IntElement};
use std::{path::Path, sync::Arc};

mod config;
mod loading;
mod kda;
mod cache;
mod batching;
mod training;
#[cfg(feature = "tensor-parallel")]
mod parallel;
#[cfg(feature = "tensor-parallel")]
pub use parallel::{KimiLinearExpertParallel, KimiLinearExpertParallelCache, KimiLinearExpertParallelCacheRecord,
    LoadedKimiLinearExpertParallel, load_huggingface_kimi_linear_expert_parallel};
pub use config::{KimiLinearConfig, KimiLinearAttentionConfig};
pub use loading::{LoadedKimiLinear, load_huggingface_kimi_linear};
pub(crate) use loading::load_sft as load_kimi_linear_sft;
pub use cache::{KimiLinearCache, KimiLinearCacheRecord};
pub use batching::{KimiLinearBatchCache, KimiLinearBatchPrefix, KimiLinearBatchDelta};
use cache::LayerCache;

struct Mla<B: Backend> {
    q: Projection<B>, kv_a: Projection<B>, kv_norm: Tensor<B, 1>,
    kv_b: Projection<B>, out: Projection<B>,
}
enum Mixer<B: Backend> { Mla(Mla<B>), Kda(kda::Kda<B>) }
struct Layer<B: Backend> {
    mixer: Mixer<B>, input_norm: Tensor<B, 1>, post_norm: Tensor<B, 1>, mlp: FeedForward<B>,
}
pub struct KimiLinearModel<B: Backend> {
    pub config: KimiLinearConfig,
    embedding: Tensor<B, 2>, layers: Vec<Layer<B>>, norm: Tensor<B, 1>, head: Projection<B>,
    routing: DeepseekV2Config,
    reduction: Option<Arc<dyn Fn(Tensor<B, 3>) -> Tensor<B, 3> + Send + Sync>>,
}
fn error(message: impl ToString) -> GenerationError { GenerationError(message.to_string()) }

impl<B: Backend> Mla<B> {
    fn project(&self, input: Tensor<B, 3>, c: &KimiLinearConfig)
        -> (Tensor<B, 4>, Tensor<B, 4>, Tensor<B, 4>) {
        let [b, s, _] = input.dims();
        let (h, rank, nd, rd, vd) = (c.num_attention_heads, c.kv_lora_rank, c.qk_nope_head_dim, c.qk_rope_head_dim, c.v_head_dim);
        let query = self.q.forward(input.clone()).reshape([b, s, h, nd + rd]).swap_dims(1, 2);
        let compressed = self.kv_a.forward(input);
        let latent = norm(compressed.clone().slice([0..b, 0..s, 0..rank]), &self.kv_norm, 1e-6);
        let expanded = self.kv_b.forward(latent).reshape([b, s, h, nd + vd]).swap_dims(1, 2);
        let position = compressed.slice([0..b, 0..s, rank..rank + rd]).reshape([b, 1, s, rd]).repeat_dim(1, h);
        let key = Tensor::cat(vec![expanded.clone().slice([0..b, 0..h, 0..s, 0..nd]), position], 3);
        let value = expanded.slice([0..b, 0..h, 0..s, nd..nd + vd]);
        (query, key, value)
    }
    fn forward(&self, input: Tensor<B, 3>, c: &KimiLinearConfig, key: &mut Option<Tensor<B, 4>>,
        value: &mut Option<Tensor<B, 4>>, position: usize) -> Tensor<B, 3> {
        let [b, s, _] = input.dims(); let device = input.device();
        let (q, k, v) = self.project(input, c);
        let k = match key { Some(old) => Tensor::cat(vec![old.clone(), k], 2), None => k };
        let v = match value { Some(old) => Tensor::cat(vec![old.clone(), v], 2), None => v };
        let mask = if position > 0 && s > 1 {
            let queries = Tensor::<B, 1, Int>::arange(position as i64..(position + s) as i64, &device).reshape([1, 1, s, 1]);
            let keys = Tensor::<B, 1, Int>::arange(0..(position + s) as i64, &device).reshape([1, 1, 1, position + s]);
            Some(keys.greater(queries).expand([b, c.num_attention_heads, s, position + s]))
        } else { None };
        let output = ruda_tensor::api::module::attention(q, k.clone(), v.clone(), mask, None,
            ruda_tensor::api::ops::AttentionModuleOptions {
                scale: Some(((c.qk_nope_head_dim + c.qk_rope_head_dim) as f64).sqrt().recip()),
                softcap: None, is_causal: position == 0 && s > 1,
            });
        *key = Some(k); *value = Some(v);
        self.out.forward(output.swap_dims(1, 2).reshape([b, s, c.num_attention_heads * c.v_head_dim]))
    }
}

impl<B: Backend> KimiLinearModel<B> {
    pub fn limits(&self) -> CausalModelLimits { self.config.limits() }
    fn fresh_cache(&self) -> KimiLinearCache<B> { KimiLinearCache::new(&self.config, self.embedding.dtype()) }
    fn feed_forward(&self, layer: &Layer<B>, input: Tensor<B, 3>) -> Result<Tensor<B, 3>, GenerationError> {
        let [b, s, d] = input.dims();
        Ok(layer.mlp.run(input.reshape([1, b * s, d]), &self.routing, self.reduction.as_ref())?.reshape([b, s, d]))
    }
}
impl<R, F, I, BT> KimiLinearModel<DeviceBackend<R, F, I, BT>>
where R: DeviceRuntime, R::Server: ComputeServer, R::Device: DeviceOps, F: FloatElement, I: IntElement, BT: BoolElement {
    fn forward_block(&self, tokens: Tensor<DeviceBackend<R, F, I, BT>, 2, Int>,
        cache: &mut KimiLinearCache<DeviceBackend<R, F, I, BT>>, last: bool)
        -> Result<Tensor<DeviceBackend<R, F, I, BT>, 3>, GenerationError> {
        let [b, s] = tokens.dims(); let c = &self.config;
        cache.validate(c, self.embedding.dtype())?;
        if b == 0 || s == 0 || cache.batch.is_some_and(|old| old != b)
            || cache.position.checked_add(s).is_none_or(|end| end > c.model_max_length)
            || tokens.device() != self.embedding.device() { return Err(error("invalid Kimi Linear cached input geometry")); }
        let mut hidden = self.embedding.clone().select(0, tokens.reshape([b * s])).reshape([b, s, c.hidden_size]);
        for (layer, state) in self.layers.iter().zip(&mut cache.layers) {
            let input = norm(hidden.clone(), &layer.input_norm, c.rms_norm_eps);
            let update = match (&layer.mixer, state) {
                (Mixer::Mla(m), LayerCache::Mla {key, value}) => m.forward(input, c, key, value, cache.position),
                (Mixer::Kda(m), LayerCache::Kda(state)) => m.forward(input, c, state, false)?.0,
                _ => return Err(error("Kimi Linear cache layer variant differs")),
            };
            hidden = hidden + update;
            let input = norm(hidden.clone(), &layer.post_norm, c.rms_norm_eps);
            hidden = hidden + self.feed_forward(layer, input)?;
        }
        if last { hidden = hidden.slice([0..b, s - 1..s, 0..c.hidden_size]); }
        let logits = self.head.forward(norm(hidden, &self.norm, c.rms_norm_eps));
        cache.position += s; cache.batch = Some(b);
        Ok(logits)
    }
}
impl<R, F, I, BT> CausalModel<DeviceBackend<R, F, I, BT>> for KimiLinearModel<DeviceBackend<R, F, I, BT>>
where R: DeviceRuntime, R::Server: ComputeServer, R::Device: DeviceOps, F: FloatElement, I: IntElement, BT: BoolElement {
    type Cache = KimiLinearCache<DeviceBackend<R, F, I, BT>>;
    fn new_cache(&self) -> Self::Cache { self.fresh_cache() }
    fn forward_cached_last(&self, tokens: Tensor<DeviceBackend<R, F, I, BT>, 2, Int>, cache: &mut Self::Cache)
        -> Tensor<DeviceBackend<R, F, I, BT>, 3> { self.forward_block(tokens, cache, true).expect("Kimi Linear forward") }
    fn try_forward_cached_last(&self, tokens: Tensor<DeviceBackend<R, F, I, BT>, 2, Int>, cache: &mut Self::Cache)
        -> Result<Tensor<DeviceBackend<R, F, I, BT>, 3>, GenerationError> { self.forward_block(tokens, cache, true) }
}
impl<R, F, I, BT> SpeculativeModel<DeviceBackend<R, F, I, BT>> for KimiLinearModel<DeviceBackend<R, F, I, BT>>
where R: DeviceRuntime, R::Server: ComputeServer, R::Device: DeviceOps, F: FloatElement, I: IntElement, BT: BoolElement {
    fn fork_cache(&self, cache: &Self::Cache) -> Self::Cache { cache.clone() }
    fn try_forward_cached_block(&self, tokens: Tensor<DeviceBackend<R, F, I, BT>, 2, Int>, cache: &mut Self::Cache)
        -> Result<Tensor<DeviceBackend<R, F, I, BT>, 3>, GenerationError> { self.forward_block(tokens, cache, false) }
}
