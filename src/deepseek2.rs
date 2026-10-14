use crate::gpu_inference::{
    TensorOps, gqa_scores, mla_absorb_query, mla_scores, mla_value_product,
};
use crate::huggingface::checkpoint::Checkpoint;
use crate::{
    CausalModel, CausalModelLimits, GenerationError, HuggingFaceLoadError, HuggingFaceLoadReport,
    SpeculativeModel,
};
use ruda_tensor::api::{
    DType, IndexingUpdateOp, Int, Tensor, TensorData,
    activation::{sigmoid, silu, softmax},
    backend::Backend,
};
use serde::Deserialize;
use std::{collections::BTreeSet, path::Path, sync::Arc};

mod cache;
mod batching;
mod routing;
mod feed_forward;
mod kimi_linear;
pub use kimi_linear::*;
mod weights;
mod mtp;
pub use mtp::{DeepseekMtp, DeepseekMtpCache, DeepseekMtpGenerationCache, DeepseekMtpGenerationCacheRecord, DeepseekMtpOutput, LoadedDeepseekMtp,
    load_huggingface_deepseek_mtp, load_huggingface_mla_mtp};
pub use mtp::{DeepseekMtpBatchCache,DeepseekMtpBatchPrefix};
#[cfg(feature="tensor-parallel")]
pub use mtp::parallel::{DeepseekMtpExpertParallel,DeepseekMtpExpertParallelCache,DeepseekMtpExpertParallelCacheRecord,
    LoadedDeepseekMtpExpertParallel,load_huggingface_deepseek_mtp_expert_parallel,load_huggingface_mla_mtp_expert_parallel};
pub use weights::{MlaFp8Backend, MlaFp8Projection, load_huggingface_mla};
pub use cache::{DeepseekV2Cache, DeepseekV2CacheRecord};
#[cfg(feature = "tensor-parallel")]
pub mod parallel;

#[derive(Clone, Debug, Deserialize)]
pub struct DeepseekV2Config {
    pub model_type: String,
    pub vocab_size: usize,
    pub hidden_size: usize,
    pub intermediate_size: usize,
    pub moe_intermediate_size: usize,
    pub num_hidden_layers: usize,
    pub num_attention_heads: usize,
    pub num_key_value_heads: usize,
    pub max_position_embeddings: usize,
    pub kv_lora_rank: usize,
    pub q_lora_rank: Option<usize>,
    pub qk_nope_head_dim: usize,
    pub qk_rope_head_dim: usize,
    pub v_head_dim: usize,
    pub n_routed_experts: usize,
    pub n_shared_experts: Option<usize>,
    pub num_experts_per_tok: usize,
    pub first_k_dense_replace: usize,
    pub moe_layer_freq: usize,
    pub norm_topk_prob: bool,
    pub routed_scaling_factor: f64,
    pub scoring_func: String,
    pub topk_method: String,
    pub n_group: usize,
    pub topk_group: usize,
    pub rms_norm_eps: f64,
    pub rope_theta: f64,
    pub rope_scaling: Option<DeepseekV2Yarn>,
    pub hidden_act: String,
    pub attention_bias: bool,
    pub attention_dropout: f64,
    pub tie_word_embeddings: bool,
    #[serde(default = "one")]
    pub pretraining_tp: usize,
    pub eos_token_id: Option<serde_json::Value>,
    #[serde(default)]
    pub quantization_config: Option<serde_json::Value>,
}

#[derive(Clone, Debug, Deserialize)]
pub struct DeepseekV2Yarn {
    #[serde(rename = "type", alias = "rope_type")]
    pub kind: String,
    pub factor: f64,
    pub original_max_position_embeddings: usize,
    pub beta_fast: f64,
    pub beta_slow: f64,
    pub mscale: f64,
    pub mscale_all_dim: f64,
}

impl DeepseekV2Config {
    pub fn limits(&self) -> CausalModelLimits {
        CausalModelLimits {
            vocab_size: self.vocab_size,
            max_sequence_length: self.max_position_embeddings,
        }
    }
    fn validate(&self) -> Result<(), HuggingFaceLoadError> {
        if !matches!(self.model_type.as_str(), "deepseek_v2" | "deepseek_v3" | "kimi_k2")
            || [
                self.vocab_size,
                self.hidden_size,
                self.intermediate_size,
                self.moe_intermediate_size,
                self.num_hidden_layers,
                self.num_attention_heads,
                self.max_position_embeddings,
                self.kv_lora_rank,
                self.qk_nope_head_dim,
                self.qk_rope_head_dim,
                self.v_head_dim,
                self.n_routed_experts,
                self.num_experts_per_tok,
                self.moe_layer_freq,
            ]
            .contains(&0)
            || self.vocab_size > i32::MAX as usize
            || self.n_routed_experts > i32::MAX as usize
            || self.num_experts_per_tok > self.n_routed_experts
            || self.q_lora_rank == Some(0)
            || self.n_shared_experts == Some(0)
            || self.num_key_value_heads != self.num_attention_heads
            || !self.qk_rope_head_dim.is_multiple_of(2)
            || self.hidden_act != "silu"
            || !routing::valid(self)
            || self.pretraining_tp != 1
            || self.attention_dropout != 0.0
            || self.tie_word_embeddings
            || !self.rms_norm_eps.is_finite()
            || self.rms_norm_eps <= 0.0
            || !self.rope_theta.is_finite()
            || self.rope_theta <= 1.0
            || !self.routed_scaling_factor.is_finite()
            || self.routed_scaling_factor <= 0.0
            || weights::block_config(self.quantization_config.as_ref()).is_err()
            || self.rope_scaling.as_ref().is_some_and(|s| {
                s.kind != "yarn"
                    || !s.factor.is_finite()
                    || s.factor < 1.0
                    || s.original_max_position_embeddings == 0
                    || !s.beta_fast.is_finite()
                    || !s.beta_slow.is_finite()
                    || s.beta_slow <= 0.0
                    || s.beta_fast < s.beta_slow
                    || !s.mscale.is_finite()
                    || !s.mscale_all_dim.is_finite()
                    || s.mscale < 0.0
                    || s.mscale_all_dim < 0.0
            })
        {
            return Err(HuggingFaceLoadError("incompatible DeepSeek/Kimi MLA geometry, routing, quantization or RoPE configuration".into()));
        }
        self.qk_nope_head_dim
            .checked_add(self.qk_rope_head_dim)
            .and_then(|n| n.checked_mul(self.num_attention_heads))
            .and_then(|n| n.checked_mul(self.hidden_size))
            .and_then(|_| self.kv_lora_rank.checked_add(self.qk_rope_head_dim))
            .ok_or_else(|| HuggingFaceLoadError("DeepSeek-V2 geometry overflow".into()))?;
        Ok(())
    }
}

fn one() -> usize { 1 }

enum Projection<B: Backend> {
    Dense { weight: Tensor<B, 2>, bias: Option<Tensor<B, 1>> },
    Packed { forward: Arc<dyn Fn(Tensor<B, 3>) -> Tensor<B, 3> + Send + Sync>,
        materialize: Arc<dyn Fn() -> Tensor<B, 2> + Send + Sync>, dtype: DType },
}
impl<B: Backend> Projection<B> {
    fn forward(&self, input: Tensor<B, 3>) -> Tensor<B, 3> {
        let Self::Dense { weight, bias } = self else {
            let Self::Packed { forward, .. } = self else { unreachable!() };
            return forward(input);
        };
        let output = input.matmul(weight.clone().transpose().unsqueeze::<3>());
        match bias {
            Some(bias) => output + bias.clone().unsqueeze::<3>(),
            None => output,
        }
    }
    fn weight(&self) -> Tensor<B, 2> {
        match self { Self::Dense { weight, .. } => weight.clone(), Self::Packed { materialize, .. } => materialize() }
    }
    fn dtype(&self) -> DType {
        match self { Self::Dense { weight, .. } => weight.dtype(), Self::Packed { dtype, .. } => *dtype }
    }
}
struct Mlp<B: Backend> {
    gate: Projection<B>,
    up: Projection<B>,
    down: Projection<B>,
}
impl<B: Backend> Mlp<B> {
    fn forward(&self, input: Tensor<B, 3>) -> Tensor<B, 3> {
        self.down
            .forward(silu(self.gate.forward(input.clone())) * self.up.forward(input))
    }
}
enum FeedForward<B: Backend> {
    Dense(Mlp<B>),
    Experts {
        router: Tensor<B, 2>,
        correction: Option<Tensor<B, 1>>,
        experts: Vec<(usize, Mlp<B>)>,
        shared: Option<Mlp<B>>,
    },
}
struct Layer<B: Backend> {
    q: Projection<B>,
    q_a: Option<(Projection<B>, Tensor<B, 1>)>,
    kv_a: Projection<B>,
    kv_norm: Tensor<B, 1>,
    kv_b: Projection<B>,
    o: Projection<B>,
    input_norm: Tensor<B, 1>,
    post_norm: Tensor<B, 1>,
    mlp: FeedForward<B>,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum DeepseekV2AttentionMode {
    Expanded,
    Absorbed,
}

pub struct DeepseekV2Model<B: Backend> {
    pub config: DeepseekV2Config,
    embedding: Tensor<B, 2>,
    layers: Vec<Layer<B>>,
    norm: Tensor<B, 1>,
    head: Projection<B>,
    attention_mode: DeepseekV2AttentionMode,
    reduction: Option<Arc<dyn Fn(Tensor<B, 3>) -> Tensor<B, 3> + Send + Sync>>,
}
pub struct LoadedDeepseekV2<B: Backend> {
    pub model: DeepseekV2Model<B>,
    pub eos_token_ids: Vec<i32>,
    pub report: HuggingFaceLoadReport,
    pub auxiliary_tensors: Vec<String>,
}

fn norm<B: Backend, const D: usize>(
    input: Tensor<B, D>,
    weight: &Tensor<B, 1>,
    epsilon: f64,
) -> Tensor<B, D> {
    let dtype = input.dtype();
    let input = input.cast(DType::F32);
    let variance = (input.clone() * input.clone()).mean_dim(D - 1);
    (input / (variance + epsilon).sqrt()).cast(dtype) * weight.clone().unsqueeze::<D>()
}
fn ops<B: Backend>() -> TensorOps<Tensor<B, 4>> {
    TensorOps {
        shape: |x| x.dims(),
        reshape: |x, s| x.reshape(s),
        swap_dims: |x, a, b| x.swap_dims(a, b),
        matmul: |a, b| a.matmul(b),
        add: |a, b| a + b,
    }
}
fn rotate<B: Backend>(input: Tensor<B, 4>, cos: Tensor<B, 4>, sin: Tensor<B, 4>) -> Tensor<B, 4> {
    let [b, h, t, d] = input.dims();
    let input = input
        .reshape([b, h, t, d / 2, 2])
        .swap_dims(3, 4)
        .reshape([b, h, t, d]);
    let first = input.clone().slice([0..b, 0..h, 0..t, 0..d / 2]);
    let second = input.clone().slice([0..b, 0..h, 0..t, d / 2..d]);
    input * cos + Tensor::cat(vec![-second, first], 3) * sin
}
fn mscale(factor: f64, scale: f64) -> f64 {
    if factor <= 1.0 {
        1.0
    } else {
        1.0 + 0.1 * scale * factor.ln()
    }
}

impl<B: Backend> DeepseekV2Model<B> {
    pub fn with_attention_mode(mut self, mode: DeepseekV2AttentionMode) -> Self {
        self.attention_mode = mode;
        self
    }

    fn rotary(
        &self,
        start: usize,
        sequence: usize,
        dtype: DType,
        device: &B::Device,
    ) -> (Tensor<B, 4>, Tensor<B, 4>, f64) {
        let c = &self.config;
        let width = c.qk_rope_head_dim;
        let mut inverse = (0..width / 2)
            .map(|i| 1.0f32 / (c.rope_theta as f32).powf((2 * i) as f32 / width as f32))
            .collect::<Vec<_>>();
        let mut amplitude = 1.0;
        let mut scale = 1.0 / ((c.qk_nope_head_dim + width) as f64).sqrt();
        if let Some(yarn) = &c.rope_scaling {
            let correction = |rotations: f64| {
                width as f64
                    * (yarn.original_max_position_embeddings as f64
                        / (rotations * 2.0 * std::f64::consts::PI))
                        .ln()
                    / (2.0 * c.rope_theta.ln())
            };
            let low = correction(yarn.beta_fast).floor().max(0.0);
            let high = correction(yarn.beta_slow).ceil().min((width - 1) as f64);
            let high = if high == low { high + 0.001 } else { high };
            for (i, frequency) in inverse.iter_mut().enumerate() {
                let ramp = ((i as f32 - low as f32) / (high - low) as f32).clamp(0.0, 1.0);
                *frequency = (*frequency / yarn.factor as f32) * ramp + *frequency * (1.0 - ramp);
            }
            amplitude = mscale(yarn.factor, yarn.mscale) / mscale(yarn.factor, yarn.mscale_all_dim);
            scale *= mscale(yarn.factor, yarn.mscale_all_dim).powi(2);
        }
        let frequencies = Tensor::<B, 2>::from_data(
            TensorData::new(
                (start..start + sequence)
                    .map(|i| i as f32)
                    .collect::<Vec<_>>(),
                [sequence, 1],
            ),
            (device, DType::F32),
        ) * Tensor::<B, 2>::from_data(
            TensorData::new(inverse, [1, width / 2]),
            (device, DType::F32),
        );
        let frequencies =
            Tensor::cat(vec![frequencies.clone(), frequencies], 1).reshape([1, 1, sequence, width]);
        (
            (frequencies.clone().cos() * amplitude).cast(dtype),
            (frequencies.sin() * amplitude).cast(dtype),
            scale,
        )
    }

    fn feed_forward(&self, layer: &Layer<B>, input: Tensor<B, 3>) -> Result<Tensor<B, 3>, GenerationError> {
        layer.mlp.run(input, &self.config, self.reduction.as_ref())
    }

    fn layer_forward(&self, layer: &Layer<B>, mut hidden: Tensor<B, 3>, state: &mut cache::LayerCache<B>,
        start: usize, cos: Tensor<B, 4>, sin: Tensor<B, 4>, scale: f64) -> Result<Tensor<B, 3>, GenerationError> {
        let c = &self.config;
        let sequence = hidden.dims()[1];
        let end = start + sequence;
        let width = c.kv_lora_rank + c.qk_rope_head_dim;
        let dtype = hidden.dtype();
        let device = hidden.device();
        let input = norm(hidden.clone(), &layer.input_norm, c.rms_norm_eps);
        let q_input = match &layer.q_a {
            Some((project, weight)) => norm(project.forward(input.clone()), weight, 1e-6),
            None => input.clone(),
        };
        let query = layer
            .q
            .forward(q_input)
            .reshape([
                1,
                sequence,
                c.num_attention_heads,
                c.qk_nope_head_dim + c.qk_rope_head_dim,
            ])
            .swap_dims(1, 2);
        let content_query = query.clone().slice([
            0..1,
            0..c.num_attention_heads,
            0..sequence,
            0..c.qk_nope_head_dim,
        ]);
        let position_query = rotate(
            query.slice([
                0..1,
                0..c.num_attention_heads,
                0..sequence,
                c.qk_nope_head_dim..c.qk_nope_head_dim + c.qk_rope_head_dim,
            ]),
            cos.clone(),
            sin.clone(),
        );
        let compressed = layer.kv_a.forward(input);
        let latent = norm(
            compressed
                .clone()
                .slice([0..1, 0..sequence, 0..c.kv_lora_rank]),
            &layer.kv_norm,
            1e-6,
        )
        .reshape([1, 1, sequence, c.kv_lora_rank]);
        let position = rotate(
            compressed
                .slice([0..1, 0..sequence, c.kv_lora_rank..width])
                .reshape([1, 1, sequence, c.qk_rope_head_dim]),
            cos.clone(),
            sin.clone(),
        );
        let history = state.append(
            Tensor::cat(vec![latent, position], 3),
            c.max_position_embeddings,
        )?;
        let latent = history
            .clone()
            .slice([0..1, 0..1, 0..end, 0..c.kv_lora_rank]);
        let position = history.slice([0..1, 0..1, 0..end, c.kv_lora_rank..width]);
        let (mut scores, values, value_weights) = match self.attention_mode {
            DeepseekV2AttentionMode::Absorbed => {
                let weights = layer.kv_b.weight().reshape([1, c.num_attention_heads,
                    c.qk_nope_head_dim + c.v_head_dim, c.kv_lora_rank]);
                let key_weights = weights.clone().slice([0..1, 0..c.num_attention_heads,
                    0..c.qk_nope_head_dim, 0..c.kv_lora_rank]);
                let value_weights = weights.slice([0..1, 0..c.num_attention_heads,
                    c.qk_nope_head_dim..c.qk_nope_head_dim + c.v_head_dim, 0..c.kv_lora_rank]).swap_dims(2, 3);
                let query = mla_absorb_query(&ops(), content_query, key_weights);
                (
                    mla_scores(&ops(), query, position_query, latent.clone(), position),
                    None,
                    Some(value_weights),
                )
            }
            DeepseekV2AttentionMode::Expanded => {
                let expanded = layer.kv_b.forward(latent.clone().reshape([1, end, c.kv_lora_rank]))
                    .reshape([
                        1,
                        end,
                        c.num_attention_heads,
                        c.qk_nope_head_dim + c.v_head_dim,
                    ])
                    .swap_dims(1, 2);
                let content_key = expanded.clone().slice([
                    0..1,
                    0..c.num_attention_heads,
                    0..end,
                    0..c.qk_nope_head_dim,
                ]);
                let value = expanded.slice([
                    0..1,
                    0..c.num_attention_heads,
                    0..end,
                    c.qk_nope_head_dim..c.qk_nope_head_dim + c.v_head_dim,
                ]);
                let query = Tensor::cat(vec![content_query, position_query], 3);
                let key = Tensor::cat(
                    vec![content_key, position.repeat_dim(1, c.num_attention_heads)],
                    3,
                );
                (gqa_scores(&ops(), query, key), Some(value), None)
            }
        };
        scores = scores * scale;
        if sequence > 1 {
            let mask = (0..sequence)
                .flat_map(|i| {
                    (0..end).map(move |j| {
                        if j <= start + i {
                            0.0f32
                        } else {
                            f32::NEG_INFINITY
                        }
                    })
                })
                .collect::<Vec<_>>();
            scores = scores
                + Tensor::<B, 4>::from_data(
                    TensorData::new(mask, [1, 1, sequence, end]),
                    (&device, dtype),
                );
        }
        let probabilities = softmax(scores.cast(DType::F32), 3).cast(dtype);
        let context = match values {
            Some(values) => probabilities.matmul(values),
            None => mla_value_product(&ops(), probabilities, latent, value_weights.expect("absorbed value weights")),
        };
        hidden = hidden
            + layer.o.forward(context.swap_dims(1, 2).reshape([
                1,
                sequence,
                c.num_attention_heads * c.v_head_dim,
            ]));
        let input = norm(hidden.clone(), &layer.post_norm, c.rms_norm_eps);
        hidden = hidden + self.feed_forward(layer, input)?;
        Ok(hidden)
    }

    fn forward_block(&self, tokens: Tensor<B, 2, Int>, cache: &mut DeepseekV2Cache<B>, last_only: bool)
        -> Result<Tensor<B, 3>, GenerationError> {
        self.forward_features(tokens, cache, last_only).map(|(logits, _)| logits)
    }

    pub fn forward_features(
        &self,
        tokens: Tensor<B, 2, Int>,
        cache: &mut DeepseekV2Cache<B>,
        last_only: bool,
    ) -> Result<(Tensor<B, 3>, Tensor<B, 3>), GenerationError> {
        self.forward_features_impl(tokens,cache,last_only,false)
    }

    fn forward_features_impl(&self,tokens:Tensor<B,2,Int>,cache:&mut DeepseekV2Cache<B>,last_only:bool,retain_all_hidden:bool)
        ->Result<(Tensor<B,3>,Tensor<B,3>),GenerationError> {
        let [batch, sequence] = tokens.dims();
        let c = &self.config;
        let start = cache.position();
        let width = c.kv_lora_rank + c.qk_rope_head_dim;
        let device = tokens.device();
        if batch != 1
            || sequence == 0
            || self.embedding.device() != device
            || !cache.compatible(
                c.num_hidden_layers,
                c.max_position_embeddings,
                width,
                &device,
                self.embedding.dtype(),
            )
            || start
                .checked_add(sequence)
                .is_none_or(|end| end > c.max_position_embeddings)
        {
            return Err(GenerationError(
                "incompatible DeepSeek-V2 cache, device or context length".into(),
            ));
        }
        let mut hidden = self
            .embedding
            .clone()
            .select(0, tokens.reshape([sequence]))
            .reshape([1, sequence, c.hidden_size]);
        let dtype = hidden.dtype();
        let (cos, sin, scale) = self.rotary(start, sequence, dtype, &device);
        let end = start + sequence;
        for (layer, state) in self.layers.iter().zip(&mut cache.layers) {
            hidden = self.layer_forward(layer, hidden, state, start, cos.clone(), sin.clone(), scale)?;
        }
        cache.position = end;
        if last_only && !retain_all_hidden {
            hidden = hidden.slice([0..1, sequence - 1..sequence, 0..c.hidden_size]);
        }
        let normalized = norm(hidden, &self.norm, c.rms_norm_eps);
        let selected=if last_only && retain_all_hidden {normalized.clone().slice([0..1,sequence-1..sequence,0..c.hidden_size])} else {normalized.clone()};
        Ok((self.head.forward(selected), normalized))
    }
}

impl<B: Backend> CausalModel<B> for DeepseekV2Model<B> {
    type Cache = DeepseekV2Cache<B>;
    fn new_cache(&self) -> Self::Cache {
        DeepseekV2Cache::new(
            self.config.num_hidden_layers,
            self.config.max_position_embeddings,
            self.config.kv_lora_rank + self.config.qk_rope_head_dim,
        )
    }
    fn forward_cached_last(
        &self,
        tokens: Tensor<B, 2, Int>,
        cache: &mut Self::Cache,
    ) -> Tensor<B, 3> {
        self.forward_block(tokens, cache, true)
            .expect("validated DeepSeek-V2 forward")
    }
    fn try_forward_cached_last(
        &self,
        tokens: Tensor<B, 2, Int>,
        cache: &mut Self::Cache,
    ) -> Result<Tensor<B, 3>, GenerationError> {
        self.forward_block(tokens, cache, true)
    }
}
impl<B: Backend> SpeculativeModel<B> for DeepseekV2Model<B> {
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

fn projection<B: Backend>(
    checkpoint: &mut weights::Weights<B>,
    prefix: &str,
    input: usize,
    output: usize,
    bias: bool,
    device: &B::Device,
    dtype: DType,
) -> Result<Projection<B>, HuggingFaceLoadError> {
    if checkpoint.is_fp8(prefix) { return checkpoint.packed_projection(prefix, input, output, bias, device, dtype); }
    let weight = checkpoint.tensor(&format!("{prefix}.weight"), [output, input], device)?;
    let bias = if bias {
        Some(checkpoint.tensor(&format!("{prefix}.bias"), [output], device)?)
    } else {
        None
    };
    if weight.dtype() != dtype || bias.as_ref().is_some_and(|b| b.dtype() != dtype) {
        return Err(HuggingFaceLoadError(format!(
            "mixed storage dtype for {prefix}"
        )));
    }
    Ok(Projection::Dense { weight, bias })
}
fn load_mlp<B: Backend>(
    checkpoint: &mut weights::Weights<B>,
    prefix: &str,
    hidden: usize,
    intermediate: usize,
    device: &B::Device,
    dtype: DType,
) -> Result<Mlp<B>, HuggingFaceLoadError> {
    Ok(Mlp {
        gate: projection(
            checkpoint,
            &format!("{prefix}.gate_proj"),
            hidden,
            intermediate,
            false,
            device,
            dtype,
        )?,
        up: projection(
            checkpoint,
            &format!("{prefix}.up_proj"),
            hidden,
            intermediate,
            false,
            device,
            dtype,
        )?,
        down: projection(
            checkpoint,
            &format!("{prefix}.down_proj"),
            intermediate,
            hidden,
            false,
            device,
            dtype,
        )?,
    })
}

pub fn load_huggingface_deepseek_v2<B: Backend>(
    directory: impl AsRef<Path>,
    device: &B::Device,
) -> Result<LoadedDeepseekV2<B>, HuggingFaceLoadError> {
    load_rank(directory.as_ref(), device, None, None)
}

struct ExpertShard {
    rank: usize,
    world: usize,
}

fn load_rank<B: Backend>(
    directory: &Path,
    device: &B::Device,
    shard: Option<ExpertShard>,
    reduction: Option<Arc<dyn Fn(Tensor<B, 3>) -> Tensor<B, 3> + Send + Sync>>,
) -> Result<LoadedDeepseekV2<B>, HuggingFaceLoadError> {
    load_rank_with(directory, device, shard, reduction, None)
}

fn load_layer<B: Backend>(checkpoint: &mut weights::Weights<B>, config: &DeepseekV2Config,
    i: usize, shard: Option<&ExpertShard>, excluded: &mut BTreeSet<String>, device: &B::Device, dtype: DType)
    -> Result<Layer<B>, HuggingFaceLoadError> {
    let p = format!("model.layers.{i}");
    let attention = format!("{p}.self_attn");
    let q_a = if let Some(rank) = config.q_lora_rank {
        Some((
            projection(
                checkpoint,
                &format!("{attention}.q_a_proj"),
                config.hidden_size,
                rank,
                config.attention_bias,
                device,
                dtype,
            )?,
            checkpoint.tensor(&format!("{attention}.q_a_layernorm.weight"), [rank], device)?,
        ))
    } else {
        None
    };
    let q = projection(
        checkpoint,
        &format!(
            "{attention}.{}",
            if q_a.is_some() { "q_b_proj" } else { "q_proj" }
        ),
        config.q_lora_rank.unwrap_or(config.hidden_size),
        config.num_attention_heads * (config.qk_nope_head_dim + config.qk_rope_head_dim),
        false,
        device,
        dtype,
    )?;
    let kv_a = projection(
        checkpoint,
        &format!("{attention}.kv_a_proj_with_mqa"),
        config.hidden_size,
        config.kv_lora_rank + config.qk_rope_head_dim,
        config.attention_bias,
        device,
        dtype,
    )?;
    let kv_norm = checkpoint.tensor(
        &format!("{attention}.kv_a_layernorm.weight"),
        [config.kv_lora_rank],
        device,
    )?;
    let kv_b = projection(checkpoint, &format!("{attention}.kv_b_proj"),
        config.kv_lora_rank, config.num_attention_heads * (config.qk_nope_head_dim + config.v_head_dim),
        false, device, dtype)?;
    let o = projection(
        checkpoint,
        &format!("{attention}.o_proj"),
        config.num_attention_heads * config.v_head_dim,
        config.hidden_size,
        config.attention_bias,
        device,
        dtype,
    )?;
    let mlp =
        if i >= config.first_k_dense_replace && i % config.moe_layer_freq == 0 {
            let router = checkpoint.tensor(
                &format!("{p}.mlp.gate.weight"),
                [config.n_routed_experts, config.hidden_size],
                device,
            )?;
            let correction = if config.topk_method == "noaux_tc" {
                Some(checkpoint.tensor(&format!("{p}.mlp.gate.e_score_correction_bias"),
                    [config.n_routed_experts], device)?)
            } else { None };
            let mut experts = Vec::new();
            for expert in 0..config.n_routed_experts {
                let prefix = format!("{p}.mlp.experts.{expert}");
                let owned = shard
                    .as_ref()
                    .is_none_or(|s| expert / (config.n_routed_experts / s.world) == s.rank);
                if owned {
                    experts.push((
                        expert,
                        load_mlp(
                            checkpoint,
                            &prefix,
                            config.hidden_size,
                            config.moe_intermediate_size,
                            device,
                            dtype,
                        )?,
                    ));
                } else {
                    for (projection, shape) in [
                        (
                            "gate_proj",
                            [config.moe_intermediate_size, config.hidden_size],
                        ),
                        (
                            "up_proj",
                            [config.moe_intermediate_size, config.hidden_size],
                        ),
                        (
                            "down_proj",
                            [config.hidden_size, config.moe_intermediate_size],
                        ),
                    ] {
                        checkpoint.exclude_projection(&format!("{prefix}.{projection}"), shape, dtype, excluded)?;
                    }
                }
            }
            let shared =
                config
                    .n_shared_experts
                    .map(|count| {
                        let intermediate =
                            config.moe_intermediate_size.checked_mul(count).ok_or_else(
                                || HuggingFaceLoadError("shared expert shape overflow".into()),
                            )?;
                        load_mlp(
                            checkpoint,
                            &format!("{p}.mlp.shared_experts"),
                            config.hidden_size,
                            intermediate,
                            device,
                            dtype,
                        )
                    })
                    .transpose()?;
            FeedForward::Experts {
                router,
                correction,
                experts,
                shared,
            }
        } else {
            FeedForward::Dense(load_mlp(
                checkpoint,
                &format!("{p}.mlp"),
                config.hidden_size,
                config.intermediate_size,
                device,
                dtype,
            )?)
        };
    let input_norm = checkpoint.tensor(
        &format!("{p}.input_layernorm.weight"),
        [config.hidden_size],
        device,
    )?;
    let post_norm = checkpoint.tensor(
        &format!("{p}.post_attention_layernorm.weight"),
        [config.hidden_size],
        device,
    )?;
    if [
        kv_norm.dtype(),
        kv_b.dtype(),
        input_norm.dtype(),
        post_norm.dtype(),
    ]
    .iter()
    .any(|&d| d != dtype)
        || q_a
            .as_ref()
            .is_some_and(|(_, weight)| weight.dtype() != dtype)
        || matches!(&mlp, FeedForward::Experts {router, ..}
            if router.dtype() != dtype && router.dtype() != DType::F32)
    {
        return Err(HuggingFaceLoadError(
            "DeepSeek-V2 normalization/router storage dtypes differ".into(),
        ));
    }
    Ok(Layer {
        q,
        q_a,
        kv_a,
        kv_norm,
        kv_b,
        o,
        input_norm,
        post_norm,
        mlp,
    })
}

fn load_rank_with<B: Backend>(directory: &Path, device: &B::Device, shard: Option<ExpertShard>,
    reduction: Option<Arc<dyn Fn(Tensor<B, 3>) -> Tensor<B, 3> + Send + Sync>>,
    packed: Option<weights::Loader<B>>) -> Result<LoadedDeepseekV2<B>, HuggingFaceLoadError> {
    let directory = directory
        .canonicalize()
        .map_err(|e| HuggingFaceLoadError(e.to_string()))?;
    let source: serde_json::Value = serde_json::from_slice(
        &std::fs::read(directory.join("config.json"))
            .map_err(|e| HuggingFaceLoadError(e.to_string()))?,
    )
    .map_err(|e| HuggingFaceLoadError(e.to_string()))?;
    let config: DeepseekV2Config = serde_json::from_value(source.clone()).map_err(|e| HuggingFaceLoadError(e.to_string()))?;
    config.validate()?;
    if shard.as_ref().is_some_and(|s| {
        s.world == 0 || s.rank >= s.world || !config.n_routed_experts.is_multiple_of(s.world)
    }) {
        return Err(HuggingFaceLoadError(
            "EP requires equal nonempty native expert shards".into(),
        ));
    }
    let mut checkpoint = weights::Weights::open(&directory, &config, packed)?;
    let embedding = checkpoint.tensor(
        "model.embed_tokens.weight",
        [config.vocab_size, config.hidden_size],
        device,
    )?;
    let dtype = embedding.dtype();
    let mut excluded = BTreeSet::new();
    let mut layers = Vec::with_capacity(config.num_hidden_layers);
    for i in 0..config.num_hidden_layers {
        layers.push(load_layer(&mut checkpoint, &config, i, shard.as_ref(), &mut excluded, device, dtype)?);
    }
    let norm = checkpoint.tensor("model.norm.weight", [config.hidden_size], device)?;
    if norm.dtype() != dtype {
        return Err(HuggingFaceLoadError(
            "DeepSeek-V2 final normalization dtype differs".into(),
        ));
    }
    let head = projection(
        &mut checkpoint,
        "lm_head",
        config.hidden_size,
        config.vocab_size,
        false,
        device,
        dtype,
    )?;
    let nextn = match source.get("num_nextn_predict_layers") {
        None => 0,
        Some(value) => value.as_u64().and_then(|n| usize::try_from(n).ok())
            .ok_or_else(|| HuggingFaceLoadError("invalid num_nextn_predict_layers".into()))?,
    };
    let auxiliary_end = config.num_hidden_layers.checked_add(nextn)
        .ok_or_else(|| HuggingFaceLoadError("MTP layer range overflow".into()))?;
    let auxiliary_tensors = checkpoint.tensors.keys().filter(|name| {
        name.strip_prefix("model.layers.").and_then(|rest| rest.split('.').next())
            .and_then(|layer| layer.parse::<usize>().ok())
            .is_some_and(|layer| (config.num_hidden_layers..auxiliary_end).contains(&layer))
    }).cloned().collect::<Vec<_>>();
    let auxiliary = auxiliary_tensors.iter().collect::<BTreeSet<_>>();
    let unused = checkpoint
        .tensors
        .keys()
        .filter(|name| !checkpoint.consumed.contains(*name) && !excluded.contains(*name) && !auxiliary.contains(name))
        .cloned()
        .collect::<Vec<_>>();
    if !unused.is_empty() {
        return Err(HuggingFaceLoadError(format!(
            "unrepresented DeepSeek-V2 tensors: {}",
            unused.join(", ")
        )));
    }
    let generation_path = directory.join("generation_config.json");
    let generation: Option<serde_json::Value> = if generation_path.is_file() {
        Some(
            serde_json::from_slice(
                &std::fs::read(&generation_path)
                    .map_err(|e| HuggingFaceLoadError(e.to_string()))?,
            )
            .map_err(|e| HuggingFaceLoadError(e.to_string()))?,
        )
    } else {
        None
    };
    let eos = generation
        .as_ref()
        .and_then(|v| v.get("eos_token_id"))
        .filter(|v| !v.is_null())
        .cloned()
        .or_else(|| config.eos_token_id.clone())
        .unwrap_or(serde_json::Value::Null);
    let eos = match eos {
        serde_json::Value::Null => vec![],
        serde_json::Value::Array(v) => v,
        v => vec![v],
    };
    let eos_token_ids = eos
        .into_iter()
        .map(|v| {
            v.as_i64()
                .and_then(|v| i32::try_from(v).ok())
                .filter(|&v| v >= 0 && (v as usize) < config.vocab_size)
                .ok_or_else(|| HuggingFaceLoadError("invalid DeepSeek-V2 EOS ID".into()))
        })
        .collect::<Result<Vec<_>, _>>()?;
    B::sync(device).map_err(|e| HuggingFaceLoadError(e.to_string()))?;
    let report = HuggingFaceLoadReport {
        config_path: directory.join("config.json"),
        weight_files: checkpoint.files.clone(),
        applied_tensors: checkpoint.consumed.len(),
        tied_word_embeddings: false,
    };
    Ok(LoadedDeepseekV2 {
        model: DeepseekV2Model {
            config,
            embedding,
            layers,
            norm,
            head,
            attention_mode: DeepseekV2AttentionMode::Expanded,
            reduction,
        },
        eos_token_ids,
        report,
        auxiliary_tensors,
    })
}
