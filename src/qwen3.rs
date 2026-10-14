use crate::gpu_inference::{TensorOps, gqa_scores, gqa_value_product};
use crate::huggingface::checkpoint::Checkpoint;
use crate::{
    CausalModel, CausalModelLimits, GenerationError, HuggingFaceLoadError, HuggingFaceLoadReport,
    LlamaKvCache, SpeculativeModel,
};
use ruda_tensor::api::{
    DType, Int, Tensor, TensorData,
    activation::{silu, softmax},
    backend::Backend,
};
use serde::Deserialize;
use std::path::Path;
use std::sync::Arc;

mod draft;
mod batching;
mod moe;
pub use moe::Qwen3MoeConfig;
#[cfg(feature = "tensor-parallel")]
pub mod parallel;
pub use draft::{
    Qwen3DraftCache, Qwen3DraftCacheRecord, Qwen3Drafted, load_huggingface_qwen3_draft,
    Qwen3DraftBatchCache,Qwen3DraftBatchPrefix,
};

#[derive(Clone, Debug, Deserialize)]
pub struct Qwen3Config {
    pub model_type: String,
    pub vocab_size: usize,
    pub hidden_size: usize,
    pub intermediate_size: usize,
    pub num_hidden_layers: usize,
    pub num_attention_heads: usize,
    pub num_key_value_heads: usize,
    pub head_dim: usize,
    pub max_position_embeddings: usize,
    pub rms_norm_eps: f64,
    pub rope_theta: f64,
    #[serde(default)]
    pub tie_word_embeddings: bool,
    #[serde(default)]
    pub attention_bias: bool,
    #[serde(default)]
    pub attention_dropout: f64,
    #[serde(default)]
    pub use_sliding_window: bool,
    #[serde(default)]
    pub rope_scaling: Option<serde_json::Value>,
    #[serde(default)]
    pub hidden_act: Option<String>,
    #[serde(default)]
    pub layer_types: Option<Vec<String>>,
    #[serde(default)]
    pub quantization_config: Option<serde_json::Value>,
    #[serde(default)]
    pub eos_token_id: Option<serde_json::Value>,
}

impl Qwen3Config {
    pub fn limits(&self) -> CausalModelLimits {
        CausalModelLimits {
            vocab_size: self.vocab_size,
            max_sequence_length: self.max_position_embeddings,
        }
    }
    fn validate(&self) -> Result<(), HuggingFaceLoadError> {
        if !matches!(self.model_type.as_str(), "qwen3" | "qwen3_moe")
            || [
                self.vocab_size,
                self.hidden_size,
                self.intermediate_size,
                self.num_hidden_layers,
                self.num_attention_heads,
                self.num_key_value_heads,
                self.head_dim,
                self.max_position_embeddings,
            ]
            .contains(&0)
            || self.vocab_size > i32::MAX as usize
            || !self.head_dim.is_multiple_of(2)
            || !self
                .num_attention_heads
                .is_multiple_of(self.num_key_value_heads)
            || !self.rms_norm_eps.is_finite()
            || self.rms_norm_eps <= 0.0
            || !self.rope_theta.is_finite()
            || self.rope_theta <= 0.0
            || self.attention_dropout != 0.0
            || self.use_sliding_window
            || self
                .quantization_config
                .as_ref()
                .is_some_and(|v| !v.is_null())
            || self.hidden_act.as_deref().is_some_and(|v| v != "silu")
            || self.layer_types.as_ref().is_some_and(|types| {
                types.len() != self.num_hidden_layers || types.iter().any(|v| v != "full_attention")
            })
        {
            return Err(HuggingFaceLoadError("requires full-attention Qwen3 dense/MoE, native floating weights and SiLU".into()));
        }
        self.num_attention_heads
            .checked_mul(self.head_dim)
            .and_then(|n| n.checked_mul(self.hidden_size))
            .ok_or_else(|| HuggingFaceLoadError("Qwen3 projection shape overflow".into()))?;
        self.rotary_parameters()?;
        Ok(())
    }

    fn rotary_parameters(&self) -> Result<crate::llama::rope::RopeParameters, HuggingFaceLoadError> {
        crate::huggingface::rope::layer_parameters(self.head_dim, self.max_position_embeddings, self.rope_theta,
            None, 1.0, self.rope_scaling.as_ref(), self.layer_types.as_deref(), self.num_hidden_layers)
    }
}

fn read_config(directory: &Path) -> Result<(Qwen3Config, Option<Qwen3MoeConfig>), HuggingFaceLoadError> {
    let mut source: serde_json::Value = serde_json::from_slice(&std::fs::read(directory.join("config.json"))
        .map_err(|e| HuggingFaceLoadError(e.to_string()))?).map_err(|e| HuggingFaceLoadError(e.to_string()))?;
    let mut rotary = source.get("rope_parameters").filter(|v| !v.is_null()).cloned()
        .or_else(|| source.get("rope_scaling").filter(|v| !v.is_null()).cloned()).unwrap_or_else(|| serde_json::json!({}));
    let object = rotary.as_object_mut().ok_or_else(|| HuggingFaceLoadError("Qwen3 RoPE parameters must be an object".into()))?;
    let defaults = ["original_max_position_embeddings", "partial_rotary_factor"];
    if object.values().any(serde_json::Value::is_object) {
        for value in object.values_mut().filter_map(serde_json::Value::as_object_mut) {
            for name in defaults { if let Some(default) = source.get(name).filter(|v| !v.is_null()) {
                value.entry(name).or_insert_with(|| default.clone());
            }}
        }
    } else {
        for name in defaults { if let Some(default) = source.get(name).filter(|v| !v.is_null()) {
            object.entry(name).or_insert_with(|| default.clone());
        }}
    }
    if let Some(theta) = object.get("rope_theta") { source["rope_theta"] = theta.clone(); }
    source["rope_scaling"] = rotary;
    let config: Qwen3Config = serde_json::from_value(source.clone()).map_err(|e| HuggingFaceLoadError(e.to_string()))?;
    let moe = if config.model_type == "qwen3_moe" {
        let moe: Qwen3MoeConfig = serde_json::from_value(source).map_err(|e| HuggingFaceLoadError(e.to_string()))?;
        moe.validate(&config)?; Some(moe)
    } else { None };
    Ok((config, moe))
}

struct Projection<B: Backend> {
    weight: Tensor<B, 2>,
    bias: Option<Tensor<B, 1>>,
}
impl<B: Backend> Projection<B> {
    fn forward(&self, x: Tensor<B, 3>) -> Tensor<B, 3> {
        let result = x.matmul(self.weight.clone().transpose().unsqueeze::<3>());
        if let Some(bias) = &self.bias {
            result + bias.clone().unsqueeze::<3>()
        } else {
            result
        }
    }
}
struct Layer<B: Backend> {
    q: Projection<B>,
    k: Projection<B>,
    v: Projection<B>,
    o: Projection<B>,
    mlp: moe::FeedForward<B>,
    input_norm: Tensor<B, 1>,
    post_norm: Tensor<B, 1>,
    q_norm: Tensor<B, 1>,
    k_norm: Tensor<B, 1>,
}

pub struct Qwen3Model<B: Backend> {
    pub config: Qwen3Config,
    pub moe_config: Option<Qwen3MoeConfig>,
    embedding: Tensor<B, 2>,
    layers: Vec<Layer<B>>,
    norm: Tensor<B, 1>,
    head: Projection<B>,
    rotary: Vec<crate::llama::rope::RotaryState>,
    reduction: Option<Arc<dyn Fn(Tensor<B, 3>) -> Tensor<B, 3> + Send + Sync>>,
}
pub struct LoadedQwen3<B: Backend> {
    pub model: Qwen3Model<B>,
    pub eos_token_ids: Vec<i32>,
    pub report: HuggingFaceLoadReport,
}

fn norm<B: Backend, const D: usize>(
    x: Tensor<B, D>,
    weight: &Tensor<B, 1>,
    epsilon: f64,
) -> Tensor<B, D> {
    let dtype = x.dtype();
    let full = x.cast(DType::F32);
    let variance = (full.clone() * full.clone()).mean_dim(D - 1);
    (full / (variance + epsilon).sqrt()).cast(dtype) * weight.clone().unsqueeze::<D>()
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
fn rotate<B: Backend>(x: Tensor<B, 4>, cos: Tensor<B, 4>, sin: Tensor<B, 4>) -> Tensor<B, 4> {
    crate::llama::rope::apply_partial(x, cos, sin)
}

impl<B: Backend> Qwen3Model<B> {
    fn forward_block(
        &self,
        tokens: Tensor<B, 2, Int>,
        cache: &mut LlamaKvCache<B>,
        last_only: bool,
    ) -> Result<Tensor<B, 3>, GenerationError> {
        self.forward_with_features(tokens, cache, last_only, &[])
            .map(|(logits, _)| logits)
    }

    fn forward_with_features(
        &self,
        tokens: Tensor<B, 2, Int>,
        cache: &mut LlamaKvCache<B>,
        last_only: bool,
        tapped_layers: &[usize],
    ) -> Result<(Tensor<B, 3>, Option<Tensor<B, 3>>), GenerationError> {
        let [batch, sequence] = tokens.dims();
        let c = &self.config;
        let start = cache.position();
        if batch != 1
            || sequence == 0
            || cache.layers.len() != c.num_hidden_layers
            || cache.capacity() != c.max_position_embeddings
            || tapped_layers
                .iter()
                .any(|&layer| layer >= c.num_hidden_layers)
            || tapped_layers.windows(2).any(|pair| pair[0] >= pair[1])
            || start
                .checked_add(sequence)
                .is_none_or(|n| n > c.max_position_embeddings)
        {
            return Err(GenerationError(
                "Qwen3 requires compatible batch-one cache and valid context length".into(),
            ));
        }
        let device = tokens.device();
        let mut hidden = self
            .embedding
            .clone()
            .select(0, tokens.reshape([sequence]))
            .reshape([1, sequence, c.hidden_size]);
        let dtype = hidden.dtype();
        let mut features = Vec::with_capacity(tapped_layers.len());
        for (index, (layer, state)) in self.layers.iter().zip(&mut cache.layers).enumerate() {
            let rotary = self.rotary[index].for_cache(&state.rotary, start)?;
            let (cos, sin) = rotary.cos_sin((start..start + sequence).map(|p| p as f32).collect(),
                1, sequence, start + sequence, dtype, &device);
            if state.sequence_length() != start {
                return Err(GenerationError("Qwen3 layer cache positions differ".into()));
            }
            let input = norm(hidden.clone(), &layer.input_norm, c.rms_norm_eps);
            let query = norm(
                layer.q.forward(input.clone()).reshape([
                    1,
                    sequence,
                    c.num_attention_heads,
                    c.head_dim,
                ]),
                &layer.q_norm,
                c.rms_norm_eps,
            )
            .swap_dims(1, 2);
            let key = norm(
                layer.k.forward(input.clone()).reshape([
                    1,
                    sequence,
                    c.num_key_value_heads,
                    c.head_dim,
                ]),
                &layer.k_norm,
                c.rms_norm_eps,
            )
            .swap_dims(1, 2);
            let value = layer
                .v
                .forward(input)
                .reshape([1, sequence, c.num_key_value_heads, c.head_dim])
                .swap_dims(1, 2);
            let query = rotate(query, cos.clone(), sin.clone());
            let key = rotate(key, cos.clone(), sin.clone());
            let (key, value, end) = state.append_growable(key, value);
            state.rotary.commit(&rotary, end)?;
            let ranges = [0..1, 0..c.num_key_value_heads, 0..end, 0..c.head_dim];
            let key = key.slice(ranges.clone());
            let value = value.slice(ranges);
            let mut scores =
                gqa_scores(&ops::<B>(), query, key) * (1.0 / (c.head_dim as f64).sqrt());
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
            let context = gqa_value_product(&ops::<B>(), probabilities, value)
                .swap_dims(1, 2)
                .reshape([1, sequence, c.num_attention_heads * c.head_dim]);
            let attention = layer.o.forward(context);
            let attention = match &self.reduction {
                Some(reduce) => reduce(attention),
                None => attention,
            };
            hidden = hidden + attention;
            let input = norm(hidden.clone(), &layer.post_norm, c.rms_norm_eps);
            let mlp = layer.mlp.forward(input)?;
            let mlp = match &self.reduction {
                Some(reduce) => reduce(mlp),
                None => mlp,
            };
            hidden = hidden + mlp;
            if tapped_layers.contains(&index) {
                features.push(hidden.clone());
            }
        }
        cache.position = start + sequence;
        if last_only {
            hidden = hidden.slice([0..1, sequence - 1..sequence, 0..c.hidden_size]);
        }
        let features = if features.is_empty() {
            None
        } else {
            Some(Tensor::cat(features, 2))
        };
        Ok((
            self.head.forward(norm(hidden, &self.norm, c.rms_norm_eps)),
            features,
        ))
    }
}

impl<B: Backend> CausalModel<B> for Qwen3Model<B> {
    type Cache = LlamaKvCache<B>;
    fn new_cache(&self) -> Self::Cache {
        LlamaKvCache::new(
            self.config.num_hidden_layers,
            self.config.max_position_embeddings,
        )
    }
    fn forward_cached_last(
        &self,
        tokens: Tensor<B, 2, Int>,
        cache: &mut Self::Cache,
    ) -> Tensor<B, 3> {
        self.forward_block(tokens, cache, true)
            .expect("validated Qwen3 forward")
    }
    fn try_forward_cached_last(
        &self,
        tokens: Tensor<B, 2, Int>,
        cache: &mut Self::Cache,
    ) -> Result<Tensor<B, 3>, GenerationError> {
        self.forward_block(tokens, cache, true)
    }
}
impl<B: Backend> SpeculativeModel<B> for Qwen3Model<B> {
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

pub fn load_huggingface_qwen3<B: Backend>(
    directory: impl AsRef<Path>,
    device: &B::Device,
) -> Result<LoadedQwen3<B>, HuggingFaceLoadError> {
    load_rank(directory.as_ref(), device, None, None)
}

struct Shard {
    rank: usize,
    world: usize,
}

fn load_rank<B: Backend>(
    directory: &Path,
    device: &B::Device,
    shard: Option<Shard>,
    reduction: Option<Arc<dyn Fn(Tensor<B, 3>) -> Tensor<B, 3> + Send + Sync>>,
) -> Result<LoadedQwen3<B>, HuggingFaceLoadError> {
    let directory = directory
        .canonicalize()
        .map_err(|e| HuggingFaceLoadError(e.to_string()))?;
    let (mut config, mut moe_config) = read_config(&directory)?;
    config.validate()?;
    let parameters = config.rotary_parameters()?;
    let rotary = (0..config.num_hidden_layers).map(|index| parameters.for_layer(index).state()).collect();
    if let Some(s) = &shard {
        if s.world == 0
            || s.rank >= s.world
            || !config.num_attention_heads.is_multiple_of(s.world)
            || ((0..config.num_hidden_layers).any(|i| moe_config.as_ref().is_none_or(|m| !m.sparse(i)))
                && !config.intermediate_size.is_multiple_of(s.world))
            || moe_config.as_ref().is_some_and(|m| (0..config.num_hidden_layers).any(|i| m.sparse(i)) && !m.moe_intermediate_size.is_multiple_of(s.world))
            || !(config.num_key_value_heads.is_multiple_of(s.world)
                || s.world.is_multiple_of(config.num_key_value_heads))
        {
            return Err(HuggingFaceLoadError(
                "Qwen3 TP requires divisible query/FFN shards and compatible KV-head groups".into(),
            ));
        }
    }
    let mut checkpoint = Checkpoint::open(&directory)?;
    let embedding = checkpoint.tensor(
        "model.embed_tokens.weight",
        [config.vocab_size, config.hidden_size],
        device,
    )?;
    let dtype = embedding.dtype();
    let mut projection = |prefix: &str,
                          input: usize,
                          output: usize,
                          bias: bool,
                          axis: Option<(usize, std::ops::Range<usize>)>|
     -> Result<Projection<B>, HuggingFaceLoadError> {
        let weight = if let Some((dimension, range)) = &axis {
            checkpoint.tensor_slice(
                &format!("{prefix}.weight"),
                [output, input],
                *dimension,
                range.clone(),
                device,
            )?
        } else {
            checkpoint.tensor(&format!("{prefix}.weight"), [output, input], device)?
        };
        let bias = if bias {
            let bias = if let Some((0, range)) = &axis {
                checkpoint.tensor_slice(
                    &format!("{prefix}.bias"),
                    [output],
                    0,
                    range.clone(),
                    device,
                )?
            } else {
                checkpoint.tensor(&format!("{prefix}.bias"), [output], device)?
            };
            if bias.dtype() != dtype {
                return Err(HuggingFaceLoadError(
                    "Qwen3 projection bias storage dtype differs".into(),
                ));
            }
            if axis.as_ref().is_some_and(|(axis, _)| *axis == 1)
                && shard.as_ref().is_some_and(|s| s.rank != 0)
            {
                None
            } else {
                Some(bias)
            }
        } else {
            None
        };
        if weight.dtype() != dtype || bias.as_ref().is_some_and(|b| b.dtype() != dtype) {
            return Err(HuggingFaceLoadError(
                "Qwen3 projection storage dtypes differ".into(),
            ));
        }
        Ok(Projection { weight, bias })
    };
    let mut layers = Vec::with_capacity(config.num_hidden_layers);
    for i in 0..config.num_hidden_layers {
        let p = format!("model.layers.{i}");
        let query = shard.as_ref().map(|s| {
            let width = config.num_attention_heads * config.head_dim / s.world;
            s.rank * width..(s.rank + 1) * width
        });
        let kv = shard.as_ref().map(|s| {
            if s.world <= config.num_key_value_heads {
                let width = config.num_key_value_heads * config.head_dim / s.world;
                s.rank * width..(s.rank + 1) * width
            } else {
                let head = s.rank / (s.world / config.num_key_value_heads);
                head * config.head_dim..(head + 1) * config.head_dim
            }
        });
        let q = projection(
            &format!("{p}.self_attn.q_proj"),
            config.hidden_size,
            config.num_attention_heads * config.head_dim,
            config.attention_bias,
            query.clone().map(|range| (0, range)),
        )?;
        let k = projection(
            &format!("{p}.self_attn.k_proj"),
            config.hidden_size,
            config.num_key_value_heads * config.head_dim,
            config.attention_bias,
            kv.clone().map(|range| (0, range)),
        )?;
        let v = projection(
            &format!("{p}.self_attn.v_proj"),
            config.hidden_size,
            config.num_key_value_heads * config.head_dim,
            config.attention_bias,
            kv.map(|range| (0, range)),
        )?;
        let o = projection(
            &format!("{p}.self_attn.o_proj"),
            config.num_attention_heads * config.head_dim,
            config.hidden_size,
            config.attention_bias,
            query.map(|range| (1, range)),
        )?;
        let mlp = moe::load(&format!("{p}.mlp"), &config, moe_config.as_ref(), i, shard.as_ref(), &mut projection)?;
        layers.push((q, k, v, o, mlp, p));
    }
    drop(projection);
    let layers = layers
        .into_iter()
        .map(|(q, k, v, o, mlp, p)| {
            Ok(Layer {
                q,
                k,
                v,
                o,
                mlp,
                input_norm: checkpoint.tensor(
                    &format!("{p}.input_layernorm.weight"),
                    [config.hidden_size],
                    device,
                )?,
                post_norm: checkpoint.tensor(
                    &format!("{p}.post_attention_layernorm.weight"),
                    [config.hidden_size],
                    device,
                )?,
                q_norm: checkpoint.tensor(
                    &format!("{p}.self_attn.q_norm.weight"),
                    [config.head_dim],
                    device,
                )?,
                k_norm: checkpoint.tensor(
                    &format!("{p}.self_attn.k_norm.weight"),
                    [config.head_dim],
                    device,
                )?,
            })
        })
        .collect::<Result<Vec<_>, HuggingFaceLoadError>>()?;
    let norm = checkpoint.tensor("model.norm.weight", [config.hidden_size], device)?;
    let head = if config.tie_word_embeddings {
        if checkpoint.tensors.contains_key("lm_head.weight") {
            let supplied = checkpoint.tensor::<B, 2>(
                "lm_head.weight",
                [config.vocab_size, config.hidden_size],
                device,
            )?;
            let actual = embedding
                .clone()
                .try_into_data()
                .map_err(|e| HuggingFaceLoadError(e.to_string()))?;
            let supplied = supplied
                .try_into_data()
                .map_err(|e| HuggingFaceLoadError(e.to_string()))?;
            if actual.dtype != supplied.dtype || actual.as_bytes() != supplied.as_bytes() {
                return Err(HuggingFaceLoadError(
                    "tied Qwen3 head differs from embedding weights".into(),
                ));
            }
        }
        Projection {
            weight: embedding.clone(),
            bias: None,
        }
    } else {
        Projection {
            weight: checkpoint.tensor(
                "lm_head.weight",
                [config.vocab_size, config.hidden_size],
                device,
            )?,
            bias: None,
        }
    };
    if norm.dtype() != dtype
        || head.weight.dtype() != dtype
        || layers.iter().any(|layer| {
            [
                &layer.input_norm,
                &layer.post_norm,
                &layer.q_norm,
                &layer.k_norm,
            ]
            .iter()
            .any(|weight| weight.dtype() != dtype)
        })
    {
        return Err(HuggingFaceLoadError(
            "Qwen3 normalization and projection storage dtypes differ".into(),
        ));
    }
    let unused = checkpoint
        .tensors
        .keys()
        .filter(|name| !checkpoint.consumed.contains(*name))
        .cloned()
        .collect::<Vec<_>>();
    if !unused.is_empty() {
        return Err(HuggingFaceLoadError(format!(
            "unrepresented Qwen3 tensors: {}",
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
                .ok_or_else(|| HuggingFaceLoadError("invalid Qwen3 EOS ID".into()))
        })
        .collect::<Result<Vec<_>, _>>()?;
    B::sync(device).map_err(|e| HuggingFaceLoadError(e.to_string()))?;
    let report = HuggingFaceLoadReport {
        config_path: directory.join("config.json"),
        weight_files: checkpoint.files,
        applied_tensors: checkpoint.consumed.len(),
        tied_word_embeddings: config.tie_word_embeddings,
    };
    if let Some(s) = shard {
        config.num_attention_heads /= s.world;
        config.num_key_value_heads = (config.num_key_value_heads / s.world).max(1);
        if (0..config.num_hidden_layers).any(|i| moe_config.as_ref().is_none_or(|m| !m.sparse(i))) {
            config.intermediate_size /= s.world;
        }
        if let Some(moe) = &mut moe_config {
            if (0..config.num_hidden_layers).any(|i| moe.sparse(i)) { moe.moe_intermediate_size /= s.world; }
        }
    }
    Ok(LoadedQwen3 {
        model: Qwen3Model {
            config,
            moe_config,
            embedding,
            layers,
            norm,
            head,
            rotary,
            reduction,
        },
        eos_token_ids,
        report,
    })
}
