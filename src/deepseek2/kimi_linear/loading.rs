use super::*;
use std::collections::BTreeSet;

pub struct LoadedKimiLinear<B: Backend> {
    pub model: KimiLinearModel<B>, pub eos_token_ids: Vec<i32>, pub report: HuggingFaceLoadReport,
}
pub fn load_huggingface_kimi_linear<B: super::super::MlaFp8Backend>(directory: impl AsRef<Path>, device: &B::Device)
    -> Result<LoadedKimiLinear<B>, HuggingFaceLoadError> {
    load_rank(directory.as_ref(), device, None, None)
}

pub(crate) fn load_sft<B: Backend>(directory: &Path, device: &B::Device)
    -> Result<crate::training::NativeSftModel<B>, HuggingFaceLoadError> {
    load_rank_with(directory, device, None, None, None)?.model.into_training()
        .map_err(|e| HuggingFaceLoadError(e.to_string()))
}
fn tensor<B: Backend, const D: usize>(checkpoint: &mut weights::Weights<B>, name: &str,
    shape: [usize; D], device: &B::Device, dtype: DType) -> Result<Tensor<B, D>, HuggingFaceLoadError> {
    let tensor = checkpoint.tensor(name, shape, device)?;
    if tensor.dtype() != dtype { return Err(HuggingFaceLoadError(format!("Kimi Linear storage dtype differs: {name}"))); }
    Ok(tensor)
}
pub(super) fn load_rank<B: super::super::MlaFp8Backend>(directory: &Path, device: &B::Device,
    shard: Option<(usize, usize)>, reduction: Option<Arc<dyn Fn(Tensor<B, 3>) -> Tensor<B, 3> + Send + Sync>>)
    -> Result<LoadedKimiLinear<B>, HuggingFaceLoadError> {
    load_rank_with(directory, device, shard, reduction, Some(weights::native_projection::<B>))
}

fn load_rank_with<B: Backend>(directory: &Path, device: &B::Device,
    shard: Option<(usize, usize)>, reduction: Option<Arc<dyn Fn(Tensor<B, 3>) -> Tensor<B, 3> + Send + Sync>>,
    loader: Option<weights::Loader<B>>)
    -> Result<LoadedKimiLinear<B>, HuggingFaceLoadError> {
    let fail = |e: std::io::Error| HuggingFaceLoadError(e.to_string());
    let directory = directory.canonicalize().map_err(fail)?;
    let config: KimiLinearConfig = serde_json::from_slice(&std::fs::read(directory.join("config.json")).map_err(fail)?)
        .map_err(|e| HuggingFaceLoadError(e.to_string()))?;
    config.validate()?;
    if shard.is_some_and(|(rank, world)| world == 0 || rank >= world
        || config.num_experts.is_none_or(|n| n < world || n % world != 0)) {
        return Err(HuggingFaceLoadError("Kimi Linear EP requires equal nonempty expert shards".into()));
    }
    let routing = config.routing();
    let mut checkpoint = weights::Weights::open(&directory, &routing, loader)?;
    let embedding = checkpoint.tensor("model.embed_tokens.weight", [config.vocab_size, config.hidden_size], device)?;
    let dtype = embedding.dtype(); let c = &config;
    let mut excluded = BTreeSet::new(); let mut layers = Vec::with_capacity(c.num_hidden_layers);
    for index in 0..c.num_hidden_layers {
        let p = format!("model.layers.{index}"); let a = format!("{p}.self_attn");
        let mixer = if c.is_kda_layer(index) {
            let linear = c.linear_attn_config.as_ref().unwrap();
            let (h, d, w) = (linear.num_heads, linear.head_dim, linear.short_conv_kernel_size); let channels = h * d;
            let q = projection(&mut checkpoint, &format!("{a}.q_proj"), c.hidden_size, channels, false, device, dtype)?;
            let k = projection(&mut checkpoint, &format!("{a}.k_proj"), c.hidden_size, channels, false, device, dtype)?;
            let v = projection(&mut checkpoint, &format!("{a}.v_proj"), c.hidden_size, channels, false, device, dtype)?;
            let q_conv = tensor(&mut checkpoint, &format!("{a}.q_conv1d.weight"), [channels, 1, w], device, dtype)?;
            let k_conv = tensor(&mut checkpoint, &format!("{a}.k_conv1d.weight"), [channels, 1, w], device, dtype)?;
            let v_conv = tensor(&mut checkpoint, &format!("{a}.v_conv1d.weight"), [channels, 1, w], device, dtype)?;
            let a_log = checkpoint.tensor(&format!("{a}.A_log"), [1, 1, h, 1], device)?;
            let dt_bias = checkpoint.tensor(&format!("{a}.dt_bias"), [channels], device)?;
            let f_a = projection(&mut checkpoint, &format!("{a}.f_a_proj"), c.hidden_size, d, false, device, dtype)?;
            let f_b = projection(&mut checkpoint, &format!("{a}.f_b_proj"), d, channels, false, device, dtype)?;
            let beta = projection(&mut checkpoint, &format!("{a}.b_proj"), c.hidden_size, h, false, device, dtype)?;
            let g_a = projection(&mut checkpoint, &format!("{a}.g_a_proj"), c.hidden_size, d, false, device, dtype)?;
            let g_b = projection(&mut checkpoint, &format!("{a}.g_b_proj"), d, channels, false, device, dtype)?;
            let output_norm = tensor(&mut checkpoint, &format!("{a}.o_norm.weight"), [d], device, dtype)?;
            let out = projection(&mut checkpoint, &format!("{a}.o_proj"), channels, c.hidden_size, false, device, dtype)?;
            Mixer::Kda(kda::Kda {q, k, v, q_conv, k_conv, v_conv, a_log, dt_bias, f_a, f_b, beta, g_a, g_b, output_norm, out})
        } else {
            let q = projection(&mut checkpoint, &format!("{a}.q_proj"), c.hidden_size,
                c.num_attention_heads * (c.qk_nope_head_dim + c.qk_rope_head_dim), false, device, dtype)?;
            let kv_a = projection(&mut checkpoint, &format!("{a}.kv_a_proj_with_mqa"), c.hidden_size,
                c.kv_lora_rank + c.qk_rope_head_dim, false, device, dtype)?;
            let kv_norm = tensor(&mut checkpoint, &format!("{a}.kv_a_layernorm.weight"), [c.kv_lora_rank], device, dtype)?;
            let kv_b = projection(&mut checkpoint, &format!("{a}.kv_b_proj"), c.kv_lora_rank,
                c.num_attention_heads * (c.qk_nope_head_dim + c.v_head_dim), false, device, dtype)?;
            let out = projection(&mut checkpoint, &format!("{a}.o_proj"), c.num_attention_heads * c.v_head_dim, c.hidden_size, false, device, dtype)?;
            Mixer::Mla(Mla {q, kv_a, kv_norm, kv_b, out})
        };
        let mlp = if c.is_moe(index) {
            let p = format!("{p}.block_sparse_moe"); let count = c.num_experts.unwrap(); let width = c.moe_intermediate_size.unwrap();
            let router = checkpoint.tensor(&format!("{p}.gate.weight"), [count, c.hidden_size], device)?;
            let correction = Some(checkpoint.tensor(&format!("{p}.gate.e_score_correction_bias"), [count], device)?);
            let mut experts = Vec::new();
            for expert in 0..count {
                let p = format!("{p}.experts.{expert}");
                if shard.is_none_or(|(rank, world)| expert / (count / world) == rank) {
                    experts.push((expert, Mlp {
                        gate: projection(&mut checkpoint, &format!("{p}.w1"), c.hidden_size, width, false, device, dtype)?,
                        up: projection(&mut checkpoint, &format!("{p}.w3"), c.hidden_size, width, false, device, dtype)?,
                        down: projection(&mut checkpoint, &format!("{p}.w2"), width, c.hidden_size, false, device, dtype)?,
                    }));
                } else {
                    for (name, shape) in [("w1", [width, c.hidden_size]), ("w3", [width, c.hidden_size]), ("w2", [c.hidden_size, width])] {
                        checkpoint.exclude_projection(&format!("{p}.{name}"), shape, dtype, &mut excluded)?;
                    }
                }
            }
            let shared = if c.num_shared_experts > 0 {
                Some(load_mlp(&mut checkpoint, &format!("{p}.shared_experts"), c.hidden_size, width * c.num_shared_experts, device, dtype)?)
            } else { None };
            FeedForward::Experts {router, correction, experts, shared}
        } else { FeedForward::Dense(load_mlp(&mut checkpoint, &format!("{p}.mlp"), c.hidden_size, c.intermediate_size, device, dtype)?) };
        let input_norm = tensor(&mut checkpoint, &format!("{p}.input_layernorm.weight"), [c.hidden_size], device, dtype)?;
        let post_norm = tensor(&mut checkpoint, &format!("{p}.post_attention_layernorm.weight"), [c.hidden_size], device, dtype)?;
        layers.push(Layer {mixer, input_norm, post_norm, mlp});
    }
    let norm = tensor(&mut checkpoint, "model.norm.weight", [c.hidden_size], device, dtype)?;
    let head = if c.tie_word_embeddings {
        if checkpoint.tensors.contains_key("lm_head.weight") {
            let supplied = tensor(&mut checkpoint, "lm_head.weight", [c.vocab_size, c.hidden_size], device, dtype)?;
            if supplied.try_into_data().map_err(|e| HuggingFaceLoadError(e.to_string()))?
                != embedding.clone().try_into_data().map_err(|e| HuggingFaceLoadError(e.to_string()))? {
                return Err(HuggingFaceLoadError("Kimi Linear tied lm_head differs from embedding".into()));
            }
        }
        Projection::Dense {weight: embedding.clone(), bias: None}
    } else { projection(&mut checkpoint, "lm_head", c.hidden_size, c.vocab_size, false, device, dtype)? };
    let unused = checkpoint.tensors.keys().filter(|n| !checkpoint.consumed.contains(*n) && !excluded.contains(*n)).cloned().collect::<Vec<_>>();
    if !unused.is_empty() { return Err(HuggingFaceLoadError(format!("unrepresented Kimi Linear weights: {}", unused.join(", ")))); }
    let generation_path = directory.join("generation_config.json");
    let generation = if generation_path.is_file() { Some(serde_json::from_slice::<serde_json::Value>(&std::fs::read(generation_path).map_err(fail)?)
        .map_err(|e| HuggingFaceLoadError(e.to_string()))?) } else { None };
    let eos = generation.as_ref().and_then(|v| v.get("eos_token_id")).filter(|v| !v.is_null()).cloned()
        .or_else(|| c.eos_token_id.clone()).unwrap_or(serde_json::Value::Null);
    let eos = match eos {serde_json::Value::Null => vec![], serde_json::Value::Array(v) => v, v => vec![v]};
    let eos_token_ids = eos.into_iter().map(|v| v.as_i64().and_then(|id| i32::try_from(id).ok())
        .filter(|&id| id >= 0 && (id as usize) < c.vocab_size).ok_or_else(|| HuggingFaceLoadError("invalid Kimi Linear EOS ID".into())))
        .collect::<Result<Vec<_>, _>>()?;
    B::sync(device).map_err(|e| HuggingFaceLoadError(e.to_string()))?;
    let report = HuggingFaceLoadReport {config_path: directory.join("config.json"), weight_files: checkpoint.files.clone(),
        applied_tensors: checkpoint.consumed.len(), tied_word_embeddings: c.tie_word_embeddings};
    Ok(LoadedKimiLinear {model: KimiLinearModel {config, embedding, layers, norm, head, routing, reduction}, eos_token_ids, report})
}
