use super::*;
use super::weights::{Part, PackedPartLoader, Weights, floating_data};
use crate::huggingface::{AwqQuantizationConfig, checkpoint::Checkpoint};
use ruda_model::module::Param;
use std::ops::Range;

fn shard(heads: usize, width: usize, rank: usize, world: usize) -> Range<usize> {
    if world <= heads {
        let size = heads / world * width;
        rank * size..(rank + 1) * size
    } else {
        let head = rank / (world / heads);
        head * width..(head + 1) * width
    }
}

pub(super) fn validate(config: &Qwen35TextConfig, world: usize) -> Result<(), HuggingFaceLoadError> {
    let compatible = |heads: usize| heads.is_multiple_of(world) || world.is_multiple_of(heads);
    if world == 0 || (if config.is_moe() {
        !config.moe.moe_intermediate_size.is_multiple_of(world) || !config.moe.shared_expert_intermediate_size.is_multiple_of(world)
    } else { !config.intermediate_size.is_multiple_of(world) })
        || !config.num_attention_heads.is_multiple_of(world) || !compatible(config.num_key_value_heads)
        || !config.linear_num_value_heads.is_multiple_of(world) || !compatible(config.linear_num_key_heads) {
        return Err(HuggingFaceLoadError("Qwen3.5 TP requires divisible query/value/FFN heads and compatible full/recurrent key-head groups".into()));
    }
    Ok(())
}

pub(super) fn load_rank<B: Backend>(directory: &Path, source: &Qwen35TextConfig,
    quantization: Option<AwqQuantizationConfig>, multimodal: bool, rank: usize, world: usize,
    device: &B::Device, reduction: Reduction<B>, packed: Option<PackedPartLoader<B>>,
    fp8_loader: Option<super::super::fp8_loading::Loader<B>>)
    -> Result<super::super::loading::LoadedQwen35Text<B>, HuggingFaceLoadError> {
    let (mut checkpoint, fp8) = super::super::fp8_loading::open(directory, fp8_loader)?;
    let base = if multimodal { "model.language_model" } else { "model" };
    let embedding_name = format!("{base}.embed_tokens.weight");
    let embedding = checkpoint.tensor::<B, 2>(&embedding_name, [source.vocab_size, source.hidden_size], device)?;
    let dtype = embedding.dtype();
    let mut w = Weights { checkpoint: &mut checkpoint, device, quantization, dtype, packed, fp8, dense_awq: 0, packed_awq: 0 };
    let c = source;
    let query = shard(c.num_attention_heads, c.head_dim, rank, world);
    let qgate = shard(c.num_attention_heads, 2 * c.head_dim, rank, world);
    let kv = shard(c.num_key_value_heads, c.head_dim, rank, world);
    let keys = c.linear_num_key_heads * c.linear_key_head_dim;
    let values = c.linear_num_value_heads * c.linear_value_head_dim;
    let linear_key = shard(c.linear_num_key_heads, c.linear_key_head_dim, rank, world);
    let linear_value = shard(c.linear_num_value_heads, c.linear_value_head_dim, rank, world);
    let value_head = shard(c.linear_num_value_heads, 1, rank, world);
    let channels = vec![linear_key.clone(), keys + linear_key.start..keys + linear_key.end,
        2 * keys + linear_value.start..2 * keys + linear_value.end];
    let column = |ranges: Vec<Range<usize>>| Part { input: 0..c.hidden_size, output: ranges, bias: true };
    let row = |range: Range<usize>| Part { input: range, output: vec![0..c.hidden_size], bias: rank == 0 };
    let mut layers = Vec::with_capacity(c.num_hidden_layers);
    for (index, kind) in c.layer_types.iter().enumerate() {
        let prefix = format!("{base}.layers.{index}");
        let mixer = match kind {
            LayerType::FullAttention => {
                let p = format!("{prefix}.self_attn");
                let q = c.num_attention_heads * c.head_dim;
                let k = c.num_key_value_heads * c.head_dim;
                Mixer::Full(attention::Attention {
                    q: w.projection(&format!("{p}.q_proj"), c.hidden_size, 2 * q, c.attention_bias, column(vec![qgate.clone()]))?,
                    k: w.projection(&format!("{p}.k_proj"), c.hidden_size, k, c.attention_bias, column(vec![kv.clone()]))?,
                    v: w.projection(&format!("{p}.v_proj"), c.hidden_size, k, c.attention_bias, column(vec![kv.clone()]))?,
                    out: w.projection(&format!("{p}.o_proj"), q, c.hidden_size, c.attention_bias, row(query.clone()))?,
                    q_norm: w.norm(&format!("{p}.q_norm"), c.head_dim, c.rms_norm_eps)?,
                    k_norm: w.norm(&format!("{p}.k_norm"), c.head_dim, c.rms_norm_eps)?,
                })
            },
            LayerType::LinearAttention => {
                let p = format!("{prefix}.linear_attn");
                Mixer::Delta(delta::Delta {
                    qkv: w.projection(&format!("{p}.in_proj_qkv"), c.hidden_size, 2 * keys + values, false, column(channels.clone()))?,
                    z: w.projection(&format!("{p}.in_proj_z"), c.hidden_size, values, false, column(vec![linear_value.clone()]))?,
                    a: w.projection(&format!("{p}.in_proj_a"), c.hidden_size, c.linear_num_value_heads, false, column(vec![value_head.clone()]))?,
                    b: w.projection(&format!("{p}.in_proj_b"), c.hidden_size, c.linear_num_value_heads, false, column(vec![value_head.clone()]))?,
                    out: w.projection(&format!("{p}.out_proj"), values, c.hidden_size, false, row(linear_value.clone()))?,
                    conv: w.rows(&format!("{p}.conv1d.weight"), [2 * keys + values, 1, c.linear_conv_kernel_dim], &channels)?,
                    a_log: w.rows(&format!("{p}.A_log"), [c.linear_num_value_heads], &[value_head.clone()])?,
                    dt_bias: w.rows(&format!("{p}.dt_bias"), [c.linear_num_value_heads], &[value_head.clone()])?,
                    norm: w.tensor(&format!("{p}.norm.weight"), [c.linear_value_head_dim])?,
                })
            },
        };
        layers.push(Layer {
            mixer,
            input_norm: w.norm(&format!("{prefix}.input_layernorm"), c.hidden_size, c.rms_norm_eps)?,
            post_norm: w.norm(&format!("{prefix}.post_attention_layernorm"), c.hidden_size, c.rms_norm_eps)?,
            mlp: super::super::moe::load(&mut w, &format!("{prefix}.mlp"), c, Some((rank, world)))?,
        });
    }
    let norm = w.norm(&format!("{base}.norm"), c.hidden_size, c.rms_norm_eps)?;
    let head = if c.tie_word_embeddings {
        if w.checkpoint.tensors.contains_key("lm_head.weight") {
            let supplied = floating_data(w.checkpoint, "lm_head.weight", [c.vocab_size, c.hidden_size])?;
            let original = floating_data(w.checkpoint, &embedding_name, [c.vocab_size, c.hidden_size])?;
            if supplied != original { return Err(HuggingFaceLoadError("tied Qwen3.5 TP head differs from embedding".into())); }
            w.checkpoint.consumed.insert("lm_head.weight".into());
        }
        Projection::Dense(Linear { weight: Param::from_tensor(embedding.clone().transpose()), bias: None })
    } else { w.projection("lm_head", c.hidden_size, c.vocab_size, false, Part::full(c.hidden_size, c.vocab_size))? };
    let dequantized_awq_linears = w.dense_awq;
    let packed_awq_linears = w.packed_awq;
    let packed_fp8_linears = w.fp8.as_ref().map_or(0, |weights| weights.projections);
    drop(w);
    let unloaded_tensors = checkpoint.tensors.keys().filter(|name| !checkpoint.consumed.contains(*name)).cloned().collect::<Vec<_>>();
    if unloaded_tensors.iter().any(|name| !(name.starts_with("mtp.") || (multimodal && name.starts_with("model.visual.")))) {
        return Err(HuggingFaceLoadError(format!("unconsumed Qwen3.5 TP tensors: {unloaded_tensors:?}")));
    }
    let mut config = source.clone();
    config.num_attention_heads /= world;
    config.num_key_value_heads = (config.num_key_value_heads / world).max(1);
    config.linear_num_value_heads /= world;
    config.linear_num_key_heads = (config.linear_num_key_heads / world).max(1);
    if config.is_moe() {
        config.moe.moe_intermediate_size /= world;
        config.moe.shared_expert_intermediate_size /= world;
    } else { config.intermediate_size /= world; }
    config.validate()?;
    B::sync(device).map_err(|e| HuggingFaceLoadError(e.to_string()))?;
    Ok(super::super::loading::LoadedQwen35Text {
        model: Qwen35TextModel { rotary: config.rotary_parameters()?, config, weight_prefix: base.into(),
            embedding: Embedding { weight: Param::from_tensor(embedding) }, layers, norm, head, reduction: Some(reduction) },
        report: HuggingFaceLoadReport { config_path: directory.join("config.json"), weight_files: checkpoint.files,
            applied_tensors: checkpoint.consumed.len(), tied_word_embeddings: source.tie_word_embeddings },
        unloaded_tensors, dequantized_awq_linears, packed_awq_linears, packed_fp8_linears,
    })
}
