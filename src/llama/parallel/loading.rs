use super::*;
use crate::huggingface::checkpoint::Checkpoint;
use ruda_nn::{Embedding, Linear, RmsNorm};
use std::ops::Range;

pub(super) fn load_rank<B: Backend>(
    directory: &Path,
    config: &LlamaConfig,
    rotary: &super::super::rope::RopeParameters,
    tied: bool,
    qwen2: bool,
    rank: usize,
    world: usize,
    device: &B::Device,
    reduction: Reduction<B>,
) -> Result<(LlamaTensorParallelRank<B>, HuggingFaceLoadReport), HuggingFaceLoadError> {
    let mut checkpoint = Checkpoint::open(directory)?;
    let embedding = checkpoint.tensor::<B, 2>(
        "model.embed_tokens.weight", [config.vocab_size, config.d_model], device,
    )?;
    let dtype = embedding.dtype();
    let width = config.head_dimension();
    let queries = config.num_query_heads / world;
    let kv_heads = (config.num_kv_heads / world).max(1);
    let query_range = rank * queries * width..(rank + 1) * queries * width;
    let kv_range = if world <= config.num_kv_heads {
        rank * kv_heads * width..(rank + 1) * kv_heads * width
    } else {
        let head = rank / (world / config.num_kv_heads);
        head * width..(head + 1) * width
    };
    let ff = config.d_ff / world;
    let ff_range = rank * ff..(rank + 1) * ff;
    let tables = rotary.tables(config, device, dtype);
    let mut layers = Vec::with_capacity(config.num_hidden_layers);
    for index in 0..config.num_hidden_layers {
        let prefix = format!("model.layers.{index}");
        let mut projection = |name: &str, input: usize, output: usize, bias: bool,
                              axis: usize, range: Range<usize>|
         -> Result<Linear<B>, HuggingFaceLoadError> {
            let name = format!("{prefix}.{name}");
            let weight = checkpoint.tensor_slice::<B, 2>(
                &format!("{name}.weight"), [output, input], axis, range.clone(), device,
            )?;
            if weight.dtype() != dtype {
                return Err(HuggingFaceLoadError(format!("{name}: TP projection storage dtype differs")));
            }
            let bias = if bias {
                let bias = checkpoint.tensor_slice::<B, 1>(
                    &format!("{name}.bias"), [output], 0, range, device,
                )?;
                if bias.dtype() != dtype {
                    return Err(HuggingFaceLoadError(format!("{name}: TP bias storage dtype differs")));
                }
                Some(Param::from_tensor(bias))
            } else { None };
            Ok(Linear { weight: Param::from_tensor(weight.transpose()), bias })
        };
        let q_proj = projection("self_attn.q_proj", config.d_model, config.d_model, qwen2, 0, query_range.clone())?;
        let k_proj = projection("self_attn.k_proj", config.d_model, config.num_kv_heads * width, qwen2, 0, kv_range.clone())?;
        let v_proj = projection("self_attn.v_proj", config.d_model, config.num_kv_heads * width, qwen2, 0, kv_range.clone())?;
        let o_proj = projection("self_attn.o_proj", config.d_model, config.d_model, false, 1, query_range.clone())?;
        let gate_proj = projection("mlp.gate_proj", config.d_model, config.d_ff, false, 0, ff_range.clone())?;
        let up_proj = projection("mlp.up_proj", config.d_model, config.d_ff, false, 0, ff_range.clone())?;
        let down_proj = projection("mlp.down_proj", config.d_ff, config.d_model, false, 1, ff_range.clone())?;
        drop(projection);
        let input_norm = checkpoint.tensor::<B, 1>(&format!("{prefix}.input_layernorm.weight"), [config.d_model], device)?;
        let post_norm = checkpoint.tensor::<B, 1>(&format!("{prefix}.post_attention_layernorm.weight"), [config.d_model], device)?;
        layers.push(LlamaDecoderLayer {
            self_attn: LlamaAttention {
                q_proj, k_proj, v_proj, o_proj, rope: tables[index].clone(),
                rotary_state: rotary.for_layer(index).runtime_state(),
                rotary_layout: RotaryLayout::HalfSplit,
                num_query_heads: queries, num_kv_heads: kv_heads, head_dimension: width,
            },
            mlp: LlamaFeedForward { gate_proj, up_proj, down_proj },
            input_layernorm: RmsNorm { gamma: Param::from_tensor(input_norm), epsilon: config.rms_norm_epsilon },
            post_attention_layernorm: RmsNorm { gamma: Param::from_tensor(post_norm), epsilon: config.rms_norm_epsilon },
        });
    }
    let norm = checkpoint.tensor::<B, 1>("model.norm.weight", [config.d_model], device)?;
    let head = if tied {
        if checkpoint.tensors.contains_key("lm_head.weight") {
            let snapshot = &checkpoint.tensors["lm_head.weight"];
            if snapshot.shape != [config.vocab_size, config.d_model].into()
                || snapshot.dtype != dtype {
                return Err(HuggingFaceLoadError("tied TP LM head geometry/dtype differs".into()));
            }
            checkpoint.consumed.insert("lm_head.weight".into());
        }
        embedding.clone()
    } else {
        let head = checkpoint.tensor::<B, 2>("lm_head.weight", [config.vocab_size, config.d_model], device)?;
        if head.dtype() != dtype {
            return Err(HuggingFaceLoadError("TP LM head storage dtype differs".into()));
        }
        head
    };
    let unused = checkpoint.tensors.keys().filter(|name| !checkpoint.consumed.contains(*name))
        .cloned().collect::<Vec<_>>();
    if !unused.is_empty() {
        return Err(HuggingFaceLoadError(format!("unrepresented Llama/Qwen2 TP tensors: {}", unused.join(", "))));
    }
    B::sync(device).map_err(|e| HuggingFaceLoadError(e.to_string()))?;
    let report = HuggingFaceLoadReport {
        config_path: directory.join("config.json"), weight_files: checkpoint.files,
        applied_tensors: checkpoint.consumed.len(), tied_word_embeddings: tied,
    };
    let model = LlamaForCausalLm {
        embed_tokens: Embedding { weight: Param::from_tensor(embedding) }, layers,
        norm: RmsNorm { gamma: Param::from_tensor(norm), epsilon: config.rms_norm_epsilon },
        lm_head: Linear { weight: Param::from_tensor(head.transpose()), bias: None },
        max_sequence_length: config.max_sequence_length,
    };
    Ok((LlamaTensorParallelRank { model, reduction }, report))
}
