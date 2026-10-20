use super::*;
use rublas::tensor_ggml::{GgmlError, GgmlGemm};
use ruda_store::gguf::{GgmlType, GgufEndian, GgufModel, GgufValue};
use ruda_tensor::TensorPrimitive;
use std::{collections::{BTreeMap, BTreeSet}, path::Path};

#[cfg(all(test, feature = "nvidia"))]
mod tests;

pub type GgufBackend<R> = DeviceBackend<R, f32, i32, u32>;

#[derive(Clone, Debug)]
pub struct GgufLoadError(pub String);
impl Display for GgufLoadError {
    fn fmt(&self, f: &mut Formatter<'_>) -> std::fmt::Result { f.write_str(&self.0) }
}
impl Error for GgufLoadError {}
impl From<std::io::Error> for GgufLoadError {
    fn from(error: std::io::Error) -> Self { Self(error.to_string()) }
}
impl From<GgmlError> for GgufLoadError {
    fn from(error: GgmlError) -> Self { Self(error.to_string()) }
}

pub struct LoadedGguf<R: DeviceRuntime> where R::Device: DeviceOps {
    pub model: GgufCausalLm<R>,
    pub metadata: BTreeMap<String, GgufValue>,
}

#[derive(Debug)]
struct Projection<R: DeviceRuntime> {
    weight: GgmlGemm<R>,
    bias: Option<GgmlGemm<R>>,
}
impl<R: DeviceRuntime> Projection<R> where R::Device: DeviceOps {
    fn forward(&self, input: Tensor<GgufBackend<R>, 3>) -> Result<Tensor<GgufBackend<R>, 3>, GgufLoadError> {
        let mut output = self.weight.forward(input.into_primitive().tensor())?;
        if let Some(bias) = &self.bias { output = bias.apply_vector(output, false)?; }
        Ok(Tensor::from_primitive(TensorPrimitive::Float(output)))
    }
}

#[derive(Debug)]
struct Layer<R: DeviceRuntime> {
    q: Projection<R>, k: Projection<R>, v: Projection<R>, o: Projection<R>,
    feed_forward: FeedForward<R>,
    input_norm: GgmlGemm<R>, post_norm: GgmlGemm<R>,
    q_norm: Option<GgmlGemm<R>>, k_norm: Option<GgmlGemm<R>>,
}

#[derive(Debug)]
struct DenseFeedForward<R: DeviceRuntime> {
    gate: Projection<R>, up: Projection<R>, down: Projection<R>,
}
impl<R: DeviceRuntime> DenseFeedForward<R> where R::Device: DeviceOps {
    fn forward(&self, input: Tensor<GgufBackend<R>, 3>) -> Result<Tensor<GgufBackend<R>, 3>, GgufLoadError> {
        self.down.forward(silu(self.gate.forward(input.clone())?) * self.up.forward(input)?)
    }
}

#[derive(Debug)]
enum FeedForward<R: DeviceRuntime> {
    Dense(DenseFeedForward<R>),
    Experts {
        router: Projection<R>,
        gate: Option<GgmlGemm<R>>,
        up: GgmlGemm<R>,
        down: GgmlGemm<R>,
        count: usize,
        selected: usize,
        normalize: bool,
        scale: f64,
        shared: Option<DenseFeedForward<R>>,
        shared_gate: Option<GgmlGemm<R>>,
    },
}
impl<R: DeviceRuntime> FeedForward<R> where R::Device: DeviceOps {
    fn forward(&self, input: Tensor<GgufBackend<R>, 3>) -> Result<Tensor<GgufBackend<R>, 3>, GgufLoadError> {
        match self {
            Self::Dense(dense) => dense.forward(input),
            Self::Experts { router, gate, up, down, count, selected, normalize, scale, shared, shared_gate } => {
                let [batch, sequence, hidden] = input.dims();
                let tokens = batch * sequence;
                let probabilities = ruda_tensor::activation::softmax(router.forward(input.clone())?, 2)
                    .reshape([tokens, *count]);
                let (mut weights, indices) = probabilities.topk_with_indices(*selected, 1);
                if *normalize {
                    let sum = weights.clone().sum_dim(1).clamp_min(6.103515625e-5);
                    weights = weights / sum;
                }
                if *scale != 0.0 && *scale != 1.0 { weights = weights * *scale; }
                let expert_input = input.clone().reshape([tokens, 1, hidden]);
                let project = |matrix: &GgmlGemm<R>, value: Tensor<GgufBackend<R>, 3>| {
                    matrix.forward_selected(value.into_primitive().tensor(), indices.clone().into_primitive(), *count)
                        .map(|output| Tensor::<GgufBackend<R>, 3>::from_primitive(TensorPrimitive::Float(output)))
                };
                let up = project(up, expert_input.clone())?;
                let activated = match gate {
                    Some(gate) => silu(project(gate, expert_input)?) * up,
                    None => silu(up),
                };
                let values = project(down, activated)?;
                let mut output = (values * weights.reshape([tokens, *selected, 1]))
                    .sum_dim(1).reshape([batch, sequence, hidden]);
                if let Some(shared) = shared {
                    let mut value = shared.forward(input.clone())?;
                    if let Some(gate) = shared_gate {
                        let gate = Tensor::<GgufBackend<R>, 3>::from_primitive(TensorPrimitive::Float(
                            gate.forward(input.into_primitive().tensor())?,
                        ));
                        value = value * ruda_tensor::activation::sigmoid(gate);
                    }
                    output = output + value;
                }
                Ok(output)
            }
        }
    }
}

/// GGUF decoder whose embeddings, projections, biases and norm weights retain
/// their file block encodings. Activations and the KV cache are floating tensors.
#[derive(Debug)]
pub struct GgufCausalLm<R: DeviceRuntime> where R::Device: DeviceOps {
    embedding: GgmlGemm<R>,
    output: Projection<R>,
    norm: GgmlGemm<R>,
    layers: Vec<Layer<R>>,
    rotary: RotaryEncoding<GgufBackend<R>>,
    half_split: bool,
    heads: usize,
    kv_heads: usize,
    head_dimension: usize,
    epsilon: f64,
    max_sequence_length: usize,
    sliding_window: usize,
    query_scale: f64,
}

fn norm<R: DeviceRuntime, const D: usize>(
    input: Tensor<GgufBackend<R>, D>, gamma: &GgmlGemm<R>, epsilon: f64,
) -> Result<Tensor<GgufBackend<R>, D>, GgufLoadError> where R::Device: DeviceOps {
    let variance = input.clone().powf_scalar(2.0).mean_dim(D - 1);
    let normalized = input / (variance + epsilon).sqrt();
    Ok(Tensor::from_primitive(TensorPrimitive::Float(
        gamma.apply_vector(normalized.into_primitive().tensor(), true)?,
    )))
}

impl<R: DeviceRuntime> GgufCausalLm<R> where R::Device: DeviceOps {
    pub fn limits(&self) -> crate::CausalModelLimits {
        crate::CausalModelLimits { vocab_size: self.embedding.rows(), max_sequence_length: self.max_sequence_length }
    }
    pub fn new_cache(&self) -> LlamaKvCache<GgufBackend<R>> {
        LlamaKvCache::new(self.layers.len(), self.max_sequence_length)
    }
    fn rotate(&self, input: Tensor<GgufBackend<R>, 4>, start: usize) -> Tensor<GgufBackend<R>, 4> {
        let width = self.rotary.theta.dims()[0] * 2;
        let [b, h, s, d] = input.dims();
        let prefix = if width == d { input.clone() } else { input.clone().slice([0..b, 0..h, 0..s, 0..width]) };
        let rotated = if self.half_split { apply_half_split_rope(&self.rotary, prefix, start) }
            else { self.rotary.apply(prefix, start) };
        if width == d { rotated }
        else { Tensor::cat(vec![rotated, input.slice([0..b, 0..h, 0..s, width..d])], 3) }
    }
    fn hidden(
        &self, tokens: Tensor<GgufBackend<R>, 2, Int>, cache: &mut LlamaKvCache<GgufBackend<R>>,
    ) -> Result<Tensor<GgufBackend<R>, 3>, GgufLoadError> {
        let [batch, sequence] = tokens.dims();
        let end = cache.position.checked_add(sequence).ok_or_else(|| GgufLoadError("GGUF cache position overflow".into()))?;
        if sequence == 0 || cache.layers.len() != self.layers.len()
            || end > self.max_sequence_length || end > cache.max_sequence_length
            || cache.layers.iter().any(|layer| layer.sequence_length() != cache.position)
        { return Err(GgufLoadError("GGUF token sequence and KV cache geometry differ".into())); }
        let mut hidden = Tensor::from_primitive(TensorPrimitive::Float(
            self.embedding.gather(tokens.into_primitive(), DType::F32)?,
        ));
        for (layer, kv) in self.layers.iter().zip(cache.layers.iter_mut()) {
            let input = norm(hidden.clone(), &layer.input_norm, self.epsilon)?;
            let mut q = layer.q.forward(input.clone())?.reshape([batch, sequence, self.heads, self.head_dimension]);
            let mut k = layer.k.forward(input.clone())?.reshape([batch, sequence, self.kv_heads, self.head_dimension]);
            if let Some(gamma) = &layer.q_norm { q = norm(q, gamma, self.epsilon)?; }
            if let Some(gamma) = &layer.k_norm { k = norm(k, gamma, self.epsilon)?; }
            let q = self.rotate(q.swap_dims(1, 2), cache.position) * self.query_scale;
            let k = self.rotate(k.swap_dims(1, 2), cache.position);
            let v = layer.v.forward(input)?.reshape([batch, sequence, self.kv_heads, self.head_dimension]).swap_dims(1, 2);
            let (k, v) = kv.append(k, v);
            let k = repeat_kv(k, self.heads / self.kv_heads);
            let v = repeat_kv(v, self.heads / self.kv_heads);
            let attended = causal_attention_windowed(q, k, v, cache.position, self.sliding_window)
                .swap_dims(1, 2).reshape([batch, sequence, self.heads * self.head_dimension]);
            hidden = hidden + layer.o.forward(attended)?;
            let input = norm(hidden.clone(), &layer.post_norm, self.epsilon)?;
            hidden = hidden + layer.feed_forward.forward(input)?;
        }
        cache.position = end;
        norm(hidden, &self.norm, self.epsilon)
    }
    pub fn forward(&self, tokens: Tensor<GgufBackend<R>, 2, Int>) -> Result<Tensor<GgufBackend<R>, 3>, GgufLoadError> {
        self.forward_cached(tokens, &mut self.new_cache())
    }
    pub fn forward_cached(
        &self, tokens: Tensor<GgufBackend<R>, 2, Int>, cache: &mut LlamaKvCache<GgufBackend<R>>,
    ) -> Result<Tensor<GgufBackend<R>, 3>, GgufLoadError> {
        self.output.forward(self.hidden(tokens, cache)?)
    }
    pub fn forward_cached_last(
        &self, tokens: Tensor<GgufBackend<R>, 2, Int>, cache: &mut LlamaKvCache<GgufBackend<R>>,
    ) -> Result<Tensor<GgufBackend<R>, 3>, GgufLoadError> {
        let hidden = self.hidden(tokens, cache)?;
        let [b, s, d] = hidden.dims();
        self.output.forward(hidden.slice([0..b, s - 1..s, 0..d]))
    }
}

impl<R: DeviceRuntime> crate::CausalModel<GgufBackend<R>> for GgufCausalLm<R> where R::Device: DeviceOps {
    type Cache = LlamaKvCache<GgufBackend<R>>;
    fn new_cache(&self) -> Self::Cache { self.new_cache() }
    fn forward_cached_last(&self, tokens: Tensor<GgufBackend<R>, 2, Int>, cache: &mut Self::Cache) -> Tensor<GgufBackend<R>, 3> {
        GgufCausalLm::forward_cached_last(self, tokens, cache).expect("GGUF cached inference failed")
    }
    fn try_forward_cached_last(&self, tokens: Tensor<GgufBackend<R>, 2, Int>, cache: &mut Self::Cache)
        -> Result<Tensor<GgufBackend<R>, 3>, crate::GenerationError>
    {
        GgufCausalLm::forward_cached_last(self, tokens, cache).map_err(|error| crate::GenerationError(error.0))
    }
}

struct Weights<'a, R: DeviceRuntime> {
    source: &'a GgufModel,
    device: &'a R::Device,
    consumed: BTreeSet<String>,
}
impl<R: DeviceRuntime> Weights<'_, R> {
    fn tensor(&mut self, name: &str, shape: &[usize]) -> Result<GgmlGemm<R>, GgufLoadError> {
        let info = self.source.tensor(name).ok_or_else(|| GgufLoadError(format!("Missing GGUF tensor {name}")))?;
        if info.shape() != shape {
            return Err(GgufLoadError(format!("{name}: expected {shape:?}, found {:?}", info.shape())));
        }
        let tensor = self.source.read_packed(name)?;
        self.consumed.insert(name.into());
        Ok(GgmlGemm::from_bytes(tensor.bytes, tensor.info.dtype, tensor.info.rows(),
            tensor.info.dimensions[0], tensor.endian == GgufEndian::Big, self.device)?)
    }
    fn projection(&mut self, name: &str, input: usize, output: usize) -> Result<Projection<R>, GgufLoadError> {
        let weight = self.tensor(&format!("{name}.weight"), &[output, input])?;
        let name = format!("{name}.bias");
        let bias = if self.source.tensor(&name).is_some() { Some(self.tensor(&name, &[output])?) } else { None };
        Ok(Projection { weight, bias })
    }
    fn dense_feed_forward(&mut self, prefix: &str, suffix: &str, hidden: usize, ff: usize)
        -> Result<DenseFeedForward<R>, GgufLoadError>
    {
        Ok(DenseFeedForward {
            gate: self.projection(&format!("{prefix}.ffn_gate{suffix}"), hidden, ff)?,
            up: self.projection(&format!("{prefix}.ffn_up{suffix}"), hidden, ff)?,
            down: self.projection(&format!("{prefix}.ffn_down{suffix}"), ff, hidden)?,
        })
    }
    fn float_vector(&mut self, name: &str, width: usize) -> Result<Vec<f32>, GgufLoadError> {
        let tensor = self.source.read_packed(name)?;
        if tensor.info.shape() != [width] { return Err(GgufLoadError(format!("Invalid RoPE vector {name}"))); }
        let big = tensor.endian == GgufEndian::Big;
        let result = match tensor.info.dtype {
            GgmlType::F32 => tensor.bytes.chunks_exact(4).map(|bytes| {
                let b = bytes.try_into().unwrap();
                f32::from_bits(if big { u32::from_be_bytes(b) } else { u32::from_le_bytes(b) })
            }).collect(),
            GgmlType::F16 | GgmlType::BF16 => tensor.bytes.chunks_exact(2).map(|bytes| {
                let b = bytes.try_into().unwrap();
                let bits = if big { u16::from_be_bytes(b) } else { u16::from_le_bytes(b) };
                if tensor.info.dtype == GgmlType::F16 { half::f16::from_bits(bits).to_f32() }
                else { half::bf16::from_bits(bits).to_f32() }
            }).collect(),
            _ => return Err(GgufLoadError("GGUF RoPE frequency factors require floating scalar storage".into())),
        };
        self.consumed.insert(name.into());
        Ok(result)
    }
}

/// Loads dense and MoE Llama, Qwen2 and Qwen3 GGUF directly, including sharded GGUF.
pub fn load_gguf<R: DeviceRuntime>(path: impl AsRef<Path>, device: &R::Device) -> Result<LoadedGguf<R>, GgufLoadError>
where R::Device: DeviceOps {
    let source = GgufModel::open(path)?;
    let metadata = source.metadata();
    let architecture = metadata.get("general.architecture").and_then(GgufValue::as_str)
        .ok_or_else(|| GgufLoadError("GGUF general.architecture is missing".into()))?;
    if !matches!(architecture, "llama" | "qwen2" | "qwen3" | "qwen2moe" | "qwen3moe") {
        return Err(GgufLoadError(format!("No native GGUF model adapter for {architecture}")));
    }
    let integer = |name: &str, default: Option<usize>| -> Result<usize, GgufLoadError> {
        let key = format!("{architecture}.{name}");
        match metadata.get(&key) {
            Some(value) => value.as_u64().and_then(|v| usize::try_from(v).ok())
                .ok_or_else(|| GgufLoadError(format!("Invalid GGUF integer {key}"))),
            None => default.ok_or_else(|| GgufLoadError(format!("Missing GGUF metadata {key}"))),
        }
    };
    let number = |name: &str, default: Option<f64>| -> Result<f64, GgufLoadError> {
        let key = format!("{architecture}.{name}");
        let value = match metadata.get(&key) {
            Some(value) => value.as_f64().ok_or_else(|| GgufLoadError(format!("Invalid GGUF number {key}")))?,
            None => default.ok_or_else(|| GgufLoadError(format!("Missing GGUF metadata {key}")))?,
        };
        if !value.is_finite() { return Err(GgufLoadError(format!("Non-finite GGUF metadata {key}"))); }
        Ok(value)
    };
    let hidden = integer("embedding_length", None)?;
    let heads = integer("attention.head_count", None)?;
    let kv_heads = integer("attention.head_count_kv", Some(heads))?;
    let head_dimension = integer("attention.key_length", hidden.checked_div(heads))?;
    let value_dimension = integer("attention.value_length", Some(head_dimension))?;
    let rotary_dimension = integer("rope.dimension_count", Some(head_dimension))?;
    let layer_count = integer("block_count", None)?;
    let ff = integer("feed_forward_length", None)?;
    let experts = integer("expert_count", Some(0))?;
    let selected_experts = integer("expert_used_count", Some(0))?;
    if (experts == 0 && selected_experts != 0)
        || (experts != 0 && (selected_experts == 0 || selected_experts > experts))
        || (matches!(architecture, "qwen2moe" | "qwen3moe") && experts == 0)
    { return Err(GgufLoadError("Invalid GGUF expert count or selection count".into())); }
    let expert_ff = match integer("expert_feed_forward_length", Some(0))? {
        0 if architecture == "llama" => ff,
        0 => ff.checked_div(selected_experts).unwrap_or(ff),
        value => value,
    };
    let shared_ff = match integer("expert_shared_feed_forward_length", Some(0))? {
        0 if architecture == "qwen2moe" => ff,
        value => value,
    };
    let expert_scale = number("expert_weights_scale", Some(1.0))?;
    if experts != 0 && expert_ff == 0 {
        return Err(GgufLoadError("GGUF expert feed-forward width is zero".into()));
    }
    let context = integer("context_length", None)?;
    let epsilon = number("attention.layer_norm_rms_epsilon", None)?;
    let theta = number("rope.freq_base", Some(10000.0))? as f32;
    if hidden == 0 || heads == 0 || kv_heads == 0 || heads % kv_heads != 0
        || head_dimension == 0 || value_dimension != head_dimension || rotary_dimension == 0
        || rotary_dimension > head_dimension || rotary_dimension % 2 != 0
        || layer_count == 0 || ff == 0 || context == 0 || epsilon <= 0.0 || theta <= 0.0 || !theta.is_finite()
    { return Err(GgufLoadError("Invalid GGUF decoder geometry or normalization parameters".into())); }
    let query_width = heads.checked_mul(head_dimension)
        .ok_or_else(|| GgufLoadError("GGUF query geometry overflows".into()))?;
    let kv_width = kv_heads.checked_mul(head_dimension)
        .ok_or_else(|| GgufLoadError("GGUF KV geometry overflows".into()))?;
    let attention_scale = number("attention.scale", Some(0.0))?;
    let query_scale = if attention_scale == 0.0 { 1.0 }
        else { attention_scale * (head_dimension as f64).sqrt() };
    let embed = source.tensor("token_embd.weight").ok_or_else(|| GgufLoadError("Missing GGUF embeddings".into()))?;
    if embed.dimensions.len() != 2 || embed.dimensions[0] != hidden {
        return Err(GgufLoadError("GGUF embedding shape differs from metadata".into()));
    }
    let vocab = embed.rows();
    if integer("vocab_size", Some(vocab))? != vocab {
        return Err(GgufLoadError("GGUF vocabulary metadata differs from embedding rows".into()));
    }
    let mut weights = Weights::<R> { source: &source, device, consumed: BTreeSet::new() };
    let embedding = weights.tensor("token_embd.weight", &[vocab, hidden])?;
    let mut layers = Vec::with_capacity(layer_count);
    for i in 0..layer_count {
        let prefix = format!("blk.{i}");
        let q_norm = if matches!(architecture, "qwen3" | "qwen3moe") { Some(weights.tensor(&format!("{prefix}.attn_q_norm.weight"), &[head_dimension])?) } else { None };
        let k_norm = if matches!(architecture, "qwen3" | "qwen3moe") { Some(weights.tensor(&format!("{prefix}.attn_k_norm.weight"), &[head_dimension])?) } else { None };
        let feed_forward = if experts == 0 {
            FeedForward::Dense(weights.dense_feed_forward(&prefix, "", hidden, ff)?)
        } else {
            let gate_name = format!("{prefix}.ffn_gate_exps.weight");
            let gate = if architecture != "llama" || source.tensor(&gate_name).is_some() {
                Some(weights.tensor(&gate_name, &[experts, expert_ff, hidden])?)
            } else { None };
            let shared = if shared_ff != 0 {
                Some(weights.dense_feed_forward(&prefix, "_shexp", hidden, shared_ff)?)
            } else { None };
            let shared_gate = if architecture == "qwen2moe" {
                Some(weights.tensor(&format!("{prefix}.ffn_gate_inp_shexp.weight"), &[hidden])?)
            } else { None };
            FeedForward::Experts {
                router: weights.projection(&format!("{prefix}.ffn_gate_inp"), hidden, experts)?,
                gate,
                up: weights.tensor(&format!("{prefix}.ffn_up_exps.weight"), &[experts, expert_ff, hidden])?,
                down: weights.tensor(&format!("{prefix}.ffn_down_exps.weight"), &[experts, hidden, expert_ff])?,
                count: experts, selected: selected_experts,
                normalize: architecture != "qwen2moe", scale: expert_scale, shared, shared_gate,
            }
        };
        layers.push(Layer {
            q: weights.projection(&format!("{prefix}.attn_q"), hidden, query_width)?,
            k: weights.projection(&format!("{prefix}.attn_k"), hidden, kv_width)?,
            v: weights.projection(&format!("{prefix}.attn_v"), hidden, kv_width)?,
            o: weights.projection(&format!("{prefix}.attn_output"), query_width, hidden)?,
            feed_forward,
            input_norm: weights.tensor(&format!("{prefix}.attn_norm.weight"), &[hidden])?,
            post_norm: weights.tensor(&format!("{prefix}.ffn_norm.weight"), &[hidden])?,
            q_norm, k_norm,
        });
    }
    let norm = weights.tensor("output_norm.weight", &[hidden])?;
    let output = if source.tensor("output.weight").is_some() { weights.projection("output", hidden, vocab)? }
        else {
            let bias = if source.tensor("output.bias").is_some() { Some(weights.tensor("output.bias", &[vocab])?) } else { None };
            Projection { weight: embedding.clone(), bias }
        };
    let rope_kind = metadata.get(&format!("{architecture}.rope.scaling.type")).and_then(GgufValue::as_str).unwrap_or("linear");
    let factor = number("rope.scaling.factor", Some(number("rope.scale_linear", Some(1.0))?))?;
    if !matches!(rope_kind, "none" | "linear" | "yarn") || factor <= 0.0 {
        return Err(GgufLoadError(format!("Unsupported GGUF RoPE scaling {rope_kind}")));
    }
    let frequency_scale = if rope_kind == "none" { 1.0 } else { 1.0 / factor };
    let mut inverse: Vec<f32> = (0..rotary_dimension / 2)
        .map(|i| theta.powf(-((2 * i) as f32) / rotary_dimension as f32)).collect();
    if source.tensor("rope_freqs.weight").is_some() {
        let factors = weights.float_vector("rope_freqs.weight", inverse.len())?;
        for (value, factor) in inverse.iter_mut().zip(factors) {
            if !factor.is_finite() || factor <= 0.0 { return Err(GgufLoadError("Invalid GGUF RoPE frequency factor".into())); }
            *value /= factor;
        }
    }
    let mut attention_factor = number("rope.scaling.attn_factor", Some(1.0))?;
    let ext_factor = number("rope.scaling.yarn_ext_factor", Some(if rope_kind == "yarn" { 1.0 } else { 0.0 }))?;
    if ext_factor != 0.0 {
        let original = integer("rope.scaling.original_context_length", Some(context))? as f64;
        let fast = number("rope.scaling.yarn_beta_fast", Some(32.0))?;
        let slow = number("rope.scaling.yarn_beta_slow", Some(1.0))?;
        if original <= 0.0 || fast <= 0.0 || slow <= 0.0 || theta <= 1.0 {
            return Err(GgufLoadError("Invalid GGUF YaRN scaling parameters".into()));
        }
        let correction = |beta: f64| rotary_dimension as f64 * (original / (beta * 2.0 * std::f64::consts::PI)).ln()
            / (2.0 * (theta as f64).ln());
        let low = correction(fast).floor().max(0.0);
        let high = correction(slow).ceil().min((rotary_dimension - 1) as f64);
        for (i, value) in inverse.iter_mut().enumerate() {
            let ramp = (1.0 - ((i as f64 - low) / (high - low).max(0.001)).clamp(0.0, 1.0)) * ext_factor;
            *value *= (frequency_scale * (1.0 - ramp) + ramp) as f32;
        }
        let log_multiplier = number("rope.scaling.yarn_log_multiplier", Some(0.0))?;
        let mscale = |multiplier: f64| if factor <= 1.0 { 1.0 } else { 1.0 + 0.1 * multiplier * factor.ln() };
        attention_factor *= mscale(1.0) / if log_multiplier == 0.0 { 1.0 } else { mscale(log_multiplier) };
    } else {
        for value in &mut inverse { *value *= frequency_scale as f32; }
        attention_factor *= number("rope.scaling.yarn_attn_factor", Some(1.0))?;
    }
    let parameters = rope::RopeParameters {
        theta, inverse_frequencies: inverse, attention_factor,
        scaling: rope::RopeScaling::Static, per_layer: Vec::new(),
    };
    let rotary = parameters.init::<GgufBackend<R>>(&LlamaConfig {
        vocab_size: vocab, d_model: query_width, d_ff: ff, num_hidden_layers: layer_count,
        num_query_heads: heads, num_kv_heads: kv_heads, max_sequence_length: context,
        rms_norm_epsilon: epsilon, rope_theta: theta,
    }, device, DType::F32);
    let remaining: Vec<_> = source.tensor_names().filter(|name| !weights.consumed.contains(*name)).collect();
    if !remaining.is_empty() { return Err(GgufLoadError(format!("Unconsumed GGUF model tensors: {}", remaining.join(", ")))); }
    let window = integer("attention.sliding_window", Some(0))?;
    let model = GgufCausalLm {
        embedding, output, norm, layers, rotary, half_split: architecture != "llama",
        heads, kv_heads, head_dimension, epsilon, max_sequence_length: context,
        sliding_window: window, query_scale,
    };
    Ok(LoadedGguf { model, metadata: metadata.clone() })
}
