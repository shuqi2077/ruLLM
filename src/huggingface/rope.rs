use super::{HuggingFaceLlamaConfig, HuggingFaceLoadError};
use crate::llama::rope::{RopeParameters, RopeScaling};
use serde_json::Value;

fn error(message: impl ToString) -> HuggingFaceLoadError { HuggingFaceLoadError(message.to_string()) }
fn number(value: Option<&Value>, name: &str) -> Result<Option<f64>, HuggingFaceLoadError> {
    value.filter(|v| !v.is_null()).map(|v| v.as_f64().filter(|n| n.is_finite())
        .ok_or_else(|| error(format!("RoPE {name} must be finite")))).transpose()
}
fn integer(value: Option<&Value>, name: &str) -> Result<Option<usize>, HuggingFaceLoadError> {
    value.filter(|v| !v.is_null()).map(|v| v.as_u64().and_then(|n| usize::try_from(n).ok()).filter(|&n| n > 0)
        .ok_or_else(|| error(format!("RoPE {name} must be a positive integer")))).transpose()
}
pub(super) fn parameters(source: &HuggingFaceLlamaConfig) -> Result<RopeParameters, HuggingFaceLoadError> {
    let fields = source.rope_parameters.as_ref().filter(|v| !v.is_null())
        .or_else(|| source.rope_scaling.as_ref().filter(|v| !v.is_null()));
    if source.num_attention_heads == 0 || source.hidden_size == 0 || source.hidden_size % source.num_attention_heads != 0 {
        return Err(error("Llama/Qwen2 require nonzero divisible attention dimensions"));
    }
    layer_parameters(source.head_dim.unwrap_or(source.hidden_size / source.num_attention_heads), source.max_position_embeddings,
        source.rope_theta.unwrap_or(10_000.0) as f64, source.original_max_position_embeddings, source.partial_rotary_factor.unwrap_or(1.0), fields,
        source.layer_types.as_deref(), source.num_hidden_layers)
}
pub(crate) fn layer_parameters(dim: usize, max_position_embeddings: usize, default_theta: f64,
    original: Option<usize>, partial: f64, fields: Option<&Value>, layer_types: Option<&[String]>, layers: usize)
    -> Result<RopeParameters, HuggingFaceLoadError> {
    let nested = fields.and_then(Value::as_object).is_some_and(|object| object.values().any(Value::is_object));
    if !nested { return static_parameters(dim, max_position_embeddings, default_theta, original, partial, fields); }
    let types = layer_types.filter(|types| types.len() == layers && layers > 0)
        .ok_or_else(|| error("per-layer RoPE requires layer_types covering all decoder layers"))?;
    let mut per_layer = Vec::with_capacity(layers);
    for name in types {
        let parameters = fields.and_then(|fields| fields.get(name)).filter(|v| v.is_object())
            .ok_or_else(|| error(format!("missing RoPE parameters for layer type {name}")))?;
        per_layer.push(static_parameters(dim, max_position_embeddings, default_theta, original, partial, Some(parameters))?);
    }
    let mut result = per_layer[0].clone();
    result.per_layer = per_layer;
    Ok(result)
}
pub(crate) fn static_parameters(dim: usize, max_position_embeddings: usize, default_theta: f64,
    original_max_position_embeddings: Option<usize>, partial_rotary_factor: f64, fields: Option<&Value>)
    -> Result<RopeParameters, HuggingFaceLoadError> {
    let fields = fields.filter(|v| !v.is_null());
    if fields.is_some_and(|v| !v.is_object()) { return Err(error("RoPE parameters must be an object")); }
    let get = |name: &str| fields.and_then(|v| v.get(name));
    if fields.is_some_and(|v| v.as_object().unwrap().values().any(Value::is_object)) {
        return Err(error("dense HF decoders require a global rotary parameter set"));
    }
    let kind = get("rope_type").or_else(|| get("type")).map(|v| v.as_str().ok_or_else(|| error("RoPE type must be a string")))
        .transpose()?.unwrap_or("default");
    if let (Some(a), Some(b)) = (get("rope_type"), get("type")) { if a != b { return Err(error("conflicting RoPE type aliases")); } }
    let theta = number(get("rope_theta"), "rope_theta")?.unwrap_or(default_theta);
    let partial = number(get("partial_rotary_factor"), "partial_rotary_factor")?.unwrap_or(partial_rotary_factor);
    if !theta.is_finite() || theta <= 0.0 || theta > f32::MAX as f64 || !partial.is_finite() || partial <= 0.0 || partial > 1.0 || max_position_embeddings == 0
        || original_max_position_embeddings == Some(0) {
        return Err(error("RoPE requires finite positive theta, context and partial_rotary_factor in (0, 1]"));
    }
    let dim = (dim as f64 * partial) as usize;
    if dim == 0 || dim % 2 != 0 { return Err(error("RoPE head dimension must be positive and even")); }
    let theta = theta as f32;
    let frequencies = (0..dim / 2).map(|i| theta.powf((2 * i) as f32 / dim as f32)).collect::<Vec<_>>();
    let mut inverse = frequencies.iter().map(|f| 1.0f32 / f).collect::<Vec<_>>();
    let mut attention_factor = 1.0;
    let mut scaling = RopeScaling::Static;
    match kind {
        "default" => {},
        "linear" => {
            let factor = number(get("factor"), "factor")?.filter(|&n| n >= 1.0).ok_or_else(|| error("linear RoPE requires factor >= 1"))? as f32;
            for frequency in &mut inverse { *frequency /= factor; }
        },
        "dynamic" => {
            if dim <= 2 { return Err(error("dynamic NTK rotary dimension must exceed two")); }
            let factor = number(get("factor"), "factor")?.filter(|&n| n >= 1.0)
                .ok_or_else(|| error("dynamic NTK RoPE requires factor >= 1"))?;
            scaling = RopeScaling::Dynamic { factor, original_length: max_position_embeddings };
        },
        "longrope" => {
            let original = integer(get("original_max_position_embeddings"), "original_max_position_embeddings")?
                .or(original_max_position_embeddings).unwrap_or(max_position_embeddings);
            let factors = |name: &str| -> Result<Vec<f32>, HuggingFaceLoadError> {
                let values = get(name).and_then(Value::as_array).filter(|v| v.len() == dim / 2)
                    .ok_or_else(|| error(format!("LongRoPE {name} requires {} entries", dim / 2)))?;
                values.iter().map(|value| number(Some(value), name)?.filter(|&n| n > 0.0 && n <= f32::MAX as f64)
                    .map(|n| n as f32).ok_or_else(|| error(format!("LongRoPE {name} entries must be positive")))).collect()
            };
            let short = factors("short_factor")?;
            let long = factors("long_factor")?;
            let inverse_long = frequencies.iter().zip(long).map(|(f, scale)| 1.0 / (scale * f)).collect::<Vec<_>>();
            for (frequency, (base, scale)) in inverse.iter_mut().zip(frequencies.iter().zip(short)) { *frequency = 1.0 / (scale * base); }
            let factor = number(get("factor"), "factor")?.unwrap_or(max_position_embeddings as f64 / original as f64);
            if factor <= 0.0 { return Err(error("LongRoPE factor must be positive")); }
            attention_factor = number(get("attention_factor"), "attention_factor")?.unwrap_or_else(|| {
                if factor <= 1.0 { 1.0 } else { (1.0 + factor.ln() / (original as f64).ln()).sqrt() }
            });
            if inverse_long.iter().any(|v| !v.is_finite() || *v <= 0.0) { return Err(error("invalid LongRoPE long frequencies")); }
            scaling = RopeScaling::Long { original_length: original, inverse_long };
        },
        "llama3" => {
            let factor = number(get("factor"), "factor")?.filter(|&n| n >= 1.0).ok_or_else(|| error("Llama 3 RoPE requires factor >= 1"))? as f32;
            let low = number(get("low_freq_factor"), "low_freq_factor")?.filter(|&n| n > 0.0).ok_or_else(|| error("missing Llama 3 low_freq_factor"))? as f32;
            let high = number(get("high_freq_factor"), "high_freq_factor")?.filter(|&n| n > low as f64).ok_or_else(|| error("Llama 3 high_freq_factor must exceed low_freq_factor"))? as f32;
            let original = original_max_position_embeddings.or(integer(get("original_max_position_embeddings"), "original_max_position_embeddings")?)
                .unwrap_or(max_position_embeddings) as f32;
            for frequency in &mut inverse {
                let wavelength = 2.0 * std::f32::consts::PI / *frequency;
                if wavelength > original / low { *frequency /= factor; }
                else if wavelength >= original / high {
                    let smooth = (original / wavelength - low) / (high - low);
                    *frequency = (1.0 - smooth) * *frequency / factor + smooth * *frequency;
                }
            }
        },
        "yarn" => {
            if theta <= 1.0 { return Err(error("YaRN requires theta > 1")); }
            let original = original_max_position_embeddings.or(integer(get("original_max_position_embeddings"), "original_max_position_embeddings")?)
                .unwrap_or(max_position_embeddings);
            if original == 0 { return Err(error("YaRN original context must be positive")); }
            let factor = number(get("factor"), "factor")?.unwrap_or(max_position_embeddings as f64 / original as f64);
            if factor < 1.0 { return Err(error("YaRN factor must be >= 1")); }
            let beta_fast = number(get("beta_fast"), "beta_fast")?.filter(|n| *n != 0.0).unwrap_or(32.0);
            let beta_slow = number(get("beta_slow"), "beta_slow")?.filter(|n| *n != 0.0).unwrap_or(1.0);
            if beta_slow <= 0.0 || beta_fast < beta_slow { return Err(error("invalid YaRN correction range")); }
            let mscale = number(get("mscale"), "mscale")?;
            let mscale_all = number(get("mscale_all_dim"), "mscale_all_dim")?;
            let magnitude = |scale: f64| if factor <= 1.0 { 1.0 } else { 0.1 * scale * factor.ln() + 1.0 };
            attention_factor = number(get("attention_factor"), "attention_factor")?.unwrap_or_else(|| {
                match (mscale, mscale_all) {
                    (Some(a), Some(b)) if a != 0.0 && b != 0.0 => magnitude(a) / magnitude(b),
                    _ => magnitude(1.0),
                }
            });
            let truncate = get("truncate").map(|v| v.as_bool().ok_or_else(|| error("YaRN truncate must be boolean"))).transpose()?.unwrap_or(true);
            let correction = |rotations: f64| (dim as f64 * (original as f64 / (rotations * 2.0 * std::f64::consts::PI)).ln()) / (2.0 * (theta as f64).ln());
            let mut low = correction(beta_fast); let mut high = correction(beta_slow);
            if truncate { low = low.floor(); high = high.ceil(); }
            low = low.max(0.0); high = high.min((dim - 1) as f64);
            if low == high { high += 0.001; }
            for (index, frequency) in inverse.iter_mut().enumerate() {
                let ramp = ((index as f32 - low as f32) / (high - low) as f32).clamp(0.0, 1.0);
                let interpolated = 1.0f32 / ((factor as f32) * frequencies[index]);
                *frequency = interpolated * ramp + *frequency * (1.0 - ramp);
            }
        },
        other => return Err(error(format!("unsupported RoPE type {other}; supported: default, linear, dynamic, longrope, llama3, yarn"))),
    }
    if !attention_factor.is_finite() || attention_factor <= 0.0 || inverse.iter().any(|v| !v.is_finite() || *v <= 0.0) {
        return Err(error("invalid scaled RoPE frequencies or attention magnitude"));
    }
    Ok(RopeParameters {theta, inverse_frequencies: inverse, attention_factor, scaling, per_layer: Vec::new()})
}
