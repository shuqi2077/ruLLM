use super::NativeSftModel;
use crate::HuggingFaceLoadError;
use ruda_tensor::api::backend::Backend;
use std::path::Path;

/// Load floating Qwen3 dense/MoE, Qwen3.5 text dense/MoE, DeepSeek V2/V3,
/// Kimi K2 or Kimi Linear into the same native training representation.
/// `B` may already be an autodiff backend; otherwise use `Module::train`.
pub fn load_huggingface_native_sft<B: Backend>(directory: impl AsRef<Path>, device: &B::Device)
    -> Result<NativeSftModel<B>, HuggingFaceLoadError> {
    let directory = directory.as_ref();
    let config: serde_json::Value = serde_json::from_slice(
        &std::fs::read(directory.join("config.json")).map_err(|e| HuggingFaceLoadError(e.to_string()))?,
    ).map_err(|e| HuggingFaceLoadError(e.to_string()))?;
    let text = config.get("text_config").unwrap_or(&config);
    if config.get("quantization_config").filter(|v| !v.is_null()).is_some()
        || text.get("quantization_config").filter(|v| !v.is_null()).is_some() {
        return Err(HuggingFaceLoadError("native SFT loading requires floating checkpoints, not packed inference quantization".into()));
    }
    let converted = match text.get("model_type").and_then(|v| v.as_str()) {
        Some("qwen3" | "qwen3_moe") => crate::load_huggingface_qwen3::<B>(directory, device)?.model.into_training(),
        Some("qwen3_5" | "qwen3_5_text" | "qwen3_5_moe" | "qwen3_5_moe_text") =>
            crate::load_huggingface_qwen35_text::<B>(directory, device)?.model.into_training(),
        Some("deepseek_v2" | "deepseek_v3" | "kimi_k2") =>
            crate::load_huggingface_deepseek_v2::<B>(directory, device)?.model.into_training(),
        Some("kimi_linear") => return crate::deepseek2::load_kimi_linear_sft::<B>(directory, device),
        other => return Err(HuggingFaceLoadError(format!("unsupported native SFT architecture {other:?}; Llama/Qwen2 use the packed training model"))),
    };
    converted.map_err(|e| HuggingFaceLoadError(e.to_string()))
}
