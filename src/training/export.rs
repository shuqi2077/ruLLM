//! Minimal Safetensors writer and Hugging Face export of packed models.
//!
//! Format: little-endian u64 header length, a JSON header mapping each name
//! to `{dtype, shape, data_offsets}`, then the concatenated raw tensors.

use crate::PackedLlamaForCausalLm;
use ruda_tensor::api::backend::Backend;
use ruda_tensor::api::{DType, TensorData};
use std::io::Write;
use std::path::Path;

fn safetensors_dtype(dtype: DType) -> Result<&'static str, String> {
    Ok(match dtype {
        DType::F32 => "F32",
        DType::F16 => "F16",
        DType::BF16 => "BF16",
        DType::F64 => "F64",
        DType::I64 => "I64",
        DType::I32 => "I32",
        DType::U8 => "U8",
        other => return Err(format!("dtype {other:?} cannot be written to Safetensors")),
    })
}

/// Write named tensors to `path` atomically (temporary file + rename).
pub fn write_safetensors(
    path: impl AsRef<Path>,
    tensors: &[(String, TensorData)],
    metadata: &[(&str, &str)],
) -> Result<(), String> {
    let path = path.as_ref();
    let mut header = serde_json::Map::new();
    if !metadata.is_empty() {
        let metadata: serde_json::Map<String, serde_json::Value> = metadata
            .iter()
            .map(|(key, value)| ((*key).to_string(), serde_json::Value::from(*value)))
            .collect();
        header.insert("__metadata__".into(), serde_json::Value::Object(metadata));
    }
    let mut offset = 0usize;
    for (name, data) in tensors {
        let bytes = data.as_bytes().len();
        header.insert(
            name.clone(),
            serde_json::json!({
                "dtype": safetensors_dtype(data.dtype)?,
                "shape": data.shape.iter().copied().collect::<Vec<usize>>(),
                "data_offsets": [offset, offset + bytes],
            }),
        );
        offset += bytes;
    }
    let mut header = serde_json::Value::Object(header).to_string().into_bytes();
    // Pad the header so tensor data starts 8-byte aligned.
    while header.len() % 8 != 0 {
        header.push(b' ');
    }
    let temporary = path.with_extension("safetensors.tmp");
    let write = || -> std::io::Result<()> {
        let mut file = std::io::BufWriter::new(std::fs::File::create(&temporary)?);
        file.write_all(&(header.len() as u64).to_le_bytes())?;
        file.write_all(&header)?;
        for (_, data) in tensors {
            file.write_all(data.as_bytes())?;
        }
        file.into_inner().map_err(|error| error.into_error())?.sync_all()?;
        std::fs::rename(&temporary, path)
    };
    write().map_err(|error| {
        let _ = std::fs::remove_file(&temporary);
        format!("cannot write {}: {error}", path.display())
    })
}

/// Export a packed (optionally LoRA-merged) model as `model.safetensors`
/// under Hugging Face Llama/Qwen2 names. Copy `config.json` and the
/// tokenizer files from the source checkpoint alongside it.
pub fn export_huggingface_safetensors<B: Backend>(
    model: &PackedLlamaForCausalLm<B>,
    path: impl AsRef<Path>,
    tie_word_embeddings: bool,
) -> Result<(), String> {
    let tensors = model.huggingface_tensors(tie_word_embeddings);
    write_safetensors(path, &tensors, &[("format", "pt")])
}
