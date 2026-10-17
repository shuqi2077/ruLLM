//! OpenAI-compatible server for a local Qwen2/Qwen2.5 checkpoint on NVIDIA.
//!
//! usage: qwen2_openai_server <model-directory> [options]
//!   --address 127.0.0.1:8000   bind address (keep loopback unless trusted)
//!   --api-key KEY              require `Authorization: Bearer KEY`
//!   --kv-gib 1.0               KV pool budget
//!   --block-size 16            tokens per KV page
//!   --max-context N            per-request context limit (default: model limit)
//!   --max-active 16            concurrently scheduled requests
//!   --max-batch-tokens 512     tokens per device batch / prefill chunk
//!   --no-prefix-cache          disable automatic prefix caching
//!   --speculative N            prompt-lookup drafts of N tokens for greedy requests
//!
//! Example request:
//!   curl http://127.0.0.1:8000/v1/chat/completions -H "Content-Type: application/json" \
//!     -d '{"messages":[{"role":"user","content":"Hello"}],"stream":true}'

use half::bf16;
use ruda_tensor::api::backend::Backend;
use ruda_tensor_device::cuda::{Cuda, CudaDevice};
use rullm::{
    OpenAiServerConfig, PagedKvCacheConfig, SpeculativeConfig, load_huggingface_qwen2_pipeline,
    serve_openai,
};
use std::{error::Error, io, path::Path};

type ModelBackend = Cuda<bf16, i32>;

fn main() -> Result<(), Box<dyn Error>> {
    let mut args = std::env::args().skip(1);
    let directory = args.next().ok_or_else(|| {
        io::Error::new(io::ErrorKind::InvalidInput, "usage: qwen2_openai_server <model-directory> [options]")
    })?;
    let mut address = "127.0.0.1:8000".to_string();
    let mut api_key = None;
    let mut kv_gib = 1.0f64;
    let mut block_size = 16usize;
    let mut max_context = None;
    let mut max_active = 16usize;
    let mut max_batch_tokens = 512usize;
    let mut prefix_cache = true;
    let mut speculative = 0usize;
    while let Some(flag) = args.next() {
        let mut value = || {
            args.next()
                .ok_or_else(|| io::Error::new(io::ErrorKind::InvalidInput, format!("{flag} needs a value")))
        };
        match flag.as_str() {
            "--address" => address = value()?,
            "--api-key" => api_key = Some(value()?),
            "--kv-gib" => kv_gib = value()?.parse()?,
            "--block-size" => block_size = value()?.parse()?,
            "--max-context" => max_context = Some(value()?.parse::<usize>()?),
            "--max-active" => max_active = value()?.parse()?,
            "--max-batch-tokens" => max_batch_tokens = value()?.parse()?,
            "--no-prefix-cache" => prefix_cache = false,
            "--speculative" => speculative = value()?.parse()?,
            other => {
                return Err(io::Error::new(io::ErrorKind::InvalidInput, format!("unknown option {other}")).into());
            }
        }
    }

    let device = CudaDevice::default();
    let pipeline = load_huggingface_qwen2_pipeline::<ModelBackend>(&directory, &device)?;
    let config = pipeline.loaded.config.clone();
    let mut eos = pipeline.loaded.default_eos_token_ids.clone();
    // Chat turns end with <|im_end|>; stop there even if the checkpoint's
    // generation config lists only <|endoftext|>.
    if let Some(id) = pipeline.tokenizer.token_to_id("<|im_end|>") {
        let id = i32::try_from(id)?;
        if !eos.contains(&id) {
            eos.push(id);
        }
    }
    let tokenizer = pipeline.tokenizer.clone();
    let model = pipeline.loaded.model.into_packed_inference();
    ModelBackend::sync(&device)?;

    let head_dim = config.d_model / config.num_query_heads;
    let bytes_per_token = 2 * config.num_hidden_layers * config.num_kv_heads * head_dim * 2;
    let budget_tokens = (kv_gib * (1u64 << 30) as f64) as usize / bytes_per_token;
    let num_pages = (budget_tokens / block_size).max(1);
    let max_sequence_length = max_context
        .unwrap_or(config.max_sequence_length)
        .min(config.max_sequence_length)
        .min(num_pages * block_size);
    let kv = PagedKvCacheConfig {
        block_size,
        num_pages,
        max_sequence_length,
    };
    let model_name = Path::new(&directory)
        .file_name()
        .and_then(|name| name.to_str())
        .unwrap_or("ruda-model")
        .to_string();
    let mut server = OpenAiServerConfig::new(address.clone(), model_name, kv);
    server.api_key = api_key;
    server.prefix_caching = prefix_cache;
    server.batching.max_active_sequences = max_active;
    server.batching.max_batch_tokens = max_batch_tokens;
    server.speculative = (speculative > 0).then_some(SpeculativeConfig {
        draft_tokens: speculative,
        ngram: 3,
    });
    println!(
        "{}",
        serde_json::json!({
            "stage": "serving", "address": address, "kv_pages": num_pages, "block_size": block_size,
            "kv_tokens": num_pages * block_size, "max_context": max_sequence_length,
            "max_active": max_active, "max_batch_tokens": max_batch_tokens,
            "prefix_cache": prefix_cache, "auth": server.api_key.is_some(),
        })
    );
    serve_openai(model, tokenizer, eos, server, &device)?;
    Ok(())
}
