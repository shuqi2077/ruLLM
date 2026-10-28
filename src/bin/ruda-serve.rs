#[path = "ruda-serve/worker.rs"]
mod worker;
use rullm::DeepseekV2AttentionMode;
use rullm::prefix_cache::PrefixCacheConfig;
use rullm::serving;
use std::{error::Error, io, path::PathBuf};

#[tokio::main(flavor = "multi_thread", worker_threads = 2)]
async fn main() -> Result<(), Box<dyn Error>> {
    let mut args = std::env::args().skip(1);
    let mut directory = None;
    let mut model_name = None;
    let mut model_id = None;
    let mut address = "127.0.0.1:8000".to_string();
    let mut queue_capacity = 64;
    let mut responses_max_output_tokens = None;
    let mut autotune_cache_directory = None;
    let mut device_indices = vec![0usize];
    let mut backend = "native".to_string();
    let mut dtype = worker::Dtype::Bf16;
    let mut libtorch_device = "cuda".to_string();
    let mut explicit_libtorch_device = false;
    let mut role = "integrated".to_string();
    let mut prefill_address = None;
    let mut prefill_transport = "host".to_string();
    let mut gpu_kv_library = None;
    let mut gpu_kv_transports = "rc,cuda_ipc".to_string();
    let mut gpu_kv_timeout_ms = 300_000u64;
    let mut gpu_kv_options = false;
    let mut mla_mode = DeepseekV2AttentionMode::Expanded;
    let mut draft_directory = None;
    let mut draft_id = None;
    let mut causal_draft_tokens = None;
    let mut mtp = false;
    let mut multimodal = false;
    let mut batch_sequences = None;
    let mut batch_tokens = None;
    let mut batch_pages = None;
    let mut batch_block = None;
    let mut active_kv_host_bytes = 0usize;
    let mut config = PrefixCacheConfig {
        block_tokens: 64,
        device_entries: 4,
        host_bytes: 0,
        disk_directory: None,
        disk_bytes: 0,
    };
    while let Some(arg) = args.next() {
        if arg == "--help" {
            println!("[--dtype bf16|fp16 (default bf16; model weights, activations and KV cache)]");
            println!(
                "ruda-serve --model-path DIR --model-id EXACT-WEIGHTS-IDENTITY [--served-model-name NAME] [--backend native|libtorch (default native; libtorch feature required)] [--libtorch-device cpu|cuda|mps (default cuda)] [--devices 0,1 (Llama/Qwen2/Qwen3(dense/MoE)/Qwen3.5(dense/MoE) TP / DeepSeek-V2/V3/Kimi-K2/Kimi-Linear EP)] [--draft-model-path DIR --draft-model-id EXACT-DRAFT-IDENTITY (Qwen3 DFlash/DSpark; single device or native TP batching)] [--mla-mode expanded|absorbed (DeepSeek-V2/V3/Kimi-K2 only; default expanded)] [--role integrated|prefill|decode] [--prefill-address HOST:PORT (prefill listen/decode connect, TCP host KV)] [--bind 127.0.0.1:8000] [--queue-capacity N] [--multimodal (native Qwen3.5(dense/MoE) JPEG/PNG; single-device integrated)] [--batch-max-sequences N --batch-max-tokens N --batch-kv-pages N --batch-block-tokens N (native Llama/Qwen2/Qwen3(dense/MoE)/Qwen3.5(dense/MoE)/DeepSeek-V2/V3/Kimi-K2/Kimi-Linear text; integrated/prefill/decode; Llama/Qwen2/Qwen3(dense/MoE)/Qwen3.5(dense/MoE) TP/DeepSeek-V2/V3/Kimi-K2/Kimi-Linear EP; prefix/host/SSD cache; Qwen3 draft; no image batching)] [--prefix-block-tokens N] [--prefix-device-entries N] [--kv-host-bytes N] [--kv-disk-directory DIR --kv-disk-bytes N]"
            );
            println!(
                "[structured-output feature: response_format JSON Schema/JSON object or grammar regex/Lark/llguidance; standard or continuous-batch serving]"
            );
            println!("[--active-kv-host-bytes N enables bounded active-request KV/recurrent-state swap for continuous batching; distinct from reusable --kv-host-bytes prefix entries; no disk spill]");
            println!("[--mtp uses checkpoint DeepSeek MTP modules for speculative decoding; single-device or native EP, continuous batching and prefill/decode KV transfer; no separate draft model]");
            println!("[gpu-kv feature: --prefill-transport ucx --gpu-kv-library PATH_TO_RUCCL_UCX_BRIDGE [--gpu-kv-transports rc,cuda_ipc] [--gpu-kv-timeout-ms 300000]; native batch prefill/decode roles, CUDA device payloads with TCP metadata, no implicit host fallback]");
            println!("[--causal-draft-tokens N selects an independent causal draft instead of DFlash/DSpark; requires --draft-model-path/--draft-model-id and native batching; Llama/Qwen2/Qwen3(dense/MoE)/Qwen3.5(dense/MoE)/DeepSeek-V2/V3/Kimi-K2/Kimi-Linear target and draft, Llama/Qwen2/Qwen3(dense/MoE)/Qwen3.5(dense/MoE) TP/DeepSeek-V2/V3/Kimi-K2/Kimi-Linear EP targets with draft on root GPU, matching token-ID mappings/decoders]");
            println!(
                "[Codex: --responses-max-output-tokens N enables /v1/responses and /v1/codex/models; checkpoint context window, stateless HTTP tool replay]"
            );
            println!(
                "[native stack-autotune feature: --autotune-cache-directory DIR enables Explore with the default validated policy and persistent tuning records]"
            );
            return Ok(());
        }
        if arg == "--multimodal" {
            multimodal = true;
            continue;
        }
        if arg == "--mtp" {mtp=true;continue;}
        let value = args
            .next()
            .ok_or_else(|| io::Error::other(format!("missing value for {arg}")))?;
        match arg.as_str() {
            "--model-path" => directory = Some(PathBuf::from(value)),
            "--model-id" => model_id = Some(value),
            "--served-model-name" => model_name = Some(value),
            "--autotune-cache-directory" => autotune_cache_directory = Some(PathBuf::from(value)),
            "--bind" => address = value,
            "--backend" => backend = value,
            "--dtype" => dtype = value.parse().map_err(io::Error::other)?,
            "--libtorch-device" => {
                libtorch_device = value;
                explicit_libtorch_device = true;
            }
            "--role" => role = value,
            "--prefill-address" => prefill_address = Some(value),
            "--prefill-transport" => prefill_transport = value,
            "--gpu-kv-library" => {gpu_kv_library=Some(PathBuf::from(value));gpu_kv_options=true;},
            "--gpu-kv-transports" => {gpu_kv_transports=value;gpu_kv_options=true;},
            "--gpu-kv-timeout-ms" => {gpu_kv_timeout_ms=value.parse()?;gpu_kv_options=true;},
            "--draft-model-path" => draft_directory = Some(PathBuf::from(value)),
            "--draft-model-id" => draft_id = Some(value),
            "--causal-draft-tokens" => causal_draft_tokens = Some(value.parse::<usize>()?),
            "--mla-mode" => {
                mla_mode = match value.as_str() {
                    "expanded" => DeepseekV2AttentionMode::Expanded,
                    "absorbed" => DeepseekV2AttentionMode::Absorbed,
                    _ => {
                        return Err(
                            io::Error::other("--mla-mode must be expanded or absorbed").into()
                        );
                    }
                }
            }
            "--queue-capacity" => queue_capacity = value.parse()?,
            "--responses-max-output-tokens" => {
                responses_max_output_tokens = Some(value.parse::<usize>()?)
            }
            "--batch-max-sequences" => batch_sequences = Some(value.parse::<usize>()?),
            "--batch-max-tokens" => batch_tokens = Some(value.parse::<usize>()?),
            "--batch-kv-pages" => batch_pages = Some(value.parse::<usize>()?),
            "--batch-block-tokens" => batch_block = Some(value.parse::<usize>()?),
            "--active-kv-host-bytes" => active_kv_host_bytes = value.parse()?,
            "--devices" => {
                device_indices = value.split(',').map(str::parse).collect::<Result<_, _>>()?
            }
            "--prefix-block-tokens" => config.block_tokens = value.parse()?,
            "--prefix-device-entries" => config.device_entries = value.parse()?,
            "--kv-host-bytes" => config.host_bytes = value.parse()?,
            "--kv-disk-directory" => config.disk_directory = Some(PathBuf::from(value)),
            "--kv-disk-bytes" => config.disk_bytes = value.parse()?,
            _ => return Err(io::Error::other(format!("unknown argument {arg}")).into()),
        }
    }
    let directory = directory
        .ok_or_else(|| io::Error::other("--model-path is required"))?
        .canonicalize()?;
    let model_id = model_id
        .filter(|id: &String| !id.is_empty())
        .ok_or_else(|| {
            io::Error::other("--model-id must identify the exact frozen weights/configuration")
        })?;
    if device_indices.is_empty()
        || device_indices
            .iter()
            .enumerate()
            .any(|(i, id)| device_indices[..i].contains(id))
    {
        return Err(io::Error::other("--devices requires distinct visible CUDA ordinals").into());
    }
    if !matches!(backend.as_str(), "native" | "libtorch") {
        return Err(io::Error::other("--backend must be native or libtorch").into());
    }
    if !matches!(libtorch_device.as_str(), "cpu" | "cuda" | "mps") {
        return Err(io::Error::other("--libtorch-device must be cpu, cuda or mps").into());
    }
    if backend == "native" && explicit_libtorch_device {
        return Err(
            io::Error::other("--libtorch-device is only valid with --backend libtorch").into(),
        );
    }
    if backend == "libtorch" && libtorch_device != "cuda" && device_indices != [0] {
        return Err(io::Error::other("CPU/MPS has no CUDA ordinal; omit --devices").into());
    }
    #[cfg(not(feature = "nvidia"))]
    if backend == "native" {
        return Err(
            io::Error::other("native serving requires the nvidia or nvidia-ptx feature").into(),
        );
    }
    #[cfg(not(feature = "libtorch"))]
    if backend == "libtorch" {
        return Err(io::Error::other("LibTorch serving requires the libtorch feature").into());
    }
    #[cfg(not(feature = "tensor-parallel"))]
    if device_indices.len() > 1 {
        return Err(
            io::Error::other("multiple devices require the tensor-parallel feature").into(),
        );
    }
    if !matches!(role.as_str(), "integrated" | "prefill" | "decode") {
        return Err(io::Error::other("--role must be integrated, prefill or decode").into());
    }
    if (role == "integrated") == prefill_address.is_some() {
        return Err(io::Error::other(
            "--prefill-address is required only for prefill/decode roles",
        )
        .into());
    }
    let prefill_only = role == "prefill";
    if !matches!(prefill_transport.as_str(),"host"|"ucx") || (prefill_transport=="host" && gpu_kv_options) {
        return Err(io::Error::other("--prefill-transport must be host or ucx; GPU KV options require ucx").into());
    }
    if prefill_transport=="ucx" && (role=="integrated" || backend!="native" || batch_sequences.is_none() || multimodal || device_indices.iter().any(|&i|i>u16::MAX as usize)) {
        return Err(io::Error::other("UCX KV handoff requires native batch prefill/decode and representable CUDA device ordinals").into());
    }
    #[cfg(not(feature="gpu-kv"))]
    if prefill_transport=="ucx" {return Err(io::Error::other("UCX KV handoff requires the gpu-kv build feature").into());}
    #[cfg(feature="gpu-kv")]
    let gpu_kv=if prefill_transport=="ucx" {
        let gpu=rullm::disaggregated::GpuKvConfig{library:gpu_kv_library.ok_or_else(||io::Error::other("--gpu-kv-library is required for ucx"))?,
            transports:gpu_kv_transports,timeout_ms:gpu_kv_timeout_ms};
        gpu.validate().map_err(io::Error::other)?;Some(gpu)
    } else {None};
    if draft_directory.is_some() != draft_id.is_some()
        || draft_id.as_ref().is_some_and(|id: &String| id.is_empty())
    {
        return Err(io::Error::other(
            "draft model path requires an explicit nonempty exact draft identity",
        )
        .into());
    }
    let draft_directory = draft_directory
        .map(|directory| directory.canonicalize())
        .transpose()?;
    if causal_draft_tokens.is_some_and(|n| n == 0) || (causal_draft_tokens.is_some() && draft_directory.is_none()) {
        return Err(io::Error::other("--causal-draft-tokens requires a positive length and an explicit draft model").into());
    }
    let batch = if let Some(max_sequences) = batch_sequences {
        Some(serving::BatchServingConfig {
            max_sequences,
            max_batch_tokens: batch_tokens.ok_or_else(|| {
                io::Error::other("--batch-max-tokens is required for batch serving")
            })?,
            kv_pages: batch_pages.ok_or_else(|| {
                io::Error::other("--batch-kv-pages is required for batch serving")
            })?,
            block_tokens: batch_block.unwrap_or(64),
            max_queued_requests: queue_capacity,
            active_kv_host_bytes,
        })
    } else {
        if batch_tokens.is_some() || batch_pages.is_some() || batch_block.is_some() || active_kv_host_bytes!=0 {
            return Err(io::Error::other("batch options require --batch-max-sequences").into());
        }
        None
    };
    if (multimodal || batch.is_some())
        && (backend != "native" || (multimodal && (role != "integrated" || device_indices.len() != 1)))
    {
        return Err(io::Error::other(
            "batch/image serving requires native CUDA; images require one integrated device",
        )
        .into());
    }
    if multimodal && batch.is_some() {
        return Err(io::Error::other(
            "combined image and continuous-batch serving is not implemented",
        )
        .into());
    }
    if batch.is_some_and(|batch| {
        batch.max_sequences == 0
            || batch.max_batch_tokens == 0
            || batch.kv_pages == 0
            || batch.block_tokens == 0
            || batch.max_queued_requests == 0
    }) {
        return Err(io::Error::other("batch limits and queue capacity must be positive").into());
    }
    if let Some(directory) = autotune_cache_directory {
        #[cfg(feature = "stack-autotune")]
        ruda::runtime::tune::stack::enable_stack_autotune(
            ruda::runtime::tune::stack::StackPolicy::default(),
            Some(directory),
        )
        .map_err(|error| io::Error::other(error.to_string()))?;
        #[cfg(not(feature = "stack-autotune"))]
        {
            let _ = directory;
            return Err(io::Error::other(
                "--autotune-cache-directory requires the stack-autotune feature",
            )
            .into());
        }
    }
    let model_name = model_name.unwrap_or_else(|| {
        directory
            .file_name()
            .unwrap_or_default()
            .to_string_lossy()
            .into_owned()
    });
    let (mut state, jobs) =
        serving::channel(model_name, queue_capacity).map_err(io::Error::other)?;
    if let Some(max_output_tokens) = responses_max_output_tokens {
        let checkpoint: serde_json::Value =
            serde_json::from_slice(&std::fs::read(directory.join("config.json"))?)?;
        let text_config = checkpoint.get("text_config").unwrap_or(&checkpoint);
        let context_window = text_config
            .get("max_position_embeddings")
            .and_then(serde_json::Value::as_u64)
            .ok_or_else(|| {
                io::Error::other("Responses requires checkpoint max_position_embeddings")
            })?;
        state = state
            .with_responses(
                usize::try_from(context_window)?,
                max_output_tokens,
                multimodal,
            )
            .map_err(io::Error::other)?;
    }
    let (ready, loaded) = tokio::sync::oneshot::channel();
    let options = worker::Options {
        directory,
        model_id,
        dtype,
        devices: device_indices,
        prefill_only,
        prefill_address,
        #[cfg(feature="gpu-kv")]
        gpu_kv,
        mla_mode,
        draft_directory,
        draft_id,
        causal_draft_tokens,
        mtp,
        cache: config,
        batch,
        multimodal,
    };
    let worker_backend = backend.clone();
    let worker = std::thread::Builder::new()
        .name(format!("ruda-{backend}-inference"))
        .spawn(move || worker::run(options, &worker_backend, &libtorch_device, jobs, ready))?;
    loaded
        .await
        .map_err(io::Error::other)?
        .map_err(io::Error::other)?;
    if prefill_only {
        println!("RUDA {backend} prefill service ready ({})",if prefill_transport=="ucx" {"UCX GPU payload, TCP metadata"} else {"TCP host KV handoff"});
        tokio::task::spawn_blocking(move || {
            worker
                .join()
                .map_err(|_| io::Error::other("prefill worker panicked"))
        })
        .await
        .map_err(io::Error::other)??;
        return Ok(());
    }
    let listener = tokio::net::TcpListener::bind(&address).await?;
    println!("RUDA {backend} model serving at http://{address}/v1");
    axum::serve(listener, serving::router_with_images(state, multimodal)).await?;
    Ok(())
}
