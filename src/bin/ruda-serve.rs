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
    let mut libtorch_device = "cuda".to_string();
    let mut explicit_libtorch_device = false;
    let mut role = "integrated".to_string();
    let mut prefill_address = None;
    let mut mla_mode = DeepseekV2AttentionMode::Expanded;
    let mut draft_directory = None;
    let mut draft_id = None;
    let mut multimodal = false;
    let mut batch_sequences = None;
    let mut batch_tokens = None;
    let mut batch_pages = None;
    let mut batch_block = None;
    let mut config = PrefixCacheConfig {
        block_tokens: 64,
        device_entries: 4,
        host_bytes: 0,
        disk_directory: None,
        disk_bytes: 0,
    };
    while let Some(arg) = args.next() {
        if arg == "--help" {
            println!(
                "ruda-serve --model-path DIR --model-id EXACT-WEIGHTS-IDENTITY [--served-model-name NAME] [--backend native|libtorch (default native; libtorch feature required)] [--libtorch-device cpu|cuda|mps (default cuda)] [--devices 0,1 (Qwen3 TP / DeepSeek-V2 EP)] [--draft-model-path DIR --draft-model-id EXACT-DRAFT-IDENTITY (single-device Qwen3 DFlash/DSpark)] [--mla-mode expanded|absorbed (DeepSeek-V2 only; default expanded)] [--role integrated|prefill|decode] [--prefill-address HOST:PORT (prefill listen/decode connect, TCP host KV)] [--bind 127.0.0.1:8000] [--queue-capacity N] [--multimodal (native Qwen3.5 JPEG/PNG; single-device integrated)] [--batch-max-sequences N --batch-max-tokens N --batch-kv-pages N --batch-block-tokens N (native Qwen3.5 text; single-device integrated; no prefix/offload/image batching)] [--prefix-block-tokens N] [--prefix-device-entries N] [--kv-host-bytes N] [--kv-disk-directory DIR --kv-disk-bytes N]"
            );
            println!(
                "[structured-output feature: response_format JSON Schema/JSON object or grammar regex/Lark/llguidance; standard or continuous-batch serving]"
            );
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
            "--libtorch-device" => {
                libtorch_device = value;
                explicit_libtorch_device = true;
            }
            "--role" => role = value,
            "--prefill-address" => prefill_address = Some(value),
            "--draft-model-path" => draft_directory = Some(PathBuf::from(value)),
            "--draft-model-id" => draft_id = Some(value),
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
        })
    } else {
        if batch_tokens.is_some() || batch_pages.is_some() || batch_block.is_some() {
            return Err(io::Error::other("batch options require --batch-max-sequences").into());
        }
        None
    };
    if (multimodal || batch.is_some())
        && (backend != "native" || role != "integrated" || device_indices.len() != 1)
    {
        return Err(io::Error::other(
            "batch/image serving currently requires single-device integrated native CUDA",
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
    if batch.is_some()
        && (config.host_bytes != 0 || config.disk_bytes != 0 || config.disk_directory.is_some())
    {
        return Err(io::Error::other("batch KV offload is not implemented; host/SSD tiers remain available in standard serving").into());
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
        devices: device_indices,
        prefill_only,
        prefill_address,
        mla_mode,
        draft_directory,
        draft_id,
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
        println!("RUDA {backend} prefill service ready (TCP host KV handoff)");
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
