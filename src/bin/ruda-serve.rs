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
    let mut device_indices = vec![0usize];
    let mut backend = "native".to_string();
    let mut libtorch_device = "cuda".to_string();
    let mut explicit_libtorch_device = false;
    let mut role = "integrated".to_string();
    let mut prefill_address = None;
    let mut mla_mode = DeepseekV2AttentionMode::Expanded;
    let mut draft_directory = None;
    let mut draft_id = None;
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
                "ruda-serve --model-path DIR --model-id EXACT-WEIGHTS-IDENTITY [--served-model-name NAME] [--backend native|libtorch (default native; libtorch feature required)] [--libtorch-device cpu|cuda|mps (default cuda)] [--devices 0,1 (Qwen3 TP / DeepSeek-V2 EP)] [--draft-model-path DIR --draft-model-id EXACT-DRAFT-IDENTITY (single-device Qwen3 DFlash/DSpark)] [--mla-mode expanded|absorbed (DeepSeek-V2 only; default expanded)] [--role integrated|prefill|decode] [--prefill-address HOST:PORT (prefill listen/decode connect, TCP host KV)] [--bind 127.0.0.1:8000] [--queue-capacity N] [--prefix-block-tokens N] [--prefix-device-entries N] [--kv-host-bytes N] [--kv-disk-directory DIR --kv-disk-bytes N]"
            );
            return Ok(());
        }
        let value = args
            .next()
            .ok_or_else(|| io::Error::other(format!("missing value for {arg}")))?;
        match arg.as_str() {
            "--model-path" => directory = Some(PathBuf::from(value)),
            "--model-id" => model_id = Some(value),
            "--served-model-name" => model_name = Some(value),
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
    let model_name = model_name.unwrap_or_else(|| {
        directory
            .file_name()
            .unwrap_or_default()
            .to_string_lossy()
            .into_owned()
    });
    let (state, jobs) = serving::channel(model_name, queue_capacity).map_err(io::Error::other)?;
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
    axum::serve(listener, serving::router(state)).await?;
    Ok(())
}
