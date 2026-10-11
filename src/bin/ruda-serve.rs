use half::bf16;
use ruda_tensor_device::cuda::{Cuda, CudaDevice};
use rullm::prefix_cache::PrefixCacheConfig;
use rullm::serving;
use rullm::{
    CausalModelLimits, DeepseekV2AttentionMode, load_huggingface_deepseek_v2,
    load_huggingface_llama, load_huggingface_qwen2, load_huggingface_qwen3,
    load_huggingface_qwen35_text,
};
use std::{error::Error, io, path::PathBuf};

type B = Cuda<bf16, i32>;

#[tokio::main(flavor = "multi_thread", worker_threads = 2)]
async fn main() -> Result<(), Box<dyn Error>> {
    let mut args = std::env::args().skip(1);
    let mut directory = None;
    let mut model_name = None;
    let mut model_id = None;
    let mut address = "127.0.0.1:8000".to_string();
    let mut queue_capacity = 64;
    let mut device_indices = vec![0usize];
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
                "ruda-serve --model-path DIR --model-id EXACT-WEIGHTS-IDENTITY [--served-model-name NAME] [--devices 0,1 (Qwen3 TP / DeepSeek-V2 EP)] [--draft-model-path DIR --draft-model-id EXACT-DRAFT-IDENTITY (single-device Qwen3 DFlash/DSpark)] [--mla-mode expanded|absorbed (DeepSeek-V2 only; default expanded)] [--role integrated|prefill|decode] [--prefill-address HOST:PORT (prefill listen/decode connect, TCP host KV)] [--bind 127.0.0.1:8000] [--queue-capacity N] [--prefix-block-tokens N] [--prefix-device-entries N] [--kv-host-bytes N] [--kv-disk-directory DIR --kv-disk-bytes N]"
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
    let worker = std::thread::Builder::new()
        .name("ruda-native-inference".into())
        .spawn(move || {
            let mut ready = Some(ready);
            let result = (|| -> Result<(), String> {
                let device = CudaDevice::new(device_indices[0]);
                let tokenizer = if prefill_only {
                    None
                } else {
                    Some(
                        tokenizers::Tokenizer::from_file(directory.join("tokenizer.json"))
                            .map_err(|e| e.to_string())?,
                    )
                };
                let tokenizer_config = if prefill_only {
                    serde_json::Value::Null
                } else {
                    serving::tokenizer_config(&directory)?
                };
                let source: serde_json::Value = serde_json::from_slice(
                    &std::fs::read(directory.join("config.json")).map_err(|e| e.to_string())?,
                )
                .map_err(|e| e.to_string())?;
                if mla_mode != DeepseekV2AttentionMode::Expanded
                    && source.get("model_type").and_then(serde_json::Value::as_str)
                        != Some("deepseek_v2")
                {
                    return Err("absorbed MLA requires a DeepSeek-V2 adapter".into());
                }
                if draft_directory.is_some()
                    && (device_indices.len() != 1
                        || source.get("model_type").and_then(serde_json::Value::as_str)
                            != Some("qwen3"))
                {
                    return Err(
                        "DFlash/DSpark currently requires a single-device Qwen3 target".into(),
                    );
                }
                let model_id = if source.get("model_type").and_then(serde_json::Value::as_str)
                    == Some("deepseek_v2")
                {
                    format!("{model_id}:mla-{mla_mode:?}")
                } else {
                    model_id
                };
                let model_id = if let Some(draft_id) = &draft_id {
                    format!("{model_id}:draft-{draft_id}")
                } else {
                    model_id
                };
                macro_rules! launch {
                    ($model:expr,$limits:expr,$eos:expr) => {{
                        let model = $model;
                        let limits = $limits;
                        let eos = $eos;
                        let on_ready = || {
                            ready
                                .take()
                                .expect("worker readiness")
                                .send(Ok::<(), String>(()))
                                .is_ok()
                        };
                        if prefill_only {
                            rullm::disaggregated::run_prefill_service::<B, _>(
                                model,
                                device,
                                limits,
                                model_id,
                                config,
                                prefill_address.as_deref().expect("prefill address"),
                                on_ready,
                            )
                            .map_err(|e| e.to_string())
                        } else {
                            serving::run_native_with_prefill::<B, _>(
                                model,
                                device,
                                limits,
                                eos,
                                tokenizer.expect("decode tokenizer"),
                                tokenizer_config,
                                model_id,
                                config,
                                jobs,
                                on_ready,
                                prefill_address
                                    .map(|address| rullm::disaggregated::RemotePrefill { address }),
                            )
                        }
                    }};
                }
                match source.get("model_type").and_then(serde_json::Value::as_str) {
                    Some("deepseek_v2") => {
                        #[cfg(feature = "tensor-parallel")]
                        if device_indices.len() > 1 {
                            let devices = device_indices
                                .iter()
                                .map(|&index| CudaDevice::new(index))
                                .collect();
                            let loaded = rullm::load_huggingface_deepseek_v2_expert_parallel::<B>(
                                &directory, devices,
                            )
                            .map_err(|e| e.to_string())?;
                            let limits = loaded.model.config.limits();
                            return launch!(
                                loaded.model.with_attention_mode(mla_mode),
                                limits,
                                loaded.eos_token_ids
                            );
                        }
                        let loaded = load_huggingface_deepseek_v2::<B>(&directory, &device)
                            .map_err(|e| e.to_string())?;
                        let limits = loaded.model.config.limits();
                        launch!(
                            loaded.model.with_attention_mode(mla_mode),
                            limits,
                            loaded.eos_token_ids
                        )
                    }
                    Some("qwen3") => {
                        #[cfg(feature = "tensor-parallel")]
                        if device_indices.len() > 1 {
                            let devices = device_indices
                                .iter()
                                .map(|&index| CudaDevice::new(index))
                                .collect();
                            let loaded = rullm::load_huggingface_qwen3_tensor_parallel::<B>(
                                &directory, devices,
                            )
                            .map_err(|e| e.to_string())?;
                            let limits = loaded.model.config.limits();
                            return launch!(loaded.model, limits, loaded.eos_token_ids);
                        }
                        let loaded = load_huggingface_qwen3::<B>(&directory, &device)
                            .map_err(|e| e.to_string())?;
                        let limits = loaded.model.config.limits();
                        if let Some(directory) = draft_directory {
                            let model =
                                rullm::load_huggingface_qwen3_draft(loaded.model, directory)
                                    .map_err(|e| e.to_string())?;
                            return launch!(model, limits, loaded.eos_token_ids);
                        }
                        launch!(loaded.model, limits, loaded.eos_token_ids)
                    }
                    Some("qwen2") => {
                        if device_indices.len() > 1 {
                            return Err("Qwen2 tensor parallel is not implemented".into());
                        }
                        let loaded = load_huggingface_qwen2::<B>(&directory, &device)
                            .map_err(|e| e.to_string())?;
                        launch!(
                            loaded.model,
                            CausalModelLimits::from(&loaded.config),
                            loaded.default_eos_token_ids
                        )
                    }
                    Some("llama") => {
                        if device_indices.len() > 1 {
                            return Err("Llama tensor parallel is not implemented".into());
                        }
                        let loaded = load_huggingface_llama::<B>(&directory, &device)
                            .map_err(|e| e.to_string())?;
                        launch!(
                            loaded.model,
                            CausalModelLimits::from(&loaded.config),
                            loaded.default_eos_token_ids
                        )
                    }
                    Some("qwen3_5" | "qwen3_5_text") => {
                        if device_indices.len() > 1 {
                            return Err("Qwen3.5 tensor parallel is not implemented".into());
                        }
                        let loaded = load_huggingface_qwen35_text::<B>(&directory, &device)
                            .map_err(|e| e.to_string())?;
                        let limits = CausalModelLimits {
                            vocab_size: loaded.model.config.vocab_size,
                            max_sequence_length: loaded.model.config.max_position_embeddings,
                        };
                        let eos = vec![loaded.model.config.eos_token_id];
                        launch!(loaded.model, limits, eos)
                    }
                    other => Err(format!("no native server model adapter for {other:?}")),
                }
            })();
            if let Err(e) = result {
                if let Some(ready) = ready {
                    let _ = ready.send(Err(e.clone()));
                }
                eprintln!("native inference worker: {e}");
            }
        })?;
    loaded
        .await
        .map_err(io::Error::other)?
        .map_err(io::Error::other)?;
    if prefill_only {
        println!("RUDA native prefill service ready (TCP host KV handoff)");
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
    println!("RUDA native model serving at http://{address}/v1");
    axum::serve(listener, serving::router(state)).await?;
    Ok(())
}
