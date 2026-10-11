use super::{progress, *};
use crate::{Qwen35ImageProcessor, Qwen35MultimodalModel, Qwen35TextModel};
use base64::Engine;
use ruda::runtime::server::ComputeServer;
use ruda_tensor::DeviceOps;
use ruda_tensor_device::{BoolElement, DeviceBackend, DeviceRuntime, FloatElement, IntElement};

pub(super) fn validate_parts(content: Option<&Value>) -> Result<(), String> {
    let Some(parts) = content.and_then(Value::as_array) else {
        return Ok(());
    };
    if parts.is_empty() {
        return Err("message content parts must not be empty".into());
    }
    for part in parts {
        match part.get("type").and_then(Value::as_str) {
            Some("text") if part.get("text").is_some_and(Value::is_string) => (),
            Some("image_url") => {
                let url = part
                    .pointer("/image_url/url")
                    .and_then(Value::as_str)
                    .ok_or("image_url requires a URL string")?;
                if !(url.starts_with("data:image/")
                    || url.starts_with("https://")
                    || url.starts_with("http://"))
                {
                    return Err("image_url requires HTTP(S) or a base64 image data URL".into());
                }
                if part
                    .pointer("/image_url/detail")
                    .and_then(Value::as_str)
                    .is_some_and(|v| v != "auto")
                {
                    return Err(
                        "image detail must be auto; checkpoint preprocessing controls resolution"
                            .into(),
                    );
                }
            }
            _ => return Err("content parts must be text or image_url".into()),
        }
    }
    Ok(())
}
pub(super) fn has_images(request: &GenerateRequest) -> bool {
    request
        .messages
        .iter()
        .flatten()
        .filter_map(|m| m.get("content").and_then(Value::as_array))
        .flatten()
        .any(|part| part.get("type").and_then(Value::as_str) == Some("image_url"))
}
pub(super) fn normalize_messages(messages: &mut [Value]) -> Result<(), String> {
    for message in messages {
        validate_parts(message.get("content"))?;
        let Some(parts) = message.get_mut("content").and_then(Value::as_array_mut) else {
            continue;
        };
        if parts
            .iter()
            .all(|part| part.get("type").and_then(Value::as_str) == Some("text"))
        {
            let text = parts
                .iter()
                .filter_map(|part| part.get("text").and_then(Value::as_str))
                .collect::<String>();
            message["content"] = json!(text);
        } else {
            for part in parts {
                if part.get("type").and_then(Value::as_str) == Some("image_url") {
                    let url = part
                        .pointer("/image_url/url")
                        .cloned()
                        .ok_or("image URL is missing")?;
                    part["type"] = json!("image");
                    part["image"] = url;
                }
            }
        }
    }
    Ok(())
}
fn load_images(request: &GenerateRequest) -> Result<Vec<Vec<u8>>, String> {
    let client = reqwest::blocking::Client::builder()
        .build()
        .map_err(|e| e.to_string())?;
    request
        .messages
        .iter()
        .flatten()
        .filter_map(|m| m.get("content").and_then(Value::as_array))
        .flatten()
        .filter(|part| part.get("type").and_then(Value::as_str) == Some("image_url"))
        .map(|part| {
            let url = part
                .pointer("/image_url/url")
                .and_then(Value::as_str)
                .ok_or("image URL is missing")?;
            if let Some(data) = url.strip_prefix("data:") {
                let (mime, body) = data.split_once(',').ok_or("invalid image data URL")?;
                if !matches!(mime, "image/png;base64" | "image/jpeg;base64") {
                    return Err("data URLs must contain base64 PNG or JPEG".into());
                }
                base64::engine::general_purpose::STANDARD
                    .decode(body)
                    .map_err(|e| e.to_string())
            } else {
                client
                    .get(url)
                    .send()
                    .and_then(reqwest::blocking::Response::error_for_status)
                    .and_then(reqwest::blocking::Response::bytes)
                    .map(|b| b.to_vec())
                    .map_err(|e| e.to_string())
            }
        })
        .collect()
}

fn execute<B: Backend, M: CausalModel<B>>(
    job: Job,
    model: &M,
    device: &B::Device,
    tokenizer: &tokenizers::Tokenizer,
    limits: CausalModelLimits,
    eos: &[i32],
    ids: Vec<i32>,
    model_id: &str,
    prefix: Option<crate::prefix_cache::PrefilledPrompt<B, M>>,
) -> Result<(), String> {
    let (greedy, sampled) = progress::configs(&job.request, eos);
    if let Err(error) = crate::generation::validate_generation(&limits, &ids, &greedy) {
        let _ = job
            .ready
            .send(Err((StatusCode::BAD_REQUEST, error.to_string())));
        return Ok(());
    }
    let constraint = match structured::Decoder::new(&job.request, tokenizer, eos) {
        Ok(constraint) => constraint,
        Err(error) => {
            let _ = job.ready.send(Err((StatusCode::BAD_REQUEST, error)));
            return Ok(());
        }
    };
    let control = GenerationControl::default();
    let session = match (job.request.temperature == Some(0.0), prefix) {
        (true, Some(prefix)) => CausalGenerationSession::new_greedy_prefilled(
            model, limits, &ids, greedy, &control, device, model_id, prefix,
        ),
        (false, Some(prefix)) => CausalGenerationSession::new_sampled_prefilled(
            model, limits, &ids, sampled, &control, device, model_id, prefix,
        ),
        (true, None) => {
            CausalGenerationSession::new_greedy(model, limits, &ids, greedy, &control, device)
        }
        (false, None) => {
            CausalGenerationSession::new_sampled(model, limits, &ids, sampled, &control, device)
        }
    };
    let mut session = match session {
        Ok(session) => session,
        Err(error) => {
            let _ = job
                .ready
                .send(Err((StatusCode::INTERNAL_SERVER_ERROR, error.to_string())));
            return Ok(());
        }
    };
    let Some(job) = progress::accept(job) else {
        return Ok(());
    };
    if let Err(error) =
        generate_response::<B, M>(&job, &mut session, tokenizer, device, ids.len(), constraint)
    {
        let _ = job.events.blocking_send(Err(error));
    }
    Ok(())
}

/// Loads vision/text weights once; image requests run the actual image-prefill model.
pub fn run_qwen35_multimodal<R, F, I, BT>(
    model: Qwen35MultimodalModel<DeviceBackend<R, F, I, BT>>,
    processor: Qwen35ImageProcessor,
    device: R::Device,
    tokenizer: tokenizers::Tokenizer,
    tokenizer_config: Value,
    model_id: String,
    cache_config: PrefixCacheConfig,
    mut jobs: mpsc::Receiver<Job>,
    on_ready: impl FnOnce() -> bool,
) -> Result<(), String>
where
    R: DeviceRuntime,
    R::Server: ComputeServer,
    R::Device: DeviceOps,
    F: FloatElement,
    I: IntElement,
    BT: BoolElement,
{
    processor
        .validate_vision(model.vision.config())
        .map_err(|e| e.to_string())?;
    let limits = CausalModelLimits {
        vocab_size: model.text.config.vocab_size,
        max_sequence_length: model.text.config.max_position_embeddings,
    };
    let eos = vec![model.text.config.eos_token_id];
    let mut cache = TieredPrefixCache::<
        DeviceBackend<R, F, I, BT>,
        Qwen35TextModel<DeviceBackend<R, F, I, BT>>,
    >::new(model_id.clone(), limits, cache_config)
    .map_err(|e| e.to_string())?;
    if !on_ready() {
        return Ok(());
    }
    while let Some(job) = jobs.blocking_recv() {
        if job.events.is_closed() {
            continue;
        }
        let ids = match progress::prompt_ids(&job, &tokenizer, &tokenizer_config) {
            Ok(ids) => ids,
            Err(error) => {
                let _ = job.ready.send(Err((StatusCode::BAD_REQUEST, error)));
                continue;
            }
        };
        let eos = match progress::eos_ids(job.chat, &eos, &tokenizer, &tokenizer_config) {
            Ok(eos) => eos,
            Err(error) => {
                let _ = job.ready.send(Err((StatusCode::BAD_REQUEST, error)));
                continue;
            }
        };
        if has_images(&job.request) {
            let prepared = load_images(&job.request)
                .and_then(|bodies| {
                    processor
                        .preprocess_encoded(&bodies)
                        .map_err(|e| e.to_string())
                })
                .and_then(|images| {
                    model
                        .prepare_image_generation(&ids, images)
                        .map_err(|e| e.to_string())
                });
            match prepared {
                Ok((conditioned, ids)) => execute(
                    job,
                    &conditioned,
                    &device,
                    &tokenizer,
                    limits,
                    &eos,
                    ids,
                    &model_id,
                    None,
                )?,
                Err(error) => {
                    let _ = job.ready.send(Err((StatusCode::BAD_REQUEST, error)));
                }
            }
        } else {
            let (greedy, _) = progress::configs(&job.request, &eos);
            if let Err(error) = crate::generation::validate_generation(&limits, &ids, &greedy) {
                let _ = job
                    .ready
                    .send(Err((StatusCode::BAD_REQUEST, error.to_string())));
                continue;
            }
            let prefix = if job.request.max_tokens.or(job.request.max_completion_tokens) == Some(0)
            {
                Ok(None)
            } else {
                cache
                    .prefill(&model.text, &ids, &device)
                    .map(Some)
                    .map_err(|e| e.to_string())
            };
            match prefix {
                Ok(prefix) => execute(
                    job,
                    &model.text,
                    &device,
                    &tokenizer,
                    limits,
                    &eos,
                    ids,
                    &model_id,
                    prefix,
                )?,
                Err(error) => {
                    let _ = job
                        .ready
                        .send(Err((StatusCode::INTERNAL_SERVER_ERROR, error)));
                }
            }
        }
    }
    Ok(())
}
