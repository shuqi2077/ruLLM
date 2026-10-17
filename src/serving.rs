use crate::prefix_cache::{PrefixCacheConfig, TieredPrefixCache};
use crate::{
    CausalGenerationSession, CausalModel, CausalModelLimits, GenerationCacheContinuation,
    GenerationControl, GenerationFinishReason, GreedyGenerationConfig, SamplingConfig,
    SamplingGenerationConfig,
};
use axum::{
    Json, Router,
    extract::State,
    http::StatusCode,
    response::{
        IntoResponse, Response,
        sse::{Event, Sse},
    },
    routing::{get, post},
};
use ruda_tensor::api::backend::Backend;
use serde::Deserialize;
use serde_json::{Value, json};
use std::{
    collections::BTreeMap,
    convert::Infallible,
    path::Path,
    sync::{
        Arc,
        atomic::{AtomicU64, Ordering},
    },
    time::{SystemTime, UNIX_EPOCH},
};
use tokio::sync::{mpsc, oneshot};

mod batch;
mod images;
mod metrics;
mod progress;
mod responses;
pub mod structured;
mod tool_calls;
pub use batch::{BatchServingConfig, run_continuous_batch, run_continuous_batch_with_prefix,run_continuous_batch_with_prefill};
#[cfg(feature="gpu-kv")]
pub use batch::run_continuous_batch_with_gpu_prefill;
pub use images::run_qwen35_multimodal;

#[derive(Clone, Deserialize)]
pub struct GenerateRequest {
    pub model: String,
    #[serde(default)]
    pub prompt: Option<String>,
    #[serde(default)]
    pub messages: Option<Vec<Value>>,
    #[serde(default)]
    pub max_tokens: Option<usize>,
    #[serde(default)]
    pub max_completion_tokens: Option<usize>,
    #[serde(default)]
    pub stream: bool,
    #[serde(default)]
    pub stream_options: Option<StreamOptions>,
    #[serde(default)]
    pub temperature: Option<f64>,
    #[serde(default)]
    pub top_p: Option<f64>,
    #[serde(default)]
    pub top_k: usize,
    #[serde(default)]
    pub seed: Option<u64>,
    #[serde(default)]
    pub stop: Option<Value>,
    #[serde(default)]
    pub n: Option<usize>,
    #[serde(default)]
    pub tools: Option<Vec<Value>>,
    #[serde(default)]
    pub tool_choice: Option<Value>,
    #[serde(default)]
    pub response_format: Option<structured::ResponseFormat>,
    #[serde(default)]
    pub grammar: Option<structured::Grammar>,
    #[serde(flatten)]
    pub template_options: BTreeMap<String, Value>,
}

#[derive(Clone, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct StreamOptions {
    #[serde(default)]
    pub include_usage: bool,
}

impl GenerateRequest {
    fn validate(&self, name: &str, chat: bool) -> Result<(), String> {
        if self.model != name {
            return Err(format!("model {} is not loaded", self.model));
        }
        if self.n.is_some_and(|n| n != 1) {
            return Err("this engine currently supports n=1".into());
        }
        if self.max_tokens.is_some() && self.max_completion_tokens.is_some() {
            return Err("supply max_tokens or max_completion_tokens, not both".into());
        }
        if self
            .temperature
            .is_some_and(|v| !v.is_finite() || v < 0.0 || v > 2.0)
            || self
                .top_p
                .is_some_and(|v| !v.is_finite() || v <= 0.0 || v > 1.0)
        {
            return Err("invalid temperature or top_p".into());
        }
        if chat {
            if self.prompt.is_some() || self.messages.as_ref().is_none_or(|m| m.is_empty()) {
                return Err("chat requires a nonempty messages array, not prompt".into());
            }
            for message in self.messages.as_ref().expect("checked messages") {
                if !matches!(
                    message.get("role").and_then(Value::as_str),
                    Some("system" | "user" | "assistant" | "tool")
                ) || message
                    .get("content")
                    .is_some_and(|v| !v.is_string() && !v.is_null() && !v.is_array())
                {
                    return Err("messages require supported roles and text or content parts".into());
                }
                images::validate_parts(message.get("content"))?;
                if message.get("content").is_none_or(Value::is_null)
                    && message.get("tool_calls").is_none()
                {
                    return Err("message requires text content or tool_calls".into());
                }
            }
        } else if self.prompt.is_none() || self.messages.is_some() || self.tools.is_some() {
            return Err("completions requires a string prompt and no messages/tools".into());
        }
        for key in self.template_options.keys() {
            if key != "enable_thinking" && key != "parallel_tool_calls" {
                return Err(format!("unsupported request field {key}"));
            }
        }
        if self.template_options.get("parallel_tool_calls").is_some_and(|value| !value.is_boolean()) {
            return Err("parallel_tool_calls must be a boolean".into());
        }
        if self
            .template_options
            .get("enable_thinking")
            .is_some_and(|v| !v.is_boolean())
        {
            return Err("enable_thinking must be a boolean".into());
        }
        if let Some(tools) = &self.tools {
            for tool in tools {
                if tool.get("type").and_then(Value::as_str) != Some("function")
                    || tool
                        .pointer("/function/name")
                        .and_then(Value::as_str)
                        .is_none_or(str::is_empty)
                {
                    return Err("tools must contain named function definitions".into());
                }
            }
        }
        if let Some(choice) = &self.tool_choice {
            let valid = matches!(choice.as_str(), Some("auto" | "none" | "required"))
                || (choice.get("type").and_then(Value::as_str) == Some("function")
                    && choice
                        .pointer("/function/name")
                        .and_then(Value::as_str)
                        .is_some());
            if !valid {
                return Err("invalid tool_choice".into());
            }
            if choice.as_str() != Some("none") && self.tools.as_ref().is_none_or(|v| v.is_empty()) {
                return Err("tool_choice requires tools".into());
            }
        }
        self.stops()?;
        structured::validate(self)?;
        Ok(())
    }

    fn stops(&self) -> Result<Vec<String>, String> {
        let values = match &self.stop {
            None | Some(Value::Null) => vec![],
            Some(Value::String(s)) => vec![s.clone()],
            Some(Value::Array(v)) => v
                .iter()
                .map(|s| {
                    s.as_str()
                        .map(str::to_owned)
                        .ok_or_else(|| "stop array must contain strings".to_string())
                })
                .collect::<Result<Vec<_>, _>>()?,
            _ => return Err("stop must be a string or string array".into()),
        };
        if values.iter().any(String::is_empty) {
            return Err("stop strings must not be empty".into());
        }
        Ok(values)
    }

    fn tools_enabled(&self) -> bool {
        self.tool_choice.as_ref().and_then(Value::as_str) != Some("none")
            && self.tools.as_ref().is_some_and(|t| !t.is_empty())
    }

    fn parallel_tool_calls(&self) -> Option<bool> {
        self.template_options.get("parallel_tool_calls").and_then(Value::as_bool)
    }
}

pub fn render_chat(
    template: &str,
    config: &Value,
    request: &GenerateRequest,
) -> Result<String, String> {
    if request.tools_enabled() && !template.contains("<tool_call>") {
        return Err("checkpoint chat template has no supported Qwen function-call format".into());
    }
    let mut environment = minijinja::Environment::new();
    environment.set_unknown_method_callback(minijinja_contrib::pycompat::unknown_method_callback);
    environment.add_function(
        "raise_exception",
        |message: String| -> Result<String, minijinja::Error> {
            Err(minijinja::Error::new(
                minijinja::ErrorKind::InvalidOperation,
                message,
            ))
        },
    );
    environment
        .add_template("chat", template)
        .map_err(|e| e.to_string())?;
    let mut messages = request.messages.clone().ok_or("messages are required")?;
    images::normalize_messages(&mut messages)?;
    for message in &mut messages {
        if let Some(calls) = message.get_mut("tool_calls").and_then(Value::as_array_mut) {
            for call in calls {
                if let Some(arguments) = call.pointer_mut("/function/arguments") {
                    if let Some(text) = arguments.as_str() {
                        *arguments = serde_json::from_str(text)
                            .map_err(|e| format!("invalid tool arguments: {e}"))?;
                    }
                }
            }
        }
    }
    let mut context = config.as_object().cloned().unwrap_or_default();
    context.insert("messages".into(), json!(messages));
    context.insert(
        "tools".into(),
        if request.tools_enabled() {
            json!(request.tools)
        } else {
            Value::Null
        },
    );
    context.insert("add_generation_prompt".into(), json!(true));
    for (key, value) in &request.template_options {
        context.insert(key.clone(), value.clone());
    }
    environment
        .get_template("chat")
        .map_err(|e| e.to_string())?
        .render(Value::Object(context))
        .map_err(|e| e.to_string())
}

#[derive(Clone)]
pub struct ServerState {
    model: String,
    jobs: mpsc::Sender<Job>,
    sequence: Arc<AtomicU64>,
    metrics: Arc<metrics::Metrics>,
    responses_config: Option<responses::Config>,
}

pub struct Job {
    request: GenerateRequest,
    chat: bool,
    id: String,
    created: u64,
    ready: oneshot::Sender<Result<(), (StatusCode, String)>>,
    events: mpsc::Sender<Result<Value, String>>,
    responses: Option<Arc<std::sync::Mutex<responses::Output>>>,
}

pub fn channel(
    model: String,
    queue_capacity: usize,
) -> Result<(ServerState, mpsc::Receiver<Job>), String> {
    if queue_capacity == 0 {
        return Err("queue capacity must be positive".into());
    }
    let (jobs, receiver) = mpsc::channel(queue_capacity);
    Ok((
        ServerState {
            model,
            jobs,
            sequence: Arc::new(AtomicU64::new(0)),
            metrics: Arc::new(metrics::Metrics::default()),
            responses_config: None,
        },
        receiver,
    ))
}

pub fn router(state: ServerState) -> Router {
    router_with_images(state, false)
}

pub fn router_with_images(state: ServerState, image_inputs: bool) -> Router {
    let metrics = state.metrics.clone();
    let chat_route = if image_inputs {
        post(chat).layer(axum::extract::DefaultBodyLimit::disable())
    } else {
        post(chat)
    };
    let responses_route = if image_inputs {
        post(responses::create).layer(axum::extract::DefaultBodyLimit::disable())
    } else {
        post(responses::create)
    };
    let routes = Router::new()
        .route("/v1/models", get(models))
        .route("/v1/codex/models", get(responses::models))
        .route("/v1/responses", responses_route)
        .route("/v1/completions", post(completions))
        .route("/v1/chat/completions", chat_route)
        .route("/metrics", get(metrics::endpoint))
        .route(
            "/ui",
            get(|| async { axum::response::Html(include_str!("serving/ui.html")) }),
        );
    routes
        .layer(axum::middleware::from_fn_with_state(
            metrics,
            metrics::record,
        ))
        .with_state(state)
}

async fn models(State(state): State<ServerState>) -> Json<Value> {
    Json(
        json!({"object":"list","data":[{"id":state.model,"object":"model","created":0,"owned_by":"ruda"}]}),
    )
}
async fn completions(
    State(state): State<ServerState>,
    request: Result<Json<GenerateRequest>, axum::extract::rejection::JsonRejection>,
) -> Response {
    match request {
        Ok(Json(request)) => submit(state, request, false).await,
        Err(e) => api_error(StatusCode::BAD_REQUEST, e.body_text()),
    }
}
async fn chat(
    State(state): State<ServerState>,
    request: Result<Json<GenerateRequest>, axum::extract::rejection::JsonRejection>,
) -> Response {
    match request {
        Ok(Json(request)) => submit(state, request, true).await,
        Err(e) => api_error(StatusCode::BAD_REQUEST, e.body_text()),
    }
}
fn api_error(status: StatusCode, message: String) -> Response {
    (status,Json(json!({"error":{"message":message,"type":if status.is_client_error(){"invalid_request_error"}else{"server_error"},"param":null,"code":null}}))).into_response()
}
async fn submit(state: ServerState, request: GenerateRequest, chat: bool) -> Response {
    submit_wire(state, request, chat, None).await
}
async fn submit_wire(
    state: ServerState,
    request: GenerateRequest,
    chat: bool,
    responses: Option<responses::Output>,
) -> Response {
    if let Err(e) = request.validate(&state.model, chat) {
        return api_error(StatusCode::BAD_REQUEST, e);
    }
    let stream = request.stream;
    let responses_wire = responses.is_some();
    let serial = state.sequence.fetch_add(1, Ordering::Relaxed);
    let created = SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .unwrap_or_default()
        .as_secs();
    let id = format!(
        "{}-{}-{created}-{serial}",
        if responses_wire {
            "resp"
        } else if chat {
            "chatcmpl"
        } else {
            "cmpl"
        },
        std::process::id()
    );
    let (ready, prepared) = oneshot::channel();
    let (events, mut receiver) = mpsc::channel(64);
    if state
        .jobs
        .send(Job {
            request,
            chat,
            id,
            created,
            ready,
            events,
            responses: responses.map(|output| Arc::new(std::sync::Mutex::new(output))),
        })
        .await
        .is_err()
    {
        return api_error(
            StatusCode::SERVICE_UNAVAILABLE,
            "native model worker is unavailable".into(),
        );
    }
    match prepared.await {
        Ok(Ok(())) => (),
        Ok(Err((status, e))) => return api_error(status, e),
        Err(_) => {
            return api_error(
                StatusCode::INTERNAL_SERVER_ERROR,
                "native model worker failed".into(),
            );
        }
    }
    if stream {
        let output = async_stream::stream! {
            while let Some(event)=receiver.recv().await {
                let value=match event {Ok(value)=>value,Err(message)=>if responses_wire {json!({"type":"response.failed","response":{"status":"failed","error":{"message":message,"code":"server_error"}}})}else{json!({"error":{"message":message,"type":"server_error"}})}};
                let mut event = Event::default();
                if responses_wire {if let Some(kind)=value.get("type").and_then(Value::as_str){event=event.event(kind);}}
                yield Ok::<Event,Infallible>(event.data(value.to_string()));
            }
            if !responses_wire {yield Ok::<Event,Infallible>(Event::default().data("[DONE]"));}
        };
        Sse::new(output).into_response()
    } else {
        match receiver.recv().await {
            Some(Ok(result)) => Json(result).into_response(),
            Some(Err(e)) => api_error(StatusCode::INTERNAL_SERVER_ERROR, e),
            None => api_error(
                StatusCode::INTERNAL_SERVER_ERROR,
                "model worker ended without a result".into(),
            ),
        }
    }
}

fn chunk(job: &Job, choice: Value, usage: Option<Value>) -> Value {
    let mut value = json!({"id":job.id,"object":if job.chat{"chat.completion.chunk"}else{"text_completion"},
        "created":job.created,"model":job.request.model,"choices":[choice]});
    if job.chat {
        value["usage"] = usage.unwrap_or(Value::Null);
    }
    value
}
fn text_chunk(job: &Job, text: &str) -> Value {
    chunk(
        job,
        if job.chat {
            json!({"index":0,"delta":{"content":text},"logprobs":null,"finish_reason":null})
        } else {
            json!({"index":0,"text":text,"logprobs":null,"finish_reason":null})
        },
        None,
    )
}

pub fn parse_tool_calls(
    text: &str,
    tools: &[Value],
    id: &str,
) -> Result<(String, Vec<Value>), String> {
    let mut remaining = text;
    let mut content = String::new();
    let mut calls = Vec::new();
    while let Some(start) = remaining.find("<tool_call>") {
        content.push_str(&remaining[..start]);
        let tail = &remaining[start + "<tool_call>".len()..];
        let end = tool_calls::body_end(tail, true)?.ok_or("unterminated model tool call")?;
        let body = tail[..end].trim();
        let call: Value = if body.starts_with("<function=") {
            let (name, arguments) =
                tool_calls::xml_prefix(body, tools, true)?.ok_or("incomplete XML tool call")?;
            json!({"name":name,"arguments":serde_json::from_str::<Value>(&arguments).map_err(|e|e.to_string())?})
        } else {
            serde_json::from_str(body).map_err(|e| format!("invalid model tool call: {e}"))?
        };
        let name = call
            .get("name")
            .and_then(Value::as_str)
            .ok_or("model tool call has no name")?;
        if !tools
            .iter()
            .any(|tool| tool.pointer("/function/name").and_then(Value::as_str) == Some(name))
        {
            return Err(format!("model selected an undeclared tool {name}"));
        }
        let arguments = call
            .get("arguments")
            .ok_or("model tool call has no arguments")?;
        let arguments = if let Some(s) = arguments.as_str() {
            let _: Value =
                serde_json::from_str(s).map_err(|e| format!("invalid model argument JSON: {e}"))?;
            s.to_owned()
        } else {
            serde_json::to_string(arguments).map_err(|e| e.to_string())?
        };
        calls.push(json!({"id":format!("call-{id}-{}",calls.len()),"type":"function","function":{"name":name,"arguments":arguments}}));
        remaining = &tail[end + "</tool_call>".len()..];
    }
    content.push_str(remaining);
    Ok((content, calls))
}

fn safe_text_end(text: &str, stops: &[String]) -> usize {
    let held = stops
        .iter()
        .flat_map(|stop| {
            stop.char_indices()
                .map(|(i, _)| i)
                .skip(1)
                .filter(move |&i| text.ends_with(&stop[..i]))
        })
        .max()
        .unwrap_or(0);
    text.len() - held
}

fn ignored_tool_tokens(
    tokenizer: &tokenizers::Tokenizer,
    tools: bool,
) -> std::collections::BTreeSet<u32> {
    if !tools {
        return Default::default();
    }
    tokenizer
        .get_added_tokens_decoder()
        .into_iter()
        .filter_map(|(id, token)| {
            (token.special && token.content != "<tool_call>" && token.content != "</tool_call>")
                .then_some(id)
        })
        .collect()
}

pub fn run_native<B: Backend, M: CausalModel<B>>(
    model: M,
    device: B::Device,
    limits: CausalModelLimits,
    eos: Vec<i32>,
    tokenizer: tokenizers::Tokenizer,
    tokenizer_config: Value,
    model_id: String,
    cache_config: PrefixCacheConfig,
    jobs: mpsc::Receiver<Job>,
    on_ready: impl FnOnce() -> bool,
) -> Result<(), String>
where
    M::Cache: Clone + GenerationCacheContinuation<B>,
{
    run_native_with_prefill(
        model,
        device,
        limits,
        eos,
        tokenizer,
        tokenizer_config,
        model_id,
        cache_config,
        jobs,
        on_ready,
        None,
    )
}

pub fn run_native_with_prefill<B: Backend, M: CausalModel<B>>(
    model: M,
    device: B::Device,
    limits: CausalModelLimits,
    eos: Vec<i32>,
    tokenizer: tokenizers::Tokenizer,
    tokenizer_config: Value,
    model_id: String,
    cache_config: PrefixCacheConfig,
    mut jobs: mpsc::Receiver<Job>,
    on_ready: impl FnOnce() -> bool,
    remote_prefill: Option<crate::disaggregated::RemotePrefill>,
) -> Result<(), String>
where
    M::Cache: Clone + GenerationCacheContinuation<B>,
{
    let mut cache = TieredPrefixCache::<B, M>::new(model_id.clone(), limits, cache_config)
        .map_err(|e| e.to_string())?;
    if !on_ready() {
        return Ok(());
    }
    while let Some(job) = jobs.blocking_recv() {
        if job.events.is_closed() {
            continue;
        }
        if images::has_images(&job.request) {
            let _ = job.ready.send(Err((
                StatusCode::BAD_REQUEST,
                "this worker is text-only; enable --multimodal for native Qwen3.5 image input"
                    .into(),
            )));
            continue;
        }
        let prompt = if job.chat {
            let template = tokenizer_config
                .get("chat_template")
                .and_then(Value::as_str);
            match template
                .ok_or_else(|| "checkpoint has no string chat_template".to_string())
                .and_then(|t| render_chat(t, &tokenizer_config, &job.request))
            {
                Ok(prompt) => prompt,
                Err(e) => {
                    let _ = job.ready.send(Err((StatusCode::BAD_REQUEST, e)));
                    continue;
                }
            }
        } else {
            job.request.prompt.clone().expect("validated string prompt")
        };
        let ids = match tokenizer
            .encode(prompt, false)
            .map_err(|e| e.to_string())
            .and_then(|encoding| {
                encoding
                    .get_ids()
                    .iter()
                    .map(|&id| i32::try_from(id).map_err(|e| e.to_string()))
                    .collect::<Result<Vec<_>, _>>()
            }) {
            Ok(ids) => ids,
            Err(e) => {
                let _ = job.ready.send(Err((StatusCode::BAD_REQUEST, e)));
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
        let max = job
            .request
            .max_tokens
            .or(job.request.max_completion_tokens)
            .unwrap_or(16);
        let temperature = job.request.temperature.unwrap_or(1.0);
        let greedy = GreedyGenerationConfig {
            max_new_tokens: max,
            eos_token_ids: eos.clone(),
        };
        let sampled = SamplingGenerationConfig {
            max_new_tokens: max,
            eos_token_ids: eos.clone(),
            sampling: SamplingConfig {
                temperature,
                top_k: job.request.top_k,
                top_p: job.request.top_p.unwrap_or(1.0),
                seed: job.request.seed,
            },
        };
        let control = GenerationControl::default();
        if let Err(e) = crate::generation::validate_generation(&limits, &ids, &greedy) {
            let _ = job
                .ready
                .send(Err((StatusCode::BAD_REQUEST, e.to_string())));
            continue;
        }
        let constraint = match structured::Decoder::new(&job.request, &tokenizer, &eos) {
            Ok(constraint) => constraint,
            Err(error) => {
                let _ = job.ready.send(Err((StatusCode::BAD_REQUEST, error)));
                continue;
            }
        };
        let prefix = if max == 0 {
            Ok(None)
        } else if let Some(remote) = &remote_prefill {
            remote
                .prefill(&model, &model_id, limits, &ids, &device)
                .map(Some)
        } else {
            cache.prefill(&model, &ids, &device).map(Some)
        };
        let session = prefix.and_then(|prefix| match (temperature == 0.0, prefix) {
            (true, Some(prefix)) => CausalGenerationSession::new_greedy_prefilled(
                &model, limits, &ids, greedy, &control, &device, &model_id, prefix,
            ),
            (true, None) => {
                CausalGenerationSession::new_greedy(&model, limits, &ids, greedy, &control, &device)
            }
            (false, Some(prefix)) => CausalGenerationSession::new_sampled_prefilled(
                &model, limits, &ids, sampled, &control, &device, &model_id, prefix,
            ),
            (false, None) => CausalGenerationSession::new_sampled(
                &model, limits, &ids, sampled, &control, &device,
            ),
        });
        let mut session = match session {
            Ok(session) => session,
            Err(e) => {
                let _ = job
                    .ready
                    .send(Err((StatusCode::INTERNAL_SERVER_ERROR, e.to_string())));
                continue;
            }
        };
        let Job {
            ready,
            request,
            chat,
            id,
            created,
            events,
            responses,
        } = job;
        if ready.send(Ok(())).is_err() {
            continue;
        }
        let (unused_ready, _) = oneshot::channel();
        let job = Job {
            ready: unused_ready,
            request,
            chat,
            id,
            created,
            events,
            responses,
        };
        if let Err(e) = generate_response::<B, M>(
            &job,
            &mut session,
            &tokenizer,
            &device,
            ids.len(),
            constraint,
        ) {
            let _ = job.events.blocking_send(Err(e));
        }
    }
    Ok(())
}

fn generate_response<B: Backend, M: CausalModel<B>>(
    job: &Job,
    session: &mut CausalGenerationSession<'_, B, M>,
    tokenizer: &tokenizers::Tokenizer,
    device: &B::Device,
    prompt_tokens: usize,
    mut constraint: Option<structured::Decoder>,
) -> Result<(), String> {
    let stops = job.request.stops()?;
    let tools = job.request.tools_enabled();
    let ignored = ignored_tool_tokens(tokenizer, tools);
    let mut decoder = tokenizer.decode_stream(!tools);
    let mut text = String::new();
    let mut emitted = 0;
    let mut stopped = false;
    if job.responses.is_some() {
        responses::begin(job)?;
    } else if job.request.stream && job.chat {
        job.events.blocking_send(Ok(chunk(job,json!({"index":0,"delta":{"role":"assistant","content":""},"finish_reason":null,"logprobs":null}),None)))
            .map_err(|e|e.to_string())?;
    }
    while !job.events.is_closed() {
        let Some(step) = structured::step(session, &mut constraint).map_err(|e| e.to_string())?
        else {
            break;
        };
        let token_id = u32::try_from(step.token_id).map_err(|e| e.to_string())?;
        if !ignored.contains(&token_id) {
            if let Some(piece) = decoder.step(token_id).map_err(|e| e.to_string())? {
                text.push_str(&piece);
            }
        }
        if let Some(end) = stops.iter().filter_map(|stop| text.find(stop)).min() {
            text.truncate(end);
            stopped = true;
        }
        if job.responses.is_some() && job.request.stream {
            emitted = responses::advance(job, &text, stopped)?;
        } else if job.request.stream && !tools {
            let end = if stopped {
                text.len()
            } else {
                safe_text_end(&text, &stops)
            };
            if end > emitted {
                job.events
                    .blocking_send(Ok(text_chunk(job, &text[emitted..end])))
                    .map_err(|e| e.to_string())?;
                emitted = end;
            }
        }
        if stopped {
            break;
        }
    }
    if !stopped {
        let ids = session
            .output()
            .generated_token_ids
            .iter()
            .map(|&id| u32::try_from(id).map_err(|e| e.to_string()))
            .collect::<Result<Vec<_>, _>>()?
            .into_iter()
            .filter(|id| !ignored.contains(id))
            .collect::<Vec<_>>();
        let complete = tokenizer.decode(&ids, !tools).map_err(|e| e.to_string())?;
        if !complete.starts_with(&text[..emitted]) {
            return Err("final tokenizer output differs from emitted stream prefix".into());
        }
        text = complete;
        if let Some(end) = stops.iter().filter_map(|stop| text.find(stop)).min() {
            if end < emitted {
                return Err("final tokenizer stop point precedes emitted text".into());
            }
            text.truncate(end);
            stopped = true;
        }
    }
    B::sync(device).map_err(|e| e.to_string())?;
    if job.events.is_closed() {
        return Ok(());
    }
    finish_response(
        job,
        text,
        emitted,
        stopped,
        session.finish_reason(),
        prompt_tokens,
        session.generated_tokens(),
    )
}

fn finish_response(
    job: &Job,
    text: String,
    emitted: usize,
    stopped: bool,
    reason: Option<GenerationFinishReason>,
    prompt_tokens: usize,
    generated_tokens: usize,
) -> Result<(), String> {
    let tools = job.request.tools_enabled();
    let (content, calls) = if tools {
        parse_tool_calls(&text, job.request.tools.as_deref().unwrap_or(&[]), &job.id)?
    } else {
        (text.clone(), vec![])
    };
    if job.request.tool_choice.as_ref().and_then(Value::as_str) == Some("required")
        && calls.is_empty()
    {
        return Err("model finished without the required tool call".into());
    }
    if job.request.parallel_tool_calls() == Some(false) && calls.len() > 1 {
        return Err("model emitted multiple calls with parallel_tool_calls=false".into());
    }
    if let Some(name) = job
        .request
        .tool_choice
        .as_ref()
        .and_then(|choice| choice.pointer("/function/name"))
        .and_then(Value::as_str)
    {
        if calls.is_empty()
            || calls
                .iter()
                .any(|call| call.pointer("/function/name").and_then(Value::as_str) != Some(name))
        {
            return Err("model did not emit the requested named tool".into());
        }
    }
    let finish = if !calls.is_empty() {
        "tool_calls"
    } else if stopped {
        "stop"
    } else if matches!(reason, Some(GenerationFinishReason::MaxNewTokens)) {
        "length"
    } else {
        "stop"
    };
    let usage = json!({"prompt_tokens":prompt_tokens,"completion_tokens":generated_tokens,"total_tokens":prompt_tokens+generated_tokens});
    if job.responses.is_some() {
        return responses::finish(job, &text, finish, prompt_tokens, generated_tokens);
    }
    if job.request.stream {
        if tools {
            if !content.is_empty() {
                job.events
                    .blocking_send(Ok(text_chunk(job, &content)))
                    .map_err(|e| e.to_string())?;
            }
            for (index, call) in calls.iter().enumerate() {
                let mut delta = call.clone();
                delta["index"] = json!(index);
                job.events.blocking_send(Ok(chunk(job,json!({"index":0,"delta":{"tool_calls":[delta]},"finish_reason":null,"logprobs":null}),None))).map_err(|e|e.to_string())?;
            }
        } else if text.len() > emitted {
            job.events
                .blocking_send(Ok(text_chunk(job, &text[emitted..])))
                .map_err(|e| e.to_string())?;
        }
        job.events
            .blocking_send(Ok(chunk(
                job,
                if job.chat {
                    json!({"index":0,"delta":{},"finish_reason":finish,"logprobs":null})
                } else {
                    json!({"index":0,"text":"","finish_reason":finish,"logprobs":null})
                },
                None,
            )))
            .map_err(|e| e.to_string())?;
        if job
            .request
            .stream_options
            .as_ref()
            .is_some_and(|o| o.include_usage)
        {
            let mut final_usage = chunk(job, Value::Null, Some(usage.clone()));
            final_usage["choices"] = json!([]);
            final_usage["usage"] = usage;
            job.events
                .blocking_send(Ok(final_usage))
                .map_err(|e| e.to_string())?;
        }
    } else {
        let mut message = json!({"role":"assistant","content":content});
        if !calls.is_empty() {
            message["tool_calls"] = json!(calls);
            if content.is_empty() {
                message["content"] = Value::Null;
            }
        }
        let choice = if job.chat {
            json!({"index":0,"message":message,"finish_reason":finish,"logprobs":null})
        } else {
            json!({"index":0,"text":content,"finish_reason":finish,"logprobs":null})
        };
        job.events
            .blocking_send(Ok(
                json!({"id":job.id,"object":if job.chat{"chat.completion"}else{"text_completion"},
            "created":job.created,"model":job.request.model,"choices":[choice],"usage":usage}),
            ))
            .map_err(|e| e.to_string())?;
    }
    Ok(())
}

pub fn tokenizer_config(directory: &Path) -> Result<Value, String> {
    serde_json::from_slice(
        &std::fs::read(directory.join("tokenizer_config.json")).map_err(|e| e.to_string())?,
    )
    .map_err(|e| e.to_string())
}
