//! Minimal OpenAI-compatible HTTP/1.1 server over [`ContinuousBatchRunner`].
//!
//! Endpoints: `GET /health`, `GET /v1/models`, `POST /v1/completions` and
//! `POST /v1/chat/completions`, each with optional `"stream": true` (server-
//! sent events). One engine thread owns the model and the scheduler; every
//! connection is parsed on its own thread and exchanges events with the
//! engine over channels. Only the standard library is used for networking.
//!
//! Security: there is no TLS. Requests are unauthenticated unless
//! `api_key` is set, in which case `Authorization: Bearer <key>` is required.
//! Bind to a loopback address unless the network is trusted.

use crate::{
    ContinuousBatchConfig, ContinuousBatchRunner, GenerationError, GreedyGenerationConfig,
    PackedLlamaForCausalLm, PagedKvCacheConfig, Qwen2ChatMessage, Qwen2ChatRole, RequestId,
    SamplingConfig, SamplingOptions, SamplingGenerationConfig, SpeculativeConfig, render_qwen2_chat_messages,
};
use ruda::runtime::server::ComputeServer;
use ruda_tensor::DeviceOps;
use ruda_tensor_device::{BoolElement, DeviceBackend, DeviceRuntime, FloatElement, IntElement};
use serde_json::{Value, json};
use std::collections::BTreeMap;
use std::io::{BufRead, BufReader, Read, Write};
use std::net::{TcpListener, TcpStream};
use std::sync::mpsc::{self, Receiver, RecvTimeoutError, Sender, TryRecvError};
use std::thread;
use std::time::{Duration, SystemTime, UNIX_EPOCH};
use tokenizers::Tokenizer;

#[derive(Debug, Clone)]
pub struct OpenAiServerConfig {
    /// For example `127.0.0.1:8000`.
    pub address: String,
    /// Name reported by `/v1/models` and in responses.
    pub model_name: String,
    /// Require `Authorization: Bearer <key>` when set.
    pub api_key: Option<String>,
    pub batching: ContinuousBatchConfig,
    pub kv: PagedKvCacheConfig,
    pub prefix_caching: bool,
    /// Prompt-lookup speculative decoding for greedy (`temperature: 0`) requests.
    pub speculative: Option<SpeculativeConfig>,
    /// Default `max_tokens` when a request omits it.
    pub default_max_tokens: usize,
    pub max_request_bytes: usize,
}

impl OpenAiServerConfig {
    pub fn new(address: impl Into<String>, model_name: impl Into<String>, kv: PagedKvCacheConfig) -> Self {
        Self {
            address: address.into(),
            model_name: model_name.into(),
            api_key: None,
            batching: ContinuousBatchConfig {
                max_active_sequences: 16,
                max_batch_tokens: 512,
            },
            kv,
            prefix_caching: true,
            speculative: None,
            default_max_tokens: 256,
            max_request_bytes: 1 << 20,
        }
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum Endpoint {
    Completion,
    Chat,
}

#[derive(Debug)]
enum PromptInput {
    Text(String),
    Chat(Vec<Qwen2ChatMessage>),
}

#[derive(Debug)]
struct Job {
    prompt: PromptInput,
    max_tokens: Option<usize>,
    sampling: Option<SamplingConfig>,
    sampling_options: SamplingOptions,
    stop: Vec<String>,
    events: Sender<Event>,
}

#[derive(Debug)]
enum Event {
    Delta(String),
    Done {
        finish_reason: &'static str,
        prompt_tokens: usize,
        completion_tokens: usize,
    },
    Error(u16, String),
}

/// Serve forever on `config.address`. `tokenizer` must match `model`;
/// `eos_token_ids` terminate generation (for Qwen chat, include `<|im_end|>`).
pub fn serve_openai<R, F, I, BT>(
    model: PackedLlamaForCausalLm<DeviceBackend<R, F, I, BT>>,
    tokenizer: Tokenizer,
    eos_token_ids: Vec<i32>,
    config: OpenAiServerConfig,
    device: &R::Device,
) -> Result<(), GenerationError>
where
    R: DeviceRuntime,
    R::Server: ComputeServer,
    R::Device: DeviceOps,
    F: FloatElement,
    I: IntElement,
    BT: BoolElement,
{
    let listener = TcpListener::bind(&config.address)
        .map_err(|error| GenerationError(format!("cannot bind {}: {error}", config.address)))?;
    let (jobs, incoming) = mpsc::channel::<Job>();
    let http = HttpSettings {
        model_name: config.model_name.clone(),
        api_key: config.api_key.clone(),
        max_request_bytes: config.max_request_bytes,
    };
    thread::Builder::new()
        .name("rullm-http-accept".into())
        .spawn(move || accept_loop(listener, jobs, http))
        .map_err(|error| GenerationError(format!("cannot start HTTP thread: {error}")))?;
    let mut runner = ContinuousBatchRunner::new(&model, config.batching, config.kv, device)?;
    runner.set_prefix_caching(config.prefix_caching);
    runner.set_speculative_decoding(config.speculative)?;
    let mut engine = Engine {
        tokenizer,
        eos_token_ids,
        max_sequence_length: config.kv.max_sequence_length,
        default_max_tokens: config.default_max_tokens,
        jobs: BTreeMap::new(),
    };
    engine.run(&mut runner, &model, &incoming, device)
}

struct HttpSettings {
    model_name: String,
    api_key: Option<String>,
    max_request_bytes: usize,
}

struct JobState {
    events: Sender<Event>,
    stop: Vec<String>,
    prompt_tokens: usize,
    generated: Vec<i32>,
    emitted: usize,
    disconnected: bool,
}

struct Engine {
    tokenizer: Tokenizer,
    eos_token_ids: Vec<i32>,
    max_sequence_length: usize,
    default_max_tokens: usize,
    jobs: BTreeMap<RequestId, JobState>,
}

impl Engine {
    fn run<R, F, I, BT>(
        &mut self,
        runner: &mut ContinuousBatchRunner<R>,
        model: &PackedLlamaForCausalLm<DeviceBackend<R, F, I, BT>>,
        incoming: &Receiver<Job>,
        device: &R::Device,
    ) -> Result<(), GenerationError>
    where
        R: DeviceRuntime,
        R::Server: ComputeServer,
        R::Device: DeviceOps,
        F: FloatElement,
        I: IntElement,
        BT: BoolElement,
    {
        loop {
            if self.jobs.is_empty() {
                match incoming.recv_timeout(Duration::from_millis(500)) {
                    Ok(job) => self.admit(runner, job),
                    Err(RecvTimeoutError::Timeout) => continue,
                    Err(RecvTimeoutError::Disconnected) => return Ok(()),
                }
            }
            loop {
                match incoming.try_recv() {
                    Ok(job) => self.admit(runner, job),
                    Err(TryRecvError::Empty) => break,
                    Err(TryRecvError::Disconnected) => return Ok(()),
                }
            }
            self.drop_disconnected(runner);
            match runner.step(model, device) {
                Ok(Some(report)) => {
                    for (id, token) in report.selected {
                        self.on_token(runner, id, token);
                    }
                }
                Ok(None) => {
                    if !self.jobs.is_empty() {
                        self.fail_all(runner, 503, "scheduler stalled: KV cache exhausted");
                    }
                }
                Err(error) => self.fail_all(runner, 500, &error.0),
            }
            while let Some(done) = runner.pop_finished() {
                self.finish(done.request_id, if done.stopped_on_eos { "stop" } else { "length" });
            }
        }
    }

    fn encode(&self, prompt: &PromptInput) -> Result<Vec<i32>, String> {
        let text = match prompt {
            PromptInput::Text(text) => text.clone(),
            PromptInput::Chat(messages) => {
                render_qwen2_chat_messages(messages, true).map_err(|error| error.0)?
            }
        };
        let encoding = self
            .tokenizer
            .encode(text, false)
            .map_err(|error| format!("tokenizer encode failed: {error}"))?;
        encoding
            .get_ids()
            .iter()
            .map(|&id| i32::try_from(id).map_err(|_| "token id exceeds i32".to_string()))
            .collect()
    }

    fn decode(&self, tokens: &[i32]) -> String {
        let ids: Vec<u32> = tokens.iter().map(|&token| token as u32).collect();
        self.tokenizer.decode(&ids, true).unwrap_or_default()
    }

    fn admit<R>(&mut self, runner: &mut ContinuousBatchRunner<R>, job: Job)
    where
        R: DeviceRuntime,
        R::Server: ComputeServer,
        R::Device: DeviceOps,
    {
        let prompt = match self.encode(&job.prompt) {
            Ok(prompt) if !prompt.is_empty() => prompt,
            Ok(_) => {
                let _ = job.events.send(Event::Error(400, "prompt is empty".into()));
                return;
            }
            Err(error) => {
                let _ = job.events.send(Event::Error(400, error));
                return;
            }
        };
        if prompt.len() >= self.max_sequence_length {
            let _ = job.events.send(Event::Error(
                400,
                format!(
                    "prompt has {} tokens; the context limit is {}",
                    prompt.len(),
                    self.max_sequence_length
                ),
            ));
            return;
        }
        let room = self.max_sequence_length - prompt.len();
        let max_new_tokens = job.max_tokens.unwrap_or(self.default_max_tokens).min(room).max(1);
        let eos_token_ids = self.eos_token_ids.clone();
        let submitted = match job.sampling {
            Some(sampling) => runner.submit_sampled_with_options(
                prompt.clone(),
                SamplingGenerationConfig {
                    max_new_tokens,
                    eos_token_ids,
                    sampling,
                },
                job.sampling_options,
            ),
            None => runner.submit(
                prompt.clone(),
                GreedyGenerationConfig {
                    max_new_tokens,
                    eos_token_ids,
                },
            ),
        };
        match submitted {
            Ok(id) => {
                self.jobs.insert(
                    id,
                    JobState {
                        events: job.events,
                        stop: job.stop,
                        prompt_tokens: prompt.len(),
                        generated: Vec::new(),
                        emitted: 0,
                        disconnected: false,
                    },
                );
            }
            Err(error) => {
                let _ = job.events.send(Event::Error(400, error.0));
            }
        }
    }

    fn on_token<R>(&mut self, runner: &mut ContinuousBatchRunner<R>, id: RequestId, token: i32)
    where
        R: DeviceRuntime,
        R::Server: ComputeServer,
        R::Device: DeviceOps,
    {
        let Some(state) = self.jobs.get_mut(&id) else {
            return;
        };
        state.generated.push(token);
        if self.eos_token_ids.contains(&token) {
            // The finished event follows from the scheduler; EOS text is not emitted.
            return;
        }
        let text = {
            let ids: Vec<u32> = state.generated.iter().map(|&t| t as u32).collect();
            self.tokenizer.decode(&ids, true).unwrap_or_default()
        };
        if let Some(cut) = earliest_stop(&text, &state.stop) {
            if cut > state.emitted {
                let delta = text[state.emitted..cut].to_string();
                if state.events.send(Event::Delta(delta)).is_err() {
                    state.disconnected = true;
                }
            }
            state.emitted = cut;
            // The request may still be running; drop it now and report a stop.
            let _ = runner.cancel(id);
            self.finish(id, "stop");
            return;
        }
        // Hold back an incomplete UTF-8 sequence and any suffix that could be
        // the start of a stop string.
        let mut safe = text.len();
        if text.ends_with('\u{FFFD}') {
            safe = text.trim_end_matches('\u{FFFD}').len();
        }
        safe = safe.saturating_sub(stop_prefix_suffix(&text[..safe], &state.stop));
        while safe > 0 && !text.is_char_boundary(safe) {
            safe -= 1;
        }
        if safe > state.emitted {
            let delta = text[state.emitted..safe].to_string();
            state.emitted = safe;
            if state.events.send(Event::Delta(delta)).is_err() {
                state.disconnected = true;
            }
        }
    }

    fn finish(&mut self, id: RequestId, finish_reason: &'static str) {
        let Some(state) = self.jobs.remove(&id) else {
            return;
        };
        let content: Vec<i32> = state
            .generated
            .iter()
            .copied()
            .filter(|token| !self.eos_token_ids.contains(token))
            .collect();
        let text = self.decode(&content);
        let end = earliest_stop(&text, &state.stop).unwrap_or(text.len());
        if end > state.emitted && text.is_char_boundary(state.emitted) {
            let _ = state
                .events
                .send(Event::Delta(text[state.emitted..end].to_string()));
        }
        let _ = state.events.send(Event::Done {
            finish_reason,
            prompt_tokens: state.prompt_tokens,
            completion_tokens: state.generated.len(),
        });
    }

    fn drop_disconnected<R>(&mut self, runner: &mut ContinuousBatchRunner<R>)
    where
        R: DeviceRuntime,
        R::Server: ComputeServer,
        R::Device: DeviceOps,
    {
        let gone: Vec<RequestId> = self
            .jobs
            .iter()
            .filter(|(_, state)| state.disconnected)
            .map(|(&id, _)| id)
            .collect();
        for id in gone {
            let _ = runner.cancel(id);
            self.jobs.remove(&id);
        }
    }

    fn fail_all<R>(&mut self, runner: &mut ContinuousBatchRunner<R>, status: u16, message: &str)
    where
        R: DeviceRuntime,
        R::Server: ComputeServer,
        R::Device: DeviceOps,
    {
        for (id, state) in std::mem::take(&mut self.jobs) {
            let _ = runner.cancel(id);
            let _ = state.events.send(Event::Error(status, message.to_string()));
        }
    }
}

fn earliest_stop(text: &str, stops: &[String]) -> Option<usize> {
    stops
        .iter()
        .filter(|stop| !stop.is_empty())
        .filter_map(|stop| text.find(stop.as_str()))
        .min()
}

/// Length of the longest suffix of `text` that is a proper prefix of a stop string.
fn stop_prefix_suffix(text: &str, stops: &[String]) -> usize {
    let mut longest = 0;
    for stop in stops {
        for length in (1..stop.len()).rev() {
            if length <= text.len()
                && stop.is_char_boundary(length)
                && text.ends_with(&stop[..length])
            {
                longest = longest.max(length);
                break;
            }
        }
    }
    longest
}

// ---------------------------------------------------------------- HTTP side

fn accept_loop(listener: TcpListener, jobs: Sender<Job>, settings: HttpSettings) {
    let settings = std::sync::Arc::new(settings);
    let mut next_id = 0u64;
    for stream in listener.incoming() {
        let Ok(stream) = stream else {
            continue;
        };
        next_id += 1;
        let jobs = jobs.clone();
        let settings = settings.clone();
        let request_id = next_id;
        let _ = thread::Builder::new()
            .name(format!("rullm-http-{request_id}"))
            .spawn(move || {
                let _ = handle_connection(stream, &jobs, &settings, request_id);
            });
    }
}

struct HttpRequest {
    method: String,
    path: String,
    headers: BTreeMap<String, String>,
    body: Vec<u8>,
}

fn read_request(stream: &TcpStream, limit: usize) -> Result<HttpRequest, (u16, String)> {
    stream
        .set_read_timeout(Some(Duration::from_secs(30)))
        .map_err(|error| (500, error.to_string()))?;
    let mut reader = BufReader::new(stream);
    let mut line = String::new();
    reader
        .read_line(&mut line)
        .map_err(|error| (400, error.to_string()))?;
    let mut parts = line.split_whitespace();
    let method = parts.next().unwrap_or_default().to_string();
    let path = parts.next().unwrap_or_default().to_string();
    if method.is_empty() || path.is_empty() {
        return Err((400, "malformed request line".into()));
    }
    let mut headers = BTreeMap::new();
    let mut header_bytes = 0usize;
    loop {
        let mut header = String::new();
        let read = reader
            .read_line(&mut header)
            .map_err(|error| (400, error.to_string()))?;
        header_bytes += read;
        if read == 0 || header_bytes > 64 * 1024 {
            return Err((400, "headers are truncated or too large".into()));
        }
        let header = header.trim_end_matches(['\r', '\n']);
        if header.is_empty() {
            break;
        }
        if let Some((name, value)) = header.split_once(':') {
            headers.insert(name.trim().to_ascii_lowercase(), value.trim().to_string());
        }
    }
    let length = headers
        .get("content-length")
        .map(|value| value.parse::<usize>())
        .transpose()
        .map_err(|_| (400, "invalid Content-Length".to_string()))?
        .unwrap_or(0);
    if length > limit {
        return Err((413, format!("request body exceeds {limit} bytes")));
    }
    let mut body = vec![0u8; length];
    reader
        .read_exact(&mut body)
        .map_err(|error| (400, error.to_string()))?;
    Ok(HttpRequest {
        method,
        path,
        headers,
        body,
    })
}

fn status_text(status: u16) -> &'static str {
    match status {
        200 => "OK",
        400 => "Bad Request",
        401 => "Unauthorized",
        404 => "Not Found",
        405 => "Method Not Allowed",
        413 => "Payload Too Large",
        503 => "Service Unavailable",
        _ => "Internal Server Error",
    }
}

fn write_json(stream: &mut TcpStream, status: u16, body: &Value) -> std::io::Result<()> {
    let body = body.to_string();
    write!(
        stream,
        "HTTP/1.1 {status} {}\r\nContent-Type: application/json\r\nContent-Length: {}\r\nConnection: close\r\n\r\n{body}",
        status_text(status),
        body.len()
    )?;
    stream.flush()
}

fn write_error(stream: &mut TcpStream, status: u16, message: &str) -> std::io::Result<()> {
    write_json(
        stream,
        status,
        &json!({"error": {"message": message, "type": "invalid_request_error", "code": status}}),
    )
}

fn unix_seconds() -> u64 {
    SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .map_or(0, |duration| duration.as_secs())
}

fn handle_connection(
    mut stream: TcpStream,
    jobs: &Sender<Job>,
    settings: &HttpSettings,
    request_id: u64,
) -> std::io::Result<()> {
    let request = match read_request(&stream, settings.max_request_bytes) {
        Ok(request) => request,
        Err((status, message)) => return write_error(&mut stream, status, &message),
    };
    if let Some(key) = &settings.api_key {
        let expected = format!("Bearer {key}");
        if request.headers.get("authorization") != Some(&expected) {
            return write_error(&mut stream, 401, "missing or invalid API key");
        }
    }
    let path = request.path.split('?').next().unwrap_or_default();
    let endpoint = match (request.method.as_str(), path) {
        ("GET", "/health") => return write_json(&mut stream, 200, &json!({"status": "ok"})),
        ("GET", "/v1/models") => {
            return write_json(
                &mut stream,
                200,
                &json!({"object": "list", "data": [{
                    "id": settings.model_name, "object": "model",
                    "created": unix_seconds(), "owned_by": "ruda",
                }]}),
            );
        }
        ("POST", "/v1/completions") => Endpoint::Completion,
        ("POST", "/v1/chat/completions") => Endpoint::Chat,
        (_, "/v1/completions" | "/v1/chat/completions" | "/v1/models" | "/health") => {
            return write_error(&mut stream, 405, "method not allowed");
        }
        _ => return write_error(&mut stream, 404, "unknown endpoint"),
    };
    let body: Value = match serde_json::from_slice(&request.body) {
        Ok(body) => body,
        Err(error) => return write_error(&mut stream, 400, &format!("invalid JSON: {error}")),
    };
    let (job, events, stream_response) = match parse_job(endpoint, &body) {
        Ok(parsed) => parsed,
        Err(message) => return write_error(&mut stream, 400, &message),
    };
    if jobs.send(job).is_err() {
        return write_error(&mut stream, 503, "engine is not running");
    }
    let created = unix_seconds();
    let id = match endpoint {
        Endpoint::Completion => format!("cmpl-{request_id}"),
        Endpoint::Chat => format!("chatcmpl-{request_id}"),
    };
    if stream_response {
        stream_events(&mut stream, endpoint, &id, created, &settings.model_name, &events)
    } else {
        collect_events(&mut stream, endpoint, &id, created, &settings.model_name, &events)
    }
}

fn number<T: TryFrom<u64>>(body: &Value, key: &str) -> Result<Option<T>, String> {
    match body.get(key) {
        None | Some(Value::Null) => Ok(None),
        Some(value) => value
            .as_u64()
            .and_then(|value| T::try_from(value).ok())
            .map(Some)
            .ok_or_else(|| format!("{key} must be a non-negative integer")),
    }
}

fn float(body: &Value, key: &str) -> Result<Option<f64>, String> {
    match body.get(key) {
        None | Some(Value::Null) => Ok(None),
        Some(value) => value
            .as_f64()
            .map(Some)
            .ok_or_else(|| format!("{key} must be a number")),
    }
}

fn parse_job(endpoint: Endpoint, body: &Value) -> Result<(Job, Receiver<Event>, bool), String> {
    let prompt = match endpoint {
        Endpoint::Completion => match body.get("prompt") {
            Some(Value::String(text)) => PromptInput::Text(text.clone()),
            Some(Value::Array(items)) if items.len() == 1 && items[0].is_string() => {
                PromptInput::Text(items[0].as_str().unwrap_or_default().to_string())
            }
            _ => return Err("prompt must be a string".into()),
        },
        Endpoint::Chat => {
            let Some(Value::Array(messages)) = body.get("messages") else {
                return Err("messages must be an array".into());
            };
            let mut parsed = Vec::with_capacity(messages.len());
            for message in messages {
                let role = match message.get("role").and_then(Value::as_str) {
                    Some("system") => Qwen2ChatRole::System,
                    Some("user") => Qwen2ChatRole::User,
                    Some("assistant") => Qwen2ChatRole::Assistant,
                    other => return Err(format!("unsupported message role {other:?}")),
                };
                let content = match message.get("content") {
                    Some(Value::String(text)) => text.clone(),
                    Some(Value::Array(parts)) => parts
                        .iter()
                        .filter_map(|part| {
                            (part.get("type").and_then(Value::as_str) == Some("text"))
                                .then(|| part.get("text").and_then(Value::as_str))
                                .flatten()
                        })
                        .collect::<Vec<_>>()
                        .join(""),
                    _ => return Err("message content must be text".into()),
                };
                parsed.push(Qwen2ChatMessage::new(role, content));
            }
            if parsed.is_empty() {
                return Err("messages must not be empty".into());
            }
            PromptInput::Chat(parsed)
        }
    };
    let max_tokens = match endpoint {
        Endpoint::Chat => number::<usize>(body, "max_completion_tokens")?
            .or(number::<usize>(body, "max_tokens")?),
        Endpoint::Completion => number::<usize>(body, "max_tokens")?,
    };
    if body.get("n").and_then(Value::as_u64).is_some_and(|n| n != 1) {
        return Err("only n = 1 is supported".into());
    }
    let temperature = float(body, "temperature")?.unwrap_or(1.0);
    let sampling = if temperature == 0.0 {
        None
    } else {
        let config = SamplingConfig {
            temperature,
            top_k: number::<usize>(body, "top_k")?.unwrap_or(0),
            top_p: float(body, "top_p")?.unwrap_or(1.0),
            seed: number::<u64>(body, "seed")?,
        };
        config.validate().map_err(|error| error.0)?;
        Some(config)
    };
    let sampling_options = if sampling.is_some() {
        let options = SamplingOptions {
            min_p: float(body, "min_p")?.unwrap_or(0.0),
            repetition_penalty: float(body, "repetition_penalty")?.unwrap_or(1.0),
            presence_penalty: float(body, "presence_penalty")?.unwrap_or(0.0),
            frequency_penalty: float(body, "frequency_penalty")?.unwrap_or(0.0),
        };
        options.validate().map_err(|error| error.0)?;
        options
    } else { SamplingOptions::default() };
    let stop = match body.get("stop") {
        None | Some(Value::Null) => Vec::new(),
        Some(Value::String(stop)) => vec![stop.clone()],
        Some(Value::Array(stops)) => stops
            .iter()
            .map(|stop| {
                stop.as_str()
                    .map(str::to_string)
                    .ok_or_else(|| "stop entries must be strings".to_string())
            })
            .collect::<Result<_, _>>()?,
        _ => return Err("stop must be a string or an array of strings".into()),
    };
    if stop.len() > 4 {
        return Err("at most four stop sequences are supported".into());
    }
    let stream = body.get("stream").and_then(Value::as_bool).unwrap_or(false);
    let (events, receiver) = mpsc::channel();
    Ok((
        Job {
            prompt,
            max_tokens,
            sampling,
            sampling_options,
            stop,
            events,
        },
        receiver,
        stream,
    ))
}

fn collect_events(
    stream: &mut TcpStream,
    endpoint: Endpoint,
    id: &str,
    created: u64,
    model: &str,
    events: &Receiver<Event>,
) -> std::io::Result<()> {
    let mut text = String::new();
    loop {
        match events.recv() {
            Ok(Event::Delta(delta)) => text.push_str(&delta),
            Ok(Event::Done {
                finish_reason,
                prompt_tokens,
                completion_tokens,
            }) => {
                let usage = json!({
                    "prompt_tokens": prompt_tokens,
                    "completion_tokens": completion_tokens,
                    "total_tokens": prompt_tokens + completion_tokens,
                });
                let body = match endpoint {
                    Endpoint::Completion => json!({
                        "id": id, "object": "text_completion", "created": created, "model": model,
                        "choices": [{"index": 0, "text": text, "logprobs": null, "finish_reason": finish_reason}],
                        "usage": usage,
                    }),
                    Endpoint::Chat => json!({
                        "id": id, "object": "chat.completion", "created": created, "model": model,
                        "choices": [{"index": 0, "message": {"role": "assistant", "content": text},
                                     "finish_reason": finish_reason}],
                        "usage": usage,
                    }),
                };
                return write_json(stream, 200, &body);
            }
            Ok(Event::Error(status, message)) => return write_error(stream, status, &message),
            Err(_) => return write_error(stream, 500, "engine stopped"),
        }
    }
}

fn stream_events(
    stream: &mut TcpStream,
    endpoint: Endpoint,
    id: &str,
    created: u64,
    model: &str,
    events: &Receiver<Event>,
) -> std::io::Result<()> {
    // Wait for the first event so request errors still get a JSON status.
    let first = match events.recv() {
        Ok(Event::Error(status, message)) => return write_error(stream, status, &message),
        Ok(event) => event,
        Err(_) => return write_error(stream, 500, "engine stopped"),
    };
    write!(
        stream,
        "HTTP/1.1 200 OK\r\nContent-Type: text/event-stream\r\nCache-Control: no-cache\r\nConnection: close\r\n\r\n"
    )?;
    let chunk = |delta: Option<&str>, finish_reason: Option<&str>, role: bool| match endpoint {
        Endpoint::Completion => json!({
            "id": id, "object": "text_completion", "created": created, "model": model,
            "choices": [{"index": 0, "text": delta.unwrap_or(""), "logprobs": null,
                         "finish_reason": finish_reason}],
        }),
        Endpoint::Chat => {
            let mut content = serde_json::Map::new();
            if role {
                content.insert("role".into(), json!("assistant"));
            }
            if let Some(delta) = delta {
                content.insert("content".into(), json!(delta));
            }
            json!({
                "id": id, "object": "chat.completion.chunk", "created": created, "model": model,
                "choices": [{"index": 0, "delta": content, "finish_reason": finish_reason}],
            })
        }
    };
    let send = |stream: &mut TcpStream, value: Value| -> std::io::Result<()> {
        write!(stream, "data: {value}\n\n")?;
        stream.flush()
    };
    if endpoint == Endpoint::Chat {
        send(stream, chunk(None, None, true))?;
    }
    let mut event = Some(first);
    loop {
        let current = match event.take() {
            Some(event) => event,
            None => match events.recv() {
                Ok(event) => event,
                Err(_) => break,
            },
        };
        match current {
            Event::Delta(delta) => send(stream, chunk(Some(&delta), None, false))?,
            Event::Done { finish_reason, .. } => {
                send(stream, chunk(None, Some(finish_reason), false))?;
                break;
            }
            Event::Error(_, message) => {
                send(stream, json!({"error": {"message": message}}))?;
                break;
            }
        }
    }
    write!(stream, "data: [DONE]\n\n")?;
    stream.flush()
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn stop_sequences_are_found_and_partial_suffixes_held_back() {
        let stops = vec!["</end>".to_string(), "\n\n".to_string()];
        assert_eq!(earliest_stop("abc</end>x\n\n", &stops), Some(3));
        assert_eq!(earliest_stop("abc", &stops), None);
        assert_eq!(stop_prefix_suffix("abc</e", &stops), 3);
        assert_eq!(stop_prefix_suffix("abc\n", &stops), 1);
        assert_eq!(stop_prefix_suffix("abc", &stops), 0);
    }

    #[test]
    fn requests_are_parsed_with_openai_defaults() {
        let (job, _, stream) = parse_job(
            Endpoint::Chat,
            &json!({"messages": [{"role": "user", "content": "hi"}], "temperature": 0, "stream": true,
                    "max_tokens": 7, "stop": "x"}),
        )
        .unwrap();
        assert!(stream);
        assert!(job.sampling.is_none());
        assert_eq!(job.max_tokens, Some(7));
        assert_eq!(job.stop, vec!["x".to_string()]);
        let (job, _, _) = parse_job(
            Endpoint::Completion,
            &json!({"prompt": "a", "top_p": 0.5, "seed": 3, "repetition_penalty": 1.1}),
        )
        .unwrap();
        let sampling = job.sampling.unwrap();
        assert_eq!(sampling.top_p, 0.5);
        assert_eq!(sampling.seed, Some(3));
        assert!(parse_job(Endpoint::Completion, &json!({"prompt": 3})).is_err());
        assert!(parse_job(Endpoint::Chat, &json!({"messages": [{"role": "tool", "content": "x"}]})).is_err());
        assert!(parse_job(Endpoint::Completion, &json!({"prompt": "a", "top_p": 2.0})).is_err());
    }
}
