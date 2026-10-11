use super::*;

#[derive(Clone)]
pub(super) struct Config {
    context_window: usize,
    max_output_tokens: usize,
    images: bool,
}

impl ServerState {
    pub fn with_responses(
        mut self,
        context_window: usize,
        max_output_tokens: usize,
        images: bool,
    ) -> Result<Self, String> {
        if max_output_tokens == 0 || max_output_tokens >= context_window {
            return Err(
                "Responses output limit must be positive and below the checkpoint context window"
                    .into(),
            );
        }
        self.responses_config = Some(Config {
            context_window,
            max_output_tokens,
            images,
        });
        Ok(self)
    }
}

pub(super) async fn models(State(state): State<ServerState>) -> Response {
    let Some(config) = state.responses_config else {
        return api_error(
            StatusCode::SERVICE_UNAVAILABLE,
            "enable Responses with --responses-max-output-tokens".into(),
        );
    };
    Json(json!({"models":[{
        "slug":state.model,"display_name":state.model,"description":null,
        "supported_reasoning_levels":[],"shell_type":"unified_exec","visibility":"list",
        "supported_in_api":true,"priority":0,"upgrade":null,"availability_nux":null,
        "base_instructions":include_str!("codex_prompt.md"),
        "supports_reasoning_summary_parameter":false,"default_reasoning_summary":"none",
        "support_verbosity":false,"default_verbosity":null,"apply_patch_tool_type":"freeform",
        "truncation_policy":{"mode":"bytes","limit":10000},
        "context_window":config.context_window,"max_context_window":config.context_window,
        "experimental_supported_tools":[],"input_modalities":if config.images {vec!["text","image"]}else{vec!["text"]},
        "supports_search_tool":false,"use_responses_lite":false
    }]})).into_response()
}

pub(super) async fn create(
    State(state): State<ServerState>,
    body: Result<Json<Value>, axum::extract::rejection::JsonRejection>,
) -> Response {
    let body = match body {
        Ok(Json(body)) => body,
        Err(e) => return api_error(StatusCode::BAD_REQUEST, e.body_text()),
    };
    let Some(config) = state.responses_config.as_ref() else {
        return api_error(
            StatusCode::SERVICE_UNAVAILABLE,
            "enable Responses with --responses-max-output-tokens".into(),
        );
    };
    match translate(body, config) {
        Ok((request, output)) => submit_wire(state, request, true, Some(output)).await,
        Err(e) => api_error(StatusCode::BAD_REQUEST, e),
    }
}

#[derive(Clone)]
struct Tool {
    name: String,
    namespace: Option<String>,
    custom: bool,
}

fn required<'a>(value: &'a Value, key: &str) -> Result<&'a str, String> {
    value
        .get(key)
        .and_then(Value::as_str)
        .ok_or_else(|| format!("missing string {key}"))
}

fn tool_name(name: &str, namespace: Option<&str>) -> String {
    namespace.map_or_else(|| name.to_owned(), |ns| format!("{ns}.{name}"))
}

fn register(
    definitions: &[Value],
    namespace: Option<&str>,
    tools: &mut Vec<Value>,
    mapping: &mut BTreeMap<String, Tool>,
) -> Result<(), String> {
    for definition in definitions {
        let kind = required(definition, "type")?;
        if kind == "namespace" && namespace.is_none() {
            let ns = required(definition, "name")?;
            let children = definition
                .get("tools")
                .and_then(Value::as_array)
                .ok_or("namespace requires tools")?;
            register(children, Some(ns), tools, mapping)?;
            continue;
        }
        if !matches!(kind, "function" | "custom") {
            return Err(format!("unsupported Responses tool {kind}"));
        }
        if definition.get("defer_loading").and_then(Value::as_bool) == Some(true) {
            return Err("deferred tool discovery is not implemented".into());
        }
        let name = required(definition, "name")?;
        let wire_name = tool_name(name, namespace);
        if mapping
            .insert(
                wire_name.clone(),
                Tool {
                    name: name.into(),
                    namespace: namespace.map(str::to_owned),
                    custom: kind == "custom",
                },
            )
            .is_some()
        {
            return Err(format!("duplicate tool {wire_name}"));
        }
        let mut function = definition.clone();
        function["name"] = json!(wire_name);
        if kind == "custom" {
            function = json!({"name":wire_name,"description":format!("{}\nReturn the exact tool input in the input string. Tool input format: {}",
                definition.get("description").and_then(Value::as_str).unwrap_or(""),
                definition.get("format").unwrap_or(&Value::Null)),
                "parameters":{"type":"object","properties":{"input":{"type":"string"}},"required":["input"],"additionalProperties":false}});
        } else {
            if definition.get("strict").and_then(Value::as_bool) == Some(true) {
                return Err("strict tool argument decoding is not implemented".into());
            }
            if definition.get("parameters").is_none() {
                return Err(format!("function {name} requires parameters"));
            }
            let object = function
                .as_object_mut()
                .ok_or("invalid function definition")?;
            object.remove("type");
            object.remove("defer_loading");
        }
        tools.push(json!({"type":"function","function":function}));
    }
    Ok(())
}

fn content(value: &Value) -> Result<Value, String> {
    if value.is_string() {
        return Ok(value.clone());
    }
    let parts = value
        .as_array()
        .ok_or("Responses content must be a string or parts array")?;
    let mut output = Vec::new();
    for part in parts {
        match required(part, "type")? {
            "input_text" | "output_text" => {
                output.push(json!({"type":"text","text":required(part,"text")?}))
            }
            "input_image" if part.get("file_id").is_none() => {
                if part
                    .get("detail")
                    .is_some_and(|v| !v.is_null() && v.as_str() != Some("auto"))
                {
                    return Err("RUDA image detail must be auto; checkpoint preprocessing controls resolution".into());
                }
                let url = required(part, "image_url")?;
                let url =
                    if let Some(body) = url.strip_prefix("data:application/octet-stream;base64,") {
                        use base64::Engine;
                        let bytes = base64::engine::general_purpose::STANDARD
                            .decode(body)
                            .map_err(|e| e.to_string())?;
                        let mime = if bytes.starts_with(b"\x89PNG\r\n\x1a\n") {
                            "image/png"
                        } else if bytes.starts_with(&[0xff, 0xd8, 0xff]) {
                            "image/jpeg"
                        } else {
                            return Err("RUDA image inputs require actual PNG or JPEG bytes".into());
                        };
                        format!("data:{mime};base64,{body}")
                    } else {
                        url.to_owned()
                    };
                output.push(json!({"type":"image_url","image_url":{"url":url}}));
            }
            kind => return Err(format!("unsupported Responses content {kind}")),
        }
    }
    Ok(json!(output))
}

fn translate(body: Value, config: &Config) -> Result<(GenerateRequest, Output), String> {
    let object = body.as_object().ok_or("Responses body must be an object")?;
    for key in ["store", "stream", "parallel_tool_calls"] {
        if body.get(key).is_some_and(|v| !v.is_boolean()) {
            return Err(format!("{key} must be a boolean"));
        }
    }
    for key in object.keys() {
        if !matches!(
            key.as_str(),
            "model"
                | "input"
                | "instructions"
                | "tools"
                | "tool_choice"
                | "parallel_tool_calls"
                | "stream"
                | "store"
                | "reasoning"
                | "include"
                | "text"
                | "prompt_cache_key"
                | "client_metadata"
                | "metadata"
                | "service_tier"
                | "stream_options"
                | "max_output_tokens"
                | "temperature"
                | "top_p"
                | "access_programs"
        ) {
            return Err(format!("unsupported Responses request field {key}"));
        }
    }
    if body.get("store").and_then(Value::as_bool) == Some(true) {
        return Err(
            "stored Responses are not implemented; use store=false and replay input".into(),
        );
    }
    if body.get("access_programs").is_some_and(|v| !v.is_null())
        || body
            .get("service_tier")
            .is_some_and(|v| !v.is_null() && v.as_str() != Some("default"))
    {
        return Err("hosted service tiers/access programs are not implemented".into());
    }
    if let Some(reasoning) = body.get("reasoning").filter(|v| !v.is_null()) {
        for (key, value) in reasoning.as_object().ok_or("reasoning must be an object")? {
            if !value.is_null()
                && !(matches!(key.as_str(), "effort" | "summary") && value.as_str() == Some("none"))
            {
                return Err(format!("unsupported reasoning control {key}"));
            }
        }
    }
    if body.get("stream_options").is_some_and(|v| !v.is_null()) {
        return Err("Responses stream_options are not implemented".into());
    }
    let mut tools = Vec::new();
    let mut mapping = BTreeMap::new();
    if let Some(definitions) = body.get("tools").filter(|v| !v.is_null()) {
        register(
            definitions.as_array().ok_or("tools must be an array")?,
            None,
            &mut tools,
            &mut mapping,
        )?;
    }
    let mut messages = Vec::new();
    if let Some(instructions) = body.get("instructions").filter(|v| !v.is_null()) {
        messages.push(json!({"role":"system","content":instructions.as_str().ok_or("instructions must be a string")?}));
    }
    let input = body.get("input").ok_or("missing Responses input")?;
    let items = if input.is_string() {
        vec![json!({"type":"message","role":"user","content":input})]
    } else {
        input
            .as_array()
            .ok_or("input must be a string or array")?
            .clone()
    };
    let mut pending = BTreeMap::<String, String>::new();
    for item in items {
        match item
            .get("type")
            .and_then(Value::as_str)
            .unwrap_or("message")
        {
            "message" => {
                let role = required(&item, "role")?;
                let role = if role == "developer" { "system" } else { role };
                messages.push(json!({"role":role,"content":content(item.get("content").ok_or("message requires content")?)?}));
            }
            "function_call" | "custom_tool_call" => {
                let name = required(&item, "name")?;
                let wire_name = tool_name(name, item.get("namespace").and_then(Value::as_str));
                let id = required(&item, "call_id")?;
                if pending.insert(id.into(), wire_name.clone()).is_some() {
                    return Err(format!("duplicate call_id {id}"));
                }
                let arguments = if item["type"] == "custom_tool_call" {
                    json!({"input":required(&item,"input")?}).to_string()
                } else {
                    required(&item, "arguments")?.into()
                };
                let _: Value = serde_json::from_str(&arguments)
                    .map_err(|e| format!("invalid replayed tool arguments: {e}"))?;
                messages.push(json!({"role":"assistant","content":null,"tool_calls":[{"id":id,"type":"function","function":{"name":wire_name,"arguments":arguments}}]}));
            }
            "function_call_output" | "custom_tool_call_output" => {
                let id = required(&item, "call_id")?;
                let name = pending
                    .remove(id)
                    .ok_or_else(|| format!("tool output {id} has no matching input call"))?;
                messages.push(json!({"role":"tool","tool_call_id":id,"name":name,"content":content(item.get("output").ok_or("tool result requires output")?)?}));
            }
            kind => return Err(format!("unsupported replay item {kind}")),
        }
    }
    if !pending.is_empty() {
        return Err("input contains tool calls without matching results".into());
    }
    let prefix_len = messages
        .iter()
        .take_while(|message| message["role"] == "system")
        .count();
    if prefix_len > 1 {
        let mut instructions = Vec::new();
        for message in &messages[..prefix_len] {
            let value = &message["content"];
            if let Some(text) = value.as_str() {
                instructions.push(text.to_owned());
            } else {
                let parts = value.as_array().ok_or("system content must be text")?;
                let mut text = String::new();
                for part in parts {
                    if part["type"] != "text" {
                        return Err("system content must be text".into());
                    }
                    text.push_str(required(part, "text")?);
                }
                instructions.push(text);
            }
        }
        messages.splice(
            ..prefix_len,
            [json!({"role":"system","content":instructions.join("\n\n")})],
        );
    }
    let max = match body.get("max_output_tokens") {
        Some(value) => usize::try_from(
            value
                .as_u64()
                .ok_or("max_output_tokens must be a positive integer")?,
        )
        .map_err(|e| e.to_string())?,
        None => config.max_output_tokens,
    };
    if max == 0 || max >= config.context_window {
        return Err("max_output_tokens is outside the checkpoint context window".into());
    }
    let choice = body.get("tool_choice").cloned().unwrap_or(json!("auto"));
    let choice = if choice.is_object() {
        let name = tool_name(
            required(&choice, "name")?,
            choice.get("namespace").and_then(Value::as_str),
        );
        json!({"type":"function","function":{"name":name}})
    } else {
        choice
    };
    let mut request = json!({"model":required(&body,"model")?,"messages":messages,"stream":body.get("stream").unwrap_or(&json!(false)),"max_tokens":max});
    if !tools.is_empty() {
        request["tools"] = json!(tools);
        request["tool_choice"] = choice;
    } else if choice != json!("auto") && choice != json!("none") {
        return Err("tool_choice requires tools".into());
    }
    for key in ["temperature", "top_p"] {
        if let Some(value) = body.get(key) {
            request[key] = value.clone();
        }
    }
    if let Some(text) = body.get("text").filter(|v| !v.is_null()) {
        let text = text.as_object().ok_or("text must be an object")?;
        for key in text.keys() {
            if key != "format" {
                return Err(format!("unsupported text control {key}"));
            }
        }
        if let Some(format) = text.get("format") {
            request["response_format"] = if format.get("type").and_then(Value::as_str)
                == Some("json_schema")
            {
                json!({"type":"json_schema","json_schema":{"name":required(format,"name")?,"description":format.get("description"),"schema":format.get("schema").ok_or("missing text format schema")?,"strict":format.get("strict")}})
            } else {
                format.clone()
            };
        }
    }
    Ok((
        serde_json::from_value(request).map_err(|e| e.to_string())?,
        Output::new(
            mapping,
            body.get("parallel_tool_calls")
                .and_then(Value::as_bool)
                .unwrap_or(true),
        ),
    ))
}

pub(super) struct Output {
    tools: BTreeMap<String, Tool>,
    cursor: usize,
    sequence: usize,
    items: Vec<Value>,
    message: Option<usize>,
    calls: usize,
    begun: bool,
    parallel: bool,
    pending: Option<(usize, String)>,
}

impl Output {
    fn new(tools: BTreeMap<String, Tool>, parallel: bool) -> Self {
        Self {
            tools,
            cursor: 0,
            sequence: 0,
            items: Vec::new(),
            message: None,
            calls: 0,
            begun: false,
            parallel,
            pending: None,
        }
    }
    fn event(&mut self, job: &Job, mut event: Value) -> Result<(), String> {
        if !job.request.stream {
            return Ok(());
        }
        event["sequence_number"] = json!(self.sequence);
        self.sequence += 1;
        job.events
            .blocking_send(Ok(event))
            .map_err(|e| e.to_string())
    }
    fn response(&self, job: &Job, status: &str, usage: Value) -> Value {
        json!({"id":job.id,"object":"response","created_at":job.created,"model":job.request.model,"status":status,"output":self.items,"usage":usage,"error":null,"incomplete_details":if status=="incomplete" {json!({"reason":"max_output_tokens"})}else{Value::Null},"end_turn":self.calls==0})
    }
    fn text(&mut self, job: &Job, piece: &str) -> Result<(), String> {
        if piece.is_empty() {
            return Ok(());
        }
        let index = if let Some(index) = self.message {
            index
        } else {
            let index = self.items.len();
            let item = json!({"type":"message","id":format!("msg-{}-{index}",job.id),"role":"assistant","status":"in_progress","content":[]});
            self.items.push(item.clone());
            self.message = Some(index);
            self.event(
                job,
                json!({"type":"response.output_item.added","output_index":index,"item":item}),
            )?;
            let part = json!({"type":"output_text","text":"","annotations":[]});
            self.items[index]["content"] = json!([part.clone()]);
            self.event(job,json!({"type":"response.content_part.added","item_id":self.items[index]["id"],"output_index":index,"content_index":0,"part":part}))?;
            index
        };
        let mut text = self.items[index]["content"][0]["text"]
            .as_str()
            .unwrap_or("")
            .to_owned();
        text.push_str(piece);
        self.items[index]["content"][0]["text"] = json!(text);
        self.event(job,json!({"type":"response.output_text.delta","item_id":self.items[index]["id"],"output_index":index,"content_index":0,"delta":piece}))
    }
    fn close_message(&mut self, job: &Job) -> Result<(), String> {
        if let Some(index) = self.message.take() {
            self.items[index]["status"] = json!("completed");
            self.event(job,json!({"type":"response.output_text.done","item_id":self.items[index]["id"],"output_index":index,"content_index":0,"text":self.items[index]["content"][0]["text"]}))?;
            self.event(job,json!({"type":"response.content_part.done","item_id":self.items[index]["id"],"output_index":index,"content_index":0,"part":self.items[index]["content"][0]}))?;
            self.event(job,json!({"type":"response.output_item.done","output_index":index,"item":self.items[index]}))?;
        }
        Ok(())
    }
    fn call_delta(&mut self, job: &Job, wire_name: &str, input: &str) -> Result<usize, String> {
        if job
            .request
            .tool_choice
            .as_ref()
            .and_then(|choice| choice.pointer("/function/name"))
            .and_then(Value::as_str)
            .is_some_and(|name| name != wire_name)
        {
            return Err("model did not emit the requested named tool".into());
        }
        let tool = self
            .tools
            .get(wire_name)
            .cloned()
            .ok_or("unknown generated Responses tool")?;
        let index = if let Some((index, name)) = &self.pending {
            if name != wire_name {
                return Err("model changed a streamed tool name".into());
            }
            *index
        } else {
            self.close_message(job)?;
            let index = self.items.len();
            let mut item = json!({"type":if tool.custom {"custom_tool_call"}else{"function_call"},"id":format!("fc-{}-{}",job.id,self.calls),"call_id":format!("call-{}-{}",job.id,self.calls),"name":tool.name,"status":"in_progress"});
            if let Some(namespace) = tool.namespace {
                item["namespace"] = json!(namespace);
            }
            item[if tool.custom { "input" } else { "arguments" }] = json!("");
            self.items.push(item.clone());
            self.pending = Some((index, wire_name.into()));
            self.event(
                job,
                json!({"type":"response.output_item.added","output_index":index,"item":item}),
            )?;
            index
        };
        let field = if tool.custom { "input" } else { "arguments" };
        let emitted = self.items[index][field]
            .as_str()
            .ok_or("missing streamed tool input")?;
        if !input.starts_with(emitted) {
            return Err("final tool input differs from the streamed prefix".into());
        }
        let delta = input[emitted.len()..].to_owned();
        self.items[index][field] = json!(input);
        if !delta.is_empty() {
            self.event(job,json!({"type":if tool.custom {"response.custom_tool_call_input.delta"}else{"response.function_call_arguments.delta"},"output_index":index,"item_id":self.items[index]["id"],"call_id":self.items[index]["call_id"],"delta":delta}))?;
        }
        Ok(index)
    }
    fn advance(&mut self, job: &Job, text: &str, final_text: bool) -> Result<usize, String> {
        let safe = if final_text {
            text.len()
        } else {
            safe_text_end(text, &job.request.stops()?)
        };
        while self.cursor < safe {
            let remaining = &text[self.cursor..safe];
            if !job.request.tools_enabled() {
                self.text(job, remaining)?;
                self.cursor = safe;
                break;
            }
            if remaining.starts_with("<tool_call>") {
                if !self.parallel && self.calls != 0 {
                    return Err(
                        "model emitted multiple calls with parallel_tool_calls=false".into(),
                    );
                }
                let Some(end) = remaining.find("</tool_call>") else {
                    if final_text {
                        return Err("unterminated model tool call".into());
                    }
                    let body = &remaining["<tool_call>".len()..];
                    let prefix = if body.trim_start().starts_with("<function=") {
                        tool_calls::xml_prefix(
                            body,
                            job.request.tools.as_deref().unwrap_or(&[]),
                            false,
                        )?
                    } else {
                        call_prefix(body)
                    };
                    if let Some((name, args)) = prefix {
                        let tool = self
                            .tools
                            .get(&name)
                            .ok_or("unknown generated Responses tool")?;
                        let input = if tool.custom {
                            field_prefix(&args, "input").and_then(string_prefix)
                        } else {
                            Some(args)
                        };
                        if let Some(input) = input {
                            self.call_delta(job, &name, &input)?;
                        }
                    }
                    break;
                };
                let end = end + "</tool_call>".len();
                let (_, calls) = parse_tool_calls(
                    &remaining[..end],
                    job.request.tools.as_deref().unwrap_or(&[]),
                    &job.id,
                )?;
                let call = &calls[0];
                let wire_name = call["function"]["name"]
                    .as_str()
                    .ok_or("missing generated tool name")?;
                if job
                    .request
                    .tool_choice
                    .as_ref()
                    .and_then(|choice| choice.pointer("/function/name"))
                    .and_then(Value::as_str)
                    .is_some_and(|name| name != wire_name)
                {
                    return Err("model did not emit the requested named tool".into());
                }
                let tool = self
                    .tools
                    .get(wire_name)
                    .cloned()
                    .ok_or("unknown generated Responses tool")?;
                let normalized_args = call["function"]["arguments"]
                    .as_str()
                    .ok_or("missing generated arguments")?;
                let body = &remaining["<tool_call>".len()..end - "</tool_call>".len()];
                let prefix = if body.trim_start().starts_with("<function=") {
                    tool_calls::xml_prefix(body, job.request.tools.as_deref().unwrap_or(&[]), true)?
                } else {
                    call_prefix(body)
                };
                let raw_args = match prefix {
                    Some((name, args)) => {
                        if name != wire_name {
                            return Err("model changed a streamed tool name".into());
                        }
                        Some(args)
                    }
                    None => None,
                };
                let args = raw_args.as_deref().unwrap_or(normalized_args);
                let parsed: Value = serde_json::from_str(args).map_err(|e| e.to_string())?;
                let normalized: Value =
                    serde_json::from_str(normalized_args).map_err(|e| e.to_string())?;
                if parsed != normalized {
                    return Err("model changed streamed tool arguments".into());
                }
                let input = if tool.custom {
                    let args: Value = serde_json::from_str(args).map_err(|e| e.to_string())?;
                    required(&args, "input")?.to_owned()
                } else {
                    args.to_owned()
                };
                let index = self.call_delta(job, wire_name, &input)?;
                let mut item = self.items[index].clone();
                let field = if tool.custom { "input" } else { "arguments" };
                item[field] = json!(input);
                item["status"] = json!("completed");
                self.items[index] = item.clone();
                let done = if tool.custom {
                    "response.custom_tool_call_input.done"
                } else {
                    "response.function_call_arguments.done"
                };
                let mut event = json!({"type":done,"output_index":index,"item_id":item["id"],"call_id":item["call_id"]});
                event[field] = json!(input);
                self.event(job, event)?;
                self.event(
                    job,
                    json!({"type":"response.output_item.done","output_index":index,"item":item}),
                )?;
                self.calls += 1;
                self.pending = None;
                self.cursor += end;
            } else {
                let end = if let Some(start) = remaining.find("<tool_call>") {
                    start
                } else if final_text {
                    remaining.len()
                } else {
                    safe_text_end(remaining, &["<tool_call>".into()])
                };
                if end == 0 {
                    break;
                }
                self.text(job, &remaining[..end])?;
                self.cursor += end;
            }
        }
        Ok(self.cursor)
    }
}

// Locate only top-level fields; braces and quotes inside strings never become
// JSON structure. Incomplete values are retained until a later token arrives.
fn value_end(text: &str) -> Option<usize> {
    let first = *text.as_bytes().first()?;
    if first == b'"' {
        let mut escape = false;
        for (index, byte) in text.bytes().enumerate().skip(1) {
            if escape {
                escape = false;
            } else if byte == b'\\' {
                escape = true;
            } else if byte == b'"' {
                return Some(index + 1);
            }
        }
        return None;
    }
    if matches!(first, b'{' | b'[') {
        let mut stack = Vec::new();
        let mut string = false;
        let mut escape = false;
        for (index, byte) in text.bytes().enumerate() {
            if string {
                if escape {
                    escape = false;
                } else if byte == b'\\' {
                    escape = true;
                } else if byte == b'"' {
                    string = false;
                }
            } else {
                match byte {
                    b'"' => string = true,
                    b'{' | b'[' => stack.push(byte),
                    b'}' | b']' => {
                        let expected = if byte == b'}' { b'{' } else { b'[' };
                        if stack.pop() != Some(expected) {
                            return None;
                        }
                        if stack.is_empty() {
                            return Some(index + 1);
                        }
                    }
                    _ => {}
                }
            }
        }
        return None;
    }
    text.bytes()
        .position(|byte| matches!(byte, b',' | b'}' | b']') || byte.is_ascii_whitespace())
}

fn field_prefix<'a>(text: &'a str, wanted: &str) -> Option<&'a str> {
    let mut remaining = text.trim_start().strip_prefix('{')?.trim_start();
    loop {
        if !remaining.starts_with('"') {
            return None;
        }
        let end = value_end(remaining)?;
        let key: String = serde_json::from_str(&remaining[..end]).ok()?;
        remaining = remaining[end..]
            .trim_start()
            .strip_prefix(':')?
            .trim_start();
        if key == wanted {
            return Some(&remaining[..value_end(remaining).unwrap_or(remaining.len())]);
        }
        remaining = remaining[value_end(remaining)?..]
            .trim_start()
            .strip_prefix(',')?
            .trim_start();
    }
}

fn string_prefix(text: &str) -> Option<String> {
    if !text.starts_with('"') {
        return None;
    }
    if let Some(end) = value_end(text) {
        return serde_json::from_str(&text[..end]).ok();
    }
    // A trailing escape or half a UTF-16 surrogate pair can occupy at most
    // twelve bytes. Hold it back instead of emitting a replacement character.
    let mut end = text.len();
    for _ in 0..=12 {
        if end == 0 {
            return None;
        }
        if text.is_char_boundary(end) {
            if let Ok(value) = serde_json::from_str::<String>(&format!("{}\"", &text[..end])) {
                return Some(value);
            }
        }
        end -= 1;
    }
    None
}

fn call_prefix(text: &str) -> Option<(String, String)> {
    let name = field_prefix(text, "name")?;
    let name: String = serde_json::from_str(name).ok()?;
    let args = field_prefix(text, "arguments")?;
    let args = if args.starts_with('"') {
        string_prefix(args)?
    } else {
        args.to_owned()
    };
    Some((name, args))
}

fn output(job: &Job) -> Result<std::sync::MutexGuard<'_, Output>, String> {
    job.responses
        .as_ref()
        .ok_or("missing Responses output state")?
        .lock()
        .map_err(|e| e.to_string())
}
pub(super) fn begin(job: &Job) -> Result<(), String> {
    let mut output = output(job)?;
    if !output.begun {
        output.begun = true;
        let response = output.response(job, "in_progress", Value::Null);
        output.event(job, json!({"type":"response.created","response":response}))?;
        output.event(
            job,
            json!({"type":"response.in_progress","response":response}),
        )?;
    }
    Ok(())
}
pub(super) fn advance(job: &Job, text: &str, final_text: bool) -> Result<usize, String> {
    output(job)?.advance(job, text, final_text)
}
pub(super) fn finish(
    job: &Job,
    text: &str,
    finish: &str,
    prompt: usize,
    generated: usize,
) -> Result<(), String> {
    let mut output = output(job)?;
    output.advance(job, text, true)?;
    output.close_message(job)?;
    let status = if finish == "length" {
        "incomplete"
    } else {
        "completed"
    };
    let response = output.response(
        job,
        status,
        json!({"input_tokens":prompt,"output_tokens":generated,"total_tokens":prompt+generated}),
    );
    if job.request.stream {
        output.event(job,json!({"type":if status=="incomplete" {"response.incomplete"}else{"response.completed"},"response":response}))
    } else {
        job.events
            .blocking_send(Ok(response))
            .map_err(|e| e.to_string())
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn config() -> Config {
        Config {
            context_window: 4096,
            max_output_tokens: 256,
            images: false,
        }
    }

    #[test]
    fn tool_results_preserve_ids_namespaces_and_custom_input() {
        let body = json!({"model":"test","store":false,"tools":[{"type":"namespace","name":"functions","tools":[{"type":"custom","name":"apply_patch","description":"Apply a patch","format":{"type":"grammar","syntax":"lark","definition":"start: /.+/"}}]}],"input":[
            {"type":"message","role":"developer","content":[{"type":"input_text","text":"Keep the workspace"}]},
            {"type":"custom_tool_call","name":"apply_patch","namespace":"functions","call_id":"original-id","input":"*** Begin Patch\n*** End Patch"},
            {"type":"custom_tool_call_output","call_id":"original-id","output":"applied"}
        ]});
        let (request, output) = translate(body, &config()).unwrap();
        let messages = request.messages.unwrap();
        assert_eq!(messages[0]["role"], "system");
        assert_eq!(messages[1]["tool_calls"][0]["id"], "original-id");
        assert_eq!(
            messages[1]["tool_calls"][0]["function"]["name"],
            "functions.apply_patch"
        );
        assert_eq!(messages[2]["tool_call_id"], "original-id");
        assert!(output.tools["functions.apply_patch"].custom);
        assert_eq!(request.max_tokens, Some(256));
    }

    #[test]
    fn stream_emits_text_before_tool_closes_and_preserves_call_replay() {
        let body = json!({"model":"test","stream":true,"tools":[{"type":"function","name":"exec_command","strict":false,"parameters":{"type":"object"}}],"input":"List files"});
        let (request, output) = translate(body, &config()).unwrap();
        let (events, mut receiver) = mpsc::channel(64);
        let (ready, _) = oneshot::channel();
        let job = Job {
            request,
            chat: true,
            id: "resp-test".into(),
            created: 1,
            ready,
            events,
            responses: Some(Arc::new(std::sync::Mutex::new(output))),
        };
        begin(&job).unwrap();
        assert_eq!(advance(&job, "Checking <tool_", false).unwrap(), 9);
        let mut first = Vec::new();
        while let Ok(event) = receiver.try_recv() {
            first.push(event.unwrap());
        }
        assert!(
            first
                .iter()
                .any(|event| event["type"] == "response.output_text.delta"
                    && event["delta"] == "Checking ")
        );
        assert!(
            !first
                .iter()
                .any(|event| event["type"] == "response.completed")
        );
        advance(
            &job,
            "Checking <tool_call>{\"name\":\"exec_command\",\"arguments\":{\"cmd\":\"l",
            false,
        )
        .unwrap();
        let mut partial = Vec::new();
        while let Ok(event) = receiver.try_recv() {
            partial.push(event.unwrap());
        }
        assert!(partial.iter().any(|event| event["type"]
            == "response.function_call_arguments.delta"
            && event["delta"] == "{\"cmd\":\"l"));
        assert!(
            !partial
                .iter()
                .any(|event| event["type"] == "response.output_item.done"
                    && event["item"]["type"] == "function_call")
        );
        let text = "Checking <tool_call>{\"name\":\"exec_command\",\"arguments\":{\"cmd\":\"ls\"}}</tool_call>";
        advance(&job, text, false).unwrap();
        finish(&job, text, "tool_calls", 20, 10).unwrap();
        let mut last = Vec::new();
        while let Ok(event) = receiver.try_recv() {
            last.push(event.unwrap());
        }
        let call = last
            .iter()
            .find(|event| {
                event["type"] == "response.output_item.done"
                    && event["item"]["type"] == "function_call"
            })
            .unwrap();
        assert_eq!(call["item"]["call_id"], "call-resp-test-0");
        assert_eq!(call["item"]["arguments"], "{\"cmd\":\"ls\"}");
        let terminal = last.last().unwrap();
        assert_eq!(terminal["type"], "response.completed");
        assert_eq!(terminal["response"]["usage"]["total_tokens"], 30);
        assert_eq!(terminal["response"]["end_turn"], false);
    }

    #[test]
    fn unsupported_state_and_unmatched_tool_outputs_are_errors() {
        assert!(
            translate(
                json!({"model":"test","store":true,"input":"hello"}),
                &config()
            )
            .is_err()
        );
        assert!(
            translate(
                json!({"model":"test","previous_response_id":"missing","input":[]}),
                &config()
            )
            .is_err()
        );
        assert!(translate(json!({"model":"test","input":[{"type":"function_call_output","call_id":"missing","output":"ok"}]}),&config()).is_err());
    }

    #[test]
    fn partial_json_keeps_unicode_escapes_and_structural_characters() {
        assert_eq!(string_prefix(r#""line\n\uD83D"#), Some("line\n".into()));
        assert_eq!(
            string_prefix(r#""line\n\uD83D\uDE00"#),
            Some("line\n😀".into())
        );
        let prefix = r#"{"arguments":{"input":"braces } [ and quote \""},"name":"apply_patch"}"#;
        let (name, args) = call_prefix(prefix).unwrap();
        assert_eq!(name, "apply_patch");
        assert_eq!(
            string_prefix(field_prefix(&args, "input").unwrap()),
            Some("braces } [ and quote \"".into())
        );
    }
}
