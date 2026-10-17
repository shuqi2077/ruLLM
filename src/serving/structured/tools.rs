use super::*;

pub(super) fn requested(request: &GenerateRequest) -> bool {
    request.tools_enabled() && (request.tools.as_ref().is_some_and(|tools| tools.iter()
        .any(|tool| tool.pointer("/function/strict").and_then(Value::as_bool) == Some(true)))
        || request.tool_choice.as_ref().is_some_and(|choice| choice.as_str() == Some("required") || choice.is_object())
        || request.parallel_tool_calls == Some(false))
}

fn selected<'a>(request: &'a GenerateRequest) -> Result<Vec<&'a Value>, String> {
    let name = request.tool_choice.as_ref().and_then(|v| v.pointer("/function/name")).and_then(Value::as_str);
    let mut names = std::collections::BTreeSet::new();
    let mut selected = Vec::new();
    for tool in request.tools.as_deref().unwrap_or(&[]) {
        let function = tool.get("function").ok_or("missing tool function")?;
        let tool_name = function.get("name").and_then(Value::as_str).filter(|v| !v.is_empty())
            .ok_or("missing tool name")?;
        if !names.insert(tool_name) { return Err(format!("duplicate tool {tool_name}")); }
        if let Some(strict) = function.get("strict") {
            if !strict.is_boolean() { return Err("tool strict must be a boolean".into()); }
        }
        if function.get("strict").and_then(Value::as_bool) == Some(true)
            && function.get("parameters").is_none_or(|s| !s.is_object() && !s.is_boolean()) {
            return Err(format!("strict tool {tool_name} requires a JSON Schema"));
        }
        if name.is_none_or(|name| name == tool_name) { selected.push(function); }
    }
    if selected.is_empty() { return Err("tool choice does not select a declared function".into()); }
    Ok(selected)
}

pub(super) fn validate(request: &GenerateRequest) -> Result<(), String> {
    selected(request).map(|_| ())
}

#[cfg(feature = "structured-output")]
fn literal(text: &str) -> String { serde_json::to_string(text).expect("string serialization") }

#[cfg(feature = "structured-output")]
fn marker(tokenizer: &tokenizers::Tokenizer, text: &str) -> String {
    let plain = literal(text);
    match tokenizer.get_added_tokens_decoder().into_iter().find(|(_, t)| t.special && t.content == text) {
        Some((id, _)) => format!("({plain} | <[{id}]>)"),
        None => plain,
    }
}

#[cfg(feature = "structured-output")]
pub(super) fn grammar(request: &GenerateRequest, tokenizer: &tokenizers::Tokenizer, eos: &[i32])
    -> Result<llguidance::api::TopLevelGrammar, String> {
    let functions = selected(request)?;
    let required = request.tool_choice.as_ref().is_some_and(|v| v.as_str() == Some("required") || v.is_object());
    let parallel = request.parallel_tool_calls != Some(false);
    let root = match (required, parallel) {
        (false, true) => "text0 (call text0)*",
        (true, true) => "text0 call text0 (call text0)*",
        (false, false) => "text0 (call text0)?",
        (true, false) => "text0 call text0",
    };
    let mut grammar = format!("start: {root}\ncall: {} ws payload ws {}\nws: /[ \\t\\r\\n]+/ | \"\"\n",
        marker(tokenizer, "<tool_call>"), marker(tokenizer, "</tool_call>"));
    let alternatives: Vec<_> = (0..functions.len()).map(|i| format!("payload_{i}")).collect();
    grammar.push_str(&format!("payload: {}\n", alternatives.join(" | ")));
    for (index, function) in functions.into_iter().enumerate() {
        let name = function["name"].as_str().ok_or("invalid tool name")?;
        let schema = if function.get("strict").and_then(Value::as_bool) == Some(true) {
            function.get("parameters").ok_or("missing strict parameters")?.clone()
        } else { Value::Bool(true) };
        // Compile parameters as a separate schema root: local $ref/$defs keep
        // their original scope rather than being moved under arguments.
        let prefix = format!("{{\"name\":{},\"arguments\":", literal(name));
        grammar.push_str(&format!("payload_{index}: {} %json {} ws \"}}\"\n", literal(&prefix), schema));
    }
    // Ordinary text is the DFA language excluding the opening tool marker.
    // A permissive `.*` alternative would let malformed tool calls bypass the
    // schema branch, including markers split across multiple BPE tokens.
    let delimiter = "<tool_call>".as_bytes();
    let specials: Vec<_> = tokenizer.get_added_tokens_decoder().into_iter()
        .filter(|(id, t)| t.special && !eos.contains(&(*id as i32))
            && t.content != "<tool_call>" && t.content != "</tool_call>")
        .map(|(id, _)| id.to_string()).collect();
    for state in 0..delimiter.len() {
        let mut alternatives = vec!["\"\"".to_owned()];
        if state == 0 {
            alternatives.push("/[^<]+/ text0".into());
        } else {
            let expected = delimiter[state] as char;
            alternatives.push(format!("/[^<{}]+/ text0", expected));
            if state + 1 < delimiter.len() {
                alternatives.push(format!("{} text{}", literal(&expected.to_string()), state + 1));
            }
        }
        alternatives.push("\"<\" text1".into());
        if !specials.is_empty() { alternatives.push(format!("<[{}]> text0", specials.join(","))); }
        grammar.push_str(&format!("text{state}: {}\n", alternatives.join(" | ")));
    }
    Ok(llguidance::api::TopLevelGrammar::from_lark(grammar))
}

#[cfg(all(test, feature = "structured-output"))]
mod tests {
    use super::*;

    fn tokenizer(special_tags: bool) -> (tokenizers::Tokenizer, i32) {
        let mut alphabet: Vec<_> = tokenizers::pre_tokenizers::byte_level::ByteLevel::alphabet().into_iter().collect();
        alphabet.sort_unstable();
        let vocabulary: serde_json::Map<String, Value> = alphabet.into_iter().enumerate()
            .map(|(id, c)| (c.to_string(), json!(id))).collect();
        let model: tokenizers::models::bpe::BPE = serde_json::from_value(json!({
            "type":"BPE", "vocab":vocabulary, "merges":[], "unk_token":null,
            "continuing_subword_prefix":null, "end_of_word_suffix":null,
            "fuse_unk":false, "byte_fallback":false, "dropout":null
        })).unwrap();
        let mut tokenizer = tokenizers::Tokenizer::new(model);
        tokenizer.with_pre_tokenizer(Some(tokenizers::pre_tokenizers::byte_level::ByteLevel::new(false, false, false)));
        tokenizer.with_decoder(Some(tokenizers::decoders::byte_level::ByteLevel::default()));
        let mut added = vec![tokenizers::AddedToken::from("<eos>", true)];
        if special_tags {
            added.push(tokenizers::AddedToken::from("<tool_call>", true));
            added.push(tokenizers::AddedToken::from("</tool_call>", true));
        }
        tokenizer.add_special_tokens(&added);
        let eos = tokenizer.token_to_id("<eos>").unwrap() as i32;
        (tokenizer, eos)
    }

    fn request(choice: Value, parallel: bool) -> GenerateRequest {
        serde_json::from_value(json!({"model":"test", "messages":[{"role":"user","content":"test"}],
            "tool_choice":choice,"parallel_tool_calls":parallel,"tools":[
                {"type":"function","function":{"name":"first","strict":true,"parameters":{
                    "$defs":{"small":{"type":"integer","minimum":0,"maximum":3}},
                    "type":"object","properties":{"n":{"$ref":"#/$defs/small"}},"required":["n"],"additionalProperties":false}}},
                {"type":"function","function":{"name":"second","strict":true,"parameters":{
                    "type":"object","properties":{"s":{"type":"string"}},"required":["s"],"additionalProperties":false}}}
            ]})).unwrap()
    }

    fn accepts(request: &GenerateRequest, text: &str, specials: bool) -> bool {
        let (tokenizer, eos) = tokenizer(specials);
        let mut decoder = Decoder::new(request, &tokenizer, &[eos]).unwrap().unwrap();
        for &token in tokenizer.encode(text, false).unwrap().get_ids() {
            let bias = decoder.bias(tokenizer.get_vocab_size(true)).unwrap();
            if !bias[token as usize].is_finite() { return false; }
            decoder.consume(token as i32).unwrap();
        }
        decoder.bias(tokenizer.get_vocab_size(true)).unwrap()[eos as usize].is_finite()
    }

    #[test]
    fn strict_schema_masks_names_types_bounds_and_additional_properties_without_text_bypass() {
        let request = request(json!("auto"), true);
        for special in [false, true] {
            for valid in ["ordinary text", "a <tool_cal is ordinary text",
                "Checking <tool_call>{\"name\":\"first\",\"arguments\":{\"n\":2}}</tool_call> done"] {
                assert!(accepts(&request, valid, special), "{valid}");
            }
            for invalid in [
                "<tool_call>{\"name\":\"unknown\",\"arguments\":{}}</tool_call>",
                "<tool_call>{\"name\":\"first\",\"arguments\":{\"n\":\"2\"}}</tool_call>",
                "<tool_call>{\"name\":\"first\",\"arguments\":{\"n\":9}}</tool_call>",
                "<tool_call>{\"name\":\"first\",\"arguments\":{\"n\":2,\"other\":0}}</tool_call>",
                "text <tool_call>not json</tool_call>",
                "<tool_call>{\"name\":\"first\",\"arguments\":{}",
            ] {
                assert!(!accepts(&request, invalid, special), "{invalid}");
            }
        }
        assert!(accepts(&request,
            "<tool_call>{\"name\":\"second\",\"arguments\":{\"s\":\"</tool_call>\"}}</tool_call>", false));
    }

    #[test]
    fn required_named_and_single_call_choices_are_enforced_before_eos() {
        let first = "<tool_call>{\"name\":\"first\",\"arguments\":{\"n\":1}}</tool_call>";
        let second = "<tool_call>{\"name\":\"second\",\"arguments\":{\"s\":\"ok\"}}</tool_call>";
        let required = request(json!("required"), true);
        assert!(!accepts(&required, "ordinary text", false));
        assert!(accepts(&required, &format!("reason {first} {second}"), false));
        let named = request(json!({"type":"function","function":{"name":"second"}}), true);
        assert!(!accepts(&named, first, false));
        assert!(accepts(&named, second, false));
        let single = request(json!("required"), false);
        assert!(accepts(&single, first, false));
        assert!(!accepts(&single, &format!("{first}{second}"), false));
        let mut disabled = request(json!("none"), false);
        assert!(!super::requested(&disabled));
        disabled.tool_choice = Some(json!({"type":"function","function":{"name":"missing"}}));
        assert!(validate(&disabled).is_err());
    }
}
