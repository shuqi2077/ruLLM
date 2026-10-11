use super::*;

fn parameter_schema<'a>(tool: &'a Value, name: &str) -> Result<&'a Value, String> {
    tool.pointer("/function/parameters/properties")
        .and_then(|properties| properties.get(name))
        .ok_or_else(|| format!("XML tool parameter {name} has no declared schema"))
}

fn string_parameter(schema: &Value) -> Result<bool, String> {
    if let Some(kind) = schema.get("type") {
        if let Some(kind) = kind.as_str() {
            return Ok(kind == "string");
        }
        if let Some(kinds) = kind.as_array() {
            if kinds.iter().any(|kind| kind.as_str() == Some("string")) {
                if kinds
                    .iter()
                    .any(|kind| !matches!(kind.as_str(), Some("string" | "null")))
                {
                    return Err("ambiguous XML string/non-string parameter schema".into());
                }
                return Ok(true);
            }
            return Ok(false);
        }
    }
    for key in ["anyOf", "oneOf"] {
        if let Some(variants) = schema.get(key).and_then(Value::as_array) {
            let kinds = variants
                .iter()
                .filter(|variant| variant.get("type").and_then(Value::as_str) != Some("null"))
                .map(string_parameter)
                .collect::<Result<Vec<_>, _>>()?;
            if let Some(first) = kinds.first() {
                if kinds.iter().all(|kind| kind == first) {
                    return Ok(*first);
                }
            }
            return Err("ambiguous XML parameter schema".into());
        }
    }
    Err("XML tool parameters require an explicit supported schema type".into())
}

fn unpad(value: &str) -> &str {
    let value = value
        .strip_prefix("\r\n")
        .or_else(|| value.strip_prefix('\n'))
        .unwrap_or(value);
    value
        .strip_suffix("\r\n")
        .or_else(|| value.strip_suffix('\n'))
        .unwrap_or(value)
}

pub(super) fn xml_prefix(
    text: &str,
    tools: &[Value],
    complete: bool,
) -> Result<Option<(String, String)>, String> {
    let Some(rest) = text.trim_start().strip_prefix("<function=") else {
        return Ok(None);
    };
    let Some(end) = rest.find('>') else {
        return Ok(None);
    };
    let name = &rest[..end];
    let tool = tools
        .iter()
        .find(|tool| tool.pointer("/function/name").and_then(Value::as_str) == Some(name))
        .ok_or_else(|| format!("model selected an undeclared tool {name}"))?;
    let mut rest = rest[end + 1..].trim_start();
    let mut entries = Vec::new();
    let mut names = std::collections::BTreeSet::new();
    loop {
        if let Some(trailing) = rest.strip_prefix("</function>") {
            if !trailing.trim().is_empty() {
                return Err("unexpected XML tool suffix".into());
            }
            return Ok(Some((name.into(), format!("{{{}}}", entries.join(",")))));
        }
        let Some(parameter) = rest.strip_prefix("<parameter=") else {
            if complete {
                return Err("XML tool call has no closing function tag".into());
            }
            return Ok(Some((name.into(), format!("{{{}", entries.join(",")))));
        };
        let Some(end) = parameter.find('>') else {
            return Ok(Some((name.into(), format!("{{{}", entries.join(",")))));
        };
        let key = &parameter[..end];
        if !names.insert(key.to_owned()) {
            return Err(format!("duplicate XML parameter {key}"));
        }
        let schema = parameter_schema(tool, key)?;
        let is_string = string_parameter(schema)?;
        let body = &parameter[end + 1..];
        let close = body.find("</parameter>");
        let raw = match close {
            Some(end) => unpad(&body[..end]),
            None if complete => return Err("unterminated XML tool parameter".into()),
            None => unpad(&body[..safe_text_end(body, &["</parameter>".into()])]),
        };
        let key = serde_json::to_string(key).map_err(|e| e.to_string())?;
        let encoded = if is_string {
            let mut value = serde_json::to_string(raw).map_err(|e| e.to_string())?;
            if close.is_none() {
                value.pop();
            }
            value
        } else if close.is_some() {
            let value = match raw.trim() {
                "True" => json!(true),
                "False" => json!(false),
                value => serde_json::from_str::<Value>(value)
                    .map_err(|e| format!("invalid XML parameter JSON: {e}"))?,
            };
            if matches!(
                schema.get("type").and_then(Value::as_str),
                Some("object" | "array")
            ) {
                raw.trim_start().to_owned()
            } else {
                serde_json::to_string(&value).map_err(|e| e.to_string())?
            }
        } else {
            let prefix = if matches!(
                schema.get("type").and_then(Value::as_str),
                Some("object" | "array")
            ) {
                raw.trim_start()
            } else {
                ""
            };
            prefix.to_owned()
        };
        entries.push(format!("{key}:{encoded}"));
        if let Some(end) = close {
            rest = body[end + "</parameter>".len()..].trim_start();
        } else {
            return Ok(Some((name.into(), format!("{{{}", entries.join(",")))));
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn xml_strings_numbers_and_stream_prefix_preserve_arguments() {
        let tools = vec![
            json!({"type":"function","function":{"name":"exec_command","parameters":{"type":"object","properties":{"cmd":{"type":"string"},"yield_time_ms":{"type":"integer"}}}}}),
        ];
        let prefix = "<function=exec_command>\n<parameter=cmd>\n  echo \"😀\"\n";
        let (_, partial) = xml_prefix(prefix, &tools, false).unwrap().unwrap();
        let text = format!(
            "{prefix}</parameter>\n<parameter=yield_time_ms>\n1000\n</parameter>\n</function>"
        );
        let (_, arguments) = xml_prefix(&text, &tools, true).unwrap().unwrap();
        assert!(arguments.starts_with(&partial));
        let value: Value = serde_json::from_str(&arguments).unwrap();
        assert_eq!(value["cmd"], "  echo \"😀\"");
        assert_eq!(value["yield_time_ms"], 1000);
        let (_, calls) =
            parse_tool_calls(&format!("<tool_call>{text}</tool_call>"), &tools, "test").unwrap();
        assert_eq!(
            serde_json::from_str::<Value>(calls[0]["function"]["arguments"].as_str().unwrap())
                .unwrap(),
            value
        );
    }
}
