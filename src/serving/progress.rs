use super::*;

pub(super) fn eos_ids(
    chat: bool,
    eos: &[i32],
    tokenizer: &tokenizers::Tokenizer,
    config: &Value,
) -> Result<Vec<i32>, String> {
    let mut ids = eos.to_vec();
    if chat {
        if let Some(value) = config.get("eos_token").filter(|value| !value.is_null()) {
            let token = value
                .as_str()
                .or_else(|| value.get("content").and_then(Value::as_str))
                .ok_or("checkpoint tokenizer eos_token must contain text")?;
            let id = tokenizer
                .token_to_id(token)
                .ok_or("checkpoint tokenizer eos_token is absent from its vocabulary")?;
            let id = i32::try_from(id).map_err(|error| error.to_string())?;
            if !ids.contains(&id) {
                ids.push(id);
            }
        }
    }
    Ok(ids)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn chat_uses_checkpoint_tokenizer_eos_without_changing_completions() {
        let vocabulary = [("<|endoftext|>".into(), 0), ("<|im_end|>".into(), 1)]
            .into_iter()
            .collect();
        let model = tokenizers::models::wordlevel::WordLevel::builder()
            .vocab(vocabulary)
            .unk_token("<|endoftext|>".into())
            .build()
            .unwrap();
        let tokenizer = tokenizers::Tokenizer::new(model);
        let config = json!({"eos_token":"<|im_end|>"});
        assert_eq!(
            eos_ids(true, &[0], &tokenizer, &config).unwrap(),
            vec![0, 1]
        );
        assert_eq!(eos_ids(false, &[0], &tokenizer, &config).unwrap(), vec![0]);
        assert_eq!(
            eos_ids(true, &[0, 1], &tokenizer, &config).unwrap(),
            vec![0, 1]
        );
        let config = json!({"eos_token":{"content":"<|im_end|>"}});
        assert_eq!(
            eos_ids(true, &[0], &tokenizer, &config).unwrap(),
            vec![0, 1]
        );
    }
}

pub(super) fn prompt_ids(
    job: &Job,
    tokenizer: &tokenizers::Tokenizer,
    config: &Value,
) -> Result<Vec<i32>, String> {
    let prompt = if job.chat {
        render_chat(
            config
                .get("chat_template")
                .and_then(Value::as_str)
                .ok_or("checkpoint has no string chat_template")?,
            config,
            &job.request,
        )?
    } else {
        job.request
            .prompt
            .clone()
            .ok_or("completion prompt is missing")?
    };
    tokenizer
        .encode(prompt, false)
        .map_err(|e| e.to_string())?
        .get_ids()
        .iter()
        .map(|&id| i32::try_from(id).map_err(|e| e.to_string()))
        .collect()
}

pub(super) fn configs(
    request: &GenerateRequest,
    eos: &[i32],
) -> (GreedyGenerationConfig, SamplingGenerationConfig) {
    let max = request
        .max_tokens
        .or(request.max_completion_tokens)
        .unwrap_or(16);
    (
        GreedyGenerationConfig {
            max_new_tokens: max,
            eos_token_ids: eos.to_vec(),
        },
        SamplingGenerationConfig {
            max_new_tokens: max,
            eos_token_ids: eos.to_vec(),
            sampling: SamplingConfig {
                temperature: request.temperature.unwrap_or(1.0),
                top_k: request.top_k,
                top_p: request.top_p.unwrap_or(1.0),
                seed: request.seed,
            },
        },
    )
}

pub(super) fn accept(job: Job) -> Option<Job> {
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
        return None;
    }
    let (ready, _) = oneshot::channel();
    Some(Job {
        ready,
        request,
        chat,
        id,
        created,
        events,
        responses,
    })
}

pub(super) struct ResponseProgress<'a> {
    decoder: Box<dyn FnMut(u32) -> Result<Option<String>, String> + 'a>,
    ids: Vec<u32>,
    text: String,
    emitted: usize,
    stopped: bool,
    stops: Vec<String>,
    ignored: std::collections::BTreeSet<u32>,
    tools: bool,
}
impl<'a> ResponseProgress<'a> {
    pub(super) fn new(job: &Job, tokenizer: &'a tokenizers::Tokenizer) -> Result<Self, String> {
        let tools = job.request.tools_enabled();
        let ignored = ignored_tool_tokens(tokenizer, tools);
        let decoder_ignored = ignored.clone();
        let mut decoder = tokenizer.decode_stream(!tools);
        if job.responses.is_some() {
            responses::begin(job)?;
        } else if job.request.stream && job.chat {
            job.events.blocking_send(Ok(chunk(job,json!({"index":0,"delta":{"role":"assistant","content":""},"finish_reason":null,"logprobs":null}),None)))
                .map_err(|e|e.to_string())?;
        }
        Ok(Self {
            decoder: Box::new(move |id| {
                if decoder_ignored.contains(&id) {
                    Ok(None)
                } else {
                    decoder.step(id).map_err(|e| e.to_string())
                }
            }),
            ids: Vec::new(),
            text: String::new(),
            emitted: 0,
            stopped: false,
            stops: job.request.stops()?,
            ignored,
            tools,
        })
    }
    pub(super) fn push(&mut self, job: &Job, token: i32) -> Result<bool, String> {
        let token = u32::try_from(token).map_err(|e| e.to_string())?;
        self.ids.push(token);
        if let Some(piece) = (self.decoder)(token)? {
            self.text.push_str(&piece);
        }
        if let Some(end) = self
            .stops
            .iter()
            .filter_map(|stop| self.text.find(stop))
            .min()
        {
            self.text.truncate(end);
            self.stopped = true;
        }
        if job.responses.is_some() && job.request.stream {
            self.emitted = responses::advance(job, &self.text, self.stopped)?;
        } else if job.request.stream && !job.request.tools_enabled() {
            let end = if self.stopped {
                self.text.len()
            } else {
                safe_text_end(&self.text, &self.stops)
            };
            if end > self.emitted {
                job.events
                    .blocking_send(Ok(text_chunk(job, &self.text[self.emitted..end])))
                    .map_err(|e| e.to_string())?;
                self.emitted = end;
            }
        }
        Ok(self.stopped)
    }
    pub(super) fn finish(
        mut self,
        job: &Job,
        tokenizer: &tokenizers::Tokenizer,
        prompt_tokens: usize,
        reason: Option<GenerationFinishReason>,
    ) -> Result<(), String> {
        if job.events.is_closed() {
            return Ok(());
        }
        if !self.stopped {
            let visible = self
                .ids
                .iter()
                .copied()
                .filter(|id| !self.ignored.contains(id))
                .collect::<Vec<_>>();
            let complete = tokenizer
                .decode(&visible, !self.tools)
                .map_err(|e| e.to_string())?;
            if !complete.starts_with(&self.text[..self.emitted]) {
                return Err("final tokenizer output differs from emitted stream prefix".into());
            }
            self.text = complete;
            if let Some(end) = self
                .stops
                .iter()
                .filter_map(|stop| self.text.find(stop))
                .min()
            {
                if end < self.emitted {
                    return Err("final tokenizer stop point precedes emitted text".into());
                }
                self.text.truncate(end);
                self.stopped = true;
            }
        }
        finish_response(
            job,
            self.text,
            self.emitted,
            self.stopped,
            reason,
            prompt_tokens,
            self.ids.len(),
        )
    }
}
