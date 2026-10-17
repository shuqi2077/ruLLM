//! Tokenized training examples, JSONL ingestion and a resumable, seeded,
//! optionally rank-sharded micro-batch loader.

use crate::{Qwen2ChatMessage, Qwen2ChatRole, render_qwen2_chat_messages};
use serde_json::Value;
use std::io::{BufRead, BufReader};
use std::path::Path;
use tokenizers::Tokenizer;

/// Label value excluded from the loss (Hugging Face convention).
pub const IGNORE_INDEX: i32 = -100;

/// One training sequence. `labels` is aligned with `input_ids` (the
/// next-token shift happens during batching); [`IGNORE_INDEX`] masks tokens.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct SftExample {
    pub input_ids: Vec<i32>,
    pub labels: Vec<i32>,
}

impl SftExample {
    /// Every token is a training target (continued pre-training).
    pub fn causal(input_ids: Vec<i32>) -> Self {
        Self {
            labels: input_ids.clone(),
            input_ids,
        }
    }

    /// Number of loss-bearing targets after the next-token shift.
    pub fn target_count(&self) -> usize {
        self.labels.iter().skip(1).filter(|&&label| label >= 0).count()
    }
}

/// How JSONL records are rendered into tokens.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum TextFormat {
    /// `{"messages": [...]}` with the Qwen2 ChatML template; only assistant
    /// turns are trained. Also accepts `{"prompt", "completion"}` and
    /// `{"text"}`.
    QwenChat,
    /// `{"prompt", "completion"}` (completion trained) or `{"text"}` (all
    /// trained), tokenized without a chat template.
    Plain,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct DataError(pub String);

impl std::fmt::Display for DataError {
    fn fmt(&self, formatter: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        formatter.write_str(&self.0)
    }
}

impl std::error::Error for DataError {}

fn encode(tokenizer: &Tokenizer, text: &str) -> Result<Vec<i32>, DataError> {
    tokenizer
        .encode(text, false)
        .map_err(|error| DataError(format!("tokenizer encode failed: {error}")))?
        .get_ids()
        .iter()
        .map(|&id| i32::try_from(id).map_err(|_| DataError("token id exceeds i32".into())))
        .collect()
}

fn chat_role(value: &Value) -> Result<Qwen2ChatRole, DataError> {
    match value.get("role").and_then(Value::as_str) {
        Some("system") => Ok(Qwen2ChatRole::System),
        Some("user") => Ok(Qwen2ChatRole::User),
        Some("assistant") => Ok(Qwen2ChatRole::Assistant),
        other => Err(DataError(format!("unsupported chat role {other:?}"))),
    }
}

/// Tokenize a chat so that exactly the assistant turns (including their
/// closing `<|im_end|>`) carry labels. Each turn boundary is located by
/// tokenizing the rendered prefix, which must be a token prefix of the
/// rendered conversation.
fn chat_example(tokenizer: &Tokenizer, messages: &[Value]) -> Result<SftExample, DataError> {
    let messages = messages
        .iter()
        .map(|message| {
            let content = message
                .get("content")
                .and_then(Value::as_str)
                .ok_or_else(|| DataError("chat content must be a string".into()))?;
            Ok(Qwen2ChatMessage::new(chat_role(message)?, content))
        })
        .collect::<Result<Vec<_>, DataError>>()?;
    let render = |messages: &[Qwen2ChatMessage], generation_prompt: bool| {
        render_qwen2_chat_messages(messages, generation_prompt).map_err(|error| DataError(error.0))
    };
    let input_ids = encode(tokenizer, &render(&messages, false)?)?;
    let mut labels = vec![IGNORE_INDEX; input_ids.len()];
    for (index, message) in messages.iter().enumerate() {
        if message.role != Qwen2ChatRole::Assistant {
            continue;
        }
        let start = encode(tokenizer, &render(&messages[..index], true)?)?;
        let end = encode(tokenizer, &render(&messages[..=index], false)?)?;
        if !input_ids.starts_with(&start) || !input_ids.starts_with(&end) {
            return Err(DataError(
                "chat turn boundary does not align with the tokenized conversation".into(),
            ));
        }
        // `end` stops after "<|im_end|>\n"; train through "<|im_end|>".
        let stop = end.len().saturating_sub(1).max(start.len());
        labels[start.len()..stop].copy_from_slice(&input_ids[start.len()..stop]);
    }
    Ok(SftExample { input_ids, labels })
}

/// Tokenized examples, truncated to `max_tokens` (`max_sequence + 1` input
/// tokens yield `max_sequence` shifted targets).
#[derive(Debug, Clone, Default)]
pub struct SftDataset {
    pub examples: Vec<SftExample>,
}

impl SftDataset {
    pub fn new(examples: Vec<SftExample>) -> Self {
        Self { examples }
    }

    /// Parse one JSON record into an example.
    pub fn parse_record(
        record: &Value,
        tokenizer: &Tokenizer,
        format: TextFormat,
        eos_token_id: Option<i32>,
    ) -> Result<SftExample, DataError> {
        if format == TextFormat::QwenChat
            && let Some(Value::Array(messages)) = record.get("messages")
        {
            return chat_example(tokenizer, messages);
        }
        let eos = eos_token_id.into_iter();
        if let (Some(prompt), Some(completion)) = (
            record.get("prompt").and_then(Value::as_str),
            record.get("completion").and_then(Value::as_str),
        ) {
            let prompt = encode(tokenizer, prompt)?;
            let completion: Vec<i32> = encode(tokenizer, completion)?.into_iter().chain(eos).collect();
            let labels = std::iter::repeat_n(IGNORE_INDEX, prompt.len())
                .chain(completion.iter().copied())
                .collect();
            let input_ids = prompt.into_iter().chain(completion).collect();
            return Ok(SftExample { input_ids, labels });
        }
        if let Some(text) = record.get("text").and_then(Value::as_str) {
            let ids = encode(tokenizer, text)?.into_iter().chain(eos).collect();
            return Ok(SftExample::causal(ids));
        }
        Err(DataError(
            "record needs \"messages\", \"prompt\"+\"completion\" or \"text\"".into(),
        ))
    }

    /// Read a JSONL file. Records shorter than two tokens or without any
    /// target are skipped; their count is returned alongside the dataset.
    pub fn from_jsonl(
        path: impl AsRef<Path>,
        tokenizer: &Tokenizer,
        format: TextFormat,
        eos_token_id: Option<i32>,
        max_tokens: usize,
    ) -> Result<(Self, usize), DataError> {
        let path = path.as_ref();
        let file = std::fs::File::open(path)
            .map_err(|error| DataError(format!("cannot open {}: {error}", path.display())))?;
        let mut examples = Vec::new();
        let mut skipped = 0usize;
        for (line_number, line) in BufReader::new(file).lines().enumerate() {
            let line = line.map_err(|error| DataError(format!("read failed: {error}")))?;
            if line.trim().is_empty() {
                continue;
            }
            let record: Value = serde_json::from_str(&line).map_err(|error| {
                DataError(format!("{}:{}: invalid JSON: {error}", path.display(), line_number + 1))
            })?;
            let mut example = Self::parse_record(&record, tokenizer, format, eos_token_id)
                .map_err(|error| DataError(format!("{}:{}: {error}", path.display(), line_number + 1)))?;
            example.input_ids.truncate(max_tokens);
            example.labels.truncate(max_tokens);
            if example.input_ids.len() < 2 || example.target_count() == 0 {
                skipped += 1;
                continue;
            }
            examples.push(example);
        }
        Ok((Self { examples }, skipped))
    }

    pub fn len(&self) -> usize {
        self.examples.len()
    }

    pub fn is_empty(&self) -> bool {
        self.examples.is_empty()
    }
}

/// One padded micro-batch: `tokens [batch, length]` row-major and
/// `targets [batch * length]` (shifted labels, padding = [`IGNORE_INDEX`]).
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct SftBatch {
    pub batch: usize,
    pub length: usize,
    pub tokens: Vec<i32>,
    pub targets: Vec<i32>,
    pub target_count: usize,
}

impl SftBatch {
    /// Right-pad shifted sequences with `pad_token_id`.
    pub fn collate(examples: &[&SftExample], pad_token_id: i32) -> Self {
        let length = examples
            .iter()
            .map(|example| example.input_ids.len().saturating_sub(1))
            .max()
            .unwrap_or(0)
            .max(1);
        let mut tokens = Vec::with_capacity(examples.len() * length);
        let mut targets = Vec::with_capacity(examples.len() * length);
        for example in examples {
            let inputs = &example.input_ids[..example.input_ids.len().saturating_sub(1)];
            let labels = example.labels.get(1..).unwrap_or(&[]);
            tokens.extend_from_slice(inputs);
            targets.extend_from_slice(labels);
            tokens.extend(std::iter::repeat_n(pad_token_id, length - inputs.len()));
            targets.extend(std::iter::repeat_n(IGNORE_INDEX, length - labels.len()));
        }
        let target_count = targets.iter().filter(|&&target| target >= 0).count();
        Self {
            batch: examples.len(),
            length,
            tokens,
            targets,
            target_count,
        }
    }
}

/// Position of an [`SftLoader`]; persisted in checkpoints.
#[derive(Debug, Clone, Copy, PartialEq, Eq, serde::Serialize, serde::Deserialize)]
pub struct SftLoaderState {
    pub epoch: u64,
    pub cursor: usize,
    pub seed: u64,
}

/// SplitMix64, used for a deterministic per-epoch permutation.
fn split_mix(state: &mut u64) -> u64 {
    *state = state.wrapping_add(0x9E37_79B9_7F4A_7C15);
    let mut value = *state;
    value = (value ^ (value >> 30)).wrapping_mul(0xBF58_476D_1CE4_E5B9);
    value = (value ^ (value >> 27)).wrapping_mul(0x94D0_49BB_1331_11EB);
    value ^ (value >> 31)
}

/// Endless micro-batch stream over a dataset. Each epoch is a seeded
/// permutation; within a window of `length_window` permuted examples the
/// order is sorted by length to reduce padding. With `world_size > 1` rank
/// `r` takes every `world_size`-th micro-batch starting at `r`.
#[derive(Debug, Clone)]
pub struct SftLoader {
    dataset: SftDataset,
    micro_batch: usize,
    pad_token_id: i32,
    length_window: usize,
    rank: usize,
    world_size: usize,
    state: SftLoaderState,
    order: Vec<usize>,
}

impl SftLoader {
    pub fn new(dataset: SftDataset, micro_batch: usize, pad_token_id: i32, seed: u64) -> Self {
        assert!(!dataset.is_empty(), "training dataset is empty");
        assert!(micro_batch > 0, "micro batch must be positive");
        let mut loader = Self {
            dataset,
            micro_batch,
            pad_token_id,
            length_window: 64,
            rank: 0,
            world_size: 1,
            state: SftLoaderState { epoch: 0, cursor: 0, seed },
            order: Vec::new(),
        };
        loader.reorder();
        loader
    }

    /// Data-parallel sharding of micro-batches.
    pub fn with_shard(mut self, rank: usize, world_size: usize) -> Self {
        assert!(world_size > 0 && rank < world_size, "invalid data shard");
        self.rank = rank;
        self.world_size = world_size;
        self
    }

    pub fn with_length_window(mut self, window: usize) -> Self {
        self.length_window = window.max(1);
        self.reorder();
        self
    }

    pub const fn state(&self) -> SftLoaderState {
        self.state
    }

    /// Resume at a saved position (same dataset, micro batch and seed).
    pub fn restore(&mut self, state: SftLoaderState) {
        self.state = state;
        self.reorder();
    }

    pub fn dataset(&self) -> &SftDataset {
        &self.dataset
    }

    /// Micro-batches per epoch across all ranks.
    pub fn batches_per_epoch(&self) -> usize {
        self.dataset.len().div_ceil(self.micro_batch)
    }

    fn reorder(&mut self) {
        let count = self.dataset.len();
        let mut order: Vec<usize> = (0..count).collect();
        let mut random = self.state.seed ^ self.state.epoch.wrapping_mul(0xA24B_AED4_963E_E407);
        for index in (1..count).rev() {
            let swap = (split_mix(&mut random) % (index as u64 + 1)) as usize;
            order.swap(index, swap);
        }
        for window in order.chunks_mut(self.length_window) {
            window.sort_by_key(|&index| self.dataset.examples[index].input_ids.len());
        }
        self.order = order;
    }

    fn next_global(&mut self) -> SftBatch {
        if self.state.cursor >= self.order.len() {
            self.state.epoch += 1;
            self.state.cursor = 0;
            self.reorder();
        }
        let end = (self.state.cursor + self.micro_batch).min(self.order.len());
        let examples: Vec<&SftExample> = self.order[self.state.cursor..end]
            .iter()
            .map(|&index| &self.dataset.examples[index])
            .collect();
        let batch = SftBatch::collate(&examples, self.pad_token_id);
        self.state.cursor = end;
        batch
    }

    /// Next micro-batch for this rank.
    pub fn next_batch(&mut self) -> SftBatch {
        let mut batch = None;
        for shard in 0..self.world_size {
            let next = self.next_global();
            if shard == self.rank {
                batch = Some(next);
            }
        }
        batch.expect("rank is inside the world")
    }
}
