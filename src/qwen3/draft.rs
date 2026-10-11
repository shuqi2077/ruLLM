use super::{Layer, Projection, Qwen3Model, norm, ops, rotate};
use crate::gpu_inference::{gqa_scores, gqa_value_product};
use crate::huggingface::checkpoint::Checkpoint;
use crate::{
    CausalModel, GenerationCacheContinuation, GenerationError, HuggingFaceLoadError,
    HuggingFaceLoadReport, LlamaKvCache, LlamaKvCacheRecord,
};
use ruda_model::record::{PrecisionSettings, Record, RecorderError};
use ruda_tensor::api::{
    DType, Int, Tensor, TensorData,
    activation::{silu, softmax},
    backend::Backend,
};
use serde::Deserialize;
use std::{collections::BTreeSet, marker::PhantomData, path::Path};

#[derive(Deserialize)]
struct DraftInner {
    mask_token_id: i32,
    target_layer_ids: Vec<usize>,
}
#[derive(Deserialize)]
struct Rope {
    rope_theta: f64,
    rope_type: String,
}
#[derive(Deserialize)]
struct DraftConfig {
    hidden_size: usize,
    intermediate_size: usize,
    num_hidden_layers: usize,
    num_attention_heads: usize,
    num_key_value_heads: usize,
    num_target_layers: usize,
    head_dim: usize,
    vocab_size: usize,
    rms_norm_eps: f64,
    max_position_embeddings: usize,
    block_size: usize,
    rope_theta: Option<f64>,
    rope_parameters: Option<Rope>,
    dflash_config: Option<DraftInner>,
    mask_token_id: Option<i32>,
    target_layer_ids: Option<Vec<usize>>,
    num_anchors: Option<usize>,
    #[serde(default)]
    markov_rank: usize,
    markov_head_type: Option<String>,
    #[serde(default)]
    enable_confidence_head: bool,
    #[serde(default)]
    attention_bias: bool,
    #[serde(default)]
    attention_dropout: f64,
    hidden_act: String,
    #[serde(default)]
    use_sliding_window: bool,
    #[serde(default)]
    quantization_config: Option<serde_json::Value>,
}

struct Draft<B: Backend> {
    config: DraftConfig,
    taps: Vec<usize>,
    mask: i32,
    theta: f64,
    anchor_first: bool,
    layers: Vec<Layer<B>>,
    fc: Projection<B>,
    hidden_norm: Tensor<B, 1>,
    norm: Tensor<B, 1>,
    markov: Option<(Tensor<B, 2>, Tensor<B, 2>)>,
}
pub struct Qwen3Drafted<B: Backend> {
    pub target: Qwen3Model<B>,
    draft: Draft<B>,
    pub report: HuggingFaceLoadReport,
}
#[derive(Clone, Debug)]
struct Verified<B: Backend> {
    tokens: Vec<i32>,
    logits: Tensor<B, 3>,
    features: Tensor<B, 3>,
    next: usize,
}
#[derive(Clone, Debug)]
pub struct Qwen3DraftCache<B: Backend> {
    target: LlamaKvCache<B>,
    draft: LlamaKvCache<B>,
    position: usize,
    pending: Option<Tensor<B, 3>>,
    verified: Option<Verified<B>>,
    feature_width: usize,
}

fn append_features<B: Backend>(pending: &mut Option<Tensor<B, 3>>, features: Tensor<B, 3>) {
    *pending = Some(match pending.take() {
        Some(previous) => Tensor::cat(vec![previous, features], 1),
        None => features,
    });
}

impl<B: Backend> Draft<B> {
    fn rotary(
        &self,
        start: usize,
        count: usize,
        device: &B::Device,
        dtype: DType,
    ) -> (Tensor<B, 4>, Tensor<B, 4>) {
        let width = self.config.head_dim;
        let inverse = (0..width / 2)
            .map(|i| 1.0f32 / (self.theta as f32).powf((2 * i) as f32 / width as f32))
            .collect::<Vec<_>>();
        let frequencies = Tensor::<B, 2>::from_data(
            TensorData::new(
                (start..start + count).map(|i| i as f32).collect::<Vec<_>>(),
                [count, 1],
            ),
            (device, DType::F32),
        ) * Tensor::<B, 2>::from_data(
            TensorData::new(inverse, [1, width / 2]),
            (device, DType::F32),
        );
        let frequencies =
            Tensor::cat(vec![frequencies.clone(), frequencies], 1).reshape([1, 1, count, width]);
        (
            frequencies.clone().cos().cast(dtype),
            frequencies.sin().cast(dtype),
        )
    }

    fn propose(
        &self,
        target: &Qwen3Model<B>,
        cache: &mut Qwen3DraftCache<B>,
        anchor: i32,
    ) -> Result<Vec<i32>, GenerationError> {
        let c = &self.config;
        let context = cache
            .pending
            .as_ref()
            .ok_or_else(|| GenerationError("draft has no committed target features".into()))?;
        let context_len = context.dims()[1];
        let old = cache.draft.position();
        if context_len == 0
            || old.checked_add(context_len) != Some(cache.position)
            || cache
                .position
                .checked_add(c.block_size)
                .is_none_or(|n| n > c.max_position_embeddings)
        {
            return Err(GenerationError(
                "draft context or positional span differs".into(),
            ));
        }
        let device = target.embedding.device();
        let dtype = target.embedding.dtype();
        let context = norm(
            self.fc.forward(context.clone()),
            &self.hidden_norm,
            c.rms_norm_eps,
        );
        let mut tokens = vec![self.mask; c.block_size];
        tokens[0] = anchor;
        let ids = Tensor::<B, 1, Int>::from_data(TensorData::new(tokens, [c.block_size]), &device);
        let mut hidden =
            target
                .embedding
                .clone()
                .select(0, ids)
                .reshape([1, c.block_size, c.hidden_size]);
        let (query_cos, query_sin) = self.rotary(cache.position, c.block_size, &device, dtype);
        let (key_cos, key_sin) = self.rotary(old, context_len + c.block_size, &device, dtype);
        let full_length = cache.position + c.block_size;
        for (layer, state) in self.layers.iter().zip(&mut cache.draft.layers) {
            if state.sequence_length() != old {
                return Err(GenerationError("draft KV positions differ".into()));
            }
            let normalized = norm(hidden.clone(), &layer.input_norm, c.rms_norm_eps);
            let query = norm(
                layer.q.forward(normalized.clone()).reshape([
                    1,
                    c.block_size,
                    c.num_attention_heads,
                    c.head_dim,
                ]),
                &layer.q_norm,
                c.rms_norm_eps,
            )
            .swap_dims(1, 2);
            let tail = Tensor::cat(vec![context.clone(), normalized], 1);
            let key = norm(
                layer.k.forward(tail.clone()).reshape([
                    1,
                    context_len + c.block_size,
                    c.num_key_value_heads,
                    c.head_dim,
                ]),
                &layer.k_norm,
                c.rms_norm_eps,
            )
            .swap_dims(1, 2);
            let value = layer
                .v
                .forward(tail)
                .reshape([
                    1,
                    context_len + c.block_size,
                    c.num_key_value_heads,
                    c.head_dim,
                ])
                .swap_dims(1, 2);
            let query = rotate(query, query_cos.clone(), query_sin.clone());
            let key = rotate(key, key_cos.clone(), key_sin.clone());
            let (key, value, end) = state.append_growable(key, value);
            if end != full_length {
                return Err(GenerationError("draft KV append span differs".into()));
            }
            let ranges = [0..1, 0..c.num_key_value_heads, 0..end, 0..c.head_dim];
            let scores =
                gqa_scores(&ops(), query, key.slice(ranges.clone())) / (c.head_dim as f64).sqrt();
            let probabilities = softmax(scores.cast(DType::F32), 3).cast(dtype);
            let attended = gqa_value_product(&ops(), probabilities, value.slice(ranges))
                .swap_dims(1, 2)
                .reshape([1, c.block_size, c.num_attention_heads * c.head_dim]);
            hidden = hidden + layer.o.forward(attended);
            let normalized = norm(hidden.clone(), &layer.post_norm, c.rms_norm_eps);
            hidden = hidden
                + layer.down.forward(
                    silu(layer.gate.forward(normalized.clone())) * layer.up.forward(normalized),
                );
        }
        cache.draft.position = full_length;
        cache.draft.truncate(cache.position)?;
        cache.pending = None;
        let logits = target
            .head
            .forward(norm(hidden, &self.norm, c.rms_norm_eps));
        let mut candidates = Vec::with_capacity(c.block_size);
        let start = if self.anchor_first { 0 } else { 1 };
        let mut previous = anchor;
        for row in start..c.block_size {
            let mut logits = logits.clone().slice([0..1, row..row + 1, 0..c.vocab_size]);
            if let Some((w1, w2)) = &self.markov {
                let index = Tensor::<B, 1, Int>::from_data([previous], &device);
                let code = w1.clone().select(0, index);
                let bias = code
                    .cast(DType::F32)
                    .matmul(w2.clone().cast(DType::F32).transpose())
                    .reshape([1, 1, c.vocab_size]);
                logits = logits.cast(DType::F32) + bias;
            }
            previous = target.greedy_token(logits)?;
            candidates.push(previous);
        }
        Ok(candidates)
    }
}

impl<B: Backend> CausalModel<B> for Qwen3Drafted<B> {
    type Cache = Qwen3DraftCache<B>;
    fn new_cache(&self) -> Self::Cache {
        Qwen3DraftCache {
            target: self.target.new_cache(),
            draft: LlamaKvCache::new(
                self.draft.config.num_hidden_layers,
                self.draft.config.max_position_embeddings,
            ),
            position: 0,
            pending: None,
            verified: None,
            feature_width: self.draft.taps.len() * self.target.config.hidden_size,
        }
    }
    fn forward_cached_last(
        &self,
        tokens: Tensor<B, 2, Int>,
        cache: &mut Self::Cache,
    ) -> Tensor<B, 3> {
        self.try_forward_cached_last(tokens, cache)
            .expect("validated Qwen3 drafted forward")
    }
    fn try_forward_cached_last(
        &self,
        tokens: Tensor<B, 2, Int>,
        cache: &mut Self::Cache,
    ) -> Result<Tensor<B, 3>, GenerationError> {
        let [batch, count] = tokens.dims();
        let c = &self.target.config;
        if batch != 1
            || count == 0
            || tokens.device() != self.target.embedding.device()
            || cache.feature_width != self.draft.taps.len() * c.hidden_size
            || cache
                .position
                .checked_add(count)
                .is_none_or(|end| end > c.max_position_embeddings)
        {
            return Err(GenerationError("incompatible Qwen3 draft request".into()));
        }
        let input = tokens
            .clone()
            .try_into_data()
            .map_err(|e| GenerationError(e.to_string()))?
            .iter::<i32>()
            .collect::<Vec<_>>();
        if input
            .iter()
            .any(|&token| token < 0 || token as usize >= c.vocab_size)
        {
            return Err(GenerationError(
                "drafted request contains invalid input tokens".into(),
            ));
        }
        if count == 1 {
            if let Some(verified) = &mut cache.verified {
                if verified.tokens.get(verified.next) == input.first() {
                    let row = verified.next;
                    let logits =
                        verified
                            .logits
                            .clone()
                            .slice([0..1, row..row + 1, 0..c.vocab_size]);
                    append_features(
                        &mut cache.pending,
                        verified.features.clone().slice([
                            0..1,
                            row..row + 1,
                            0..cache.feature_width,
                        ]),
                    );
                    verified.next += 1;
                    cache.position += 1;
                    if verified.next == verified.tokens.len() {
                        cache.verified = None;
                    }
                    return Ok(logits);
                }
            }
        }
        cache.verified = None;
        cache.target.truncate(cache.position)?;
        let can_draft = count == 1
            && cache.position > 0
            && cache.pending.is_some()
            && cache
                .position
                .checked_add(self.draft.config.block_size + 1)
                .is_some_and(|end| {
                    end <= c.max_position_embeddings
                        && end <= self.draft.config.max_position_embeddings
                });
        if can_draft {
            let mut span = vec![input[0]];
            span.extend(self.draft.propose(&self.target, cache, input[0])?);
            let input = Tensor::<B, 2, Int>::from_data(
                TensorData::new(span.clone(), [1, span.len()]),
                &tokens.device(),
            );
            let (logits, features) = self.target.forward_with_features(
                input,
                &mut cache.target,
                false,
                &self.draft.taps,
            )?;
            let features = features
                .ok_or_else(|| GenerationError("target did not capture draft features".into()))?;
            append_features(
                &mut cache.pending,
                features.clone().slice([0..1, 0..1, 0..cache.feature_width]),
            );
            cache.position += 1;
            let first = logits.clone().slice([0..1, 0..1, 0..c.vocab_size]);
            if span.len() > 1 {
                cache.verified = Some(Verified {
                    tokens: span,
                    logits,
                    features,
                    next: 1,
                });
            }
            Ok(first)
        } else {
            let (logits, features) = self.target.forward_with_features(
                tokens,
                &mut cache.target,
                true,
                &self.draft.taps,
            )?;
            append_features(
                &mut cache.pending,
                features.ok_or_else(|| {
                    GenerationError("target did not capture draft features".into())
                })?,
            );
            cache.position += count;
            Ok(logits)
        }
    }
}

pub struct Qwen3DraftCacheRecord<B: Backend> {
    version: u32,
    model_id: String,
    target: LlamaKvCacheRecord<B>,
    draft: LlamaKvCacheRecord<B>,
    position: usize,
    feature_width: usize,
    pending: Option<TensorData>,
    backend: PhantomData<B>,
}
impl<B: Backend> Record<B> for Qwen3DraftCacheRecord<B> {
    type Item<S: PrecisionSettings> = (
        u32,
        String,
        <LlamaKvCacheRecord<B> as Record<B>>::Item<S>,
        <LlamaKvCacheRecord<B> as Record<B>>::Item<S>,
        usize,
        usize,
        Option<TensorData>,
    );
    fn into_item<S: PrecisionSettings>(self) -> Self::Item<S> {
        (
            self.version,
            self.model_id,
            self.target.into_item::<S>(),
            self.draft.into_item::<S>(),
            self.position,
            self.feature_width,
            self.pending,
        )
    }
    fn from_item<S: PrecisionSettings>(item: Self::Item<S>, device: &B::Device) -> Self {
        Self {
            version: item.0,
            model_id: item.1,
            target: LlamaKvCacheRecord::from_item::<S>(item.2, device),
            draft: LlamaKvCacheRecord::from_item::<S>(item.3, device),
            position: item.4,
            feature_width: item.5,
            pending: item.6,
            backend: PhantomData,
        }
    }
}
fn invalid(error: impl std::fmt::Display) -> RecorderError {
    RecorderError::Unknown(format!("invalid draft continuation: {error}"))
}
impl<B: Backend> GenerationCacheContinuation<B> for Qwen3DraftCache<B> {
    type Record = Qwen3DraftCacheRecord<B>;
    fn generation_position(&self) -> usize {
        self.position
    }
    fn capture_generation(
        &self,
        model_id: &str,
        device: &B::Device,
    ) -> Result<Self::Record, RecorderError> {
        if model_id.is_empty() || self.draft.position() > self.position {
            return Err(invalid("model or native draft position differs"));
        }
        B::sync(device).map_err(invalid)?;
        let mut target = self.target.clone();
        target.truncate(self.position).map_err(invalid)?;
        let pending = self
            .pending
            .as_ref()
            .map(|features| features.clone().try_into_data().map_err(invalid))
            .transpose()?;
        check_pending(
            pending.as_ref(),
            self.position - self.draft.position(),
            self.feature_width,
        )?;
        Ok(Qwen3DraftCacheRecord {
            version: 1,
            model_id: model_id.into(),
            target: target.capture_generation(model_id, device)?,
            draft: self
                .draft
                .capture_generation(&format!("{model_id}:draft"), device)?,
            position: self.position,
            feature_width: self.feature_width,
            pending,
            backend: PhantomData,
        })
    }
    fn restore_generation(
        record: Self::Record,
        model_id: &str,
        template: &Self,
        position: usize,
        device: &B::Device,
    ) -> Result<Self, RecorderError> {
        if record.version != 1
            || record.model_id != model_id
            || model_id.is_empty()
            || record.position != position
            || record.feature_width != template.feature_width
        {
            return Err(invalid("identity or draft feature geometry differs"));
        }
        let draft_item = record
            .draft
            .into_item::<ruda_model::record::FullPrecisionSettings>();
        let draft_position = draft_item.2;
        if draft_position > position {
            return Err(invalid("draft context is ahead of committed target"));
        }
        check_pending(
            record.pending.as_ref(),
            position - draft_position,
            record.feature_width,
        )?;
        let target = LlamaKvCache::restore_generation(
            record.target,
            model_id,
            &template.target,
            position,
            device,
        )?;
        let draft = LlamaKvCache::restore_generation(
            LlamaKvCacheRecord::from_item::<ruda_model::record::FullPrecisionSettings>(
                draft_item, device,
            ),
            &format!("{model_id}:draft"),
            &template.draft,
            draft_position,
            device,
        )?;
        let pending = record.pending.map(|data| {
            let dtype = data.dtype;
            Tensor::from_data(data, (device, dtype))
        });
        Ok(Self {
            target,
            draft,
            position,
            pending,
            verified: None,
            feature_width: record.feature_width,
        })
    }
}
fn check_pending(
    data: Option<&TensorData>,
    rows: usize,
    width: usize,
) -> Result<(), RecorderError> {
    match data {
        None if rows == 0 => Ok(()),
        Some(data)
            if rows > 0
                && width > 0
                && data.shape == [1, rows, width]
                && matches!(data.dtype, DType::F16 | DType::BF16 | DType::F32)
                && rows
                    .checked_mul(width)
                    .and_then(|n| n.checked_mul(data.dtype.size()))
                    == Some(data.as_bytes().len()) =>
        {
            Ok(())
        }
        _ => Err(invalid(
            "pending target features do not match committed input count",
        )),
    }
}

pub fn load_huggingface_qwen3_draft<B: Backend>(
    target: Qwen3Model<B>,
    directory: impl AsRef<Path>,
) -> Result<Qwen3Drafted<B>, HuggingFaceLoadError> {
    let directory = directory
        .as_ref()
        .canonicalize()
        .map_err(|e| HuggingFaceLoadError(e.to_string()))?;
    let mut config: DraftConfig = serde_json::from_slice(
        &std::fs::read(directory.join("config.json"))
            .map_err(|e| HuggingFaceLoadError(e.to_string()))?,
    )
    .map_err(|e| HuggingFaceLoadError(e.to_string()))?;
    let inner = config.dflash_config.take();
    let (mask, taps) = match inner {
        Some(inner) => (inner.mask_token_id, inner.target_layer_ids),
        None => (
            config
                .mask_token_id
                .ok_or_else(|| HuggingFaceLoadError("draft mask ID missing".into()))?,
            config
                .target_layer_ids
                .take()
                .ok_or_else(|| HuggingFaceLoadError("target feature layers missing".into()))?,
        ),
    };
    let theta = match (&config.rope_parameters, config.rope_theta) {
        (Some(rope), _) if rope.rope_type == "default" => rope.rope_theta,
        (None, Some(theta)) => theta,
        _ => {
            return Err(HuggingFaceLoadError(
                "unsupported draft RoPE definition".into(),
            ));
        }
    };
    let anchor_first = config.num_anchors.is_some() || config.markov_rank > 0;
    if target.reduction.is_some()
        || [
            config.hidden_size,
            config.intermediate_size,
            config.num_hidden_layers,
            config.num_attention_heads,
            config.num_key_value_heads,
            config.head_dim,
            config.block_size,
        ]
        .contains(&0)
        || !config
            .num_attention_heads
            .is_multiple_of(config.num_key_value_heads)
        || !config.head_dim.is_multiple_of(2)
        || config.hidden_size != target.config.hidden_size
        || config.vocab_size != target.config.vocab_size
        || config.num_target_layers != target.config.num_hidden_layers
        || config.block_size < 2
        || config.block_size > config.max_position_embeddings
        || config.max_position_embeddings < target.config.max_position_embeddings
        || taps.is_empty()
        || taps
            .iter()
            .any(|&layer| layer >= target.config.num_hidden_layers)
        || taps.windows(2).any(|pair| pair[0] >= pair[1])
        || mask < 0
        || mask as usize >= config.vocab_size
        || config.hidden_act != "silu"
        || config.attention_bias
        || config.attention_dropout != 0.0
        || config.use_sliding_window
        || !theta.is_finite()
        || theta <= 0.0
        || !config.rms_norm_eps.is_finite()
        || config.rms_norm_eps <= 0.0
        || config.markov_rank > 0 && config.markov_head_type.as_deref() != Some("vanilla")
        || config
            .quantization_config
            .as_ref()
            .is_some_and(|v| !v.is_null())
    {
        return Err(HuggingFaceLoadError("requires a compatible floating DFlash/vanilla DSpark drafter and a single-rank Qwen3 target".into()));
    }
    let device = target.embedding.device();
    let dtype = target.embedding.dtype();
    let mut checkpoint = Checkpoint::open(&directory)?;
    fn projection<B: Backend>(
        checkpoint: &mut Checkpoint,
        prefix: &str,
        input: usize,
        output: usize,
        device: &B::Device,
    ) -> Result<Projection<B>, HuggingFaceLoadError> {
        Ok(Projection {
            weight: checkpoint.tensor(&format!("{prefix}.weight"), [output, input], device)?,
            bias: None,
        })
    }
    let mut layers = Vec::with_capacity(config.num_hidden_layers);
    for layer in 0..config.num_hidden_layers {
        let p = format!("layers.{layer}");
        let h = config.hidden_size;
        layers.push(Layer {
            q: projection(
                &mut checkpoint,
                &format!("{p}.self_attn.q_proj"),
                h,
                config.num_attention_heads * config.head_dim,
                &device,
            )?,
            k: projection(
                &mut checkpoint,
                &format!("{p}.self_attn.k_proj"),
                h,
                config.num_key_value_heads * config.head_dim,
                &device,
            )?,
            v: projection(
                &mut checkpoint,
                &format!("{p}.self_attn.v_proj"),
                h,
                config.num_key_value_heads * config.head_dim,
                &device,
            )?,
            o: projection(
                &mut checkpoint,
                &format!("{p}.self_attn.o_proj"),
                config.num_attention_heads * config.head_dim,
                h,
                &device,
            )?,
            gate: projection(
                &mut checkpoint,
                &format!("{p}.mlp.gate_proj"),
                h,
                config.intermediate_size,
                &device,
            )?,
            up: projection(
                &mut checkpoint,
                &format!("{p}.mlp.up_proj"),
                h,
                config.intermediate_size,
                &device,
            )?,
            down: projection(
                &mut checkpoint,
                &format!("{p}.mlp.down_proj"),
                config.intermediate_size,
                h,
                &device,
            )?,
            input_norm: checkpoint.tensor(&format!("{p}.input_layernorm.weight"), [h], &device)?,
            post_norm: checkpoint.tensor(
                &format!("{p}.post_attention_layernorm.weight"),
                [h],
                &device,
            )?,
            q_norm: checkpoint.tensor(
                &format!("{p}.self_attn.q_norm.weight"),
                [config.head_dim],
                &device,
            )?,
            k_norm: checkpoint.tensor(
                &format!("{p}.self_attn.k_norm.weight"),
                [config.head_dim],
                &device,
            )?,
        });
    }
    let fc = projection(
        &mut checkpoint,
        "fc",
        config.hidden_size * taps.len(),
        config.hidden_size,
        &device,
    )?;
    let hidden_norm = checkpoint.tensor("hidden_norm.weight", [config.hidden_size], &device)?;
    let norm = checkpoint.tensor("norm.weight", [config.hidden_size], &device)?;
    let markov = if config.markov_rank > 0 {
        Some((
            checkpoint.tensor(
                "markov_head.markov_w1.weight",
                [config.vocab_size, config.markov_rank],
                &device,
            )?,
            checkpoint.tensor(
                "markov_head.markov_w2.weight",
                [config.vocab_size, config.markov_rank],
                &device,
            )?,
        ))
    } else {
        None
    };
    for (name, original) in [
        ("embed_tokens.weight", &target.embedding),
        ("lm_head.weight", &target.head.weight),
    ] {
        if checkpoint.tensors.contains_key(name) {
            let supplied = checkpoint
                .tensor::<B, 2>(name, [config.vocab_size, config.hidden_size], &device)?
                .try_into_data()
                .map_err(|e| HuggingFaceLoadError(e.to_string()))?;
            let actual = original
                .clone()
                .try_into_data()
                .map_err(|e| HuggingFaceLoadError(e.to_string()))?;
            if supplied.dtype != actual.dtype || supplied.as_bytes() != actual.as_bytes() {
                return Err(HuggingFaceLoadError(format!(
                    "{name} is not identical to the target checkpoint"
                )));
            }
        }
    }
    let mut excluded = BTreeSet::new();
    for name in ["confidence_head.proj.weight", "confidence_head.proj.bias"] {
        if let Some(snapshot) = checkpoint.tensors.get(name) {
            if !config.enable_confidence_head
                || snapshot.dtype != dtype
                || snapshot.shape.len() != if name.ends_with("bias") { 1 } else { 2 }
                || snapshot.shape[0] != 1
            {
                return Err(HuggingFaceLoadError(
                    "incompatible optional confidence-head storage".into(),
                ));
            }
            excluded.insert(name.to_string());
        }
    }
    let unused = checkpoint
        .tensors
        .keys()
        .filter(|name| !checkpoint.consumed.contains(*name) && !excluded.contains(*name))
        .cloned()
        .collect::<Vec<_>>();
    if !unused.is_empty()
        || checkpoint
            .tensors
            .iter()
            .any(|(name, snapshot)| checkpoint.consumed.contains(name) && snapshot.dtype != dtype)
    {
        return Err(HuggingFaceLoadError(format!(
            "unrepresented or mixed-dtype draft tensors: {}",
            unused.join(", ")
        )));
    }
    B::sync(&device).map_err(|e| HuggingFaceLoadError(e.to_string()))?;
    let report = HuggingFaceLoadReport {
        config_path: directory.join("config.json"),
        weight_files: checkpoint.files,
        applied_tensors: checkpoint.consumed.len(),
        tied_word_embeddings: false,
    };
    Ok(Qwen3Drafted {
        target,
        draft: Draft {
            config,
            taps,
            mask,
            theta,
            anchor_first,
            layers,
            fc,
            hidden_norm,
            norm,
            markov,
        },
        report,
    })
}
