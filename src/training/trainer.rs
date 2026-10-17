//! AdamW trainer for packed Llama/Qwen2 models (full fine-tuning or LoRA).
//!
//! One optimizer step consumes `gradient_accumulation` micro-batches; each
//! micro-batch loss is normalized by the step's total valid-token count so
//! accumulated gradients equal the gradient of the step's token-mean loss.
//! Gradients are clipped by their global L2 norm; non-finite steps are
//! skipped. Checkpoints are written atomically and only the newest
//! `keep_checkpoints` are retained.

use super::data::{SftBatch, SftLoader, SftLoaderState};
use super::lora::LlamaLora;
use super::ops::{TrainingBackend, cross_entropy_chunk_rows};
use crate::PackedLlamaForCausalLm;
use ruda_model::module::{AutodiffModule, Module, ModuleMapper, ModuleVisitor, Param};
use ruda_model::record::{BinFileRecorder, FullPrecisionSettings, Recorder};
use ruda_optim::adaptor::OptimizerAdaptor;
use ruda_optim::{AdamW, AdamWConfig, GradientsAccumulator, GradientsParams, Optimizer};
use ruda_tensor::api::backend::AutodiffBackend;
use ruda_tensor::api::{DType, Int, Tensor, TensorData};
use std::path::{Path, PathBuf};
use std::time::{Duration, Instant};

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct TrainError(pub String);

impl std::fmt::Display for TrainError {
    fn fmt(&self, formatter: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        formatter.write_str(&self.0)
    }
}

impl std::error::Error for TrainError {}

fn error(message: impl std::fmt::Display) -> TrainError {
    TrainError(message.to_string())
}

/// Linear warmup, then cosine decay to `min_learning_rate` at `total_steps`.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct LearningRateSchedule {
    pub peak: f64,
    pub min: f64,
    pub warmup_steps: usize,
    pub total_steps: usize,
}

impl LearningRateSchedule {
    /// Learning rate for optimizer step `step` (zero-based).
    pub fn at(&self, step: usize) -> f64 {
        if self.warmup_steps > 0 && step < self.warmup_steps {
            return self.peak * (step + 1) as f64 / self.warmup_steps as f64;
        }
        let decay = self.total_steps.saturating_sub(self.warmup_steps).max(1);
        let progress = (step.saturating_sub(self.warmup_steps) as f64 / decay as f64).min(1.0);
        self.min + 0.5 * (self.peak - self.min) * (1.0 + (std::f64::consts::PI * progress).cos())
    }
}

#[derive(Debug, Clone, PartialEq)]
pub struct TrainConfig {
    pub schedule: LearningRateSchedule,
    pub weight_decay: f32,
    pub beta_1: f32,
    pub beta_2: f32,
    pub epsilon: f32,
    /// Global L2 norm limit; `None` disables clipping.
    pub max_grad_norm: Option<f32>,
    /// Micro-batches per optimizer step.
    pub gradient_accumulation: usize,
    /// FP32 bytes budget for one cross-entropy logits chunk.
    pub loss_chunk_bytes: usize,
    /// Optimizer steps between checkpoints (`0` = only at the end).
    pub checkpoint_every: usize,
    pub checkpoint_dir: PathBuf,
    /// Newest checkpoints kept on disk (at least one).
    pub keep_checkpoints: usize,
    /// Optimizer steps between progress callbacks.
    pub log_every: usize,
}

impl TrainConfig {
    pub fn new(total_steps: usize, learning_rate: f64, checkpoint_dir: impl Into<PathBuf>) -> Self {
        Self {
            schedule: LearningRateSchedule {
                peak: learning_rate,
                min: learning_rate * 0.1,
                warmup_steps: (total_steps / 20).max(1),
                total_steps,
            },
            weight_decay: 0.0,
            beta_1: 0.9,
            beta_2: 0.95,
            epsilon: 1e-8,
            max_grad_norm: Some(1.0),
            gradient_accumulation: 1,
            loss_chunk_bytes: 128 << 20,
            checkpoint_every: 0,
            checkpoint_dir: checkpoint_dir.into(),
            keep_checkpoints: 1,
            log_every: 1,
        }
    }
}

/// Non-tensor trainer position, persisted as `state.json`.
#[derive(Debug, Clone, Copy, PartialEq, serde::Serialize, serde::Deserialize)]
pub struct TrainerState {
    pub step: usize,
    pub tokens: u64,
    pub skipped_steps: u64,
    pub elapsed_seconds: f64,
    pub loader: Option<SftLoaderState>,
}

/// Result of one optimizer step.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct StepStats {
    pub step: usize,
    pub loss: f32,
    pub grad_norm: f32,
    pub learning_rate: f64,
    pub tokens: usize,
    pub applied: bool,
}

/// Progress snapshot passed to the callback of [`SftTrainer::train`].
#[derive(Debug, Clone, PartialEq)]
pub struct TrainProgress {
    pub stats: StepStats,
    pub total_steps: usize,
    pub tokens_seen: u64,
    pub elapsed: Duration,
    /// Target tokens per second over the most recent `log_every` steps.
    pub recent_tokens_per_second: f64,
    /// Remaining time at the recent step rate.
    pub eta: Option<Duration>,
    pub epoch: u64,
    pub last_checkpoint: Option<(usize, Duration)>,
    pub skipped_steps: u64,
}

/// Sum of squared gradient entries over a module's parameters (FP32).
struct SquaredNorm<'a, B: AutodiffBackend> {
    grads: &'a GradientsParams,
    total: Option<Tensor<B::InnerBackend, 1>>,
}

impl<B: AutodiffBackend> ModuleVisitor<B> for SquaredNorm<'_, B> {
    fn visit_float<const D: usize>(&mut self, param: &Param<Tensor<B, D>>) {
        if let Some(gradient) = self.grads.get::<B::InnerBackend, D>(param.id) {
            let squared = gradient.cast(DType::F32).square().sum();
            self.total = Some(match self.total.take() {
                Some(total) => total + squared,
                None => squared,
            });
        }
    }
}

/// Multiply every parameter gradient by `factor`.
struct ScaleGradients<'a, B: AutodiffBackend> {
    grads: &'a mut GradientsParams,
    factor: f32,
    _backend: core::marker::PhantomData<B>,
}

impl<B: AutodiffBackend> ModuleVisitor<B> for ScaleGradients<'_, B> {
    fn visit_float<const D: usize>(&mut self, param: &Param<Tensor<B, D>>) {
        if let Some(gradient) = self.grads.remove::<B::InnerBackend, D>(param.id) {
            self.grads
                .register::<B::InnerBackend, D>(param.id, gradient.mul_scalar(self.factor));
        }
    }
}

/// Global L2 norm of `grads` over `module`; clips in place when above `max`.
pub(super) fn clip_global_norm<B: AutodiffBackend, M: AutodiffModule<B>>(
    module: &M,
    grads: &mut GradientsParams,
    max: Option<f32>,
) -> Result<f32, TrainError> {
    let mut visitor = SquaredNorm::<B> { grads, total: None };
    module.visit(&mut visitor);
    let Some(total) = visitor.total else {
        return Ok(0.0);
    };
    let squared = total
        .into_data()
        .convert::<f32>()
        .to_vec::<f32>()
        .map_err(|failure| error(format!("cannot read gradient norm: {failure:?}")))?[0];
    let norm = squared.sqrt();
    if let Some(max) = max
        && norm.is_finite()
        && norm > max
    {
        let mut scale = ScaleGradients::<B> {
            grads,
            factor: max / (norm + 1e-6),
            _backend: core::marker::PhantomData,
        };
        module.visit(&mut scale);
    }
    Ok(norm)
}

type FullOptimizer<B> = OptimizerAdaptor<AdamW, PackedLlamaForCausalLm<B>, B>;
type LoraOptimizer<B> = OptimizerAdaptor<AdamW, LlamaLora<B>, B>;

/// Trainable float parameters are checkpointed as Safetensors keyed by
/// `<visit index>:<ParamId>`, bit-exact in their own dtypes. The module
/// recorders convert every float to the backend's default element on load,
/// which turns FP32 LoRA adapters (and FP32 master weights) on a BF16 backend
/// into BF16 and breaks resume. Visiting the module directly also avoids the
/// non-leaf `require_grad` failure of loading records into autodiff modules.
pub(super) fn save_trainable<B: ruda_tensor::api::backend::Backend, M: Module<B>>(module: &M, path: &Path) -> Result<(), String> {
    let mut collector = ParamCollector(Vec::new());
    module.visit(&mut collector);
    super::export::write_safetensors(path, &collector.0, &[("format", "rullm-trainable")])
}

/// Replace every float parameter of `module` (same architecture as when
/// saved) with the stored value, dtype and `ParamId`, keeping `require_grad`.
pub(super) fn load_trainable<B: ruda_tensor::api::backend::Backend, M: Module<B>>(
    module: &M,
    path: &Path,
    device: &B::Device,
) -> Result<M, String> {
    let mut stored = read_safetensors(path)?;
    stored.sort_by(|a, b| a.0.cmp(&b.0));
    let mut expected = ParamCounter(0);
    module.visit(&mut expected);
    if expected.0 != stored.len() {
        return Err(format!(
            "checkpoint {} holds {} tensors, the module has {} float parameters",
            path.display(),
            stored.len(),
            expected.0
        ));
    }
    let mut loader = ParamLoader { stored: stored.into_iter(), device: device.clone(), failure: None };
    let loaded = module.clone().map(&mut loader);
    match loader.failure {
        Some(failure) => Err(failure),
        None => Ok(loaded),
    }
}

/// Minimal Safetensors reader for files written by `write_safetensors`.
fn read_safetensors(path: &Path) -> Result<Vec<(String, TensorData)>, String> {
    let bytes = std::fs::read(path).map_err(|failure| format!("cannot read {}: {failure}", path.display()))?;
    let invalid = || format!("{} is not a valid Safetensors file", path.display());
    let header_length = bytes
        .get(..8)
        .map(|prefix| u64::from_le_bytes(prefix.try_into().expect("eight bytes")) as usize)
        .ok_or_else(invalid)?;
    let body = 8usize.checked_add(header_length).filter(|&end| end <= bytes.len()).ok_or_else(invalid)?;
    let header: serde_json::Map<String, serde_json::Value> =
        serde_json::from_slice(&bytes[8..body]).map_err(|_| invalid())?;
    let mut tensors = Vec::new();
    for (name, entry) in header {
        if name == "__metadata__" {
            continue;
        }
        let dtype = match entry["dtype"].as_str() {
            Some("F32") => DType::F32,
            Some("F16") => DType::F16,
            Some("BF16") => DType::BF16,
            Some("F64") => DType::F64,
            other => return Err(format!("{name}: unsupported checkpoint dtype {other:?}")),
        };
        let shape: Vec<usize> = serde_json::from_value(entry["shape"].clone()).map_err(|_| invalid())?;
        let [start, end]: [usize; 2] =
            serde_json::from_value(entry["data_offsets"].clone()).map_err(|_| invalid())?;
        let data = bytes.get(body + start..body + end).ok_or_else(invalid)?.to_vec();
        tensors.push((name, TensorData::from_bytes_vec(data, shape, dtype)));
    }
    Ok(tensors)
}

struct ParamCollector(Vec<(String, TensorData)>);

impl<B: ruda_tensor::api::backend::Backend> ModuleVisitor<B> for ParamCollector {
    fn visit_float<const D: usize>(&mut self, param: &Param<Tensor<B, D>>) {
        let name = format!("{:06}:{}", self.0.len(), param.id.serialize());
        self.0.push((name, param.val().into_data()));
    }
}

struct ParamCounter(usize);

impl<B: ruda_tensor::api::backend::Backend> ModuleVisitor<B> for ParamCounter {
    fn visit_float<const D: usize>(&mut self, _param: &Param<Tensor<B, D>>) {
        self.0 += 1;
    }
}

struct ParamLoader<I: Iterator<Item = (String, TensorData)>, Device> {
    stored: I,
    device: Device,
    failure: Option<String>,
}

impl<B, I> ModuleMapper<B> for ParamLoader<I, B::Device>
where
    B: ruda_tensor::api::backend::Backend,
    I: Iterator<Item = (String, TensorData)>,
{
    fn map_float<const D: usize>(&mut self, param: Param<Tensor<B, D>>) -> Param<Tensor<B, D>> {
        let Some((name, data)) = self.stored.next() else {
            self.failure.get_or_insert_with(|| "checkpoint has fewer tensors than the module".into());
            return param;
        };
        let dims = param.dims();
        if !data.shape.iter().copied().eq(dims.iter().copied()) {
            self.failure.get_or_insert_with(|| format!("{name}: stored shape {:?} != parameter shape {dims:?}", data.shape));
            return param;
        }
        let Some(id) = name
            .split_once(':')
            .and_then(|(_, id)| ruda_model::module::ParamId::try_deserialize(id))
        else {
            self.failure.get_or_insert_with(|| format!("{name}: invalid parameter id"));
            return param;
        };
        let require_grad = param.is_require_grad();
        let dtype = data.dtype;
        let tensor = Tensor::<B, D>::from_data(data, (&self.device, dtype));
        Param::initialized(id, tensor).set_require_grad(require_grad)
    }
}

enum Trainable<B: AutodiffBackend + TrainingBackend> {
    Full(FullOptimizer<B>),
    Lora {
        adapters: LlamaLora<B>,
        optimizer: LoraOptimizer<B>,
    },
}

/// Supervised fine-tuning / continued pre-training driver.
pub struct SftTrainer<B: AutodiffBackend + TrainingBackend> {
    model: PackedLlamaForCausalLm<B>,
    trainable: Trainable<B>,
    config: TrainConfig,
    state: TrainerState,
    device: B::Device,
    vocabulary: usize,
    last_checkpoint: Option<(usize, Instant)>,
    #[cfg(feature = "distributed")]
    data_parallel: Option<ruccl::PeerId>,
}

impl<B: AutodiffBackend + TrainingBackend> SftTrainer<B> {
    fn optimizer_config(config: &TrainConfig) -> AdamWConfig {
        AdamWConfig::new()
            .with_beta_1(config.beta_1)
            .with_beta_2(config.beta_2)
            .with_epsilon(config.epsilon)
            .with_weight_decay(config.weight_decay)
    }

    /// Full fine-tuning: every model parameter is trained.
    pub fn full(model: PackedLlamaForCausalLm<B>, config: TrainConfig, device: B::Device) -> Self {
        let optimizer = Self::optimizer_config(&config).init::<B, PackedLlamaForCausalLm<B>>();
        Self::build(model, Trainable::Full(optimizer), config, device)
    }

    /// LoRA: the base model is frozen and only `adapters` are trained.
    pub fn lora(
        model: PackedLlamaForCausalLm<B>,
        adapters: LlamaLora<B>,
        config: TrainConfig,
        device: B::Device,
    ) -> Self {
        let optimizer = Self::optimizer_config(&config).init::<B, LlamaLora<B>>();
        Self::build(model.no_grad(), Trainable::Lora { adapters, optimizer }, config, device)
    }

    fn build(
        model: PackedLlamaForCausalLm<B>,
        trainable: Trainable<B>,
        config: TrainConfig,
        device: B::Device,
    ) -> Self {
        assert!(config.gradient_accumulation > 0, "gradient accumulation must be positive");
        let vocabulary = model.geometry().vocab_size;
        Self {
            model,
            trainable,
            config,
            state: TrainerState {
                step: 0,
                tokens: 0,
                skipped_steps: 0,
                elapsed_seconds: 0.0,
                loader: None,
            },
            device,
            vocabulary,
            last_checkpoint: None,
            #[cfg(feature = "distributed")]
            data_parallel: None,
        }
    }

    /// Average gradients across ruCCL peers (already registered by the
    /// caller) before every optimizer step. Shard the loader with
    /// [`SftLoader::with_shard`] accordingly.
    #[cfg(feature = "distributed")]
    pub fn with_data_parallel(mut self, peer: ruccl::PeerId) -> Self {
        self.data_parallel = Some(peer);
        self
    }

    pub fn state(&self) -> TrainerState {
        self.state
    }

    pub fn device(&self) -> &B::Device { &self.device }

    pub(super) fn objective_contract(&self) -> serde_json::Value {
        let c = &self.config;
        serde_json::json!({"peak": c.schedule.peak, "min": c.schedule.min,
            "warmup": c.schedule.warmup_steps, "steps": c.schedule.total_steps,
            "weight_decay": c.weight_decay, "beta1": c.beta_1, "beta2": c.beta_2,
            "epsilon": c.epsilon, "max_grad_norm": c.max_grad_norm,
            "accumulation": c.gradient_accumulation, "loss_chunk_bytes": c.loss_chunk_bytes})
    }

    pub(super) fn loss_chunk_rows(&self) -> usize {
        cross_entropy_chunk_rows(self.vocabulary, self.config.loss_chunk_bytes)
    }

    pub(super) fn objective_step(&mut self, count: usize, tokens: usize, update_allowed: bool,
        mut objective: impl FnMut(&PackedLlamaForCausalLm<B>, Option<&LlamaLora<B>>, usize) -> Result<Tensor<B, 1>, TrainError>,
    ) -> Result<StepStats, TrainError> {
        if count == 0 { return Err(error("an objective update needs at least one sample")); }
        let mut sum = 0.0;
        let mut accumulator = GradientsAccumulator::<PackedLlamaForCausalLm<B>>::new();
        let mut lora_accumulator = GradientsAccumulator::<LlamaLora<B>>::new();
        for index in 0..count {
            let loss = objective(&self.model, self.adapters(), index)?;
            sum += read_scalar(loss.clone())?;
            let backward = loss.backward();
            match &self.trainable {
                Trainable::Full(_) => accumulator.accumulate::<B>(&self.model,
                    GradientsParams::from_grads(backward, &self.model)),
                Trainable::Lora { adapters, .. } => lora_accumulator.accumulate::<B>(adapters,
                    GradientsParams::from_grads(backward, adapters)),
            }
        }
        let grads = match &self.trainable {
            Trainable::Full(_) => accumulator.grads(),
            Trainable::Lora { .. } => lora_accumulator.grads(),
        };
        let mut grads = self.synchronize(grads)?;
        let norm = match &self.trainable {
            Trainable::Full(_) => clip_global_norm::<B, _>(&self.model, &mut grads, self.config.max_grad_norm)?,
            Trainable::Lora { adapters, .. } => clip_global_norm::<B, _>(adapters, &mut grads, self.config.max_grad_norm)?,
        };
        let learning_rate = self.config.schedule.at(self.state.step);
        sum = self.objective_mean(sum)?;
        let allowed = self.objective_mean(if update_allowed { 1.0 } else { 0.0 })?;
        let applied = norm.is_finite() && sum.is_finite() && allowed == 1.0;
        if applied {
            match &mut self.trainable {
                Trainable::Full(optimizer) => self.model = optimizer.step(learning_rate, self.model.clone(), grads),
                Trainable::Lora { adapters, optimizer } => *adapters = optimizer.step(learning_rate, adapters.clone(), grads),
            }
        } else { self.state.skipped_steps += 1; }
        self.state.step += 1;
        self.state.tokens += tokens as u64;
        Ok(StepStats { step: self.state.step, loss: sum, grad_norm: norm, learning_rate, tokens, applied })
    }

    pub fn model(&self) -> &PackedLlamaForCausalLm<B> {
        &self.model
    }

    pub fn adapters(&self) -> Option<&LlamaLora<B>> {
        match &self.trainable {
            Trainable::Lora { adapters, .. } => Some(adapters),
            Trainable::Full(_) => None,
        }
    }

    /// Mean local normalization weight; combined with mean-reduced gradients
    /// this yields a global weighted mean, including unequal replica batches.
    pub(super) fn objective_normalizer(&self, local: usize) -> Result<f32, TrainError> {
        let count = self.objective_mean(local as f32)?;
        if count <= 0.0 { return Err(error("objective has no valid targets")); }
        Ok(count)
    }

    pub(super) fn objective_mean(&self, value: f32) -> Result<f32, TrainError> {
        #[cfg(feature = "distributed")]
        if let Some(peer) = self.data_parallel {
            let tensor = Tensor::<B::InnerBackend, 1>::from_data([value], &self.device);
            let primitive = ruccl::all_reduce::<B::InnerBackend>(peer,
                tensor.into_primitive().tensor(), ruccl::ReduceOperation::Mean)
                .map_err(|failure| error(format!("objective reduction failed: {failure:?}")))?;
            return read_scalar(Tensor::<B::InnerBackend, 1>::from_primitive(
                ruda_tensor::api::TensorPrimitive::Float(primitive)));
        }
        Ok(value)
    }

    /// Final model and (for LoRA) adapters.
    pub fn into_parts(self) -> (PackedLlamaForCausalLm<B>, Option<LlamaLora<B>>) {
        match self.trainable {
            Trainable::Lora { adapters, .. } => (self.model, Some(adapters)),
            Trainable::Full(_) => (self.model, None),
        }
    }
}

impl<B: AutodiffBackend + TrainingBackend> SftTrainer<B> {
    fn micro_loss(&self, batch: &SftBatch, normalizer: usize) -> Tensor<B, 1> {
        let tokens = Tensor::<B, 2, Int>::from_data(
            TensorData::new(batch.tokens.clone(), [batch.batch, batch.length]),
            &self.device,
        );
        let targets = Tensor::<B, 1, Int>::from_data(
            TensorData::new(batch.targets.clone(), [batch.batch * batch.length]),
            &self.device,
        );
        let chunk = cross_entropy_chunk_rows(self.vocabulary, self.config.loss_chunk_bytes);
        let adapters = match &self.trainable {
            Trainable::Lora { adapters, .. } => Some(adapters),
            Trainable::Full(_) => None,
        };
        self.model.train_loss(tokens, targets, normalizer, adapters, chunk)
    }

    fn validate(&self, batch: &SftBatch) -> Result<(), TrainError> {
        if let Some(&token) = batch
            .tokens
            .iter()
            .find(|&&token| token < 0 || token as usize >= self.vocabulary)
        {
            return Err(error(format!("token {token} is outside vocabulary [0, {})", self.vocabulary)));
        }
        if let Some(&target) = batch
            .targets
            .iter()
            .find(|&&target| target >= self.vocabulary as i32)
        {
            return Err(error(format!("target {target} is outside the vocabulary")));
        }
        Ok(())
    }

    #[cfg(feature = "distributed")]
    pub(super) fn synchronize(&self, grads: GradientsParams) -> Result<GradientsParams, TrainError> {
        match self.data_parallel {
            Some(peer) => grads
                .all_reduce::<B::InnerBackend>(peer, ruccl::ReduceOperation::Mean)
                .map_err(|failure| error(format!("gradient all-reduce failed: {failure:?}"))),
            None => Ok(grads),
        }
    }

    #[cfg(not(feature = "distributed"))]
    pub(super) fn synchronize(&self, grads: GradientsParams) -> Result<GradientsParams, TrainError> {
        Ok(grads)
    }
}

impl<B: AutodiffBackend + TrainingBackend> SftTrainer<B> {
    /// One optimizer step over `micro_batches` (gradient accumulation).
    pub fn step(&mut self, micro_batches: &[SftBatch]) -> Result<StepStats, TrainError> {
        if micro_batches.is_empty() {
            return Err(error("an optimizer step needs at least one micro-batch"));
        }
        for batch in micro_batches {
            self.validate(batch)?;
        }
        let normalizer: usize = micro_batches.iter().map(|batch| batch.target_count).sum();
        let learning_rate = self.config.schedule.at(self.state.step);
        let mut loss_sum = 0.0f32;
        let (grads, norm) = match &self.trainable {
            Trainable::Full(_) => {
                let mut accumulator = GradientsAccumulator::<PackedLlamaForCausalLm<B>>::new();
                for batch in micro_batches {
                    let loss = self.micro_loss(batch, normalizer);
                    loss_sum += read_scalar(loss.clone())?;
                    let grads = GradientsParams::from_grads(loss.backward(), &self.model);
                    accumulator.accumulate::<B>(&self.model, grads);
                }
                let mut grads = self.synchronize(accumulator.grads())?;
                let norm =
                    clip_global_norm::<B, _>(&self.model, &mut grads, self.config.max_grad_norm)?;
                (grads, norm)
            }
            Trainable::Lora { adapters, .. } => {
                let mut accumulator = GradientsAccumulator::<LlamaLora<B>>::new();
                for batch in micro_batches {
                    let loss = self.micro_loss(batch, normalizer);
                    loss_sum += read_scalar(loss.clone())?;
                    let grads = GradientsParams::from_grads(loss.backward(), adapters);
                    accumulator.accumulate::<B>(adapters, grads);
                }
                let mut grads = self.synchronize(accumulator.grads())?;
                let norm =
                    clip_global_norm::<B, _>(adapters, &mut grads, self.config.max_grad_norm)?;
                (grads, norm)
            }
        };
        let applied = norm.is_finite() && loss_sum.is_finite();
        if applied {
            match &mut self.trainable {
                Trainable::Full(optimizer) => {
                    self.model = optimizer.step(learning_rate, self.model.clone(), grads);
                }
                Trainable::Lora { adapters, optimizer } => {
                    *adapters = optimizer.step(learning_rate, adapters.clone(), grads);
                }
            }
        } else {
            self.state.skipped_steps += 1;
        }
        self.state.step += 1;
        self.state.tokens += normalizer as u64;
        Ok(StepStats {
            step: self.state.step,
            loss: loss_sum,
            grad_norm: norm,
            learning_rate,
            tokens: normalizer,
            applied,
        })
    }
}

pub(super) fn read_scalar<B: ruda_tensor::api::backend::Backend>(tensor: Tensor<B, 1>) -> Result<f32, TrainError> {
    tensor
        .into_data()
        .convert::<f32>()
        .to_vec::<f32>()
        .map_err(|failure| error(format!("cannot read loss: {failure:?}")))
        .map(|values| values[0])
}

const LATEST_FILE: &str = "latest";
const STATE_FILE: &str = "state.json";
const TRAINABLE_FILE: &str = "trainable.safetensors";
const OPTIMIZER_FILE: &str = "optimizer";

fn io(failure: std::io::Error) -> TrainError {
    error(format!("checkpoint I/O failed: {failure}"))
}

impl<B: AutodiffBackend + TrainingBackend> SftTrainer<B> {
    /// Write `checkpoint_dir/step-N` atomically, point `latest` at it and
    /// delete older checkpoints beyond `keep_checkpoints`. For LoRA only the
    /// adapters are saved; the frozen base is reloaded from its source.
    pub fn save_checkpoint(&mut self) -> Result<PathBuf, TrainError> {
        self.save_checkpoint_with(|_| Ok(()))
    }

    pub(super) fn save_checkpoint_with(&mut self, extra: impl FnOnce(&Path) -> Result<(), TrainError>) -> Result<PathBuf, TrainError> {
        let root = &self.config.checkpoint_dir;
        std::fs::create_dir_all(root).map_err(io)?;
        let name = format!("step-{:08}", self.state.step);
        let destination = root.join(&name);
        let temporary = root.join(format!(".{name}.tmp"));
        if temporary.exists() {
            std::fs::remove_dir_all(&temporary).map_err(io)?;
        }
        std::fs::create_dir(&temporary).map_err(io)?;
        let recorder = BinFileRecorder::<FullPrecisionSettings>::new();
        let record = |failure| error(format!("cannot record checkpoint: {failure:?}"));
        match &self.trainable {
            Trainable::Full(optimizer) => {
                save_trainable(&self.model, &temporary.join(TRAINABLE_FILE)).map_err(error)?;
                Recorder::<B>::record(&recorder, optimizer.to_record(), temporary.join(OPTIMIZER_FILE))
                    .map_err(record)?;
            }
            Trainable::Lora { adapters, optimizer } => {
                save_trainable(adapters, &temporary.join(TRAINABLE_FILE)).map_err(error)?;
                Recorder::<B>::record(&recorder, optimizer.to_record(), temporary.join(OPTIMIZER_FILE))
                    .map_err(record)?;
            }
        }
        let state = serde_json::to_vec_pretty(&self.state).map_err(error)?;
        std::fs::write(temporary.join(STATE_FILE), state).map_err(io)?;
        extra(&temporary)?;
        if destination.exists() {
            std::fs::remove_dir_all(&destination).map_err(io)?;
        }
        std::fs::rename(&temporary, &destination).map_err(io)?;
        std::fs::write(root.join(LATEST_FILE), &name).map_err(io)?;
        self.prune_checkpoints()?;
        self.last_checkpoint = Some((self.state.step, Instant::now()));
        Ok(destination)
    }

    fn prune_checkpoints(&self) -> Result<(), TrainError> {
        let mut checkpoints: Vec<PathBuf> = std::fs::read_dir(&self.config.checkpoint_dir)
            .map_err(io)?
            .filter_map(Result::ok)
            .map(|entry| entry.path())
            .filter(|path| {
                path.is_dir()
                    && path
                        .file_name()
                        .and_then(|name| name.to_str())
                        .is_some_and(|name| name.starts_with("step-"))
            })
            .collect();
        checkpoints.sort();
        let keep = self.config.keep_checkpoints.max(1);
        let excess = checkpoints.len().saturating_sub(keep);
        for path in checkpoints.into_iter().take(excess) {
            std::fs::remove_dir_all(&path).map_err(io)?;
        }
        Ok(())
    }
}

impl<B: AutodiffBackend + TrainingBackend> SftTrainer<B> {
    /// Restore the newest checkpoint under `checkpoint_dir`, if any. Returns
    /// the restored step. The loader position is restored by [`Self::train`].
    pub fn resume(&mut self) -> Result<Option<usize>, TrainError> {
        let latest = self.config.checkpoint_dir.join(LATEST_FILE);
        if !latest.is_file() {
            return Ok(None);
        }
        let name = std::fs::read_to_string(&latest).map_err(io)?;
        self.load_checkpoint(&self.config.checkpoint_dir.join(name.trim()))
            .map(|step| Some(step))
    }

    /// Restore a specific checkpoint directory.
    pub fn load_checkpoint(&mut self, directory: &Path) -> Result<usize, TrainError> {
        let state: TrainerState = serde_json::from_slice(
            &std::fs::read(directory.join(STATE_FILE)).map_err(io)?,
        )
        .map_err(|failure| error(format!("invalid trainer state: {failure}")))?;
        let recorder = BinFileRecorder::<FullPrecisionSettings>::new();
        let load = |failure| error(format!("cannot load checkpoint: {failure:?}"));
        let trainable = std::mem::replace(
            &mut self.trainable,
            Trainable::Full(Self::optimizer_config(&self.config).init()),
        );
        self.trainable = match trainable {
            Trainable::Full(optimizer) => {
                self.model = load_trainable(&self.model, &directory.join(TRAINABLE_FILE), &self.device)
                    .map_err(error)?;
                let record = Recorder::<B>::load(&recorder, directory.join(OPTIMIZER_FILE), &self.device)
                    .map_err(load)?;
                Trainable::Full(optimizer.load_record(record))
            }
            Trainable::Lora { adapters, optimizer } => {
                let adapters = load_trainable(&adapters, &directory.join(TRAINABLE_FILE), &self.device)
                    .map_err(error)?;
                let record = Recorder::<B>::load(&recorder, directory.join(OPTIMIZER_FILE), &self.device)
                    .map_err(load)?;
                Trainable::Lora {
                    adapters,
                    optimizer: optimizer.load_record(record),
                }
            }
        };
        self.state = state;
        self.last_checkpoint = Some((state.step, Instant::now()));
        Ok(state.step)
    }
}

impl<B: AutodiffBackend + TrainingBackend> SftTrainer<B> {
    /// Train until `schedule.total_steps`, resuming the loader position from
    /// the trainer state when present. `on_progress` runs every `log_every`
    /// steps and must not block. Checkpoints are written every
    /// `checkpoint_every` steps and once at the end.
    pub fn train(
        &mut self,
        loader: &mut SftLoader,
        mut on_progress: impl FnMut(&TrainProgress),
    ) -> Result<TrainerState, TrainError> {
        if let Some(position) = self.state.loader {
            loader.restore(position);
        }
        let total_steps = self.config.schedule.total_steps;
        let started = Instant::now();
        let elapsed_before = self.state.elapsed_seconds;
        let log_every = self.config.log_every.max(1);
        let mut window_started = Instant::now();
        let mut window_tokens = 0u64;
        let mut window_steps = 0usize;
        while self.state.step < total_steps {
            let micro_batches: Vec<SftBatch> = (0..self.config.gradient_accumulation)
                .map(|_| loader.next_batch())
                .collect();
            let stats = self.step(&micro_batches)?;
            self.state.loader = Some(loader.state());
            self.state.elapsed_seconds = elapsed_before + started.elapsed().as_secs_f64();
            window_tokens += stats.tokens as u64;
            window_steps += 1;
            let checkpoint_due = self.config.checkpoint_every > 0
                && self.state.step % self.config.checkpoint_every == 0;
            if checkpoint_due || self.state.step == total_steps {
                self.save_checkpoint()?;
            }
            if self.state.step % log_every == 0 || self.state.step == total_steps {
                let window = window_started.elapsed();
                let seconds_per_step = window.as_secs_f64() / window_steps.max(1) as f64;
                let remaining = total_steps - self.state.step;
                on_progress(&TrainProgress {
                    stats,
                    total_steps,
                    tokens_seen: self.state.tokens,
                    elapsed: Duration::from_secs_f64(self.state.elapsed_seconds),
                    recent_tokens_per_second: window_tokens as f64 / window.as_secs_f64().max(1e-9),
                    eta: (window_steps > 0)
                        .then(|| Duration::from_secs_f64(seconds_per_step * remaining as f64)),
                    epoch: loader.state().epoch,
                    last_checkpoint: self
                        .last_checkpoint
                        .map(|(step, at)| (step, at.elapsed())),
                    skipped_steps: self.state.skipped_steps,
                });
                window_started = Instant::now();
                window_tokens = 0;
                window_steps = 0;
            }
        }
        Ok(self.state)
    }
}
