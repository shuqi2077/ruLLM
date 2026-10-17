//! Preference and on-policy training of packed causal models, full or LoRA.
use super::trainer::{clip_global_norm, load_trainable, read_scalar, save_trainable};
use super::{
    LinearLogProbsBackend, LlamaLora, SftTrainer, StepStats, TrainConfig, TrainError,
    TrainingBackend,
};
use crate::PackedLlamaForCausalLm;
use ruda_model::module::Module;
use ruda_model::record::{BinFileRecorder, FullPrecisionSettings, Recorder};
use ruda_nn::{Linear, LinearConfig};
use ruda_optim::{
    AdamW, AdamWConfig, GradientsParams, Optimizer, adaptor::OptimizerAdaptor,
};
use ruda_tensor::api::{
    DType, Int, Tensor, TensorData,
    activation::log_sigmoid,
    backend::{AutodiffBackend, Backend},
};
use serde::{Deserialize, Serialize};
use std::path::{Path, PathBuf};

mod packed;
use packed::{probabilities_many, critic_many, behavior_many};
pub(super) use packed::reward_many;

fn err(e: impl std::fmt::Display) -> TrainError {
    TrainError(e.to_string())
}

/// Prompt and generated continuation, without padding. Include EOS in completion
/// when it was generated; `terminal` distinguishes EOS from length truncation.
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub struct Response {
    pub prompt: Vec<i32>,
    pub completion: Vec<i32>,
    pub terminal: bool,
    /// Actual paged append end for each prompt/completion token. None denotes
    /// the original full-prefix scoring contract. Includes the last token's
    /// hypothetical append when generation stops before it enters KV.
    #[serde(default)]
    pub rotary_context: Option<Vec<usize>>,
    /// Captured from the sampling engine, not reconstructed from rounded or
    /// newly updated weights. Used only for old-policy rollout statistics.
    #[serde(default)]
    pub behavior_log_probs: Option<Vec<f32>>,
}

impl Response {
    pub(super) fn validate<B: Backend>(&self, model: &PackedLlamaForCausalLm<B>) -> Result<(), TrainError> {
        let geometry = model.geometry();
        if self.prompt.is_empty()
            || self.completion.is_empty()
            || self.prompt.len() + self.completion.len() > geometry.max_sequence_length
            || self
                .prompt
                .iter()
                .chain(&self.completion)
                .any(|&t| t < 0 || t as usize >= geometry.vocab_size)
        {
            return Err(err(
                "response needs nonempty prompt/completion within model context and vocabulary",
            ));
        }
        if let Some(contexts) = &self.rotary_context {
            let length = self.prompt.len() + self.completion.len();
            if contexts.len() != length || contexts.iter().enumerate()
                .any(|(position, &end)| end <= position || end > length)
                || contexts.windows(2).any(|pair| pair[0] > pair[1]) {
                return Err(err("response rotary append contexts do not cover its actual tokens"));
            }
        }
        if self.behavior_log_probs.as_ref().is_some_and(|values| values.len() != self.completion.len()
            || values.iter().any(|x| !x.is_finite() || *x > 0.0)) {
            return Err(err("behavior log probabilities must match completion tokens"));
        }
        Ok(())
    }
    pub(super) fn inputs<B: Backend>(&self, device: &B::Device, include_last: bool) -> Tensor<B, 2, Int> {
        let mut tokens = self.prompt.clone();
        tokens.extend(&self.completion);
        if !include_last {
            tokens.pop();
        }
        let length = tokens.len();
        Tensor::from_data(TensorData::new(tokens, [1, length]), device)
    }
}

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub struct PreferencePair {
    pub chosen: Response,
    pub rejected: Response,
}

/// Explicit, immutable reference, which may itself contain frozen adapters.
pub struct ReferencePolicy<B: Backend> {
    model: PackedLlamaForCausalLm<B>,
    adapters: Option<LlamaLora<B>>,
    pub id: String,
}
impl<B: Backend> ReferencePolicy<B> {
    pub fn new(
        model: PackedLlamaForCausalLm<B>,
        adapters: Option<LlamaLora<B>>,
        id: String,
    ) -> Result<Self, TrainError> {
        if id.is_empty() {
            return Err(err("reference identity must be nonempty"));
        }
        Ok(Self {
            model: model.no_grad(),
            adapters: adapters.map(Module::no_grad),
            id,
        })
    }
}

fn probabilities<B: TrainingBackend + LinearLogProbsBackend>(
    model: &PackedLlamaForCausalLm<B>,
    adapters: Option<&LlamaLora<B>>,
    response: &Response,
    device: &B::Device,
    chunk: usize,
) -> Tensor<B, 1> {
    if response.rotary_context.is_some() {
        return probabilities_many(model, adapters, &[response], device, chunk).remove(0);
    }
    // Runtime RoPE depends on the append context. Score each action on the
    // very prefix used by the categorical sampler, not the final response's
    // longer rotary context (which would change the old-policy distribution).
    if model.paged_rotary_states().iter().any(Option::is_some) {
        let mut prefix = response.prompt.clone();
        let mut selected = Vec::with_capacity(response.completion.len());
        for &token in &response.completion {
            let length = prefix.len();
            let mut targets = vec![-100; length];
            targets[length - 1] = token;
            selected.push(
                model
                    .train_log_probs(
                        Tensor::from_data(TensorData::new(prefix.clone(), [1, length]), device),
                        Tensor::from_data(TensorData::new(targets, [length]), device),
                        adapters,
                        chunk,
                    )
                    .slice([length - 1..length]),
            );
            prefix.push(token);
        }
        return Tensor::cat(selected, 0);
    }
    let start = response.prompt.len() - 1;
    let end = start + response.completion.len();
    let mut targets = vec![-100; start];
    targets.extend(&response.completion);
    model
        .train_log_probs(
            response.inputs(device, false),
            Tensor::from_data(TensorData::new(targets, [end]), device),
            adapters,
            chunk,
        )
        .slice([start..end])
}

fn values<B: Backend>(tensor: Tensor<B, 1>) -> Result<Vec<f32>, TrainError> {
    tensor
        .into_data()
        .convert::<f32>()
        .to_vec::<f32>()
        .map_err(|e| err(format!("cannot read rollout: {e:?}")))
}
fn vector<B: Backend>(data: &[f32], device: &B::Device) -> Tensor<B, 1> {
    Tensor::from_data(
        TensorData::new(data.to_vec(), [data.len()]),
        (device, DType::F32),
    )
}

/// Mutable application cursor/buffer and exact categorical-sampler RNG. Store
/// any in-flight rollout and update-epoch cursor in `data` before checkpointing.
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub struct PostTrainingState {
    pub sampling_seed: u64,
    pub data: serde_json::Value,
}
impl PostTrainingState {
    pub fn new(sampling_seed: u64) -> Self {
        Self {
            sampling_seed,
            data: serde_json::Value::Null,
        }
    }
}

#[derive(Serialize, Deserialize, PartialEq)]
pub(super) struct Metadata {
    version: u32,
    algorithm: String,
    reference: String,
    actor: serde_json::Value,
    options: serde_json::Value,
    state: PostTrainingState,
}
pub(super) fn metadata<B: AutodiffBackend + TrainingBackend>(
    actor: &SftTrainer<B>,
    algorithm: &str,
    reference: &str,
    options: serde_json::Value,
    state: &PostTrainingState,
) -> Metadata {
    Metadata {
        version: 1,
        algorithm: algorithm.into(),
        reference: reference.into(),
        actor: actor.objective_contract(),
        options,
        state: state.clone(),
    }
}
pub(super) fn write_metadata(path: &Path, meta: &Metadata) -> Result<(), TrainError> {
    std::fs::write(
        path.join("post-training.json"),
        serde_json::to_vec(meta).map_err(err)?,
    )
    .map_err(err)
}
pub(super) fn read_metadata(path: &Path, expected: &Metadata) -> Result<PostTrainingState, TrainError> {
    let actual: Metadata =
        serde_json::from_slice(&std::fs::read(path.join("post-training.json")).map_err(err)?)
            .map_err(err)?;
    if actual.version != expected.version
        || actual.algorithm != expected.algorithm
        || actual.reference != expected.reference
        || actual.actor != expected.actor
        || actual.options != expected.options
    {
        return Err(err(
            "checkpoint post-training algorithm, reference identity or options differ",
        ));
    }
    Ok(actual.state)
}

/// Full-vocabulary, temperature-one sampling. There is no top-k/p rejection
/// distribution to mismatch the old-policy log probabilities used in updates.
fn sample<B: AutodiffBackend + TrainingBackend>(
    actor: &SftTrainer<B>,
    prompt: &[i32],
    max_new_tokens: usize,
    eos: &[i32],
    state: &mut PostTrainingState,
) -> Result<Response, TrainError> {
    let geometry = actor.model().geometry();
    if prompt.is_empty()
        || max_new_tokens == 0
        || prompt.len() + max_new_tokens > geometry.max_sequence_length
        || prompt
            .iter()
            .chain(eos)
            .any(|&t| t < 0 || t as usize >= geometry.vocab_size)
    {
        return Err(err("invalid rollout prompt, token limit or EOS IDs"));
    }
    let model = actor.model().clone().no_grad();
    let adapters = actor.adapters().cloned().map(Module::no_grad);
    let mut tokens = prompt.to_vec();
    let mut terminal = false;
    for _ in 0..max_new_tokens {
        let input = Tensor::from_data(
            TensorData::new(tokens.clone(), [1, tokens.len()]),
            actor.device(),
        );
        let logits = values(model.train_last_logits(input, adapters.as_ref()))?;
        let maximum = logits.iter().copied().fold(f32::NEG_INFINITY, f32::max);
        let weights: Vec<f64> = logits
            .iter()
            .map(|&x| ((x - maximum) as f64).exp())
            .collect();
        let total: f64 = weights.iter().sum();
        if !total.is_finite() || total <= 0.0 {
            return Err(err("nonfinite rollout distribution"));
        }
        state.sampling_seed = state.sampling_seed.wrapping_add(0x9E3779B97F4A7C15);
        let mut z = state.sampling_seed;
        z = (z ^ (z >> 30)).wrapping_mul(0xBF58476D1CE4E5B9);
        z = (z ^ (z >> 27)).wrapping_mul(0x94D049BB133111EB);
        z ^= z >> 31;
        let draw = ((z >> 11) as f64) / ((1u64 << 53) as f64) * total;
        let mut cumulative = 0.0;
        let mut token = weights.len() - 1;
        for (index, weight) in weights.iter().enumerate() {
            cumulative += weight;
            if draw < cumulative {
                token = index;
                break;
            }
        }
        tokens.push(token as i32);
        if eos.contains(&(token as i32)) {
            terminal = true;
            break;
        }
    }
    Ok(Response {
        prompt: prompt.to_vec(),
        completion: tokens[prompt.len()..].to_vec(),
        terminal,
        rotary_context: None,
        behavior_log_probs: None,
    })
}

pub struct DpoTrainer<B: AutodiffBackend + TrainingBackend> {
    pub actor: SftTrainer<B>,
    reference: ReferencePolicy<B>,
    beta: f32,
    pub state: PostTrainingState,
}
impl<B: AutodiffBackend + TrainingBackend + LinearLogProbsBackend> DpoTrainer<B> {
    pub fn new(
        actor: SftTrainer<B>,
        reference: ReferencePolicy<B>,
        beta: f32,
        seed: u64,
    ) -> Result<Self, TrainError> {
        if !beta.is_finite() || beta <= 0.0 {
            return Err(err("DPO beta must be positive"));
        }
        Ok(Self {
            actor,
            reference,
            beta,
            state: PostTrainingState::new(seed),
        })
    }
    /// One effective-batch update; chosen/rejected likelihoods are completion
    /// sums and every preference pair has equal weight, independent of length.
    pub fn step(&mut self, pairs: &[PreferencePair]) -> Result<StepStats, TrainError> {
        if pairs.is_empty() { return Err(err("DPO requires a nonempty local preference batch")); }
        for pair in pairs {
            pair.chosen.validate(self.actor.model())?;
            pair.rejected.validate(self.actor.model())?;
            pair.chosen.validate(&self.reference.model)?;
            pair.rejected.validate(&self.reference.model)?;
            if pair.chosen.prompt != pair.rejected.prompt {
                return Err(err("DPO pair prompts differ"));
            }
        }
        let normalizer = self.actor.objective_normalizer(pairs.len())?;
        let device = self.actor.device().clone();
        let chunk = self.actor.loss_chunk_rows();
        let reference = &self.reference;
        let beta = self.beta;
        let tokens = pairs
            .iter()
            .map(|p| p.chosen.completion.len() + p.rejected.completion.len())
            .sum();
        let responses: Vec<_> = pairs.iter().map(|p| &p.chosen)
            .chain(pairs.iter().map(|p| &p.rejected)).collect();
        self.actor
            .objective_step(1, tokens, true, |model, adapters, _| {
                let policy = Tensor::cat(probabilities_many(model, adapters, &responses, &device, chunk)
                    .into_iter().map(Tensor::sum).collect(), 0);
                let frozen = Tensor::cat(probabilities_many(&reference.model, reference.adapters.as_ref(),
                    &responses, &device, chunk).into_iter().map(|x| x.sum().detach()).collect(), 0);
                let n = pairs.len();
                let ratio = policy - frozen;
                Ok(log_sigmoid((ratio.clone().slice([0..n]) - ratio.slice([n..2*n])) * beta)
                    .neg().sum() / normalizer)
            })
    }
    pub fn save_checkpoint(&mut self) -> Result<PathBuf, TrainError> {
        let meta = metadata(
            &self.actor,
            "dpo",
            &self.reference.id,
            serde_json::json!({"beta": self.beta}),
            &self.state,
        );
        self.actor
            .save_checkpoint_with(|p| write_metadata(p, &meta))
    }
    pub fn load_checkpoint(&mut self, path: &Path) -> Result<usize, TrainError> {
        let state = read_metadata(
            path,
            &metadata(
                &self.actor,
                "dpo",
                &self.reference.id,
                serde_json::json!({"beta": self.beta}),
                &self.state,
            ),
        )?;
        let step = self.actor.load_checkpoint(path)?;
        self.state = state;
        Ok(step)
    }
}

/// Independent trainable value backbone and scalar head. Its unused LM head
/// receives no gradients; value-path parameters have their own AdamW state.
#[derive(Module, Debug)]
pub struct ValueCritic<B: Backend> {
    pub backbone: PackedLlamaForCausalLm<B>,
    pub head: Linear<B>,
}
impl<B: TrainingBackend> ValueCritic<B> {
    pub fn new(backbone: PackedLlamaForCausalLm<B>, device: &B::Device) -> Self {
        let width = backbone.lora_shapes().hidden;
        Self {
            backbone,
            head: LinearConfig::new(width, 1).init(device),
        }
    }
    fn response_values(&self, response: &Response, device: &B::Device) -> Tensor<B, 1> {
        if response.rotary_context.is_some() {
            return critic_many(self, &[response], device).remove(0);
        }
        if self
            .backbone
            .paged_rotary_states()
            .iter()
            .any(Option::is_some)
        {
            let mut prefix = response.prompt.clone();
            let mut selected = Vec::with_capacity(response.completion.len() + 1);
            for t in 0..=response.completion.len() {
                let length = prefix.len();
                let hidden = self.backbone.forward_train_hidden(
                    Tensor::from_data(TensorData::new(prefix.clone(), [1, length]), device),
                    None,
                );
                let width = hidden.dims()[2];
                selected.push(
                    self.head
                        .forward(hidden.slice([0..1, length - 1..length, 0..width]))
                        .reshape([1])
                        .cast(DType::F32),
                );
                if t < response.completion.len() {
                    prefix.push(response.completion[t]);
                }
            }
            return Tensor::cat(selected, 0);
        }
        let hidden = self
            .backbone
            .forward_train_hidden(response.inputs(device, true), None);
        let start = response.prompt.len() - 1;
        let end = start + response.completion.len() + 1;
        self.head
            .forward(hidden)
            .reshape([end])
            .slice([start..end])
            .cast(DType::F32)
    }
}

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub struct PpoConfig {
    pub clip_ratio: f32,
    pub value_clip: Option<f32>,
    pub value_coefficient: f32,
    pub gamma: f32,
    pub gae_lambda: f32,
    pub kl_coefficient: f32,
}
impl Default for PpoConfig {
    fn default() -> Self {
        Self {
            clip_ratio: 0.2,
            value_clip: Some(0.2),
            value_coefficient: 0.5,
            gamma: 1.0,
            gae_lambda: 0.95,
            kl_coefficient: 0.0,
        }
    }
}
impl PpoConfig {
    fn validate(&self) -> Result<(), TrainError> {
        if !self.clip_ratio.is_finite()
            || !(0.0..1.0).contains(&self.clip_ratio)
            || !self.gamma.is_finite()
            || !(0.0..=1.0).contains(&self.gamma)
            || !self.gae_lambda.is_finite()
            || !(0.0..=1.0).contains(&self.gae_lambda)
            || !self.value_coefficient.is_finite()
            || self.value_coefficient < 0.0
            || !self.kl_coefficient.is_finite()
            || self.kl_coefficient < 0.0
            || self.value_clip.is_some_and(|x| !x.is_finite() || x < 0.0)
        {
            return Err(err("invalid PPO configuration"));
        }
        Ok(())
    }
}

/// Fixed rollout-policy statistics. Reuse this same object for all PPO epochs;
/// recomputing old probabilities after updates would invalidate clipping.
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub struct PpoRollout {
    pub response: Response,
    pub old_log_probs: Vec<f32>,
    pub old_values: Vec<f32>,
    pub advantages: Vec<f32>,
    pub returns: Vec<f32>,
}
impl PpoRollout {
    fn validate<B: Backend>(&self, model: &PackedLlamaForCausalLm<B>) -> Result<(), TrainError> {
        self.response.validate(model)?;
        let n = self.response.completion.len();
        if [
            &self.old_log_probs,
            &self.old_values,
            &self.advantages,
            &self.returns,
        ]
        .iter()
        .any(|v| v.len() != n || v.iter().any(|x| !x.is_finite()))
        {
            return Err(err(
                "PPO rollout statistics must match response tokens and be finite",
            ));
        }
        Ok(())
    }
}

type CriticOptimizer<B> = OptimizerAdaptor<AdamW, ValueCritic<B>, B>;
pub struct PpoTrainer<B: AutodiffBackend + TrainingBackend> {
    pub actor: SftTrainer<B>,
    pub critic: ValueCritic<B>,
    critic_optimizer: CriticOptimizer<B>,
    critic_config: TrainConfig,
    reference: ReferencePolicy<B>,
    pub config: PpoConfig,
    pub state: PostTrainingState,
}
impl<B: AutodiffBackend + TrainingBackend + LinearLogProbsBackend> PpoTrainer<B> {
    pub fn new(
        actor: SftTrainer<B>,
        critic: ValueCritic<B>,
        critic_config: TrainConfig,
        reference: ReferencePolicy<B>,
        config: PpoConfig,
        seed: u64,
    ) -> Result<Self, TrainError> {
        config.validate()?;
        let optimizer = AdamWConfig::new()
            .with_beta_1(critic_config.beta_1)
            .with_beta_2(critic_config.beta_2)
            .with_epsilon(critic_config.epsilon)
            .with_weight_decay(critic_config.weight_decay)
            .init();
        Ok(Self {
            actor,
            critic,
            critic_optimizer: optimizer,
            critic_config,
            reference,
            config,
            state: PostTrainingState::new(seed),
        })
    }
    pub fn sample(
        &mut self,
        prompt: &[i32],
        max_new_tokens: usize,
        eos: &[i32],
    ) -> Result<Response, TrainError> {
        sample(&self.actor, prompt, max_new_tokens, eos, &mut self.state)
    }
    /// Capture before updating. Caller supplies the actual per-token rewards;
    /// terminal trajectories bootstrap zero, length-truncated ones use V(last).
    pub fn prepare_rollout(
        &self,
        response: Response,
        rewards: &[f32],
    ) -> Result<PpoRollout, TrainError> {
        self.config.validate()?;
        response.validate(self.actor.model())?;
        response.validate(&self.critic.backbone)?;
        response.validate(&self.reference.model)?;
        if rewards.len() != response.completion.len() || rewards.iter().any(|x| !x.is_finite()) {
            return Err(err("one finite reward per completion token is required"));
        }
        let device = self.actor.device();
        let chunk = self.actor.loss_chunk_rows();
        let old = match &response.behavior_log_probs {
            Some(logp) => logp.clone(),
            None => values(probabilities(
            &self.actor.model().clone().no_grad(),
            self.actor.adapters().cloned().map(Module::no_grad).as_ref(),
            &response,
            device,
            chunk,
        ))?,
        };
        let reference = values(probabilities(
            &self.reference.model,
            self.reference.adapters.as_ref(),
            &response,
            device,
            chunk,
        ))?;
        let old_values = values(
            self.critic
                .clone()
                .no_grad()
                .response_values(&response, device),
        )?;
        let n = rewards.len();
        let mut advantages = vec![0.0; n];
        let mut returns = vec![0.0; n];
        let mut gae = 0.0;
        for t in (0..n).rev() {
            let next = if t + 1 == n && response.terminal {
                0.0
            } else {
                old_values[t + 1]
            };
            let reward = rewards[t] - self.config.kl_coefficient * (old[t] - reference[t]);
            let delta = reward + self.config.gamma * next - old_values[t];
            gae = delta + self.config.gamma * self.config.gae_lambda * gae;
            advantages[t] = gae;
            returns[t] = gae + old_values[t];
        }
        Ok(PpoRollout {
            response,
            old_log_probs: old,
            old_values: old_values[..n].to_vec(),
            advantages,
            returns,
        })
    }
    /// Token-mean clipped policy/value losses over the entire effective batch.
    pub fn step(&mut self, rollouts: &[PpoRollout]) -> Result<StepStats, TrainError> {
        if rollouts.is_empty() { return Err(err("PPO requires a nonempty local rollout batch")); }
        self.config.validate()?;
        for rollout in rollouts {
            rollout.validate(self.actor.model())?;
            rollout.response.validate(&self.critic.backbone)?;
        }
        let tokens: usize = rollouts.iter().map(|r| r.response.completion.len()).sum();
        let normalizer = self.actor.objective_normalizer(tokens)?;
        let device = self.actor.device().clone();
        let chunk = self.actor.loss_chunk_rows();
        let responses: Vec<_> = rollouts.iter().map(|r| &r.response).collect();
        let predicted = critic_many(&self.critic, &responses, &device);
        let mut critic_losses = Vec::new();
        for (rollout, predicted) in rollouts.iter().zip(predicted) {
            let n = rollout.response.completion.len();
            let predicted = predicted.slice([0..n]);
            let target = vector(&rollout.returns, &device);
            let loss = (predicted.clone() - target.clone()).square();
            let loss = if let Some(clip) = self.config.value_clip {
                let old = vector(&rollout.old_values, &device);
                let clipped = old.clone() + (predicted - old).clamp(-clip, clip);
                loss.max_pair((clipped - target).square())
            } else {
                loss
            };
            let loss = loss.sum() * (0.5 * self.config.value_coefficient / normalizer);
            critic_losses.push(loss);
        }
        let loss = Tensor::cat(critic_losses, 0).sum();
        let mut critic_loss = read_scalar(loss.clone())?;
        let mut grads = self.actor.synchronize(GradientsParams::from_grads(loss.backward(), &self.critic))?;
        let norm =
            clip_global_norm::<B, _>(&self.critic, &mut grads, self.critic_config.max_grad_norm)?;
        critic_loss = self.actor.objective_mean(critic_loss)?;
        let config = &self.config;
        let mut stats = self.actor.objective_step(
            1,
            tokens,
            norm.is_finite() && critic_loss.is_finite(),
            |model, adapters, _| {
                let current = probabilities_many(model, adapters, &responses, &device, chunk);
                let mut losses = Vec::new();
                for (r, logp) in rollouts.iter().zip(current) {
                let ratio = (logp - vector(&r.old_log_probs, &device)).exp();
                let advantage = vector(&r.advantages, &device);
                losses.push((ratio.clone() * advantage.clone())
                    .min_pair(
                        ratio.clamp(1.0 - config.clip_ratio, 1.0 + config.clip_ratio) * advantage,
                    )
                    .sum()
                    .neg()
                    / normalizer);
                }
                Ok(Tensor::cat(losses, 0).sum())
            },
        )?;
        if stats.applied {
            self.critic = self.critic_optimizer.step(
                self.critic_config.schedule.at(stats.step - 1),
                self.critic.clone(),
                grads,
            );
        }
        stats.loss += critic_loss;
        stats.grad_norm = stats.grad_norm.max(norm);
        Ok(stats)
    }
    fn options(&self) -> serde_json::Value {
        let c = &self.critic_config;
        serde_json::json!({"ppo": self.config, "critic": {"peak": c.schedule.peak, "min": c.schedule.min,
            "warmup": c.schedule.warmup_steps, "steps": c.schedule.total_steps, "beta1": c.beta_1,
            "beta2": c.beta_2, "epsilon": c.epsilon, "weight_decay": c.weight_decay, "max_grad_norm": c.max_grad_norm}})
    }
    pub fn save_checkpoint(&mut self) -> Result<PathBuf, TrainError> {
        let meta = metadata(
            &self.actor,
            "ppo",
            &self.reference.id,
            self.options(),
            &self.state,
        );
        let critic = &self.critic;
        let optimizer = &self.critic_optimizer;
        self.actor.save_checkpoint_with(|p| {
            save_trainable(critic, &p.join("critic.safetensors")).map_err(err)?;
            Recorder::<B>::record(
                &BinFileRecorder::<FullPrecisionSettings>::new(),
                optimizer.to_record(),
                p.join("critic-optimizer"),
            )
            .map_err(|e| err(format!("critic optimizer record: {e:?}")))?;
            write_metadata(p, &meta)
        })
    }
    pub fn load_checkpoint(&mut self, path: &Path) -> Result<usize, TrainError> {
        let state = read_metadata(
            path,
            &metadata(
                &self.actor,
                "ppo",
                &self.reference.id,
                self.options(),
                &self.state,
            ),
        )?;
        let critic = load_trainable(
            &self.critic,
            &path.join("critic.safetensors"),
            self.actor.device(),
        )
        .map_err(err)?;
        let record = Recorder::<B>::load(
            &BinFileRecorder::<FullPrecisionSettings>::new(),
            path.join("critic-optimizer"),
            self.actor.device(),
        )
        .map_err(|e| err(format!("critic optimizer load: {e:?}")))?;
        let step = self.actor.load_checkpoint(path)?;
        self.critic = critic;
        self.critic_optimizer = self.critic_optimizer.clone().load_record(record);
        self.state = state;
        Ok(step)
    }
}

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub struct GrpoConfig {
    pub clip_ratio: f32,
    pub kl_coefficient: f32,
    pub reward_epsilon: f32,
}
impl Default for GrpoConfig {
    fn default() -> Self {
        Self {
            clip_ratio: 0.2,
            kl_coefficient: 0.04,
            reward_epsilon: 1e-4,
        }
    }
}
impl GrpoConfig {
    fn validate(&self) -> Result<(), TrainError> {
        if !self.clip_ratio.is_finite()
            || !(0.0..1.0).contains(&self.clip_ratio)
            || !self.kl_coefficient.is_finite()
            || self.kl_coefficient < 0.0
            || !self.reward_epsilon.is_finite()
            || self.reward_epsilon <= 0.0
        {
            return Err(err("invalid GRPO configuration"));
        }
        Ok(())
    }
}
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub struct GrpoRollout {
    pub response: Response,
    pub old_log_probs: Vec<f32>,
    pub reference_log_probs: Vec<f32>,
    pub advantage: f32,
}
pub struct GrpoTrainer<B: AutodiffBackend + TrainingBackend> {
    pub actor: SftTrainer<B>,
    reference: ReferencePolicy<B>,
    pub config: GrpoConfig,
    pub state: PostTrainingState,
}
impl<B: AutodiffBackend + TrainingBackend + LinearLogProbsBackend> GrpoTrainer<B> {
    pub fn new(
        actor: SftTrainer<B>,
        reference: ReferencePolicy<B>,
        config: GrpoConfig,
        seed: u64,
    ) -> Result<Self, TrainError> {
        config.validate()?;
        Ok(Self {
            actor,
            reference,
            config,
            state: PostTrainingState::new(seed),
        })
    }
    pub fn sample(
        &mut self,
        prompt: &[i32],
        max_new_tokens: usize,
        eos: &[i32],
    ) -> Result<Response, TrainError> {
        sample(&self.actor, prompt, max_new_tokens, eos, &mut self.state)
    }
    /// A complete prompt group (at least two responses). Outcome rewards are
    /// normalized with population standard deviation; equal rewards give zero.
    pub fn prepare_group(
        &self,
        responses: Vec<Response>,
        rewards: &[f32],
    ) -> Result<Vec<GrpoRollout>, TrainError> {
        self.config.validate()?;
        if responses.len() < 2
            || responses.len() != rewards.len()
            || rewards.iter().any(|x| !x.is_finite())
            || responses.iter().any(|r| r.prompt != responses[0].prompt)
        {
            return Err(err(
                "GRPO requires a complete same-prompt group and finite rewards",
            ));
        }
        let mean = rewards.iter().sum::<f32>() / rewards.len() as f32;
        let std =
            (rewards.iter().map(|r| (r - mean).powi(2)).sum::<f32>() / rewards.len() as f32).sqrt();
        let model = self.actor.model().clone().no_grad();
        let adapters = self.actor.adapters().cloned().map(Module::no_grad);
        for response in &responses {
            response.validate(&model)?;
            response.validate(&self.reference.model)?;
        }
        let refs: Vec<_> = responses.iter().collect();
        let old = behavior_many(&model, adapters.as_ref(), &refs, self.actor.device(), self.actor.loss_chunk_rows())?;
        let reference = probabilities_many(&self.reference.model, self.reference.adapters.as_ref(), &refs,
            self.actor.device(), self.actor.loss_chunk_rows());
        responses
            .into_iter()
            .zip(rewards)
            .zip(old.into_iter().zip(reference))
            .map(|((response, reward), (old, reference))| {
                let old_log_probs = old;
                let reference_log_probs = values(reference)?;
                Ok(GrpoRollout {
                    response,
                    old_log_probs,
                    reference_log_probs,
                    advantage: (reward - mean) / (std + self.config.reward_epsilon),
                })
            })
            .collect()
    }
    /// Mean of per-response token means, not a length-biased global token mean.
    pub fn step(&mut self, rollouts: &[GrpoRollout]) -> Result<StepStats, TrainError> {
        if rollouts.is_empty() { return Err(err("GRPO requires a nonempty local rollout batch")); }
        self.config.validate()?;
        for r in rollouts {
            r.response.validate(self.actor.model())?;
            if !r.advantage.is_finite()
                || [&r.old_log_probs, &r.reference_log_probs].iter().any(|v| {
                    v.len() != r.response.completion.len() || v.iter().any(|x| !x.is_finite())
                })
            {
                return Err(err("invalid fixed GRPO rollout statistics"));
            }
        }
        let normalizer = self.actor.objective_normalizer(rollouts.len())?;
        let device = self.actor.device().clone();
        let chunk = self.actor.loss_chunk_rows();
        let config = &self.config;
        let tokens = rollouts.iter().map(|r| r.response.completion.len()).sum();
        let responses: Vec<_> = rollouts.iter().map(|r| &r.response).collect();
        self.actor
            .objective_step(1, tokens, true, |model, adapters, _| {
                let current = probabilities_many(model, adapters, &responses, &device, chunk);
                let mut losses = Vec::new();
                for (r, logp) in rollouts.iter().zip(current) {
                let ratio = (logp.clone() - vector(&r.old_log_probs, &device)).exp();
                let policy = (ratio.clone() * r.advantage).min_pair(
                    ratio.clamp(1.0 - config.clip_ratio, 1.0 + config.clip_ratio) * r.advantage,
                );
                let delta = vector(&r.reference_log_probs, &device) - logp;
                let kl = delta.clone().exp() - delta - 1.0;
                losses.push((kl * config.kl_coefficient - policy).mean() / normalizer);
                }
                Ok(Tensor::cat(losses, 0).sum())
            })
    }
    pub fn save_checkpoint(&mut self) -> Result<PathBuf, TrainError> {
        let meta = metadata(
            &self.actor,
            "grpo",
            &self.reference.id,
            serde_json::to_value(&self.config).map_err(err)?,
            &self.state,
        );
        self.actor
            .save_checkpoint_with(|p| write_metadata(p, &meta))
    }
    pub fn load_checkpoint(&mut self, path: &Path) -> Result<usize, TrainError> {
        let state = read_metadata(
            path,
            &metadata(
                &self.actor,
                "grpo",
                &self.reference.id,
                serde_json::to_value(&self.config).map_err(err)?,
                &self.state,
            ),
        )?;
        let step = self.actor.load_checkpoint(path)?;
        self.state = state;
        Ok(step)
    }
}
