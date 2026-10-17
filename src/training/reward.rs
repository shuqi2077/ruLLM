//! Pairwise reward learning and explicit outcome-reward scoring.
use super::post_training::{metadata, read_metadata, write_metadata, reward_many};
use super::trainer::{load_trainable, read_scalar, save_trainable};
use super::{
    LlamaLora, PostTrainingState, PreferencePair, Response, SftTrainer, StepStats, TrainConfig,
    TrainError, TrainingBackend,
};
use crate::PackedLlamaForCausalLm;
use ruda_model::module::Module;
use ruda_model::record::{BinFileRecorder, FullPrecisionSettings, Recorder};
use ruda_nn::{Linear, LinearConfig};
use ruda_optim::{AdamW, AdamWConfig, Optimizer, adaptor::OptimizerAdaptor};
use ruda_tensor::api::{DType, Tensor, activation::log_sigmoid, backend::AutodiffBackend};
use std::path::{Path, PathBuf};

fn error(value: impl std::fmt::Display) -> TrainError {
    TrainError(value.to_string())
}

fn score<B: TrainingBackend>(
    model: &PackedLlamaForCausalLm<B>,
    adapters: Option<&LlamaLora<B>>,
    head: &Linear<B>,
    response: &Response,
    device: &B::Device,
) -> Tensor<B, 1> {
    if response.rotary_context.is_some() {
        return reward_many(model, adapters, head, &[response], device);
    }
    let length = response.prompt.len() + response.completion.len();
    let hidden = model.forward_train_hidden(response.inputs(device, true), adapters);
    head.forward(hidden.slice([0..1, length - 1..length, 0..model.lora_shapes().hidden]))
        .reshape([1])
        .cast(DType::F32)
}

/// Bradley-Terry preference training: equal-weight mean of
/// `-log_sigmoid(score(chosen) - score(rejected))` at the last response token.
/// The actor can be full or LoRA. Its backbone and the independently optimized
/// scalar head receive derivatives from the same forward/backward, not LM loss.
pub struct RewardTrainer<B: AutodiffBackend + TrainingBackend> {
    pub actor: SftTrainer<B>,
    pub head: Linear<B>,
    head_optimizer: OptimizerAdaptor<AdamW, Linear<B>, B>,
    head_config: TrainConfig,
    pub state: PostTrainingState,
}

impl<B: AutodiffBackend + TrainingBackend> RewardTrainer<B> {
    pub fn new(actor: SftTrainer<B>, head_config: TrainConfig) -> Self {
        let head = LinearConfig::new(actor.model().lora_shapes().hidden, 1)
            .with_bias(false)
            .init(actor.device());
        let head_optimizer = AdamWConfig::new()
            .with_beta_1(head_config.beta_1)
            .with_beta_2(head_config.beta_2)
            .with_epsilon(head_config.epsilon)
            .with_weight_decay(head_config.weight_decay)
            .init();
        Self {
            actor,
            head,
            head_optimizer,
            head_config,
            state: PostTrainingState::new(0),
        }
    }

    pub fn step(&mut self, pairs: &[PreferencePair]) -> Result<StepStats, TrainError> {
        if pairs.is_empty() { return Err(error("reward training requires a nonempty local preference batch")); }
        for pair in pairs {
            pair.chosen.validate(self.actor.model())?;
            pair.rejected.validate(self.actor.model())?;
            if pair.chosen.prompt != pair.rejected.prompt {
                return Err(error("reward preference pair prompts differ"));
            }
        }
        let normalizer = self.actor.objective_normalizer(pairs.len())?;
        let device = self.actor.device().clone();
        let tokens = pairs
            .iter()
            .map(|pair| pair.chosen.completion.len() + pair.rejected.completion.len())
            .sum();
        let head = &self.head;
        let responses: Vec<_> = pairs.iter().map(|p| &p.chosen)
            .chain(pairs.iter().map(|p| &p.rejected)).collect();
        let (stats, grads) = self.actor.objective_step_with_aux(
            1,
            tokens,
            head,
            self.head_config.max_grad_norm,
            |model, adapters, _| {
                let scores = reward_many(model, adapters, head, &responses, &device);
                let n = pairs.len();
                Ok(log_sigmoid(scores.clone().slice([0..n]) - scores.slice([n..2*n]))
                    .neg().sum() / normalizer)
            },
        )?;
        if stats.applied {
            self.head = self.head_optimizer.step(
                self.head_config.schedule.at(stats.step - 1),
                self.head.clone(),
                grads,
            );
        }
        Ok(stats)
    }

    /// Frozen scoring of the entire unpadded response, including generated EOS.
    /// No reward normalization, centering, clipping or KL term is inserted.
    pub fn score(&self, response: &Response) -> Result<f32, TrainError> {
        response.validate(self.actor.model())?;
        read_scalar(score(
            &self.actor.model().clone().no_grad(),
            self.actor.adapters().cloned().map(Module::no_grad).as_ref(),
            &self.head.clone().no_grad(),
            response,
            self.actor.device(),
        ))
    }

    /// One packed backbone forward for all complete responses, in input order.
    pub fn score_batch(&self, responses: &[Response]) -> Result<Vec<f32>, TrainError> {
        if responses.is_empty() { return Err(error("reward scoring needs at least one response")); }
        for response in responses { response.validate(self.actor.model())?; }
        let refs: Vec<_> = responses.iter().collect();
        reward_many(&self.actor.model().clone().no_grad(),
            self.actor.adapters().cloned().map(Module::no_grad).as_ref(),
            &self.head.clone().no_grad(), &refs, self.actor.device())
            .into_data().convert::<f32>().to_vec::<f32>()
            .map_err(|e| error(format!("cannot read rewards: {e:?}")))
    }

    pub fn token_rewards_batch(&self, responses: &[Response]) -> Result<Vec<Vec<f32>>, TrainError> {
        Ok(responses.iter().zip(self.score_batch(responses)?).map(|(response, reward)| {
            let mut rewards = vec![0.0; response.completion.len()];
            *rewards.last_mut().expect("validated completion") = reward;
            rewards
        }).collect())
    }

    /// Explicit sparse outcome rewards for PPO: zero except the last action.
    /// GRPO can consume `score` directly as its per-response outcome reward.
    pub fn token_rewards(&self, response: &Response) -> Result<Vec<f32>, TrainError> {
        let reward = self.score(response)?;
        let mut rewards = vec![0.0; response.completion.len()];
        *rewards
            .last_mut()
            .expect("score validated nonempty completion") = reward;
        Ok(rewards)
    }

    fn options(&self) -> serde_json::Value {
        let c = &self.head_config;
        serde_json::json!({"loss": "bradley_terry", "pooling": "last_response_token",
            "head": {"peak": c.schedule.peak, "min": c.schedule.min,
            "warmup": c.schedule.warmup_steps, "steps": c.schedule.total_steps,
            "beta1": c.beta_1, "beta2": c.beta_2, "epsilon": c.epsilon,
            "weight_decay": c.weight_decay, "max_grad_norm": c.max_grad_norm}})
    }

    /// Actor, scalar head, both AdamW states and caller cursor commit together.
    /// Retention is controlled by the actor's existing TrainConfig.
    pub fn save_checkpoint(&mut self) -> Result<PathBuf, TrainError> {
        let meta = metadata(&self.actor, "reward", "", self.options(), &self.state);
        let head = &self.head;
        let optimizer = &self.head_optimizer;
        self.actor.save_checkpoint_with(|path| {
            save_trainable(head, &path.join("reward-head.safetensors")).map_err(error)?;
            Recorder::<B>::record(
                &BinFileRecorder::<FullPrecisionSettings>::new(),
                optimizer.to_record(),
                path.join("reward-head-optimizer"),
            )
            .map_err(|e| error(format!("reward head optimizer record: {e:?}")))?;
            write_metadata(path, &meta)
        })
    }

    pub fn load_checkpoint(&mut self, path: &Path) -> Result<usize, TrainError> {
        let state = read_metadata(
            path,
            &metadata(&self.actor, "reward", "", self.options(), &self.state),
        )?;
        let head = load_trainable(
            &self.head,
            &path.join("reward-head.safetensors"),
            self.actor.device(),
        )
        .map_err(error)?;
        let record = Recorder::<B>::load(
            &BinFileRecorder::<FullPrecisionSettings>::new(),
            path.join("reward-head-optimizer"),
            self.actor.device(),
        )
        .map_err(|e| error(format!("reward head optimizer load: {e:?}")))?;
        let step = self.actor.load_checkpoint(path)?;
        self.head = head;
        self.head_optimizer = self.head_optimizer.clone().load_record(record);
        self.state = state;
        Ok(step)
    }
}
