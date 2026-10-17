use super::*;
use ruda_nn::attention::PackedSequenceLayout;

pub(super) struct ResponseLayout {
    pub tokens: Vec<i32>,
    pub targets: Vec<i32>,
    pub layout: PackedSequenceLayout,
    pub actions: Vec<core::ops::Range<usize>>,
    pub append_contexts: Option<Vec<usize>>,
}

impl ResponseLayout {
    pub fn new(responses: &[&Response], include_last: bool) -> Self {
        let mut tokens = Vec::new();
        let mut targets = Vec::new();
        let mut boundaries = vec![0];
        let mut actions = Vec::new();
        let mut contexts = Vec::new();
        for response in responses {
            let offset = tokens.len();
            tokens.extend(&response.prompt);
            tokens.extend(&response.completion);
            if !include_last { tokens.pop(); }
            let length = tokens.len() - offset;
            match &response.rotary_context {
                Some(history) => contexts.extend_from_slice(&history[..length]),
                None => contexts.extend(core::iter::repeat_n(length, length)),
            }
            let start = offset + response.prompt.len() - 1;
            let end = start + response.completion.len();
            targets.resize(start, -100);
            targets.extend(&response.completion);
            targets.resize(tokens.len(), -100);
            actions.push(start..end + usize::from(include_last));
            boundaries.push(tokens.len());
        }
        let layout = PackedSequenceLayout::new(boundaries, tokens.len());
        let append_contexts = responses.iter().any(|r| r.rotary_context.is_some()).then_some(contexts);
        Self { tokens, targets, layout, actions, append_contexts }
    }

    pub fn inputs<B: Backend>(&self, device: &B::Device) -> Tensor<B, 1, Int> {
        Tensor::from_data(TensorData::new(self.tokens.clone(), [self.tokens.len()]), device)
    }
}

/// Each vector contains only the actions of one response, in input order.
pub(super) fn probabilities_many<B: TrainingBackend + LinearLogProbsBackend>(
    model: &PackedLlamaForCausalLm<B>, adapters: Option<&LlamaLora<B>>,
    responses: &[&Response], device: &B::Device, chunk: usize,
) -> Vec<Tensor<B, 1>> {
    if model.paged_rotary_states().iter().any(Option::is_some) && responses.iter().any(|r| r.rotary_context.is_none()) {
        return prefix_statistics(model, adapters, responses, device, chunk);
    }
    let batch = ResponseLayout::new(responses, false);
    let logp = model.train_packed_log_probs_with_contexts(batch.inputs(device),
        Tensor::from_data(TensorData::new(batch.targets.clone(), [batch.targets.len()]), device),
        &batch.layout, adapters, chunk, batch.append_contexts.as_deref());
    batch.actions.into_iter().map(|range| logp.clone().slice([range])).collect()
}

pub(super) fn critic_many<B: TrainingBackend>(critic: &ValueCritic<B>, responses: &[&Response],
    device: &B::Device) -> Vec<Tensor<B, 1>> {
    // The critic does not need an LM projection or LinearLogProbsBackend.
    if critic.backbone.paged_rotary_states().iter().any(Option::is_some) && responses.iter().any(|r| r.rotary_context.is_none()) {
        let mut selected: Vec<Vec<Tensor<B, 1>>> = responses.iter().map(|_| Vec::new()).collect();
        let maximum = responses.iter().map(|r| r.completion.len()).max().unwrap_or(0);
        for step in 0..=maximum {
            let active: Vec<_> = responses.iter().enumerate().filter(|(_, r)| step <= r.completion.len()).collect();
            let mut tokens = Vec::new();
            let mut contexts = Vec::new();
            let mut boundaries = vec![0];
            for (_, response) in &active {
                tokens.extend(&response.prompt);
                tokens.extend(&response.completion[..step]);
                let length = response.prompt.len() + step;
                contexts.extend(response.rotary_context.as_ref().map(|v| v[..length].to_vec())
                    .unwrap_or_else(|| vec![length; length]));
                boundaries.push(tokens.len());
            }
            let count = tokens.len();
            let layout = PackedSequenceLayout::new(boundaries, count);
            let hidden = critic.backbone.forward_train_packed_hidden_with_contexts(
                Tensor::from_data(TensorData::new(tokens, [count]), device), &layout, None, Some(&contexts));
            let values = critic.head.forward(hidden).reshape([count]).cast(DType::F32);
            for ((index, _), range) in active.into_iter().zip(layout.boundaries().windows(2)) {
                selected[index].push(values.clone().slice([range[1]-1..range[1]]));
            }
        }
        return selected.into_iter().map(|parts| Tensor::cat(parts, 0)).collect();
    }
    let batch = ResponseLayout::new(responses, true);
    let hidden = critic.backbone.forward_train_packed_hidden_with_contexts(batch.inputs(device), &batch.layout, None,
        batch.append_contexts.as_deref());
    let values = critic.head.forward(hidden).reshape([batch.tokens.len()]).cast(DType::F32);
    batch.actions.into_iter().map(|range| values.clone().slice([range])).collect()
}

pub(super) fn behavior_many<B: TrainingBackend + LinearLogProbsBackend>(
    model: &PackedLlamaForCausalLm<B>, adapters: Option<&LlamaLora<B>>,
    responses: &[&Response], device: &B::Device, chunk: usize,
) -> Result<Vec<Vec<f32>>, TrainError> {
    let mut output: Vec<_> = responses.iter().map(|r| r.behavior_log_probs.clone()).collect();
    let missing: Vec<_> = responses.iter().enumerate().filter(|(_, r)| r.behavior_log_probs.is_none()).collect();
    if !missing.is_empty() {
        let refs: Vec<_> = missing.iter().map(|(_, r)| **r).collect();
        for ((index, _), logp) in missing.into_iter().zip(probabilities_many(model, adapters, &refs, device, chunk)) {
            output[index] = Some(values(logp)?);
        }
    }
    Ok(output.into_iter().map(|v| v.expect("captured or computed behavior statistics")).collect())
}

fn prefix_statistics<B: TrainingBackend + LinearLogProbsBackend>(
    model: &PackedLlamaForCausalLm<B>, adapters: Option<&LlamaLora<B>>,
    responses: &[&Response], device: &B::Device, chunk: usize,
) -> Vec<Tensor<B, 1>> {
    let mut selected: Vec<Vec<Tensor<B, 1>>> = responses.iter().map(|_| Vec::new()).collect();
    let maximum = responses.iter().map(|r| r.completion.len()).max().unwrap_or(0);
    for step in 0..maximum {
        let active: Vec<_> = responses.iter().enumerate().filter(|(_, r)| step < r.completion.len()).collect();
        let mut tokens = Vec::new();
        let mut targets = Vec::new();
        let mut contexts = Vec::new();
        let mut boundaries = vec![0];
        for (_, response) in &active {
            tokens.extend(&response.prompt);
            tokens.extend(&response.completion[..step]);
            let length = response.prompt.len() + step;
            contexts.extend(response.rotary_context.as_ref().map(|v| v[..length].to_vec())
                .unwrap_or_else(|| vec![length; length]));
            targets.resize(tokens.len()-1, -100);
            targets.push(response.completion[step]);
            boundaries.push(tokens.len());
        }
        let layout = PackedSequenceLayout::new(boundaries, tokens.len());
        let count = tokens.len();
        let logp = model.train_packed_log_probs_with_contexts(
            Tensor::from_data(TensorData::new(tokens, [count]), device),
            Tensor::from_data(TensorData::new(targets, [count]), device), &layout, adapters, chunk, Some(&contexts));
        for ((index, _), range) in active.into_iter().zip(layout.boundaries().windows(2)) {
            selected[index].push(logp.clone().slice([range[1]-1..range[1]]));
        }
    }
    selected.into_iter().map(|parts| Tensor::cat(parts, 0)).collect()
}

pub(in crate::training) fn reward_many<B: TrainingBackend>(model: &PackedLlamaForCausalLm<B>,
    adapters: Option<&LlamaLora<B>>, head: &Linear<B>, responses: &[&Response],
    device: &B::Device) -> Tensor<B, 1> {
    let batch = ResponseLayout::new(responses, true);
    let hidden = model.forward_train_packed_hidden_with_contexts(batch.inputs(device), &batch.layout, adapters,
        batch.append_contexts.as_deref());
    let all = head.forward(hidden).reshape([batch.tokens.len()]).cast(DType::F32);
    Tensor::cat(batch.layout.boundaries().windows(2)
        .map(|range| all.clone().slice([range[1]-1..range[1]])).collect(), 0)
}

impl<B: AutodiffBackend + TrainingBackend + LinearLogProbsBackend> PpoTrainer<B> {
    /// Snapshot an entire ragged response batch with isolated packed forwards.
    /// Each reward vector belongs to one response; GAE restarts per document.
    pub fn prepare_rollouts(&self, responses: Vec<Response>, rewards: &[Vec<f32>])
        -> Result<Vec<PpoRollout>, TrainError> {
        self.config.validate()?;
        if responses.is_empty() || responses.len() != rewards.len() {
            return Err(err("PPO needs a nonempty response batch and matching rewards"));
        }
        for (response, reward) in responses.iter().zip(rewards) {
            response.validate(self.actor.model())?;
            response.validate(&self.critic.backbone)?;
            response.validate(&self.reference.model)?;
            if reward.len() != response.completion.len() || reward.iter().any(|x| !x.is_finite()) {
                return Err(err("one finite reward per completion token is required"));
            }
        }
        let model = self.actor.model().clone().no_grad();
        let adapters = self.actor.adapters().cloned().map(Module::no_grad);
        let refs: Vec<_> = responses.iter().collect();
        let device = self.actor.device();
        let chunk = self.actor.loss_chunk_rows();
        let old = behavior_many(&model, adapters.as_ref(), &refs, device, chunk)?;
        let reference = probabilities_many(&self.reference.model, self.reference.adapters.as_ref(), &refs, device, chunk);
        let predicted = critic_many(&self.critic.clone().no_grad(), &refs, device);
        responses.into_iter().zip(rewards).zip(old.into_iter().zip(reference).zip(predicted))
            .map(|((response, rewards), ((old, reference), predicted))| {
                let reference = values(reference)?;
                let predicted = values(predicted)?;
                let n = rewards.len();
                let mut advantages = vec![0.0; n];
                let mut returns = vec![0.0; n];
                let mut gae = 0.0;
                for t in (0..n).rev() {
                    let next = if t + 1 == n && response.terminal { 0.0 } else { predicted[t + 1] };
                    let reward = rewards[t] - self.config.kl_coefficient * (old[t] - reference[t]);
                    let delta = reward + self.config.gamma * next - predicted[t];
                    gae = delta + self.config.gamma * self.config.gae_lambda * gae;
                    advantages[t] = gae;
                    returns[t] = gae + predicted[t];
                }
                Ok(PpoRollout { response, old_log_probs: old, old_values: predicted[..n].to_vec(), advantages, returns })
            }).collect()
    }
}
