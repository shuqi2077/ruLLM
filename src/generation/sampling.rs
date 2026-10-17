use super::{
    CausalModel, CausalModelLimits, RudaPackedModel, GenerationError, GreedyGenerationConfig,
    TokenGenerationOutput, ControlledGenerationOutput, GenerationControl, GenerationEvent,
    generate_with_selector_stream, read_last_logits,
};
use crate::{LlamaConfig, LlamaForCausalLm, PackedLlamaForCausalLm};
use ruda::runtime::server::ComputeServer;
use chacha20::ChaCha12Rng;
use ruda_core::rand::{RngExt, SeedableRng, get_seeded_rng};
use ruda_tensor::DeviceOps;
use ruda_tensor::api::backend::Backend;
use std::ops::ControlFlow;
use ruda_tensor_device::{BoolElement, DeviceBackend, DeviceRuntime, FloatElement, IntElement};

mod distribution;
use distribution::SamplingWorkspace;
mod state;
pub use state::TokenSamplerState;

/// Temperature, top-k, then nucleus filtering for categorical token sampling.
#[derive(Debug, Clone, Copy, PartialEq, serde::Serialize, serde::Deserialize)]
pub struct SamplingConfig {
    /// Finite and strictly positive. Use the greedy API for deterministic argmax.
    pub temperature: f64,
    /// Zero disables top-k. Tokens tied at the kth logit are retained.
    pub top_k: usize,
    /// Probability mass to retain, in (0, 1]. One disables nucleus filtering.
    pub top_p: f64,
    /// Request-local random seed; None initializes from system entropy.
    pub seed: Option<u64>,
}

#[derive(Debug, Clone, Copy, PartialEq, serde::Serialize, serde::Deserialize)]
#[serde(default)]
pub struct SamplingOptions {
    /// Drop tokens whose probability is below `min_p` times the most likely
    /// token's probability, in [0, 1]. Zero disables it.
    pub min_p: f64,
    /// Hugging Face repetition penalty over prompt and generated tokens:
    /// positive logits are divided by it, negative ones multiplied. One
    /// disables it; must be finite and positive.
    pub repetition_penalty: f64,
    /// OpenAI presence penalty, subtracted once from every token that already
    /// appears in the generated output, in [-2, 2].
    pub presence_penalty: f64,
    /// OpenAI frequency penalty, subtracted per occurrence in the generated
    /// output, in [-2, 2].
    pub frequency_penalty: f64,
}

impl Default for SamplingConfig {
    fn default() -> Self {
        Self {
            temperature: 1.0,
            top_k: 0,
            top_p: 1.0,
            seed: None,
        }
    }
}

impl Default for SamplingOptions {
    fn default() -> Self {
        Self {
            min_p: 0.0,
            repetition_penalty: 1.0,
            presence_penalty: 0.0,
            frequency_penalty: 0.0,
        }
    }
}

impl SamplingConfig {
    pub fn validate(&self) -> Result<(), GenerationError> {
        if !self.temperature.is_finite() || self.temperature <= 0.0 {
            return Err(GenerationError(
                "sampling temperature must be finite and positive".into(),
            ));
        }
        if !self.top_p.is_finite() || self.top_p <= 0.0 || self.top_p > 1.0 {
            return Err(GenerationError("sampling top_p must be in (0, 1]".into()));
        }
        Ok(())
    }
}

impl SamplingOptions {
    pub fn validate(&self) -> Result<(), GenerationError> {
        if !self.min_p.is_finite() || !(0.0..=1.0).contains(&self.min_p) {
            return Err(GenerationError("sampling min_p must be in [0, 1]".into()));
        }
        if !self.repetition_penalty.is_finite() || self.repetition_penalty <= 0.0 {
            return Err(GenerationError(
                "repetition_penalty must be finite and positive".into(),
            ));
        }
        for (name, value) in [
            ("presence_penalty", self.presence_penalty),
            ("frequency_penalty", self.frequency_penalty),
        ] {
            if !value.is_finite() || !(-2.0..=2.0).contains(&value) {
                return Err(GenerationError(format!("{name} must be in [-2, 2]")));
            }
        }
        Ok(())
    }

    fn has_penalties(&self) -> bool {
        self.repetition_penalty != 1.0
            || self.presence_penalty != 0.0
            || self.frequency_penalty != 0.0
    }
}

/// Apply repetition (prompt + generated) and presence/frequency (generated)
/// penalties to a copy of `logits`.
fn penalized_logits(
    logits: &[f32],
    config: &SamplingOptions,
    prompt: &[i32],
    generated: &[i32],
) -> Vec<f32> {
    let mut output = logits.to_vec();
    let in_vocabulary = |token: i32| (token >= 0 && (token as usize) < logits.len()).then_some(token as usize);
    if config.repetition_penalty != 1.0 {
        let mut seen = vec![false; logits.len()];
        for index in prompt.iter().chain(generated).copied().filter_map(in_vocabulary) {
            seen[index] = true;
        }
        let penalty = config.repetition_penalty as f32;
        for (value, seen) in output.iter_mut().zip(seen) {
            if seen && value.is_finite() {
                *value = if *value > 0.0 { *value / penalty } else { *value * penalty };
            }
        }
    }
    if config.presence_penalty != 0.0 || config.frequency_penalty != 0.0 {
        let mut counts = std::collections::BTreeMap::<usize, u32>::new();
        for index in generated.iter().copied().filter_map(in_vocabulary) {
            *counts.entry(index).or_default() += 1;
        }
        for (index, count) in counts {
            if output[index].is_finite() {
                output[index] -= (config.presence_penalty
                    + config.frequency_penalty * f64::from(count)) as f32;
            }
        }
    }
    output
}

#[derive(Debug, Clone, PartialEq)]
pub struct SamplingGenerationConfig {
    pub max_new_tokens: usize,
    pub eos_token_ids: Vec<i32>,
    pub sampling: SamplingConfig,
}

impl Default for SamplingGenerationConfig {
    fn default() -> Self {
        Self {
            max_new_tokens: 32,
            eos_token_ids: Vec::new(),
            sampling: SamplingConfig::default(),
        }
    }
}

/// A request-local sampler. Cloning preserves its exact current RNG state.
#[derive(Debug)]
pub struct TokenSampler {
    config: SamplingConfig,
    options: SamplingOptions,
    rng: ChaCha12Rng,
    workspace: SamplingWorkspace,
}

impl Clone for TokenSampler {
    fn clone(&self) -> Self {
        Self {
            config: self.config,
            options: self.options,
            rng: ChaCha12Rng::deserialize_state(&self.rng.serialize_state()),
            // Scratch buffers are not semantic state. Avoid copying a full
            // vocabulary when the scheduler previews a request's next draw.
            workspace: SamplingWorkspace::default(),
        }
    }
}

impl TokenSampler {
    pub fn new(config: SamplingConfig) -> Result<Self, GenerationError> {
        Self::new_with_options(config, SamplingOptions::default())
    }

    pub fn new_with_options(config: SamplingConfig, options: SamplingOptions) -> Result<Self, GenerationError> {
        config.validate()?;
        options.validate()?;
        let rng = config
            .seed
            .map(ChaCha12Rng::seed_from_u64)
            .unwrap_or_else(|| ChaCha12Rng::from_rng(&mut get_seeded_rng()));
        Ok(Self { config, options, rng, workspace: SamplingWorkspace::default() })
    }

    /// Release reusable vocabulary buffers without changing configuration or
    /// RNG state. Useful when retaining many idle samplers between requests.
    pub fn clear_workspace(&mut self) {
        self.workspace = SamplingWorkspace::default();
    }

    pub fn propose(&mut self, logits: &[f32]) -> Result<(i32, Vec<f64>), GenerationError> {
        let probabilities = self.workspace.fill_with_min_p(logits, self.config, self.options.min_p)?.to_vec();
        let token = super::speculative::draw(&probabilities, self.rng.random())?;
        Ok((token, probabilities))
    }

    /// Rejection-corrected selection from the processed target distribution.
    /// q is the exact distribution used to draw candidate, not new draft logits.
    pub fn verify_draft(&mut self, candidate: i32, q: &[f64], logits: &[f32])
        -> Result<(i32, bool), GenerationError> {
        if candidate < 0 || candidate as usize >= q.len() || q.len() != logits.len()
            || q.iter().any(|x| !x.is_finite() || *x < 0.0) || q[candidate as usize] <= 0.0 {
            return Err(GenerationError("invalid draft candidate distribution".into()));
        }
        let p = self.workspace.fill_with_min_p(logits, self.config, self.options.min_p)?;
        if self.rng.random::<f64>() < (p[candidate as usize] / q[candidate as usize]).min(1.0) {
            return Ok((candidate, true));
        }
        let residual = p.iter().zip(q).map(|(p, q)| (p - q).max(0.0)).collect::<Vec<_>>();
        Ok((super::speculative::draw(&residual, self.rng.random())?, false))
    }

    /// Sample one vocabulary index. Negative infinity masks a token; NaN,
    /// positive infinity and a fully masked vocabulary return an error.
    pub fn sample(&mut self, logits: &[f32]) -> Result<i32, GenerationError> {
        self.sample_with_history(logits, &[], &[])
    }

    /// Sample with the configured penalties applied against `prompt` and the
    /// tokens `generated` so far. Without penalties this equals [`Self::sample`].
    pub fn sample_with_history(
        &mut self,
        logits: &[f32],
        prompt: &[i32],
        generated: &[i32],
    ) -> Result<i32, GenerationError> {
        let penalized;
        let values = if self.options.has_penalties() {
            penalized = penalized_logits(logits, &self.options, prompt, generated);
            penalized.as_slice()
        } else {
            logits
        };
        let probabilities = self.workspace.fill_with_min_p(values, self.config, self.options.min_p)?;
        let draw = self.rng.random::<f64>();
        let mut cumulative = 0.0;
        let mut last = None;
        for (token, probability) in probabilities.iter().copied().enumerate() {
            if probability == 0.0 {
                continue;
            }
            last = Some(token);
            cumulative += probability;
            if draw < cumulative {
                return Ok(token as i32);
            }
        }
        // Summation rounding can leave the normalized CDF just below one.
        Ok(last.expect("validated nonempty sampling distribution") as i32)
    }
}

pub(super) fn filtered_probabilities(
    logits: &[f32],
    config: SamplingConfig,
) -> Result<Vec<f64>, GenerationError> {
    Ok(SamplingWorkspace::default().fill(logits, config)?.to_vec())
}

pub fn generate_sampled<B: Backend>(
    model: &LlamaForCausalLm<B>,
    model_config: &LlamaConfig,
    prompt_token_ids: &[i32],
    generation: &SamplingGenerationConfig,
    device: &B::Device,
) -> Result<TokenGenerationOutput, GenerationError> {
    generate_sampled_impl(model, model_config, prompt_token_ids, generation, device)
}

pub fn generate_sampled_packed<B: Backend>(
    model: &PackedLlamaForCausalLm<B>,
    model_config: &LlamaConfig,
    prompt_token_ids: &[i32],
    generation: &SamplingGenerationConfig,
    device: &B::Device,
) -> Result<TokenGenerationOutput, GenerationError> {
    generate_sampled_impl(model, model_config, prompt_token_ids, generation, device)
}

pub fn generate_sampled_packed_ruda<R, F, I, BT>(
    model: &PackedLlamaForCausalLm<DeviceBackend<R, F, I, BT>>,
    model_config: &LlamaConfig,
    prompt_token_ids: &[i32],
    generation: &SamplingGenerationConfig,
    device: &R::Device,
) -> Result<TokenGenerationOutput, GenerationError>
where
    R: DeviceRuntime,
    R::Server: ComputeServer,
    R::Device: DeviceOps,
    F: FloatElement,
    I: IntElement,
    BT: BoolElement,
{
    generate_sampled_impl(
        &RudaPackedModel(model),
        model_config,
        prompt_token_ids,
        generation,
        device,
    )
}

pub(super) fn generate_sampled_impl<B: Backend, M: CausalModel<B>>(
    model: &M,
    model_config: &LlamaConfig,
    prompt_token_ids: &[i32],
    generation: &SamplingGenerationConfig,
    device: &B::Device,
) -> Result<TokenGenerationOutput, GenerationError> {
    generate_causal_sampled(model, &model_config.into(), prompt_token_ids, generation, device)
}

pub fn generate_causal_sampled<B: Backend, M: CausalModel<B>>(
    model: &M,
    model_limits: &CausalModelLimits,
    prompt_token_ids: &[i32],
    generation: &SamplingGenerationConfig,
    device: &B::Device,
) -> Result<TokenGenerationOutput, GenerationError> {
    Ok(generate_causal_sampled_stream(
        model, model_limits, prompt_token_ids, generation,
        &GenerationControl::default(), device, |_| ControlFlow::Continue(()),
    )?.output)
}

/// Sample with the same filtering/RNG semantics as `generate_causal_sampled`,
/// while observing token callbacks, stop sequences and cooperative cancellation.
pub fn generate_causal_sampled_stream<B: Backend, M: CausalModel<B>>(
    model: &M,
    model_limits: &CausalModelLimits,
    prompt_token_ids: &[i32],
    generation: &SamplingGenerationConfig,
    control: &GenerationControl,
    device: &B::Device,
    on_token: impl FnMut(GenerationEvent<'_>) -> ControlFlow<()>,
) -> Result<ControlledGenerationOutput, GenerationError> {
    generate_causal_sampled_stream_with_options(model, model_limits, prompt_token_ids, generation,
        SamplingOptions::default(), control, device, on_token)
}

pub fn generate_causal_sampled_stream_with_options<B: Backend, M: CausalModel<B>>(
    model: &M,
    model_limits: &CausalModelLimits,
    prompt_token_ids: &[i32],
    generation: &SamplingGenerationConfig,
    options: SamplingOptions,
    control: &GenerationControl,
    device: &B::Device,
    on_token: impl FnMut(GenerationEvent<'_>) -> ControlFlow<()>,
) -> Result<ControlledGenerationOutput, GenerationError> {
    let mut sampler = TokenSampler::new_with_options(generation.sampling, options)?;
    let limits = GreedyGenerationConfig {
        max_new_tokens: generation.max_new_tokens,
        eos_token_ids: generation.eos_token_ids.clone(),
    };
    let mut generated = Vec::new();
    generate_with_selector_stream(
        model, model_limits, prompt_token_ids, &limits, control, device,
        |_, logits| {
            let row = read_last_logits(logits)?;
            let token = sampler.sample_with_history(&row, prompt_token_ids, &generated)?;
            generated.push(token);
            Ok(token)
        },
        on_token,
    )
}

#[cfg(test)]
mod tests;
