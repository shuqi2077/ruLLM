use super::*;
use crate::training::{NativeSftModel, TrainError};
use crate::training::native::{Architecture, ProjectionTarget};

fn projection<B: Backend>(model: &mut NativeSftModel<B>, prefix: &str, value: Projection<B>, target: ProjectionTarget) {
    model.add_matrix(format!("{prefix}.weight"), value.weight, target);
    if let Some(bias) = value.bias { model.add_vector(format!("{prefix}.bias"), bias); }
}

fn mlp<B: Backend>(model: &mut NativeSftModel<B>, prefix: &str, value: moe::Mlp<B>) {
    projection(model, &format!("{prefix}.gate_proj"), value.gate, ProjectionTarget::FeedForwardInput);
    projection(model, &format!("{prefix}.up_proj"), value.up, ProjectionTarget::FeedForwardInput);
    projection(model, &format!("{prefix}.down_proj"), value.down, ProjectionTarget::FeedForwardOutput);
}

impl<B: Backend> Qwen3Model<B> {
    /// Register the existing dense/MoE weights for full fine-tuning or LoRA.
    /// For an inference backend, use `Module::train` on the returned module.
    pub fn into_training(self) -> Result<NativeSftModel<B>, TrainError> {
        if self.reduction.is_some() {
            return Err(TrainError("convert the unsharded Qwen3 model for SFT; inference TP closures have no backward".into()));
        }
        let mut model = NativeSftModel::new(Architecture::Qwen3(self.config.clone(), self.moe_config.clone()), "model".into());
        model.add_matrix("model.embed_tokens.weight".into(), self.embedding, ProjectionTarget::None);
        for (index, layer) in self.layers.into_iter().enumerate() {
            let p = format!("model.layers.{index}");
            let a = format!("{p}.self_attn");
            projection(&mut model, &format!("{a}.q_proj"), layer.q, ProjectionTarget::AttentionInput);
            projection(&mut model, &format!("{a}.k_proj"), layer.k, ProjectionTarget::AttentionInput);
            projection(&mut model, &format!("{a}.v_proj"), layer.v, ProjectionTarget::AttentionInput);
            projection(&mut model, &format!("{a}.o_proj"), layer.o, ProjectionTarget::AttentionOutput);
            model.add_vector(format!("{a}.q_norm.weight"), layer.q_norm);
            model.add_vector(format!("{a}.k_norm.weight"), layer.k_norm);
            model.add_vector(format!("{p}.input_layernorm.weight"), layer.input_norm);
            model.add_vector(format!("{p}.post_attention_layernorm.weight"), layer.post_norm);
            match layer.mlp {
                moe::FeedForward::Dense(value) => mlp(&mut model, &format!("{p}.mlp"), value),
                moe::FeedForward::Experts { router, experts, .. } => {
                    projection(&mut model, &format!("{p}.mlp.gate"), router, ProjectionTarget::None);
                    for (expert, value) in experts.into_iter().enumerate() {
                        mlp(&mut model, &format!("{p}.mlp.experts.{expert}"), value);
                    }
                }
            }
        }
        model.add_vector("model.norm.weight".into(), self.norm);
        if !self.config.tie_word_embeddings { projection(&mut model, "lm_head", self.head, ProjectionTarget::None); }
        Ok(model)
    }
}
