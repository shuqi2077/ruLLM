use super::*;
use crate::training::{NativeSftModel, TrainError};
use crate::training::native::{Architecture, ProjectionTarget};

fn projection<B: Backend>(model: &mut NativeSftModel<B>, prefix: &str, value: Projection<B>, target: ProjectionTarget) -> Result<(), TrainError> {
    let Projection::Dense(value) = value else {
        return Err(TrainError(format!("{prefix}: SFT requires floating weights, not an inference-only packed projection")));
    };
    model.add_matrix(format!("{prefix}.weight"), value.weight.val().transpose(), target);
    if let Some(bias) = value.bias { model.add_vector(format!("{prefix}.bias"), bias.val()); }
    Ok(())
}

fn mlp<B: Backend>(model: &mut NativeSftModel<B>, prefix: &str, value: Mlp<B>) -> Result<(), TrainError> {
    projection(model, &format!("{prefix}.gate_proj"), value.gate, ProjectionTarget::FeedForwardInput)?;
    projection(model, &format!("{prefix}.up_proj"), value.up, ProjectionTarget::FeedForwardInput)?;
    projection(model, &format!("{prefix}.down_proj"), value.down, ProjectionTarget::FeedForwardOutput)
}

impl<B: Backend> Qwen35TextModel<B> {
    /// Preserve full-attention/linear-attention layer order and routed/shared
    /// experts, replacing inference-only kernels with differentiable forwards.
    pub fn into_training(self) -> Result<NativeSftModel<B>, TrainError> {
        if self.reduction.is_some() {
            return Err(TrainError("convert the unsharded Qwen3.5 text model for SFT".into()));
        }
        let base = self.weight_prefix;
        let mut model = NativeSftModel::new(Architecture::Qwen35(self.config.clone()), base.clone());
        model.add_matrix(format!("{base}.embed_tokens.weight"), self.embedding.weight.val(), ProjectionTarget::None);
        for (index, layer) in self.layers.into_iter().enumerate() {
            let p = format!("{base}.layers.{index}");
            model.add_vector(format!("{p}.input_layernorm.weight"), layer.input_norm.weight);
            model.add_vector(format!("{p}.post_attention_layernorm.weight"), layer.post_norm.weight);
            match layer.mixer {
                Mixer::Full(value) => {
                    let a = format!("{p}.self_attn");
                    projection(&mut model, &format!("{a}.q_proj"), value.q, ProjectionTarget::AttentionInput)?;
                    projection(&mut model, &format!("{a}.k_proj"), value.k, ProjectionTarget::AttentionInput)?;
                    projection(&mut model, &format!("{a}.v_proj"), value.v, ProjectionTarget::AttentionInput)?;
                    projection(&mut model, &format!("{a}.o_proj"), value.out, ProjectionTarget::AttentionOutput)?;
                    model.add_vector(format!("{a}.q_norm.weight"), value.q_norm.weight);
                    model.add_vector(format!("{a}.k_norm.weight"), value.k_norm.weight);
                }
                Mixer::Delta(value) => {
                    let a = format!("{p}.linear_attn");
                    for (name, value) in [("in_proj_qkv", value.qkv), ("in_proj_z", value.z), ("in_proj_a", value.a), ("in_proj_b", value.b)] {
                        projection(&mut model, &format!("{a}.{name}"), value, ProjectionTarget::AttentionInput)?;
                    }
                    projection(&mut model, &format!("{a}.out_proj"), value.out, ProjectionTarget::AttentionOutput)?;
                    model.add_convolution(format!("{a}.conv1d.weight"), value.conv);
                    model.add_vector(format!("{a}.A_log"), value.a_log);
                    model.add_vector(format!("{a}.dt_bias"), value.dt_bias);
                    model.add_vector(format!("{a}.norm.weight"), value.norm);
                }
            }
            match layer.mlp {
                FeedForward::Dense(value) => mlp(&mut model, &format!("{p}.mlp"), value)?,
                FeedForward::Experts { router, experts, shared, shared_gate, .. } => {
                    projection(&mut model, &format!("{p}.mlp.gate"), router, ProjectionTarget::None)?;
                    projection(&mut model, &format!("{p}.mlp.shared_expert_gate"), shared_gate, ProjectionTarget::None)?;
                    mlp(&mut model, &format!("{p}.mlp.shared_expert"), shared)?;
                    for (expert, value) in experts.into_iter().enumerate() {
                        mlp(&mut model, &format!("{p}.mlp.experts.{expert}"), value)?;
                    }
                }
            }
        }
        model.add_vector(format!("{base}.norm.weight"), self.norm.weight);
        if !self.config.tie_word_embeddings { projection(&mut model, "lm_head", self.head, ProjectionTarget::None)?; }
        Ok(model)
    }
}
