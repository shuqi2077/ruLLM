use super::*;
use crate::training::{NativeSftModel, TrainError};
use crate::training::native::{Architecture, ProjectionTarget};
use super::super::training::{feed_forward, projection};

impl<B: Backend> KimiLinearModel<B> {
    pub fn into_training(self) -> Result<NativeSftModel<B>, TrainError> {
        if self.reduction.is_some() { return Err(TrainError("convert an unsharded Kimi Linear model for SFT".into())); }
        let mut model = NativeSftModel::new(Architecture::KimiLinear(self.config.clone()), "model".into());
        model.add_matrix("model.embed_tokens.weight".into(), self.embedding, ProjectionTarget::None);
        for (index, layer) in self.layers.into_iter().enumerate() {
            let p = format!("model.layers.{index}");
            let a = format!("{p}.self_attn");
            model.add_vector(format!("{p}.input_layernorm.weight"), layer.input_norm);
            model.add_vector(format!("{p}.post_attention_layernorm.weight"), layer.post_norm);
            match layer.mixer {
                Mixer::Mla(value) => {
                    projection(&mut model, &format!("{a}.q_proj"), value.q, ProjectionTarget::AttentionInput)?;
                    projection(&mut model, &format!("{a}.kv_a_proj_with_mqa"), value.kv_a, ProjectionTarget::AttentionInput)?;
                    projection(&mut model, &format!("{a}.kv_b_proj"), value.kv_b, ProjectionTarget::AttentionInput)?;
                    projection(&mut model, &format!("{a}.o_proj"), value.out, ProjectionTarget::AttentionOutput)?;
                    model.add_vector(format!("{a}.kv_a_layernorm.weight"), value.kv_norm);
                }
                Mixer::Kda(value) => {
                    for (name, value) in [("q_proj", value.q), ("k_proj", value.k), ("v_proj", value.v),
                        ("f_a_proj", value.f_a), ("f_b_proj", value.f_b), ("b_proj", value.beta), ("g_a_proj", value.g_a), ("g_b_proj", value.g_b)] {
                        projection(&mut model, &format!("{a}.{name}"), value, ProjectionTarget::AttentionInput)?;
                    }
                    projection(&mut model, &format!("{a}.o_proj"), value.out, ProjectionTarget::AttentionOutput)?;
                    for (name, value) in [("q_conv1d", value.q_conv), ("k_conv1d", value.k_conv), ("v_conv1d", value.v_conv)] {
                        model.add_convolution(format!("{a}.{name}.weight"), value);
                    }
                    model.add_state(format!("{a}.A_log"), value.a_log);
                    model.add_vector(format!("{a}.dt_bias"), value.dt_bias);
                    model.add_vector(format!("{a}.o_norm.weight"), value.output_norm);
                }
            }
            let moe = self.config.is_moe(index);
            feed_forward(&mut model, &format!("{p}.{}", if moe { "block_sparse_moe" } else { "mlp" }),
                layer.mlp, moe, self.config.num_experts.unwrap_or(0))?;
        }
        model.add_vector("model.norm.weight".into(), self.norm);
        if !self.config.tie_word_embeddings { projection(&mut model, "lm_head", self.head, ProjectionTarget::None)?; }
        Ok(model)
    }
}
