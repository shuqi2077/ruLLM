use super::*;
use crate::training::{NativeSftModel, TrainError};
use crate::training::native::{Architecture, ProjectionTarget};

pub(super) fn projection<B: Backend>(model: &mut NativeSftModel<B>, prefix: &str, value: Projection<B>, target: ProjectionTarget) -> Result<(), TrainError> {
    let Projection::Dense { weight, bias } = value else {
        return Err(TrainError(format!("{prefix}: load floating MLA/Kimi weights for SFT; packed inference closures have no backward")));
    };
    model.add_matrix(format!("{prefix}.weight"), weight, target);
    if let Some(bias) = bias { model.add_vector(format!("{prefix}.bias"), bias); }
    Ok(())
}

pub(super) fn mlp<B: Backend>(model: &mut NativeSftModel<B>, prefix: &str, value: Mlp<B>, kimi: bool) -> Result<(), TrainError> {
    let (gate, up, down) = if kimi { ("w1", "w3", "w2") } else { ("gate_proj", "up_proj", "down_proj") };
    projection(model, &format!("{prefix}.{gate}"), value.gate, ProjectionTarget::FeedForwardInput)?;
    projection(model, &format!("{prefix}.{up}"), value.up, ProjectionTarget::FeedForwardInput)?;
    projection(model, &format!("{prefix}.{down}"), value.down, ProjectionTarget::FeedForwardOutput)
}

pub(super) fn feed_forward<B: Backend>(model: &mut NativeSftModel<B>, prefix: &str, value: FeedForward<B>, kimi: bool, expected_experts: usize) -> Result<(), TrainError> {
    match value {
        FeedForward::Dense(value) => mlp(model, prefix, value, false),
        FeedForward::Experts { router, correction, experts, shared } => {
            if experts.len() != expected_experts || experts.iter().enumerate().any(|(index, (id, _))| index != *id) {
                return Err(TrainError("SFT needs the complete expert set, not an inference EP shard".into()));
            }
            model.add_matrix(format!("{prefix}.gate.weight"), router, ProjectionTarget::None);
            if let Some(correction) = correction {
                model.add_constant(format!("{prefix}.gate.e_score_correction_bias"), correction);
            }
            for (expert, value) in experts { mlp(model, &format!("{prefix}.experts.{expert}"), value, kimi)?; }
            if let Some(shared) = shared { mlp(model, &format!("{prefix}.shared_experts"), shared, false)?; }
            Ok(())
        }
    }
}

impl<B: Backend> DeepseekV2Model<B> {
    pub fn into_training(self) -> Result<NativeSftModel<B>, TrainError> {
        if self.reduction.is_some() { return Err(TrainError("convert an unsharded MLA model for SFT".into())); }
        let mut model = NativeSftModel::new(Architecture::Mla(self.config.clone()), "model".into());
        model.add_matrix("model.embed_tokens.weight".into(), self.embedding, ProjectionTarget::None);
        for (index, layer) in self.layers.into_iter().enumerate() {
            let p = format!("model.layers.{index}");
            let a = format!("{p}.self_attn");
            let compressed_query = layer.q_a.is_some();
            if let Some((value, norm)) = layer.q_a {
                projection(&mut model, &format!("{a}.q_a_proj"), value, ProjectionTarget::AttentionInput)?;
                model.add_vector(format!("{a}.q_a_layernorm.weight"), norm);
            }
            projection(&mut model, &format!("{a}.{}", if compressed_query { "q_b_proj" } else { "q_proj" }), layer.q, ProjectionTarget::AttentionInput)?;
            projection(&mut model, &format!("{a}.kv_a_proj_with_mqa"), layer.kv_a, ProjectionTarget::AttentionInput)?;
            projection(&mut model, &format!("{a}.kv_b_proj"), layer.kv_b, ProjectionTarget::AttentionInput)?;
            projection(&mut model, &format!("{a}.o_proj"), layer.o, ProjectionTarget::AttentionOutput)?;
            model.add_vector(format!("{a}.kv_a_layernorm.weight"), layer.kv_norm);
            model.add_vector(format!("{p}.input_layernorm.weight"), layer.input_norm);
            model.add_vector(format!("{p}.post_attention_layernorm.weight"), layer.post_norm);
            feed_forward(&mut model, &format!("{p}.mlp"), layer.mlp, false, self.config.n_routed_experts)?;
        }
        model.add_vector("model.norm.weight".into(), self.norm);
        if !self.config.tie_word_embeddings { projection(&mut model, "lm_head", self.head, ProjectionTarget::None)?; }
        Ok(model)
    }
}

pub(crate) fn route<B: Backend>(logits: Tensor<B, 2>, correction: Option<&Tensor<B, 1>>, c: &DeepseekV2Config)
    -> (Tensor<B, 2>, Tensor<B, 2, Int>) {
    routing::route(logits, correction, c)
}

pub(crate) fn rotary<B: Backend>(c: &DeepseekV2Config, sequence: usize, dtype: DType, device: &B::Device)
    -> (Tensor<B, 4>, Tensor<B, 4>, f64) {
    let width = c.qk_rope_head_dim;
    let mut inverse = (0..width / 2).map(|i| 1.0f32 / (c.rope_theta as f32).powf((2 * i) as f32 / width as f32)).collect::<Vec<_>>();
    let mut amplitude = 1.0;
    let mut scale = 1.0 / ((c.qk_nope_head_dim + width) as f64).sqrt();
    if let Some(yarn) = &c.rope_scaling {
        let correction = |rotations: f64| width as f64
            * (yarn.original_max_position_embeddings as f64 / (rotations * 2.0 * std::f64::consts::PI)).ln() / (2.0 * c.rope_theta.ln());
        let low = correction(yarn.beta_fast).floor().max(0.0);
        let high = correction(yarn.beta_slow).ceil().min((width - 1) as f64);
        let high = if high == low { high + 0.001 } else { high };
        for (i, frequency) in inverse.iter_mut().enumerate() {
            let ramp = ((i as f32 - low as f32) / (high - low) as f32).clamp(0.0, 1.0);
            *frequency = (*frequency / yarn.factor as f32) * ramp + *frequency * (1.0 - ramp);
        }
        amplitude = mscale(yarn.factor, yarn.mscale) / mscale(yarn.factor, yarn.mscale_all_dim);
        scale *= mscale(yarn.factor, yarn.mscale_all_dim).powi(2);
    }
    let positions = Tensor::<B, 2>::from_data(TensorData::new((0..sequence).map(|p| p as f32).collect::<Vec<_>>(), [sequence, 1]), (device, DType::F32));
    let frequencies = positions * Tensor::<B, 2>::from_data(TensorData::new(inverse, [1, width / 2]), (device, DType::F32));
    let frequencies = Tensor::cat(vec![frequencies.clone(), frequencies], 1).reshape([1, 1, sequence, width]);
    ((frequencies.clone().cos() * amplitude).cast(dtype), (frequencies.sin() * amplitude).cast(dtype), scale)
}
