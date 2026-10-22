use super::*;
use ruda_tensor::api::{activation::sigmoid, module::attention, ops::AttentionModuleOptions};

impl<B: TrainingBackend> NativeSftModel<B> {
    pub(super) fn qwen_attention(&self, input: Tensor<B, 3>, prefix: &str, c: &Qwen3Config,
        layer: usize, adapters: Option<&NativeLora<B>>) -> Tensor<B, 3> {
        let [batch, sequence, _] = input.dims();
        let (heads, kv, dim) = (c.num_attention_heads, c.num_key_value_heads, c.head_dim);
        let p = format!("{prefix}.self_attn");
        let q = self.normalized(self.project(&format!("{p}.q_proj"), input.clone(), adapters)
            .reshape([batch, sequence, heads, dim]), &format!("{p}.q_norm"), c.rms_norm_eps, false).swap_dims(1, 2);
        let k = self.normalized(self.project(&format!("{p}.k_proj"), input.clone(), adapters)
            .reshape([batch, sequence, kv, dim]), &format!("{p}.k_norm"), c.rms_norm_eps, false).swap_dims(1, 2);
        let v = self.project(&format!("{p}.v_proj"), input, adapters).reshape([batch, sequence, kv, dim]);
        let parameters = c.rotary_parameters().expect("validated Qwen3 rotary parameters");
        let (cos, sin) = parameters.for_layer(layer).state().cos_sin(
            (0..sequence).map(|p| p as f32).collect(), 1, sequence, sequence, q.dtype(), &q.device());
        let q = crate::llama::rope::apply_partial(q, cos.clone(), sin.clone()).swap_dims(1, 2);
        let k = crate::llama::rope::apply_partial(k, cos, sin).swap_dims(1, 2);
        let context = super::super::flash_attention(q, k, v, 0).reshape([batch, sequence, heads * dim]);
        self.project(&format!("{p}.o_proj"), context, adapters)
    }

    pub(super) fn qwen35_attention(&self, input: Tensor<B, 3>, prefix: &str, c: &Qwen35TextConfig,
        adapters: Option<&NativeLora<B>>) -> Tensor<B, 3> {
        let [batch, sequence, _] = input.dims();
        let (heads, kv, dim) = (c.num_attention_heads, c.num_key_value_heads, c.head_dim);
        let p = format!("{prefix}.self_attn");
        let qgate = self.project(&format!("{p}.q_proj"), input.clone(), adapters).reshape([batch, sequence, heads, 2 * dim]);
        let gate = qgate.clone().slice([0..batch, 0..sequence, 0..heads, dim..2 * dim]).reshape([batch, sequence, heads * dim]);
        let q = self.normalized(qgate.slice([0..batch, 0..sequence, 0..heads, 0..dim]),
            &format!("{p}.q_norm"), c.rms_norm_eps, true).swap_dims(1, 2);
        let k = self.normalized(self.project(&format!("{p}.k_proj"), input.clone(), adapters)
            .reshape([batch, sequence, kv, dim]), &format!("{p}.k_norm"), c.rms_norm_eps, true).swap_dims(1, 2);
        let v = self.project(&format!("{p}.v_proj"), input, adapters).reshape([batch, sequence, kv, dim]);
        let (cos, sin) = c.rotary_parameters().expect("validated Qwen3.5 rotary parameters").state().cos_sin(
            (0..sequence).map(|p| p as f32).collect(), 1, sequence, sequence, q.dtype(), &q.device());
        let q = crate::llama::rope::apply_partial(q, cos.clone(), sin.clone()).swap_dims(1, 2);
        let k = crate::llama::rope::apply_partial(k, cos, sin).swap_dims(1, 2);
        let context = super::super::flash_attention(q, k, v, 0).reshape([batch, sequence, heads * dim]);
        self.project(&format!("{p}.o_proj"), context * sigmoid(gate), adapters)
    }

    pub(super) fn mla_attention(&self, input: Tensor<B, 3>, prefix: &str, c: &DeepseekV2Config,
        adapters: Option<&NativeLora<B>>) -> Tensor<B, 3> {
        let [batch, sequence, _] = input.dims();
        let (heads, rank, nd, rd, vd) = (c.num_attention_heads, c.kv_lora_rank, c.qk_nope_head_dim, c.qk_rope_head_dim, c.v_head_dim);
        let p = format!("{prefix}.self_attn");
        let (query_prefix, query_input) = if c.q_lora_rank.is_some() {
            let projected = self.project(&format!("{p}.q_a_proj"), input.clone(), adapters);
            (format!("{p}.q_b_proj"), self.normalized(projected, &format!("{p}.q_a_layernorm"), 1e-6, false))
        } else { (format!("{p}.q_proj"), input.clone()) };
        let query = self.project(&query_prefix, query_input, adapters).reshape([batch, sequence, heads, nd + rd]).swap_dims(1, 2);
        let compressed = self.project(&format!("{p}.kv_a_proj_with_mqa"), input, adapters);
        let latent = self.normalized(compressed.clone().slice([0..batch, 0..sequence, 0..rank]),
            &format!("{p}.kv_a_layernorm"), 1e-6, false);
        let expanded = self.project(&format!("{p}.kv_b_proj"), latent, adapters)
            .reshape([batch, sequence, heads, nd + vd]).swap_dims(1, 2);
        let (cos, sin, scale) = crate::deepseek2::training::rotary::<B>(c, sequence, query.dtype(), &query.device());
        let query_position = rotate_mla(query.clone().slice([0..batch, 0..heads, 0..sequence, nd..nd + rd]), cos.clone(), sin.clone());
        let key_position = rotate_mla(compressed.slice([0..batch, 0..sequence, rank..rank + rd])
            .reshape([batch, 1, sequence, rd]), cos, sin).repeat_dim(1, heads);
        let q = Tensor::cat(vec![query.slice([0..batch, 0..heads, 0..sequence, 0..nd]), query_position], 3);
        let k = Tensor::cat(vec![expanded.clone().slice([0..batch, 0..heads, 0..sequence, 0..nd]), key_position], 3);
        let v = expanded.slice([0..batch, 0..heads, 0..sequence, nd..nd + vd]);
        let context = attention(q, k, v, None, None, AttentionModuleOptions { scale: Some(scale), softcap: None, is_causal: true })
            .swap_dims(1, 2).reshape([batch, sequence, heads * vd]);
        self.project(&format!("{p}.o_proj"), context, adapters)
    }

    pub(super) fn kimi_mla_attention(&self, input: Tensor<B, 3>, prefix: &str, c: &KimiLinearConfig,
        adapters: Option<&NativeLora<B>>) -> Tensor<B, 3> {
        let [batch, sequence, _] = input.dims();
        let (heads, rank, nd, rd, vd) = (c.num_attention_heads, c.kv_lora_rank, c.qk_nope_head_dim, c.qk_rope_head_dim, c.v_head_dim);
        let p = format!("{prefix}.self_attn");
        let q = self.project(&format!("{p}.q_proj"), input.clone(), adapters).reshape([batch, sequence, heads, nd + rd]).swap_dims(1, 2);
        let compressed = self.project(&format!("{p}.kv_a_proj_with_mqa"), input, adapters);
        let latent = self.normalized(compressed.clone().slice([0..batch, 0..sequence, 0..rank]), &format!("{p}.kv_a_layernorm"), 1e-6, false);
        let expanded = self.project(&format!("{p}.kv_b_proj"), latent, adapters).reshape([batch, sequence, heads, nd + vd]).swap_dims(1, 2);
        let position = compressed.slice([0..batch, 0..sequence, rank..rank + rd]).reshape([batch, 1, sequence, rd]).repeat_dim(1, heads);
        let k = Tensor::cat(vec![expanded.clone().slice([0..batch, 0..heads, 0..sequence, 0..nd]), position], 3);
        let v = expanded.slice([0..batch, 0..heads, 0..sequence, nd..nd + vd]);
        let context = attention(q, k, v, None, None, AttentionModuleOptions {
            scale: Some(((nd + rd) as f64).sqrt().recip()), softcap: None, is_causal: true,
        }).swap_dims(1, 2).reshape([batch, sequence, heads * vd]);
        self.project(&format!("{p}.o_proj"), context, adapters)
    }
}

fn rotate_mla<B: Backend>(input: Tensor<B, 4>, cos: Tensor<B, 4>, sin: Tensor<B, 4>) -> Tensor<B, 4> {
    let [b, h, s, d] = input.dims();
    let input = input.reshape([b, h, s, d / 2, 2]).swap_dims(3, 4).reshape([b, h, s, d]);
    crate::llama::rope::apply_partial(input, cos, sin)
}
