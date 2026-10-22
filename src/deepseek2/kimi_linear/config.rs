use super::*;
use serde::Deserialize;
use std::collections::BTreeSet;

#[derive(Clone, Debug, Deserialize)]
pub struct KimiLinearAttentionConfig {
    pub num_heads: usize, pub head_dim: usize, pub short_conv_kernel_size: usize,
    pub full_attn_layers: Vec<usize>, pub kda_layers: Vec<usize>,
}
#[derive(Clone, Debug, Deserialize)]
pub struct KimiLinearConfig {
    pub model_type: String,
    pub vocab_size: usize, pub hidden_size: usize, pub intermediate_size: usize,
    pub num_hidden_layers: usize, pub num_attention_heads: usize,
    #[serde(alias = "max_position_embeddings")]
    pub model_max_length: usize,
    pub kv_lora_rank: usize, pub qk_nope_head_dim: usize, pub qk_rope_head_dim: usize, pub v_head_dim: usize,
    #[serde(default)] pub q_lora_rank: Option<usize>,
    pub mla_use_nope: bool,
    pub rms_norm_eps: f64, pub hidden_act: String,
    #[serde(default)] pub tie_word_embeddings: bool,
    #[serde(default)] pub num_experts: Option<usize>,
    #[serde(default)] pub num_experts_per_token: Option<usize>,
    #[serde(default)] pub num_shared_experts: usize,
    #[serde(default)] pub moe_intermediate_size: Option<usize>,
    #[serde(default = "yes")] pub moe_renormalize: bool,
    #[serde(default = "sigmoid_name")] pub moe_router_activation_func: String,
    #[serde(default = "one")] pub num_expert_group: usize,
    #[serde(default = "one")] pub topk_group: usize,
    #[serde(default = "scale")] pub routed_scaling_factor: f64,
    #[serde(default)] pub first_k_dense_replace: usize,
    #[serde(default = "one")] pub moe_layer_freq: usize,
    #[serde(default)] pub num_nextn_predict_layers: usize,
    #[serde(default)] pub linear_attn_config: Option<KimiLinearAttentionConfig>,
    #[serde(default)] pub eos_token_id: Option<serde_json::Value>,
    #[serde(default)] pub quantization_config: Option<serde_json::Value>,
}
fn one() -> usize { 1 }
fn yes() -> bool { true }
fn scale() -> f64 { 1.0 }
fn sigmoid_name() -> String { "sigmoid".into() }
impl KimiLinearConfig {
    pub fn limits(&self) -> CausalModelLimits { CausalModelLimits {vocab_size: self.vocab_size, max_sequence_length: self.model_max_length} }
    pub fn is_kda_layer(&self, index: usize) -> bool { self.linear_attn_config.as_ref().is_some_and(|c| c.kda_layers.contains(&(index + 1))) }
    pub(crate) fn is_moe(&self, index: usize) -> bool { self.num_experts.is_some() && index >= self.first_k_dense_replace && index % self.moe_layer_freq == 0 }
    pub(super) fn mla_layers(&self) -> usize { (0..self.num_hidden_layers).filter(|&i| !self.is_kda_layer(i)).count() }
    pub(crate) fn routing(&self) -> DeepseekV2Config {
        DeepseekV2Config {
            model_type: self.model_type.clone(), vocab_size: self.vocab_size, hidden_size: self.hidden_size,
            intermediate_size: self.intermediate_size, moe_intermediate_size: self.moe_intermediate_size.unwrap_or(0),
            num_hidden_layers: self.num_hidden_layers, num_attention_heads: self.num_attention_heads,
            num_key_value_heads: self.num_attention_heads, max_position_embeddings: self.model_max_length,
            kv_lora_rank: self.kv_lora_rank, q_lora_rank: None, qk_nope_head_dim: self.qk_nope_head_dim,
            qk_rope_head_dim: self.qk_rope_head_dim, v_head_dim: self.v_head_dim,
            n_routed_experts: self.num_experts.unwrap_or(0), n_shared_experts: (self.num_shared_experts > 0).then_some(self.num_shared_experts),
            num_experts_per_tok: self.num_experts_per_token.unwrap_or(0), first_k_dense_replace: self.first_k_dense_replace,
            moe_layer_freq: self.moe_layer_freq, norm_topk_prob: self.moe_renormalize, routed_scaling_factor: self.routed_scaling_factor,
            scoring_func: self.moe_router_activation_func.clone(), topk_method: "noaux_tc".into(), n_group: self.num_expert_group,
            topk_group: self.topk_group, rms_norm_eps: self.rms_norm_eps, rope_theta: 10000.0, rope_scaling: None,
            hidden_act: self.hidden_act.clone(), attention_bias: false, attention_dropout: 0.0,
            tie_word_embeddings: self.tie_word_embeddings, pretraining_tp: 1, eos_token_id: self.eos_token_id.clone(),
            quantization_config: self.quantization_config.clone(),
        }
    }
    pub fn validate(&self) -> Result<(), HuggingFaceLoadError> {
        let fail = || HuggingFaceLoadError("incompatible Kimi Linear configuration".into());
        if self.model_type != "kimi_linear" || !self.mla_use_nope || self.q_lora_rank.is_some()
            || self.hidden_act != "silu" || self.num_nextn_predict_layers != 0
            || [self.vocab_size, self.hidden_size, self.intermediate_size, self.num_hidden_layers, self.num_attention_heads,
                self.model_max_length, self.kv_lora_rank, self.qk_nope_head_dim, self.qk_rope_head_dim, self.v_head_dim, self.moe_layer_freq].contains(&0)
            || self.vocab_size > i32::MAX as usize || self.model_max_length > i32::MAX as usize
            || !self.rms_norm_eps.is_finite() || self.rms_norm_eps <= 0.0
            || !self.routed_scaling_factor.is_finite() || self.routed_scaling_factor <= 0.0 { return Err(fail()); }
        if let Some(experts) = self.num_experts {
            if experts == 0 || experts > i32::MAX as usize || self.num_experts_per_token.is_none_or(|k| k == 0 || k > experts)
                || self.moe_intermediate_size.is_none_or(|d| d == 0) || !super::super::routing::valid(&self.routing()) { return Err(fail()); }
            self.moe_intermediate_size.unwrap().checked_mul(self.num_shared_experts).ok_or_else(fail)?;
        }
        if let Some(c) = &self.linear_attn_config {
            let kda = c.kda_layers.iter().copied().collect::<BTreeSet<_>>();
            let full = c.full_attn_layers.iter().copied().collect::<BTreeSet<_>>();
            if [c.num_heads, c.head_dim, c.short_conv_kernel_size].contains(&0)
                || kda.len() != c.kda_layers.len() || full.len() != c.full_attn_layers.len()
                || !kda.is_disjoint(&full) || kda.union(&full).copied().collect::<BTreeSet<_>>() != (1..=self.num_hidden_layers).collect() { return Err(fail()); }
            c.num_heads.checked_mul(c.head_dim).and_then(|n| n.checked_mul(self.hidden_size)).ok_or_else(fail)?;
        }
        self.qk_nope_head_dim.checked_add(self.qk_rope_head_dim).and_then(|n| n.checked_mul(self.num_attention_heads))
            .and_then(|n| n.checked_mul(self.hidden_size)).ok_or_else(fail)?;
        self.kv_lora_rank.checked_add(self.qk_rope_head_dim).and_then(|n| n.checked_mul(self.hidden_size)).ok_or_else(fail)?;
        self.qk_nope_head_dim.checked_add(self.v_head_dim).and_then(|n| n.checked_mul(self.num_attention_heads))
            .and_then(|n| n.checked_mul(self.kv_lora_rank)).ok_or_else(fail)?;
        weights::block_config(self.quantization_config.as_ref())?;
        Ok(())
    }
}
