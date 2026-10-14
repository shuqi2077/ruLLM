use crate::HuggingFaceLoadError;
use serde::Deserialize;

#[derive(Debug, Clone, Copy, Deserialize, PartialEq, Eq)]
#[serde(rename_all = "snake_case")]
pub enum LayerType {
    LinearAttention,
    FullAttention,
}

#[derive(Debug, Clone, Deserialize)]
pub struct RopeConfig {
    pub rope_type: String,
    pub rope_theta: f64,
    pub partial_rotary_factor: f64,
    pub mrope_section: [usize; 3],
    pub mrope_interleaved: bool,
    #[serde(flatten)]
    pub scaling: std::collections::BTreeMap<String, serde_json::Value>,
}

#[derive(Debug, Clone, Deserialize)]
pub struct Qwen35MoeConfig {
    #[serde(default)] pub num_experts: usize,
    #[serde(default)] pub num_experts_per_tok: usize,
    #[serde(default)] pub moe_intermediate_size: usize,
    #[serde(default)] pub shared_expert_intermediate_size: usize,
}

#[derive(Debug, Clone, Deserialize)]
pub struct Qwen35TextConfig {
    pub model_type: String,
    pub hidden_size: usize,
    #[serde(default)]
    pub intermediate_size: usize,
    #[serde(flatten)]
    pub moe: Qwen35MoeConfig,
    pub vocab_size: usize,
    pub num_hidden_layers: usize,
    pub num_attention_heads: usize,
    pub num_key_value_heads: usize,
    pub head_dim: usize,
    pub max_position_embeddings: usize,
    #[serde(default)]
    pub original_max_position_embeddings: Option<usize>,
    pub layer_types: Vec<LayerType>,
    pub linear_conv_kernel_dim: usize,
    pub linear_key_head_dim: usize,
    pub linear_value_head_dim: usize,
    pub linear_num_key_heads: usize,
    pub linear_num_value_heads: usize,
    pub rms_norm_eps: f64,
    pub hidden_act: String,
    pub attention_bias: bool,
    pub attention_dropout: f64,
    #[serde(default)]
    pub tie_word_embeddings: bool,
    pub rope_parameters: RopeConfig,
    pub eos_token_id: i32,
    pub mamba_ssm_dtype: String,
    #[serde(default)]
    pub quantization_config: Option<serde_json::Value>,
}

impl Qwen35TextConfig {
    pub fn is_moe(&self) -> bool { self.model_type == "qwen3_5_moe_text" }

    pub(crate) fn rotary_parameters(&self) -> Result<crate::llama::rope::RopeParameters, HuggingFaceLoadError> {
        let rope = &self.rope_parameters;
        let mut fields: serde_json::Map<String, serde_json::Value> = rope.scaling.clone().into_iter().collect();
        fields.insert("rope_type".into(), rope.rope_type.clone().into());
        fields.insert("rope_theta".into(), rope.rope_theta.into());
        fields.insert("partial_rotary_factor".into(), rope.partial_rotary_factor.into());
        crate::huggingface::rope::static_parameters(self.head_dim, self.max_position_embeddings,
            rope.rope_theta, self.original_max_position_embeddings, rope.partial_rotary_factor,
            Some(&serde_json::Value::Object(fields)))
    }

    pub fn validate(&self) -> Result<(), HuggingFaceLoadError> {
        let fail = |message: &str| Err(HuggingFaceLoadError(message.into()));
        if !matches!(self.model_type.as_str(), "qwen3_5_text" | "qwen3_5_moe_text")
            || self.hidden_act != "silu"
            || self.attention_dropout != 0.0
            || self.mamba_ssm_dtype != "float32"
            || self.quantization_config.is_some()
        {
            return fail(
                "unsupported Qwen3.5 text architecture, activation, RoPE, dropout, state dtype or quantization",
            );
        }
        if [
            self.hidden_size,
            self.vocab_size,
            self.num_hidden_layers,
            self.num_attention_heads,
            self.num_key_value_heads,
            self.head_dim,
            self.max_position_embeddings,
            self.linear_conv_kernel_dim,
            self.linear_key_head_dim,
            self.linear_value_head_dim,
            self.linear_num_key_heads,
            self.linear_num_value_heads,
        ]
        .contains(&0)
            || self.layer_types.len() != self.num_hidden_layers
        {
            return fail(
                "Qwen3.5 dimensions must be positive and layer_types must cover every layer",
            );
        }
        if self.is_moe() {
            let moe = &self.moe;
            if moe.num_experts == 0 || moe.num_experts > i32::MAX as usize
                || moe.num_experts_per_tok == 0 || moe.num_experts_per_tok > moe.num_experts
                || moe.moe_intermediate_size == 0 || moe.shared_expert_intermediate_size == 0
                || moe.moe_intermediate_size.checked_mul(2).and_then(|n| n.checked_mul(self.hidden_size)).is_none()
                || moe.shared_expert_intermediate_size.checked_mul(self.hidden_size).is_none() {
                return fail("invalid Qwen3.5 MoE expert, shared-expert or routing geometry");
            }
        } else if self.intermediate_size == 0 {
            return fail("dense Qwen3.5 requires a positive intermediate_size");
        }
        if self.num_attention_heads % self.num_key_value_heads != 0
            || self.linear_num_value_heads % self.linear_num_key_heads != 0
        {
            return fail("Qwen3.5 query/value head counts must be divisible by key head counts");
        }
        let rope = &self.rope_parameters;
        let rotary = self.head_dim as f64 * rope.partial_rotary_factor;
        if !rotary.is_finite()
            || rotary < 2.0
            || rotary > self.head_dim as f64
            || rotary.fract() != 0.0
            || rotary as usize % 2 != 0
            || !rope.rope_theta.is_finite()
            || rope.rope_theta <= 0.0
            || !self.rms_norm_eps.is_finite()
            || self.rms_norm_eps <= 0.0
            || self.eos_token_id < 0
            || self.eos_token_id as usize >= self.vocab_size
        {
            return fail("invalid Qwen3.5 rotary, normalization or EOS configuration");
        }
        for (a, b) in [
            (self.num_attention_heads, self.head_dim.saturating_mul(2)),
            (self.linear_num_key_heads, self.linear_key_head_dim),
            (self.linear_num_value_heads, self.linear_value_head_dim),
            (self.vocab_size, self.hidden_size),
        ] {
            if a.checked_mul(b).is_none_or(|n| n > u32::MAX as usize) {
                return fail("Qwen3.5 tensor dimensions exceed supported indexing");
            }
        }
        self.rotary_parameters()?;
        Ok(())
    }
}
