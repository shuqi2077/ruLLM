//! Document-isolated training on the actual loaded Llama/Qwen2 parameters.
use super::*;
use ruda_nn::attention::{PackedAttentionOptions, PackedSequenceLayout, packed_scaled_dot_product_attention};
use ruda_nn::loss::PackedCausalLanguageModel;

fn rotate_documents<B: Backend>(
    input: Tensor<B, 3>, layout: &PackedSequenceLayout,
    rope: &RotaryEncoding<B>, rotary_layout: RotaryLayout, state: Option<&rope::RotaryState>,
) -> Tensor<B, 3> {
    let [tokens, heads, width] = input.dims();
    if tokens == 0 { return input; }
    let mut documents = Vec::new();
    for range in layout.boundaries().windows(2) {
        let length = range[1] - range[0];
        if length == 0 { continue; }
        let document = input.clone().slice_dim(0, range[0]..range[1])
            .reshape([1, length, heads, width]).swap_dims(1, 2);
        let rotated = if let Some(state) = state { state.fresh().apply(document, 0, layout.max_length()) } else { match rotary_layout {
            RotaryLayout::Interleaved => rope.apply(document, 0),
            RotaryLayout::HalfSplit => apply_half_split_rope(rope, document, 0),
        }};
        documents.push(rotated.swap_dims(1, 2).reshape([length, heads, width]));
    }
    Tensor::cat(documents, 0)
}

fn attend_documents<B: Backend>(
    query: Tensor<B, 3>, key: Tensor<B, 3>, value: Tensor<B, 3>,
    layout: &PackedSequenceLayout, rope: &RotaryEncoding<B>,
    rotary_layout: RotaryLayout, state: Option<&rope::RotaryState>, output: &Linear<B>,
) -> Tensor<B, 2> {
    let [tokens, heads, width] = query.dims();
    let query = rotate_documents(query, layout, rope, rotary_layout, state);
    let key = rotate_documents(key, layout, rope, rotary_layout, state);
    let options = PackedAttentionOptions { causal: true, ..Default::default() };
    let context = packed_scaled_dot_product_attention(query, key, value, layout, layout, options, None);
    output.forward(context.reshape([tokens, heads * width]))
}

impl<B: Backend> LlamaAttention<B> {
    fn forward_packed(&self, input: Tensor<B, 2>, layout: &PackedSequenceLayout) -> Tensor<B, 2> {
        let tokens = input.dims()[0];
        let query = self.q_proj.forward(input.clone()).reshape([tokens, self.num_query_heads, self.head_dimension]);
        let key = self.k_proj.forward(input.clone()).reshape([tokens, self.num_kv_heads, self.head_dimension]);
        let value = self.v_proj.forward(input).reshape([tokens, self.num_kv_heads, self.head_dimension]);
        attend_documents(query, key, value, layout, &self.rope, self.rotary_layout, self.rotary_state.as_ref(), &self.o_proj)
    }
}

impl<B: Backend> PackedLlamaAttention<B> {
    fn forward_packed(&self, input: Tensor<B, 2>, layout: &PackedSequenceLayout) -> Tensor<B, 2> {
        let tokens = input.dims()[0];
        let query_width = self.num_query_heads * self.head_dimension;
        let kv_width = self.num_kv_heads * self.head_dimension;
        // Q/K/V projections still execute once for the whole token payload;
        // only RoPE and attention are partitioned by actual document geometry.
        let projected = linear(input, self.qkv_weight.val(), self.qkv_bias.as_ref().map(Param::val));
        let query = projected.clone().slice_dim(1, 0..query_width)
            .reshape([tokens, self.num_query_heads, self.head_dimension]);
        let key = projected.clone().slice_dim(1, query_width..query_width + kv_width)
            .reshape([tokens, self.num_kv_heads, self.head_dimension]);
        let value = projected.slice_dim(1, query_width + kv_width..query_width + 2 * kv_width)
            .reshape([tokens, self.num_kv_heads, self.head_dimension]);
        attend_documents(query, key, value, layout, &self.rope, self.rotary_layout, self.rotary_state.as_ref(), &self.o_proj)
    }
}

impl<B: Backend> LlamaDecoderLayer<B> {
    fn forward_packed(&self, input: Tensor<B, 2>, layout: &PackedSequenceLayout) -> Tensor<B, 2> {
        let [tokens, width] = input.dims();
        let hidden = input.clone() + self.self_attn.forward_packed(self.input_layernorm.forward(input), layout);
        let normalized = self.post_attention_layernorm.forward(hidden.clone());
        hidden + self.mlp.forward(normalized.reshape([1, tokens, width])).reshape([tokens, width])
    }
}

impl<B: Backend> PackedLlamaDecoderLayer<B> {
    fn forward_packed(&self, input: Tensor<B, 2>, layout: &PackedSequenceLayout) -> Tensor<B, 2> {
        let [tokens, width] = input.dims();
        let hidden = input.clone() + self.self_attn.forward_packed(self.input_layernorm.forward(input), layout);
        let normalized = self.post_attention_layernorm.forward(hidden.clone());
        hidden + self.mlp.forward(normalized.reshape([1, tokens, width])).reshape([tokens, width])
    }
}

impl<B: Backend> LlamaForCausalLm<B> {
    /// Flat-token backbone with causal attention and reset positions per document.
    /// No dense global mask or padded samples are constructed. Uses the existing
    /// model parameters, aliases and Q/K/V biases/rotary layout without copying
    /// or repacking the model. Attention uses per-document FP32 score matrices.
    pub fn forward_packed_hidden(&self, tokens: Tensor<B, 1, Int>, layout: &PackedSequenceLayout) -> Tensor<B, 2> {
        let count = tokens.dims()[0];
        assert_eq!(count, layout.tokens(), "packed layout does not cover actual tokens");
        assert!(layout.max_length() <= self.max_sequence_length, "a document exceeds configured RoPE capacity");
        let width = self.embed_tokens.weight.val().dims()[1];
        let mut hidden = self.embed_tokens.forward(tokens.reshape([1, count])).reshape([count, width]);
        for layer in &self.layers { hidden = layer.forward_packed(hidden, layout); }
        self.norm.forward(hidden)
    }

    /// Flat logits `[actual_tokens,vocabulary]`; prefer the chunked loss when
    /// full-vocabulary logits for every token are not needed by the application.
    pub fn forward_packed(&self, tokens: Tensor<B, 1, Int>, layout: &PackedSequenceLayout) -> Tensor<B, 2> {
        self.lm_head.forward(self.forward_packed_hidden(tokens, layout))
    }
}

impl<B: Backend> PackedLlamaForCausalLm<B> {
    /// Isolated packed documents while retaining fused trainable QKV/gate-up
    /// projections. Projection packing and token packing are independent: the
    /// same parameter records work for dense and document-packed training.
    pub fn forward_packed_hidden(&self, tokens: Tensor<B, 1, Int>, layout: &PackedSequenceLayout) -> Tensor<B, 2> {
        let count = tokens.dims()[0];
        assert_eq!(count, layout.tokens(), "packed layout does not cover actual tokens");
        assert!(layout.max_length() <= self.max_sequence_length, "a document exceeds configured RoPE capacity");
        let width = self.embed_tokens.weight.val().dims()[1];
        let mut hidden = self.embed_tokens.forward(tokens.reshape([1, count])).reshape([count, width]);
        for layer in &self.layers { hidden = layer.forward_packed(hidden, layout); }
        self.norm.forward(hidden)
    }

    /// Flat full-vocabulary logits using the already packed model projections.
    pub fn forward_packed(&self, tokens: Tensor<B, 1, Int>, layout: &PackedSequenceLayout) -> Tensor<B, 2> {
        self.lm_head.forward(self.forward_packed_hidden(tokens, layout))
    }
}

impl<B: TrainingBackend> PackedLlamaForCausalLm<B> {
    /// Flat, document-isolated training with optional LoRA. Projections and
    /// feed-forward layers process the entire payload; each attention call
    /// uses one document's actual rotary context and memory-efficient backward.
    pub fn forward_train_packed_hidden(&self, tokens: Tensor<B, 1, Int>,
        layout: &PackedSequenceLayout, lora: Option<&LlamaLora<B>>) -> Tensor<B, 2> {
        self.forward_train_packed_hidden_with_contexts(tokens, layout, lora, None)
    }

    /// Replay actual paged append contexts, one end position per token. This
    /// preserves the rotary phases of cached keys when runtime RoPE changes.
    pub fn forward_train_packed_hidden_with_contexts(&self, tokens: Tensor<B, 1, Int>,
        layout: &PackedSequenceLayout, lora: Option<&LlamaLora<B>>,
        append_contexts: Option<&[usize]>) -> Tensor<B, 2> {
        let count = tokens.dims()[0];
        assert_eq!(count, layout.tokens(), "packed layout must cover the tokens");
        assert!(count > 0 && layout.max_length() <= self.max_sequence_length);
        if let Some(contexts) = append_contexts { assert_eq!(contexts.len(), count); }
        if let Some(lora) = lora {
            assert_eq!(lora.layers.len(), self.layers.len());
        }
        let width = self.lora_shapes().hidden;
        let scale = lora.map_or(0.0, LlamaLora::scale);
        let mut hidden = self.embed_tokens.forward(tokens.reshape([1, count]));
        for (index, layer) in self.layers.iter().enumerate() {
            let adapter = lora.map(|lora| &lora.layers[index]);
            let attention = &layer.self_attn;
            let normalized = layer.input_layernorm.forward(hidden.clone());
            let projected = with_lora(linear(normalized.clone(), attention.qkv_weight.val(),
                attention.qkv_bias.as_ref().map(Param::val)), normalized,
                adapter.and_then(|a| a.qkv.as_ref()), scale);
            let query_width = attention.num_query_heads * attention.head_dimension;
            let kv_width = attention.num_kv_heads * attention.head_dimension;
            let mut contexts = Vec::new();
            for range in layout.boundaries().windows(2) {
                let (start, end) = (range[0], range[1]);
                let length = end - start;
                if length == 0 { continue; }
                let split = |range: core::ops::Range<usize>, heads| projected.clone()
                    .slice([0..1, start..end, range])
                    .reshape([1, length, heads, attention.head_dimension]);
                let tables = append_contexts.and_then(|contexts| attention.rotary_state.as_ref().map(|state| {
                    let mut previous = None;
                    let contexts: Vec<_> = contexts[start..end].iter().map(|&end| {
                        let context = state.paged_context(previous, end);
                        previous = Some(context);
                        context
                    }).collect();
                    state.paged_table(0, &contexts, projected.dtype(), &projected.device())
                }));
                let rotate = |x: Tensor<B, 4>| {
                    let x = x.swap_dims(1, 2);
                    let x = match &tables {
                        Some(table) => {
                            let dimension = table.dims()[1];
                            let cos = table.clone().slice([0..length, 0..dimension, 0..1])
                                .reshape([1, 1, length, dimension]);
                            let sin = table.clone().slice([0..length, 0..dimension, 1..2])
                                .reshape([1, 1, length, dimension]);
                            rope::apply_partial(x, cos, sin)
                        }
                        None => attention.apply_rope(x, 0),
                    };
                    x.swap_dims(1, 2)
                };
                let query = rotate(split(0..query_width, attention.num_query_heads));
                let key = rotate(split(query_width..query_width + kv_width, attention.num_kv_heads));
                let value = split(query_width + kv_width..query_width + 2 * kv_width, attention.num_kv_heads);
                contexts.push(flash_attention(query, key, value, attention.sliding_window)
                    .reshape([1, length, query_width]));
            }
            let context = Tensor::cat(contexts, 1);
            hidden = hidden + with_lora(attention.o_proj.forward(context.clone()), context,
                adapter.and_then(|a| a.output.as_ref()), scale);
            let normalized = layer.post_attention_layernorm.forward(hidden.clone());
            let mlp = &layer.mlp;
            let gate_up = with_lora(linear(normalized.clone(), mlp.gate_up_weight.val(), None),
                normalized, adapter.and_then(|a| a.gate_up.as_ref()), scale);
            let gate = gate_up.clone().slice([0..1, 0..count, 0..mlp.d_ff]);
            let up = gate_up.slice([0..1, 0..count, mlp.d_ff..2 * mlp.d_ff]);
            let activated = silu(gate) * up;
            hidden = hidden + with_lora(mlp.down_proj.forward(activated.clone()), activated,
                adapter.and_then(|a| a.down.as_ref()), scale);
        }
        self.norm.forward(hidden).reshape([count, width])
    }

    /// State-aligned selected-token log probabilities on isolated flat documents.
    pub fn train_packed_log_probs(&self, tokens: Tensor<B, 1, Int>, targets: Tensor<B, 1, Int>,
        layout: &PackedSequenceLayout, lora: Option<&LlamaLora<B>>, chunk_rows: usize) -> Tensor<B, 1>
    where B: crate::training::LinearLogProbsBackend {
        self.train_packed_log_probs_with_contexts(tokens, targets, layout, lora, chunk_rows, None)
    }

    pub fn train_packed_log_probs_with_contexts(&self, tokens: Tensor<B, 1, Int>, targets: Tensor<B, 1, Int>,
        layout: &PackedSequenceLayout, lora: Option<&LlamaLora<B>>, chunk_rows: usize,
        append_contexts: Option<&[usize]>) -> Tensor<B, 1>
    where B: crate::training::LinearLogProbsBackend {
        crate::training::linear_log_probs(self.forward_train_packed_hidden_with_contexts(tokens, layout, lora, append_contexts),
            self.lm_head.weight.val(), targets, chunk_rows)
    }
}

impl<B: Backend> PackedCausalLanguageModel<B> for LlamaForCausalLm<B> {
    fn forward_packed_hidden(&self, tokens: Tensor<B, 1, Int>, layout: &PackedSequenceLayout) -> Tensor<B, 2> {
        LlamaForCausalLm::forward_packed_hidden(self, tokens, layout)
    }

    fn project(&self, hidden: Tensor<B, 2>) -> Tensor<B, 2> { self.lm_head.forward(hidden) }
}

impl<B: Backend> PackedCausalLanguageModel<B> for PackedLlamaForCausalLm<B> {
    fn forward_packed_hidden(&self, tokens: Tensor<B, 1, Int>, layout: &PackedSequenceLayout) -> Tensor<B, 2> {
        PackedLlamaForCausalLm::forward_packed_hidden(self, tokens, layout)
    }

    fn project(&self, hidden: Tensor<B, 2>) -> Tensor<B, 2> { self.lm_head.forward(hidden) }
}
