use super::*;
use crate::gpu_inference::{gqa_scores, gqa_value_product};
use ruda_tensor::api::activation::{sigmoid, softmax};
use ruda_tensor_tch::{LibTorch, TchElement};

type Torch<E> = LibTorch<E>;

impl<E: TchElement> Mlp<Torch<E>> {
    fn forward(&self, input: Tensor<Torch<E>, 3>) -> Tensor<Torch<E>, 3> {
        let gate = self.gate.forward(input.clone());
        let dtype = gate.dtype();
        self.down
            .forward(silu(gate.cast(DType::F32)).cast(dtype) * self.up.forward(input))
    }
}

impl<E: TchElement> Norm<Torch<E>> {
    fn forward<const D: usize>(
        &self,
        input: Tensor<Torch<E>, D>,
    ) -> Result<Tensor<Torch<E>, D>, GenerationError> {
        let dtype = input.dtype();
        let full = input.cast(DType::F32);
        let inverse = (full.clone().square().mean_dim(D - 1) + self.epsilon).rsqrt();
        Ok(
            (full * inverse * (self.weight.clone().cast(DType::F32) + 1.0).unsqueeze::<D>())
                .cast(dtype),
        )
    }
}

impl<E: TchElement> attention::Attention<Torch<E>> {
    fn forward(
        &self,
        x: Tensor<Torch<E>, 3>,
        c: &Qwen35TextConfig,
        cos: Tensor<Torch<E>, 4>,
        sin: Tensor<Torch<E>, 4>,
        key_cache: &mut Option<Tensor<Torch<E>, 4>>,
        value_cache: &mut Option<Tensor<Torch<E>, 4>>,
        position: usize,
    ) -> Result<Tensor<Torch<E>, 3>, GenerationError> {
        let [b, s, _] = x.dims();
        let (h, k, d) = (c.num_attention_heads, c.num_key_value_heads, c.head_dim);
        let qgate = self.q.forward(x.clone()).reshape([b, s, h, 2 * d]);
        let gate = qgate
            .clone()
            .slice([0..b, 0..s, 0..h, d..2 * d])
            .reshape([b, s, h * d]);
        let q = self
            .q_norm
            .forward(qgate.slice([0..b, 0..s, 0..h, 0..d]))?
            .swap_dims(1, 2);
        let key = self
            .k_norm
            .forward(self.k.forward(x.clone()).reshape([b, s, k, d]))?
            .swap_dims(1, 2);
        let value = self.v.forward(x).reshape([b, s, k, d]).swap_dims(1, 2);
        let q = attention::rotate(q, cos.clone(), sin.clone());
        let key = attention::rotate(key, cos, sin);
        let key = match key_cache.as_ref() {
            Some(old) => Tensor::cat(vec![old.clone(), key], 2),
            None => key,
        };
        let value = match value_cache.as_ref() {
            Some(old) => Tensor::cat(vec![old.clone(), value], 2),
            None => value,
        };
        *key_cache = Some(key.clone());
        *value_cache = Some(value.clone());
        let mask =
            attention::causal_mask::<Torch<E>>(s, position, key.dims()[2], q.dtype(), &q.device());
        let ops = attention::shared_tensor_ops::<Torch<E>>();
        let scores = gqa_scores(&ops, q, key) * (d as f64).sqrt().recip();
        let scores = match mask {
            Some(mask) => scores + mask,
            None => scores,
        };
        let dtype = scores.dtype();
        let probabilities = softmax(scores.cast(DType::F32), 3).cast(dtype);
        let out = gqa_value_product(&ops, probabilities, value)
            .swap_dims(1, 2)
            .reshape([b, s, h * d]);
        Ok(self.out.forward(out * sigmoid(gate)))
    }
}

impl<E: TchElement> delta::Delta<Torch<E>> {
    fn forward(
        &self,
        x: Tensor<Torch<E>, 3>,
        c: &Qwen35TextConfig,
        conv_cache: &mut Option<Tensor<Torch<E>, 3>>,
        state: &mut Option<Tensor<Torch<E>, 4>>,
    ) -> Result<Tensor<Torch<E>, 3>, GenerationError> {
        let [batch, seq, _] = x.dims();
        let (kh, vh, kd, vd) = (
            c.linear_num_key_heads,
            c.linear_num_value_heads,
            c.linear_key_head_dim,
            c.linear_value_head_dim,
        );
        let (keys, values) = (kh * kd, vh * vd);
        let channels = keys * 2 + values;
        let width = c.linear_conv_kernel_dim;
        let dtype = x.dtype();
        let device = x.device();
        let mixed = self.qkv.forward(x.clone()).swap_dims(1, 2);
        let previous = conv_cache
            .as_ref()
            .map(Tensor::clone)
            .unwrap_or_else(|| Tensor::zeros([batch, channels, width], (&device, dtype)));
        let history = Tensor::cat(vec![previous, mixed], 2);
        *conv_cache = Some(
            history
                .clone()
                .slice([0..batch, 0..channels, seq..seq + width]),
        );
        let mixed = delta::causal_convolution(history, self.conv.clone(), seq);
        let query = mixed
            .clone()
            .slice([0..batch, 0..seq, 0..keys])
            .reshape([batch, seq, kh, kd])
            .swap_dims(1, 2);
        let key = mixed
            .clone()
            .slice([0..batch, 0..seq, keys..2 * keys])
            .reshape([batch, seq, kh, kd])
            .swap_dims(1, 2);
        let value = mixed
            .slice([0..batch, 0..seq, 2 * keys..channels])
            .reshape([batch, seq, vh, vd])
            .swap_dims(1, 2);
        let l2 = |x: Tensor<Torch<E>, 4>| {
            let inv = (x.clone().square().sum_dim(3) + 1e-6)
                .cast(DType::F32)
                .rsqrt()
                .cast(dtype);
            x * inv
        };
        let query = attention::repeat_heads(l2(query), vh / kh);
        let key = attention::repeat_heads(l2(key), vh / kh);
        let beta = sigmoid(self.b.forward(x.clone())).swap_dims(1, 2);
        let a = self.a.forward(x.clone()).cast(DType::F32)
            + self.dt_bias.clone().cast(DType::F32).unsqueeze::<3>();
        let softplus = a
            .clone()
            .exp()
            .log1p()
            .mask_where(a.clone().greater_elem(20.0), a);
        let decay = -(self.a_log.clone().cast(DType::F32).exp().unsqueeze::<3>() * softplus)
            .swap_dims(1, 2);
        let recurrent_decode = state.is_some() && seq == 1;
        let initial = state
            .as_ref()
            .map(Tensor::clone)
            .unwrap_or_else(|| Tensor::zeros([batch, vh, kd, vd], (&device, DType::F32)));
        let query = query.into_primitive().tensor();
        let key = key.into_primitive().tensor();
        let value = value.into_primitive().tensor();
        let beta = beta.into_primitive().tensor();
        let decay = decay.into_primitive().tensor();
        let initial = initial.into_primitive().tensor();
        let (output, final_state) = super::libtorch_delta::gated_delta(
            &query.tensor,
            &key.tensor,
            &value.tensor,
            &beta.tensor,
            &decay.tensor,
            &initial.tensor,
            recurrent_decode,
        )?;
        *state = Some(Tensor::from_primitive(TensorPrimitive::Float(
            ruda_tensor_tch::TchTensor::new(final_state),
        )));
        let y = Tensor::<Torch<E>, 4>::from_primitive(TensorPrimitive::Float(
            ruda_tensor_tch::TchTensor::new(output),
        ))
        .cast(dtype)
        .swap_dims(1, 2)
        .cast(DType::F32);
        let z = self
            .z
            .forward(x)
            .reshape([batch, seq, vh, vd])
            .cast(DType::F32);
        let inverse = (y.clone().square().mean_dim(3) + c.rms_norm_eps).rsqrt();
        let normalized =
            (y * inverse).cast(dtype).cast(self.norm.dtype()) * self.norm.clone().unsqueeze::<4>();
        let gated = (normalized.cast(DType::F32) * silu(z))
            .cast(dtype)
            .reshape([batch, seq, values]);
        Ok(self.out.forward(gated))
    }
}

impl<E: TchElement> Layer<Torch<E>> {
    fn forward(
        &self,
        hidden: Tensor<Torch<E>, 3>,
        config: &Qwen35TextConfig,
        cos: Tensor<Torch<E>, 4>,
        sin: Tensor<Torch<E>, 4>,
        state: &mut LayerCache<Torch<E>>,
        position: usize,
    ) -> Result<Tensor<Torch<E>, 3>, GenerationError> {
        let normalized = self.input_norm.forward(hidden.clone())?;
        let update = match (&self.mixer, state) {
            (Mixer::Full(attn), LayerCache::Full { key, value }) => {
                attn.forward(normalized, config, cos, sin, key, value, position)?
            }
            (Mixer::Delta(delta), LayerCache::Delta { convolution, state }) => {
                delta.forward(normalized, config, convolution, state)?
            }
            _ => return Err(GenerationError("Qwen3.5 cache layer type mismatch".into())),
        };
        let hidden = hidden + update;
        Ok(hidden.clone() + self.mlp.forward(self.post_norm.forward(hidden)?))
    }
}

impl<E: TchElement> Qwen35TextModel<Torch<E>> {
    fn forward_hidden(
        &self,
        tokens: Tensor<Torch<E>, 2, Int>,
        cache: &mut Qwen35Cache<Torch<E>>,
    ) -> Result<Tensor<Torch<E>, 3>, GenerationError> {
        let [batch, sequence] = tokens.dims();
        if batch == 0
            || sequence == 0
            || cache.batch.is_some_and(|n| n != batch)
            || cache.layers.len() != self.layers.len()
            || cache
                .position
                .checked_add(sequence)
                .is_none_or(|n| n > self.config.max_position_embeddings)
        {
            return Err(GenerationError(
                "invalid Qwen3.5 batch, sequence or cache capacity".into(),
            ));
        }
        let mut hidden = self.embedding.forward(tokens);
        let (cos, sin) = attention::text_rope(
            &self.config,
            cache.position,
            sequence,
            hidden.dtype(),
            &hidden.device(),
        );
        for (layer, state) in self.layers.iter().zip(cache.layers.iter_mut()) {
            hidden = layer.forward(
                hidden,
                &self.config,
                cos.clone(),
                sin.clone(),
                state,
                cache.position,
            )?;
        }
        cache.position += sequence;
        cache.batch = Some(batch);
        Ok(hidden)
    }

    pub fn forward_cached(
        &self,
        tokens: Tensor<Torch<E>, 2, Int>,
        cache: &mut Qwen35Cache<Torch<E>>,
    ) -> Result<Tensor<Torch<E>, 3>, GenerationError> {
        let hidden = self.forward_hidden(tokens, cache)?;
        Ok(self.head.forward(self.norm.forward(hidden)?))
    }

    pub fn forward_cached_last(
        &self,
        tokens: Tensor<Torch<E>, 2, Int>,
        cache: &mut Qwen35Cache<Torch<E>>,
    ) -> Result<Tensor<Torch<E>, 3>, GenerationError> {
        let hidden = self.forward_hidden(tokens, cache)?;
        let [batch, sequence, width] = hidden.dims();
        Ok(self.head.forward(self.norm.forward(hidden.slice([
            0..batch,
            sequence - 1..sequence,
            0..width,
        ]))?))
    }
}

impl<E: TchElement> CausalModel<Torch<E>> for Qwen35TextModel<Torch<E>> {
    type Cache = Qwen35Cache<Torch<E>>;
    fn new_cache(&self) -> Self::Cache {
        self.new_cache()
    }
    fn forward_cached_last(
        &self,
        tokens: Tensor<Torch<E>, 2, Int>,
        cache: &mut Self::Cache,
    ) -> Tensor<Torch<E>, 3> {
        self.forward_cached_last(tokens, cache)
            .expect("validated Qwen3.5 cached generation")
    }
    fn try_forward_cached_last(
        &self,
        tokens: Tensor<Torch<E>, 2, Int>,
        cache: &mut Self::Cache,
    ) -> Result<Tensor<Torch<E>, 3>, GenerationError> {
        self.forward_cached_last(tokens, cache)
    }
}

impl<E: TchElement> crate::SpeculativeModel<Torch<E>> for Qwen35TextModel<Torch<E>> {
    fn fork_cache(&self, cache: &Self::Cache) -> Self::Cache {
        cache.clone()
    }
    fn try_forward_cached_block(
        &self,
        tokens: Tensor<Torch<E>, 2, Int>,
        cache: &mut Self::Cache,
    ) -> Result<Tensor<Torch<E>, 3>, GenerationError> {
        self.forward_cached(tokens, cache)
    }
}
