use super::*;
use ruda_tensor::api::{activation::{sigmoid, silu}, module::conv1d, ops::ConvOptions};

fn softplus<B: Backend, const D: usize>(input: Tensor<B, D>) -> Tensor<B, D> {
    let large = input.clone().greater_elem(20.0);
    input.clone().clamp_max(20.0).exp().log1p().mask_where(large, input)
}

fn convolution<B: Backend>(input: Tensor<B, 3>, weight: Tensor<B, 3>, round_before_silu: bool) -> Tensor<B, 3> {
    let [batch, sequence, channels] = input.dims();
    let width = weight.dims()[2];
    let dtype = input.dtype();
    let history = Tensor::cat(vec![Tensor::zeros([batch, channels, width], (&input.device(), dtype)), input.swap_dims(1, 2)], 2);
    let output = conv1d(history.cast(DType::F32), weight.cast(DType::F32), None,
        ConvOptions::new([1], [0], [1], channels));
    let output = if round_before_silu { output.cast(dtype).cast(DType::F32) } else { output };
    silu(output).cast(dtype).slice([0..batch, 0..channels, 1..sequence + 1]).swap_dims(1, 2)
}

fn repeat_heads<B: Backend>(input: Tensor<B, 4>, repetitions: usize) -> Tensor<B, 4> {
    if repetitions == 1 { return input; }
    let [batch, heads, sequence, width] = input.dims();
    input.unsqueeze_dim::<5>(2).repeat_dim(2, repetitions).reshape([batch, heads * repetitions, sequence, width])
}

/// Exact delta update, expressed through differentiable tensor operations. The
/// keywise decay case is KDA; scalar head decay is Qwen3.5's gated delta rule.
fn delta_rule<B: Backend>(query: Tensor<B, 4>, key: Tensor<B, 4>, value: Tensor<B, 4>,
    beta: Tensor<B, 3>, log_decay: Tensor<B, 4>) -> Tensor<B, 4> {
    let [batch, heads, sequence, key_dim] = query.dims();
    let value_dim = value.dims()[3];
    let decay_dim = log_decay.dims()[3];
    let dtype = value.dtype();
    let mut state = Tensor::<B, 4>::zeros([batch, heads, key_dim, value_dim], (&query.device(), DType::F32));
    let mut output = Vec::with_capacity(sequence);
    for token in 0..sequence {
        let q = query.clone().slice([0..batch, 0..heads, token..token + 1, 0..key_dim]).cast(DType::F32);
        let k = key.clone().slice([0..batch, 0..heads, token..token + 1, 0..key_dim]).cast(DType::F32);
        let v = value.clone().slice([0..batch, 0..heads, token..token + 1, 0..value_dim]).cast(DType::F32);
        let decay = log_decay.clone().slice([0..batch, 0..heads, token..token + 1, 0..decay_dim]).exp().swap_dims(2, 3);
        state = state * decay;
        let prediction = k.clone().matmul(state.clone());
        let rate = beta.clone().slice([0..batch, 0..heads, token..token + 1]).cast(DType::F32).reshape([batch, heads, 1, 1]);
        state = state + k.swap_dims(2, 3).matmul((v - prediction) * rate);
        output.push((q.mul_scalar((key_dim as f64).sqrt().recip()).matmul(state.clone())).cast(dtype));
    }
    Tensor::cat(output, 2)
}

impl<B: TrainingBackend> NativeSftModel<B> {
    pub(super) fn gated_delta(&self, input: Tensor<B, 3>, prefix: &str, c: &Qwen35TextConfig,
        adapters: Option<&NativeLora<B>>) -> Tensor<B, 3> {
        let [batch, sequence, _] = input.dims();
        let (kh, vh, kd, vd) = (c.linear_num_key_heads, c.linear_num_value_heads, c.linear_key_head_dim, c.linear_value_head_dim);
        let (keys, values) = (kh * kd, vh * vd);
        let channels = 2 * keys + values;
        let dtype = input.dtype();
        let p = format!("{prefix}.linear_attn");
        let mixed = convolution(self.project(&format!("{p}.in_proj_qkv"), input.clone(), adapters),
            self.convolution(&format!("{p}.conv1d.weight")), true);
        let q = mixed.clone().slice([0..batch, 0..sequence, 0..keys]).reshape([batch, sequence, kh, kd]).swap_dims(1, 2);
        let k = mixed.clone().slice([0..batch, 0..sequence, keys..2 * keys]).reshape([batch, sequence, kh, kd]).swap_dims(1, 2);
        let v = mixed.slice([0..batch, 0..sequence, 2 * keys..channels]).reshape([batch, sequence, vh, vd]).swap_dims(1, 2);
        let l2 = |x: Tensor<B, 4>| {
            let inverse = (x.clone().square().sum_dim(3) + 1e-6).cast(DType::F32).rsqrt().cast(dtype);
            x * inverse
        };
        let q = repeat_heads(l2(q), vh / kh);
        let k = repeat_heads(l2(k), vh / kh);
        let beta = sigmoid(self.project(&format!("{p}.in_proj_b"), input.clone(), adapters)).swap_dims(1, 2);
        let a = self.project(&format!("{p}.in_proj_a"), input.clone(), adapters).cast(DType::F32)
            + self.vector(&format!("{p}.dt_bias")).cast(DType::F32).unsqueeze::<3>();
        let decay = -(self.vector(&format!("{p}.A_log")).cast(DType::F32).exp().unsqueeze::<3>() * softplus(a));
        let y = delta_rule(q, k, v, beta, decay.swap_dims(1, 2).reshape([batch, vh, sequence, 1]))
            .swap_dims(1, 2).cast(DType::F32);
        let z = self.project(&format!("{p}.in_proj_z"), input, adapters).reshape([batch, sequence, vh, vd]).cast(DType::F32);
        let gamma = Tensor::<B, 1>::ones([vd], (&y.device(), DType::F32));
        let normalized = rms_norm(y, gamma, c.rms_norm_eps).cast(dtype);
        let norm = self.vector(&format!("{p}.norm.weight"));
        let normalized = normalized.cast(norm.dtype()) * norm.unsqueeze::<4>();
        let gated = (normalized.cast(DType::F32) * silu(z)).cast(dtype).reshape([batch, sequence, values]);
        self.project(&format!("{p}.out_proj"), gated, adapters)
    }

    pub(super) fn kda(&self, input: Tensor<B, 3>, prefix: &str, c: &KimiLinearConfig,
        adapters: Option<&NativeLora<B>>) -> Tensor<B, 3> {
        let linear = c.linear_attn_config.as_ref().expect("validated KDA configuration");
        let [batch, sequence, _] = input.dims();
        let (heads, dim) = (linear.num_heads, linear.head_dim);
        let channels = heads * dim;
        let dtype = input.dtype();
        let p = format!("{prefix}.self_attn");
        let conv = |name: &str| convolution(self.project(&format!("{p}.{name}_proj"), input.clone(), adapters),
            self.convolution(&format!("{p}.{name}_conv1d.weight")), false);
        let l2 = |x: Tensor<B, 3>| {
            let x = x.reshape([batch, sequence, heads, dim]).cast(DType::F32);
            (x.clone() * (x.square().sum_dim(3) + 1e-6).rsqrt()).cast(dtype).swap_dims(1, 2)
        };
        let q = l2(conv("q"));
        let k = l2(conv("k"));
        let v = conv("v").reshape([batch, sequence, heads, dim]).swap_dims(1, 2);
        let f = self.project(&format!("{p}.f_b_proj"), self.project(&format!("{p}.f_a_proj"), input.clone(), adapters), adapters).cast(DType::F32)
            + self.vector(&format!("{p}.dt_bias")).cast(DType::F32).unsqueeze::<3>();
        let a = self.states[self.layout.states[&format!("{p}.A_log")]].val().cast(DType::F32).exp();
        let decay = -(softplus(f).reshape([batch, sequence, heads, dim]) * a);
        let beta = sigmoid(self.project(&format!("{p}.b_proj"), input.clone(), adapters).cast(DType::F32)).swap_dims(1, 2);
        let y = delta_rule(q, k, v, beta, decay.swap_dims(1, 2)).swap_dims(1, 2).cast(DType::F32);
        let gate = sigmoid(self.project(&format!("{p}.g_b_proj"), self.project(&format!("{p}.g_a_proj"), input, adapters), adapters)
            .reshape([batch, sequence, heads, dim]).cast(DType::F32));
        let variance = y.clone().square().mean_dim(3);
        let output = (y * (variance + c.rms_norm_eps).rsqrt()
            * self.vector(&format!("{p}.o_norm.weight")).cast(DType::F32).unsqueeze::<4>() * gate)
            .cast(dtype).reshape([batch, sequence, channels]);
        self.project(&format!("{p}.o_proj"), output, adapters)
    }
}
