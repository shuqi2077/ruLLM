use super::*;
use ruda_tensor::api::{activation::{sigmoid, silu}, module::conv1d, ops::ConvOptions};
use rudnn::kda::{KdaInput, chunk_kda, chunk_kda_with_history};

pub(super) struct Kda<B: Backend> {
    pub q: Projection<B>, pub k: Projection<B>, pub v: Projection<B>,
    pub q_conv: Tensor<B, 3>, pub k_conv: Tensor<B, 3>, pub v_conv: Tensor<B, 3>,
    pub a_log: Tensor<B, 4>, pub dt_bias: Tensor<B, 1>,
    pub f_a: Projection<B>, pub f_b: Projection<B>, pub beta: Projection<B>,
    pub g_a: Projection<B>, pub g_b: Projection<B>, pub output_norm: Tensor<B, 1>, pub out: Projection<B>,
}
#[derive(Clone)]
pub(super) struct State<B: Backend> {
    pub conv: Option<[Tensor<B, 3>; 3]>, pub recurrent: Option<Tensor<B, 4>>,
}
impl<B: Backend> State<B> {
    pub(super) fn empty() -> Self { Self {conv: None, recurrent: None} }
}
pub(super) struct History<B: Backend> {pub conv: [Tensor<B, 3>; 3], pub states: Vec<Tensor<B, 4>>}

fn convolution<B: Backend>(input: Tensor<B, 3>, previous: Tensor<B, 3>, weight: Tensor<B, 3>)
    -> (Tensor<B, 3>, Tensor<B, 3>, Tensor<B, 3>) {
    let [b, s, channels] = input.dims(); let width = weight.dims()[2]; let dtype = input.dtype();
    let history = Tensor::cat(vec![previous, input.swap_dims(1, 2)], 2);
    let output = conv1d(history.clone().cast(DType::F32), weight.cast(DType::F32), None, ConvOptions::new([1], [0], [1], channels));
    let output = silu(output).cast(dtype).slice([0..b, 0..channels, 1..s + 1]).swap_dims(1, 2);
    let final_state = history.clone().slice([0..b, 0..channels, s..s + width]);
    (output, final_state, history)
}
impl<R, F, I, BT> Kda<DeviceBackend<R, F, I, BT>>
where R: DeviceRuntime, R::Server: ComputeServer, R::Device: DeviceOps, F: FloatElement, I: IntElement, BT: BoolElement {
    pub(super) fn forward(&self, input: Tensor<DeviceBackend<R, F, I, BT>, 3>, config: &KimiLinearConfig,
        state: &mut State<DeviceBackend<R, F, I, BT>>, record: bool)
        -> Result<(Tensor<DeviceBackend<R, F, I, BT>, 3>, Option<History<DeviceBackend<R, F, I, BT>>>), GenerationError> {
        let c = config.linear_attn_config.as_ref().ok_or_else(|| error("missing Kimi KDA dimensions"))?;
        let [b, s, _] = input.dims(); let (h, d, w) = (c.num_heads, c.head_dim, c.short_conv_kernel_size);
        let channels = h * d; let dtype = input.dtype(); let device = input.device();
        let previous = state.conv.clone().unwrap_or_else(|| std::array::from_fn(|_| Tensor::zeros([b, channels, w], (&device, dtype))));
        let [cq, ck, cv] = previous;
        let (q, cq, hq) = convolution(self.q.forward(input.clone()), cq, self.q_conv.clone());
        let (k, ck, hk) = convolution(self.k.forward(input.clone()), ck, self.k_conv.clone());
        let (v, cv, hv) = convolution(self.v.forward(input.clone()), cv, self.v_conv.clone());
        let l2 = |x: Tensor<DeviceBackend<R, F, I, BT>, 3>| {
            let x = x.reshape([b, s, h, d]).cast(DType::F32);
            (x.clone() * (x.square().sum_dim(3) + 1e-6).rsqrt()).cast(dtype).swap_dims(1, 2)
        };
        let g = self.f_b.forward(self.f_a.forward(input.clone())).cast(DType::F32)
            + self.dt_bias.clone().cast(DType::F32).unsqueeze::<3>();
        let softplus = g.clone().exp().log1p().mask_where(g.clone().greater_elem(20.0), g);
        let decay = -(softplus.reshape([b, s, h, d]) * self.a_log.clone().cast(DType::F32).exp());
        let beta = sigmoid(self.beta.forward(input.clone()).cast(DType::F32)).swap_dims(1, 2);
        let initial = state.recurrent.clone().unwrap_or_else(|| Tensor::zeros([b, h, d, d], (&device, DType::F32)));
        let inputs = KdaInput {query: l2(q).into_primitive().tensor(), key: l2(k).into_primitive().tensor(),
            value: v.reshape([b, s, h, d]).swap_dims(1, 2).into_primitive().tensor(),
            beta: beta.into_primitive().tensor(), log_decay: decay.swap_dims(1, 2).into_primitive().tensor(),
            initial_state: initial.into_primitive().tensor(), query_scale: (d as f32).sqrt().recip()};
        let (result, history) = if record {
            let history = chunk_kda_with_history(inputs, 64).map_err(error)?;
            (history.result, Some(History {conv: [hq, hk, hv], states: history.states.into_iter()
                .map(|x| Tensor::from_primitive(TensorPrimitive::Float(x))).collect()}))
        } else { (chunk_kda(inputs, 64, false).map_err(error)?, None) };
        state.conv = Some([cq, ck, cv]);
        state.recurrent = Some(Tensor::from_primitive(TensorPrimitive::Float(result.final_state)));
        let y = Tensor::<DeviceBackend<R, F, I, BT>, 4>::from_primitive(TensorPrimitive::Float(result.output))
            .swap_dims(1, 2).cast(DType::F32);
        let gate = sigmoid(self.g_b.forward(self.g_a.forward(input)).reshape([b, s, h, d]).cast(DType::F32));
        let variance = y.clone().square().mean_dim(3);
        let output = (y * (variance + config.rms_norm_eps).rsqrt()
            * self.output_norm.clone().cast(DType::F32).unsqueeze::<4>() * gate).cast(dtype).reshape([b, s, channels]);
        Ok((self.out.forward(output), history))
    }
}
