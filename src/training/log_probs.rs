use ruda::runtime::server::ComputeServer;
use ruda_autodiff::{
    Autodiff,
    checkpoint::{base::Checkpointer, strategy::CheckpointStrategy},
    grads::Gradients,
    ops::{Backward, Ops, OpsKind},
};
use ruda_tensor::api::{DType, IndexingUpdateOp, Int, Tensor};
use ruda_tensor::ops::FloatTensorOps;
use ruda_tensor::{
    Backend, DeviceOps, TensorPrimitive,
    tensor::{FloatTensor, IntTensor},
};
use ruda_tensor_device::{BoolElement, DeviceBackend, DeviceRuntime, FloatElement, IntElement};

pub trait LinearLogProbsBackend: Backend {
    fn linear_log_probs(
        hidden: FloatTensor<Self>,
        weight: FloatTensor<Self>,
        targets: IntTensor<Self>,
        chunk_rows: usize,
    ) -> FloatTensor<Self>;
}

/// Selected-token log probabilities with chunked full-vocabulary projection
/// and recomputation in backward. Negative targets have zero value/gradient.
pub fn linear_log_probs<B: LinearLogProbsBackend>(
    hidden: Tensor<B, 2>,
    weight: Tensor<B, 2>,
    targets: Tensor<B, 1, Int>,
    chunk_rows: usize,
) -> Tensor<B, 1> {
    Tensor::from_primitive(TensorPrimitive::Float(B::linear_log_probs(
        hidden.into_primitive().tensor(),
        weight.into_primitive().tensor(),
        targets.into_primitive(),
        chunk_rows,
    )))
}

fn calculate<B: Backend>(
    hidden: Tensor<B, 2>,
    weight: Tensor<B, 2>,
    targets: Tensor<B, 1, Int>,
    chunk_rows: usize,
    upstream: Option<Tensor<B, 1>>,
) -> (Tensor<B, 1>, Option<(Tensor<B, 2>, Tensor<B, 2>)>) {
    let [rows, width] = hidden.dims();
    let [weight_width, vocabulary] = weight.dims();
    assert_eq!(width, weight_width);
    assert_eq!(targets.dims(), [rows]);
    if rows == 0 {
        let device = hidden.device();
        return (
            Tensor::zeros([0], (&device, DType::F32)),
            upstream.map(|_| {
                (
                    Tensor::zeros([0, width], (&device, hidden.dtype())),
                    Tensor::zeros([width, vocabulary], (&device, weight.dtype())),
                )
            }),
        );
    }
    let mask = targets
        .clone()
        .greater_equal_elem(0)
        .float()
        .cast(DType::F32);
    let safe = targets.clamp_min(0);
    let mut values = Vec::new();
    let mut dh = Vec::new();
    let mut dw = upstream
        .as_ref()
        .map(|_| Tensor::<B, 2>::zeros([width, vocabulary], (&hidden.device(), DType::F32)));
    for start in (0..rows).step_by(chunk_rows.max(1)) {
        let end = (start + chunk_rows.max(1)).min(rows);
        let count = end - start;
        let h = hidden.clone().slice([start..end, 0..width]);
        let logits = h.clone().matmul(weight.clone()).cast(DType::F32);
        let shifted = logits.clone() - logits.max_dim(1);
        let sum = shifted.clone().exp().sum_dim(1);
        let target = safe.clone().slice([start..end]).reshape([count, 1]);
        let valid = mask.clone().slice([start..end]).reshape([count, 1]);
        values.push(
            ((shifted.clone().gather(1, target.clone()) - sum.clone().log()) * valid.clone())
                .reshape([count]),
        );
        if let (Some(upstream), Some(dw)) = (upstream.as_ref(), dw.as_mut()) {
            let gradient = (shifted.exp() / sum)
                .scatter(1, target, valid.clone().neg(), IndexingUpdateOp::Add)
                .neg()
                * valid
                * upstream.clone().slice([start..end]).reshape([count, 1]);
            let gradient = gradient.cast(hidden.dtype());
            dh.push(gradient.clone().matmul(weight.clone().transpose()));
            *dw = dw.clone() + h.transpose().matmul(gradient).cast(DType::F32);
        }
    }
    (
        Tensor::cat(values, 0),
        dw.map(|dw| (Tensor::cat(dh, 0), dw.cast(weight.dtype()))),
    )
}

impl<R, F, I, BT> LinearLogProbsBackend for DeviceBackend<R, F, I, BT>
where
    R: DeviceRuntime,
    R::Server: ComputeServer,
    R::Device: DeviceOps,
    F: FloatElement,
    I: IntElement,
    BT: BoolElement,
{
    fn linear_log_probs(
        hidden: FloatTensor<Self>,
        weight: FloatTensor<Self>,
        targets: IntTensor<Self>,
        chunk_rows: usize,
    ) -> FloatTensor<Self> {
        calculate::<Self>(
            Tensor::from_primitive(TensorPrimitive::Float(hidden)),
            Tensor::from_primitive(TensorPrimitive::Float(weight)),
            Tensor::from_primitive(targets),
            chunk_rows,
            None,
        )
        .0
        .into_primitive()
        .tensor()
    }
}

#[derive(Debug)]
struct LogProbsBackward;

#[derive(Clone)]
struct LogProbsState<B: Backend> {
    hidden: FloatTensor<B>,
    weight: FloatTensor<B>,
    targets: IntTensor<B>,
    chunk_rows: usize,
}

impl<B: Backend> core::fmt::Debug for LogProbsState<B> {
    fn fmt(&self, f: &mut core::fmt::Formatter<'_>) -> core::fmt::Result {
        f.write_str("LogProbsState")
    }
}

impl<B: Backend> Backward<B, 2> for LogProbsBackward {
    type State = LogProbsState<B>;
    fn backward(self, ops: Ops<Self::State, 2>, grads: &mut Gradients, _: &mut Checkpointer) {
        let upstream =
            Tensor::from_primitive(TensorPrimitive::Float(grads.consume::<B>(&ops.node)));
        let (_, gradients) = calculate::<B>(
            Tensor::from_primitive(TensorPrimitive::Float(ops.state.hidden)),
            Tensor::from_primitive(TensorPrimitive::Float(ops.state.weight)),
            Tensor::from_primitive(ops.state.targets),
            ops.state.chunk_rows,
            Some(upstream),
        );
        let (hidden, weight) = gradients.expect("log probability gradients");
        for (parent, gradient) in ops.parents.into_iter().zip([hidden, weight]) {
            if let Some(parent) = parent {
                grads.register::<B>(parent.id, gradient.into_primitive().tensor());
            }
        }
    }
}

impl<B: Backend, C: CheckpointStrategy> LinearLogProbsBackend for Autodiff<B, C> {
    fn linear_log_probs(
        hidden: FloatTensor<Self>,
        weight: FloatTensor<Self>,
        targets: IntTensor<Self>,
        chunk_rows: usize,
    ) -> FloatTensor<Self> {
        let context = Self::float_autodiff_context(&hidden)
            .merge(Self::float_autodiff_context(&weight))
            .merge(targets.context);
        let hidden = Self::float_with_autodiff_context(hidden, context);
        let weight = Self::float_with_autodiff_context(weight, context);
        let state: LogProbsState<B> = LogProbsState {
            hidden: hidden.primitive.clone(),
            weight: weight.primitive.clone(),
            targets: targets.primitive,
            chunk_rows,
        };
        let output = calculate::<B>(
            Tensor::from_primitive(TensorPrimitive::Float(state.hidden.clone())),
            Tensor::from_primitive(TensorPrimitive::Float(state.weight.clone())),
            Tensor::from_primitive(state.targets.clone()),
            chunk_rows,
            None,
        )
        .0
        .into_primitive()
        .tensor();
        match <LogProbsBackward as Backward<B, 2>>::prepare::<C>(
            LogProbsBackward,
            [hidden.node, weight.node],
        )
        .compute_bound()
        .stateful()
        {
            OpsKind::Tracked(prep) => prep.finish(state, output),
            OpsKind::UnTracked(prep) => prep.finish(output),
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use ruda_tensor::api::activation::log_softmax;
    use ruda_tensor_host::Host;

    #[test]
    fn chunked_log_probs_and_weighted_backward_match_composed_host() {
        type B = Autodiff<Host>;
        let device = Default::default();
        let h = Tensor::<B, 2>::from_data([[0.1, 0.4], [0.3, -0.5], [0.8, 0.2]], &device)
            .require_grad();
        let w =
            Tensor::<B, 2>::from_data([[0.2, -0.7, 0.4], [0.6, 0.3, -0.2]], &device).require_grad();
        let hr = h.clone().detach().require_grad();
        let wr = w.clone().detach().require_grad();
        let target = Tensor::<B, 1, Int>::from_data([2, -100, 0], &device);
        let actual = linear_log_probs(h.clone(), w.clone(), target, 1);
        let selected = log_softmax(hr.clone().matmul(wr.clone()), 1)
            .gather(1, Tensor::<B, 2, Int>::from_data([[2], [0], [0]], &device))
            .reshape([3]);
        let expected = selected * Tensor::<B, 1>::from_data([1.0, 0.0, 1.0], &device);
        let upstream = Tensor::<B, 1>::from_data([-0.7, 3.0, 0.2], &device);
        let ga = (actual.clone() * upstream.clone()).sum().backward();
        let ge = (expected.clone() * upstream).sum().backward();
        let check = |a: Vec<f32>, b: Vec<f32>| {
            for (a, b) in a.into_iter().zip(b) {
                assert!((a - b).abs() < 1e-6, "{a} != {b}");
            }
        };
        check(
            actual.into_data().to_vec::<f32>().unwrap(),
            expected.into_data().to_vec::<f32>().unwrap(),
        );
        check(
            h.grad(&ga).unwrap().into_data().to_vec::<f32>().unwrap(),
            hr.grad(&ge).unwrap().into_data().to_vec::<f32>().unwrap(),
        );
        check(
            w.grad(&ga).unwrap().into_data().to_vec::<f32>().unwrap(),
            wr.grad(&ge).unwrap().into_data().to_vec::<f32>().unwrap(),
        );
    }
}
