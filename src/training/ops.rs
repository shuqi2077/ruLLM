//! Training operators with hand-written backward passes:
//!
//! - [`flash_attention`]: causal (optionally windowed) GQA attention whose
//!   forward is the tiled online-softmax kernel and whose backward
//!   recomputes probabilities from the saved log-sum-exp (O(S) memory).
//! - [`linear_cross_entropy`]: LM-head projection fused with token
//!   cross-entropy, processed in row chunks so the full `[tokens, vocab]`
//!   logits are never resident.

use crate::paged_attention::{
    KvLayout, PagedAttentionBatch, attention_backward_raw, attention_forward_raw,
};
use ruda::runtime::server::ComputeServer;
use ruda_autodiff::Autodiff;
use ruda_autodiff::checkpoint::{base::Checkpointer, strategy::CheckpointStrategy};
use ruda_autodiff::grads::Gradients;
use ruda_autodiff::ops::{Backward, Ops, OpsKind};
use ruda_kernel::tensor::RudaTensor;
use ruda_kernel::tensor::contiguous::into_contiguous;
use ruda_tensor::api::{DType, IndexingUpdateOp, Int, Tensor};
use ruda_tensor::tensor::{FloatTensor, IntTensor};
use ruda_tensor::{Backend, DeviceOps, TensorPrimitive};
use ruda_tensor_device::{BoolElement, DeviceBackend, DeviceRuntime, FloatElement, IntElement};

/// Backends that provide [`flash_attention`].
pub trait FlashAttentionBackend: Backend {
    /// `query [B, S, Hq, D]`, `key/value [B, S, Hkv, D]` from position zero;
    /// returns `[B, S, Hq, D]`. `window == 0` is plain causal attention.
    fn flash_attention(
        query: FloatTensor<Self>,
        key: FloatTensor<Self>,
        value: FloatTensor<Self>,
        window: usize,
    ) -> FloatTensor<Self>;
}

/// Backends that provide [`linear_cross_entropy`].
pub trait LinearCrossEntropyBackend: Backend {
    /// Mean token cross-entropy of `hidden [N, d] @ weight [d, V]` against
    /// `targets [N]`; negative targets are ignored. The sum is divided by
    /// `normalizer` (the valid-token count of the whole optimizer step).
    fn linear_cross_entropy(
        hidden: FloatTensor<Self>,
        weight: FloatTensor<Self>,
        targets: IntTensor<Self>,
        normalizer: usize,
        chunk_rows: usize,
    ) -> FloatTensor<Self>;
}

/// Everything the Llama/Qwen training forward needs.
pub trait TrainingBackend: FlashAttentionBackend + LinearCrossEntropyBackend {}

impl<B: FlashAttentionBackend + LinearCrossEntropyBackend> TrainingBackend for B {}

/// Tensor-level [`FlashAttentionBackend::flash_attention`].
pub fn flash_attention<B: FlashAttentionBackend>(
    query: Tensor<B, 4>,
    key: Tensor<B, 4>,
    value: Tensor<B, 4>,
    window: usize,
) -> Tensor<B, 4> {
    let output = B::flash_attention(
        query.into_primitive().tensor(),
        key.into_primitive().tensor(),
        value.into_primitive().tensor(),
        window,
    );
    Tensor::from_primitive(TensorPrimitive::Float(output))
}

/// Tensor-level [`LinearCrossEntropyBackend::linear_cross_entropy`]; returns
/// a one-element FP32 loss.
pub fn linear_cross_entropy<B: LinearCrossEntropyBackend>(
    hidden: Tensor<B, 2>,
    weight: Tensor<B, 2>,
    targets: Tensor<B, 1, Int>,
    normalizer: usize,
    chunk_rows: usize,
) -> Tensor<B, 1> {
    let loss = B::linear_cross_entropy(
        hidden.into_primitive().tensor(),
        weight.into_primitive().tensor(),
        targets.into_primitive(),
        normalizer,
        chunk_rows,
    );
    Tensor::from_primitive(TensorPrimitive::Float(loss))
}

/// Rows per cross-entropy chunk so one FP32 `[rows, vocab]` block stays near
/// `budget_bytes`.
pub fn cross_entropy_chunk_rows(vocabulary: usize, budget_bytes: usize) -> usize {
    (budget_bytes / (vocabulary.max(1) * 4)).max(1)
}

fn dense_attention_forward<R: DeviceRuntime>(
    query: RudaTensor<R>,
    key: RudaTensor<R>,
    value: RudaTensor<R>,
    window: usize,
) -> (RudaTensor<R>, RudaTensor<R>) {
    let [batch, sequence, _, head_dim] = query.meta.shape().dims::<4>();
    let [key_batch, key_sequence, kv_heads, key_dim] = key.meta.shape().dims::<4>();
    assert_eq!([key_batch, key_sequence, key_dim], [batch, sequence, head_dim]);
    let key = into_contiguous(key);
    let value = into_contiguous(value);
    let metadata =
        PagedAttentionBatch::dense(&query.client, &query.device, batch, 0, sequence, sequence);
    let layout = KvLayout::sequence_major(sequence, kv_heads, head_dim);
    attention_forward_raw(query, &key, &value, &metadata, layout, kv_heads, window)
}

impl<R, F, I, BT> FlashAttentionBackend for DeviceBackend<R, F, I, BT>
where
    R: DeviceRuntime,
    R::Server: ComputeServer,
    R::Device: DeviceOps,
    F: FloatElement,
    I: IntElement,
    BT: BoolElement,
{
    fn flash_attention(
        query: FloatTensor<Self>,
        key: FloatTensor<Self>,
        value: FloatTensor<Self>,
        window: usize,
    ) -> FloatTensor<Self> {
        dense_attention_forward(query, key, value, window).0
    }
}

struct FlashAttentionState<R: DeviceRuntime> {
    query: RudaTensor<R>,
    key: RudaTensor<R>,
    value: RudaTensor<R>,
    output: RudaTensor<R>,
    lse: RudaTensor<R>,
    window: usize,
}

impl<R: DeviceRuntime> Clone for FlashAttentionState<R> {
    fn clone(&self) -> Self {
        Self {
            query: self.query.clone(),
            key: self.key.clone(),
            value: self.value.clone(),
            output: self.output.clone(),
            lse: self.lse.clone(),
            window: self.window,
        }
    }
}

impl<R: DeviceRuntime> core::fmt::Debug for FlashAttentionState<R> {
    fn fmt(&self, formatter: &mut core::fmt::Formatter<'_>) -> core::fmt::Result {
        formatter
            .debug_struct("FlashAttentionState")
            .field("query", &self.query.meta.shape())
            .field("key", &self.key.meta.shape())
            .field("window", &self.window)
            .finish_non_exhaustive()
    }
}

#[derive(Debug)]
struct FlashAttentionBackward;

impl<R, F, I, BT> Backward<DeviceBackend<R, F, I, BT>, 3> for FlashAttentionBackward
where
    R: DeviceRuntime,
    R::Server: ComputeServer,
    R::Device: DeviceOps,
    F: FloatElement,
    I: IntElement,
    BT: BoolElement,
{
    type State = FlashAttentionState<R>;

    fn backward(
        self,
        ops: Ops<Self::State, 3>,
        grads: &mut Gradients,
        _checkpointer: &mut Checkpointer,
    ) {
        let state = ops.state;
        let grad_output = grads.consume::<DeviceBackend<R, F, I, BT>>(&ops.node);
        let (grad_query, grad_key, grad_value) = attention_backward_raw(
            state.query,
            state.key,
            state.value,
            state.output,
            state.lse,
            grad_output,
            state.window,
        );
        let [query, key, value] = ops.parents;
        for (parent, gradient) in [(query, grad_query), (key, grad_key), (value, grad_value)] {
            if let Some(parent) = parent {
                grads.register::<DeviceBackend<R, F, I, BT>>(parent.id, gradient);
            }
        }
    }
}

impl<R, F, I, BT, C> FlashAttentionBackend for Autodiff<DeviceBackend<R, F, I, BT>, C>
where
    R: DeviceRuntime,
    R::Server: ComputeServer,
    R::Device: DeviceOps,
    F: FloatElement,
    I: IntElement,
    BT: BoolElement,
    C: CheckpointStrategy,
{
    fn flash_attention(
        query: FloatTensor<Self>,
        key: FloatTensor<Self>,
        value: FloatTensor<Self>,
        window: usize,
    ) -> FloatTensor<Self> {
        let (output, lse) = dense_attention_forward(
            query.primitive.clone(),
            key.primitive.clone(),
            value.primitive.clone(),
            window,
        );
        let prepared = <FlashAttentionBackward as Backward<DeviceBackend<R, F, I, BT>, 3>>::prepare::<C>(
            FlashAttentionBackward,
            [query.node.clone(), key.node.clone(), value.node.clone()],
        )
        .compute_bound()
        .stateful();
        match prepared {
            OpsKind::Tracked(prep) => {
                let state = FlashAttentionState {
                    query: query.primitive,
                    key: key.primitive,
                    value: value.primitive,
                    output: output.clone(),
                    lse,
                    window,
                };
                prep.finish(state, output)
            }
            OpsKind::UnTracked(prep) => prep.finish(output),
        }
    }
}

/// Chunked loss and, when requested, `(d hidden, d weight)` for a unit
/// upstream gradient. Logits, log-sum-exp and probabilities are FP32.
fn chunked_cross_entropy<B: Backend>(
    hidden: Tensor<B, 2>,
    weight: Tensor<B, 2>,
    targets: Tensor<B, 1, Int>,
    normalizer: usize,
    chunk_rows: usize,
    with_gradients: bool,
) -> (Tensor<B, 1>, Option<(Tensor<B, 2>, Tensor<B, 2>)>) {
    let [rows, width] = hidden.dims();
    let [weight_width, vocabulary] = weight.dims();
    assert_eq!(width, weight_width, "LM head width mismatch");
    assert_eq!(targets.dims(), [rows], "one target per hidden row");
    let device = hidden.device();
    let hidden_dtype = hidden.dtype();
    let weight_dtype = weight.dtype();
    let scale = 1.0 / normalizer.max(1) as f32;
    let valid = targets.clone().greater_equal_elem(0);
    let mask = valid.float().cast(DType::F32);
    let safe_targets = targets.clamp_min(0);
    let chunk_rows = chunk_rows.max(1);
    let mut loss = Tensor::<B, 1>::zeros([1], (&device, DType::F32));
    let mut grad_hidden = Vec::new();
    let mut grad_weight = with_gradients
        .then(|| Tensor::<B, 2>::zeros([width, vocabulary], (&device, DType::F32)));
    let mut start = 0;
    while start < rows {
        let end = (start + chunk_rows).min(rows);
        let count = end - start;
        let hidden_chunk = hidden.clone().slice([start..end, 0..width]);
        let logits = hidden_chunk.clone().matmul(weight.clone()).cast(DType::F32);
        let maximum = logits.clone().max_dim(1);
        let shifted = logits - maximum;
        let sum_exp = shifted.clone().exp().sum_dim(1);
        let chunk_targets = safe_targets.clone().slice([start..end]).reshape([count, 1]);
        let chunk_mask = mask.clone().slice([start..end]).reshape([count, 1]);
        let picked = shifted.clone().gather(1, chunk_targets.clone());
        let row_loss = (sum_exp.clone().log() - picked) * chunk_mask.clone();
        loss = loss + row_loss.sum();
        if let Some(grad_weight) = grad_weight.as_mut() {
            let probabilities = shifted.exp() / sum_exp;
            let gradient = probabilities
                .scatter(1, chunk_targets, chunk_mask.clone().neg(), IndexingUpdateOp::Add)
                * chunk_mask;
            let gradient = gradient.mul_scalar(scale).cast(hidden_dtype);
            grad_hidden.push(gradient.clone().matmul(weight.clone().transpose()));
            *grad_weight = grad_weight.clone()
                + hidden_chunk.transpose().matmul(gradient).cast(DType::F32);
        }
        start = end;
    }
    let gradients = grad_weight.map(|grad_weight| {
        (Tensor::cat(grad_hidden, 0), grad_weight.cast(weight_dtype))
    });
    (loss.mul_scalar(scale), gradients)
}

impl<R, F, I, BT> LinearCrossEntropyBackend for DeviceBackend<R, F, I, BT>
where
    R: DeviceRuntime,
    R::Server: ComputeServer,
    R::Device: DeviceOps,
    F: FloatElement,
    I: IntElement,
    BT: BoolElement,
{
    fn linear_cross_entropy(
        hidden: FloatTensor<Self>,
        weight: FloatTensor<Self>,
        targets: IntTensor<Self>,
        normalizer: usize,
        chunk_rows: usize,
    ) -> FloatTensor<Self> {
        let (loss, _) = chunked_cross_entropy::<Self>(
            Tensor::from_primitive(TensorPrimitive::Float(hidden)),
            Tensor::from_primitive(TensorPrimitive::Float(weight)),
            Tensor::from_primitive(targets),
            normalizer,
            chunk_rows,
            false,
        );
        loss.into_primitive().tensor()
    }
}

struct CrossEntropyState<B: Backend> {
    grad_hidden: FloatTensor<B>,
    grad_weight: FloatTensor<B>,
}

impl<B: Backend> Clone for CrossEntropyState<B> {
    fn clone(&self) -> Self {
        Self {
            grad_hidden: self.grad_hidden.clone(),
            grad_weight: self.grad_weight.clone(),
        }
    }
}

impl<B: Backend> core::fmt::Debug for CrossEntropyState<B> {
    fn fmt(&self, formatter: &mut core::fmt::Formatter<'_>) -> core::fmt::Result {
        formatter.write_str("CrossEntropyState")
    }
}

#[derive(Debug)]
struct LinearCrossEntropyBackward;

impl<B: Backend> Backward<B, 2> for LinearCrossEntropyBackward {
    type State = CrossEntropyState<B>;

    fn backward(
        self,
        ops: Ops<Self::State, 2>,
        grads: &mut Gradients,
        _checkpointer: &mut Checkpointer,
    ) {
        let upstream = Tensor::<B, 1>::from_primitive(TensorPrimitive::Float(
            grads.consume::<B>(&ops.node),
        ));
        let [hidden, weight] = ops.parents;
        let scaled = |gradient: FloatTensor<B>| {
            let gradient = Tensor::<B, 2>::from_primitive(TensorPrimitive::Float(gradient));
            let factor = upstream.clone().cast(gradient.dtype()).reshape([1, 1]);
            (gradient * factor).into_primitive().tensor()
        };
        if let Some(hidden) = hidden {
            grads.register::<B>(hidden.id, scaled(ops.state.grad_hidden.clone()));
        }
        if let Some(weight) = weight {
            grads.register::<B>(weight.id, scaled(ops.state.grad_weight));
        }
    }
}

impl<B: Backend, C: CheckpointStrategy> LinearCrossEntropyBackend for Autodiff<B, C> {
    fn linear_cross_entropy(
        hidden: FloatTensor<Self>,
        weight: FloatTensor<Self>,
        targets: IntTensor<Self>,
        normalizer: usize,
        chunk_rows: usize,
    ) -> FloatTensor<Self> {
        let inner = |with_gradients| {
            chunked_cross_entropy::<B>(
                Tensor::from_primitive(TensorPrimitive::Float(hidden.primitive.clone())),
                Tensor::from_primitive(TensorPrimitive::Float(weight.primitive.clone())),
                Tensor::from_primitive(targets.clone()),
                normalizer,
                chunk_rows,
                with_gradients,
            )
        };
        let prepared = <LinearCrossEntropyBackward as Backward<B, 2>>::prepare::<C>(
            LinearCrossEntropyBackward,
            [hidden.node.clone(), weight.node.clone()],
        )
        .compute_bound()
        .stateful();
        match prepared {
            OpsKind::Tracked(prep) => {
                let (loss, gradients) = inner(true);
                let (grad_hidden, grad_weight) = gradients.expect("gradients were requested");
                let state = CrossEntropyState {
                    grad_hidden: grad_hidden.into_primitive().tensor(),
                    grad_weight: grad_weight.into_primitive().tensor(),
                };
                prep.finish(state, loss.into_primitive().tensor())
            }
            OpsKind::UnTracked(prep) => prep.finish(inner(false).0.into_primitive().tensor()),
        }
    }
}
