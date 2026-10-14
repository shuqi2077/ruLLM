use super::*;
use crate::{HuggingFaceLoadError, huggingface::checkpoint::{Checkpoint, StackedMatrixSource}};
use ruda_model::module::Param;
use ruda_tensor::api::{IndexingUpdateOp, activation::{sigmoid, softmax}};
use std::ops::Range;

pub(super) enum FeedForward<B: Backend> {
    Dense(Mlp<B>),
    Experts { router: Projection<B>, experts: Vec<Mlp<B>>, shared: Mlp<B>, shared_gate: Projection<B>, top_k: usize },
}

impl<B: Backend> FeedForward<B> {
    fn forward_with(&self, input: Tensor<B, 3>, run: impl Fn(&Mlp<B>, Tensor<B, 3>) -> Result<Tensor<B, 3>, GenerationError>)
        -> Result<Tensor<B, 3>, GenerationError> {
        let Self::Experts { router, experts, shared, shared_gate, top_k } = self else {
            let Self::Dense(mlp) = self else { unreachable!() }; return run(mlp, input);
        };
        let [batch, sequence, hidden] = input.dims();
        let tokens = batch.checked_mul(sequence).ok_or_else(|| GenerationError("Qwen3.5 MoE token count overflow".into()))?;
        let slots = tokens.checked_mul(*top_k).filter(|&n| n <= i32::MAX as usize)
            .ok_or_else(|| GenerationError("Qwen3.5 MoE assignments exceed i32 indexing".into()))?;
        let dtype = input.dtype(); let device = input.device();
        let logits = router.forward(input.clone()).reshape([tokens, experts.len()]);
        let (weights, selected) = softmax(logits.cast(DType::F32), 1).topk_with_indices(*top_k, 1);
        let weights = (weights.clone() / weights.sum_dim(1)).cast(dtype).reshape([slots, 1]);
        let selected = selected.try_into_data().map_err(|e| GenerationError(e.to_string()))?.iter::<i32>().collect::<Vec<_>>();
        if selected.len() != slots || selected.iter().any(|&id| id < 0 || id as usize >= experts.len()) {
            return Err(GenerationError("invalid Qwen3.5 MoE expert assignments".into()));
        }
        let flat = input.clone().reshape([tokens, hidden]);
        let mut routed = Tensor::<B, 2>::zeros([tokens, hidden], (&device, dtype));
        for (expert_id, expert) in experts.iter().enumerate() {
            let slots = selected.iter().enumerate().filter(|(_, id)| **id as usize == expert_id).map(|(slot, _)| slot).collect::<Vec<_>>();
            if slots.is_empty() { continue; }
            let count = slots.len();
            let rows = slots.iter().map(|&slot| (slot / *top_k) as i32).collect::<Vec<_>>();
            let rows = Tensor::<B, 1, Int>::from_data(TensorData::new(rows, [count]), &device);
            let indices = Tensor::<B, 1, Int>::from_data(TensorData::new(slots.into_iter().map(|slot| slot as i32).collect::<Vec<_>>(), [count]), &device);
            let output = run(expert, flat.clone().select(0, rows.clone()).reshape([1, count, hidden]))?
                .reshape([count, hidden]) * weights.clone().select(0, indices);
            routed = routed.select_assign(0, rows, output.cast(dtype), IndexingUpdateOp::Add);
        }
        let shared = run(shared, input.clone())? * sigmoid(shared_gate.forward(input));
        Ok(routed.reshape([batch, sequence, hidden]) + shared)
    }
}

impl<R, F, I, BT> FeedForward<DeviceBackend<R, F, I, BT>>
where R: DeviceRuntime, R::Server: ComputeServer, R::Device: DeviceOps,
    F: FloatElement, I: IntElement, BT: BoolElement {
    pub(super) fn forward(&self, input: Tensor<DeviceBackend<R, F, I, BT>, 3>)
        -> Result<Tensor<DeviceBackend<R, F, I, BT>, 3>, GenerationError> {
        self.forward_with(input, |mlp, value| mlp.forward(value))
    }
}

#[cfg(feature = "libtorch")]
impl<E: ruda_tensor_tch::TchElement> FeedForward<ruda_tensor_tch::LibTorch<E>> {
    pub(super) fn forward(&self, input: Tensor<ruda_tensor_tch::LibTorch<E>, 3>)
        -> Result<Tensor<ruda_tensor_tch::LibTorch<E>, 3>, GenerationError> {
        self.forward_with(input, |mlp, value| Ok(mlp.forward(value)))
    }
}

pub(super) trait Weights<B: Backend> {
    fn checkpoint(&mut self) -> &mut Checkpoint;
    fn device(&self) -> B::Device;
    fn fp8_block(&self) -> Option<[usize; 2]>;
    fn fp8_projection(&mut self, weight: TensorData, scales: TensorData) -> Result<Projection<B>, HuggingFaceLoadError>;
    fn mlp_projection(&mut self, prefix: &str, input: usize, output: usize, part: Option<(usize, Range<usize>)>)
        -> Result<Projection<B>, HuggingFaceLoadError>;
}

fn interval(width: usize, shard: Option<(usize, usize)>) -> Option<Range<usize>> {
    shard.map(|(rank, world)| { let size = width / world; rank * size..(rank + 1) * size })
}

fn dense<B: Backend>(weights: &mut impl Weights<B>, prefix: &str, hidden: usize, width: usize, shard: Option<(usize, usize)>)
    -> Result<Mlp<B>, HuggingFaceLoadError> {
    let range = interval(width, shard);
    Ok(Mlp {
        gate: weights.mlp_projection(&format!("{prefix}.gate_proj"), hidden, width, range.clone().map(|r| (0, r)))?,
        up: weights.mlp_projection(&format!("{prefix}.up_proj"), hidden, width, range.clone().map(|r| (0, r)))?,
        down: weights.mlp_projection(&format!("{prefix}.down_proj"), width, hidden, range.map(|r| (1, r)))?,
    })
}

pub(super) fn load<B: Backend>(weights: &mut impl Weights<B>, prefix: &str, config: &Qwen35TextConfig, shard: Option<(usize, usize)>)
    -> Result<FeedForward<B>, HuggingFaceLoadError> {
    let hidden = config.hidden_size;
    if !config.is_moe() { return dense(weights, prefix, hidden, config.intermediate_size, shard).map(FeedForward::Dense); }
    let moe = &config.moe;
    let router = weights.mlp_projection(&format!("{prefix}.gate"), hidden, moe.num_experts, None)?;
    let shared_gate = weights.mlp_projection(&format!("{prefix}.shared_expert_gate"), hidden, 1, None)?;
    let shared = dense(weights, &format!("{prefix}.shared_expert"), hidden, moe.shared_expert_intermediate_size, shard)?;
    let mut experts = Vec::with_capacity(moe.num_experts);
    let gate_up_name = format!("{prefix}.experts.gate_up_proj");
    let down_name = format!("{prefix}.experts.down_proj");
    let stacked_fp8 = weights.checkpoint().tensors.get(&gate_up_name).is_some_and(|tensor| tensor.dtype == DType::U8);
    if stacked_fp8 {
        let width = moe.moe_intermediate_size;
        let block = weights.fp8_block().ok_or_else(|| HuggingFaceLoadError("stacked FP8 experts require block configuration".into()))?;
        let mut gate_up = super::fp8_loading::Stacked::open(weights.checkpoint(), &gate_up_name, [moe.num_experts, 2 * width, hidden], block)?;
        let mut down = super::fp8_loading::Stacked::open(weights.checkpoint(), &down_name, [moe.num_experts, hidden, width], block)?;
        let range = interval(width, shard).unwrap_or(0..width);
        for expert in 0..moe.num_experts {
            let (matrix, scales) = gate_up.matrix(expert, range.clone(), 0..hidden)?;
            let gate = weights.fp8_projection(matrix, scales)?;
            let (matrix, scales) = gate_up.matrix(expert, width + range.start..width + range.end, 0..hidden)?;
            let up = weights.fp8_projection(matrix, scales)?;
            let (matrix, scales) = down.matrix(expert, 0..hidden, range.clone())?;
            let down = weights.fp8_projection(matrix, scales)?;
            experts.push(Mlp { gate, up, down });
        }
    } else if weights.checkpoint().tensors.contains_key(&gate_up_name) || weights.checkpoint().tensors.contains_key(&down_name) {
        let width = moe.moe_intermediate_size;
        let mut gate_up: StackedMatrixSource = weights.checkpoint().stacked_matrix_source(&gate_up_name, [moe.num_experts, 2 * width, hidden])?;
        let mut down: StackedMatrixSource = weights.checkpoint().stacked_matrix_source(&down_name, [moe.num_experts, hidden, width])?;
        let range = interval(width, shard).unwrap_or(0..width);
        let device = weights.device();
        let projection = |data: TensorData| {
            let dtype = data.dtype;
            Projection::Dense(Linear { weight: Param::from_tensor(Tensor::<B, 2>::from_data(data, (&device, dtype)).transpose()), bias: None })
        };
        for expert in 0..moe.num_experts {
            experts.push(Mlp {
                gate: projection(gate_up.matrix(expert, range.clone(), 0..hidden)?),
                up: projection(gate_up.matrix(expert, width + range.start..width + range.end, 0..hidden)?),
                down: projection(down.matrix(expert, 0..hidden, range.clone())?),
            });
        }
    } else {
        for expert in 0..moe.num_experts {
            experts.push(dense(weights, &format!("{prefix}.experts.{expert}"), hidden, moe.moe_intermediate_size, shard)?);
        }
    }
    Ok(FeedForward::Experts { router, experts, shared, shared_gate, top_k: moe.num_experts_per_tok })
}
