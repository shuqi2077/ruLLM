use super::*;
use ruda_tensor::api::IndexingUpdateOp;

#[derive(Clone, Debug, Deserialize)]
pub struct Qwen3MoeConfig {
    pub num_experts: usize, pub num_experts_per_tok: usize, pub moe_intermediate_size: usize,
    pub decoder_sparse_step: usize,
    #[serde(default)] pub mlp_only_layers: Vec<usize>,
    pub norm_topk_prob: bool,
}
impl Qwen3MoeConfig {
    pub(super) fn sparse(&self, layer: usize) -> bool {
        self.num_experts > 0 && !self.mlp_only_layers.contains(&layer) && (layer + 1) % self.decoder_sparse_step == 0
    }
    pub(super) fn validate(&self, config: &Qwen3Config) -> Result<(), HuggingFaceLoadError> {
        if self.decoder_sparse_step == 0 || self.num_experts > i32::MAX as usize
            || (self.num_experts > 0 && (self.num_experts_per_tok == 0 || self.num_experts_per_tok > self.num_experts || self.moe_intermediate_size == 0))
            || self.mlp_only_layers.iter().any(|&i| i >= config.num_hidden_layers)
            || self.mlp_only_layers.iter().enumerate().any(|(i, layer)| self.mlp_only_layers[..i].contains(layer))
            || self.moe_intermediate_size.checked_mul(config.hidden_size).is_none() {
            return Err(HuggingFaceLoadError("invalid Qwen3 MoE expert/routing/layer geometry".into()));
        }
        Ok(())
    }
}
pub(super) struct Mlp<B: Backend> {pub gate: Projection<B>, pub up: Projection<B>, pub down: Projection<B>}
impl<B: Backend> Mlp<B> {
    fn forward(&self, input: Tensor<B, 3>) -> Tensor<B, 3> {
        self.down.forward(silu(self.gate.forward(input.clone())) * self.up.forward(input))
    }
}
pub(super) enum FeedForward<B: Backend> {
    Dense(Mlp<B>),
    Experts {router: Projection<B>, experts: Vec<Mlp<B>>, top_k: usize, normalize: bool},
}
impl<B: Backend> FeedForward<B> {
    pub(super) fn forward(&self, input: Tensor<B, 3>) -> Result<Tensor<B, 3>, GenerationError> {
        let Self::Experts {router, experts, top_k, normalize} = self else {
            let Self::Dense(mlp) = self else { unreachable!() }; return Ok(mlp.forward(input));
        };
        let [b, s, d] = input.dims(); let tokens = b.checked_mul(s).ok_or_else(|| GenerationError("Qwen3 MoE token count overflow".into()))?;
        let slots = tokens.checked_mul(*top_k).filter(|&n| n <= i32::MAX as usize)
            .ok_or_else(|| GenerationError("Qwen3 MoE assignment count exceeds i32 indexing".into()))?;
        let dtype = input.dtype(); let device = input.device();
        let logits = router.forward(input.clone()).reshape([tokens, experts.len()]).cast(DType::F32);
        let (mut weights, selected) = softmax(logits, 1).topk_with_indices(*top_k, 1);
        if *normalize { weights = weights.clone() / weights.sum_dim(1); }
        let weights = weights.cast(dtype).reshape([slots, 1]);
        let selected = selected.try_into_data().map_err(|e| GenerationError(e.to_string()))?.iter::<i32>().collect::<Vec<_>>();
        if selected.len() != slots || selected.iter().any(|&id| id < 0 || id as usize >= experts.len()) {
            return Err(GenerationError("Qwen3 MoE router returned invalid expert assignments".into()));
        }
        let flat = input.reshape([tokens, d]); let mut result = Tensor::<B, 2>::zeros([tokens, d], (&device, dtype));
        for (expert_id, expert) in experts.iter().enumerate() {
            let assignments = selected.iter().enumerate().filter(|(_, id)| **id as usize == expert_id).map(|(slot, _)| slot).collect::<Vec<_>>();
            if assignments.is_empty() { continue; }
            let count = assignments.len();
            let rows = assignments.iter().map(|&slot| (slot / *top_k) as i32).collect::<Vec<_>>();
            let indices = assignments.into_iter().map(|slot| slot as i32).collect::<Vec<_>>();
            let rows = Tensor::<B, 1, Int>::from_data(TensorData::new(rows, [count]), &device);
            let indices = Tensor::<B, 1, Int>::from_data(TensorData::new(indices, [count]), &device);
            let hidden = flat.clone().select(0, rows.clone()).reshape([1, count, d]);
            let output = expert.forward(hidden).reshape([count, d]) * weights.clone().select(0, indices);
            result = result.select_assign(0, rows, output, IndexingUpdateOp::Add);
        }
        Ok(result.reshape([b, s, d]))
    }
}

pub(super) fn load<B: Backend>(prefix: &str, config: &Qwen3Config, moe: Option<&Qwen3MoeConfig>, layer: usize, shard: Option<&Shard>,
    projection: &mut impl FnMut(&str, usize, usize, bool, Option<(usize, std::ops::Range<usize>)>) -> Result<Projection<B>, HuggingFaceLoadError>)
    -> Result<FeedForward<B>, HuggingFaceLoadError> {
    let dense = |prefix: &str, width: usize,
        projection: &mut dyn FnMut(&str, usize, usize, bool, Option<(usize, std::ops::Range<usize>)>) -> Result<Projection<B>, HuggingFaceLoadError>|
        -> Result<Mlp<B>, HuggingFaceLoadError> {
        let range = shard.map(|s| {let local = width / s.world; s.rank * local..(s.rank + 1) * local});
        Ok(Mlp {
            gate: projection(&format!("{prefix}.gate_proj"), config.hidden_size, width, false, range.clone().map(|r| (0, r)))?,
            up: projection(&format!("{prefix}.up_proj"), config.hidden_size, width, false, range.clone().map(|r| (0, r)))?,
            down: projection(&format!("{prefix}.down_proj"), width, config.hidden_size, false, range.map(|r| (1, r)))?,
        })
    };
    if let Some(moe) = moe.filter(|m| m.sparse(layer)) {
        let router = projection(&format!("{prefix}.gate"), config.hidden_size, moe.num_experts, false, None)?;
        let mut experts = Vec::with_capacity(moe.num_experts);
        for i in 0..moe.num_experts { experts.push(dense(&format!("{prefix}.experts.{i}"), moe.moe_intermediate_size, projection)?); }
        Ok(FeedForward::Experts {router, experts, top_k: moe.num_experts_per_tok, normalize: moe.norm_topk_prob})
    } else { Ok(FeedForward::Dense(dense(prefix, config.intermediate_size, projection)?)) }
}
