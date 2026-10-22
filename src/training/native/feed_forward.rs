use super::*;
use ruda_tensor::api::{activation::{sigmoid, silu, softmax}, IndexingUpdateOp};

impl<B: Backend> NativeSftModel<B> {
    fn mlp(&self, input: Tensor<B, 3>, prefix: &str, kimi_expert: bool, adapters: Option<&NativeLora<B>>) -> Tensor<B, 3> {
        let (gate, up, down) = if kimi_expert { ("w1", "w3", "w2") } else { ("gate_proj", "up_proj", "down_proj") };
        let activated = silu(self.project(&format!("{prefix}.{gate}"), input.clone(), adapters))
            * self.project(&format!("{prefix}.{up}"), input, adapters);
        self.project(&format!("{prefix}.{down}"), activated, adapters)
    }

    pub(super) fn feed_forward(&self, input: Tensor<B, 3>, prefix: &str, layer: usize, adapters: Option<&NativeLora<B>>)
        -> Result<Tensor<B, 3>, TrainError> {
        let kimi = matches!(&self.layout.architecture, Architecture::KimiLinear(c) if c.is_moe(layer));
        let prefix = format!("{prefix}.{}", if kimi { "block_sparse_moe" } else { "mlp" });
        if !self.layout.matrices.contains_key(&format!("{prefix}.gate.weight")) {
            return Ok(self.mlp(input, &prefix, false, adapters));
        }
        let [batch, sequence, hidden] = input.dims();
        let tokens = batch.checked_mul(sequence).ok_or_else(|| TrainError("MoE token count overflow".into()))?;
        let dtype = input.dtype();
        let device = input.device();
        let (weights, selected, experts, top_k, full_precision) = match &self.layout.architecture {
            Architecture::Qwen3(_, Some(c)) => {
                let logits = self.project(&format!("{prefix}.gate"), input.clone(), None).reshape([tokens, c.num_experts]).cast(DType::F32);
                let (mut weights, selected) = softmax(logits, 1).topk_with_indices(c.num_experts_per_tok, 1);
                if c.norm_topk_prob { weights = weights.clone() / weights.sum_dim(1); }
                (weights, selected, c.num_experts, c.num_experts_per_tok, false)
            }
            Architecture::Qwen35(c) => {
                let logits = self.project(&format!("{prefix}.gate"), input.clone(), None).reshape([tokens, c.moe.num_experts]).cast(DType::F32);
                let (weights, selected) = softmax(logits, 1).topk_with_indices(c.moe.num_experts_per_tok, 1);
                let weights = weights.clone() / weights.sum_dim(1);
                (weights, selected, c.moe.num_experts, c.moe.num_experts_per_tok, false)
            }
            architecture => {
                let c = match architecture { Architecture::Mla(c) => c.clone(), Architecture::KimiLinear(c) => c.routing(), _ => unreachable!() };
                let logits = input.clone().reshape([tokens, hidden]).cast(DType::F32)
                    .matmul(self.matrix(&format!("{prefix}.gate.weight")).cast(DType::F32).transpose());
                let correction = self.layout.vectors.get(&format!("{prefix}.gate.e_score_correction_bias"))
                    .map(|&index| self.vectors[index].val());
                let (weights, selected) = crate::deepseek2::training::route(logits, correction.as_ref(), &c);
                (weights, selected, c.n_routed_experts, c.num_experts_per_tok, true)
            }
        };
        let slots = tokens.checked_mul(top_k).filter(|&n| n <= i32::MAX as usize)
            .ok_or_else(|| TrainError("MoE assignments exceed i32 indexing".into()))?;
        let selected = selected.try_into_data().map_err(|e| TrainError(e.to_string()))?.iter::<i32>().collect::<Vec<_>>();
        if selected.len() != slots || selected.iter().any(|&id| id < 0 || id as usize >= experts) {
            return Err(TrainError("invalid MoE expert assignments".into()));
        }
        let accumulation_dtype = if full_precision { DType::F32 } else { dtype };
        let weights = weights.cast(accumulation_dtype).reshape([slots, 1]);
        let flat = input.clone().reshape([tokens, hidden]);
        let mut output = Tensor::<B, 2>::zeros([tokens, hidden], (&device, accumulation_dtype));
        for expert in 0..experts {
            let assignments = selected.iter().enumerate().filter(|(_, id)| **id as usize == expert).map(|(slot, _)| slot).collect::<Vec<_>>();
            if assignments.is_empty() { continue; }
            let count = assignments.len();
            let rows = Tensor::<B, 1, Int>::from_data(TensorData::new(assignments.iter().map(|&slot| (slot / top_k) as i32).collect::<Vec<_>>(), [count]), &device);
            let indices = Tensor::<B, 1, Int>::from_data(TensorData::new(assignments.into_iter().map(|slot| slot as i32).collect::<Vec<_>>(), [count]), &device);
            let selected_input = flat.clone().select(0, rows.clone()).reshape([1, count, hidden]);
            let update = self.mlp(selected_input, &format!("{prefix}.experts.{expert}"), kimi, adapters)
                .reshape([count, hidden]).cast(accumulation_dtype) * weights.clone().select(0, indices);
            output = output.select_assign(0, rows, update, IndexingUpdateOp::Add);
        }
        let mut output = output.reshape([batch, sequence, hidden]).cast(dtype);
        if self.layout.matrices.contains_key(&format!("{prefix}.shared_expert.gate_proj.weight")) {
            output = output + self.mlp(input.clone(), &format!("{prefix}.shared_expert"), false, adapters)
                * sigmoid(self.project(&format!("{prefix}.shared_expert_gate"), input.clone(), None));
        }
        if self.layout.matrices.contains_key(&format!("{prefix}.shared_experts.gate_proj.weight")) {
            output = output + self.mlp(input, &format!("{prefix}.shared_experts"), false, adapters);
        }
        Ok(output)
    }
}
