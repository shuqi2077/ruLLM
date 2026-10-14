use super::*;

impl<B: Backend> FeedForward<B> {
    pub(super) fn run(&self, input: Tensor<B, 3>, c: &DeepseekV2Config,
        reduction: Option<&Arc<dyn Fn(Tensor<B, 3>) -> Tensor<B, 3> + Send + Sync>>)
        -> Result<Tensor<B, 3>, GenerationError> {
        let FeedForward::Experts {
            router,
            correction,
            experts,
            shared,
        } = self
        else {
            let FeedForward::Dense(mlp) = self else {
                unreachable!()
            };
            return Ok(mlp.forward(input));
        };
        let [_, sequence, hidden] = input.dims();
        let device = input.device();
        let dtype = input.dtype();
        let flat = input.clone().reshape([sequence, hidden]);
        let logits = flat
            .clone()
            .cast(DType::F32)
            .matmul(router.clone().cast(DType::F32).transpose());
        let (weights, indices) = routing::route(logits, correction.as_ref(), c);
        let ids = indices
            .try_into_data()
            .map_err(|e| GenerationError(e.to_string()))?
            .iter::<i32>()
            .collect::<Vec<_>>();
        if ids.len() != sequence * c.num_experts_per_tok
            || ids
                .iter()
                .any(|&id| id < 0 || id as usize >= c.n_routed_experts)
        {
            return Err(GenerationError(
                "DeepSeek router returned invalid expert assignments".into(),
            ));
        }
        let weights = weights.reshape([sequence * c.num_experts_per_tok, 1]);
        let mut routed = Tensor::<B, 2>::zeros([sequence, hidden], (&device, DType::F32));
        for (expert_id, expert) in experts {
            let assignments = ids
                .iter()
                .enumerate()
                .filter(|(_, id)| **id as usize == *expert_id)
                .map(|(slot, _)| slot)
                .collect::<Vec<_>>();
            if assignments.is_empty() {
                continue;
            }
            let rows = assignments
                .iter()
                .map(|&slot| (slot / c.num_experts_per_tok) as i32)
                .collect::<Vec<_>>();
            let slots = assignments
                .iter()
                .map(|&slot| i32::try_from(slot).map_err(|e| GenerationError(e.to_string())))
                .collect::<Result<Vec<_>, _>>()?;
            let count = rows.len();
            let rows = Tensor::<B, 1, Int>::from_data(TensorData::new(rows, [count]), &device);
            let slots = Tensor::<B, 1, Int>::from_data(TensorData::new(slots, [count]), &device);
            let selected = flat
                .clone()
                .select(0, rows.clone())
                .reshape([1, count, hidden]);
            let output = expert
                .forward(selected)
                .reshape([count, hidden])
                .cast(DType::F32)
                * weights.clone().select(0, slots);
            routed = routed.select_assign(0, rows, output, IndexingUpdateOp::Add);
        }
        let routed = routed.reshape([1, sequence, hidden]);
        let routed = match reduction {
            Some(reduce) => reduce(routed),
            None => routed,
        };
        let mut routed = routed.cast(dtype);
        if let Some(shared) = shared {
            routed = routed + shared.forward(input);
        }
        Ok(routed)
    }
}
