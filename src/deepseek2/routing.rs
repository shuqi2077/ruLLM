use super::*;

pub(super) fn valid(c: &DeepseekV2Config) -> bool {
    if c.n_group == 0 || c.topk_group == 0 || c.topk_group > c.n_group
        || c.n_routed_experts % c.n_group != 0 {
        return false;
    }
    let per_group = c.n_routed_experts / c.n_group;
    if c.num_experts_per_tok > c.topk_group * per_group { return false; }
    match (c.model_type.as_str(), c.topk_method.as_str(), c.scoring_func.as_str()) {
        ("deepseek_v2", "greedy", "softmax") => c.n_group == 1 && c.topk_group == 1,
        ("deepseek_v2", "group_limited_greedy", "softmax") => true,
        ("deepseek_v3" | "kimi_k2", "noaux_tc", "sigmoid") => per_group >= 2,
        ("kimi_linear", "noaux_tc", "sigmoid" | "softmax") => per_group >= 2,
        _ => false,
    }
}

pub(super) fn route<B: Backend>(logits: Tensor<B, 2>, correction: Option<&Tensor<B, 1>>,
    c: &DeepseekV2Config) -> (Tensor<B, 2>, Tensor<B, 2, Int>) {
    let [tokens, experts] = logits.dims();
    let mut scores = if c.scoring_func == "sigmoid" { sigmoid(logits) } else { softmax(logits, 1) };
    if c.model_type == "kimi_linear" {
        if let Some(bias) = correction { scores = scores + bias.clone().cast(DType::F32).unsqueeze::<2>(); }
    }
    let mut choice = match correction {
        Some(bias) if c.model_type != "kimi_linear" => scores.clone() + bias.clone().cast(DType::F32).unsqueeze::<2>(),
        Some(_) => scores.clone(),
        None => scores.clone(),
    };
    if c.topk_group < c.n_group {
        let per_group = experts / c.n_group;
        let grouped = choice.clone().reshape([tokens, c.n_group, per_group]);
        let group_scores = if c.topk_method == "noaux_tc" {
            grouped.topk(2, 2).sum_dim(2)
        } else { grouped.max_dim(2) }.reshape([tokens, c.n_group]);
        let selected = group_scores.topk_with_indices(c.topk_group, 1).1;
        let groups = Tensor::<B, 1, Int>::arange(0..c.n_group as i64, &scores.device())
            .reshape([1, 1, c.n_group]);
        let mask = selected.reshape([tokens, c.topk_group, 1]).equal(groups)
            .any_dim(1).reshape([tokens, c.n_group, 1]).expand([tokens, c.n_group, per_group])
            .reshape([tokens, experts]);
        let masked = if matches!(c.model_type.as_str(), "kimi_k2" | "kimi_linear") || c.topk_method == "group_limited_greedy" {
            0.0
        } else { f32::NEG_INFINITY };
        choice = choice.mask_fill(mask.bool_not(), masked);
    }
    let indices = choice.topk_with_indices(c.num_experts_per_tok, 1).1;
    let mut weights = scores.gather(1, indices.clone());
    let normalized = c.num_experts_per_tok > 1 && c.norm_topk_prob;
    if normalized { weights = weights.clone() / (weights.sum_dim(1) + 1e-20); }
    if c.model_type != "deepseek_v2" || !normalized { weights = weights * c.routed_scaling_factor; }
    (weights, indices)
}
