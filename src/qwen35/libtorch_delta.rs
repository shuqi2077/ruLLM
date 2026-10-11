use crate::GenerationError;
use tch::{Kind, Tensor};

pub(super) fn gated_delta(
    query: &Tensor,
    key: &Tensor,
    value: &Tensor,
    beta: &Tensor,
    decay: &Tensor,
    initial: &Tensor,
    recurrent_decode: bool,
) -> Result<(Tensor, Tensor), GenerationError> {
    let shape = query.size();
    if shape.len() != 4 || shape.iter().any(|&n| n <= 0) {
        return Err(GenerationError(
            "invalid LibTorch gated-delta query shape".into(),
        ));
    }
    let (batch, heads, sequence, kd) = (shape[0], shape[1], shape[2], shape[3]);
    let value_shape = value.size();
    if value_shape.len() != 4 || value_shape[..3] != shape[..3] || value_shape[3] <= 0 {
        return Err(GenerationError(
            "invalid LibTorch gated-delta value shape".into(),
        ));
    }
    let vd = value_shape[3];
    if key.size() != shape
        || beta.size() != [batch, heads, sequence]
        || decay.size() != [batch, heads, sequence]
        || initial.size() != [batch, heads, kd, vd]
        || !matches!(query.kind(), Kind::Float | Kind::Half | Kind::BFloat16)
        || [query, key, value, beta]
            .iter()
            .any(|t| t.kind() != query.kind() || t.device() != query.device())
        || [decay, initial]
            .iter()
            .any(|t| t.kind() != Kind::Float || t.device() != query.device())
    {
        return Err(GenerationError(
            "LibTorch gated-delta geometry/device/FP32 state mismatch".into(),
        ));
    }
    let scale = (kd as f32).sqrt().recip() as f64;
    if recurrent_decode {
        if sequence != 1 {
            return Err(GenerationError(
                "recurrent decode requires one token".into(),
            ));
        }
        let q = query.to_kind(Kind::Float).transpose(-2, -1) * scale;
        let k = key.to_kind(Kind::Float).transpose(-2, -1);
        let state = initial * decay.exp().unsqueeze(-1);
        let prediction = (&state * &k).sum_dim_intlist(&[2i64][..], true, Kind::Float);
        let residual =
            (value.to_kind(Kind::Float) - prediction) * beta.to_kind(Kind::Float).unsqueeze(-1);
        let state = state + k * residual;
        let output = (&state * q).sum_dim_intlist(&[2i64][..], true, Kind::Float);
        return Ok((output, state));
    }
    let chunk = 64i64;
    let identity = Tensor::eye(chunk, (Kind::Float, query.device()));
    let mut state = initial.shallow_clone();
    let mut outputs = Vec::with_capacity((sequence as usize).div_ceil(chunk as usize));
    for start in (0..sequence).step_by(chunk as usize) {
        let length = chunk.min(sequence - start);
        let pad = [0, 0, 0, chunk - length];
        let q = query
            .narrow(2, start, length)
            .to_kind(Kind::Float)
            .constant_pad_nd(pad)
            * scale;
        let k = key
            .narrow(2, start, length)
            .to_kind(Kind::Float)
            .constant_pad_nd(pad);
        let v = value
            .narrow(2, start, length)
            .to_kind(Kind::Float)
            .constant_pad_nd(pad);
        let b = beta
            .narrow(2, start, length)
            .to_kind(Kind::Float)
            .constant_pad_nd([0, chunk - length])
            .unsqueeze(-1);
        let cumulative = decay
            .narrow(2, start, length)
            .constant_pad_nd([0, chunk - length])
            .cumsum(2, Kind::Float);
        let exponential = cumulative.exp().unsqueeze(-1);
        let difference = cumulative.unsqueeze(-1) - cumulative.unsqueeze(-2);
        let factor = difference.tril(0).exp();
        let key_beta = &k * &b;
        let triangular = (key_beta.matmul(&k.transpose(-2, -1)) * &factor).tril(-1);
        let system = &identity + triangular;
        let rhs = Tensor::cat(&[&v * &b, key_beta * &exponential], -1);
        let solved = system
            .f_linalg_solve_triangular(&rhs, false, true, true)
            .map_err(|error| {
                GenerationError(format!("LibTorch gated-delta triangular solve: {error}"))
            })?;
        let residual = solved.narrow(-1, 0, vd) - solved.narrow(-1, vd, kd).matmul(&state);
        let attention = (q.matmul(&k.transpose(-2, -1)) * factor).tril(0);
        let output = (q * exponential).matmul(&state) + attention.matmul(&residual);
        outputs.push(output.narrow(2, 0, length));
        let last = cumulative.select(2, chunk - 1);
        let tail_key = k * (last.unsqueeze(-1) - cumulative).exp().unsqueeze(-1);
        state = state * last.exp().unsqueeze(-1).unsqueeze(-1)
            + tail_key.transpose(-2, -1).matmul(&residual);
    }
    Ok((Tensor::cat(&outputs, 2), state))
}
