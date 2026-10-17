use super::*;
use half::bf16;
use ruda_tensor::api::TensorData;
use ruda_tensor_device::cuda::{Cuda, CudaDevice};

/// Deterministic values in [-1, 1).
fn values(count: usize, seed: u64) -> Vec<f32> {
    let mut state = seed.wrapping_mul(0x9E37_79B9_7F4A_7C15) | 1;
    (0..count)
        .map(|_| {
            state ^= state << 13;
            state ^= state >> 7;
            state ^= state << 17;
            ((state >> 40) as f32 / (1u64 << 24) as f32) * 2.0 - 1.0
        })
        .collect()
}

std::thread_local! {
    /// Sliding window used by `paged_case!` and the reference (`0` = none).
    static WINDOW: std::cell::Cell<usize> = const { std::cell::Cell::new(0) };
}

fn window() -> usize {
    WINDOW.with(std::cell::Cell::get)
}

struct Case {
    head_dim: usize,
    query_heads: usize,
    kv_heads: usize,
    block_size: usize,
    sequence: usize,
    starts: Vec<usize>,
}

/// Physical page of logical block `block` for batch row `row`: scrambled so
/// that rows interleave inside the pool.
fn page_of(row: usize, block: usize, rows: usize) -> usize {
    // 5 is coprime with rows * 64 for rows < 5, so this is injective.
    (block * rows + row) * 5 % (rows * 64) + 1
}

fn reference(
    case: &Case,
    query: &[f32],
    key_pool: &[f32],
    value_pool: &[f32],
    table: &[u32],
    stride: usize,
) -> Vec<f32> {
    let [d, hq, hkv, s] = [case.head_dim, case.query_heads, case.kv_heads, case.sequence];
    let group = hq / hkv;
    let scale = 1.0 / (d as f64).sqrt();
    let mut output = vec![0.0f32; case.starts.len() * s * hq * d];
    for (b, &start) in case.starts.iter().enumerate() {
        for t in 0..s {
            for h in 0..hq {
                let kv = h / group;
                let limit = start + t + 1;
                let low = if window() > 0 { limit.saturating_sub(window()) } else { 0 };
                let q = &query[((b * s + t) * hq + h) * d..][..d];
                let address = |position: usize| {
                    let page = table[b * stride + position / case.block_size] as usize;
                    ((page * hkv + kv) * case.block_size + position % case.block_size) * d
                };
                let scores: Vec<f64> = (low..limit)
                    .map(|p| {
                        let k = &key_pool[address(p)..][..d];
                        q.iter().zip(k).map(|(&a, &b)| a as f64 * b as f64).sum::<f64>() * scale
                    })
                    .collect();
                let max = scores.iter().copied().fold(f64::NEG_INFINITY, f64::max);
                let weights: Vec<f64> = scores.iter().map(|s| (s - max).exp()).collect();
                let total: f64 = weights.iter().sum();
                let out = &mut output[((b * s + t) * hq + h) * d..][..d];
                for (offset, w) in weights.iter().enumerate() {
                    let v = &value_pool[address(low + offset)..][..d];
                    for (o, &x) in out.iter_mut().zip(v) {
                        *o += (w / total * x as f64) as f32;
                    }
                }
            }
        }
    }
    output
}

macro_rules! paged_case {
    ($backend:ty, $element:ty, $case:expr, $dtype:expr, $tolerance:expr, $round:expr) => {{
        type TB = $backend;
        let case: Case = $case;
        let device = CudaDevice::default();
        let rows = case.starts.len();
        let [d, hq, hkv, s] = [case.head_dim, case.query_heads, case.kv_heads, case.sequence];
        let contexts: Vec<usize> = case.starts.iter().map(|&start| start + s).collect();
        let stride = contexts.iter().map(|c| c.div_ceil(case.block_size)).max().unwrap();
        let pages = rows * 64 + 2;
        let mut table = vec![0u32; rows * stride];
        for row in 0..rows {
            for block in 0..contexts[row].div_ceil(case.block_size) {
                table[row * stride + block] = page_of(row, block, rows) as u32;
            }
        }
        let round = $round;
        let pool_len = pages * hkv * case.block_size * d;
        let query: Vec<f32> = values(rows * s * hq * d, 1).into_iter().map(round).collect();
        let key_pool: Vec<f32> = values(pool_len, 2).into_iter().map(round).collect();
        let value_pool: Vec<f32> = values(pool_len, 3).into_iter().map(round).collect();
        let to_tensor = |data: &[f32], shape: [usize; 4]| {
            let typed: Vec<$element> = data.iter().map(|&x| <$element>::from_f32(x)).collect();
            Tensor::<TB, 4>::from_data(TensorData::new(typed, shape), (&device, $dtype))
        };
        let pool_shape = [pages, hkv, case.block_size, d];
        let key = to_tensor(&key_pool, pool_shape).into_primitive().tensor();
        let value = to_tensor(&value_pool, pool_shape).into_primitive().tensor();
        let q = to_tensor(&query, [rows, s, hq, d]);
        let starts: Vec<u32> = case.starts.iter().map(|&x| x as u32).collect();
        let context_u32: Vec<u32> = contexts.iter().map(|&x| x as u32).collect();
        let batch = PagedAttentionBatch::upload(
            &key.client, &key.device, &table, stride, &starts, &context_u32,
            case.block_size, s,
        ).unwrap();
        let actual: Vec<f32> = paged_attention(q, &key, &value, &batch, hkv, window())
            .into_data()
            .to_vec::<$element>()
            .unwrap()
            .into_iter()
            .map(|x| x.to_f32())
            .collect();
        let expected = reference(&case, &query, &key_pool, &value_pool, &table, stride);
        let mut worst = 0.0f32;
        for (index, (&a, &e)) in actual.iter().zip(&expected).enumerate() {
            assert!(a.is_finite(), "non-finite output at {index}");
            worst = worst.max((a - e).abs());
            assert!(
                (a - e).abs() <= $tolerance,
                "d={d} hq={hq} hkv={hkv} s={s} starts={:?}: [{index}] {a} != {e}",
                case.starts
            );
        }
        eprintln!(
            "paged attention {} d={d} hq={hq} hkv={hkv} bs={} s={s} starts={:?} max_abs_err={worst:e}",
            stringify!($element), case.block_size, case.starts
        );
    }};
}

trait FromF32 {
    fn from_f32(x: f32) -> Self;
    fn to_f32(self) -> f32;
}
impl FromF32 for f32 {
    fn from_f32(x: f32) -> Self {
        x
    }
    fn to_f32(self) -> f32 {
        self
    }
}
impl FromF32 for half::f16 {
    fn from_f32(x: f32) -> Self {
        half::f16::from_f32(x)
    }
    fn to_f32(self) -> f32 {
        half::f16::to_f32(self)
    }
}
impl FromF32 for bf16 {
    fn from_f32(x: f32) -> Self {
        bf16::from_f32(x)
    }
    fn to_f32(self) -> f32 {
        bf16::to_f32(self)
    }
}

fn cases() -> Vec<Case> {
    vec![
        // Decode, Qwen2.5-0.5B geometry (group 7), uneven contexts across pages.
        Case { head_dim: 64, query_heads: 14, kv_heads: 2, block_size: 16, sequence: 1, starts: vec![0, 15, 40] },
        // Prefill from zero with a tail tile, MHA.
        Case { head_dim: 64, query_heads: 4, kv_heads: 4, block_size: 16, sequence: 37, starts: vec![0, 0] },
        // Chunked/prefix continuation with GQA and a wider head.
        Case { head_dim: 128, query_heads: 8, kv_heads: 2, block_size: 32, sequence: 21, starts: vec![5, 70] },
        // Several rows per warp across two warps of a block.
        Case { head_dim: 64, query_heads: 8, kv_heads: 1, block_size: 16, sequence: 9, starts: vec![3] },
    ]
}

#[test]
fn paged_attention_tensor_core_bf16_matches_reference() {
    for case in cases() {
        paged_case!(Cuda<bf16, i32>, bf16, case, DType::BF16, 2.0e-2, |x: f32| bf16::from_f32(x).to_f32());
    }
}

/// FP16 WMMA path: the only tensor-core type on Turing (sm_75), where bf16
/// falls back to the scalar kernel.
#[test]
fn paged_attention_tensor_core_f16_matches_reference() {
    use half::f16;
    for case in cases() {
        paged_case!(Cuda<f16, i32>, f16, case, DType::F16, 4.0e-3, |x: f32| f16::from_f32(x).to_f32());
    }
    WINDOW.with(|cell| cell.set(5));
    let case = Case { head_dim: 64, query_heads: 4, kv_heads: 2, block_size: 16, sequence: 23, starts: vec![0, 9] };
    paged_case!(Cuda<f16, i32>, f16, case, DType::F16, 4.0e-3, |x: f32| f16::from_f32(x).to_f32());
    WINDOW.with(|cell| cell.set(0));
}

/// Sliding windows in both kernels against the windowed reference.
#[test]
fn paged_attention_sliding_window_matches_reference() {
    for window in [1usize, 5, 17] {
        let case = Case { head_dim: 64, query_heads: 4, kv_heads: 2, block_size: 16, sequence: 23, starts: vec![0, 9] };
        WINDOW.with(|cell| cell.set(window));
        paged_case!(Cuda<bf16, i32>, bf16, case, DType::BF16, 2.0e-2, |x: f32| bf16::from_f32(x).to_f32());
        let case = Case { head_dim: 40, query_heads: 4, kv_heads: 2, block_size: 8, sequence: 23, starts: vec![0, 9] };
        paged_case!(Cuda<f32, i32>, f32, case, DType::F32, 2.0e-5, |x: f32| x);
        WINDOW.with(|cell| cell.set(0));
    }
}

#[test]
fn paged_attention_scalar_f32_matches_reference() {
    let mut all = cases();
    // Head width that is not a multiple of 16.
    all.push(Case { head_dim: 40, query_heads: 6, kv_heads: 2, block_size: 8, sequence: 5, starts: vec![0, 11] });
    for case in all {
        paged_case!(Cuda<f32, i32>, f32, case, DType::F32, 2.0e-5, |x: f32| x);
    }
}

#[test]
fn paged_attention_scalar_bf16_odd_head_matches_reference() {
    let case = Case { head_dim: 40, query_heads: 6, kv_heads: 2, block_size: 8, sequence: 5, starts: vec![0, 11] };
    paged_case!(Cuda<bf16, i32>, bf16, case, DType::BF16, 2.0e-2, |x: f32| bf16::from_f32(x).to_f32());
}

/// RoPE + paged scatter against a host computation for both rotary layouts.
#[test]
fn paged_qkv_rope_writes_rotated_query_and_paged_kv() {
    type TB = Cuda<f32, i32>;
    let device = CudaDevice::default();
    let [rows, s, hq, hkv, d, block_size] = [2usize, 5, 4, 2, 8, 4];
    let starts = [3usize, 0];
    let max_positions = 16;
    let width = (hq + 2 * hkv) * d;
    let projected = values(rows * s * width, 7);
    let angles: Vec<f32> = (0..max_positions * d)
        .map(|i| (i / d) as f32 * 0.1 + (i % d) as f32 * 0.03)
        .collect();
    let mut frequencies = Vec::with_capacity(max_positions * d * 2);
    for angle in &angles {
        frequencies.push(angle.cos());
        frequencies.push(angle.sin());
    }
    let stride = 3usize;
    let table: Vec<u32> = vec![4, 1, 6, 2, 0, 5];
    let pages = 8;
    for half_split in [true, false] {
        let projected_tensor = Tensor::<TB, 3>::from_data(
            TensorData::new(projected.clone(), [rows, s, width]), (&device, DType::F32));
        let frequency_tensor = Tensor::<TB, 3>::from_data(
            TensorData::new(frequencies.clone(), [max_positions, d, 2]), (&device, DType::F32));
        let pool = || Tensor::<TB, 4>::zeros([pages, hkv, block_size, d], (&device, DType::F32))
            .into_primitive().tensor();
        let (key_pool, value_pool) = (pool(), pool());
        let starts_u32: Vec<u32> = starts.iter().map(|&x| x as u32).collect();
        let contexts: Vec<u32> = starts.iter().map(|&x| (x + s) as u32).collect();
        let batch = PagedAttentionBatch::upload(
            &key_pool.client, &key_pool.device, &table, stride, &starts_u32, &contexts, block_size, s,
        ).unwrap();
        let query = paged_qkv_rope(projected_tensor, frequency_tensor, &key_pool, &value_pool,
            &batch, hq, hkv, d, half_split, false)
            .into_data().to_vec::<f32>().unwrap();
        let read = |tensor: &RudaTensor<_>| {
            Tensor::<TB, 4>::from_primitive(TensorPrimitive::Float(tensor.clone()))
                .into_data().to_vec::<f32>().unwrap()
        };
        let (keys, values_out) = (read(&key_pool), read(&value_pool));
        let rotate = |row: &[f32], base: usize, dim: usize, position: usize| -> f32 {
            let (pair, frequency, negate) = if half_split {
                let half = d / 2;
                if dim < half { (dim + half, 2 * dim, true) } else { (dim - half, 2 * (dim - half), false) }
            } else if dim % 2 == 0 { (dim + 1, dim, true) } else { (dim - 1, dim, false) };
            let rotated = if negate { -row[base + pair] } else { row[base + pair] };
            let index = (position * d + frequency) * 2;
            row[base + dim] * frequencies[index] + rotated * frequencies[index + 1]
        };
        for b in 0..rows {
            for t in 0..s {
                let row = &projected[(b * s + t) * width..][..width];
                let position = starts[b] + t;
                for h in 0..hq {
                    for dim in 0..d {
                        let expected = rotate(row, h * d, dim, position);
                        let actual = query[((b * s + t) * hq + h) * d + dim];
                        assert!((actual - expected).abs() < 1e-5, "q half_split={half_split}");
                    }
                }
                let page = table[b * stride + position / block_size] as usize;
                for h in 0..hkv {
                    let slot = ((page * hkv + h) * block_size + position % block_size) * d;
                    for dim in 0..d {
                        let expected = rotate(row, hq * d + h * d, dim, position);
                        assert!((keys[slot + dim] - expected).abs() < 1e-5, "k half_split={half_split}");
                        let expected_value = row[(hq + hkv) * d + h * d + dim];
                        assert_eq!(values_out[slot + dim], expected_value, "v half_split={half_split}");
                    }
                }
            }
        }
    }
}
