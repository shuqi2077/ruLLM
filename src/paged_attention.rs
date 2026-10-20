//! Paged-KV attention kernels for the CUDA (and any Ruda device) inference path.
//!
//! KV pools use the layout `[num_pages, kv_heads, block_size, head_dim]`. A
//! dense per-batch cache `[batch, kv_heads, capacity, head_dim]` is the same
//! layout with one page per batch row and `block_size == capacity`, so the
//! dense prefill path reuses these kernels through an identity block table.
//!
//! Query rows are arranged per `(batch, kv_head)` as `row = token * group +
//! head_in_group`, where `group = query_heads / kv_heads`. All query heads that
//! share a KV head therefore share every K/V tile loaded into shared memory,
//! and single-token decode still fills `group` rows of a 16-row MMA tile.
//! Row `row` may attend to keys `< start_position + token + 1` (causal), which
//! covers full prefill, chunked/prefix continuation and decode in one kernel.

#![allow(clippy::too_many_arguments)]

use ruda::runtime::server::ComputeServer;
use ruda_core::ir::features::MmaConfig;
use ruda_kernel::dsl as kernel_dsl;
use ruda_kernel::dsl::calculate_ruda_count_elemwise;
use ruda_kernel::dsl::client::ComputeClient;
use ruda_kernel::dsl::prelude::*;
use ruda_kernel::tensor::RudaTensor;
use ruda_kernel::tensor::allocation::empty_device_contiguous_dtype;
use ruda_kernel::tensor::contiguous::into_contiguous;
use ruda_core::tensor::{DType, Shape};
use ruda_tensor::api::Tensor;
use ruda_tensor::{DeviceOps, TensorPrimitive};
use ruda_tensor_device::{BoolElement, DeviceBackend, DeviceRuntime, FloatElement, IntElement};

/// Rows of one tensor-core tile; also the number of keys per K/V tile.
const TILE: usize = 16;
/// Padding (in elements) appended to every shared-memory row of Q/K/V tiles.
/// Keeps WMMA leading dimensions a multiple of 8 while breaking the 128-byte
/// bank period of power-of-two head dimensions.
const ROW_PADDING: usize = 8;
const MAX_WARPS: usize = 4;

/// Element strides of K/V addressed through a block table:
/// `page * page_stride + kv_head * head_stride + slot * slot_stride + dim`.
/// Rows may also use a sliding window: keys `[limit - window, limit)`.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(crate) struct KvLayout {
    pub page_stride: usize,
    pub head_stride: usize,
    pub slot_stride: usize,
    pub block_size: usize,
}

impl KvLayout {
    /// Pools `[num_pages, kv_heads, block_size, head_dim]`; also a dense
    /// cache `[batch, kv_heads, capacity, head_dim]` with one page per row.
    pub const fn pool(kv_heads: usize, block_size: usize, head_dim: usize) -> Self {
        Self {
            page_stride: kv_heads * block_size * head_dim,
            head_stride: block_size * head_dim,
            slot_stride: head_dim,
            block_size,
        }
    }

    /// Training activations `[batch, sequence, kv_heads, head_dim]`.
    pub const fn sequence_major(sequence: usize, kv_heads: usize, head_dim: usize) -> Self {
        Self {
            page_stride: sequence * kv_heads * head_dim,
            head_stride: head_dim,
            slot_stride: kv_heads * head_dim,
            block_size: sequence,
        }
    }
}

/// First key a row may attend to under a sliding window (`0` = unbounded).
#[ruda]
fn window_start(limit: usize, window: usize) -> usize {
    let mut low = 0usize;
    if window > 0 && limit > window {
        low = limit - window;
    }
    low
}

#[ruda]
fn rotary_value<F: Float>(
    projected: &kernel_dsl::prelude::Tensor<F>,
    frequencies: &kernel_dsl::prelude::Tensor<F>,
    row: usize,
    column_base: usize,
    dimension: usize,
    frequency_row: usize,
    #[comptime] rotary_dim: usize,
    #[comptime] half_split: bool,
) -> F {
    let source = row + (column_base + dimension) * projected.stride(2);
    let mut result = projected[source];
    if dimension < rotary_dim {
    let mut paired_dimension = 0usize;
    let mut frequency_index = dimension;
    let mut negate = false;
    if half_split {
        let half = comptime!(rotary_dim / 2);
        frequency_index = 2 * (dimension % half);
        if dimension < half {
            paired_dimension = dimension + half;
            negate = true;
        } else {
            paired_dimension = dimension - half;
        }
    } else {
        if dimension % 2 == 0 {
            paired_dimension = dimension + 1;
            negate = true;
        } else {
            paired_dimension = dimension - 1;
        }
    }
    let paired = row + (column_base + paired_dimension) * projected.stride(2);
    let rotated = if negate {
        F::new(0.0_f32) - projected[paired]
    } else {
        projected[paired]
    };
    let frequency = frequency_row * frequencies.stride(0) + frequency_index * frequencies.stride(1);
        result = projected[source] * frequencies[frequency]
            + rotated * frequencies[frequency + frequencies.stride(2)];
    }
    result
}

/// Splits a packed `[B, S, Hq*D + 2*Hkv*D]` projection, applies RoPE at each
/// row's absolute position, writes Q as contiguous `[B, S, Hq, D]` and
/// scatters K/V into their reserved paged slots.
#[ruda(launch)]
fn paged_qkv_rope_kernel<F: Float>(
    projected: &kernel_dsl::prelude::Tensor<F>,
    frequencies: &kernel_dsl::prelude::Tensor<F>,
    query: &mut Array<F>,
    key_pool: &mut Array<F>,
    value_pool: &mut Array<F>,
    block_table: &Array<u32>,
    start_positions: &Array<u32>,
    sequence: u32,
    query_heads: u32,
    kv_heads: u32,
    block_size: u32,
    table_stride: u32,
    #[comptime] head_dim: usize,
    #[comptime] rotary_dim: usize,
    #[comptime] half_split: bool,
    #[comptime] per_token: bool,
    #[define(F)] _dtype: StorageType,
) {
    let sequence = sequence as usize;
    let query_heads = query_heads as usize;
    let kv_heads = kv_heads as usize;
    let block_size = block_size as usize;
    let table_stride = table_stride as usize;
    let batch_count = start_positions.len();
    let query_elements = batch_count * sequence * query_heads * head_dim;
    let kv_elements = batch_count * sequence * kv_heads * head_dim;
    let position = ABSOLUTE_POS;
    if position >= query_elements + 2 * kv_elements {
        terminate!();
    }
    let query_width = query_heads * head_dim;
    let kv_width = kv_heads * head_dim;

    if position < query_elements {
        let dimension = position % head_dim;
        let head = (position / head_dim) % query_heads;
        let token = (position / head_dim / query_heads) % sequence;
        let batch = position / head_dim / query_heads / sequence;
        let row = batch * projected.stride(0) + token * projected.stride(1);
        let absolute = start_positions[batch] as usize + token;
        let frequency_row = if per_token { batch * sequence + token } else { absolute };
        query[position] = rotary_value::<F>(
            projected,
            frequencies,
            row,
            head * head_dim,
            dimension,
            frequency_row,
            rotary_dim,
            half_split,
        );
    } else {
        let is_value = position >= query_elements + kv_elements;
        let local = if is_value {
            position - query_elements - kv_elements
        } else {
            position - query_elements
        };
        let dimension = local % head_dim;
        let head = (local / head_dim) % kv_heads;
        let token = (local / head_dim / kv_heads) % sequence;
        let batch = local / head_dim / kv_heads / sequence;
        let row = batch * projected.stride(0) + token * projected.stride(1);
        let absolute = start_positions[batch] as usize + token;
        let frequency_row = if per_token { batch * sequence + token } else { absolute };
        let page = block_table[batch * table_stride + absolute / block_size] as usize;
        let slot = ((page * kv_heads + head) * block_size + absolute % block_size) * head_dim
            + dimension;
        if is_value {
            let source = row + (query_width + kv_width + head * head_dim + dimension)
                * projected.stride(2);
            value_pool[slot] = projected[source];
        } else {
            key_pool[slot] = rotary_value::<F>(
                projected,
                frequencies,
                row,
                query_width + head * head_dim,
                dimension,
                frequency_row,
                rotary_dim,
                half_split,
            );
        }
    }
}

/// FlashAttention-2 style kernel on 16x16x16 tensor-core fragments.
///
/// Grid: `(row_blocks, batch * kv_heads)`, block: `(32, NUM_WARPS)`. Each warp
/// owns 16 query rows; the block cooperatively streams 16-key K/V tiles from
/// the page pool into shared memory. Scores and the running output stay in
/// FP32; probabilities are rounded to the model dtype before the P·V MMA and
/// the softmax denominator sums exactly those rounded probabilities.
#[ruda(launch)]
fn paged_flash_attention_kernel<F: Float>(
    query: &kernel_dsl::prelude::Tensor<F>,
    key_pool: &Array<F>,
    value_pool: &Array<F>,
    block_table: &Array<u32>,
    start_positions: &Array<u32>,
    context_lengths: &Array<u32>,
    output: &mut Array<F>,
    lse: &mut Array<f32>,
    sequence: u32,
    query_heads: u32,
    kv_heads: u32,
    block_size: u32,
    table_stride: u32,
    page_stride: u32,
    head_stride: u32,
    slot_stride: u32,
    window: u32,
    scale: f32,
    #[comptime] head_dim: usize,
    #[comptime] num_warps: usize,
    #[define(F)] _dtype: StorageType,
) {
    let page_stride = page_stride as usize;
    let head_stride = head_stride as usize;
    let slot_stride = slot_stride as usize;
    let window = window as usize;
    let row_stride = comptime!(head_dim + 8);
    let row_stride_u32 = comptime!((head_dim + 8) as u32);
    let chunks = comptime!(head_dim / 16);
    let threads = comptime!(num_warps * 32);
    let block_rows = comptime!(num_warps * 16);

    let lane = UNIT_POS_X as usize;
    let warp = UNIT_POS_Y as usize;
    let thread = warp * 32 + lane;

    let sequence = sequence as usize;
    let query_heads = query_heads as usize;
    let kv_heads = kv_heads as usize;
    let block_size = block_size as usize;
    let table_stride = table_stride as usize;
    let group = query_heads / kv_heads;
    let total_rows = sequence * group;

    let batch_head = RUDA_POS_Y as usize;
    let batch = batch_head / kv_heads;
    let kv_head = batch_head % kv_heads;
    let block_row = RUDA_POS_X as usize * block_rows;
    let start = start_positions[batch] as usize;
    let context = context_lengths[batch] as usize;

    let block_end = if block_row + block_rows < total_rows {
        block_row + block_rows
    } else {
        total_rows
    };
    let last_token = (block_end - 1) / group;
    let causal_limit = start + last_token + 1;
    let block_limit = if causal_limit < context {
        causal_limit
    } else {
        context
    };
    let tiles = (block_limit + 15) / 16;
    // The block's first row has the earliest window start.
    let first_tile = window_start(start + block_row / group + 1, window) / 16;

    let mut q_tiles = SharedMemory::<F>::new(comptime!(num_warps * 16 * (head_dim + 8)));
    let mut k_tile = SharedMemory::<F>::new(comptime!(16 * (head_dim + 8)));
    let mut v_tile = SharedMemory::<F>::new(comptime!(16 * (head_dim + 8)));
    let mut scores_tiles = SharedMemory::<f32>::new(comptime!(num_warps * 256));
    let mut probability_tiles = SharedMemory::<F>::new(comptime!(num_warps * 256));
    let mut output_tiles = SharedMemory::<f32>::new(comptime!(num_warps * 16 * head_dim));

    let warp_row = block_row + warp * 16;
    let q_base = warp * 16 * row_stride;
    let s_base = warp * 256;
    let o_base = warp * 16 * head_dim;

    let mut element = lane;
    while element < 16 * head_dim {
        let tile_row = element / head_dim;
        let dimension = element % head_dim;
        let global_row = warp_row + tile_row;
        let mut value = F::new(0.0_f32);
        if global_row < total_rows {
            let token = global_row / group;
            let head = kv_head * group + global_row % group;
            value = query[batch * query.stride(0)
                + token * query.stride(1)
                + head * query.stride(2)
                + dimension * query.stride(3)];
        }
        q_tiles[q_base + tile_row * row_stride + dimension] = value;
        output_tiles[o_base + element] = 0.0f32;
        element += 32;
    }

    // Two lanes per row; lane parity selects the 8-column half of each tile.
    let row = lane / 2;
    let half = lane % 2;
    let global_row = warp_row + row;
    let row_valid = global_row < total_rows;
    let mut row_limit = 0usize;
    let mut row_low = 0usize;
    if row_valid {
        let limit = start + global_row / group + 1;
        row_limit = if limit < context { limit } else { context };
        row_low = window_start(limit, window);
    }
    let mut running_max = f32::new(-3.0e38);
    let mut running_sum = 0.0f32;
    sync_ruda();

    for tile in first_tile..tiles {
        let key_start = tile * 16;
        let mut element = thread;
        while element < 16 * head_dim {
            let key = element / head_dim;
            let dimension = element % head_dim;
            let position = key_start + key;
            let mut key_value = F::new(0.0_f32);
            let mut value_value = F::new(0.0_f32);
            if position < block_limit {
                let page = block_table[batch * table_stride + position / block_size] as usize;
                let address = page * page_stride
                    + kv_head * head_stride
                    + (position % block_size) * slot_stride
                    + dimension;
                key_value = key_pool[address];
                value_value = value_pool[address];
            }
            k_tile[key * row_stride + dimension] = key_value;
            v_tile[key * row_stride + dimension] = value_value;
            element += threads;
        }
        sync_ruda();

        let scores = cmma::Matrix::<f32>::from_value(
            cmma::MatrixIdent::Accumulator,
            16usize,
            16usize,
            16usize,
            cmma::MatrixLayout::Undefined,
            0.0f32,
        );
        for chunk in 0..chunks {
            let query_fragment = cmma::Matrix::<F>::from_slice(
                cmma::MatrixIdent::A,
                16usize,
                16usize,
                16usize,
                cmma::MatrixLayout::RowMajor,
                &q_tiles.slice(q_base + chunk * 16, q_base + 16 * row_stride),
                row_stride_u32,
            );
            // K stored [key, dim] row-major is K^T in column-major order.
            let key_fragment = cmma::Matrix::<F>::from_slice(
                cmma::MatrixIdent::B,
                16usize,
                16usize,
                16usize,
                cmma::MatrixLayout::ColMajor,
                &k_tile.slice(chunk * 16, 16 * row_stride),
                row_stride_u32,
            );
            cmma::execute::<F, F, f32, f32, _>(&query_fragment, &key_fragment, &scores, &scores);
        }
        cmma::store(
            &mut scores_tiles.slice_mut(s_base, s_base + 256),
            &scores,
            16u32,
            cmma::MatrixLayout::RowMajor,
        );
        sync_ruda();

        let mut tile_max = f32::new(-3.0e38);
        for column_offset in 0..8usize {
            let column = half * 8 + column_offset;
            let key = key_start + column;
            if key < row_limit && key >= row_low {
                let score = scores_tiles[s_base + row * 16 + column] * scale;
                if score > tile_max {
                    tile_max = score;
                }
            }
        }
        let partner_max = plane_shuffle_xor(tile_max, 1u32);
        if partner_max > tile_max {
            tile_max = partner_max;
        }
        let new_max = if tile_max > running_max {
            tile_max
        } else {
            running_max
        };
        let alpha = (running_max - new_max).exp();
        let mut tile_sum = 0.0f32;
        for column_offset in 0..8usize {
            let column = half * 8 + column_offset;
            let mut probability = 0.0f32;
            let key = key_start + column;
            if key < row_limit && key >= row_low {
                probability =
                    (scores_tiles[s_base + row * 16 + column] * scale - new_max).exp();
            }
            let rounded = F::cast_from(probability);
            tile_sum += f32::cast_from(rounded);
            probability_tiles[s_base + row * 16 + column] = rounded;
        }
        tile_sum += plane_shuffle_xor(tile_sum, 1u32);
        running_sum = running_sum * alpha + tile_sum;
        running_max = new_max;
        sync_ruda();

        let probability_fragment = cmma::Matrix::<F>::from_slice(
            cmma::MatrixIdent::A,
            16usize,
            16usize,
            16usize,
            cmma::MatrixLayout::RowMajor,
            &probability_tiles.slice(s_base, s_base + 256),
            16u32,
        );
        for chunk in 0..chunks {
            let value_fragment = cmma::Matrix::<F>::from_slice(
                cmma::MatrixIdent::B,
                16usize,
                16usize,
                16usize,
                cmma::MatrixLayout::RowMajor,
                &v_tile.slice(chunk * 16, 16 * row_stride),
                row_stride_u32,
            );
            let product = cmma::Matrix::<f32>::from_value(
                cmma::MatrixIdent::Accumulator,
                16usize,
                16usize,
                16usize,
                cmma::MatrixLayout::Undefined,
                0.0f32,
            );
            cmma::execute::<F, F, f32, f32, _>(
                &probability_fragment,
                &value_fragment,
                &product,
                &product,
            );
            cmma::store(
                &mut scores_tiles.slice_mut(s_base, s_base + 256),
                &product,
                16u32,
                cmma::MatrixLayout::RowMajor,
            );
            sync_ruda();
            for column_offset in 0..8usize {
                let column = half * 8 + column_offset;
                let index = o_base + row * head_dim + chunk * 16 + column;
                output_tiles[index] =
                    output_tiles[index] * alpha + scores_tiles[s_base + row * 16 + column];
            }
            sync_ruda();
        }
    }

    if row_valid {
        let token = global_row / group;
        let head = kv_head * group + global_row % group;
        let output_row = (batch * sequence + token) * query_heads + head;
        let output_base = output_row * head_dim;
        let mut inverse = 0.0f32;
        let mut row_lse = f32::new(-3.0e38);
        if running_sum > 0.0f32 {
            inverse = 1.0f32 / running_sum;
            row_lse = running_max + running_sum.ln();
        }
        if half == 0 {
            lse[output_row] = row_lse;
        }
        let half_width = comptime!(head_dim / 2);
        for offset in 0..half_width {
            let dimension = half * half_width + offset;
            output[output_base + dimension] =
                F::cast_from(output_tiles[o_base + row * head_dim + dimension] * inverse);
        }
    }
}

/// Portable online-softmax fallback: one warp per query row, any head width
/// and dtype. Used when tensor-core fragments are unavailable (FP32 models,
/// head widths that are not a multiple of 16, devices without CMMA).
#[ruda(launch)]
fn paged_attention_scalar_kernel<F: Float>(
    query: &kernel_dsl::prelude::Tensor<F>,
    key_pool: &Array<F>,
    value_pool: &Array<F>,
    block_table: &Array<u32>,
    start_positions: &Array<u32>,
    context_lengths: &Array<u32>,
    output: &mut Array<F>,
    lse: &mut Array<f32>,
    sequence: u32,
    query_heads: u32,
    kv_heads: u32,
    block_size: u32,
    table_stride: u32,
    page_stride: u32,
    head_stride: u32,
    slot_stride: u32,
    window: u32,
    scale: f32,
    #[comptime] head_dim: usize,
    #[define(F)] _dtype: StorageType,
) {
    let page_stride = page_stride as usize;
    let head_stride = head_stride as usize;
    let slot_stride = slot_stride as usize;
    let lane = UNIT_POS_X as usize;
    let sequence = sequence as usize;
    let query_heads = query_heads as usize;
    let kv_heads = kv_heads as usize;
    let block_size = block_size as usize;
    let table_stride = table_stride as usize;
    let group = query_heads / kv_heads;

    let global_row = RUDA_POS_X as usize;
    let batch_head = RUDA_POS_Y as usize;
    let batch = batch_head / kv_heads;
    let kv_head = batch_head % kv_heads;
    let token = global_row / group;
    let head = kv_head * group + global_row % group;
    let start = start_positions[batch] as usize;
    let context = context_lengths[batch] as usize;
    let causal_limit = start + token + 1;
    let limit = if causal_limit < context {
        causal_limit
    } else {
        context
    };
    let low = window_start(causal_limit, window as usize);

    let mut query_values = SharedMemory::<f32>::new(head_dim);
    let mut accumulators = SharedMemory::<f32>::new(head_dim);
    let mut dimension = lane;
    while dimension < head_dim {
        query_values[dimension] = f32::cast_from(
            query[batch * query.stride(0)
                + token * query.stride(1)
                + head * query.stride(2)
                + dimension * query.stride(3)],
        );
        accumulators[dimension] = 0.0f32;
        dimension += 32;
    }
    sync_ruda();

    let mut running_max = f32::new(-3.0e38);
    let mut running_sum = 0.0f32;
    for position in low..limit {
        let page = block_table[batch * table_stride + position / block_size] as usize;
        let base =
            page * page_stride + kv_head * head_stride + (position % block_size) * slot_stride;
        let mut dot = 0.0f32;
        let mut dimension = lane;
        while dimension < head_dim {
            dot += query_values[dimension] * f32::cast_from(key_pool[base + dimension]);
            dimension += 32;
        }
        let score = plane_sum(dot) * scale;
        let new_max = if score > running_max {
            score
        } else {
            running_max
        };
        let alpha = (running_max - new_max).exp();
        let probability = (score - new_max).exp();
        running_sum = running_sum * alpha + probability;
        let mut dimension = lane;
        while dimension < head_dim {
            accumulators[dimension] = accumulators[dimension] * alpha
                + probability * f32::cast_from(value_pool[base + dimension]);
            dimension += 32;
        }
        running_max = new_max;
    }

    let mut inverse = 0.0f32;
    let mut row_lse = f32::new(-3.0e38);
    if running_sum > 0.0f32 {
        inverse = 1.0f32 / running_sum;
        row_lse = running_max + running_sum.ln();
    }
    let output_row = (batch * sequence + token) * query_heads + head;
    if lane == 0 {
        lse[output_row] = row_lse;
    }
    let output_base = output_row * head_dim;
    let mut dimension = lane;
    while dimension < head_dim {
        output[output_base + dimension] = F::cast_from(accumulators[dimension] * inverse);
        dimension += 32;
    }
}

/// `delta[row] = sum_d dO[row, d] * O[row, d]` over contiguous `[rows, D]`.
#[ruda(launch)]
fn attention_backward_delta_kernel<F: Float>(
    output: &Array<F>,
    grad_output: &Array<F>,
    delta: &mut Array<f32>,
    #[comptime] head_dim: usize,
    #[define(F)] _dtype: StorageType,
) {
    let row = RUDA_POS_X as usize;
    let lane = UNIT_POS_X as usize;
    let base = row * head_dim;
    let mut sum = 0.0f32;
    let mut dimension = lane;
    while dimension < head_dim {
        sum += f32::cast_from(output[base + dimension])
            * f32::cast_from(grad_output[base + dimension]);
        dimension += 32;
    }
    let total = plane_sum(sum);
    if lane == 0 {
        delta[row] = total;
    }
}

/// dQ for causal (optionally windowed) attention from position zero over
/// contiguous `q/dO [B, S, Hq, D]`, `k/v [B, S, Hkv, D]`. Grid
/// `(S * group, B * Hkv)`, one warp per query row; `P = exp(S - lse)`.
#[ruda(launch)]
fn attention_backward_query_kernel<F: Float>(
    query: &Array<F>,
    key: &Array<F>,
    value: &Array<F>,
    grad_output: &Array<F>,
    lse: &Array<f32>,
    delta: &Array<f32>,
    grad_query: &mut Array<F>,
    sequence: u32,
    query_heads: u32,
    kv_heads: u32,
    window: u32,
    scale: f32,
    #[comptime] head_dim: usize,
    #[define(F)] _dtype: StorageType,
) {
    let lane = UNIT_POS_X as usize;
    let sequence = sequence as usize;
    let query_heads = query_heads as usize;
    let kv_heads = kv_heads as usize;
    let group = query_heads / kv_heads;
    let global_row = RUDA_POS_X as usize;
    let batch_head = RUDA_POS_Y as usize;
    let batch = batch_head / kv_heads;
    let kv_head = batch_head % kv_heads;
    let token = global_row / group;
    let head = kv_head * group + global_row % group;
    let row = (batch * sequence + token) * query_heads + head;
    let limit = token + 1;
    let low = window_start(limit, window as usize);
    let row_lse = lse[row];
    let row_delta = delta[row];

    let mut query_values = SharedMemory::<f32>::new(head_dim);
    let mut grad_values = SharedMemory::<f32>::new(head_dim);
    let mut accumulators = SharedMemory::<f32>::new(head_dim);
    let mut dimension = lane;
    while dimension < head_dim {
        query_values[dimension] = f32::cast_from(query[row * head_dim + dimension]);
        grad_values[dimension] = f32::cast_from(grad_output[row * head_dim + dimension]);
        accumulators[dimension] = 0.0f32;
        dimension += 32;
    }
    sync_ruda();

    for position in low..limit {
        let base = ((batch * sequence + position) * kv_heads + kv_head) * head_dim;
        let mut dot = 0.0f32;
        let mut grad_dot = 0.0f32;
        let mut dimension = lane;
        while dimension < head_dim {
            dot += query_values[dimension] * f32::cast_from(key[base + dimension]);
            grad_dot += grad_values[dimension] * f32::cast_from(value[base + dimension]);
            dimension += 32;
        }
        let probability = (plane_sum(dot) * scale - row_lse).exp();
        let grad_score = probability * (plane_sum(grad_dot) - row_delta);
        let mut dimension = lane;
        while dimension < head_dim {
            accumulators[dimension] += grad_score * f32::cast_from(key[base + dimension]);
            dimension += 32;
        }
    }

    let mut dimension = lane;
    while dimension < head_dim {
        grad_query[row * head_dim + dimension] = F::cast_from(accumulators[dimension] * scale);
        dimension += 32;
    }
}

/// dK and dV with the layouts of [`attention_backward_query_kernel`]. Grid
/// `(S, B * Hkv)`, one warp per key position summing over every query head
/// of its GQA group and every token that sees the key, so no atomics.
#[ruda(launch)]
fn attention_backward_key_value_kernel<F: Float>(
    query: &Array<F>,
    key: &Array<F>,
    value: &Array<F>,
    grad_output: &Array<F>,
    lse: &Array<f32>,
    delta: &Array<f32>,
    grad_key: &mut Array<F>,
    grad_value: &mut Array<F>,
    sequence: u32,
    query_heads: u32,
    kv_heads: u32,
    window: u32,
    scale: f32,
    #[comptime] head_dim: usize,
    #[define(F)] _dtype: StorageType,
) {
    let lane = UNIT_POS_X as usize;
    let sequence = sequence as usize;
    let query_heads = query_heads as usize;
    let kv_heads = kv_heads as usize;
    let window = window as usize;
    let group = query_heads / kv_heads;
    let position = RUDA_POS_X as usize;
    let batch_head = RUDA_POS_Y as usize;
    let batch = batch_head / kv_heads;
    let kv_head = batch_head % kv_heads;
    let kv_row = (batch * sequence + position) * kv_heads + kv_head;
    // Token t sees this key iff position <= t and (no window or t < position + window).
    let mut end_token = sequence;
    if window > 0 && position + window < sequence {
        end_token = position + window;
    }

    let mut key_values = SharedMemory::<f32>::new(head_dim);
    let mut value_values = SharedMemory::<f32>::new(head_dim);
    let mut key_accumulators = SharedMemory::<f32>::new(head_dim);
    let mut value_accumulators = SharedMemory::<f32>::new(head_dim);
    let mut dimension = lane;
    while dimension < head_dim {
        key_values[dimension] = f32::cast_from(key[kv_row * head_dim + dimension]);
        value_values[dimension] = f32::cast_from(value[kv_row * head_dim + dimension]);
        key_accumulators[dimension] = 0.0f32;
        value_accumulators[dimension] = 0.0f32;
        dimension += 32;
    }
    sync_ruda();

    for member in 0..group {
        let head = kv_head * group + member;
        for token in position..end_token {
            let row = (batch * sequence + token) * query_heads + head;
            let base = row * head_dim;
            let mut dot = 0.0f32;
            let mut grad_dot = 0.0f32;
            let mut dimension = lane;
            while dimension < head_dim {
                dot += f32::cast_from(query[base + dimension]) * key_values[dimension];
                grad_dot +=
                    f32::cast_from(grad_output[base + dimension]) * value_values[dimension];
                dimension += 32;
            }
            let probability = (plane_sum(dot) * scale - lse[row]).exp();
            let grad_score = probability * (plane_sum(grad_dot) - delta[row]);
            let mut dimension = lane;
            while dimension < head_dim {
                value_accumulators[dimension] +=
                    probability * f32::cast_from(grad_output[base + dimension]);
                key_accumulators[dimension] +=
                    grad_score * f32::cast_from(query[base + dimension]);
                dimension += 32;
            }
        }
    }

    let mut dimension = lane;
    while dimension < head_dim {
        grad_key[kv_row * head_dim + dimension] =
            F::cast_from(key_accumulators[dimension] * scale);
        grad_value[kv_row * head_dim + dimension] = F::cast_from(value_accumulators[dimension]);
        dimension += 32;
    }
}

/// Device-resident per-batch metadata shared by every layer of one forward.

#[derive(Debug, Clone)]
pub(crate) struct PagedAttentionBatch<R: DeviceRuntime> {
    pub block_table: RudaTensor<R>,
    pub start_positions: RudaTensor<R>,
    pub context_lengths: RudaTensor<R>,
    pub batch: usize,
    pub sequence: usize,
    pub table_stride: usize,
    pub block_size: usize,
    /// Largest `context_length` in the batch (for RoPE table bounds checks).
    pub max_context: usize,
}

impl<R: DeviceRuntime> PagedAttentionBatch<R> {
    /// `block_table` is row-major `[batch, table_stride]`.
    pub fn upload(
        client: &ComputeClient<R>,
        device: &R::Device,
        block_table: &[u32],
        table_stride: usize,
        start_positions: &[u32],
        context_lengths: &[u32],
        block_size: usize,
        sequence: usize,
    ) -> Result<Self, String> {
        let batch = start_positions.len();
        if batch == 0 || sequence == 0 || block_size == 0 || table_stride == 0 {
            return Err("paged attention batch dimensions must be non-zero".into());
        }
        if context_lengths.len() != batch || block_table.len() != batch * table_stride {
            return Err("paged attention metadata lengths do not match the batch".into());
        }
        let mut max_context = 0usize;
        for row in 0..batch {
            let start = start_positions[row] as usize;
            let context = context_lengths[row] as usize;
            if context != start + sequence {
                return Err(format!(
                    "paged row {row}: context {context} must equal start {start} + {sequence} new tokens"
                ));
            }
            if context.div_ceil(block_size) > table_stride {
                return Err(format!(
                    "paged row {row}: context {context} needs more than {table_stride} pages"
                ));
            }
            max_context = max_context.max(context);
        }
        Ok(Self {
            block_table: upload_u32(client, device, block_table),
            start_positions: upload_u32(client, device, start_positions),
            context_lengths: upload_u32(client, device, context_lengths),
            batch,
            sequence,
            table_stride,
            block_size,
            max_context,
        })
    }
}

impl<R: DeviceRuntime> PagedAttentionBatch<R> {
    /// One page per batch row, every row appending `sequence` tokens at
    /// `position` (dense caches and training activations).
    pub fn dense(
        client: &ComputeClient<R>,
        device: &R::Device,
        batch: usize,
        position: usize,
        sequence: usize,
        block_size: usize,
    ) -> Self {
        let table = (0..batch as u32).collect::<Vec<_>>();
        let starts = vec![position as u32; batch];
        let contexts = vec![(position + sequence) as u32; batch];
        Self::upload(client, device, &table, 1, &starts, &contexts, block_size, sequence)
            .expect("a dense buffer always forms a valid single-page table")
    }
}

fn upload_u32<R: DeviceRuntime>(
    client: &ComputeClient<R>,
    device: &R::Device,
    values: &[u32],
) -> RudaTensor<R> {
    let handle = client.create_from_slice(u32::as_bytes(values));
    RudaTensor::new_contiguous(
        client.clone(),
        device.clone(),
        Shape::new([values.len()]),
        handle,
        DType::U32,
    )
}

fn shared_memory_bytes(head_dim: usize, warps: usize, element_bytes: usize) -> usize {
    let row_stride = head_dim + ROW_PADDING;
    let query = warps * TILE * row_stride * element_bytes;
    let key_value = 2 * TILE * row_stride * element_bytes;
    let scores = warps * TILE * TILE * 4;
    let probabilities = warps * TILE * TILE * element_bytes;
    let output = warps * TILE * head_dim * 4;
    query + key_value + scores + probabilities + output
}

/// Number of warps for the tensor-core kernel, or `None` to use the portable
/// kernel.
fn tensor_core_warps<R: DeviceRuntime>(
    client: &ComputeClient<R>,
    dtype: DType,
    head_dim: usize,
    total_rows: usize,
) -> Option<usize> {
    if !matches!(dtype, DType::F16 | DType::BF16) || head_dim == 0 || head_dim % TILE != 0 {
        return None;
    }
    let element: ruda_core::ir::StorageType = dtype.into();
    let accumulator: ruda_core::ir::StorageType = DType::F32.into();
    let config = MmaConfig {
        a_type: element,
        b_type: element,
        cd_type: accumulator,
        m: TILE as u32,
        n: TILE as u32,
        k: TILE as u32,
    };
    let properties = client.properties();
    if !properties.features.matmul.cmma.contains(&config) {
        return None;
    }
    // Stay inside the default static shared-memory window even when the
    // device reports a larger opt-in maximum.
    let budget = (properties.hardware.max_shared_memory_size as usize).min(48 * 1024);
    let wanted = total_rows.div_ceil(TILE).clamp(1, MAX_WARPS);
    (1..=wanted)
        .rev()
        .find(|&warps| shared_memory_bytes(head_dim, warps, 2) <= budget)
}

/// Causal attention over paged K/V. `query` is logically `[B, S, Hq, D]`
/// (any strides); the result is contiguous `[B, S, Hq, D]`.
pub(crate) fn paged_attention<R, F, I, BT>(
    query: Tensor<DeviceBackend<R, F, I, BT>, 4>,
    key_pool: &RudaTensor<R>,
    value_pool: &RudaTensor<R>,
    batch: &PagedAttentionBatch<R>,
    kv_heads: usize,
    window: usize,
) -> Tensor<DeviceBackend<R, F, I, BT>, 4>
where
    R: DeviceRuntime,
    R::Server: ComputeServer,
    R::Device: DeviceOps,
    F: FloatElement,
    I: IntElement,
    BT: BoolElement,
{
    let query = query.into_primitive().tensor();
    let head_dim = query.meta.shape()[3];
    let pool_shape = key_pool.meta.shape().dims::<4>();
    assert_eq!(pool_shape[1], kv_heads, "KV pool head count mismatch");
    assert_eq!(pool_shape[2], batch.block_size, "KV pool block size mismatch");
    assert_eq!(pool_shape[3], head_dim, "KV pool head width mismatch");
    let layout = KvLayout::pool(kv_heads, batch.block_size, head_dim);
    let (output, _) =
        attention_forward_raw(query, key_pool, value_pool, batch, layout, kv_heads, window);
    Tensor::from_primitive(TensorPrimitive::Float(output))
}

fn empty_like_device<R: DeviceRuntime>(like: &RudaTensor<R>, shape: Shape, dtype: DType) -> RudaTensor<R> {
    empty_device_contiguous_dtype::<R>(like.client.clone(), like.device.clone(), shape, dtype)
}

/// Causal attention forward on raw device tensors. `query` is logically
/// `[B, S, Hq, D]` (any strides); K/V follow `layout`. Returns contiguous
/// `[B, S, Hq, D]` output and `[B, S, Hq]` FP32 natural-log sum-exp of the
/// scaled scores (consumed by [`attention_backward_raw`]).
pub(crate) fn attention_forward_raw<R: DeviceRuntime>(
    query: RudaTensor<R>,
    key_pool: &RudaTensor<R>,
    value_pool: &RudaTensor<R>,
    batch: &PagedAttentionBatch<R>,
    layout: KvLayout,
    kv_heads: usize,
    window: usize,
) -> (RudaTensor<R>, RudaTensor<R>) {
    let [batch_size, sequence, query_heads, head_dim] = query.meta.shape().dims::<4>();
    assert_eq!(batch_size, batch.batch, "attention batch size mismatch");
    assert_eq!(sequence, batch.sequence, "attention sequence mismatch");
    assert_eq!(layout.block_size, batch.block_size, "attention block size mismatch");
    assert!(kv_heads > 0 && query_heads.is_multiple_of(kv_heads));
    assert!(head_dim > 0, "attention needs a non-empty head");
    assert_eq!(value_pool.meta.shape(), key_pool.meta.shape());
    assert_eq!(key_pool.dtype, query.dtype);
    assert_eq!(value_pool.dtype, query.dtype);
    for stride in [layout.page_stride, layout.head_stride, layout.slot_stride] {
        assert!(stride <= u32::MAX as usize, "K/V stride exceeds u32");
    }
    let key_pool = &into_contiguous(key_pool.clone());
    let value_pool = &into_contiguous(value_pool.clone());

    let output = empty_like_device(
        &query,
        Shape::new([batch_size, sequence, query_heads, head_dim]),
        query.dtype,
    );
    let lse = empty_like_device(&query, Shape::new([batch_size, sequence, query_heads]), DType::F32);
    let client = query.client.clone();
    let group = query_heads / kv_heads;
    let total_rows = sequence * group;
    let scale = 1.0f32 / (head_dim as f32).sqrt();
    let dtype = output.dtype;
    match tensor_core_warps(&client, dtype, head_dim, total_rows) {
        Some(warps) => {
            let row_blocks = total_rows.div_ceil(warps * TILE);
            paged_flash_attention_kernel::launch::<R>(
                &client,
                RudaCount::Static(row_blocks as u32, (batch_size * kv_heads) as u32, 1),
                RudaDim::new_2d(32, warps as u32),
                query.into_tensor_arg(),
                key_pool.clone().into_array_arg(),
                value_pool.clone().into_array_arg(),
                batch.block_table.clone().into_array_arg(),
                batch.start_positions.clone().into_array_arg(),
                batch.context_lengths.clone().into_array_arg(),
                output.clone().into_array_arg(),
                lse.clone().into_array_arg(),
                sequence as u32,
                query_heads as u32,
                kv_heads as u32,
                layout.block_size as u32,
                batch.table_stride as u32,
                layout.page_stride as u32,
                layout.head_stride as u32,
                layout.slot_stride as u32,
                window as u32,
                scale,
                head_dim,
                warps,
                dtype.into(),
            );
        }
        None => {
            paged_attention_scalar_kernel::launch::<R>(
                &client,
                RudaCount::Static(total_rows as u32, (batch_size * kv_heads) as u32, 1),
                RudaDim::new_1d(32),
                query.into_tensor_arg(),
                key_pool.clone().into_array_arg(),
                value_pool.clone().into_array_arg(),
                batch.block_table.clone().into_array_arg(),
                batch.start_positions.clone().into_array_arg(),
                batch.context_lengths.clone().into_array_arg(),
                output.clone().into_array_arg(),
                lse.clone().into_array_arg(),
                sequence as u32,
                query_heads as u32,
                kv_heads as u32,
                layout.block_size as u32,
                batch.table_stride as u32,
                layout.page_stride as u32,
                layout.head_stride as u32,
                layout.slot_stride as u32,
                window as u32,
                scale,
                head_dim,
                dtype.into(),
            );
        }
    }
    (output, lse)
}

/// Fused Q/K/V split + RoPE + paged K/V scatter. Returns Q as contiguous
/// `[B, S, Hq, D]`; K/V are written in place into the pools at the slots
/// described by `batch` (which must already be reserved by the caller).
pub(crate) fn paged_qkv_rope<R, F, I, BT>(
    projected: Tensor<DeviceBackend<R, F, I, BT>, 3>,
    frequencies: Tensor<DeviceBackend<R, F, I, BT>, 3>,
    key_pool: &RudaTensor<R>,
    value_pool: &RudaTensor<R>,
    batch: &PagedAttentionBatch<R>,
    query_heads: usize,
    kv_heads: usize,
    head_dim: usize,
    half_split: bool,
    per_token: bool,
) -> Tensor<DeviceBackend<R, F, I, BT>, 4>
where
    R: DeviceRuntime,
    R::Server: ComputeServer,
    R::Device: DeviceOps,
    F: FloatElement,
    I: IntElement,
    BT: BoolElement,
{
    let frequencies = if frequencies.dtype() == projected.dtype() {
        frequencies
    } else {
        frequencies.cast(projected.dtype())
    };
    let projected = projected.into_primitive().tensor();
    let frequencies = frequencies.into_primitive().tensor();
    let [batch_size, sequence, width] = projected.meta.shape().dims::<3>();
    assert_eq!(batch_size, batch.batch, "paged projection batch mismatch");
    assert_eq!(sequence, batch.sequence, "paged projection sequence mismatch");
    assert_eq!(width, (query_heads + 2 * kv_heads) * head_dim);
    assert!(head_dim.is_multiple_of(2), "RoPE needs an even head width");
    assert!(
        (if per_token { batch_size * sequence } else { batch.max_context }) <= frequencies.meta.shape()[0],
        "paged context exceeds the RoPE table"
    );
    let rotary_dim = frequencies.meta.shape()[1];
    assert!(rotary_dim > 0 && rotary_dim <= head_dim && rotary_dim.is_multiple_of(2));
    assert_eq!(frequencies.meta.shape()[2], 2);
    assert_eq!(key_pool.dtype, projected.dtype);
    assert_eq!(value_pool.dtype, projected.dtype);

    let query = empty_device_contiguous_dtype::<R>(
        projected.client.clone(),
        projected.device.clone(),
        Shape::new([batch_size, sequence, query_heads, head_dim]),
        projected.dtype,
    );
    let elements = batch_size * sequence * (query_heads + 2 * kv_heads) * head_dim;
    let client = projected.client.clone();
    let ruda_dim = RudaDim::new(client.properties(), elements);
    let ruda_count = calculate_ruda_count_elemwise(&client, elements, ruda_dim);
    let dtype = query.dtype;
    paged_qkv_rope_kernel::launch::<R>(
        &client,
        ruda_count,
        ruda_dim,
        projected.into_tensor_arg(),
        frequencies.into_tensor_arg(),
        query.clone().into_array_arg(),
        key_pool.clone().into_array_arg(),
        value_pool.clone().into_array_arg(),
        batch.block_table.clone().into_array_arg(),
        batch.start_positions.clone().into_array_arg(),
        sequence as u32,
        query_heads as u32,
        kv_heads as u32,
        batch.block_size as u32,
        batch.table_stride as u32,
        head_dim,
        rotary_dim,
        half_split,
        per_token,
        dtype.into(),
    );
    Tensor::from_primitive(TensorPrimitive::Float(query))
}

/// Causal flash attention over a dense `[B, Hkv, capacity, D]` cache, treated
/// as one page per batch row. `query` is `[B, Hq, S, D]` at absolute position
/// `position`; the result is contiguous `[B, S, Hq, D]`.
pub(crate) fn dense_flash_attention<R, F, I, BT>(
    query: Tensor<DeviceBackend<R, F, I, BT>, 4>,
    key: Tensor<DeviceBackend<R, F, I, BT>, 4>,
    value: Tensor<DeviceBackend<R, F, I, BT>, 4>,
    position: usize,
) -> Tensor<DeviceBackend<R, F, I, BT>, 4>
where
    R: DeviceRuntime,
    R::Server: ComputeServer,
    R::Device: DeviceOps,
    F: FloatElement,
    I: IntElement,
    BT: BoolElement,
{
    dense_flash_attention_windowed(query, key, value, position, 0)
}

/// [`dense_flash_attention`] with a sliding window (`0` = unbounded).
pub(crate) fn dense_flash_attention_windowed<R, F, I, BT>(
    query: Tensor<DeviceBackend<R, F, I, BT>, 4>,
    key: Tensor<DeviceBackend<R, F, I, BT>, 4>,
    value: Tensor<DeviceBackend<R, F, I, BT>, 4>,
    position: usize,
    window: usize,
) -> Tensor<DeviceBackend<R, F, I, BT>, 4>
where
    R: DeviceRuntime,
    R::Server: ComputeServer,
    R::Device: DeviceOps,
    F: FloatElement,
    I: IntElement,
    BT: BoolElement,
{
    let [batch_size, _, sequence, _] = query.dims();
    let [_, kv_heads, capacity, _] = key.dims();
    let key = into_contiguous(key.into_primitive().tensor());
    let value = into_contiguous(value.into_primitive().tensor());
    let batch =
        PagedAttentionBatch::dense(&key.client, &key.device, batch_size, position, sequence, capacity);
    paged_attention(query.swap_dims(1, 2), &key, &value, &batch, kv_heads, window)
}

/// Gradients of causal attention computed from position zero on contiguous
/// `q [B,S,Hq,D]`, `k/v [B,S,Hkv,D]`, given the forward output, its
/// log-sum-exp and `dO [B,S,Hq,D]`. Returns `(dQ, dK, dV)` in the input dtype.
pub(crate) fn attention_backward_raw<R: DeviceRuntime>(
    query: RudaTensor<R>,
    key: RudaTensor<R>,
    value: RudaTensor<R>,
    output: RudaTensor<R>,
    lse: RudaTensor<R>,
    grad_output: RudaTensor<R>,
    window: usize,
) -> (RudaTensor<R>, RudaTensor<R>, RudaTensor<R>) {
    let query = into_contiguous(query);
    let key = into_contiguous(key);
    let value = into_contiguous(value);
    let output = into_contiguous(output);
    let lse = into_contiguous(lse);
    let grad_output = into_contiguous(grad_output);
    assert_eq!(grad_output.dtype, query.dtype, "attention gradient dtype mismatch");
    let [batch, sequence, query_heads, head_dim] = query.meta.shape().dims::<4>();
    let kv_heads = key.meta.shape()[2];
    assert!(kv_heads > 0 && query_heads.is_multiple_of(kv_heads));
    let group = query_heads / kv_heads;
    let dtype = query.dtype;
    let scale = 1.0f32 / (head_dim as f32).sqrt();
    let client = query.client.clone();
    let delta = empty_like_device(&query, Shape::new([batch, sequence, query_heads]), DType::F32);
    attention_backward_delta_kernel::launch::<R>(
        &client,
        RudaCount::Static((batch * sequence * query_heads) as u32, 1, 1),
        RudaDim::new_1d(32),
        output.into_array_arg(),
        grad_output.clone().into_array_arg(),
        delta.clone().into_array_arg(),
        head_dim,
        dtype.into(),
    );
    let grad_query = empty_like_device(&query, query.meta.shape().clone(), dtype);
    let grad_key = empty_like_device(&key, key.meta.shape().clone(), dtype);
    let grad_value = empty_like_device(&value, value.meta.shape().clone(), dtype);
    attention_backward_query_kernel::launch::<R>(
        &client,
        RudaCount::Static((sequence * group) as u32, (batch * kv_heads) as u32, 1),
        RudaDim::new_1d(32),
        query.clone().into_array_arg(),
        key.clone().into_array_arg(),
        value.clone().into_array_arg(),
        grad_output.clone().into_array_arg(),
        lse.clone().into_array_arg(),
        delta.clone().into_array_arg(),
        grad_query.clone().into_array_arg(),
        sequence as u32,
        query_heads as u32,
        kv_heads as u32,
        window as u32,
        scale,
        head_dim,
        dtype.into(),
    );
    attention_backward_key_value_kernel::launch::<R>(
        &client,
        RudaCount::Static(sequence as u32, (batch * kv_heads) as u32, 1),
        RudaDim::new_1d(32),
        query.into_array_arg(),
        key.into_array_arg(),
        value.into_array_arg(),
        grad_output.into_array_arg(),
        lse.into_array_arg(),
        delta.into_array_arg(),
        grad_key.clone().into_array_arg(),
        grad_value.clone().into_array_arg(),
        sequence as u32,
        query_heads as u32,
        kv_heads as u32,
        window as u32,
        scale,
        head_dim,
        dtype.into(),
    );
    (grad_query, grad_key, grad_value)
}

#[cfg(all(test, feature = "nvidia"))]
mod tests;
