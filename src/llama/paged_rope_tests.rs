use super::*;
use crate::{
    KvPageId, KvReservationId, PagedKvCacheConfig, PagedLlamaEngine, RequestId, ScheduledBatch,
    ScheduledBatchKind, ScheduledSequence,
};
use ruda_tensor_device::cuda::{Cuda, CudaDevice};

#[test]
fn paged_partial_static_dynamic_and_long_rope_match_same_dense_chunks() {
    type B = Cuda<f32, i32>;
    let device = CudaDevice::default();
    let config = LlamaConfig {
        vocab_size: 32,
        d_model: 64,
        d_ff: 96,
        num_hidden_layers: 2,
        num_query_heads: 4,
        num_kv_heads: 2,
        max_sequence_length: 32,
        rms_norm_epsilon: 1e-6,
        rope_theta: 10000.0,
    };
    for scaling in [
        rope::RopeScaling::Static,
        rope::RopeScaling::Dynamic {
            factor: 2.0,
            original_length: 4,
        },
        rope::RopeScaling::Long {
            original_length: 4,
            inverse_long: vec![0.5, 0.005, 0.0005, 0.00005],
        },
    ] {
        let mut model = LlamaForCausalLm::<B>::init_qwen2(&config, &device)
            .unwrap()
            .into_packed_inference();
        let parameters = rope::RopeParameters {
            theta: 10000.0,
            inverse_frequencies: vec![1.0, 0.01, 0.001, 0.0001],
            attention_factor: 1.1,
            scaling,
            per_layer: vec![],
        };
        for layer in &mut model.layers {
            layer.self_attn.rope = parameters.init(&config, &device, DType::F32);
            layer.self_attn.rotary_state = parameters.runtime_state();
        }
        let engine = PagedLlamaEngine::new(
            &model,
            PagedKvCacheConfig {
                block_size: 4,
                num_pages: 8,
                max_sequence_length: 32,
            },
            &device,
        )
        .unwrap();
        let mut dense_cache = model.new_cache();
        let mut start = 0;
        for (index, count) in [3, 3, 2].into_iter().enumerate() {
            let tokens: Vec<i32> = (start..start + count).map(|i| (i + 1) as i32).collect();
            let batch = ScheduledBatch {
                id: index as u64,
                kind: ScheduledBatchKind::Prefill,
                sequences: vec![ScheduledSequence {
                    request_id: RequestId(1),
                    token_ids: tokens.clone(),
                    start_position: start,
                    context_length: start + count,
                    block_table: vec![KvPageId(0), KvPageId(1)],
                    reservation_id: KvReservationId(index as u64),
                }],
            };
            let actual = engine
                .execute(&model, &batch, &device)
                .unwrap()
                .into_data()
                .to_vec::<f32>()
                .unwrap();
            let expected = model
                .forward_cached_last(
                    Tensor::from_data(TensorData::new(tokens, [1, count]), &device),
                    &mut dense_cache,
                )
                .into_data()
                .to_vec::<f32>()
                .unwrap();
            for (a, b) in actual.iter().zip(expected) {
                assert!((a - b).abs() < 2e-3, "{a} != {b}");
            }
            engine.commit_batch(&batch, &[count]).unwrap();
            start += count;
        }
    }
}
