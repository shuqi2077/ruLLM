use super::*;
use ruda_driver_cuda::{CudaDevice, CudaRuntime};
use ruda_kernel::tensor::readback::into_data_sync;

#[derive(serde::Deserialize)]
struct Reference {
    tokens: Vec<Vec<i32>>,
    shape: [usize; 3],
    logits: Vec<f32>,
}

#[test]
#[ignore = "requires independent GGML/PyTorch fixtures in RUDA_GGUF_FIXTURES"]
fn packed_gguf_models_match_independent_full_and_cached_logits() {
    let root = std::path::PathBuf::from(std::env::var_os("RUDA_GGUF_FIXTURES")
        .expect("RUDA_GGUF_FIXTURES must identify the independent model fixtures"));
    let device = CudaDevice::default();
    for architecture in ["llama", "qwen2", "qwen3"] {
        let path = root.join(format!("{architecture}.gguf"));
        let source = GgufModel::open(&path).unwrap();
        let expected: Reference = serde_json::from_slice(
            &std::fs::read(root.join(format!("{architecture}.json"))).unwrap(),
        ).unwrap();
        let loaded = load_gguf::<CudaRuntime>(&path, &device).unwrap();
        let model = loaded.model;
        let check_storage = |name: &str, weight: &GgmlGemm<CudaRuntime>| {
            let original = source.read_packed(name).unwrap();
            assert_eq!(weight.format(), original.info.dtype);
            assert_eq!(weight.packed_bytes(), original.bytes.len());
            assert_eq!(weight.original_storage().dtype, DType::U8);
            assert_eq!(into_data_sync(weight.original_storage().clone()).to_vec::<u8>().unwrap(), original.bytes);
        };
        check_storage("token_embd.weight", &model.embedding);
        check_storage("output_norm.weight", &model.norm);
        check_storage(if architecture == "qwen2" { "token_embd.weight" } else { "output.weight" }, &model.output.weight);
        for (index, layer) in model.layers.iter().enumerate() {
            for (name, projection) in [("attn_q", &layer.q), ("attn_k", &layer.k),
                ("attn_v", &layer.v), ("attn_output", &layer.o), ("ffn_gate", &layer.gate),
                ("ffn_up", &layer.up), ("ffn_down", &layer.down)] {
                check_storage(&format!("blk.{index}.{name}.weight"), &projection.weight);
                if let Some(bias) = &projection.bias {
                    check_storage(&format!("blk.{index}.{name}.bias"), bias);
                }
            }
        }
        let batch = expected.tokens.len();
        let length = expected.tokens[0].len();
        let tokens = Tensor::<GgufBackend<CudaRuntime>, 2, Int>::from_data(
            TensorData::new(expected.tokens.concat(), [batch, length]), &device,
        );
        let compare = |actual: Tensor<GgufBackend<CudaRuntime>, 3>| {
            assert_eq!(actual.dims(), expected.shape);
            let values = actual.into_data().to_vec::<f32>().unwrap();
            assert_eq!(values.len(), expected.logits.len());
            for (index, (actual, reference)) in values.iter().zip(&expected.logits).enumerate() {
                assert!(actual.is_finite() && (actual - reference).abs() <= 2e-4 * reference.abs().max(1.0),
                    "{architecture} logit {index}: {actual} != {reference}");
            }
        };
        compare(model.forward(tokens.clone()).unwrap());
        let mut cache = model.new_cache();
        let mut chunks = Vec::new();
        let mut start = 0;
        for size in [2, 1, 2] {
            chunks.push(model.forward_cached(tokens.clone().slice([0..batch, start..start + size]), &mut cache).unwrap());
            start += size;
            assert_eq!(cache.position, start);
        }
        compare(Tensor::cat(chunks, 1));
        let mut cache = model.new_cache();
        let last = model.forward_cached_last(tokens, &mut cache).unwrap().into_data().to_vec::<f32>().unwrap();
        for b in 0..batch {
            for v in 0..expected.shape[2] {
                let reference = expected.logits[(b * length + length - 1) * expected.shape[2] + v];
                assert!((last[b * expected.shape[2] + v] - reference).abs() <= 2e-4 * reference.abs().max(1.0));
            }
        }
        check_storage("token_embd.weight", &model.embedding);
    }
}
