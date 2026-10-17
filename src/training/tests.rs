//! GPU checks of the training operators against composed autodiff
//! references, and end-to-end trainer behaviour on a tiny model.

use super::*;
use crate::{LlamaConfig, LlamaForCausalLm};
use half::bf16;
use ruda_autodiff::Autodiff;
use ruda_tensor::api::activation::{log_softmax, softmax};
use ruda_tensor::api::backend::{AutodiffBackend, Backend};
use ruda_tensor::api::{DType, Int, Tensor, TensorData};
use ruda_tensor_device::cuda::{Cuda, CudaDevice};

type F32Train = Autodiff<Cuda<f32, i32>>;
type Bf16Train = Autodiff<Cuda<bf16, i32>>;

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

/// Composed causal GQA attention on `[B, S, H, D]` inputs.
fn reference_attention<B: Backend>(
    query: Tensor<B, 4>,
    key: Tensor<B, 4>,
    value: Tensor<B, 4>,
    window: usize,
) -> Tensor<B, 4> {
    let [batch, sequence, query_heads, head_dim] = query.dims();
    let kv_heads = key.dims()[2];
    let group = query_heads / kv_heads;
    let expand = |tensor: Tensor<B, 4>| {
        tensor
            .swap_dims(1, 2)
            .unsqueeze_dim::<5>(2)
            .repeat_dim(2, group)
            .reshape([batch, query_heads, sequence, head_dim])
    };
    let mut mask = Vec::with_capacity(sequence * sequence);
    for row in 0..sequence {
        for column in 0..sequence {
            let visible = column <= row && (window == 0 || row - column < window);
            mask.push(if visible { 0.0f32 } else { -1.0e9 });
        }
    }
    let mask = Tensor::<B, 4>::from_data(
        TensorData::new(mask, [1, 1, sequence, sequence]),
        (&query.device(), DType::F32),
    );
    let scale = 1.0 / (head_dim as f32).sqrt();
    let scores = query
        .swap_dims(1, 2)
        .cast(DType::F32)
        .matmul(expand(key).cast(DType::F32).swap_dims(2, 3))
        .mul_scalar(scale)
        + mask;
    softmax(scores, 3)
        .matmul(expand(value).cast(DType::F32))
        .swap_dims(1, 2)
}

fn max_difference<B: Backend, const D: usize>(left: Tensor<B, D>, right: Tensor<B, D>) -> f32 {
    let left = left.cast(DType::F32).into_data().to_vec::<f32>().unwrap();
    let right = right.cast(DType::F32).into_data().to_vec::<f32>().unwrap();
    assert_eq!(left.len(), right.len());
    left.iter()
        .zip(&right)
        .map(|(a, b)| (a - b).abs())
        .fold(0.0, f32::max)
}

fn check_attention_gradients<B: AutodiffBackend + FlashAttentionBackend>(
    dtype: DType,
    head_dim: usize,
    window: usize,
    tolerance: f32,
) {
    let device = Default::default();
    let [batch, sequence, query_heads, kv_heads] = [2usize, 37, 4, 2];
    let tensor = |seed, heads| {
        Tensor::<B, 4>::from_data(
            TensorData::new(values(batch * sequence * heads * head_dim, seed), [batch, sequence, heads, head_dim]),
            (&device, DType::F32),
        )
        .cast(dtype)
        .detach()
        .require_grad()
    };
    let (query, key, value) = (tensor(1, query_heads), tensor(2, kv_heads), tensor(3, kv_heads));
    let weights = Tensor::<B, 4>::from_data(
        TensorData::new(values(batch * sequence * query_heads * head_dim, 4), [batch, sequence, query_heads, head_dim]),
        (&device, DType::F32),
    );
    let fused = flash_attention(query.clone(), key.clone(), value.clone(), window);
    let mut fused_grads = (fused.clone().cast(DType::F32) * weights.clone()).sum().backward();
    let reference = reference_attention(query.clone(), key.clone(), value.clone(), window);
    let mut reference_grads = (reference.clone() * weights).sum().backward();
    let output = max_difference(fused.inner(), reference.inner());
    let mut worst = output;
    for (name, input) in [("query", &query), ("key", &key), ("value", &value)] {
        let fused = input.grad_remove(&mut fused_grads).unwrap();
        let reference = input.grad_remove(&mut reference_grads).unwrap();
        let difference = max_difference(fused, reference);
        eprintln!("flash attention {dtype:?} d={head_dim} window={window} d{name} max_abs={difference:e}");
        worst = worst.max(difference);
    }
    assert!(worst <= tolerance, "flash attention gradient error {worst} > {tolerance}");
}

#[test]
fn flash_attention_backward_matches_composed_reference() {
    for window in [0, 7] {
        check_attention_gradients::<F32Train>(DType::F32, 40, window, 2.0e-4);
        check_attention_gradients::<Bf16Train>(DType::BF16, 64, window, 6.0e-2);
    }
}

#[test]
fn linear_cross_entropy_matches_log_softmax_reference() {
    type B = F32Train;
    let device = Default::default();
    let [rows, width, vocabulary] = [13usize, 24, 50];
    let hidden = Tensor::<B, 2>::from_data(TensorData::new(values(rows * width, 5), [rows, width]), &device)
        .require_grad();
    let weight = Tensor::<B, 2>::from_data(TensorData::new(values(width * vocabulary, 6), [width, vocabulary]), &device)
        .require_grad();
    let targets: Vec<i32> = (0..rows as i32).map(|row| if row % 4 == 3 { -100 } else { row * 7 % 50 }).collect();
    let valid = targets.iter().filter(|&&target| target >= 0).count();
    let target_tensor = Tensor::<B, 1, Int>::from_data(TensorData::new(targets.clone(), [rows]), &device);
    // A chunk smaller than the row count exercises the chunked path.
    let fused = linear_cross_entropy(hidden.clone(), weight.clone(), target_tensor, valid, 4);
    let mut fused_grads = fused.clone().mul_scalar(3.0).backward();

    let log_probabilities = log_softmax(hidden.clone().matmul(weight.clone()), 1);
    let safe: Vec<i32> = targets.iter().map(|&target| target.max(0)).collect();
    let mask: Vec<f32> = targets.iter().map(|&target| if target >= 0 { 1.0 } else { 0.0 }).collect();
    let picked = log_probabilities.gather(
        1,
        Tensor::<B, 2, Int>::from_data(TensorData::new(safe, [rows, 1]), &device),
    );
    let mask = Tensor::<B, 2>::from_data(TensorData::new(mask, [rows, 1]), &device);
    let reference = (picked * mask).sum().neg().div_scalar(valid as f32);
    let mut reference_grads = reference.clone().mul_scalar(3.0).backward();

    assert!(max_difference(fused.inner(), reference.inner()) < 1e-5);
    for input in [&hidden, &weight] {
        let difference = max_difference(
            input.grad_remove(&mut fused_grads).unwrap(),
            input.grad_remove(&mut reference_grads).unwrap(),
        );
        assert!(difference < 1e-5, "cross-entropy gradient error {difference}");
    }
}

fn tiny_config() -> LlamaConfig {
    LlamaConfig {
        vocab_size: 48,
        d_model: 64,
        d_ff: 96,
        num_hidden_layers: 2,
        num_query_heads: 4,
        num_kv_heads: 2,
        max_sequence_length: 64,
        rms_norm_epsilon: 1.0e-6,
        rope_theta: 10_000.0,
    }
}

fn tiny_dataset() -> SftDataset {
    SftDataset::new(
        (0..8)
            .map(|index| {
                let ids: Vec<i32> = (0..12).map(|step| (index * 3 + step * 5) % 48).collect();
                let mut labels = ids.clone();
                labels[..4].fill(IGNORE_INDEX);
                SftExample { input_ids: ids, labels }
            })
            .collect(),
    )
}

fn run_trainer(lora: bool, directory: &std::path::Path, steps: usize) -> (Vec<f32>, TrainerState) {
    type B = F32Train;
    let device = CudaDevice::default();
    let model = LlamaForCausalLm::<B>::init_qwen2(&tiny_config(), &device)
        .unwrap()
        .into_packed_training();
    let mut config = TrainConfig::new(steps, 3.0e-3, directory);
    config.gradient_accumulation = 2;
    config.checkpoint_every = 2;
    config.keep_checkpoints = 1;
    let mut trainer = if lora {
        let adapters = LlamaLora::<B>::new(
            model.lora_shapes(),
            LoraConfig { rank: 4, alpha: 8, targets: LoraTargets::ALL, seed: 3 },
            &device,
        );
        SftTrainer::lora(model, adapters, config, device)
    } else {
        SftTrainer::full(model, config, device)
    };
    trainer.resume().unwrap();
    let mut loader = SftLoader::new(tiny_dataset(), 2, 0, 11);
    let mut losses = Vec::new();
    let state = trainer
        .train(&mut loader, |progress| losses.push(progress.stats.loss))
        .unwrap();
    (losses, state)
}

#[test]
fn trainer_reduces_loss_checkpoints_and_resumes() {
    for lora in [false, true] {
        let directory = std::env::temp_dir().join(format!("rullm-trainer-test-{lora}-{}", std::process::id()));
        let _ = std::fs::remove_dir_all(&directory);
        let (losses, state) = run_trainer(lora, &directory, 12);
        eprintln!("trainer lora={lora} losses={losses:?}");
        assert_eq!(state.step, 12);
        assert!(losses.last().unwrap() < losses.first().unwrap(), "loss must decrease");
        let checkpoints: Vec<_> = std::fs::read_dir(&directory)
            .unwrap()
            .filter_map(Result::ok)
            .filter(|entry| entry.file_name().to_string_lossy().starts_with("step-"))
            .collect();
        assert_eq!(checkpoints.len(), 1, "only the newest checkpoint is kept");
        // Resuming a finished run performs no further steps.
        let (more, resumed) = run_trainer(lora, &directory, 12);
        assert!(more.is_empty());
        assert_eq!(resumed.step, 12);
        assert_eq!(resumed.loader, state.loader);
        std::fs::remove_dir_all(&directory).unwrap();
    }
}

#[test]
fn merged_lora_matches_adapter_forward_and_exports() {
    type B = F32Train;
    let device = CudaDevice::default();
    let model = LlamaForCausalLm::<B>::init_qwen2(&tiny_config(), &device)
        .unwrap()
        .into_packed_training();
    let mut adapters = LlamaLora::<B>::new(model.lora_shapes(), LoraConfig::default(), &device);
    for layer in &mut adapters.layers {
        if let Some(pair) = layer.qkv.as_mut() {
            pair.b = ruda_model::module::Param::from_tensor(
                Tensor::random(pair.b.dims(), ruda_tensor::Distribution::Normal(0.0, 0.05), &device).detach(),
            );
        }
    }
    let tokens = Tensor::<B, 2, Int>::from_data(TensorData::new((0..10).collect::<Vec<i32>>(), [1, 10]), &device);
    let adapted = model.forward_train_hidden(tokens.clone(), Some(&adapters));
    let merged = model.merge_lora(&adapters);
    let plain = merged.forward_train_hidden(tokens, None);
    assert!(max_difference(adapted.inner(), plain.inner()) < 1e-4);
    let path = std::env::temp_dir().join(format!("rullm-export-{}.safetensors", std::process::id()));
    export_huggingface_safetensors(&merged, &path, true).unwrap();
    let bytes = std::fs::read(&path).unwrap();
    let header_length = u64::from_le_bytes(bytes[..8].try_into().unwrap()) as usize;
    let header: serde_json::Value = serde_json::from_slice(&bytes[8..8 + header_length]).unwrap();
    assert_eq!(header["model.layers.0.self_attn.q_proj.weight"]["shape"], serde_json::json!([64, 64]));
    assert_eq!(header["model.layers.1.self_attn.k_proj.bias"]["shape"], serde_json::json!([32]));
    assert!(header.get("lm_head.weight").is_none());
    std::fs::remove_file(&path).unwrap();
}

/// Merge + export at a realistic hidden width in BF16 (the Qwen2.5-0.5B
/// shapes: hidden 896, 14/2 heads, rank 8) against a host computation, both
/// on the device weights and on the bytes written to Safetensors.
#[test]
fn bf16_lora_merge_and_export_match_host_at_model_width() {
    type B = Bf16Train;
    let device = CudaDevice::default();
    let config = LlamaConfig {
        vocab_size: 64,
        d_model: 896,
        d_ff: 1024,
        num_hidden_layers: 1,
        num_query_heads: 14,
        num_kv_heads: 2,
        max_sequence_length: 32,
        rms_norm_epsilon: 1.0e-6,
        rope_theta: 10_000.0,
    };
    let model = LlamaForCausalLm::<B>::init_qwen2(&config, &device).unwrap().into_packed_training();
    let mut adapters = LlamaLora::<B>::new(
        model.lora_shapes(),
        LoraConfig { rank: 8, alpha: 16, targets: LoraTargets::ALL, seed: 1 },
        &device,
    );
    for layer in &mut adapters.layers {
        for pair in [layer.qkv.as_mut(), layer.down.as_mut()].into_iter().flatten() {
            pair.b = ruda_model::module::Param::from_tensor(
                Tensor::<B, 2>::random(pair.b.dims(), ruda_tensor::Distribution::Normal(0.0, 0.05), &device)
                    .cast(DType::F32)
                    .detach(),
            );
        }
    }
    let host = |tensor: Tensor<B, 2>| -> Vec<f32> {
        tensor.cast(DType::F32).into_data().to_vec::<f32>().unwrap()
    };
    let scale = adapters.scale() as f32;
    let rank = 8usize;
    // expected[i][o] = bf16(bf16(W[i][o]) + bf16(scale * (A B)[i][o])) for W [in, out].
    let expected = |weight: &[f32], a: &[f32], b: &[f32], inputs: usize, outputs: usize| -> Vec<f32> {
        let mut result = vec![0.0f32; inputs * outputs];
        for i in 0..inputs {
            for o in 0..outputs {
                let delta: f32 = (0..rank).map(|r| a[i * rank + r] * b[r * outputs + o]).sum::<f32>() * scale;
                let sum = bf16::from_f32(weight[i * outputs + o]).to_f32() + bf16::from_f32(delta).to_f32();
                result[i * outputs + o] = bf16::from_f32(sum).to_f32();
            }
        }
        result
    };
    let check = |label: &str, actual: &[f32], expected: &[f32]| {
        assert_eq!(actual.len(), expected.len(), "{label} length");
        let mut worst = 0.0f32;
        for (index, (&a, &e)) in actual.iter().zip(expected).enumerate() {
            assert!(a.is_finite(), "{label}[{index}] is {a}");
            let error = (a - e).abs() / e.abs().max(1.0);
            worst = worst.max(error);
            assert!(error <= 1.6e-2, "{label}[{index}]: {a} != {e}");
        }
        eprintln!("{label}: max relative error {worst:e}");
    };

    let layer = &model.layers[0];
    let pair_qkv = adapters.layers[0].qkv.as_ref().unwrap();
    let pair_down = adapters.layers[0].down.as_ref().unwrap();
    let [hidden, qkv_out] = layer.self_attn.qkv_weight.dims();
    let d_ff = layer.mlp.d_ff;
    let expected_qkv = expected(&host(layer.self_attn.qkv_weight.val()), &host(pair_qkv.a.val()),
        &host(pair_qkv.b.val()), hidden, qkv_out);
    let expected_down = expected(&host(layer.mlp.down_proj.weight.val()), &host(pair_down.a.val()),
        &host(pair_down.b.val()), d_ff, hidden);

    let merged = model.merge_lora(&adapters);
    check("device qkv", &host(merged.layers[0].self_attn.qkv_weight.val()), &expected_qkv);
    check("device down", &host(merged.layers[0].mlp.down_proj.weight.val()), &expected_down);

    let path = std::env::temp_dir().join(format!("rullm-export-wide-{}.safetensors", std::process::id()));
    export_huggingface_safetensors(&merged, &path, true).unwrap();
    let bytes = std::fs::read(&path).unwrap();
    std::fs::remove_file(&path).unwrap();
    let header_length = u64::from_le_bytes(bytes[..8].try_into().unwrap()) as usize;
    let header: serde_json::Value = serde_json::from_slice(&bytes[8..8 + header_length]).unwrap();
    let read = |name: &str| -> (Vec<usize>, Vec<f32>) {
        let entry = &header[name];
        assert_eq!(entry["dtype"], "BF16", "{name} dtype");
        let shape: Vec<usize> = serde_json::from_value(entry["shape"].clone()).unwrap();
        let [start, end]: [usize; 2] = serde_json::from_value(entry["data_offsets"].clone()).unwrap();
        let data = &bytes[8 + header_length + start..8 + header_length + end];
        let values = data
            .chunks_exact(2)
            .map(|pair| bf16::from_le_bytes([pair[0], pair[1]]).to_f32())
            .collect();
        (shape, values)
    };
    // Hugging Face layout is [out, in]: transpose the [in, out] expectation.
    let transpose = |values: &[f32], inputs: usize, outputs: usize, columns: std::ops::Range<usize>| -> Vec<f32> {
        columns.flat_map(|o| (0..inputs).map(move |i| values[i * outputs + o])).collect()
    };
    let query_width = config.num_query_heads * (config.d_model / config.num_query_heads);
    let (shape, q) = read("model.layers.0.self_attn.q_proj.weight");
    assert_eq!(shape, vec![query_width, hidden]);
    check("exported q_proj", &q, &transpose(&expected_qkv, hidden, qkv_out, 0..query_width));
    let (shape, down) = read("model.layers.0.mlp.down_proj.weight");
    assert_eq!(shape, vec![hidden, d_ff]);
    check("exported down_proj", &down, &transpose(&expected_down, d_ff, hidden, 0..hidden));
}

/// BF16 backend + FP32 LoRA adapters: a checkpoint must restore the adapters
/// bit-exactly in FP32 (the module recorders would convert them to BF16) and
/// training must continue from the same loss as an uninterrupted run.
#[test]
fn bf16_lora_resume_keeps_fp32_adapters_exact() {
    type B = Bf16Train;
    let device = CudaDevice::default();
    let directory = std::env::temp_dir().join(format!("rullm-resume-bf16-{}", std::process::id()));
    let _ = std::fs::remove_dir_all(&directory);
    let build = |steps: usize| {
        let model = LlamaForCausalLm::<B>::init_qwen2(&tiny_config(), &device).unwrap().into_packed_training();
        let adapters = LlamaLora::<B>::new(
            model.lora_shapes(),
            LoraConfig { rank: 4, alpha: 8, targets: LoraTargets::ALL, seed: 3 },
            &device,
        );
        let mut config = TrainConfig::new(steps, 3.0e-3, &directory);
        config.gradient_accumulation = 2;
        config.checkpoint_every = 2;
        config.keep_checkpoints = 1;
        SftTrainer::lora(model, adapters, config, device.clone())
    };
    let adapter_values = |adapters: &LlamaLora<B>| -> Vec<(DType, Vec<u8>)> {
        let pair = adapters.layers[0].qkv.as_ref().unwrap();
        [pair.a.val(), pair.b.val()]
            .into_iter()
            .map(|tensor| (tensor.dtype(), tensor.into_data().as_bytes().to_vec()))
            .collect()
    };
    let mut first = build(4);
    let mut loader = SftLoader::new(tiny_dataset(), 2, 0, 11);
    first.train(&mut loader, |_| {}).unwrap();
    let (_, trained) = first.into_parts();
    let trained = adapter_values(trained.as_ref().unwrap());
    assert!(trained.iter().all(|(dtype, _)| *dtype == DType::F32), "adapters must train in FP32");

    let mut resumed = build(4);
    assert_eq!(resumed.resume().unwrap(), Some(4));
    let (_, restored) = resumed.into_parts();
    assert_eq!(adapter_values(restored.as_ref().unwrap()), trained, "restored adapters differ");
    std::fs::remove_dir_all(&directory).unwrap();
}
