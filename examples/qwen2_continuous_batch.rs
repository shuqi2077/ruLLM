//! Continuous batching with paged KV on NVIDIA.
//!
//! usage: qwen2_continuous_batch <model-directory> <max-new-tokens> <prompt>...
//!
//! Every prompt is submitted at once; the scheduler interleaves batched
//! prefills and batched one-token decodes over a shared page pool. Each
//! request is then re-run alone through the dense single-sequence path and
//! the generated token IDs are compared.

use half::bf16;
use ruda_tensor::api::backend::Backend;
use ruda_tensor::api::{DType, Int, Tensor, TensorData};
use ruda_tensor_device::cuda::{Cuda, CudaDevice};
use rullm::{
    ContinuousBatchConfig, ContinuousBatchRunner, GreedyGenerationConfig, PagedKvCacheConfig,
    generate_greedy_packed_ruda, load_huggingface_qwen2_pipeline,
};
use std::collections::BTreeMap;
use std::{error::Error, io, time::Instant};

type ModelBackend = Cuda<bf16, i32>;

/// Largest top-2 logit gap (FP32 view of BF16 logits) treated as a tie: a few
/// BF16 ulps at the typical |logit| of 10-30.
const NEAR_TIE_LOGITS: f32 = 0.5;

fn main() -> Result<(), Box<dyn Error>> {
    let mut args = std::env::args().skip(1);
    let usage = || {
        io::Error::new(
            io::ErrorKind::InvalidInput,
            "usage: qwen2_continuous_batch <model-directory> <max-new-tokens> <prompt>...",
        )
    };
    let directory = args.next().ok_or_else(usage)?;
    let max_new_tokens: usize = args.next().ok_or_else(usage)?.parse()?;
    let prompts: Vec<String> = args.collect();
    if prompts.is_empty() || max_new_tokens == 0 {
        return Err(usage().into());
    }

    let device = CudaDevice::default();
    let started = Instant::now();
    let pipeline = load_huggingface_qwen2_pipeline::<ModelBackend>(&directory, &device)?;
    let config = pipeline.loaded.config.clone();
    let eos = pipeline.loaded.default_eos_token_ids.clone();
    let encoded = prompts
        .iter()
        .map(|prompt| pipeline.encode(prompt, false))
        .collect::<Result<Vec<_>, _>>()?;
    let model = pipeline.loaded.model.into_packed_inference();
    ModelBackend::sync(&device)?;
    println!(
        "{}",
        serde_json::json!({"stage": "loaded", "seconds": started.elapsed().as_secs_f64()})
    );

    let longest = encoded.iter().map(Vec::len).max().unwrap_or(1);
    let block_size = 16;
    let max_sequence_length = (longest + max_new_tokens).min(config.max_sequence_length);
    let pages_per_request = max_sequence_length.div_ceil(block_size);
    let kv = PagedKvCacheConfig {
        block_size,
        num_pages: pages_per_request * encoded.len(),
        max_sequence_length,
    };
    let batching = ContinuousBatchConfig {
        max_active_sequences: encoded.len(),
        max_batch_tokens: longest.max(encoded.len()).max(64),
    };
    let mut runner = ContinuousBatchRunner::new(&model, batching, kv, &device)?;
    println!(
        "{}",
        serde_json::json!({
            "stage": "kv-allocated", "pages": kv.num_pages, "block_size": block_size,
            "pool_mib": runner.engine().pool_bytes() as f64 / (1024.0 * 1024.0),
        })
    );

    let mut order = BTreeMap::new();
    for (index, prompt) in encoded.iter().enumerate() {
        let id = runner.submit(
            prompt.clone(),
            GreedyGenerationConfig {
                max_new_tokens,
                eos_token_ids: eos.clone(),
            },
        )?;
        order.insert(id, index);
    }

    let started = Instant::now();
    let mut steps = 0usize;
    let mut processed = 0usize;
    let mut finished = Vec::new();
    while let Some(report) = runner.step(&model, &device)? {
        steps += 1;
        processed += report.processed_tokens;
        while let Some(done) = runner.pop_finished() {
            finished.push(done);
        }
    }
    ModelBackend::sync(&device)?;
    let seconds = started.elapsed().as_secs_f64();
    let generated: usize = finished.iter().map(|f| f.generated_token_ids.len()).sum();
    println!(
        "{}",
        serde_json::json!({
            "stage": "batched", "seconds": seconds, "steps": steps,
            "processed_tokens": processed, "generated_tokens": generated,
            "generated_tokens_per_second": generated as f64 / seconds,
        })
    );
    if finished.len() != encoded.len() {
        return Err(io::Error::other(format!(
            "{} of {} requests finished",
            finished.len(),
            encoded.len()
        ))
        .into());
    }

    let mut mismatches = 0usize;
    let mut near_ties = 0usize;
    let mut hard_divergences = 0usize;
    finished.sort_by_key(|done| order[&done.request_id]);
    for done in &finished {
        let index = order[&done.request_id];
        let reference = generate_greedy_packed_ruda(
            &model,
            &config,
            &encoded[index],
            &GreedyGenerationConfig {
                max_new_tokens,
                eos_token_ids: eos.clone(),
            },
            &device,
        )?;
        let matches = reference.generated_token_ids == done.generated_token_ids;
        mismatches += usize::from(!matches);
        // BF16 batched and single-sequence runs use different GEMM shapes and
        // attention kernels, so greedy decoding can flip at a near-tie. At the
        // first divergence, score both candidate tokens with one dense
        // prefill over the shared context: a gap within `NEAR_TIE_LOGITS` is
        // a rounding-level tie, anything larger is a real defect.
        let divergence = reference
            .generated_token_ids
            .iter()
            .zip(&done.generated_token_ids)
            .position(|(a, b)| a != b);
        let mut tie_gap = None;
        if let Some(position) = divergence {
            let mut context = encoded[index].clone();
            context.extend_from_slice(&done.generated_token_ids[..position]);
            let tokens = Tensor::<ModelBackend, 2, Int>::from_data(
                TensorData::new(context.clone(), [1, context.len()]),
                &device,
            );
            let mut cache = model.new_cache();
            let logits = model
                .forward_cached_last_ruda(tokens, &mut cache)
                .cast(DType::F32)
                .into_data()
                .to_vec::<f32>()
                .map_err(|error| io::Error::other(format!("cannot read dense logits: {error:?}")))?;
            let single = reference.generated_token_ids[position] as usize;
            let batched = done.generated_token_ids[position] as usize;
            let gap = (logits[single] - logits[batched]).abs();
            tie_gap = Some(gap);
            if gap <= NEAR_TIE_LOGITS {
                near_ties += 1;
            } else {
                hard_divergences += 1;
            }
        }
        let ids = done
            .generated_token_ids
            .iter()
            .map(|&id| u32::try_from(id))
            .collect::<Result<Vec<_>, _>>()?;
        let text = pipeline.tokenizer.decode(&ids, true).map_err(io::Error::other)?;
        println!(
            "{}",
            serde_json::json!({
                "stage": "request", "index": index, "prompt": prompts[index],
                "generated_token_ids": done.generated_token_ids,
                "single_sequence_token_ids": reference.generated_token_ids,
                "matches_single_sequence": matches, "stopped_on_eos": done.stopped_on_eos,
                "first_divergence": divergence, "divergence_logit_gap": tie_gap,
                "text": text,
            })
        );
    }
    println!(
        "{}",
        serde_json::json!({
            "stage": "summary", "requests": finished.len(), "mismatches": mismatches,
            "near_tie_divergences": near_ties, "hard_divergences": hard_divergences,
            "near_tie_logits": NEAR_TIE_LOGITS,
        })
    );
    if hard_divergences > 0 {
        return Err(io::Error::other(format!("{hard_divergences} requests diverged beyond a near-tie")).into());
    }
    Ok(())
}
