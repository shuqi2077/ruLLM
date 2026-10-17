//! Supervised fine-tuning (LoRA or full) of a local Qwen2/Qwen2.5 checkpoint.
//!
//! usage: qwen2_sft <model-dir> <train.jsonl> <output-dir> [options]
//!   --steps 200            optimizer steps
//!   --lr 2e-4              peak learning rate (warmup 5%, cosine to 10%)
//!   --micro-batch 4        sequences per micro-batch
//!   --accumulation 4       micro-batches per optimizer step
//!   --max-length 1024      tokens per sequence (after truncation)
//!   --full                 full fine-tuning instead of LoRA
//!   --rank 16 --alpha 32   LoRA rank / alpha
//!   --checkpoint-every 50  optimizer steps between checkpoints (newest kept)
//!   --f32                  train in FP32 instead of BF16
//!
//! JSONL records: {"messages": [...]}, {"prompt", "completion"} or {"text"}.
//! Progress is one JSON line per step. Re-running with the same output
//! directory resumes from the newest checkpoint. At the end the merged
//! weights are written to `<output-dir>/model.safetensors`; copy
//! `config.json` and the tokenizer files from the source directory to load it.

use half::bf16;
use ruda_autodiff::Autodiff;
use ruda_tensor::api::backend::Backend;
use ruda_tensor_device::cuda::{Cuda, CudaDevice};
use rullm::training::{LoraTargets, TextFormat, TrainProgress};
use rullm::{
    LlamaLora, LoraConfig, SftDataset, SftLoader, SftTrainer, TrainConfig,
    export_huggingface_safetensors, load_huggingface_qwen2_pipeline,
};
use std::{error::Error, io, path::PathBuf, time::Instant};

struct Options {
    model: PathBuf,
    data: PathBuf,
    output: PathBuf,
    steps: usize,
    learning_rate: f64,
    micro_batch: usize,
    accumulation: usize,
    max_length: usize,
    full: bool,
    rank: usize,
    alpha: usize,
    checkpoint_every: usize,
    f32: bool,
}

fn parse() -> Result<Options, Box<dyn Error>> {
    let mut args = std::env::args().skip(1);
    let usage = || io::Error::new(io::ErrorKind::InvalidInput, "usage: qwen2_sft <model-dir> <train.jsonl> <output-dir> [options]");
    let mut options = Options {
        model: args.next().ok_or_else(usage)?.into(),
        data: args.next().ok_or_else(usage)?.into(),
        output: args.next().ok_or_else(usage)?.into(),
        steps: 200,
        learning_rate: 2e-4,
        micro_batch: 4,
        accumulation: 4,
        max_length: 1024,
        full: false,
        rank: 16,
        alpha: 32,
        checkpoint_every: 50,
        f32: false,
    };
    while let Some(flag) = args.next() {
        let mut value = || args.next().ok_or_else(|| io::Error::new(io::ErrorKind::InvalidInput, format!("{flag} needs a value")));
        match flag.as_str() {
            "--steps" => options.steps = value()?.parse()?,
            "--lr" => options.learning_rate = value()?.parse()?,
            "--micro-batch" => options.micro_batch = value()?.parse()?,
            "--accumulation" => options.accumulation = value()?.parse()?,
            "--max-length" => options.max_length = value()?.parse()?,
            "--full" => options.full = true,
            "--rank" => options.rank = value()?.parse()?,
            "--alpha" => options.alpha = value()?.parse()?,
            "--checkpoint-every" => options.checkpoint_every = value()?.parse()?,
            "--f32" => options.f32 = true,
            other => return Err(io::Error::new(io::ErrorKind::InvalidInput, format!("unknown option {other}")).into()),
        }
    }
    Ok(options)
}

fn report(progress: &TrainProgress) {
    let stats = progress.stats;
    println!(
        "{}",
        serde_json::json!({
            "stage": "train", "step": stats.step, "total_steps": progress.total_steps,
            "loss": stats.loss, "grad_norm": stats.grad_norm, "lr": stats.learning_rate,
            "applied": stats.applied, "skipped_steps": progress.skipped_steps,
            "epoch": progress.epoch, "tokens_seen": progress.tokens_seen,
            "elapsed_s": progress.elapsed.as_secs_f64(),
            "recent_tokens_per_s": progress.recent_tokens_per_second,
            "eta_s": progress.eta.map(|eta| eta.as_secs_f64()),
            "eta_basis": "mean seconds per step over the last logging window",
            "last_checkpoint_step": progress.last_checkpoint.map(|(step, _)| step),
            "last_checkpoint_age_s": progress.last_checkpoint.map(|(_, age)| age.as_secs_f64()),
        })
    );
}

fn run<B: ruda_tensor::api::backend::AutodiffBackend + rullm::training::TrainingBackend<Device = CudaDevice>>(
    options: Options,
) -> Result<(), Box<dyn Error>> {
    let device = CudaDevice::default();
    let started = Instant::now();
    let pipeline = load_huggingface_qwen2_pipeline::<B>(&options.model, &device)?;
    let eos = pipeline.tokenizer.token_to_id("<|im_end|>").map(|id| id as i32)
        .or_else(|| pipeline.loaded.default_eos_token_ids.first().copied());
    let (dataset, skipped) = SftDataset::from_jsonl(
        &options.data, &pipeline.tokenizer, TextFormat::QwenChat, eos, options.max_length + 1,
    )?;
    let tied = pipeline.loaded.report.tied_word_embeddings;
    let model = pipeline.loaded.model.into_packed_training();
    B::sync(&device)?;
    println!("{}", serde_json::json!({
        "stage": "loaded", "seconds": started.elapsed().as_secs_f64(), "examples": dataset.len(),
        "skipped_examples": skipped, "mode": if options.full { "full" } else { "lora" },
    }));
    let mut config = TrainConfig::new(options.steps, options.learning_rate, options.output.join("checkpoints"));
    config.gradient_accumulation = options.accumulation;
    config.checkpoint_every = options.checkpoint_every;
    let mut trainer = if options.full {
        SftTrainer::full(model, config, device.clone())
    } else {
        let adapters = LlamaLora::<B>::new(
            model.lora_shapes(),
            LoraConfig { rank: options.rank, alpha: options.alpha, targets: LoraTargets::ALL, seed: 0 },
            &device,
        );
        SftTrainer::lora(model, adapters, config, device.clone())
    };
    if let Some(step) = trainer.resume()? {
        println!("{}", serde_json::json!({"stage": "resumed", "step": step}));
    }
    let mut loader = SftLoader::new(dataset, options.micro_batch, eos.unwrap_or(0), 0);
    trainer.train(&mut loader, report)?;
    let (model, adapters) = trainer.into_parts();
    let model = match adapters {
        Some(adapters) => model.merge_lora(&adapters),
        None => model,
    };
    let path = options.output.join("model.safetensors");
    export_huggingface_safetensors(&model, &path, tied)?;
    println!("{}", serde_json::json!({"stage": "exported", "path": path, "seconds": started.elapsed().as_secs_f64()}));
    Ok(())
}

fn main() -> Result<(), Box<dyn Error>> {
    let options = parse()?;
    std::fs::create_dir_all(&options.output)?;
    if options.f32 {
        run::<Autodiff<Cuda<f32, i32>>>(options)
    } else {
        run::<Autodiff<Cuda<bf16, i32>>>(options)
    }
}
