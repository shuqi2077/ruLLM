//! Supervised fine-tuning and continued pre-training for packed Llama/Qwen2
//! models: memory-efficient training operators, LoRA, tokenized datasets
//! with resumable shuffling, an AdamW trainer with gradient accumulation,
//! global-norm clipping, warmup + cosine schedule, atomic checkpoints and
//! progress reporting, and Hugging Face Safetensors export.

pub(crate) mod data;
pub(crate) mod export;
pub(crate) mod lora;
pub(crate) mod ops;
pub(crate) mod trainer;

pub use data::{
    IGNORE_INDEX, SftBatch, SftDataset, SftExample, SftLoader, SftLoaderState, TextFormat,
};
pub use export::{export_huggingface_safetensors, write_safetensors};
pub use lora::{LlamaLora, LoraConfig, LoraLayer, LoraPair, LoraShapes, LoraTargets};
pub use ops::{
    FlashAttentionBackend, LinearCrossEntropyBackend, TrainingBackend, cross_entropy_chunk_rows,
    flash_attention, linear_cross_entropy,
};
pub use trainer::{
    LearningRateSchedule, SftTrainer, StepStats, TrainConfig, TrainError, TrainProgress,
    TrainerState,
};

#[cfg(all(test, feature = "nvidia"))]
mod tests;
