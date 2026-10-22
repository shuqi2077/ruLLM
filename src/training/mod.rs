//! Supervised fine-tuning and continued pre-training for Llama/Qwen2,
//! Qwen3/Qwen3.5 dense/MoE, DeepSeek MLA and Kimi Linear: training operators,
//! LoRA, tokenized datasets
//! with resumable shuffling, an AdamW trainer with gradient accumulation,
//! global-norm clipping, warmup + cosine schedule, atomic checkpoints and
//! progress reporting, and Hugging Face Safetensors export.

pub(crate) mod data;
pub(crate) mod export;
pub(crate) mod lora;
pub(crate) mod ops;
pub(crate) mod log_probs;
pub mod post_training;
pub mod reward;
mod rollout;
pub(crate) mod trainer;
mod model;
pub(crate) mod native;
mod loading;

pub use model::SftModel;
pub use native::{NativeLora, NativeSftModel};
pub use loading::load_huggingface_native_sft;

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
pub use log_probs::{LinearLogProbsBackend, linear_log_probs};
pub use reward::RewardTrainer;
pub use post_training::{DpoTrainer, GrpoTrainer, PpoTrainer, ReferencePolicy, ValueCritic,
    Response, PagedResponse, ResponseRecord, PreferencePair, PpoConfig, GrpoConfig, PpoRollout, GrpoRollout, PostTrainingState};

#[cfg(all(test, feature = "nvidia"))]
mod tests;
