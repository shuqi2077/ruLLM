//! Model loading, generation and scheduling on Ruda tensors and compute libraries.

mod ruda_inference;
mod generation;
mod huggingface;
mod llama;
mod continuous_batch;
mod batch_kv;
mod batch_prefix;
mod prefix_transfer;
pub use prefix_transfer::{PrefixShard,PrefixTensors,GpuPrefixBackend};
#[cfg(feature = "tensor-parallel")]
mod parallel_batch;
#[cfg(feature = "tensor-parallel")]
pub use parallel_batch::{ParallelBatchCache,ParallelBatchPrefix,ParallelBatchDelta};
pub use batch_kv::{DenseBatchCache,DenseBatchPrefix,DenseBatchDelta};
mod causal_draft_batch;
pub use causal_draft_batch::{CausalDraftBatch, CausalDraftBatchConfig, CausalDraftBatchCache,
    CausalDraftBatchPrefix, SpeculativeBatchModel, DenseCausalBatchModel,
    CausalBatchModel,CausalBatchCache,CausalBatchPrefix,CausalBatchDelta};
mod openai_server;
pub mod training;
mod paged_attention;
mod paged_engine;
mod paged_kv;
mod qwen35;
mod qwen3;
mod deepseek2;
pub use deepseek2::{DeepseekV2Config, DeepseekV2Yarn, DeepseekV2Model, DeepseekV2Cache,
    DeepseekV2CacheRecord, DeepseekV2AttentionMode, LoadedDeepseekV2, load_huggingface_deepseek_v2};
pub use deepseek2::{MlaFp8Backend, MlaFp8Projection, load_huggingface_mla};
pub use deepseek2::{DeepseekMtp, DeepseekMtpCache, DeepseekMtpGenerationCache, DeepseekMtpGenerationCacheRecord, DeepseekMtpOutput, LoadedDeepseekMtp,
    load_huggingface_deepseek_mtp, load_huggingface_mla_mtp};
pub use deepseek2::{KimiLinearConfig, KimiLinearAttentionConfig, KimiLinearModel, KimiLinearCache,
    KimiLinearCacheRecord, LoadedKimiLinear, load_huggingface_kimi_linear,
    KimiLinearBatchCache, KimiLinearBatchPrefix, KimiLinearBatchDelta};
pub use deepseek2::{DeepseekV2Config as MlaConfig, DeepseekV2Model as MlaModel,
    DeepseekV2Cache as MlaCache, LoadedDeepseekV2 as LoadedMla};
#[cfg(feature = "tensor-parallel")]
pub use deepseek2::parallel::{DeepseekV2ExpertParallel, DeepseekV2ExpertParallelCache,
    DeepseekV2ExpertParallelCacheRecord, LoadedDeepseekV2ExpertParallel, load_huggingface_deepseek_v2_expert_parallel};
#[cfg(feature = "tensor-parallel")]
pub use deepseek2::parallel::load_huggingface_mla_expert_parallel;
pub use deepseek2::{DeepseekMtpBatchCache,DeepseekMtpBatchPrefix};
#[cfg(feature="tensor-parallel")]
pub use deepseek2::{DeepseekMtpExpertParallel,DeepseekMtpExpertParallelCache,DeepseekMtpExpertParallelCacheRecord,
    LoadedDeepseekMtpExpertParallel,load_huggingface_deepseek_mtp_expert_parallel,load_huggingface_mla_mtp_expert_parallel};
#[cfg(feature = "tensor-parallel")]
pub use deepseek2::{KimiLinearExpertParallel, KimiLinearExpertParallelCache, KimiLinearExpertParallelCacheRecord,
    LoadedKimiLinearExpertParallel, load_huggingface_kimi_linear_expert_parallel};
pub use qwen3::{Qwen3Config,Qwen3MoeConfig,Qwen3Model,LoadedQwen3,load_huggingface_qwen3};
pub use qwen3::{Qwen3Drafted, Qwen3DraftCache, Qwen3DraftCacheRecord, load_huggingface_qwen3_draft};
pub use qwen3::{Qwen3DraftBatchCache,Qwen3DraftBatchPrefix};
#[cfg(feature = "tensor-parallel")]
pub use qwen3::parallel::{Qwen3TensorParallel, Qwen3TensorParallelCache, Qwen3TensorParallelCacheRecord,
    LoadedQwen3TensorParallel, load_huggingface_qwen3_tensor_parallel,Qwen3DraftTensorParallel};
pub mod prefix_cache;
pub mod disaggregated;
#[cfg(feature = "serving")]
pub mod serving;

pub use openai_server::{OpenAiServerConfig, serve_openai};
pub use training::{
    LlamaLora, LoraConfig, LoraTargets, SftDataset, SftLoader, SftTrainer, TrainConfig,
    export_huggingface_safetensors,
};
pub use training::{DpoTrainer, PpoTrainer, GrpoTrainer, RewardTrainer, ReferencePolicy, ValueCritic,
    Response, PagedResponse, ResponseRecord, PreferencePair, PpoConfig, GrpoConfig, PpoRollout, GrpoRollout, PostTrainingState};
pub use paged_engine::{BatchStepReport, ContinuousBatchRunner, PagedLlamaEngine};

pub use qwen35::{Qwen35TextConfig, Qwen35MoeConfig, Qwen35TextModel, Qwen35Cache, Qwen35LayerType, Qwen35RopeConfig, LoadedQwen35Text, load_huggingface_qwen35_text, load_huggingface_qwen35_text_packed};
pub use qwen35::{LoadedQwen35Vision, Qwen35VisionConfig, Qwen35VisionModel, Qwen35VisionOutput, load_huggingface_qwen35_vision};
pub use qwen35::{Qwen35MultimodalCache, Qwen35MultimodalModel, load_huggingface_qwen35_multimodal};
pub use qwen35::{Qwen35MultimodalCacheRecord,Qwen35ImageConditioned,Qwen35CachedMultimodalDecoder};
pub use qwen35::{Qwen35ImageProcessor, Qwen35PreparedImages, Qwen35RgbImage};
pub use qwen35::{Qwen35BatchCache,Qwen35BatchDelta};
#[cfg(feature = "tensor-parallel")]
pub use qwen35::parallel::{Qwen35TensorParallel, Qwen35TensorParallelCache,
    Qwen35TensorParallelCacheRecord, LoadedQwen35TensorParallel,
    load_huggingface_qwen35_tensor_parallel, load_huggingface_qwen35_tensor_parallel_packed};
pub use qwen35::{Qwen35CacheRecord,Qwen35LayerCacheState};

pub use generation::*;
pub use huggingface::*;
pub use llama::{
    LlamaAttention, LlamaConfig, LlamaConfigError, LlamaDecoderLayer, LlamaFeedForward,
    LlamaForCausalLm, LlamaKvCache, LlamaKvCacheRecord, LlamaLayerCache, PackedLlamaForCausalLm,
};
pub use llama::awq::{AwqBackend, AwqLlamaForCausalLm};
#[cfg(feature = "tensor-parallel")]
pub use llama::parallel::{
    LlamaTensorParallel, LlamaTensorParallelRank, LlamaTensorParallelCache,
    LlamaTensorParallelCacheRecord, LoadedLlamaTensorParallel,
    load_huggingface_llama_tensor_parallel, load_huggingface_qwen2_tensor_parallel,
};
pub use continuous_batch::{
    ContinuousBatchConfig, ContinuousBatchError, ContinuousBatchScheduler, ContinuousBatchSnapshot, PagedBatchSnapshot,
    FinishedGeneration, ScheduledBatch, ScheduledBatchKind, ScheduledSequence, SpeculativeConfig,
    CancelledGeneration, ContinuousBatchOptions, KvAdmissionPolicy,
    BatchFence, BatchLaunchFailure, FencedBatchScheduler,
    DeviceBatchModel, DeviceBatchExecutor, ExecutedBatch, BatchSamplingState,
    PrefixBatchModel,PrefixBatchExecutor,PrefixBatchStep,ActiveKvSwapConfig,ActiveKvSwapStats,
};
pub use paged_kv::*;

/// Optional backend adapters and host-side execution controls.
pub mod runtime;
pub mod backends;

/// Shared whole-generation calibration, enabled explicitly for native targets.
#[cfg(all(feature = "stack-autotune", any(target_os = "linux", target_os = "windows", target_os = "macos", target_os = "android")))]
pub mod autotune;

/// Shared device attention tensor transformations (candidate implementation).
pub mod gpu_inference;

/// Shared packed paged attention/MLA and local MoE building blocks.
/// Model adapters must provide their own projections, routing policy and cache ownership.
pub mod device_inference {
    pub use rudnn::paged_attention::{DevicePlan, HostPlan, PagedAttentionError};
    pub use rudnn::moe::{SwiGluExperts, RoutingOptions, GroupRoutingOptions, route_sigmoid_grouped, GroupedStrategy};
}
