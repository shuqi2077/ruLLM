//! Model loading, generation and scheduling on Ruda tensors and compute libraries.

mod ruda_inference;
mod generation;
mod huggingface;
mod llama;
mod continuous_batch;
mod paged_kv;
mod qwen35;
mod qwen3;
mod deepseek2;
pub use deepseek2::{DeepseekV2Config, DeepseekV2Yarn, DeepseekV2Model, DeepseekV2Cache,
    DeepseekV2CacheRecord, DeepseekV2AttentionMode, LoadedDeepseekV2, load_huggingface_deepseek_v2};
#[cfg(feature = "tensor-parallel")]
pub use deepseek2::parallel::{DeepseekV2ExpertParallel, DeepseekV2ExpertParallelCache,
    DeepseekV2ExpertParallelCacheRecord, LoadedDeepseekV2ExpertParallel, load_huggingface_deepseek_v2_expert_parallel};
pub use qwen3::{Qwen3Config,Qwen3Model,LoadedQwen3,load_huggingface_qwen3};
pub use qwen3::{Qwen3Drafted, Qwen3DraftCache, Qwen3DraftCacheRecord, load_huggingface_qwen3_draft};
#[cfg(feature = "tensor-parallel")]
pub use qwen3::parallel::{Qwen3TensorParallel, Qwen3TensorParallelCache, Qwen3TensorParallelCacheRecord,
    LoadedQwen3TensorParallel, load_huggingface_qwen3_tensor_parallel};
pub mod prefix_cache;
pub mod disaggregated;
#[cfg(feature = "serving")]
pub mod serving;

pub use qwen35::{Qwen35TextConfig, Qwen35TextModel, Qwen35Cache, Qwen35LayerType, Qwen35RopeConfig, LoadedQwen35Text, load_huggingface_qwen35_text, load_huggingface_qwen35_text_packed};
pub use qwen35::{LoadedQwen35Vision, Qwen35VisionConfig, Qwen35VisionModel, Qwen35VisionOutput, load_huggingface_qwen35_vision};
pub use qwen35::{Qwen35MultimodalCache, Qwen35MultimodalModel, load_huggingface_qwen35_multimodal};
pub use qwen35::{Qwen35MultimodalCacheRecord,Qwen35ImageConditioned,Qwen35CachedMultimodalDecoder};
pub use qwen35::{Qwen35ImageProcessor, Qwen35PreparedImages, Qwen35RgbImage};
pub use qwen35::Qwen35BatchCache;
pub use qwen35::{Qwen35CacheRecord,Qwen35LayerCacheState};

pub use generation::*;
pub use huggingface::*;
pub use llama::{
    LlamaAttention, LlamaConfig, LlamaConfigError, LlamaDecoderLayer, LlamaFeedForward,
    LlamaForCausalLm, LlamaKvCache, LlamaKvCacheRecord, LlamaLayerCache, PackedLlamaForCausalLm,
};
pub use llama::awq::{AwqBackend, AwqLlamaForCausalLm};
pub use continuous_batch::{
    ContinuousBatchConfig, ContinuousBatchError, ContinuousBatchScheduler, ContinuousBatchSnapshot,
    FinishedGeneration, ScheduledBatch, ScheduledBatchKind, ScheduledSequence,
    CancelledGeneration, ContinuousBatchOptions, KvAdmissionPolicy,
    BatchFence, BatchLaunchFailure, FencedBatchScheduler,
    DeviceBatchModel, DeviceBatchExecutor, ExecutedBatch,
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
