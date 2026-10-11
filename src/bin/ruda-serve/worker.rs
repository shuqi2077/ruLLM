use ruda_tensor::api::backend::Backend;
use rullm::prefix_cache::PrefixCacheConfig;
use rullm::{
    CausalModel, CausalModelLimits, DeepseekV2AttentionMode, GenerationCacheContinuation,
    Qwen35Cache, Qwen35TextModel, load_huggingface_deepseek_v2, load_huggingface_llama,
    load_huggingface_qwen2, load_huggingface_qwen3, load_huggingface_qwen35_text, serving,
};
use std::path::PathBuf;
use tokio::sync::{mpsc, oneshot};

pub struct Options {
    pub directory: PathBuf,
    pub model_id: String,
    pub devices: Vec<usize>,
    pub prefill_only: bool,
    pub prefill_address: Option<String>,
    pub mla_mode: DeepseekV2AttentionMode,
    pub draft_directory: Option<PathBuf>,
    pub draft_id: Option<String>,
    pub cache: PrefixCacheConfig,
}

struct Worker {
    options: Options,
    source: serde_json::Value,
    tokenizer: Option<tokenizers::Tokenizer>,
    tokenizer_config: serde_json::Value,
    jobs: mpsc::Receiver<serving::Job>,
}

type Ready = Option<oneshot::Sender<Result<(), String>>>;

pub fn run(
    options: Options,
    backend: &str,
    libtorch_device: &str,
    jobs: mpsc::Receiver<serving::Job>,
    ready: oneshot::Sender<Result<(), String>>,
) {
    let mut ready = Some(ready);
    let result = (|| {
        let worker = Worker::new(options, jobs)?;
        match backend {
            "native" => {
                #[cfg(feature = "nvidia")]
                {
                    worker.run_cuda(&mut ready)
                }
                #[cfg(not(feature = "nvidia"))]
                {
                    Err("native CUDA serving requires the nvidia or nvidia-ptx feature".into())
                }
            }
            "libtorch" => {
                #[cfg(feature = "libtorch")]
                {
                    use ruda_tensor_tch::{LibTorch, LibTorchDevice};
                    let device = match libtorch_device {
                        "cpu" => LibTorchDevice::Cpu,
                        "cuda" => {
                            for &index in &worker.options.devices {
                                if !tch::Cuda::is_available()
                                    || index as u64 >= tch::Cuda::device_count() as u64
                                {
                                    return Err(format!(
                                        "LibTorch CUDA device {index} is unavailable"
                                    ));
                                }
                            }
                            LibTorchDevice::Cuda(worker.options.devices[0])
                        }
                        "mps" => {
                            if !tch::utils::has_mps() {
                                return Err("this LibTorch installation has no MPS support".into());
                            }
                            LibTorchDevice::Mps
                        }
                        _ => return Err("--libtorch-device must be cpu, cuda or mps".into()),
                    };
                    let _no_grad = tch::no_grad_guard();
                    if worker.options.devices.len() > 1 {
                        #[cfg(feature = "tensor-parallel")]
                        {
                            if libtorch_device != "cuda"
                                || !ruda_tensor_tch::distributed::nccl_available()
                            {
                                return Err("LibTorch TP/EP requires a Linux NCCL-enabled CUDA LibTorch build".into());
                            }
                            let devices = worker
                                .options
                                .devices
                                .iter()
                                .map(|&index| LibTorchDevice::Cuda(index))
                                .collect();
                            return worker
                                .run_parallel::<LibTorch<half::bf16>>(devices, &mut ready);
                        }
                        #[cfg(not(feature = "tensor-parallel"))]
                        {
                            return Err(
                                "multiple LibTorch devices require the tensor-parallel feature"
                                    .into(),
                            );
                        }
                    }
                    worker.run_single::<LibTorch<half::bf16>>(device, &mut ready)
                }
                #[cfg(not(feature = "libtorch"))]
                {
                    Err("LibTorch serving requires the libtorch feature".into())
                }
            }
            _ => Err("--backend must be native or libtorch".into()),
        }
    })();
    if let Err(error) = result {
        if let Some(ready) = ready {
            let _ = ready.send(Err(error.clone()));
        }
        eprintln!("{backend} inference worker: {error}");
    }
}

impl Worker {
    fn new(mut options: Options, jobs: mpsc::Receiver<serving::Job>) -> Result<Self, String> {
        let source: serde_json::Value = serde_json::from_slice(
            &std::fs::read(options.directory.join("config.json")).map_err(|e| e.to_string())?,
        )
        .map_err(|e| e.to_string())?;
        let model_type = source.get("model_type").and_then(serde_json::Value::as_str);
        if options.mla_mode != DeepseekV2AttentionMode::Expanded
            && model_type != Some("deepseek_v2")
        {
            return Err("absorbed MLA requires a DeepSeek-V2 adapter".into());
        }
        if options.draft_directory.is_some()
            && (options.devices.len() != 1 || model_type != Some("qwen3"))
        {
            return Err("DFlash/DSpark currently requires a single-device Qwen3 target".into());
        }
        if model_type == Some("deepseek_v2") {
            options.model_id = format!("{}:mla-{:?}", options.model_id, options.mla_mode);
        }
        if let Some(id) = &options.draft_id {
            options.model_id = format!("{}:draft-{id}", options.model_id);
        }
        let tokenizer = if options.prefill_only {
            None
        } else {
            Some(
                tokenizers::Tokenizer::from_file(options.directory.join("tokenizer.json"))
                    .map_err(|e| e.to_string())?,
            )
        };
        let tokenizer_config = if options.prefill_only {
            serde_json::Value::Null
        } else {
            serving::tokenizer_config(&options.directory)?
        };
        Ok(Self {
            options,
            source,
            tokenizer,
            tokenizer_config,
            jobs,
        })
    }

    fn launch<B: Backend, M: CausalModel<B>>(
        self,
        model: M,
        device: B::Device,
        limits: CausalModelLimits,
        eos: Vec<i32>,
        ready: &mut Ready,
    ) -> Result<(), String>
    where
        M::Cache: Clone + GenerationCacheContinuation<B>,
    {
        let on_ready = || ready.take().expect("worker readiness").send(Ok(())).is_ok();
        if self.options.prefill_only {
            rullm::disaggregated::run_prefill_service::<B, _>(
                model,
                device,
                limits,
                self.options.model_id,
                self.options.cache,
                self.options
                    .prefill_address
                    .as_deref()
                    .expect("prefill address"),
                on_ready,
            )
            .map_err(|e| e.to_string())
        } else {
            serving::run_native_with_prefill::<B, _>(
                model,
                device,
                limits,
                eos,
                self.tokenizer.expect("decode tokenizer"),
                self.tokenizer_config,
                self.options.model_id,
                self.options.cache,
                self.jobs,
                on_ready,
                self.options
                    .prefill_address
                    .map(|address| rullm::disaggregated::RemotePrefill { address }),
            )
        }
    }

    fn run_single<B: Backend>(self, device: B::Device, ready: &mut Ready) -> Result<(), String>
    where
        Qwen35TextModel<B>: CausalModel<B, Cache = Qwen35Cache<B>>,
    {
        match self
            .source
            .get("model_type")
            .and_then(serde_json::Value::as_str)
        {
            Some("deepseek_v2") => {
                let loaded = load_huggingface_deepseek_v2::<B>(&self.options.directory, &device)
                    .map_err(|e| e.to_string())?;
                let limits = loaded.model.config.limits();
                let model = loaded.model.with_attention_mode(self.options.mla_mode);
                self.launch::<B, _>(model, device, limits, loaded.eos_token_ids, ready)
            }
            Some("qwen3") => {
                let loaded = load_huggingface_qwen3::<B>(&self.options.directory, &device)
                    .map_err(|e| e.to_string())?;
                let limits = loaded.model.config.limits();
                if let Some(directory) = &self.options.draft_directory {
                    let model = rullm::load_huggingface_qwen3_draft(loaded.model, directory)
                        .map_err(|e| e.to_string())?;
                    return self.launch::<B, _>(model, device, limits, loaded.eos_token_ids, ready);
                }
                self.launch::<B, _>(loaded.model, device, limits, loaded.eos_token_ids, ready)
            }
            Some("qwen2") => {
                let loaded = load_huggingface_qwen2::<B>(&self.options.directory, &device)
                    .map_err(|e| e.to_string())?;
                self.launch::<B, _>(
                    loaded.model,
                    device,
                    CausalModelLimits::from(&loaded.config),
                    loaded.default_eos_token_ids,
                    ready,
                )
            }
            Some("llama") => {
                let loaded = load_huggingface_llama::<B>(&self.options.directory, &device)
                    .map_err(|e| e.to_string())?;
                self.launch::<B, _>(
                    loaded.model,
                    device,
                    CausalModelLimits::from(&loaded.config),
                    loaded.default_eos_token_ids,
                    ready,
                )
            }
            Some("qwen3_5" | "qwen3_5_text") => {
                let loaded = load_huggingface_qwen35_text::<B>(&self.options.directory, &device)
                    .map_err(|e| e.to_string())?;
                let limits = CausalModelLimits {
                    vocab_size: loaded.model.config.vocab_size,
                    max_sequence_length: loaded.model.config.max_position_embeddings,
                };
                let eos = vec![loaded.model.config.eos_token_id];
                self.launch::<B, _>(loaded.model, device, limits, eos, ready)
            }
            other => Err(format!("no server model adapter for {other:?}")),
        }
    }

    #[cfg(feature = "nvidia")]
    fn run_cuda(self, ready: &mut Ready) -> Result<(), String> {
        use ruda_tensor_device::cuda::{Cuda, CudaDevice};
        type B = Cuda<half::bf16, i32>;
        let device = CudaDevice::new(self.options.devices[0]);
        if self.options.devices.len() == 1 {
            return self.run_single::<B>(device, ready);
        }
        #[cfg(feature = "tensor-parallel")]
        {
            let devices = self
                .options
                .devices
                .iter()
                .map(|&id| CudaDevice::new(id))
                .collect();
            self.run_parallel::<B>(devices, ready)
        }
        #[cfg(not(feature = "tensor-parallel"))]
        {
            Err("multiple native devices require the tensor-parallel feature".into())
        }
    }
    #[cfg(feature = "tensor-parallel")]
    fn run_parallel<B: ruda_tensor::distributed::DistributedBackend>(
        self,
        devices: Vec<B::Device>,
        ready: &mut Ready,
    ) -> Result<(), String> {
        let device = devices[0].clone();
        match self
            .source
            .get("model_type")
            .and_then(serde_json::Value::as_str)
        {
            Some("deepseek_v2") => {
                let loaded = rullm::load_huggingface_deepseek_v2_expert_parallel::<B>(
                    &self.options.directory,
                    devices,
                )
                .map_err(|e| e.to_string())?;
                let limits = loaded.model.config.limits();
                let model = loaded.model.with_attention_mode(self.options.mla_mode);
                self.launch::<B, _>(model, device, limits, loaded.eos_token_ids, ready)
            }
            Some("qwen3") => {
                let loaded = rullm::load_huggingface_qwen3_tensor_parallel::<B>(
                    &self.options.directory,
                    devices,
                )
                .map_err(|e| e.to_string())?;
                let limits = loaded.model.config.limits();
                self.launch::<B, _>(loaded.model, device, limits, loaded.eos_token_ids, ready)
            }
            other => Err(format!(
                "tensor/expert parallel serving is not implemented for {other:?}"
            )),
        }
    }
}
