use ruda_tensor::api::backend::Backend;
use rullm::prefix_cache::PrefixCacheConfig;
use rullm::{
    CausalModel, CausalModelLimits, DeepseekV2AttentionMode, GenerationCacheContinuation,
    Qwen35Cache, Qwen35TextModel, load_huggingface_mla, load_huggingface_llama,
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
    #[cfg(feature="gpu-kv")]
    pub gpu_kv:Option<rullm::disaggregated::GpuKvConfig>,
    pub mla_mode: DeepseekV2AttentionMode,
    pub draft_directory: Option<PathBuf>,
    pub draft_id: Option<String>,
    pub causal_draft_tokens: Option<usize>,
    pub mtp: bool,
    pub cache: PrefixCacheConfig,
    pub batch: Option<serving::BatchServingConfig>,
    pub multimodal: bool,
}

struct Worker {
    options: Options,
    source: serde_json::Value,
    tokenizer: Option<tokenizers::Tokenizer>,
    tokenizer_config: serde_json::Value,
    jobs: mpsc::Receiver<serving::Job>,
}

type Ready = Option<oneshot::Sender<Result<(), String>>>;

#[cfg(feature="nvidia")]
fn load_causal_batch(directory:&std::path::Path,device:&ruda_tensor_device::cuda::CudaDevice,mla:DeepseekV2AttentionMode)
    ->Result<(rullm::CausalBatchModel<rullm::backends::Nvidia<half::bf16,i32,u8>>,Vec<i32>),String> {
    type B=rullm::backends::Nvidia<half::bf16,i32,u8>;
    use rullm::DenseCausalBatchModel as Model;
    let source:serde_json::Value=serde_json::from_slice(&std::fs::read(directory.join("config.json")).map_err(|e|e.to_string())?)
        .map_err(|e|e.to_string())?;
    match source.get("model_type").and_then(serde_json::Value::as_str) {
        Some("llama")=>{
            let loaded=load_huggingface_llama::<B>(directory,device).map_err(|e|e.to_string())?;
            Ok((Model::Llama(loaded.model).into(),loaded.default_eos_token_ids))
        },
        Some("qwen2")=>{
            let loaded=load_huggingface_qwen2::<B>(directory,device).map_err(|e|e.to_string())?;
            Ok((Model::Llama(loaded.model).into(),loaded.default_eos_token_ids))
        },
        Some("qwen3"|"qwen3_moe")=>{
            let loaded=load_huggingface_qwen3::<B>(directory,device).map_err(|e|e.to_string())?;
            Ok((Model::Qwen3(loaded.model).into(),loaded.eos_token_ids))
        },
        Some("deepseek_v2"|"deepseek_v3"|"kimi_k2")=>{
            let loaded=load_huggingface_mla::<B>(directory,device).map_err(|e|e.to_string())?;
            Ok((Model::DeepseekV2(loaded.model.with_attention_mode(mla)).into(),loaded.eos_token_ids))
        },
        Some("qwen3_5"|"qwen3_5_text"|"qwen3_5_moe"|"qwen3_5_moe_text")=>{
            let loaded=load_huggingface_qwen35_text::<B>(directory,device).map_err(|e|e.to_string())?;
            let eos=vec![loaded.model.config.eos_token_id];
            Ok((rullm::CausalBatchModel::Qwen35(loaded.model),eos))
        },
        Some("kimi_linear")=>{
            let loaded=rullm::load_huggingface_kimi_linear::<B>(directory,device).map_err(|e|e.to_string())?;
            Ok((rullm::CausalBatchModel::KimiLinear(loaded.model),loaded.eos_token_ids))
        },
        other=>Err(format!("no independent causal-draft batch adapter for {other:?}")),
    }
}

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
                                || !ruda_tensor_tch::distributed::peer_available()
                            {
                                return Err(
                                    "LibTorch TP/EP requires the ruCCL CUDA peer adapter".into()
                                );
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
        if options.mtp {
            if !matches!(model_type,Some("deepseek_v2"|"deepseek_v3"|"kimi_k2")) || options.draft_directory.is_some()
                || options.causal_draft_tokens.is_some() || source.get("num_nextn_predict_layers").and_then(serde_json::Value::as_u64).unwrap_or(0)==0 {
                return Err("--mtp requires checkpoint MTP modules and cannot select an independent draft simultaneously".into());
            }
            options.model_id=format!("{}:mtp",options.model_id);
        }
        if options.multimodal
            && (!matches!(model_type, Some("qwen3_5" | "qwen3_5_text" | "qwen3_5_moe" | "qwen3_5_moe_text"))
                || options.devices.len() != 1
                || options.prefill_only
                || options.prefill_address.is_some())
        {
            return Err(
                "image serving currently requires single-device integrated native Qwen3.5"
                    .into(),
            );
        }
        if options.batch.is_some() && (!matches!(model_type,Some("llama"|"qwen2"|"qwen3"|"qwen3_moe"|"qwen3_5"|"qwen3_5_text"|"qwen3_5_moe"|"qwen3_5_moe_text"|"deepseek_v2"|"deepseek_v3"|"kimi_k2"|"kimi_linear"))
            || (options.devices.len()!=1 && !matches!(model_type,Some("llama"|"qwen2"|"qwen3"|"qwen3_moe"|"qwen3_5"|"qwen3_5_text"|"qwen3_5_moe"|"qwen3_5_moe_text"|"deepseek_v2"|"deepseek_v3"|"kimi_k2"|"kimi_linear")))) {
            return Err("continuous batching requires a supported native model".into());
        }
        if options.mla_mode != DeepseekV2AttentionMode::Expanded
            && !matches!(model_type, Some("deepseek_v2"|"deepseek_v3"|"kimi_k2"))
        {
            return Err("absorbed MLA requires a DeepSeek/Kimi MLA adapter".into());
        }
        if options.causal_draft_tokens.is_some()
            && (options.batch.is_none() || options.multimodal
                || !matches!(model_type,Some("llama"|"qwen2"|"qwen3"|"qwen3_moe"|"qwen3_5"|"qwen3_5_text"|"qwen3_5_moe"|"qwen3_5_moe_text"|"deepseek_v2"|"deepseek_v3"|"kimi_k2"|"kimi_linear"))) {
            return Err("independent causal drafting requires native Llama/Qwen2/Qwen3/Qwen3.5/DeepSeek/Kimi continuous batching".into());
        }
        if options.draft_directory.is_some() && options.causal_draft_tokens.is_none()
            && (model_type != Some("qwen3") || (options.devices.len()!=1 && options.batch.is_none()))
        {
            return Err("DFlash/DSpark requires Qwen3; multiple target devices require native continuous batching".into());
        }
        if matches!(model_type, Some("deepseek_v2"|"deepseek_v3"|"kimi_k2")) {
            options.model_id = format!("{}:mla-{:?}", options.model_id, options.mla_mode);
        }
        if let Some(id) = &options.draft_id {
            let kind=if options.causal_draft_tokens.is_some() {"causal-draft"} else {"draft"};
            options.model_id = format!("{}:{kind}-{id}", options.model_id);
        }
        if options.batch.is_some() && options.devices.len()>1 {
            options.model_id=format!("{}:batch-{}-ranks",options.model_id,options.devices.len());
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

    fn run_single<B: Backend + rullm::MlaFp8Backend>(self, device: B::Device, ready: &mut Ready) -> Result<(), String>
    where
        Qwen35TextModel<B>: CausalModel<B, Cache = Qwen35Cache<B>>,
    {
        match self
            .source
            .get("model_type")
            .and_then(serde_json::Value::as_str)
        {
            Some("deepseek_v2"|"deepseek_v3"|"kimi_k2") => {
                if self.options.mtp {
                    let loaded=rullm::load_huggingface_mla_mtp::<B>(&self.options.directory,&device).map_err(|e|e.to_string())?;
                    let limits=loaded.model.target.config.limits();let model=loaded.model.with_attention_mode(self.options.mla_mode);
                    return self.launch::<B,_>(model,device,limits,loaded.eos_token_ids,ready);
                }
                let loaded = load_huggingface_mla::<B>(&self.options.directory, &device)
                    .map_err(|e| e.to_string())?;
                let limits = loaded.model.config.limits();
                let model = loaded.model.with_attention_mode(self.options.mla_mode);
                self.launch::<B, _>(model, device, limits, loaded.eos_token_ids, ready)
            }
            Some("qwen3"|"qwen3_moe") => {
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
            Some("qwen3_5" | "qwen3_5_text" | "qwen3_5_moe" | "qwen3_5_moe_text") => {
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
    fn launch_batch<B:rullm::GpuPrefixBackend,M:rullm::PrefixBatchModel<B>>(
        self,model:M,device:B::Device,eos:Vec<i32>,config:serving::BatchServingConfig,ready:&mut Ready,
    )->Result<(),String> {
        #[cfg(feature="gpu-kv")]
        if let Some(gpu)=self.options.gpu_kv.clone() {
            if self.options.prefill_only {
                let kv=rullm::PagedKvCacheConfig{block_size:config.block_tokens,num_pages:config.kv_pages,max_sequence_length:model.batch_limits().max_sequence_length};
                return rullm::disaggregated::run_gpu_batch_prefill_service::<B,_>(model,device,self.options.model_id,
                    self.options.cache,kv,config.max_batch_tokens,self.options.prefill_address.as_deref().expect("GPU prefill address"),gpu,
                    ||ready.take().expect("worker readiness").send(Ok(())).is_ok()).map_err(|e|e.to_string());
            }
            use ruda_tensor::device::Device;
            let devices=self.options.devices.iter().map(|&index|Ok(B::Device::from_id(ruda_core::device::DeviceId{
                type_id:0,index_id:u16::try_from(index).map_err(|e|e.to_string())?}))).collect::<Result<Vec<_>,String>>()?;
            let remote=rullm::disaggregated::RemotePrefill{address:self.options.prefill_address.expect("GPU decode address")};
            return serving::run_continuous_batch_with_gpu_prefill::<B,_>(model,device,devices,eos,self.tokenizer.expect("batch tokenizer"),
                self.tokenizer_config,config,self.options.model_id,self.options.cache,remote,gpu,self.jobs,
                ||ready.take().expect("worker readiness").send(Ok(())).is_ok());
        }
        if self.options.prefill_only {
            let kv=rullm::PagedKvCacheConfig{block_size:config.block_tokens,num_pages:config.kv_pages,
                max_sequence_length:model.batch_limits().max_sequence_length};
            return rullm::disaggregated::run_batch_prefill_service::<B,_>(model,device,self.options.model_id,
                self.options.cache,kv,config.max_batch_tokens,self.options.prefill_address.as_deref().expect("prefill address"),
                ||ready.take().expect("worker readiness").send(Ok(())).is_ok()).map_err(|e|e.to_string());
        }
        let remote=self.options.prefill_address.map(|address|rullm::disaggregated::RemotePrefill{address});
        serving::run_continuous_batch_with_prefill::<B,_>(model,device,eos,self.tokenizer.expect("batch tokenizer"),
            self.tokenizer_config,config,self.options.model_id,self.options.cache,remote,self.jobs,
            ||ready.take().expect("worker readiness").send(Ok(())).is_ok())
    }

    #[cfg(feature = "nvidia")]
    fn run_cuda(self, ready: &mut Ready) -> Result<(), String> {
        use ruda_tensor_device::cuda::{Cuda, CudaDevice};
        type B = Cuda<half::bf16, i32>;
        let device = CudaDevice::new(self.options.devices[0]);
        if self.options.multimodal {
            type Native = rullm::backends::Nvidia<half::bf16, i32, u8>;
            let model = rullm::load_huggingface_qwen35_multimodal::<Native>(
                &self.options.directory,
                &device,
            )
            .map_err(|e| e.to_string())?;
            let processor = rullm::Qwen35ImageProcessor::from_huggingface(&self.options.directory)
                .map_err(|e| e.to_string())?;
            return serving::run_qwen35_multimodal(
                model,
                processor,
                device,
                self.tokenizer.expect("image tokenizer"),
                self.tokenizer_config,
                self.options.model_id,
                self.options.cache,
                self.jobs,
                || ready.take().expect("worker readiness").send(Ok(())).is_ok(),
            );
        }
        if let Some(config) = self.options.batch {
            type Native = rullm::backends::Nvidia<half::bf16, i32, u8>;
            if let Some(draft_tokens)=self.options.causal_draft_tokens {
                use sha2::{Digest,Sha256};
                let directory=self.options.draft_directory.as_ref().expect("causal draft directory");
                let tokenizer=|directory:&std::path::Path|->Result<serde_json::Value,String> {
                    let tokenizer=tokenizers::Tokenizer::from_file(directory.join("tokenizer.json")).map_err(|e|e.to_string())?;
                    let vocabulary=tokenizer.get_vocab(true).into_iter().collect::<std::collections::BTreeMap<_,_>>();
                    let special=tokenizer.get_added_tokens_decoder().into_iter().filter(|(_,token)|token.special)
                        .map(|(id,_)|id).collect::<std::collections::BTreeSet<_>>();
                    serde_json::to_value((vocabulary,tokenizer.get_decoder(),special)).map_err(|e|e.to_string())
                };
                let target_tokenizer=tokenizer(&self.options.directory)?;
                if target_tokenizer!=tokenizer(directory)? {return Err("target and causal draft token-ID mappings/decoders differ".into());}
                let tokenizer_id=format!("{:x}",Sha256::digest(serde_json::to_vec(&target_tokenizer).map_err(|e|e.to_string())?));
                let (draft,_)=load_causal_batch(directory,&device,DeepseekV2AttentionMode::Expanded)?;
                let draft_config=rullm::CausalDraftBatchConfig {
                    draft_tokens,draft_model_id:self.options.draft_id.clone().expect("causal draft identity"),
                    target_tokenizer_id:tokenizer_id.clone(),draft_tokenizer_id:tokenizer_id,
                };
                if self.options.devices.len()>1 {
                    #[cfg(feature="tensor-parallel")]
                    {
                        let devices=self.options.devices.iter().map(|&id|CudaDevice::new(id)).collect();
                        return match self.source.get("model_type").and_then(serde_json::Value::as_str) {
                            Some(kind @ ("llama"|"qwen2"))=>{
                                let loaded=if kind=="qwen2" {
                                    rullm::load_huggingface_qwen2_tensor_parallel::<Native>(&self.options.directory,devices)
                                } else {
                                    rullm::load_huggingface_llama_tensor_parallel::<Native>(&self.options.directory,devices)
                                }.map_err(|e|e.to_string())?;
                                let model=rullm::CausalDraftBatch::new(loaded.model,draft,draft_config).map_err(|e|e.to_string())?;
                                self.launch_batch::<Native,_>(model,device,loaded.eos_token_ids,config,ready)
                            },
                            Some("qwen3"|"qwen3_moe")=>{
                                let loaded=rullm::load_huggingface_qwen3_tensor_parallel::<Native>(&self.options.directory,devices).map_err(|e|e.to_string())?;
                                let model=rullm::CausalDraftBatch::new(loaded.model,draft,draft_config).map_err(|e|e.to_string())?;
                                self.launch_batch::<Native,_>(model,device,loaded.eos_token_ids,config,ready)
                            },
                            Some("deepseek_v2"|"deepseek_v3"|"kimi_k2")=>{
                                let loaded=rullm::load_huggingface_mla_expert_parallel::<Native>(&self.options.directory,devices).map_err(|e|e.to_string())?;
                                let model=rullm::CausalDraftBatch::new(loaded.model.with_attention_mode(self.options.mla_mode),draft,draft_config).map_err(|e|e.to_string())?;
                                self.launch_batch::<Native,_>(model,device,loaded.eos_token_ids,config,ready)
                            },
                            Some("qwen3_5"|"qwen3_5_text"|"qwen3_5_moe"|"qwen3_5_moe_text")=>{
                                let loaded=rullm::load_huggingface_qwen35_tensor_parallel::<Native>(&self.options.directory,devices).map_err(|e|e.to_string())?;
                                let eos=vec![loaded.model.config.eos_token_id];
                                let model=rullm::CausalDraftBatch::new(loaded.model,draft,draft_config).map_err(|e|e.to_string())?;
                                self.launch_batch::<Native,_>(model,device,eos,config,ready)
                            },
                            Some("kimi_linear")=>{
                                let loaded=rullm::load_huggingface_kimi_linear_expert_parallel::<Native>(&self.options.directory,devices).map_err(|e|e.to_string())?;
                                let model=rullm::CausalDraftBatch::new(loaded.model,draft,draft_config).map_err(|e|e.to_string())?;
                                self.launch_batch::<Native,_>(model,device,loaded.eos_token_ids,config,ready)
                            },
                            _=>Err("no parallel causal target adapter".into()),
                        };
                    }
                    #[cfg(not(feature="tensor-parallel"))]
                    return Err("parallel causal drafting requires the tensor-parallel feature".into());
                }
                let (target,eos)=load_causal_batch(&self.options.directory,&device,self.options.mla_mode)?;
                let model=rullm::CausalDraftBatch::new(target,draft,draft_config).map_err(|e|e.to_string())?;
                return self.launch_batch::<Native,_>(model,device,eos,config,ready);
            }
            if self.options.devices.len()>1 {
                #[cfg(feature = "tensor-parallel")]
                {
                    let devices=self.options.devices.iter().map(|&id|CudaDevice::new(id)).collect();
                    return match self.source.get("model_type").and_then(serde_json::Value::as_str) {
                        Some(kind @ ("llama"|"qwen2"))=>{
                            let loaded=if kind=="qwen2" {
                                rullm::load_huggingface_qwen2_tensor_parallel::<Native>(&self.options.directory,devices)
                            } else {
                                rullm::load_huggingface_llama_tensor_parallel::<Native>(&self.options.directory,devices)
                            }.map_err(|e|e.to_string())?;
                            self.launch_batch::<Native,_>(loaded.model,device,loaded.eos_token_ids,config,ready)
                        },
                        Some("qwen3"|"qwen3_moe")=>{
                            let loaded=rullm::load_huggingface_qwen3_tensor_parallel::<Native>(&self.options.directory,devices).map_err(|e|e.to_string())?;
                            if let Some(directory)=&self.options.draft_directory {
                                let model=loaded.model.with_draft(directory).map_err(|e|e.to_string())?;
                                return self.launch_batch::<Native,_>(model,device,loaded.eos_token_ids,config,ready);
                            }
                            self.launch_batch::<Native,_>(loaded.model,device,loaded.eos_token_ids,config,ready)
                        },
                        Some("deepseek_v2"|"deepseek_v3"|"kimi_k2")=>{
                            if self.options.mtp {
                                let loaded=rullm::load_huggingface_mla_mtp_expert_parallel::<Native>(&self.options.directory,devices).map_err(|e|e.to_string())?;
                                let model=loaded.model.with_attention_mode(self.options.mla_mode);
                                return self.launch_batch::<Native,_>(model,device,loaded.eos_token_ids,config,ready);
                            }
                            let loaded=rullm::load_huggingface_mla_expert_parallel::<Native>(&self.options.directory,devices).map_err(|e|e.to_string())?;
                            let model=loaded.model.with_attention_mode(self.options.mla_mode);
                            self.launch_batch::<Native,_>(model,device,loaded.eos_token_ids,config,ready)
                        },
                        Some("qwen3_5"|"qwen3_5_text"|"qwen3_5_moe"|"qwen3_5_moe_text")=>{
                            let loaded=rullm::load_huggingface_qwen35_tensor_parallel::<Native>(&self.options.directory,devices).map_err(|e|e.to_string())?;
                            let eos=vec![loaded.model.config.eos_token_id];
                            self.launch_batch::<Native,_>(loaded.model,device,eos,config,ready)
                        },
                        Some("kimi_linear")=>{
                            let loaded=rullm::load_huggingface_kimi_linear_expert_parallel::<Native>(&self.options.directory,devices).map_err(|e|e.to_string())?;
                            self.launch_batch::<Native,_>(loaded.model,device,loaded.eos_token_ids,config,ready)
                        },
                        _=>Err("no parallel continuous-batch model adapter".into()),
                    };
                }
                #[cfg(not(feature = "tensor-parallel"))]
                return Err("parallel batching requires the tensor-parallel feature".into());
            }
            return match self.source.get("model_type").and_then(serde_json::Value::as_str) {
                Some("llama")=>{
                    let loaded=load_huggingface_llama::<Native>(&self.options.directory,&device).map_err(|e|e.to_string())?;
                    self.launch_batch::<Native,_>(loaded.model,device,loaded.default_eos_token_ids,config,ready)
                },
                Some("qwen2")=>{
                    let loaded=load_huggingface_qwen2::<Native>(&self.options.directory,&device).map_err(|e|e.to_string())?;
                    self.launch_batch::<Native,_>(loaded.model,device,loaded.default_eos_token_ids,config,ready)
                },
                Some("qwen3"|"qwen3_moe")=>{
                    let loaded=load_huggingface_qwen3::<Native>(&self.options.directory,&device).map_err(|e|e.to_string())?;
                    if let Some(directory)=&self.options.draft_directory {
                        let model=rullm::load_huggingface_qwen3_draft(loaded.model,directory).map_err(|e|e.to_string())?;
                        return self.launch_batch::<Native,_>(model,device,loaded.eos_token_ids,config,ready);
                    }
                    self.launch_batch::<Native,_>(loaded.model,device,loaded.eos_token_ids,config,ready)
                },
                Some("deepseek_v2"|"deepseek_v3"|"kimi_k2")=>{
                    if self.options.mtp {
                        let loaded=rullm::load_huggingface_mla_mtp::<Native>(&self.options.directory,&device).map_err(|e|e.to_string())?;
                        let model=loaded.model.with_attention_mode(self.options.mla_mode);
                        return self.launch_batch::<Native,_>(model,device,loaded.eos_token_ids,config,ready);
                    }
                    let loaded=load_huggingface_mla::<Native>(&self.options.directory,&device).map_err(|e|e.to_string())?;
                    let model=loaded.model.with_attention_mode(self.options.mla_mode);
                    self.launch_batch::<Native,_>(model,device,loaded.eos_token_ids,config,ready)
                },
                Some("qwen3_5"|"qwen3_5_text"|"qwen3_5_moe"|"qwen3_5_moe_text")=>{
                    let loaded=load_huggingface_qwen35_text::<Native>(&self.options.directory,&device).map_err(|e|e.to_string())?;
                    let eos=vec![loaded.model.config.eos_token_id];
                    self.launch_batch::<Native,_>(loaded.model,device,eos,config,ready)
                },
                Some("kimi_linear")=>{
                    let loaded=rullm::load_huggingface_kimi_linear::<Native>(&self.options.directory,&device).map_err(|e|e.to_string())?;
                    self.launch_batch::<Native,_>(loaded.model,device,loaded.eos_token_ids,config,ready)
                },
                _=>Err("no continuous-batch model adapter".into()),
            };
        }
        if self.source.get("model_type").and_then(serde_json::Value::as_str)==Some("kimi_linear") {
            type Native = rullm::backends::Nvidia<half::bf16,i32,u8>;
            if self.options.devices.len()==1 {
                let loaded=rullm::load_huggingface_kimi_linear::<Native>(&self.options.directory,&device).map_err(|e|e.to_string())?;
                let limits=loaded.model.limits();
                return self.launch::<Native,_>(loaded.model,device,limits,loaded.eos_token_ids,ready);
            }
            #[cfg(feature="tensor-parallel")]
            {
                let devices=self.options.devices.iter().map(|&id|CudaDevice::new(id)).collect();
                let loaded=rullm::load_huggingface_kimi_linear_expert_parallel::<Native>(&self.options.directory,devices).map_err(|e|e.to_string())?;
                let limits=loaded.model.limits();
                return self.launch::<Native,_>(loaded.model,device,limits,loaded.eos_token_ids,ready);
            }
            #[cfg(not(feature="tensor-parallel"))]
            return Err("Kimi Linear EP requires the tensor-parallel feature".into());
        }
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
    fn run_parallel<B: ruda_tensor::distributed::DistributedBackend + rullm::MlaFp8Backend>(
        self,
        devices: Vec<B::Device>,
        ready: &mut Ready,
    ) -> Result<(), String>
    where Qwen35TextModel<B>: CausalModel<B, Cache=Qwen35Cache<B>> + rullm::SpeculativeModel<B> + Sync {
        let device = devices[0].clone();
        match self
            .source
            .get("model_type")
            .and_then(serde_json::Value::as_str)
        {
            Some(kind @ ("llama" | "qwen2")) => {
                let loaded = if kind == "qwen2" {
                    rullm::load_huggingface_qwen2_tensor_parallel::<B>(&self.options.directory, devices)
                } else {
                    rullm::load_huggingface_llama_tensor_parallel::<B>(&self.options.directory, devices)
                }.map_err(|e| e.to_string())?;
                let limits = loaded.model.limits();
                self.launch::<B, _>(loaded.model, device, limits, loaded.eos_token_ids, ready)
            }
            Some("deepseek_v2"|"deepseek_v3"|"kimi_k2") => {
                if self.options.mtp {
                    let loaded=rullm::load_huggingface_mla_mtp_expert_parallel::<B>(&self.options.directory,devices).map_err(|e|e.to_string())?;
                    let limits=loaded.model.config.limits();let model=loaded.model.with_attention_mode(self.options.mla_mode);
                    return self.launch::<B,_>(model,device,limits,loaded.eos_token_ids,ready);
                }
                let loaded = rullm::load_huggingface_mla_expert_parallel::<B>(
                    &self.options.directory,
                    devices,
                )
                .map_err(|e| e.to_string())?;
                let limits = loaded.model.config.limits();
                let model = loaded.model.with_attention_mode(self.options.mla_mode);
                self.launch::<B, _>(model, device, limits, loaded.eos_token_ids, ready)
            }
            Some("qwen3"|"qwen3_moe") => {
                let loaded = rullm::load_huggingface_qwen3_tensor_parallel::<B>(
                    &self.options.directory,
                    devices,
                )
                .map_err(|e| e.to_string())?;
                let limits = loaded.model.config.limits();
                self.launch::<B, _>(loaded.model, device, limits, loaded.eos_token_ids, ready)
            }
            Some("qwen3_5" | "qwen3_5_text" | "qwen3_5_moe" | "qwen3_5_moe_text") => {
                let loaded = rullm::load_huggingface_qwen35_tensor_parallel::<B>(&self.options.directory, devices)
                    .map_err(|e| e.to_string())?;
                let limits = loaded.model.limits();
                let eos = vec![loaded.model.config.eos_token_id];
                self.launch::<B, _>(loaded.model, device, limits, eos, ready)
            }
            other => Err(format!(
                "tensor/expert parallel serving is not implemented for {other:?}"
            )),
        }
    }
}
