use super::{
    progress::{self, ResponseProgress},
    *,
};
use crate::{
    ContinuousBatchConfig, ContinuousBatchOptions, DeviceBatchExecutor, DeviceBatchModel,
    PagedKvCacheConfig, RequestId, PrefixBatchExecutor, PrefixBatchModel,
    ContinuousBatchError, ContinuousBatchSnapshot, FinishedGeneration, GenerationError,
};

#[derive(Clone, Copy)]
pub struct BatchServingConfig {
    pub max_sequences: usize,
    pub max_batch_tokens: usize,
    pub block_tokens: usize,
    pub kv_pages: usize,
    pub max_queued_requests: usize,
    pub active_kv_host_bytes: usize,
}

struct Active<'a> {
    job: Job,
    progress: ResponseProgress<'a>,
    prompt_tokens: usize,
    constraint: Option<structured::Decoder>,
}

trait BatchEngine {
    fn submit(&mut self, ids: Vec<i32>, config: GreedyGenerationConfig) -> Result<RequestId, ContinuousBatchError>;
    fn submit_sampled(&mut self, ids: Vec<i32>, config: SamplingGenerationConfig) -> Result<RequestId, ContinuousBatchError>;
    fn cancel(&mut self, id: RequestId) -> Result<(), ContinuousBatchError>;
    fn snapshot(&self) -> ContinuousBatchSnapshot;
    fn pop_finished(&mut self) -> Option<FinishedGeneration>;
    fn is_idle(&self) -> bool;
    fn pop_failed(&mut self)->Option<(RequestId,String)> {None}
    fn waiting_for_prefill(&self)->bool {false}
    fn step(&mut self, processor: &mut dyn FnMut(RequestId, &mut [f32]) -> Result<(), GenerationError>)
        -> Result<Option<Vec<(RequestId, i32)>>, ContinuousBatchError>;
}

macro_rules! batch_engine {
    ($executor:ident, $model:ident, $tokens:expr) => {
        impl<B: Backend, M: $model<B>> BatchEngine for $executor<'_, B, M> {
            fn submit(&mut self, ids: Vec<i32>, config: GreedyGenerationConfig) -> Result<RequestId, ContinuousBatchError> {
                self.submit(ids, config)
            }
            fn submit_sampled(&mut self, ids: Vec<i32>, config: SamplingGenerationConfig) -> Result<RequestId, ContinuousBatchError> {
                self.submit_sampled(ids, config)
            }
            fn cancel(&mut self, id: RequestId) -> Result<(), ContinuousBatchError> { self.cancel(id).map(|_| ()) }
            fn snapshot(&self) -> ContinuousBatchSnapshot { self.snapshot() }
            fn pop_finished(&mut self) -> Option<FinishedGeneration> { self.pop_finished() }
            fn is_idle(&self) -> bool { self.is_idle() }
            fn step(&mut self, processor: &mut dyn FnMut(RequestId, &mut [f32]) -> Result<(), GenerationError>)
                -> Result<Option<Vec<(RequestId, i32)>>, ContinuousBatchError> {
                self.step_with_logits_processor(processor).map(|step| step.map($tokens))
            }
        }
    };
}
batch_engine!(DeviceBatchExecutor, DeviceBatchModel, |step: crate::ExecutedBatch|
    step.batch.sequences.iter().zip(step.generated_token_ids).map(|(row, token)| (row.request_id, token)).collect());
impl<B:Backend,M:PrefixBatchModel<B>> BatchEngine for PrefixBatchExecutor<'_,B,M> {
    fn submit(&mut self,ids:Vec<i32>,config:GreedyGenerationConfig)->Result<RequestId,ContinuousBatchError> {self.submit(ids,config)}
    fn submit_sampled(&mut self,ids:Vec<i32>,config:SamplingGenerationConfig)->Result<RequestId,ContinuousBatchError> {self.submit_sampled(ids,config)}
    fn cancel(&mut self,id:RequestId)->Result<(),ContinuousBatchError> {self.cancel(id).map(|_|())}
    fn snapshot(&self)->ContinuousBatchSnapshot {self.snapshot()}
    fn pop_finished(&mut self)->Option<FinishedGeneration> {self.pop_finished()}
    fn is_idle(&self)->bool {self.is_idle()}
    fn pop_failed(&mut self)->Option<(RequestId,String)> {self.pop_failed()}
    fn waiting_for_prefill(&self)->bool {self.waiting_for_prefill()}
    fn step(&mut self,processor:&mut dyn FnMut(RequestId,&mut [f32])->Result<(),GenerationError>)
        ->Result<Option<Vec<(RequestId,i32)>>,ContinuousBatchError> {
        self.step_with_logits_processor(processor).map(|step|step.map(|step|step.tokens))
    }
}

fn admit<'a, E: BatchEngine>(
    job: Job,
    executor: &mut E,
    active: &mut BTreeMap<RequestId, Active<'a>>,
    tokenizer: &'a tokenizers::Tokenizer,
    config: &Value,
    eos: &[i32],
) {
    if job.events.is_closed() {
        return;
    }
    let submission = (|| {
        if images::has_images(&job.request) {
            return Err("continuous text batching does not accept image requests".into());
        }
        let ids = progress::prompt_ids(&job, tokenizer, config)?;
        let eos = progress::eos_ids(job.chat, eos, tokenizer, config)?;
        let constraint = structured::Decoder::new(&job.request, tokenizer, &eos)?;
        let prompt_tokens = ids.len();
        let (greedy, sampled) = progress::configs(&job.request, &eos);
        let id = if job.request.temperature == Some(0.0) {
            executor.submit(ids, greedy)
        } else {
            executor.submit_sampled(ids, sampled)
        }
        .map_err(|e| e.to_string())?;
        Ok::<_, String>((id, prompt_tokens, constraint))
    })();
    let (id, prompt_tokens, constraint) = match submission {
        Ok(value) => value,
        Err(error) => {
            let _ = job.ready.send(Err((StatusCode::BAD_REQUEST, error)));
            return;
        }
    };
    let Some(job) = progress::accept(job) else {
        let _ = executor.cancel(id);
        return;
    };
    match ResponseProgress::new(&job, tokenizer) {
        Ok(progress) => {
            active.insert(
                id,
                Active {
                    job,
                    progress,
                    prompt_tokens,
                    constraint,
                },
            );
        }
        Err(error) => {
            let _ = executor.cancel(id);
            let _ = job.events.blocking_send(Err(error));
        }
    }
}

/// Shared-weight, real model batches; one executor forward processes all scheduled rows.
pub fn run_continuous_batch<B: Backend, M: DeviceBatchModel<B>>(
    model: M,
    device: B::Device,
    eos: Vec<i32>,
    tokenizer: tokenizers::Tokenizer,
    tokenizer_config: Value,
    config: BatchServingConfig,
    jobs: mpsc::Receiver<Job>,
    on_ready: impl FnOnce() -> bool,
) -> Result<(), String> {
    if config.active_kv_host_bytes!=0 {return Err("active KV swap requires the prefix-capable batch executor".into());}
    let mut executor = DeviceBatchExecutor::<B, M>::new(
        &model,
        device,
        ContinuousBatchConfig {
            max_active_sequences: config.max_sequences,
            max_batch_tokens: config.max_batch_tokens,
        },
        PagedKvCacheConfig {
            block_size: config.block_tokens,
            num_pages: config.kv_pages,
            max_sequence_length: model.batch_limits().max_sequence_length,
        },
        ContinuousBatchOptions {
            max_pending_requests: Some(config.max_queued_requests),
            ..Default::default()
        },
    )
    .map_err(|e| e.to_string())?;
    run_batch_loop(&mut executor, eos, tokenizer, tokenizer_config, config, jobs, on_ready)
}

/// Batched prefill/decode with exact model-scoped prefix reuse and bounded host/SSD tiers.
pub fn run_continuous_batch_with_prefix<B: Backend, M: PrefixBatchModel<B>>(
    model: M,
    device: B::Device,
    eos: Vec<i32>,
    tokenizer: tokenizers::Tokenizer,
    tokenizer_config: Value,
    config: BatchServingConfig,
    model_id: String,
    prefixes: PrefixCacheConfig,
    jobs: mpsc::Receiver<Job>,
    on_ready: impl FnOnce() -> bool,
) -> Result<(), String> {
    run_continuous_batch_with_prefill(model,device,eos,tokenizer,tokenizer_config,config,model_id,prefixes,None,jobs,on_ready)
}

pub fn run_continuous_batch_with_prefill<B:Backend,M:PrefixBatchModel<B>>(
    model:M,device:B::Device,eos:Vec<i32>,tokenizer:tokenizers::Tokenizer,tokenizer_config:Value,
    config:BatchServingConfig,model_id:String,prefixes:PrefixCacheConfig,
    remote:Option<crate::disaggregated::RemotePrefill>,jobs:mpsc::Receiver<Job>,on_ready:impl FnOnce()->bool,
)->Result<(),String> {
    let mut executor = PrefixBatchExecutor::<B, M>::new(
        &model, device,
        ContinuousBatchConfig { max_active_sequences: config.max_sequences, max_batch_tokens: config.max_batch_tokens },
        PagedKvCacheConfig { block_size: config.block_tokens, num_pages: config.kv_pages,
            max_sequence_length: model.batch_limits().max_sequence_length },
        ContinuousBatchOptions { max_pending_requests: Some(config.max_queued_requests), ..Default::default() },
        model_id, prefixes,
    ).map_err(|e| e.to_string())?;
    executor.set_active_kv_swap(crate::ActiveKvSwapConfig{host_bytes:config.active_kv_host_bytes}).map_err(|e|e.to_string())?;
    if let Some(remote)=remote {executor.set_remote_prefill(remote).map_err(|e|e.to_string())?;}
    run_batch_loop(&mut executor, eos, tokenizer, tokenizer_config, config, jobs, on_ready)
}

#[cfg(feature="gpu-kv")]
pub fn run_continuous_batch_with_gpu_prefill<B:crate::GpuPrefixBackend,M:PrefixBatchModel<B>>(
    model:M,device:B::Device,devices:Vec<B::Device>,eos:Vec<i32>,tokenizer:tokenizers::Tokenizer,tokenizer_config:Value,
    config:BatchServingConfig,model_id:String,prefixes:PrefixCacheConfig,
    remote:crate::disaggregated::RemotePrefill,gpu:crate::disaggregated::GpuKvConfig,jobs:mpsc::Receiver<Job>,on_ready:impl FnOnce()->bool,
)->Result<(),String> {
    let mut executor=PrefixBatchExecutor::<B,M>::new(&model,device,
        ContinuousBatchConfig{max_active_sequences:config.max_sequences,max_batch_tokens:config.max_batch_tokens},
        PagedKvCacheConfig{block_size:config.block_tokens,num_pages:config.kv_pages,max_sequence_length:model.batch_limits().max_sequence_length},
        ContinuousBatchOptions{max_pending_requests:Some(config.max_queued_requests),..Default::default()},model_id,prefixes).map_err(|e|e.to_string())?;
    executor.set_active_kv_swap(crate::ActiveKvSwapConfig{host_bytes:config.active_kv_host_bytes}).map_err(|e|e.to_string())?;
    executor.set_remote_gpu_prefill(remote,devices,gpu).map_err(|e|e.to_string())?;
    run_batch_loop(&mut executor,eos,tokenizer,tokenizer_config,config,jobs,on_ready)
}

fn run_batch_loop<E: BatchEngine>(
    executor: &mut E,
    eos: Vec<i32>,
    tokenizer: tokenizers::Tokenizer,
    tokenizer_config: Value,
    config: BatchServingConfig,
    mut jobs: mpsc::Receiver<Job>,
    on_ready: impl FnOnce() -> bool,
) -> Result<(), String> {
    if config.max_queued_requests == 0 { return Err("batch queue capacity must be positive".into()); }
    let admission_limit = config.max_sequences.checked_add(config.max_queued_requests)
        .ok_or("batch admission count overflows")?;
    if !on_ready() {
        return Ok(());
    }
    let mut active = BTreeMap::<RequestId, Active<'_>>::new();
    let mut closed = false;
    loop {
        if active.is_empty() {
            if closed {
                break;
            }
            match jobs.blocking_recv() {
                Some(job) => admit(
                    job,
                    executor,
                    &mut active,
                    &tokenizer,
                    &tokenizer_config,
                    &eos,
                ),
                None => {
                    closed = true;
                    continue;
                }
            }
        }
        // Admission occurs between completed batches, including while older rows decode.
        while active.len() < admission_limit
            && executor.snapshot().pending_requests < config.max_queued_requests
        {
            match jobs.try_recv() {
                Ok(job) => admit(
                    job,
                    executor,
                    &mut active,
                    &tokenizer,
                    &tokenizer_config,
                    &eos,
                ),
                Err(mpsc::error::TryRecvError::Empty) => break,
                Err(mpsc::error::TryRecvError::Disconnected) => {
                    closed = true;
                    break;
                }
            }
        }
        let disconnected = active
            .iter()
            .filter(|(_, request)| request.job.events.is_closed())
            .map(|(&id, _)| id)
            .collect::<Vec<_>>();
        for id in disconnected {
            executor.cancel(id).map_err(|e| e.to_string())?;
            active.remove(&id);
        }
        let step = match executor.step(&mut |id, values| {
            if let Some(request) = active.get_mut(&id) {
                if let Some(constraint) = &mut request.constraint {
                    constraint.mask_values(values)?;
                }
            }
            Ok(())
        }) {
            Ok(step) => step,
            Err(error) => {
                let message = error.to_string();
                for request in active.values() {
                    let _ = request.job.events.blocking_send(Err(message.clone()));
                }
                return Err(message);
            }
        };
        if let Some(step) = &step {
            for &(id, token) in step {
                let Some(request) = active.get_mut(&id) else {
                    continue;
                };
                if let Some(constraint) = &mut request.constraint {
                    if let Err(error) = constraint.consume(token) {
                        executor.cancel(id).map_err(|e| e.to_string())?;
                        let request = active
                            .remove(&id)
                            .expect("active constrained row");
                        let _ = request.job.events.blocking_send(Err(error.to_string()));
                        continue;
                    }
                }
                match request.progress.push(&request.job, token) {
                    Ok(false) => (),
                    result => {
                        executor.cancel(id).map_err(|e| e.to_string())?;
                        let request = active.remove(&id).expect("active batch row");
                        let result = match result {
                            Ok(true) => request.progress.finish(
                                &request.job,
                                &tokenizer,
                                request.prompt_tokens,
                                None,
                            ),
                            Err(error) => Err(error),
                            Ok(false) => unreachable!(),
                        };
                        if let Err(error) = result {
                            let _ = request.job.events.blocking_send(Err(error));
                        }
                    }
                }
            }
        }
        while let Some(finished) = executor.pop_finished() {
            if let Some(request) = active.remove(&finished.request_id) {
                let reason = if finished.stopped_on_eos {
                    finished
                        .generated_token_ids
                        .last()
                        .copied()
                        .map(GenerationFinishReason::EosToken)
                } else {
                    Some(GenerationFinishReason::MaxNewTokens)
                };
                if let Err(error) =
                    request
                        .progress
                        .finish(&request.job, &tokenizer, request.prompt_tokens, reason)
                {
                    let _ = request.job.events.blocking_send(Err(error));
                }
            }
        }
        while let Some((id,error))=executor.pop_failed() {
            if let Some(request)=active.remove(&id) {let _=request.job.events.blocking_send(Err(error));}
        }
        if step.is_none() && !executor.is_idle() {
            if executor.waiting_for_prefill() {
                std::thread::park_timeout(std::time::Duration::from_millis(10));
                continue;
            }
            let error =
                "continuous batch cannot advance with the configured KV page pool".to_string();
            for request in active.values() {
                let _ = request.job.events.blocking_send(Err(error.clone()));
            }
            return Err(error);
        }
    }
    Ok(())
}
