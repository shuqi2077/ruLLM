use super::*;
use crate::{CausalModelLimits, GenerationError, SamplingGenerationConfig};
use ruda_tensor::api::{FloatDType, Tensor, backend::Backend};

/// A real batched model adapter. Cache forks must isolate subsequent writes.
/// The model consumes scheduled rows together, not by looping over model forwards.
pub trait DeviceBatchModel<B: Backend> {
    type Cache;
    fn batch_limits(&self) -> CausalModelLimits;
    fn new_batch_cache(&self, config: PagedKvCacheConfig) -> Self::Cache;
    fn fork_batch_cache(&self, cache: &Self::Cache) -> Self::Cache;
    fn forward_batch(
        &self,
        batch: &ScheduledBatch,
        cache: &mut Self::Cache,
        device: &B::Device,
    ) -> Result<Tensor<B, 3>, GenerationError>;
    /// Called only after device completion, when finished/cancelled pages can be released.
    fn retain_batch_cache(&self, cache: &mut Self::Cache, scheduler: &ContinuousBatchScheduler);
    fn sync_batch(&self, device: &B::Device) -> Result<(), GenerationError> {
        B::sync(device).map_err(|e|GenerationError(e.to_string()))
    }
    fn lookahead_tokens(&self)->usize {0}
    fn forward_batch_lookahead(&self,batch:&ScheduledBatch,_lookahead:&BTreeMap<RequestId,PagedKvAppendReservation>,
        cache:&mut Self::Cache,device:&B::Device)->Result<Tensor<B,3>,GenerationError> {
        self.forward_batch(batch,cache,device)
    }
    fn forward_batch_sampling(&self,batch:&ScheduledBatch,lookahead:&BTreeMap<RequestId,PagedKvAppendReservation>,
        cache:&mut Self::Cache,device:&B::Device,_sampling:&mut BatchSamplingState)->Result<Tensor<B,3>,GenerationError> {
        self.forward_batch_lookahead(batch,lookahead,cache,device)
    }
    fn select_batch_token(&self,_cache:&mut Self::Cache,_id:RequestId,values:&[f32],sampler:&mut Option<TokenSampler>)
        ->Result<i32,GenerationError> {
        match sampler {
            Some(sampler)=>sampler.sample(values),
            None=>super::sampling::greedy_token(values,0).map_err(|e|GenerationError(e.to_string())),
        }
    }
}

#[derive(Debug, Clone)]
pub struct ExecutedBatch {
    pub batch: ScheduledBatch,
    pub generated_token_ids: Vec<i32>,
}

/// Synchronous batch boundaries with asynchronous GPU work inside each forward.
/// The model is borrowed once; weights are shared across all resident requests.
pub struct DeviceBatchExecutor<'a, B: Backend, M: DeviceBatchModel<B>> {
    pub(super) model: &'a M,
    pub(super) device: B::Device,
    pub(super) scheduler: ContinuousBatchScheduler,
    pub(super) cache: M::Cache,
}

impl<'a, B: Backend, M: DeviceBatchModel<B>> DeviceBatchExecutor<'a, B, M> {
    pub fn new(
        model: &'a M,
        device: B::Device,
        config: ContinuousBatchConfig,
        kv: PagedKvCacheConfig,
        options: ContinuousBatchOptions,
    ) -> Result<Self, ContinuousBatchError> {
        if kv.max_sequence_length > model.batch_limits().max_sequence_length {
            return Err(ContinuousBatchError(
                "KV sequence limit exceeds model capacity".into(),
            ));
        }
        let scheduler = ContinuousBatchScheduler::with_options(config, kv, options)?;
        Ok(Self {
            model,
            device,
            scheduler,
            cache: model.new_batch_cache(kv),
        })
    }

    pub fn submit(
        &mut self,
        prompt: Vec<i32>,
        generation: GreedyGenerationConfig,
    ) -> Result<RequestId, ContinuousBatchError> {
        self.validate(&prompt, &generation)?;
        self.scheduler.submit(prompt, generation)
    }

    pub fn submit_sampled(
        &mut self,
        prompt: Vec<i32>,
        generation: SamplingGenerationConfig,
    ) -> Result<RequestId, ContinuousBatchError> {
        self.validate(
            &prompt,
            &GreedyGenerationConfig {
                max_new_tokens: generation.max_new_tokens,
                eos_token_ids: generation.eos_token_ids.clone(),
            },
        )?;
        self.scheduler.submit_sampled(prompt, generation)
    }

    fn validate(
        &self,
        prompt: &[i32],
        generation: &GreedyGenerationConfig,
    ) -> Result<(), ContinuousBatchError> {
        crate::generation::validate_generation(&self.model.batch_limits(), prompt, generation)
            .map(|_| ())
            .map_err(|e| ContinuousBatchError(e.to_string()))
    }

    pub fn step(&mut self) -> Result<Option<ExecutedBatch>, ContinuousBatchError> {
        self.step_inner(None,None)
    }

    pub fn step_with_logits_processor(
        &mut self,
        mut processor: impl FnMut(RequestId, &mut [f32]) -> Result<(), GenerationError>,
    ) -> Result<Option<ExecutedBatch>, ContinuousBatchError> {
        self.step_inner(Some(&mut processor),None)
    }

    pub(super) fn step_inner(
        &mut self,
        processor: Option<&mut dyn FnMut(RequestId, &mut [f32]) -> Result<(), GenerationError>>,
        observer: Option<&mut dyn FnMut(&ScheduledBatch,&M::Cache,&Tensor<B,3>)>,
    ) -> Result<Option<ExecutedBatch>, ContinuousBatchError> {
        let Some(batch) = self.scheduler.schedule()? else {
            return Ok(None);
        };
        let mut working = self.model.fork_batch_cache(&self.cache);
        let lookahead=match self.scheduler.reserve_lookahead(&batch,self.model.lookahead_tokens()) {
            Ok(reserved)=>reserved,
            Err(error)=>{self.scheduler.fail_batch(batch.id)?;return Err(error);},
        };
        let vocabulary = self.model.batch_limits().vocab_size;
        let mut sampling=match self.scheduler.batch_sampling_state(batch.id) {
            Ok(sampling)=>sampling,
            Err(error)=>{self.scheduler.fail_batch(batch.id)?;return Err(error);},
        };
        let mut saved_logits=None;
        let result = self
            .model
            .forward_batch_sampling(&batch, &lookahead, &mut working, &self.device, &mut sampling)
            .and_then(|logits| {
                if logits.dims() != [batch.batch_size(), 1, vocabulary] {
                    return Err(GenerationError("device batch logit shape mismatch".into()));
                }
                if observer.is_some() {saved_logits=Some(logits.clone());}
                logits
                    .cast(FloatDType::F32)
                    .try_into_data()
                    .map_err(|e| GenerationError(e.to_string()))?
                    .to_vec::<f32>()
                    .map_err(|e| GenerationError(e.to_string()))
            });
        // An error/panic is not evidence of device completion. A failed sync
        // leaves this batch in flight; its pages cannot be cancelled or reused.
        self.model.sync_batch(&self.device)
            .map_err(|e| ContinuousBatchError(format!("batch completion: {e}")))?;
        if result.is_ok() {
            if let (Some(observer),Some(logits))=(observer,saved_logits.as_ref()) {
                observer(&batch,&working,logits);
                self.model.sync_batch(&self.device).map_err(|e|ContinuousBatchError(format!("batch prefix completion: {e}")))?;
            }
        }
        let selection = result
            .map_err(|e| ContinuousBatchError(e.to_string()))
            .and_then(|mut values| {
                if let Some(processor) = processor {
                    for (sequence, row) in batch
                        .sequences
                        .iter()
                        .zip(values.chunks_exact_mut(vocabulary))
                    {
                        if sequence.context_length<self.scheduler.active[&sequence.request_id].prompt_token_ids.len() {continue;}
                        processor(sequence.request_id, row)
                            .map_err(|e| ContinuousBatchError(e.to_string()))?;
                    }
                }
                self.scheduler
                    .select_batch_tokens_with(batch.id, &values, vocabulary, sampling, |id,row,sampler|
                        self.model.select_batch_token(&mut working,id,row,sampler)
                            .map_err(|e|ContinuousBatchError(e.to_string())))
            });
        let selection = match selection {
            Ok(selection) => selection,
            Err(error) => {
                self.scheduler.fail_batch(batch.id)?;
                return Err(error);
            }
        };
        let generated_token_ids = self.scheduler.complete_selected_batch(selection)?;
        self.model.retain_batch_cache(&mut working, &self.scheduler);
        self.cache = working;
        Ok(Some(ExecutedBatch {
            batch,
            generated_token_ids,
        }))
    }

    pub fn cancel(
        &mut self,
        id: RequestId,
    ) -> Result<Option<CancelledGeneration>, ContinuousBatchError> {
        let cancelled = self.scheduler.cancel(id)?;
        self.model
            .retain_batch_cache(&mut self.cache, &self.scheduler);
        Ok(cancelled)
    }
    pub fn pop_finished(&mut self) -> Option<FinishedGeneration> {
        self.scheduler.pop_finished()
    }
    pub fn is_idle(&self) -> bool {
        self.scheduler.is_idle()
    }
    pub fn snapshot(&self) -> ContinuousBatchSnapshot {
        self.scheduler.snapshot()
    }
    pub fn cache(&self) -> &M::Cache {
        &self.cache
    }
}
