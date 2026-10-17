use super::*;
use crate::{CausalModelLimits,GenerationError,SamplingGenerationConfig};
use crate::batch_prefix::{Prefix,PrefixStore};
use crate::prefix_cache::{PrefixCacheConfig,PrefixCacheStats};
use ruda_tensor::api::{backend::Backend,FloatDType,Tensor};
#[path="active_swap.rs"]
mod active_swap;
pub use active_swap::{ActiveKvSwapConfig,ActiveKvSwapStats};
use active_swap::{ActiveKvSwap,SwapRestore};

/// Complete request-local state, including recurrent/convolution state where
/// applicable. Snapshot/restore must never reuse another live request's pages.
pub trait PrefixBatchModel<B:Backend>:DeviceBatchModel<B> {
    type Prefix:Clone;
    fn capture_prefix(&self,cache:&Self::Cache,id:RequestId)->Result<Self::Prefix,GenerationError>;
    fn prefix_position(&self,prefix:&Self::Prefix)->usize;
    fn prefix_bytes(&self,prefix:&Self::Prefix)->usize;
    fn restore_prefix(&self,cache:&mut Self::Cache,id:RequestId,table:&[KvPageId],prefix:&Self::Prefix)->Result<(),GenerationError>;
    fn encode_prefix(&self,prefix:&Self::Prefix,model_id:&str)->Result<Vec<u8>,GenerationError>;
    fn decode_prefix(&self,bytes:&[u8],model_id:&str,device:&B::Device)->Result<Self::Prefix,GenerationError>;
    fn prefix_tensor_transfer(&self)->bool {false}
    fn export_prefix_tensors(&self,_prefix:&Self::Prefix,_shard:crate::PrefixShard)->Result<crate::PrefixTensors<B>,GenerationError> {
        Err(GenerationError("model does not expose semantic prefix tensors".into()))
    }
    fn import_prefix_tensors(&self,_tensors:&crate::PrefixTensors<B>,_shard:crate::PrefixShard,_device:&B::Device)->Result<Self::Prefix,GenerationError> {
        Err(GenerationError("model does not import semantic prefix tensors".into()))
    }
}

pub struct PrefixBatchStep {
    /// None means every emitted token came from already cached native logits.
    pub batch:Option<ScheduledBatch>,
    pub tokens:Vec<(RequestId,i32)>,
}
pub struct PrefixBatchExecutor<'a,B:Backend,M:PrefixBatchModel<B>> {
    inner:DeviceBatchExecutor<'a,B,M>,
    prefixes:PrefixStore<B,M>,
    pending:BTreeMap<RequestId,Prefix<B,M>>,
    ready:BTreeMap<RequestId,Tensor<B,3>>,
    failed:VecDeque<(RequestId,String)>,
    handoff:BTreeSet<RequestId>,
    swap:ActiveKvSwap,
    swap_restore:Option<SwapRestore<B,M>>,
    #[cfg(feature="serving")]
    remote:Option<crate::disaggregated::RemoteBatchNetwork>,
    #[cfg(feature="gpu-kv")]
    remote_gpu:Option<crate::disaggregated::GpuRemoteNetwork<B>>,
}
impl<'a,B:Backend,M:PrefixBatchModel<B>> PrefixBatchExecutor<'a,B,M> {
    pub fn new(model:&'a M,device:B::Device,config:ContinuousBatchConfig,kv:PagedKvCacheConfig,
        options:ContinuousBatchOptions,model_id:String,prefixes:PrefixCacheConfig)->Result<Self,ContinuousBatchError> {
        let chunk=prefixes.block_tokens;
        let store=PrefixStore::new(model_id,model.batch_limits(),prefixes).map_err(|e|ContinuousBatchError(e.to_string()))?;
        let mut inner=DeviceBatchExecutor::new(model,device,config,kv,options)?;
        inner.scheduler.prefill_chunk_tokens=store.enabled().then_some(chunk);
        Ok(Self{inner,prefixes:store,pending:BTreeMap::new(),ready:BTreeMap::new(),failed:VecDeque::new(),handoff:BTreeSet::new(),
            swap:ActiveKvSwap::default(),swap_restore:None,
            #[cfg(feature="serving")] remote:None,
            #[cfg(feature="gpu-kv")] remote_gpu:None})
    }
    #[cfg(feature="serving")]
    pub fn set_remote_prefill(&mut self,remote:crate::disaggregated::RemotePrefill)->Result<(),ContinuousBatchError> {
        if !self.is_idle() {return Err(ContinuousBatchError("configure remote prefill before submitting requests".into()));}
        #[cfg(feature="gpu-kv")]
        if self.remote_gpu.is_some() {return Err(ContinuousBatchError("GPU prefill is already configured".into()));}
        let capacity=self.inner.scheduler.options.max_pending_requests
            .ok_or_else(||ContinuousBatchError("remote prefill requires a bounded pending request queue".into()))?;
        self.remote=Some(crate::disaggregated::RemoteBatchNetwork::new(remote,self.prefixes.model_id().into(),
            self.inner.model.batch_limits(),capacity).map_err(|e|ContinuousBatchError(e.to_string()))?);
        if self.inner.scheduler.prefill_chunk_tokens.is_none() {
            self.inner.scheduler.prefill_chunk_tokens=Some(self.inner.scheduler.config.max_batch_tokens);
        }
        Ok(())
    }
    #[cfg(feature="gpu-kv")]
    pub fn set_remote_gpu_prefill(&mut self,remote:crate::disaggregated::RemotePrefill,devices:Vec<B::Device>,gpu:crate::disaggregated::GpuKvConfig)
        ->Result<(),ContinuousBatchError> where B:crate::GpuPrefixBackend {
        if !self.is_idle() || self.remote.is_some() || !self.inner.model.prefix_tensor_transfer() {
            return Err(ContinuousBatchError("configure GPU prefill on an idle semantic-prefix model without host prefill".into()));
        }
        let capacity=self.inner.scheduler.options.max_pending_requests
            .ok_or_else(||ContinuousBatchError("GPU prefill requires a bounded pending request queue".into()))?;
        self.remote_gpu=Some(crate::disaggregated::GpuRemoteNetwork::new(remote,self.prefixes.model_id().into(),
            self.inner.model.batch_limits(),devices,gpu,capacity).map_err(|e|ContinuousBatchError(e.to_string()))?);
        if self.inner.scheduler.prefill_chunk_tokens.is_none() {self.inner.scheduler.prefill_chunk_tokens=Some(self.inner.scheduler.config.max_batch_tokens);}
        Ok(())
    }
    pub fn submit(&mut self,prompt:Vec<i32>,generation:GreedyGenerationConfig)->Result<RequestId,ContinuousBatchError> {
        self.check_swap_queue(generation.max_new_tokens)?;
        let prefix=if generation.max_new_tokens>0 {self.lookup(&prompt)?} else {None};
        let id=self.inner.submit(prompt,generation)?;
        self.stage(id,prefix)?;
        Ok(id)
    }
    pub fn submit_sampled(&mut self,prompt:Vec<i32>,generation:SamplingGenerationConfig)->Result<RequestId,ContinuousBatchError> {
        self.check_swap_queue(generation.max_new_tokens)?;
        let prefix=if generation.max_new_tokens>0 {self.lookup(&prompt)?} else {None};
        let id=self.inner.submit_sampled(prompt,generation)?;
        self.stage(id,prefix)?;
        Ok(id)
    }
    fn stage(&mut self,id:RequestId,prefix:Option<Prefix<B,M>>)->Result<(),ContinuousBatchError> {
        #[cfg(feature="gpu-kv")]
        if let Some(remote)=&mut self.remote_gpu {
            if let Some(request)=self.inner.scheduler.pending.iter().find(|r|r.id==id) {
                if !prefix.as_ref().is_some_and(|p|p.tokens==request.prompt_token_ids) {
                    if let Err(e)=remote.submit(id,request.prompt_token_ids.clone()) {
                        self.inner.cancel(id)?;return Err(ContinuousBatchError(e.to_string()));
                    }
                    self.inner.scheduler.deferred_prefill.insert(id);return Ok(());
                }
                self.handoff.insert(id);
            }
        }
        #[cfg(feature="serving")]
        if let Some(remote)=&mut self.remote {
            if let Some(request)=self.inner.scheduler.pending.iter().find(|r|r.id==id) {
                if !prefix.as_ref().is_some_and(|p|p.tokens==request.prompt_token_ids) {
                    if let Err(e)=remote.submit(id,request.prompt_token_ids.clone()) {
                        self.inner.cancel(id)?;return Err(ContinuousBatchError(e.to_string()));
                    }
                    self.inner.scheduler.deferred_prefill.insert(id);
                    return Ok(());
                }
                self.handoff.insert(id);
            }
        }
        if let Some(prefix)=prefix {self.pending.insert(id,prefix);}
        Ok(())
    }
    #[cfg(feature="serving")]
    fn poll_remote(&mut self)->Result<(),ContinuousBatchError> {
        loop {
            let Some((id,result))=self.remote.as_mut().and_then(|r|r.poll()) else {break;};
            let Some(request)=self.inner.scheduler.pending.iter().find(|r|r.id==id) else {continue;};
            let result=result.and_then(|bytes|self.prefixes.import_handoff(self.inner.model,
                &request.prompt_token_ids,&bytes,&self.inner.device).map_err(|e|e.to_string()));
            match result {
                Ok(prefix)=>{
                    if let Err(e)=self.prefixes.insert(self.inner.model,prefix.clone(),&self.inner.device) {
                        eprintln!("remote prefix tier write: {e}");
                    }
                    self.pending.insert(id,prefix);self.handoff.insert(id);self.inner.scheduler.deferred_prefill.remove(&id);
                },
                Err(error)=>{self.inner.cancel(id)?;self.failed.push_back((id,error));},
            }
        }
        Ok(())
    }
    fn lookup(&mut self,prompt:&[i32])->Result<Option<Prefix<B,M>>,ContinuousBatchError> {
        self.prefixes.lookup(self.inner.model,prompt,&self.inner.device).map_err(|e|ContinuousBatchError(e.to_string()))
    }
    #[cfg(feature="gpu-kv")]
    fn poll_gpu_remote(&mut self)->Result<(),ContinuousBatchError> {
        loop {
            let Some((id,result))=self.remote_gpu.as_mut().and_then(|r|r.poll()) else {break;};
            let Some(request)=self.inner.scheduler.pending.iter().find(|r|r.id==id) else {continue;};
            let result=result.and_then(|handoff|handoff.import(self.inner.model,&request.prompt_token_ids,&self.inner.device).map_err(|e|e.to_string()));
            match result {
                Ok(prefix)=>{
                    if let Err(e)=self.prefixes.insert(self.inner.model,prefix.clone(),&self.inner.device) {eprintln!("GPU prefix tier write: {e}");}
                    self.pending.insert(id,prefix);self.handoff.insert(id);self.inner.scheduler.deferred_prefill.remove(&id);
                },
                Err(error)=>{self.inner.cancel(id)?;self.failed.push_back((id,error));},
            }
        }
        Ok(())
    }
    fn restore_ready(&mut self)->Result<(),ContinuousBatchError> {
        if self.inner.scheduler.in_flight.is_some() {return Err(ContinuousBatchError("batch is still in flight".into()));}
        #[cfg(feature="serving")]
        self.poll_remote()?;
        #[cfg(feature="gpu-kv")]
        self.poll_gpu_remote()?;
        self.resume_swapped()?;
        self.inner.scheduler.promote_pending()?;
        let active=self.pending.keys().filter(|id|self.inner.scheduler.active.contains_key(id)).copied().collect::<Vec<_>>();
        for id in active {
            let prefix=&self.pending[&id];let length=prefix.tokens.len();
            if !self.inner.scheduler.kv_cache.can_append(id,length)? {
                if self.handoff.contains(&id) {
                    self.inner.scheduler.deferred_prefill.insert(id);
                    continue;
                }
                self.pending.remove(&id);
                continue;
            }
            let reservation=self.inner.scheduler.kv_cache.begin_append(id,length)?;
            let mut working=self.inner.model.fork_batch_cache(&self.inner.cache);
            let restored=self.inner.model.restore_prefix(&mut working,id,&reservation.block_table,&prefix.state);
            self.inner.model.sync_batch(&self.inner.device).map_err(|e|ContinuousBatchError(format!("prefix restore completion: {e}")))?;
            if let Err(error)=restored {
                self.inner.scheduler.kv_cache.cancel_append(reservation.id)?;
                return Err(ContinuousBatchError(error.to_string()));
            }
            self.inner.scheduler.kv_cache.commit_append(reservation.id)?;
            self.inner.cache=working;
            let prefix=self.pending.remove(&id).unwrap();
            self.handoff.remove(&id);self.inner.scheduler.deferred_prefill.remove(&id);
            if length==self.inner.scheduler.active[&id].prompt_token_ids.len() {self.ready.insert(id,prefix.logits);}
        }
        Ok(())
    }
    pub fn step(&mut self)->Result<Option<PrefixBatchStep>,ContinuousBatchError> {
        self.step_with_logits_processor(|_,_|Ok(()))
    }
    pub fn step_with_logits_processor(&mut self,mut processor:impl FnMut(RequestId,&mut [f32])->Result<(),GenerationError>)
        ->Result<Option<PrefixBatchStep>,ContinuousBatchError> {
        self.restore_ready()?;
        if self.ready.is_empty() && self.relieve_kv_pressure()? {self.restore_ready()?;}
        if !self.ready.is_empty() {
            let ids=self.ready.keys().copied().collect::<Vec<_>>();let mut tokens=Vec::with_capacity(ids.len());
            for id in ids {
                let mut values=self.ready[&id].clone().cast(FloatDType::F32).try_into_data()
                    .map_err(|e|ContinuousBatchError(e.to_string()))?.to_vec::<f32>().map_err(|e|ContinuousBatchError(e.to_string()))?;
                processor(id,&mut values).map_err(|e|ContinuousBatchError(e.to_string()))?;
                let token=self.inner.scheduler.complete_prefilled(id,&values,self.inner.model.batch_limits())?;
                tokens.push((id,token));self.ready.remove(&id);
            }
            self.inner.model.retain_batch_cache(&mut self.inner.cache,&self.inner.scheduler);
            return Ok(Some(PrefixBatchStep{batch:None,tokens}));
        }
        let prompts=self.inner.scheduler.active.iter().filter(|(_,r)|r.phase==RequestPhase::NeedsPrefill)
            .map(|(&id,r)|(id,r.prompt_token_ids.clone())).collect::<BTreeMap<_,_>>();
        let model=self.inner.model;let device=self.inner.device.clone();let store=&mut self.prefixes;
        let mut observer=|batch:&ScheduledBatch,cache:&M::Cache,logits:&Tensor<B,3>| {
            if batch.kind!=ScheduledBatchKind::Prefill || !store.enabled() {return;}
            for (row,sequence) in batch.sequences.iter().enumerate() {
                let result=(||->Result<(),GenerationError> {
                    let tokens=prompts.get(&sequence.request_id).ok_or_else(||GenerationError("missing batch prefix token key".into()))?[..sequence.context_length].to_vec();
                    let state=model.capture_prefix(cache,sequence.request_id)?;
                    let logits=logits.clone().slice([row..row+1,0..1,0..model.batch_limits().vocab_size]);
                    store.insert(model,Prefix{tokens,state,logits},&device)
                })();
                if let Err(error)=result {eprintln!("batch prefix tier write: {error}");}
            }
        };
        let step=self.inner.step_inner(Some(&mut processor),Some(&mut observer))?;
        Ok(step.map(|step|PrefixBatchStep {
            tokens:step.batch.sequences.iter().zip(step.generated_token_ids).filter(|(r,_)|
                step.batch.kind!=ScheduledBatchKind::Prefill || prompts.get(&r.request_id).is_some_and(|p|r.context_length==p.len()))
                .map(|(r,t)|(r.request_id,t)).collect(),batch:Some(step.batch),
        }))
    }
    pub fn cancel(&mut self,id:RequestId)->Result<Option<CancelledGeneration>,ContinuousBatchError> {
        self.complete_swap_restore()?;
        let result=match self.cancel_swapped(id) {Some(result)=>Some(result),None=>self.inner.cancel(id)?};
        self.pending.remove(&id);self.ready.remove(&id);self.handoff.remove(&id);
        #[cfg(feature="serving")]
        if let Some(remote)=&mut self.remote {remote.cancel(id);}
        #[cfg(feature="gpu-kv")]
        if let Some(remote)=&mut self.remote_gpu {remote.cancel(id);}
        Ok(result)
    }
    pub fn pop_failed(&mut self)->Option<(RequestId,String)> {self.failed.pop_front()}
    pub fn waiting_for_prefill(&self)->bool {
        #[cfg(feature="gpu-kv")]
        if let Some(remote)=&self.remote_gpu {return remote.waiting();}
        #[cfg(feature="serving")]
        if let Some(remote)=&self.remote {return remote.waiting();}
        false
    }
    pub fn pop_finished(&mut self)->Option<FinishedGeneration> {self.inner.pop_finished()}
    pub fn is_idle(&self)->bool {self.inner.is_idle() && self.swap.is_empty() && self.swap_restore.is_none()}
    pub fn snapshot(&self)->ContinuousBatchSnapshot {
        let mut snapshot=self.inner.snapshot();snapshot.pending_requests+=self.swap.len();
        snapshot.has_in_flight_batch|=self.swap_restore.is_some();snapshot
    }
    pub fn prefix_stats(&self)->PrefixCacheStats {self.prefixes.stats()}
    pub fn cache(&self)->&M::Cache {self.inner.cache()}
}

impl ContinuousBatchScheduler {
    fn complete_prefilled(&mut self,id:RequestId,values:&[f32],limits:CausalModelLimits)->Result<i32,ContinuousBatchError> {
        let request=self.active.get_mut(&id).ok_or_else(||ContinuousBatchError("missing prefilled request".into()))?;
        if self.in_flight.is_some() || request.phase!=RequestPhase::NeedsPrefill
            || self.kv_cache.committed_tokens(id)?!=request.prompt_token_ids.len() || values.len()!=limits.vocab_size {
            return Err(ContinuousBatchError("prefilled request has incomplete state or logits".into()));
        }
        let (token,sampler)=if let Some(sampler)=&request.sampler {
            let mut next=sampler.clone();let token=next.sample_with_history(values,&request.prompt_token_ids,&request.generated_token_ids)
                .map_err(|e|ContinuousBatchError(e.to_string()))?;
            next.clear_workspace();(token,Some(next))
        } else {(super::sampling::greedy_token(values,0)?,None)};
        request.generated_token_ids.push(token);request.history.push(token);request.sampler=sampler;
        let stopped_on_eos=request.eos.contains(&token);
        if stopped_on_eos || request.generated_token_ids.len()>=request.generation.max_new_tokens {
            self.kv_cache.remove_sequence(id)?;
            let request=self.active.remove(&id).unwrap();
            self.finished.push_back(FinishedGeneration{request_id:id,prompt_token_ids:request.prompt_token_ids,
                generated_token_ids:request.generated_token_ids,stopped_on_eos});
        } else {request.phase=RequestPhase::AwaitingDecode(token);}
        Ok(token)
    }
}
