use super::*;
use ruda_tensor::api::TensorData;

#[derive(Debug,Clone,Copy,Default,PartialEq,Eq)]
pub struct ActiveKvSwapConfig {
    /// Separate from reusable prefix-cache memory; zero disables active swap.
    pub host_bytes:usize,
}

#[derive(Debug,Clone,Copy,Default,PartialEq,Eq)]
pub struct ActiveKvSwapStats {
    pub host_bytes:usize,
    pub host_limit:usize,
    pub suspended_requests:usize,
    pub swaps_out:u64,
    pub swaps_in:u64,
    pub budget_misses:u64,
}

struct SwappedRequest {
    request:ActiveRequest,
    position:usize,
    prefix:Vec<u8>,
    logits:Option<Vec<u8>>,
    bytes:usize,
}

#[derive(Default)]
pub(super) struct ActiveKvSwap {
    config:ActiveKvSwapConfig,
    entries:BTreeMap<RequestId,SwappedRequest>,
    stats:ActiveKvSwapStats,
}
impl ActiveKvSwap {
    pub(super) fn is_empty(&self)->bool {self.entries.is_empty()}
    pub(super) fn len(&self)->usize {self.entries.len()}
}

// Keep all source/destination handles and the page reservation alive across a
// failed completion fence. Retrying the fence never repeats the restore write.
pub(super) struct SwapRestore<B:Backend,M:PrefixBatchModel<B>> {
    id:RequestId,
    reservation:Option<KvReservationId>,
    cache:M::Cache,
    _prefix:Option<M::Prefix>,
    logits:Option<Tensor<B,3>>,
    error:Option<GenerationError>,
}

impl<B:Backend,M:PrefixBatchModel<B>> PrefixBatchExecutor<'_,B,M> {
    pub fn set_active_kv_swap(&mut self,config:ActiveKvSwapConfig)->Result<(),ContinuousBatchError> {
        if !self.is_idle() {return Err(ContinuousBatchError("configure active KV swap before submitting requests".into()));}
        self.swap.config=config;
        self.swap.stats.host_limit=config.host_bytes;
        if config.host_bytes>0 && self.inner.scheduler.prefill_chunk_tokens.is_none() {
            self.inner.scheduler.prefill_chunk_tokens=Some(self.inner.scheduler.config.max_batch_tokens);
        }
        Ok(())
    }

    pub fn active_kv_swap_stats(&self)->ActiveKvSwapStats {
        ActiveKvSwapStats{suspended_requests:self.swap.len(),..self.swap.stats}
    }

    pub(super) fn check_swap_queue(&self,max_new_tokens:usize)->Result<(),ContinuousBatchError> {
        if max_new_tokens>0 && self.inner.scheduler.options.max_pending_requests
            .is_some_and(|limit|self.inner.scheduler.pending.len()+self.swap.len()>=limit) {
            return Err(ContinuousBatchError("pending/suspended request queue is full".into()));
        }
        Ok(())
    }

    fn refresh_swap_admission(&mut self) {
        self.inner.scheduler.admission_ceiling=self.swap.entries.keys().next().copied();
    }

    /// Suspend the exact committed request, without re-prefill or resampling.
    /// False leaves it on device when disabled, unknown, or over the host limit.
    pub fn swap_out(&mut self,id:RequestId)->Result<bool,ContinuousBatchError> {
        self.complete_swap_restore()?;
        if self.swap.config.host_bytes==0 {return Ok(false);}
        let scheduler=&self.inner.scheduler;
        if scheduler.in_flight.is_some() {return Err(ContinuousBatchError("cannot swap an in-flight batch".into()));}
        if !scheduler.active.contains_key(&id) {return Ok(false);}
        let position=scheduler.kv_cache.committed_tokens(id)?;
        let remaining=self.swap.config.host_bytes.saturating_sub(self.swap.stats.host_bytes);
        let prefix=if position==0 {Vec::new()} else {
            let state=self.inner.model.capture_prefix(&self.inner.cache,id)
                .map_err(|e|ContinuousBatchError(e.to_string()))?;
            if self.inner.model.prefix_position(&state)!=position {
                return Err(ContinuousBatchError("active swap prefix is not at the committed position".into()));
            }
            let logits_bytes=self.ready.get(&id).map_or(0,|t|t.dims().into_iter()
                .fold(t.dtype().size(),usize::saturating_mul));
            if self.inner.model.prefix_bytes(&state).saturating_add(logits_bytes)>remaining {
                self.swap.stats.budget_misses+=1;return Ok(false);
            }
            self.inner.model.encode_prefix(&state,self.prefixes.model_id())
                .map_err(|e|ContinuousBatchError(e.to_string()))?
        };
        let logits=self.ready.get(&id).map(|logits| {
            let data=logits.clone().try_into_data().map_err(|e|ContinuousBatchError(e.to_string()))?;
            rmp_serde::to_vec_named(&data).map_err(|e|ContinuousBatchError(e.to_string()))
        }).transpose()?;
        let bytes=prefix.capacity().checked_add(logits.as_ref().map_or(0,Vec::capacity))
            .ok_or_else(||ContinuousBatchError("active KV host byte count overflow".into()))?;
        if bytes>remaining {self.swap.stats.budget_misses+=1;return Ok(false);}
        self.inner.model.sync_batch(&self.inner.device)
            .map_err(|e|ContinuousBatchError(format!("active swap download completion: {e}")))?;
        // Host copy and completion fence precede reclaiming any physical page.
        self.inner.scheduler.kv_cache.remove_sequence(id)?;
        let request=self.inner.scheduler.active.remove(&id).expect("active swap request");
        self.ready.remove(&id);
        self.swap.entries.insert(id,SwappedRequest{request,position,prefix,logits,bytes});
        self.swap.stats.host_bytes+=bytes;
        self.swap.stats.swaps_out+=1;
        self.refresh_swap_admission();
        self.inner.model.retain_batch_cache(&mut self.inner.cache,&self.inner.scheduler);
        Ok(true)
    }

    fn next_tokens(&self,id:RequestId,request:&ActiveRequest,position:usize,ready:bool)->usize {
        if ready {return 0;}
        if let Some(prefix)=self.pending.get(&id) {return prefix.tokens.len().saturating_sub(position);}
        match request.phase {
            RequestPhase::AwaitingDecode(_)=>1,
            RequestPhase::NeedsPrefill=>{
                let remaining=request.prompt_token_ids.len().saturating_sub(position);
                remaining.min(self.inner.scheduler.prefill_chunk_tokens.unwrap_or(remaining))
                    .min(self.inner.scheduler.config.max_batch_tokens)
            },
            RequestPhase::InFlight(_)=>0,
        }
    }

    fn resume_pages(&self,id:RequestId,entry:&SwappedRequest)->usize {
        let kv=self.inner.scheduler.kv_cache.config();
        let pages=(entry.position+self.next_tokens(id,&entry.request,entry.position,entry.logits.is_some())).div_ceil(kv.block_size);
        if self.inner.scheduler.options.kv_admission==KvAdmissionPolicy::ReserveSequenceCapacity {
            pages.max(ContinuousBatchScheduler::request_page_budget(entry.request.prompt_token_ids.len(),&entry.request.generation,kv.block_size))
        } else {pages}
    }

    fn can_resume(&self,id:RequestId)->bool {
        let scheduler=&self.inner.scheduler;
        if scheduler.active.len()>=scheduler.config.max_active_sequences {return false;}
        let pages=self.resume_pages(id,&self.swap.entries[&id]);
        let block=scheduler.kv_cache.config().block_size;
        let mut available=scheduler.kv_cache.free_page_count();
        for (&older,request) in scheduler.active.range(..id) {
            let Ok(position)=scheduler.kv_cache.committed_tokens(older) else {return false;};
            let next=self.next_tokens(older,request,position,self.ready.contains_key(&older));
            let needed=(position+next).div_ceil(block).saturating_sub(position.div_ceil(block));
            let Some(left)=available.checked_sub(needed) else {return false;};available=left;
        }
        if pages>available {return false;}
        if scheduler.options.kv_admission==KvAdmissionPolicy::ReserveSequenceCapacity {
            let kv=scheduler.kv_cache.config();
            let reserved=scheduler.active.values().map(|r|ContinuousBatchScheduler::request_page_budget(
                r.prompt_token_ids.len(),&r.generation,kv.block_size)).sum::<usize>();
            if pages>kv.num_pages.saturating_sub(reserved) {return false;}
        }
        true
    }

    /// Restore one suspended request if both its prefix and its next scheduled
    /// input fit. The host state remains authoritative until the GPU fence.
    pub fn swap_in(&mut self,id:RequestId)->Result<bool,ContinuousBatchError> {
        self.complete_swap_restore()?;
        if self.inner.scheduler.in_flight.is_some() {return Err(ContinuousBatchError("cannot restore during an in-flight batch".into()));}
        let Some(entry)=self.swap.entries.get(&id) else {return Ok(false);};
        if !self.can_resume(id) {return Ok(false);}
        let position=entry.position;
        let state=if position==0 {None} else {Some(self.inner.model.decode_prefix(
            &entry.prefix,self.prefixes.model_id(),&self.inner.device).map_err(|e|ContinuousBatchError(e.to_string()))?)};
        if state.as_ref().is_some_and(|p|self.inner.model.prefix_position(p)!=position) {
            return Err(ContinuousBatchError("restored active prefix position differs from saved state".into()));
        }
        let logits=entry.logits.as_ref().map(|bytes| {
            let data:TensorData=rmp_serde::from_slice(bytes).map_err(|e|ContinuousBatchError(e.to_string()))?;
            if data.shape!=[1,1,self.inner.model.batch_limits().vocab_size].into() || !data.dtype.is_float() {
                return Err(ContinuousBatchError("active swap logits have invalid shape or dtype".into()));
            }
            let dtype=data.dtype;Ok(Tensor::<B,3>::from_data(data,(&self.inner.device,dtype)))
        }).transpose()?;
        self.inner.scheduler.kv_cache.create_sequence(id)?;
        let reservation=if position==0 {None} else {
            match self.inner.scheduler.kv_cache.begin_append(id,position) {
                Ok(reservation)=>Some(reservation),
                Err(error)=>{self.inner.scheduler.kv_cache.remove_sequence(id)?;return Err(error.into());},
            }
        };
        let mut cache=self.inner.model.fork_batch_cache(&self.inner.cache);
        let error=match (&state,&reservation) {
            (Some(state),Some(reservation))=>self.inner.model.restore_prefix(&mut cache,id,&reservation.block_table,state).err(),
            _=>None,
        };
        self.swap_restore=Some(SwapRestore{id,reservation:reservation.map(|r|r.id),cache,_prefix:state,logits,error});
        self.complete_swap_restore()?;
        Ok(true)
    }

    pub(super) fn complete_swap_restore(&mut self)->Result<(),ContinuousBatchError> {
        if self.swap_restore.is_none() {return Ok(());}
        self.inner.model.sync_batch(&self.inner.device)
            .map_err(|e|ContinuousBatchError(format!("active swap upload completion (reservation retained): {e}")))?;
        let restore=self.swap_restore.as_ref().expect("swap restore transaction");
        if let Some(error)=&restore.error {
            let error=error.to_string();let id=restore.id;
            if let Some(reservation)=restore.reservation {self.inner.scheduler.kv_cache.cancel_append(reservation)?;}
            self.inner.scheduler.kv_cache.remove_sequence(id)?;
            self.swap_restore=None;
            return Err(ContinuousBatchError(error));
        }
        if let Some(reservation)=restore.reservation {self.inner.scheduler.kv_cache.commit_append(reservation)?;}
        let restore=self.swap_restore.take().expect("completed restore transaction");
        let entry=self.swap.entries.remove(&restore.id).expect("host state retained through GPU completion");
        self.inner.cache=restore.cache;
        self.inner.scheduler.active.insert(restore.id,entry.request);
        if let Some(logits)=restore.logits {self.ready.insert(restore.id,logits);}
        self.swap.stats.host_bytes-=entry.bytes;
        self.swap.stats.swaps_in+=1;
        self.refresh_swap_admission();
        Ok(())
    }

    pub(super) fn resume_swapped(&mut self)->Result<(),ContinuousBatchError> {
        self.complete_swap_restore()?;
        while let Some(id)=self.swap.entries.keys().next().copied() {
            if !self.swap_in(id)? {break;}
        }
        Ok(())
    }

    pub(super) fn cancel_swapped(&mut self,id:RequestId)->Option<CancelledGeneration> {
        let entry=self.swap.entries.remove(&id)?;
        self.swap.stats.host_bytes-=entry.bytes;
        self.inner.scheduler.deferred_prefill.remove(&id);
        self.refresh_swap_admission();
        Some(CancelledGeneration{request_id:id,prompt_token_ids:entry.request.prompt_token_ids,
            generated_token_ids:entry.request.generated_token_ids})
    }

    /// On an actual page stall, preserve the oldest ready request and evict
    /// newer requests until it can advance. Never recompute an evicted prefix.
    pub(super) fn relieve_kv_pressure(&mut self)->Result<bool,ContinuousBatchError> {
        if self.swap.config.host_bytes==0 {return Ok(false);}
        let scheduler=&self.inner.scheduler;
        let mut target=self.swap.entries.keys().next().copied();
        for (&id,request) in &scheduler.active {
            if scheduler.deferred_prefill.contains(&id) && !self.handoff.contains(&id) {continue;}
            let position=scheduler.kv_cache.committed_tokens(id)?;
            let next=self.next_tokens(id,request,position,self.ready.contains_key(&id));
            if next>0 && scheduler.kv_cache.can_append(id,next)? {return Ok(false);}
            if next>0 {target=Some(target.map_or(id,|oldest|oldest.min(id)));}
        }
        let Some(target)=target else {return Ok(false);};
        let victims=self.inner.scheduler.active.keys().rev().filter(|&&id|id>target).copied().collect::<Vec<_>>();
        let mut changed=false;
        for victim in victims {
            let fits=if self.swap.entries.contains_key(&target) {self.can_resume(target)} else {
                let scheduler=&self.inner.scheduler;let position=scheduler.kv_cache.committed_tokens(target)?;
                scheduler.kv_cache.can_append(target,self.next_tokens(target,&scheduler.active[&target],position,false))?
            };
            if fits {break;}
            changed|=self.swap_out(victim)?;
        }
        if self.swap.entries.contains_key(&target) {changed|=self.swap_in(target)?;}
        Ok(changed)
    }
}
