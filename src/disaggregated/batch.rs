use super::*;
use crate::{PagedKvCacheConfig,PagedKvCacheManager,PrefixBatchModel,RequestId,
    ScheduledBatch,ScheduledBatchKind,ScheduledSequence};
use crate::batch_prefix::{Prefix,PrefixStore};

pub(super) fn prefill<B:Backend,M:PrefixBatchModel<B>>(model:&M,device:&B::Device,tokens:&[i32],
    mut kv:PagedKvCacheConfig,chunk:usize,store:&mut PrefixStore<B,M>)->Result<Prefix<B,M>,GenerationError> {
    crate::generation::validate_generation(&model.batch_limits(),tokens,&GreedyGenerationConfig {
        max_new_tokens:0,eos_token_ids:Vec::new(),
    })?;
    let reused=store.lookup(model,tokens,device)?;
    if let Some(prefix)=&reused {if prefix.tokens.len()==tokens.len() {return Ok(prefix.clone());}}
    let pages=tokens.len().div_ceil(kv.block_size);
    if pages>kv.num_pages {return Err(error("prefill prompt exceeds the configured KV page pool"));}
    kv.num_pages=pages;
    let mut manager=PagedKvCacheManager::new(kv).map_err(error)?;
    let id=RequestId(1);manager.create_sequence(id).map_err(error)?;
    let mut cache=model.new_batch_cache(kv);let mut position=0;
    if let Some(prefix)=reused {
        let reservation=manager.begin_append(id,prefix.tokens.len()).map_err(error)?;
        let result=model.restore_prefix(&mut cache,id,&reservation.block_table,&prefix.state);
        model.sync_batch(device)?;result?;manager.commit_append(reservation.id).map_err(error)?;
        position=prefix.tokens.len();
    }
    let mut last=None;let mut batch_id=1;
    while position<tokens.len() {
        let end=position.saturating_add(chunk).min(tokens.len());
        let reservation=manager.begin_append(id,end-position).map_err(error)?;
        let batch=ScheduledBatch{id:batch_id,kind:ScheduledBatchKind::Prefill,sequences:vec![ScheduledSequence {
            request_id:id,token_ids:tokens[position..end].to_vec(),start_position:position,context_length:end,
            block_table:reservation.block_table,reservation_id:reservation.id,samples_token:end==tokens.len(),
        }]};
        let result=model.forward_batch(&batch,&mut cache,device);
        model.sync_batch(device)?;
        let logits=result?;
        if logits.dims()!=[1,1,model.batch_limits().vocab_size] {return Err(error("prefill batch logits shape differs"));}
        manager.commit_append(reservation.id).map_err(error)?;
        position=end;last=Some(logits);batch_id+=1;
    }
    let prefix=Prefix{tokens:tokens.to_vec(),state:model.capture_prefix(&cache,id)?,logits:last.ok_or_else(||error("missing prefill logits"))?};
    model.sync_batch(device)?;
    if let Err(e)=store.insert(model,prefix.clone(),device) {eprintln!("remote batch prefix tier write: {e}");}
    Ok(prefix)
}

/// Prefill-only handoff for the complete native batch adapter state. No token
/// is selected or decoded to obtain this snapshot. Version 2 is deliberately
/// distinct from the single-session version 1 cache representation.
pub fn run_batch_prefill_service<B:Backend,M:PrefixBatchModel<B>>(
    model:M,device:B::Device,model_id:String,cache_config:PrefixCacheConfig,
    kv:PagedKvCacheConfig,max_batch_tokens:usize,address:&str,on_ready:impl FnOnce()->bool,
)->Result<(),GenerationError> {
    kv.validate().map_err(error)?;
    if max_batch_tokens==0 {return Err(error("prefill chunk token limit must be positive"));}
    let limits=model.batch_limits();
    if kv.max_sequence_length!=limits.max_sequence_length {return Err(error("prefill model and KV context limits differ"));}
    let chunk=cache_config.block_tokens.min(max_batch_tokens);
    let mut store=PrefixStore::<B,M>::new(model_id.clone(),limits,cache_config)?;
    let listener=TcpListener::bind(address).map_err(error)?;
    if !on_ready() {return Ok(());}
    for incoming in listener.incoming() {
        let mut stream=match incoming {Ok(s)=>s,Err(e)=>{eprintln!("batch prefill connection: {e}");continue;}};
        let result=(|| {
            let request:PrefillRequest=rmp_serde::from_slice(&read_frame(&mut stream)?).map_err(error)?;
            if !matches!(request.version,2|3) || request.model_id!=model_id || request.limits!=limits {
                return Err(error("batch prefill model identity, version or limits differ"));
            }
            let prefix=prefill(&model,&device,&request.tokens,kv,chunk,&mut store)?;
            store.export_handoff(&model,&prefix,&device,request.version==3)
        })();
        let response=match result {
            Ok(mut bytes)=>{bytes.insert(0,0);bytes},
            Err(e)=>{let mut bytes=vec![1];bytes.extend_from_slice(e.to_string().as_bytes());bytes},
        };
        if let Err(e)=write_frame(&mut stream,&response) {eprintln!("batch prefill handoff: {e}");}
    }
    Ok(())
}

#[cfg(feature="serving")]
mod network {
    use super::*;
    use std::collections::BTreeMap;
    use tokio::{io::{AsyncReadExt,AsyncWriteExt},sync::mpsc,task::JoinHandle};

    async fn exchange(address:String,request:PrefillRequest)->Result<Vec<u8>,String> {
        let result=async {
            let payload=rmp_serde::to_vec(&request).map_err(error)?;
            let mut stream=tokio::net::TcpStream::connect(address).await.map_err(error)?;
            stream.set_nodelay(true).map_err(error)?;
            stream.write_all(MAGIC).await.map_err(error)?;
            stream.write_all(&(payload.len() as u64).to_be_bytes()).await.map_err(error)?;
            stream.write_all(&Sha256::digest(&payload)).await.map_err(error)?;
            stream.write_all(&payload).await.map_err(error)?;
            stream.flush().await.map_err(error)?;
            let mut magic=[0;8];let mut length=[0;8];let mut digest=[0;32];
            stream.read_exact(&mut magic).await.map_err(error)?;
            if &magic!=MAGIC {return Err(error("incompatible batch prefill protocol"));}
            stream.read_exact(&mut length).await.map_err(error)?;
            stream.read_exact(&mut digest).await.map_err(error)?;
            let length=usize::try_from(u64::from_be_bytes(length)).map_err(error)?;
            let mut response=Vec::new();response.try_reserve_exact(length).map_err(error)?;response.resize(length,0);
            stream.read_exact(&mut response).await.map_err(error)?;
            if Sha256::digest(&response)[..]!=digest {return Err(error("batch prefill payload digest differs"));}
            match response.first().copied() {
                Some(0)=>{response.remove(0);Ok(response)},
                Some(1)=>Err(error(String::from_utf8_lossy(&response[1..]))),
                _=>Err(error("invalid batch prefill response")),
            }
        }.await;
        result.map_err(|e|e.to_string())
    }

    pub(crate) struct RemoteBatchNetwork {
        runtime:Option<tokio::runtime::Runtime>,address:String,model_id:String,limits:CausalModelLimits,capacity:usize,
        sender:mpsc::Sender<(RequestId,Result<Vec<u8>,String>)>,
        receiver:mpsc::Receiver<(RequestId,Result<Vec<u8>,String>)>,
        requests:BTreeMap<RequestId,JoinHandle<()>>,
    }
    impl RemoteBatchNetwork {
        pub fn new(remote:RemotePrefill,model_id:String,limits:CausalModelLimits,capacity:usize)->Result<Self,GenerationError> {
            if capacity==0 || remote.address.is_empty() {return Err(error("remote prefill needs an address and bounded positive queue"));}
            let runtime=tokio::runtime::Builder::new_multi_thread().worker_threads(1).enable_io().build().map_err(error)?;
            let (sender,receiver)=mpsc::channel(capacity);
            Ok(Self{runtime:Some(runtime),address:remote.address,model_id,limits,capacity,sender,receiver,requests:BTreeMap::new()})
        }
        pub fn submit(&mut self,id:RequestId,tokens:Vec<i32>)->Result<(),GenerationError> {
            if self.requests.len()>=self.capacity || self.requests.contains_key(&id) {return Err(error("remote prefill queue is full or request already exists"));}
            let address=self.address.clone();let sender=self.sender.clone();let owner=std::thread::current();
            let request=PrefillRequest{version:3,model_id:self.model_id.clone(),limits:self.limits,tokens};
            let task=self.runtime.as_ref().unwrap().spawn(async move {
                let result=exchange(address,request).await;
                if sender.send((id,result)).await.is_ok() {owner.unpark();}
            });
            self.requests.insert(id,task);Ok(())
        }
        pub fn poll(&mut self)->Option<(RequestId,Result<Vec<u8>,String>)> {
            while let Ok((id,result))=self.receiver.try_recv() {
                if self.requests.remove(&id).is_some() {return Some((id,result));}
            }
            None
        }
        pub fn cancel(&mut self,id:RequestId) {if let Some(task)=self.requests.remove(&id) {task.abort();}}
        pub fn waiting(&self)->bool {!self.requests.is_empty()}
    }
    impl Drop for RemoteBatchNetwork {
        fn drop(&mut self) {
            for task in self.requests.values() {task.abort();}
            if let Some(runtime)=self.runtime.take() {runtime.shutdown_background();}
        }
    }
}
#[cfg(feature="serving")]
pub(crate) use network::RemoteBatchNetwork;
