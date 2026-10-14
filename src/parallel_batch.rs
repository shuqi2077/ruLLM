use crate::{DeviceBatchModel,PrefixBatchModel};
use ruda_tensor::api::backend::Backend;

pub struct ParallelBatchCache<B:Backend,M:DeviceBatchModel<B>> {
    pub(crate) ranks:Vec<M::Cache>,
    pub(crate) devices:Vec<B::Device>,
}

pub struct ParallelBatchPrefix<B:Backend,M:PrefixBatchModel<B>> {
    pub(crate) ranks:Vec<M::Prefix>,
    pub(crate) devices:Vec<B::Device>,
    pub(crate) position:usize,
}
impl<B:Backend,M:PrefixBatchModel<B>> Clone for ParallelBatchPrefix<B,M> {
    fn clone(&self)->Self {Self{ranks:self.ranks.clone(),devices:self.devices.clone(),position:self.position}}
}

pub struct ParallelBatchDelta<B:Backend,M:crate::SpeculativeBatchModel<B>> {
    pub(crate) ranks:Vec<M::Delta>,pub(crate) devices:Vec<B::Device>,
}
impl<B:Backend,M:crate::SpeculativeBatchModel<B>> Clone for ParallelBatchDelta<B,M> {
    fn clone(&self)->Self {Self{ranks:self.ranks.clone(),devices:self.devices.clone()}}
}

macro_rules! parallel_batch_model {
    ($group:ident,$rank:ident) => {crate::parallel_batch::parallel_batch_model!($group,$rank,|_group|crate::PrefixShard::default());};
    ($group:ident,$rank:ident,|$this:ident|$layout:expr) => {
        impl<B:ruda_tensor::distributed::DistributedBackend> crate::DeviceBatchModel<B> for $group<B>
        where $rank<B>:crate::PrefixBatchModel<B>+Sync,
            <$rank<B> as crate::DeviceBatchModel<B>>::Cache:Send {
            type Cache=crate::ParallelBatchCache<B,$rank<B>>;
            fn batch_limits(&self)->crate::CausalModelLimits {self.ranks[0].batch_limits()}
            fn new_batch_cache(&self,config:crate::PagedKvCacheConfig)->Self::Cache {
                Self::Cache{ranks:self.ranks.iter().map(|m|m.new_batch_cache(config)).collect(),devices:self.devices.clone()}
            }
            fn fork_batch_cache(&self,cache:&Self::Cache)->Self::Cache {
                Self::Cache{ranks:self.ranks.iter().zip(&cache.ranks).map(|(m,c)|m.fork_batch_cache(c)).collect(),devices:cache.devices.clone()}
            }
            fn retain_batch_cache(&self,cache:&mut Self::Cache,scheduler:&crate::ContinuousBatchScheduler) {
                for (model,state) in self.ranks.iter().zip(&mut cache.ranks) {model.retain_batch_cache(state,scheduler);}
            }
            fn sync_batch(&self,device:&B::Device)->Result<(),crate::GenerationError> {
                if self.devices.first()!=Some(device) {return Err(crate::GenerationError("parallel batch root device differs".into()));}
                let mut failure=None;
                for (model,device) in self.ranks.iter().zip(&self.devices) {
                    if let Err(error)=model.sync_batch(device) {failure=Some(error);}
                }
                if let Some(error)=failure {Err(error)} else {Ok(())}
            }
            fn forward_batch(&self,batch:&crate::ScheduledBatch,cache:&mut Self::Cache,device:&B::Device)
                ->Result<ruda_tensor::api::Tensor<B,3>,crate::GenerationError> {
                self.forward_batch_lookahead(batch,&std::collections::BTreeMap::new(),cache,device)
            }
            fn lookahead_tokens(&self)->usize {self.ranks.iter().map(|rank|rank.lookahead_tokens()).min().unwrap_or(0)}
            fn forward_batch_lookahead(&self,batch:&crate::ScheduledBatch,
                lookahead:&std::collections::BTreeMap<crate::RequestId,crate::PagedKvAppendReservation>,
                cache:&mut Self::Cache,device:&B::Device)->Result<ruda_tensor::api::Tensor<B,3>,crate::GenerationError> {
                if cache.devices!=self.devices || cache.ranks.len()!=self.ranks.len() || self.devices.first()!=Some(device) {
                    return Err(crate::GenerationError("parallel batch topology or root device differs".into()));
                }
                std::thread::scope(|scope| {
                    let mut handles=Vec::with_capacity(self.ranks.len());
                    for ((model,state),device) in self.ranks.iter().zip(&mut cache.ranks).zip(&self.devices) {
                        handles.push(scope.spawn(move || {
                            let output=model.forward_batch_lookahead(batch,lookahead,state,device);
                            B::sync(device).map_err(|e|crate::GenerationError(format!("parallel batch rank completion: {e}")))?;
                            output
                        }));
                    }
                    let mut root=None;let mut error=None;
                    for (rank,handle) in handles.into_iter().enumerate() {
                        match handle.join() {
                            Ok(Ok(output)) if rank==0=>root=Some(output),Ok(Ok(_))=>(),
                            Ok(Err(e))=>error=Some(e),Err(_)=>error=Some(crate::GenerationError("parallel batch rank panicked".into())),
                        }
                    }
                    if let Some(error)=error {Err(error)} else {root.ok_or_else(||crate::GenerationError("parallel batch has no root logits".into()))}
                })
            }
        }
        impl<B:ruda_tensor::distributed::DistributedBackend> crate::PrefixBatchModel<B> for $group<B>
        where $rank<B>:crate::PrefixBatchModel<B>+Sync,
            <$rank<B> as crate::DeviceBatchModel<B>>::Cache:Send {
            type Prefix=crate::ParallelBatchPrefix<B,$rank<B>>;
            fn prefix_tensor_transfer(&self)->bool {self.ranks.iter().all(|m|m.prefix_tensor_transfer())}
            fn export_prefix_tensors(&self,prefix:&Self::Prefix,shard:crate::PrefixShard)->Result<crate::PrefixTensors<B>,crate::GenerationError> {
                if shard!=crate::PrefixShard::default() || prefix.ranks.len()!=self.ranks.len() || prefix.devices!=self.devices {
                    return Err(crate::GenerationError("parallel tensor prefix source topology differs".into()));
                }
                let $this=self;let layout=$layout;
                let mut result:Option<crate::PrefixTensors<B>>=None;
                for (rank,((model,state),device)) in self.ranks.iter().zip(&prefix.ranks).zip(&self.devices).enumerate() {
                    let part=model.export_prefix_tensors(state,crate::PrefixShard{rank,world:self.ranks.len(),..layout})?;
                    model.sync_batch(device)?;
                    if let Some(result)=&mut result {result.merge(part)?;} else {result=Some(part);}
                }
                result.ok_or_else(||crate::GenerationError("parallel tensor prefix has no ranks".into()))
            }
            fn import_prefix_tensors(&self,tensors:&crate::PrefixTensors<B>,shard:crate::PrefixShard,device:&B::Device)
                ->Result<Self::Prefix,crate::GenerationError> {
                if shard!=crate::PrefixShard::default() || self.devices.first()!=Some(device) {
                    return Err(crate::GenerationError("parallel tensor prefix destination topology differs".into()));
                }
                let $this=self;let layout=$layout;
                let mut ranks=Vec::with_capacity(self.ranks.len());
                for (rank,(model,device)) in self.ranks.iter().zip(&self.devices).enumerate() {
                    let prefix=model.import_prefix_tensors(tensors,crate::PrefixShard{rank,world:self.ranks.len(),..layout},device)?;
                    if model.prefix_position(&prefix)!=tensors.position {return Err(crate::GenerationError("parallel tensor prefix position differs".into()));}
                    model.sync_batch(device)?;ranks.push(prefix);
                }
                Ok(Self::Prefix{ranks,devices:self.devices.clone(),position:tensors.position})
            }
            fn capture_prefix(&self,cache:&<Self as crate::DeviceBatchModel<B>>::Cache,id:crate::RequestId)
                ->Result<Self::Prefix,crate::GenerationError> {
                if cache.ranks.len()!=self.ranks.len() || cache.devices!=self.devices {
                    return Err(crate::GenerationError("parallel prefix topology differs".into()));
                }
                let mut ranks=Vec::with_capacity(self.ranks.len());let mut position=None;
                for ((model,state),device) in self.ranks.iter().zip(&cache.ranks).zip(&self.devices) {
                    let prefix=model.capture_prefix(state,id)?;
                    B::sync(device).map_err(|e|crate::GenerationError(e.to_string()))?;
                    let p=model.prefix_position(&prefix);
                    if position.is_some_and(|previous|previous!=p) {return Err(crate::GenerationError("parallel prefix rank positions differ".into()));}
                    position=Some(p);ranks.push(prefix);
                }
                Ok(Self::Prefix{ranks,devices:self.devices.clone(),position:position.unwrap_or(0)})
            }
            fn prefix_position(&self,prefix:&Self::Prefix)->usize {prefix.position}
            fn prefix_bytes(&self,prefix:&Self::Prefix)->usize {
                self.ranks.iter().zip(&prefix.ranks).fold(0usize,|sum,(m,p)|sum.saturating_add(m.prefix_bytes(p)))
            }
            fn restore_prefix(&self,cache:&mut <Self as crate::DeviceBatchModel<B>>::Cache,id:crate::RequestId,
                table:&[crate::KvPageId],prefix:&Self::Prefix)->Result<(),crate::GenerationError> {
                if cache.ranks.len()!=self.ranks.len() || prefix.ranks.len()!=self.ranks.len()
                    || cache.devices!=self.devices || prefix.devices!=self.devices
                    || self.ranks.iter().zip(&prefix.ranks).any(|(m,p)|m.prefix_position(p)!=prefix.position) {
                    return Err(crate::GenerationError("parallel prefix destination topology/position differs".into()));
                }
                let mut failure=None;
                for (((model,state),prefix),device) in self.ranks.iter().zip(&mut cache.ranks).zip(&prefix.ranks).zip(&self.devices) {
                    if failure.is_none() {if let Err(error)=model.restore_prefix(state,id,table,prefix) {failure=Some(error);}}
                    if let Err(error)=B::sync(device) {failure=Some(crate::GenerationError(format!("parallel prefix completion: {error}")));}
                }
                if let Some(error)=failure {Err(error)} else {Ok(())}
            }
            fn encode_prefix(&self,prefix:&Self::Prefix,model_id:&str)->Result<Vec<u8>,crate::GenerationError> {
                if prefix.devices!=self.devices || prefix.ranks.len()!=self.ranks.len() {
                    return Err(crate::GenerationError("parallel prefix serialization topology differs".into()));
                }
                let mut ranks=Vec::with_capacity(self.ranks.len());
                for (rank,((model,prefix),device)) in self.ranks.iter().zip(&prefix.ranks).zip(&self.devices).enumerate() {
                    B::sync(device).map_err(|e|crate::GenerationError(e.to_string()))?;
                    let identity=format!("{}:{}:{}:{rank}",model_id,stringify!($group),self.ranks.len());
                    ranks.push(model.encode_prefix(prefix,&identity)?);
                }
                rmp_serde::to_vec(&(1u32,model_id,prefix.position,ranks)).map_err(|e|crate::GenerationError(e.to_string()))
            }
            fn decode_prefix(&self,bytes:&[u8],model_id:&str,device:&B::Device)->Result<Self::Prefix,crate::GenerationError> {
                let (version,identity,position,ranks):(u32,String,usize,Vec<Vec<u8>>)=rmp_serde::from_slice(bytes)
                    .map_err(|e|crate::GenerationError(e.to_string()))?;
                if version!=1 || identity!=model_id || self.devices.first()!=Some(device) || ranks.len()!=self.ranks.len() {
                    return Err(crate::GenerationError("parallel prefix serialized identity/topology differs".into()));
                }
                let mut decoded=Vec::with_capacity(ranks.len());
                for (rank,((model,bytes),device)) in self.ranks.iter().zip(ranks).zip(&self.devices).enumerate() {
                    let identity=format!("{}:{}:{}:{rank}",model_id,stringify!($group),self.ranks.len());
                    let prefix=model.decode_prefix(&bytes,&identity,device)?;
                    if model.prefix_position(&prefix)!=position {return Err(crate::GenerationError("parallel prefix serialized position differs".into()));}
                    decoded.push(prefix);
                }
                Ok(Self::Prefix{ranks:decoded,devices:self.devices.clone(),position})
            }
        }
    };
}
pub(crate) use parallel_batch_model;

macro_rules! parallel_speculative_model {
    ($group:ident,$rank:ident)=>{
        impl<B:ruda_tensor::distributed::DistributedBackend> crate::SpeculativeBatchModel<B> for $group<B>
        where $rank<B>:crate::SpeculativeBatchModel<B>+Sync,
            <$rank<B> as crate::DeviceBatchModel<B>>::Cache:Send {
            type Delta=crate::ParallelBatchDelta<B,$rank<B>>;
            fn begin_batch_speculation(&self,cache:&mut <Self as crate::DeviceBatchModel<B>>::Cache,batch:&crate::ScheduledBatch)
                ->Result<(),crate::GenerationError> {
                if cache.devices!=self.devices || cache.ranks.len()!=self.ranks.len() {return Err(crate::GenerationError("parallel speculative topology differs".into()));}
                for (model,state) in self.ranks.iter().zip(&mut cache.ranks) {model.begin_batch_speculation(state,batch)?;}
                Ok(())
            }
            fn forward_batch_block(&self,batch:&crate::ScheduledBatch,cache:&mut <Self as crate::DeviceBatchModel<B>>::Cache,
                device:&B::Device)->Result<ruda_tensor::api::Tensor<B,3>,crate::GenerationError> {
                if cache.devices!=self.devices || cache.ranks.len()!=self.ranks.len() || self.devices.first()!=Some(device) {
                    return Err(crate::GenerationError("parallel speculative topology/root differs".into()));
                }
                std::thread::scope(|scope| {
                    let mut handles=Vec::with_capacity(self.ranks.len());
                    for ((model,state),device) in self.ranks.iter().zip(&mut cache.ranks).zip(&self.devices) {
                        handles.push(scope.spawn(move || {
                            let output=model.forward_batch_block(batch,state,device);
                            model.sync_batch(device)?;output
                        }));
                    }
                    let mut root=None;let mut failure=None;
                    for (rank,handle) in handles.into_iter().enumerate() {
                        match handle.join() {
                            Ok(Ok(output)) if rank==0=>root=Some(output),Ok(Ok(_))=>(),
                            Ok(Err(error))=>failure=Some(error),Err(_)=>failure=Some(crate::GenerationError("parallel speculative rank panicked".into())),
                        }
                    }
                    if let Some(error)=failure {Err(error)} else {root.ok_or_else(||crate::GenerationError("parallel speculative root is absent".into()))}
                })
            }
            fn capture_batch_delta(&self,cache:&<Self as crate::DeviceBatchModel<B>>::Cache,id:crate::RequestId,start:usize,end:usize)
                ->Result<Self::Delta,crate::GenerationError> {
                if cache.devices!=self.devices || cache.ranks.len()!=self.ranks.len() {return Err(crate::GenerationError("parallel delta topology differs".into()));}
                let ranks=self.ranks.iter().zip(&cache.ranks).map(|(m,c)|m.capture_batch_delta(c,id,start,end)).collect::<Result<Vec<_>,_>>()?;
                Ok(Self::Delta{ranks,devices:self.devices.clone()})
            }
            fn truncate_batch_cache(&self,cache:&mut <Self as crate::DeviceBatchModel<B>>::Cache,id:crate::RequestId,position:usize)
                ->Result<(),crate::GenerationError> {
                if cache.devices!=self.devices || cache.ranks.len()!=self.ranks.len() {return Err(crate::GenerationError("parallel truncate topology differs".into()));}
                for (model,state) in self.ranks.iter().zip(&mut cache.ranks) {model.truncate_batch_cache(state,id,position)?;}
                Ok(())
            }
            fn append_batch_delta(&self,cache:&mut <Self as crate::DeviceBatchModel<B>>::Cache,id:crate::RequestId,table:&[crate::KvPageId],delta:&Self::Delta,offset:usize)
                ->Result<(),crate::GenerationError> {
                if cache.devices!=self.devices || cache.ranks.len()!=self.ranks.len() || delta.devices!=self.devices || delta.ranks.len()!=self.ranks.len() {
                    return Err(crate::GenerationError("parallel append delta topology differs".into()));
                }
                for ((model,state),delta) in self.ranks.iter().zip(&mut cache.ranks).zip(&delta.ranks) {model.append_batch_delta(state,id,table,delta,offset)?;}
                Ok(())
            }
        }
    }
}
pub(crate) use parallel_speculative_model;
