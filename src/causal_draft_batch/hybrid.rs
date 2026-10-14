use super::*;
use crate::{DenseBatchCache,DenseBatchDelta,DenseBatchPrefix,Qwen35TextModel,Qwen35BatchCache,Qwen35BatchDelta,Qwen35Cache,
    KimiLinearModel,KimiLinearBatchCache,KimiLinearBatchPrefix,KimiLinearBatchDelta};
use ruda::runtime::server::ComputeServer;
use ruda_tensor::DeviceOps;
use ruda_tensor_device::{BoolElement,DeviceBackend,DeviceRuntime,FloatElement,IntElement};

pub enum CausalBatchModel<B:Backend> {Dense(DenseCausalBatchModel<B>),Qwen35(Qwen35TextModel<B>),KimiLinear(KimiLinearModel<B>)}
pub enum CausalBatchCache<B:Backend> {Dense(DenseBatchCache<B>),Qwen35(Qwen35BatchCache<B>),KimiLinear(KimiLinearBatchCache<B>)}
#[derive(Clone)]
pub enum CausalBatchPrefix<B:Backend> {Dense(DenseBatchPrefix<B>),Qwen35(Qwen35Cache<B>),KimiLinear(KimiLinearBatchPrefix<B>)}
#[derive(Clone)]
pub enum CausalBatchDelta<B:Backend> {Dense(DenseBatchDelta<B>),Qwen35(Qwen35BatchDelta<B>),KimiLinear(KimiLinearBatchDelta<B>)}
impl<B:Backend> From<DenseCausalBatchModel<B>> for CausalBatchModel<B> {
    fn from(model:DenseCausalBatchModel<B>)->Self {Self::Dense(model)}
}

macro_rules! cache {
    ($model:expr,$state:expr,$m:ident,$c:ident,$body:expr)=>{match ($model,$state) {
        (CausalBatchModel::Dense($m),CausalBatchCache::Dense($c))=>$body,
        (CausalBatchModel::Qwen35($m),CausalBatchCache::Qwen35($c))=>$body,
        (CausalBatchModel::KimiLinear($m),CausalBatchCache::KimiLinear($c))=>$body,
        _=>Err(error("causal model and cache variants differ")),
    }};
}

impl<R,F,I,BT> DeviceBatchModel<DeviceBackend<R,F,I,BT>> for CausalBatchModel<DeviceBackend<R,F,I,BT>>
where R:DeviceRuntime,R::Server:ComputeServer,R::Device:DeviceOps,F:FloatElement,I:IntElement,BT:BoolElement {
    type Cache=CausalBatchCache<DeviceBackend<R,F,I,BT>>;
    fn batch_limits(&self)->CausalModelLimits {match self {Self::Dense(m)=>m.batch_limits(),Self::Qwen35(m)=>m.batch_limits(),Self::KimiLinear(m)=>m.batch_limits()}}
    fn new_batch_cache(&self,config:PagedKvCacheConfig)->Self::Cache {match self {
        Self::Dense(m)=>CausalBatchCache::Dense(m.new_batch_cache(config)),Self::Qwen35(m)=>CausalBatchCache::Qwen35(m.new_batch_cache(config)),
        Self::KimiLinear(m)=>CausalBatchCache::KimiLinear(m.new_batch_cache(config)),
    }}
    fn fork_batch_cache(&self,state:&Self::Cache)->Self::Cache {match (self,state) {
        (Self::Dense(m),CausalBatchCache::Dense(c))=>CausalBatchCache::Dense(m.fork_batch_cache(c)),
        (Self::Qwen35(m),CausalBatchCache::Qwen35(c))=>CausalBatchCache::Qwen35(m.fork_batch_cache(c)),
        (Self::KimiLinear(m),CausalBatchCache::KimiLinear(c))=>CausalBatchCache::KimiLinear(m.fork_batch_cache(c)),
        _=>panic!("causal model and cache variants differ"),
    }}
    fn retain_batch_cache(&self,state:&mut Self::Cache,scheduler:&ContinuousBatchScheduler) {match (self,state) {
        (Self::Dense(m),CausalBatchCache::Dense(c))=>m.retain_batch_cache(c,scheduler),
        (Self::Qwen35(m),CausalBatchCache::Qwen35(c))=>m.retain_batch_cache(c,scheduler),
        (Self::KimiLinear(m),CausalBatchCache::KimiLinear(c))=>m.retain_batch_cache(c,scheduler),
        _=>panic!("causal model and cache variants differ"),
    }}
    fn forward_batch(&self,batch:&ScheduledBatch,state:&mut Self::Cache,device:&R::Device)
        ->Result<Tensor<DeviceBackend<R,F,I,BT>,3>,GenerationError> {cache!(self,state,m,c,m.forward_batch(batch,c,device))}
    fn sync_batch(&self,device:&R::Device)->Result<(),GenerationError> {match self {Self::Dense(m)=>m.sync_batch(device),Self::Qwen35(m)=>m.sync_batch(device),Self::KimiLinear(m)=>m.sync_batch(device)}}
}

impl<R,F,I,BT> PrefixBatchModel<DeviceBackend<R,F,I,BT>> for CausalBatchModel<DeviceBackend<R,F,I,BT>>
where R:DeviceRuntime,R::Server:ComputeServer,R::Device:DeviceOps,F:FloatElement,I:IntElement,BT:BoolElement {
    type Prefix=CausalBatchPrefix<DeviceBackend<R,F,I,BT>>;
    fn prefix_tensor_transfer(&self)->bool {match self {Self::Dense(m)=>m.prefix_tensor_transfer(),Self::Qwen35(m)=>m.prefix_tensor_transfer(),Self::KimiLinear(m)=>m.prefix_tensor_transfer()}}
    fn export_prefix_tensors(&self,prefix:&Self::Prefix,shard:crate::PrefixShard)
        ->Result<crate::PrefixTensors<DeviceBackend<R,F,I,BT>>,GenerationError> {match (self,prefix) {
        (Self::Dense(m),CausalBatchPrefix::Dense(p))=>m.export_prefix_tensors(p,shard),
        (Self::Qwen35(m),CausalBatchPrefix::Qwen35(p))=>m.export_prefix_tensors(p,shard),
        (Self::KimiLinear(m),CausalBatchPrefix::KimiLinear(p))=>m.export_prefix_tensors(p,shard),
        _=>Err(error("causal prefix tensor variants differ")),
    }}
    fn import_prefix_tensors(&self,tensors:&crate::PrefixTensors<DeviceBackend<R,F,I,BT>>,shard:crate::PrefixShard,device:&R::Device)
        ->Result<Self::Prefix,GenerationError> {match self {
        Self::Dense(m)=>m.import_prefix_tensors(tensors,shard,device).map(CausalBatchPrefix::Dense),
        Self::Qwen35(m)=>m.import_prefix_tensors(tensors,shard,device).map(CausalBatchPrefix::Qwen35),
        Self::KimiLinear(m)=>m.import_prefix_tensors(tensors,shard,device).map(CausalBatchPrefix::KimiLinear),
    }}
    fn capture_prefix(&self,state:&Self::Cache,id:RequestId)->Result<Self::Prefix,GenerationError> {match (self,state) {
        (Self::Dense(m),CausalBatchCache::Dense(c))=>m.capture_prefix(c,id).map(CausalBatchPrefix::Dense),
        (Self::Qwen35(m),CausalBatchCache::Qwen35(c))=>m.capture_prefix(c,id).map(CausalBatchPrefix::Qwen35),
        (Self::KimiLinear(m),CausalBatchCache::KimiLinear(c))=>m.capture_prefix(c,id).map(CausalBatchPrefix::KimiLinear),
        _=>Err(error("causal model and prefix source differ")),
    }}
    fn prefix_position(&self,prefix:&Self::Prefix)->usize {match (self,prefix) {
        (Self::Dense(m),CausalBatchPrefix::Dense(p))=>m.prefix_position(p),
        (Self::Qwen35(m),CausalBatchPrefix::Qwen35(p))=>m.prefix_position(p),
        (Self::KimiLinear(m),CausalBatchPrefix::KimiLinear(p))=>m.prefix_position(p),
        _=>panic!("causal model and prefix variants differ"),
    }}
    fn prefix_bytes(&self,prefix:&Self::Prefix)->usize {match (self,prefix) {
        (Self::Dense(m),CausalBatchPrefix::Dense(p))=>m.prefix_bytes(p),
        (Self::Qwen35(m),CausalBatchPrefix::Qwen35(p))=>m.prefix_bytes(p),
        (Self::KimiLinear(m),CausalBatchPrefix::KimiLinear(p))=>m.prefix_bytes(p),
        _=>panic!("causal model and prefix variants differ"),
    }}
    fn restore_prefix(&self,state:&mut Self::Cache,id:RequestId,table:&[KvPageId],prefix:&Self::Prefix)->Result<(),GenerationError> {
        match (self,state,prefix) {
            (Self::Dense(m),CausalBatchCache::Dense(c),CausalBatchPrefix::Dense(p))=>m.restore_prefix(c,id,table,p),
            (Self::Qwen35(m),CausalBatchCache::Qwen35(c),CausalBatchPrefix::Qwen35(p))=>m.restore_prefix(c,id,table,p),
            (Self::KimiLinear(m),CausalBatchCache::KimiLinear(c),CausalBatchPrefix::KimiLinear(p))=>m.restore_prefix(c,id,table,p),
            _=>Err(error("causal prefix destination variants differ")),
        }
    }
    fn encode_prefix(&self,prefix:&Self::Prefix,model_id:&str)->Result<Vec<u8>,GenerationError> {
        let record=match (self,prefix) {
            (Self::Dense(m),CausalBatchPrefix::Dense(p))=>(0u8,m.encode_prefix(p,model_id)?),
            (Self::Qwen35(m),CausalBatchPrefix::Qwen35(p))=>(1u8,m.encode_prefix(p,model_id)?),
            (Self::KimiLinear(m),CausalBatchPrefix::KimiLinear(p))=>(2u8,m.encode_prefix(p,model_id)?),
            _=>return Err(error("causal prefix serialization variants differ")),
        };
        rmp_serde::to_vec(&record).map_err(error)
    }
    fn decode_prefix(&self,bytes:&[u8],model_id:&str,device:&R::Device)->Result<Self::Prefix,GenerationError> {
        let (kind,record):(u8,Vec<u8>)=rmp_serde::from_slice(bytes).map_err(error)?;
        match (self,kind) {
            (Self::Dense(m),0)=>m.decode_prefix(&record,model_id,device).map(CausalBatchPrefix::Dense),
            (Self::Qwen35(m),1)=>m.decode_prefix(&record,model_id,device).map(CausalBatchPrefix::Qwen35),
            (Self::KimiLinear(m),2)=>m.decode_prefix(&record,model_id,device).map(CausalBatchPrefix::KimiLinear),
            _=>Err(error("causal prefix record and model variants differ")),
        }
    }
}

impl<R,F,I,BT> SpeculativeBatchModel<DeviceBackend<R,F,I,BT>> for CausalBatchModel<DeviceBackend<R,F,I,BT>>
where R:DeviceRuntime,R::Server:ComputeServer,R::Device:DeviceOps,F:FloatElement,I:IntElement,BT:BoolElement {
    type Delta=CausalBatchDelta<DeviceBackend<R,F,I,BT>>;
    fn begin_batch_speculation(&self,state:&mut Self::Cache,batch:&ScheduledBatch)->Result<(),GenerationError> {
        cache!(self,state,m,c,m.begin_batch_speculation(c,batch))
    }
    fn forward_batch_block(&self,batch:&ScheduledBatch,state:&mut Self::Cache,device:&R::Device)
        ->Result<Tensor<DeviceBackend<R,F,I,BT>,3>,GenerationError> {cache!(self,state,m,c,m.forward_batch_block(batch,c,device))}
    fn capture_batch_delta(&self,state:&Self::Cache,id:RequestId,start:usize,end:usize)->Result<Self::Delta,GenerationError> {match (self,state) {
        (Self::Dense(m),CausalBatchCache::Dense(c))=>m.capture_batch_delta(c,id,start,end).map(CausalBatchDelta::Dense),
        (Self::Qwen35(m),CausalBatchCache::Qwen35(c))=>m.capture_batch_delta(c,id,start,end).map(CausalBatchDelta::Qwen35),
        (Self::KimiLinear(m),CausalBatchCache::KimiLinear(c))=>m.capture_batch_delta(c,id,start,end).map(CausalBatchDelta::KimiLinear),
        _=>Err(error("causal delta source variants differ")),
    }}
    fn truncate_batch_cache(&self,state:&mut Self::Cache,id:RequestId,position:usize)->Result<(),GenerationError> {
        cache!(self,state,m,c,m.truncate_batch_cache(c,id,position))
    }
    fn append_batch_delta(&self,state:&mut Self::Cache,id:RequestId,table:&[KvPageId],delta:&Self::Delta,offset:usize)->Result<(),GenerationError> {
        match (self,state,delta) {
            (Self::Dense(m),CausalBatchCache::Dense(c),CausalBatchDelta::Dense(d))=>m.append_batch_delta(c,id,table,d,offset),
            (Self::Qwen35(m),CausalBatchCache::Qwen35(c),CausalBatchDelta::Qwen35(d))=>m.append_batch_delta(c,id,table,d,offset),
            (Self::KimiLinear(m),CausalBatchCache::KimiLinear(c),CausalBatchDelta::KimiLinear(d))=>m.append_batch_delta(c,id,table,d,offset),
            _=>Err(error("causal delta destination variants differ")),
        }
    }
}
