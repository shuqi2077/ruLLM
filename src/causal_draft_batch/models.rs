use crate::{CausalModelLimits, ContinuousBatchScheduler, DenseBatchCache, DenseBatchDelta, DenseBatchPrefix,
    DeepseekV2Model, DeviceBatchModel, GenerationError, KvPageId, LlamaForCausalLm, PagedKvCacheConfig,
    PrefixBatchModel, Qwen3Model, RequestId, ScheduledBatch, SpeculativeBatchModel};
use ruda::runtime::server::ComputeServer;
use ruda_tensor::{DeviceOps, api::{backend::Backend, Tensor}};
use ruda_tensor_device::{BoolElement, DeviceBackend, DeviceRuntime, FloatElement, IntElement};

pub enum DenseCausalBatchModel<B: Backend> {
    Llama(LlamaForCausalLm<B>),
    Qwen3(Qwen3Model<B>),
    DeepseekV2(DeepseekV2Model<B>),
}

macro_rules! dispatch {
    ($value:expr,$model:ident,$body:expr)=>{match $value {
        DenseCausalBatchModel::Llama($model)=>$body,
        DenseCausalBatchModel::Qwen3($model)=>$body,
        DenseCausalBatchModel::DeepseekV2($model)=>$body,
    }};
}

impl<R,F,I,BT> DeviceBatchModel<DeviceBackend<R,F,I,BT>> for DenseCausalBatchModel<DeviceBackend<R,F,I,BT>>
where R:DeviceRuntime,R::Server:ComputeServer,R::Device:DeviceOps,F:FloatElement,I:IntElement,BT:BoolElement {
    type Cache=DenseBatchCache<DeviceBackend<R,F,I,BT>>;
    fn batch_limits(&self)->CausalModelLimits {dispatch!(self,m,m.batch_limits())}
    fn new_batch_cache(&self,config:PagedKvCacheConfig)->Self::Cache {dispatch!(self,m,m.new_batch_cache(config))}
    fn fork_batch_cache(&self,cache:&Self::Cache)->Self::Cache {dispatch!(self,m,m.fork_batch_cache(cache))}
    fn retain_batch_cache(&self,cache:&mut Self::Cache,scheduler:&ContinuousBatchScheduler) {dispatch!(self,m,m.retain_batch_cache(cache,scheduler))}
    fn sync_batch(&self,device:&R::Device)->Result<(),GenerationError> {dispatch!(self,m,m.sync_batch(device))}
    fn forward_batch(&self,batch:&ScheduledBatch,cache:&mut Self::Cache,device:&R::Device)
        ->Result<Tensor<DeviceBackend<R,F,I,BT>,3>,GenerationError> {dispatch!(self,m,m.forward_batch(batch,cache,device))}
}

impl<R,F,I,BT> PrefixBatchModel<DeviceBackend<R,F,I,BT>> for DenseCausalBatchModel<DeviceBackend<R,F,I,BT>>
where R:DeviceRuntime,R::Server:ComputeServer,R::Device:DeviceOps,F:FloatElement,I:IntElement,BT:BoolElement {
    type Prefix=DenseBatchPrefix<DeviceBackend<R,F,I,BT>>;
    fn prefix_tensor_transfer(&self)->bool {dispatch!(self,m,m.prefix_tensor_transfer())}
    fn export_prefix_tensors(&self,prefix:&Self::Prefix,shard:crate::PrefixShard)
        ->Result<crate::PrefixTensors<DeviceBackend<R,F,I,BT>>,GenerationError> {dispatch!(self,m,m.export_prefix_tensors(prefix,shard))}
    fn import_prefix_tensors(&self,tensors:&crate::PrefixTensors<DeviceBackend<R,F,I,BT>>,shard:crate::PrefixShard,device:&R::Device)
        ->Result<Self::Prefix,GenerationError> {dispatch!(self,m,m.import_prefix_tensors(tensors,shard,device))}
    fn capture_prefix(&self,cache:&Self::Cache,id:RequestId)->Result<Self::Prefix,GenerationError> {dispatch!(self,m,m.capture_prefix(cache,id))}
    fn prefix_position(&self,prefix:&Self::Prefix)->usize {prefix.position()}
    fn prefix_bytes(&self,prefix:&Self::Prefix)->usize {prefix.bytes()}
    fn restore_prefix(&self,cache:&mut Self::Cache,id:RequestId,table:&[KvPageId],prefix:&Self::Prefix)->Result<(),GenerationError> {
        dispatch!(self,m,m.restore_prefix(cache,id,table,prefix))
    }
    fn encode_prefix(&self,prefix:&Self::Prefix,model_id:&str)->Result<Vec<u8>,GenerationError> {dispatch!(self,m,m.encode_prefix(prefix,model_id))}
    fn decode_prefix(&self,bytes:&[u8],model_id:&str,device:&R::Device)->Result<Self::Prefix,GenerationError> {
        dispatch!(self,m,m.decode_prefix(bytes,model_id,device))
    }
}

impl<R,F,I,BT> SpeculativeBatchModel<DeviceBackend<R,F,I,BT>> for DenseCausalBatchModel<DeviceBackend<R,F,I,BT>>
where R:DeviceRuntime,R::Server:ComputeServer,R::Device:DeviceOps,F:FloatElement,I:IntElement,BT:BoolElement {
    type Delta=DenseBatchDelta<DeviceBackend<R,F,I,BT>>;
    fn forward_batch_block(&self,batch:&ScheduledBatch,cache:&mut Self::Cache,device:&R::Device)
        ->Result<Tensor<DeviceBackend<R,F,I,BT>,3>,GenerationError> {dispatch!(self,m,m.forward_batch_block(batch,cache,device))}
    fn capture_batch_delta(&self,cache:&Self::Cache,id:RequestId,start:usize,end:usize)->Result<Self::Delta,GenerationError> {
        cache.capture_delta(id,start,end)
    }
    fn truncate_batch_cache(&self,cache:&mut Self::Cache,id:RequestId,position:usize)->Result<(),GenerationError> {cache.truncate(id,position)}
    fn append_batch_delta(&self,cache:&mut Self::Cache,id:RequestId,table:&[KvPageId],delta:&Self::Delta,offset:usize)
        ->Result<(),GenerationError> {cache.append_delta(id,table,delta,offset)}
}
