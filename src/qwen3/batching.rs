use super::*;
use crate::{ContinuousBatchScheduler,DenseBatchCache,DeviceBatchModel,PagedKvCacheConfig,ScheduledBatch};
use ruda::runtime::server::ComputeServer;
use ruda_tensor::DeviceOps;
use ruda_tensor_device::{BoolElement,DeviceBackend,DeviceRuntime,FloatElement,IntElement};

crate::batch_kv::dense_prefix_model!(Qwen3Model,|model|vec![(model.config.num_key_value_heads,model.config.head_dim,
    model.config.head_dim,model.embedding.dtype());model.layers.len()]);
crate::batch_kv::dense_speculative_model!(Qwen3Model);

impl<R,F,I,BT> DeviceBatchModel<DeviceBackend<R,F,I,BT>> for Qwen3Model<DeviceBackend<R,F,I,BT>>
where R:DeviceRuntime,R::Server:ComputeServer,R::Device:DeviceOps,F:FloatElement,I:IntElement,BT:BoolElement {
    type Cache=DenseBatchCache<DeviceBackend<R,F,I,BT>>;
    fn batch_limits(&self)->CausalModelLimits {
        CausalModelLimits{vocab_size:self.config.vocab_size,max_sequence_length:self.config.max_position_embeddings}
    }
    fn new_batch_cache(&self,config:PagedKvCacheConfig)->Self::Cache {DenseBatchCache::new(config,self.layers.len())}
    fn fork_batch_cache(&self,cache:&Self::Cache)->Self::Cache {cache.clone()}
    fn retain_batch_cache(&self,cache:&mut Self::Cache,scheduler:&ContinuousBatchScheduler) {cache.retain(scheduler);}
    fn forward_batch(&self,batch:&ScheduledBatch,cache:&mut Self::Cache,device:&R::Device)
        ->Result<Tensor<DeviceBackend<R,F,I,BT>,3>,GenerationError> {
        self.forward_batch_features(batch,cache,device,true,&[]).map(|(logits,_)|logits)
    }
}

impl<R,F,I,BT> Qwen3Model<DeviceBackend<R,F,I,BT>>
where R:DeviceRuntime,R::Server:ComputeServer,R::Device:DeviceOps,F:FloatElement,I:IntElement,BT:BoolElement {
    fn forward_batch_logits(&self,batch:&ScheduledBatch,cache:&mut DenseBatchCache<DeviceBackend<R,F,I,BT>>,device:&R::Device,last_only:bool)
        ->Result<Tensor<DeviceBackend<R,F,I,BT>,3>,GenerationError> {
        self.forward_batch_features(batch,cache,device,last_only,&[]).map(|(logits,_)|logits)
    }
    pub(super) fn forward_batch_features(&self,batch:&ScheduledBatch,cache:&mut DenseBatchCache<DeviceBackend<R,F,I,BT>>,device:&R::Device,
        last_only:bool,taps:&[usize])->Result<(Tensor<DeviceBackend<R,F,I,BT>,3>,Option<Tensor<DeviceBackend<R,F,I,BT>,3>>),GenerationError> {
        cache.validate(batch,self.batch_limits(),self.layers.len())?;
        if taps.iter().any(|&i|i>=self.layers.len()) || taps.windows(2).any(|w|w[0]>=w[1]) {return Err(GenerationError("invalid batched target feature taps".into()));}
        let c=&self.config;let (b,s)=(batch.batch_size(),batch.next_n());
        let ids=Tensor::<DeviceBackend<R,F,I,BT>,1,Int>::from_data(TensorData::new(batch.token_matrix(),[b*s]),device);
        let mut hidden=self.embedding.clone().select(0,ids).reshape([b,s,c.hidden_size]);
        let dtype=hidden.dtype();
        let mut features=Vec::with_capacity(taps.len());
        for (index,layer) in self.layers.iter().enumerate() {
            let mut cos_rows=Vec::with_capacity(b);let mut sin_rows=Vec::with_capacity(b);let mut rotary_updates=Vec::with_capacity(b);
            for row in &batch.sequences {
                let state=cache.rotary_state(row.request_id,index,row.start_position,&self.rotary[index])?;
                let positions=(row.start_position..row.context_length).map(|p|p as f32).collect();
                let (cos,sin)=state.cos_sin(positions,1,s,row.context_length,dtype,device);
                cos_rows.push(cos);sin_rows.push(sin);rotary_updates.push((row.request_id,row.context_length,state));
            }
            let (cos,sin)=(Tensor::cat(cos_rows,0),Tensor::cat(sin_rows,0));
            let input=norm(hidden.clone(),&layer.input_norm,c.rms_norm_eps);
            let q=norm(layer.q.forward(input.clone()).reshape([b,s,c.num_attention_heads,c.head_dim]),&layer.q_norm,c.rms_norm_eps).swap_dims(1,2);
            let k=norm(layer.k.forward(input.clone()).reshape([b,s,c.num_key_value_heads,c.head_dim]),&layer.k_norm,c.rms_norm_eps).swap_dims(1,2);
            let v=layer.v.forward(input).reshape([b,s,c.num_key_value_heads,c.head_dim]).swap_dims(1,2);
            let q=rotate(q,cos.clone(),sin.clone());let k=rotate(k,cos.clone(),sin.clone());
            let context=cache.attention(index,batch,q,k,v,(c.head_dim as f32).sqrt().recip())?
                .swap_dims(1,2).reshape([b,s,c.num_attention_heads*c.head_dim]);
            for (id,end,state) in rotary_updates {cache.commit_rotary(id,index,end,&state)?;}
            let update=layer.o.forward(context);
            let update=match &self.reduction {Some(reduce)=>reduce(update),None=>update};
            hidden=hidden+update;
            let input=norm(hidden.clone(),&layer.post_norm,c.rms_norm_eps);
            let mlp=layer.mlp.forward(input)?;
            let mlp=match &self.reduction {Some(reduce)=>reduce(mlp),None=>mlp};
            hidden=hidden+mlp;
            if taps.contains(&index) {features.push(hidden.clone());}
        }
        if last_only {hidden=hidden.slice([0..b,s-1..s,0..c.hidden_size]);}
        let logits=self.head.forward(norm(hidden,&self.norm,c.rms_norm_eps));
        cache.commit(batch);
        Ok((logits,if features.is_empty() {None} else {Some(Tensor::cat(features,2))}))
    }
}
