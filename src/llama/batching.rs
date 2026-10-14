use super::*;
use crate::{CausalModelLimits,ContinuousBatchScheduler,DenseBatchCache,DeviceBatchModel,GenerationError,PagedKvCacheConfig,ScheduledBatch};

crate::batch_kv::dense_prefix_model!(LlamaForCausalLm,|model|model.layers.iter().map(|layer| {
    let a=&layer.self_attn;(a.num_kv_heads,a.head_dimension,a.head_dimension,model.embed_tokens.weight.val().dtype())
}).collect::<Vec<_>>());
crate::batch_kv::dense_speculative_model!(LlamaForCausalLm);

impl<R,F,I,BT> DeviceBatchModel<DeviceBackend<R,F,I,BT>> for LlamaForCausalLm<DeviceBackend<R,F,I,BT>>
where R:DeviceRuntime,R::Server:ComputeServer,R::Device:DeviceOps,F:FloatElement,I:IntElement,BT:BoolElement {
    type Cache=DenseBatchCache<DeviceBackend<R,F,I,BT>>;
    fn batch_limits(&self)->CausalModelLimits {
        CausalModelLimits{vocab_size:self.embed_tokens.weight.val().dims()[0],max_sequence_length:self.max_sequence_length}
    }
    fn new_batch_cache(&self,config:PagedKvCacheConfig)->Self::Cache {DenseBatchCache::new(config,self.layers.len())}
    fn fork_batch_cache(&self,cache:&Self::Cache)->Self::Cache {cache.clone()}
    fn retain_batch_cache(&self,cache:&mut Self::Cache,scheduler:&ContinuousBatchScheduler) {cache.retain(scheduler);}
    fn forward_batch(&self,batch:&ScheduledBatch,cache:&mut Self::Cache,device:&R::Device)
        ->Result<Tensor<DeviceBackend<R,F,I,BT>,3>,GenerationError> {
        self.forward_batch_logits(batch,cache,device,true)
    }
}

impl<R,F,I,BT> LlamaForCausalLm<DeviceBackend<R,F,I,BT>>
where R:DeviceRuntime,R::Server:ComputeServer,R::Device:DeviceOps,F:FloatElement,I:IntElement,BT:BoolElement {
    fn forward_batch_logits(&self,batch:&ScheduledBatch,cache:&mut DenseBatchCache<DeviceBackend<R,F,I,BT>>,device:&R::Device,last_only:bool)
        ->Result<Tensor<DeviceBackend<R,F,I,BT>,3>,GenerationError> {
        self.forward_batch_logits_with_reduction(batch,cache,device,last_only,None)
    }
    pub(super) fn forward_batch_logits_with_reduction(&self,batch:&ScheduledBatch,cache:&mut DenseBatchCache<DeviceBackend<R,F,I,BT>>,
        device:&R::Device,last_only:bool,
        reduction:Option<&(dyn Fn(Tensor<DeviceBackend<R,F,I,BT>,3>)->Tensor<DeviceBackend<R,F,I,BT>,3>+Send+Sync)>)
        ->Result<Tensor<DeviceBackend<R,F,I,BT>,3>,GenerationError> {
        cache.validate(batch,self.batch_limits(),self.layers.len())?;
        let (b,s)=(batch.batch_size(),batch.next_n());
        let tokens=Tensor::from_data(TensorData::new(batch.token_matrix(),[b,s]),device);
        let mut hidden=self.embed_tokens.forward(tokens);
        for (index,layer) in self.layers.iter().enumerate() {
            let attn=&layer.self_attn;let (h,kh,d)=(attn.num_query_heads,attn.num_kv_heads,attn.head_dimension);
            let input=layer.input_layernorm.forward(hidden.clone());
            let q=attn.q_proj.forward(input.clone()).reshape([b,s,h,d]).swap_dims(1,2);
            let k=attn.k_proj.forward(input.clone()).reshape([b,s,kh,d]).swap_dims(1,2);
            let v=attn.v_proj.forward(input).reshape([b,s,kh,d]).swap_dims(1,2);
            let mut queries=Vec::with_capacity(b);let mut keys=Vec::with_capacity(b);
            let mut rotary_updates=Vec::with_capacity(b);
            for (row,sequence) in batch.sequences.iter().enumerate() {
                let query=q.clone().slice([row..row+1,0..h,0..s,0..d]);
                let key=k.clone().slice([row..row+1,0..kh,0..s,0..d]);
                match &attn.rotary_state {
                    Some(template)=>{
                        let state=cache.rotary_state(sequence.request_id,index,sequence.start_position,template)?;
                        queries.push(state.apply(query,sequence.start_position,sequence.context_length));
                        keys.push(state.apply(key,sequence.start_position,sequence.context_length));
                        rotary_updates.push((sequence.request_id,sequence.context_length,state));
                    }
                    None=>{queries.push(attn.apply_rope(query,sequence.start_position));keys.push(attn.apply_rope(key,sequence.start_position));}
                }
            }
            let context=cache.attention(index,batch,Tensor::cat(queries,0),Tensor::cat(keys,0),v,(d as f32).sqrt().recip())?;
            for (id,end,state) in rotary_updates {cache.commit_rotary(id,index,end,&state)?;}
            let attention=attn.finish(context);
            let attention=match reduction {Some(reduce)=>reduce(attention),None=>attention};
            hidden=hidden+attention;
            let mlp=layer.mlp.forward(layer.post_attention_layernorm.forward(hidden.clone()));
            let mlp=match reduction {Some(reduce)=>reduce(mlp),None=>mlp};
            hidden=hidden+mlp;
        }
        let width=hidden.dims()[2];
        if last_only {hidden=hidden.slice([0..b,s-1..s,0..width]);}
        let logits=self.lm_head.forward(self.norm.forward(hidden));
        cache.commit(batch);
        Ok(logits)
    }
}
