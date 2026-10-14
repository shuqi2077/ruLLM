use super::*;
use crate::{ContinuousBatchScheduler,DenseBatchCache,DeviceBatchModel,PagedKvCacheConfig,ScheduledBatch};
use ruda::runtime::server::ComputeServer;
use ruda_tensor::DeviceOps;
use ruda_tensor_device::{BoolElement,DeviceBackend,DeviceRuntime,FloatElement,IntElement};

crate::batch_kv::dense_prefix_model!(DeepseekV2Model,|model|{
    let c=&model.config;let layout=match model.attention_mode {
        DeepseekV2AttentionMode::Expanded=>(c.num_attention_heads,c.qk_nope_head_dim+c.qk_rope_head_dim,c.v_head_dim,model.embedding.dtype()),
        DeepseekV2AttentionMode::Absorbed=>(1,c.kv_lora_rank+c.qk_rope_head_dim,c.kv_lora_rank,model.embedding.dtype()),
    };vec![layout;model.layers.len()]
});
crate::batch_kv::dense_speculative_model!(DeepseekV2Model);

impl<R,F,I,BT> DeviceBatchModel<DeviceBackend<R,F,I,BT>> for DeepseekV2Model<DeviceBackend<R,F,I,BT>>
where R:DeviceRuntime,R::Server:ComputeServer,R::Device:DeviceOps,F:FloatElement,I:IntElement,BT:BoolElement {
    type Cache=DenseBatchCache<DeviceBackend<R,F,I,BT>>;
    fn batch_limits(&self)->CausalModelLimits {self.config.limits()}
    fn new_batch_cache(&self,config:PagedKvCacheConfig)->Self::Cache {DenseBatchCache::new(config,self.layers.len())}
    fn fork_batch_cache(&self,cache:&Self::Cache)->Self::Cache {cache.clone()}
    fn retain_batch_cache(&self,cache:&mut Self::Cache,scheduler:&ContinuousBatchScheduler) {cache.retain(scheduler);}
    fn forward_batch(&self,batch:&ScheduledBatch,cache:&mut Self::Cache,device:&R::Device)
        ->Result<Tensor<DeviceBackend<R,F,I,BT>,3>,GenerationError> {
        self.forward_batch_logits(batch,cache,device,true)
    }
}

impl<R,F,I,BT> DeepseekV2Model<DeviceBackend<R,F,I,BT>>
where R:DeviceRuntime,R::Server:ComputeServer,R::Device:DeviceOps,F:FloatElement,I:IntElement,BT:BoolElement {
    fn forward_batch_logits(&self,batch:&ScheduledBatch,cache:&mut DenseBatchCache<DeviceBackend<R,F,I,BT>>,device:&R::Device,last_only:bool)
        ->Result<Tensor<DeviceBackend<R,F,I,BT>,3>,GenerationError> {
        self.forward_batch_features(batch,cache,device,last_only).map(|(logits,_)|logits)
    }
    pub(super) fn forward_batch_features(&self,batch:&ScheduledBatch,cache:&mut DenseBatchCache<DeviceBackend<R,F,I,BT>>,device:&R::Device,last_only:bool)
        ->Result<(Tensor<DeviceBackend<R,F,I,BT>,3>,Tensor<DeviceBackend<R,F,I,BT>,3>),GenerationError> {
        cache.validate(batch,self.batch_limits(),self.layers.len())?;
        let c=&self.config;let (b,s)=(batch.batch_size(),batch.next_n());
        let ids=Tensor::<DeviceBackend<R,F,I,BT>,1,Int>::from_data(TensorData::new(batch.token_matrix(),[b*s]),device);
        let mut hidden=self.embedding.clone().select(0,ids).reshape([b,s,c.hidden_size]);
        let dtype=hidden.dtype();let mut cosines=Vec::with_capacity(b);let mut sines=Vec::with_capacity(b);let mut scale=0.0;
        for row in &batch.sequences {
            let (cos,sin,coefficient)=self.rotary(row.start_position,s,dtype,device);
            cosines.push(cos);sines.push(sin);scale=coefficient;
        }
        let cos=Tensor::cat(cosines,0);let sin=Tensor::cat(sines,0);
        for (index,layer) in self.layers.iter().enumerate() {
            hidden=self.layer_batch_forward(layer,index,hidden,batch,cache,cos.clone(),sin.clone(),scale)?;
        }
        let hidden=norm(hidden,&self.norm,c.rms_norm_eps);
        let selected=if last_only {hidden.clone().slice([0..b,s-1..s,0..c.hidden_size])} else {hidden.clone()};
        let logits=self.head.forward(selected);
        cache.commit(batch);Ok((logits,hidden))
    }
    pub(super) fn layer_batch_forward(&self,layer:&Layer<DeviceBackend<R,F,I,BT>>,index:usize,
        mut hidden:Tensor<DeviceBackend<R,F,I,BT>,3>,batch:&ScheduledBatch,cache:&mut DenseBatchCache<DeviceBackend<R,F,I,BT>>,
        cos:Tensor<DeviceBackend<R,F,I,BT>,4>,sin:Tensor<DeviceBackend<R,F,I,BT>,4>,scale:f64)
        ->Result<Tensor<DeviceBackend<R,F,I,BT>,3>,GenerationError> {
            let c=&self.config;let (b,s)=(batch.batch_size(),batch.next_n());
            let (h,rank,rd,nd,vd)=(c.num_attention_heads,c.kv_lora_rank,c.qk_rope_head_dim,c.qk_nope_head_dim,c.v_head_dim);
            let input=norm(hidden.clone(),&layer.input_norm,c.rms_norm_eps);
            let q_input=match &layer.q_a {Some((projection,weight))=>norm(projection.forward(input.clone()),weight,1e-6),None=>input.clone()};
            let query=layer.q.forward(q_input).reshape([b,s,h,nd+rd]).swap_dims(1,2);
            let content_query=query.clone().slice([0..b,0..h,0..s,0..nd]);
            let position_query=rotate(query.slice([0..b,0..h,0..s,nd..nd+rd]),cos.clone(),sin.clone());
            let compressed=layer.kv_a.forward(input);
            let latent=norm(compressed.clone().slice([0..b,0..s,0..rank]),&layer.kv_norm,1e-6).reshape([b,1,s,rank]);
            let position=rotate(compressed.slice([0..b,0..s,rank..rank+rd]).reshape([b,1,s,rd]),cos.clone(),sin.clone());
            let context=match self.attention_mode {
                DeepseekV2AttentionMode::Absorbed=>{
                    let weights=layer.kv_b.weight().reshape([1,h,nd+vd,rank]);
                    let key_weights=weights.clone().slice([0..1,0..h,0..nd,0..rank]);
                    let value_weights=weights.slice([0..1,0..h,nd..nd+vd,0..rank]).swap_dims(2,3);
                    let query=Tensor::cat(vec![mla_absorb_query(&ops(),content_query,key_weights),position_query],3);
                    let key=Tensor::cat(vec![latent.clone(),position],3);
                    cache.attention(index,batch,query,key,latent,scale as f32)?.matmul(value_weights)
                },
                DeepseekV2AttentionMode::Expanded=>{
                    let expanded=layer.kv_b.forward(latent.reshape([b,s,rank]))
                        .reshape([b,s,h,nd+vd]).swap_dims(1,2);
                    let key=Tensor::cat(vec![expanded.clone().slice([0..b,0..h,0..s,0..nd]),position.repeat_dim(1,h)],3);
                    let value=expanded.slice([0..b,0..h,0..s,nd..nd+vd]);
                    let query=Tensor::cat(vec![content_query,position_query],3);
                    cache.attention(index,batch,query,key,value,scale as f32)?
                },
            };
            hidden=hidden+layer.o.forward(context.swap_dims(1,2).reshape([b,s,h*vd]));
            let input=norm(hidden.clone(),&layer.post_norm,c.rms_norm_eps).reshape([1,b*s,c.hidden_size]);
            hidden=hidden+self.feed_forward(layer,input)?.reshape([b,s,c.hidden_size]);
            Ok(hidden)
    }
}
