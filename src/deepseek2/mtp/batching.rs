use super::*;
use crate::{ContinuousBatchScheduler,DenseBatchCache,DenseBatchDelta,DenseBatchPrefix,DeviceBatchModel,
    PrefixBatchModel,PagedKvCacheConfig,PagedKvAppendReservation,RequestId,KvPageId,ScheduledBatch,ScheduledBatchKind};
use ruda::runtime::server::ComputeServer;
use ruda_tensor::DeviceOps;
use ruda_tensor_device::{BoolElement,DeviceBackend,DeviceRuntime,FloatElement,IntElement};
use std::collections::BTreeMap;

#[derive(Clone)]
struct BatchVerified<B:Backend> {tokens:Vec<i32>,logits:Tensor<B,3>,hidden:Tensor<B,3>,delta:DenseBatchDelta<B>,next:usize}
#[derive(Clone)]
struct Request<B:Backend> {tails:Vec<Option<Tensor<B,3>>>,verified:Option<BatchVerified<B>>}
#[derive(Clone)]
pub struct DeepseekMtpBatchCache<B:Backend> {
    target:DenseBatchCache<B>,predictors:Vec<DenseBatchCache<B>>,requests:BTreeMap<RequestId,Request<B>>,
}
#[derive(Clone)]
pub struct DeepseekMtpBatchPrefix<B:Backend> {
    target:DenseBatchPrefix<B>,predictors:Vec<Option<DenseBatchPrefix<B>>>,tails:Vec<Option<Tensor<B,3>>>,
}

impl<R,F,I,BT> DeepseekMtp<DeviceBackend<R,F,I,BT>>
where R:DeviceRuntime,R::Server:ComputeServer,R::Device:DeviceOps,F:FloatElement,I:IntElement,BT:BoolElement {
    fn predictor_batch(&self,depth:usize,batch:&ScheduledBatch,previous:Tensor<DeviceBackend<R,F,I,BT>,3>,
        cache:&mut DenseBatchCache<DeviceBackend<R,F,I,BT>>,device:&R::Device)
        ->Result<Tensor<DeviceBackend<R,F,I,BT>,3>,GenerationError> {
        let c=&self.target.config;let (b,s)=(batch.batch_size(),batch.next_n());
        let predictor=self.predictors.get(depth).ok_or_else(||GenerationError("MTP depth differs".into()))?;
        cache.validate(batch,c.limits(),1)?;
        if previous.dims()!=[b,s,c.hidden_size] || previous.dtype()!=self.target.embedding.dtype() || previous.device()!=*device {
            return Err(GenerationError("MTP batched upstream geometry differs".into()));
        }
        let ids=Tensor::<DeviceBackend<R,F,I,BT>,1,Int>::from_data(TensorData::new(batch.token_matrix(),[b*s]),device);
        let embedding=self.target.embedding.clone().select(0,ids).reshape([b,s,c.hidden_size]);
        let input=predictor.projection.forward(Tensor::cat(vec![norm(embedding,&predictor.embedding_norm,c.rms_norm_eps),
            norm(previous,&predictor.hidden_norm,c.rms_norm_eps)],2));
        let mut cos=Vec::with_capacity(b);let mut sin=Vec::with_capacity(b);let mut scale=0.0;
        for row in &batch.sequences {
            let start=row.start_position.checked_add(depth+1).filter(|&p|p.checked_add(s).is_some_and(|end|end<=c.max_position_embeddings))
                .ok_or_else(||GenerationError("MTP batched shifted context exceeds capacity".into()))?;
            let (a,z,k)=self.target.rotary(start,s,input.dtype(),device);cos.push(a);sin.push(z);scale=k;
        }
        let output=self.target.layer_batch_forward(&predictor.decoder,0,input,batch,cache,Tensor::cat(cos,0),Tensor::cat(sin,0),scale)?;
        let hidden_states=norm(output,&predictor.output_norm,c.rms_norm_eps);
        cache.commit(batch);
        Ok(hidden_states)
    }

    fn confirm_batch(&self,batch:&ScheduledBatch,hidden:Tensor<DeviceBackend<R,F,I,BT>,3>,
        cache:&mut DeepseekMtpBatchCache<DeviceBackend<R,F,I,BT>>,device:&R::Device)->Result<(),GenerationError> {
        let width=self.target.config.hidden_size;let s=batch.next_n();let block=cache.target.config().block_size;
        let old=batch.sequences.iter().map(|row|cache.requests[&row.request_id].tails.clone()).collect::<Vec<_>>();
        let mut upstream=batch.sequences.iter().enumerate().map(|(i,row)|Some((row.start_position,
            hidden.clone().slice([i..i+1,0..s,0..width])))).collect::<Vec<_>>();
        for (i,row) in batch.sequences.iter().enumerate() {
            cache.requests.get_mut(&row.request_id).unwrap().tails[0]=Some(hidden.clone().slice([i..i+1,s-1..s,0..width]));
        }
        for depth in 0..self.predictors.len() {
            let mut groups=BTreeMap::<usize,Vec<(usize,crate::ScheduledSequence,Tensor<DeviceBackend<R,F,I,BT>,3>)>>::new();
            for (i,row) in batch.sequences.iter().enumerate() {
                let first=row.start_position.max(depth+1);
                if first>=row.context_length {continue;}
                let count=row.context_length-first;
                let (upstream_start,values)=upstream[i].as_ref().ok_or_else(||GenerationError("missing MTP batched upstream".into()))?;
                let previous=if first==row.start_position {
                    let tail=old[i][depth].clone().ok_or_else(||GenerationError("missing MTP confirmed hidden tail".into()))?;
                    if count==1 {tail} else {Tensor::cat(vec![tail,values.clone().slice([0..1,0..count-1,0..width])],1)}
                } else {
                    values.clone().slice([0..1,first-1-upstream_start..row.context_length-1-upstream_start,0..width])
                };
                let mut write=row.clone();write.token_ids=row.token_ids[first-row.start_position..].to_vec();
                write.start_position=first-depth-1;write.context_length=row.context_length-depth-1;
                write.block_table.truncate(write.context_length.div_ceil(block));
                groups.entry(count).or_default().push((i,write,previous));
            }
            let mut next=(0..batch.batch_size()).map(|_|None).collect::<Vec<_>>();
            for (count,rows) in groups {
                let work=ScheduledBatch{id:batch.id,kind:batch.kind,sequences:rows.iter().map(|r|r.1.clone()).collect()};
                let previous=Tensor::cat(rows.iter().map(|r|r.2.clone()).collect(),0);
                let output=self.predictor_batch(depth,&work,previous,&mut cache.predictors[depth],device)?;
                for (number,(i,_,_)) in rows.into_iter().enumerate() {
                    let row=&batch.sequences[i];let values=output.clone().slice([number..number+1,0..count,0..width]);
                    cache.requests.get_mut(&row.request_id).unwrap().tails[depth+1]=Some(values.clone().slice([0..1,count-1..count,0..width]));
                    next[i]=Some((row.context_length-count,values));
                }
            }
            upstream=next;
        }
        Ok(())
    }

    fn propose_batch(&self,batch:&ScheduledBatch,lookahead:&BTreeMap<RequestId,PagedKvAppendReservation>,count:usize,
        cache:&DeepseekMtpBatchCache<DeviceBackend<R,F,I,BT>>,device:&R::Device)->Result<Vec<Vec<i32>>,GenerationError> {
        let mut states=cache.predictors.clone();let block=cache.target.config().block_size;let width=self.target.config.hidden_size;
        let mut tails=batch.sequences.iter().map(|row|cache.requests[&row.request_id].tails.clone()).collect::<Vec<_>>();
        let mut spans=batch.sequences.iter().map(|row|vec![row.token_ids[0]]).collect::<Vec<_>>();
        for step in 0..count {
            let previous=tails.clone();
            for depth in step..self.predictors.len() {
                let mut work=batch.clone();let mut inputs=Vec::with_capacity(batch.batch_size());
                for (i,row) in work.sequences.iter_mut().enumerate() {
                    inputs.push(previous[i][depth].clone().ok_or_else(||GenerationError("MTP proposal prefix is not primed".into()))?);
                    row.token_ids=vec![spans[i][step]];row.start_position=states[depth].sequence_length(row.request_id).unwrap_or(0);
                    row.context_length=row.start_position+1;
                    row.block_table=lookahead[&row.request_id].block_table[..row.context_length.div_ceil(block)].to_vec();
                }
                let output=self.predictor_batch(depth,&work,Tensor::cat(inputs,0),&mut states[depth],device)?;
                let logits=if depth==step {Some(self.target.head.forward(output.clone()))} else {None};
                for (i,tail) in tails.iter_mut().enumerate() {
                    tail[depth+1]=Some(output.clone().slice([i..i+1,0..1,0..width]));
                    if let Some(logits)=&logits {
                        spans[i].push(self.target.greedy_token(logits.clone().slice([i..i+1,0..1,0..self.target.config.vocab_size]))?);
                    }
                }
            }
        }
        Ok(spans)
    }

    fn predictor_geometry(&self)->Vec<(usize,usize,usize,DType)> {
        let c=&self.target.config;let dtype=self.target.embedding.dtype();
        vec![match self.target.attention_mode {
            DeepseekV2AttentionMode::Expanded=>(c.num_attention_heads,c.qk_nope_head_dim+c.qk_rope_head_dim,c.v_head_dim,dtype),
            DeepseekV2AttentionMode::Absorbed=>(1,c.kv_lora_rank+c.qk_rope_head_dim,c.kv_lora_rank,dtype),
        }]
    }
    fn validate_prefix(&self,prefix:&DeepseekMtpBatchPrefix<DeviceBackend<R,F,I,BT>>)->Result<(),GenerationError> {
        let position=prefix.target.position();let c=&self.target.config;
        if position==0 || position>c.max_position_embeddings || prefix.predictors.len()!=self.predictors.len()
            || prefix.tails.len()!=self.predictors.len()+1 {return Err(GenerationError("MTP prefix geometry differs".into()));}
        for (depth,state) in prefix.predictors.iter().enumerate() {
            let expected=position.saturating_sub(depth+1);
            match state {
                Some(state) if state.position()==expected && expected>0=>state.validate(&self.predictor_geometry(),c.max_position_embeddings)?,
                None if expected==0=>(),_=>return Err(GenerationError("MTP predictor prefix position differs".into())),
            }
        }
        for (depth,tail) in prefix.tails.iter().enumerate() {
            match tail {
                Some(tail) if position>depth && tail.dims()==[1,1,c.hidden_size] && tail.dtype()==self.target.embedding.dtype()=>(),
                None if position<=depth=>(),_=>return Err(GenerationError("MTP prefix hidden tail differs".into())),
            }
        }
        Ok(())
    }
}

impl<R,F,I,BT> DeviceBatchModel<DeviceBackend<R,F,I,BT>> for DeepseekMtp<DeviceBackend<R,F,I,BT>>
where R:DeviceRuntime,R::Server:ComputeServer,R::Device:DeviceOps,F:FloatElement,I:IntElement,BT:BoolElement {
    type Cache=DeepseekMtpBatchCache<DeviceBackend<R,F,I,BT>>;
    fn batch_limits(&self)->CausalModelLimits {self.target.config.limits()}
    fn new_batch_cache(&self,config:PagedKvCacheConfig)->Self::Cache {
        Self::Cache{target:self.target.new_batch_cache(config),predictors:(0..self.predictors.len()).map(|_|DenseBatchCache::new(config,1)).collect(),requests:BTreeMap::new()}
    }
    fn fork_batch_cache(&self,cache:&Self::Cache)->Self::Cache {cache.clone()}
    fn retain_batch_cache(&self,cache:&mut Self::Cache,scheduler:&ContinuousBatchScheduler) {
        cache.target.retain(scheduler);for state in &mut cache.predictors {state.retain(scheduler);}
        cache.requests.retain(|id,_|scheduler.kv_cache().contains_sequence(*id));
    }
    fn lookahead_tokens(&self)->usize {self.predictors.len()}
    fn forward_batch(&self,batch:&ScheduledBatch,cache:&mut Self::Cache,device:&R::Device)->Result<Tensor<DeviceBackend<R,F,I,BT>,3>,GenerationError> {
        self.forward_batch_lookahead(batch,&BTreeMap::new(),cache,device)
    }
    fn forward_batch_lookahead(&self,batch:&ScheduledBatch,lookahead:&BTreeMap<RequestId,PagedKvAppendReservation>,cache:&mut Self::Cache,device:&R::Device)
        ->Result<Tensor<DeviceBackend<R,F,I,BT>,3>,GenerationError> {
        cache.target.validate(batch,self.batch_limits(),self.target.layers.len())?;
        if cache.predictors.len()!=self.predictors.len() {return Err(GenerationError("MTP batch predictor count differs".into()));}
        for row in &batch.sequences {cache.requests.entry(row.request_id).or_insert_with(||Request{tails:vec![None;self.predictors.len()+1],verified:None});}
        if batch.kind==ScheduledBatchKind::Prefill {
            let (logits,hidden)=self.target.forward_batch_features(batch,&mut cache.target,device,true)?;
            for row in &batch.sequences {cache.requests.get_mut(&row.request_id).unwrap().verified=None;}
            self.confirm_batch(batch,hidden,cache,device)?;return Ok(logits);
        }
        if batch.next_n()!=1 {return Err(GenerationError("MTP decode requires one committed token per row".into()));}
        let c=&self.target.config;let mut logits=(0..batch.batch_size()).map(|_|None).collect::<Vec<_>>();
        let mut hidden=logits.clone();let mut ordinary=Vec::new();let mut proposals=BTreeMap::<usize,Vec<usize>>::new();
        for (i,row) in batch.sequences.iter().enumerate() {
            let request=cache.requests.get_mut(&row.request_id).unwrap();
            if let Some(mut verified)=request.verified.take() {
                if verified.tokens.get(verified.next)==row.token_ids.first() {
                    let step=verified.next;cache.target.append_delta(row.request_id,&row.block_table,&verified.delta,step-1)?;
                    logits[i]=Some(verified.logits.clone().slice([0..1,step..step+1,0..c.vocab_size]));
                    hidden[i]=Some(verified.hidden.clone().slice([0..1,step..step+1,0..c.hidden_size]));
                    verified.next+=1;if verified.next<verified.tokens.len() {request.verified=Some(verified);}continue;
                }
            }
            let available=lookahead.get(&row.request_id).map_or(0,|r|r.token_count.saturating_sub(1))
                .min(self.predictors.len()).min(c.max_position_embeddings-row.context_length);
            if available>0 && request.tails.iter().all(Option::is_some) {proposals.entry(available).or_default().push(i);} else {ordinary.push(i);}
        }
        if !ordinary.is_empty() {
            let work=ScheduledBatch{id:batch.id,kind:batch.kind,sequences:ordinary.iter().map(|&i|batch.sequences[i].clone()).collect()};
            let (values,features)=self.target.forward_batch_features(&work,&mut cache.target,device,true)?;
            for (number,&i) in ordinary.iter().enumerate() {
                logits[i]=Some(values.clone().slice([number..number+1,0..1,0..c.vocab_size]));
                hidden[i]=Some(features.clone().slice([number..number+1,0..1,0..c.hidden_size]));
            }
        }
        for (count,indices) in proposals {
            let mut work=ScheduledBatch{id:batch.id,kind:batch.kind,sequences:indices.iter().map(|&i|batch.sequences[i].clone()).collect()};
            let spans=self.propose_batch(&work,lookahead,count,cache,device)?;
            for (row,tokens) in work.sequences.iter_mut().zip(spans) {
                row.token_ids=tokens;row.context_length=row.start_position+count+1;
                row.block_table=lookahead[&row.request_id].block_table[..row.context_length.div_ceil(cache.target.config().block_size)].to_vec();
            }
            let (values,features)=self.target.forward_batch_features(&work,&mut cache.target,device,false)?;
            for (number,&i) in indices.iter().enumerate() {
                let row=&work.sequences[number];let delta=cache.target.capture_delta(row.request_id,row.start_position+1,row.context_length)?;
                cache.target.truncate(row.request_id,row.start_position+1)?;
                let values=values.clone().slice([number..number+1,0..count+1,0..c.vocab_size]);
                let features=features.clone().slice([number..number+1,0..count+1,0..c.hidden_size]);
                logits[i]=Some(values.clone().slice([0..1,0..1,0..c.vocab_size]));hidden[i]=Some(features.clone().slice([0..1,0..1,0..c.hidden_size]));
                cache.requests.get_mut(&row.request_id).unwrap().verified=Some(BatchVerified{tokens:row.token_ids.clone(),logits:values,hidden:features,delta,next:1});
            }
        }
        let hidden=Tensor::cat(hidden.into_iter().map(|v|v.ok_or_else(||GenerationError("missing MTP confirmed hidden row".into()))).collect::<Result<Vec<_>,_>>()?,0);
        self.confirm_batch(batch,hidden,cache,device)?;
        Ok(Tensor::cat(logits.into_iter().map(|v|v.ok_or_else(||GenerationError("missing MTP logits row".into()))).collect::<Result<Vec<_>,_>>()?,0))
    }
}

impl<R,F,I,BT> PrefixBatchModel<DeviceBackend<R,F,I,BT>> for DeepseekMtp<DeviceBackend<R,F,I,BT>>
where R:DeviceRuntime,R::Server:ComputeServer,R::Device:DeviceOps,F:FloatElement,I:IntElement,BT:BoolElement {
    type Prefix=DeepseekMtpBatchPrefix<DeviceBackend<R,F,I,BT>>;
    fn prefix_tensor_transfer(&self)->bool {true}
    fn capture_prefix(&self,cache:&Self::Cache,id:RequestId)->Result<Self::Prefix,GenerationError> {
        let request=cache.requests.get(&id).ok_or_else(||GenerationError("missing MTP prefix request".into()))?;
        let predictors=cache.predictors.iter().map(|state|if state.sequence_length(id).unwrap_or(0)>0 {state.capture_prefix(id).map(Some)} else {Ok(None)})
            .collect::<Result<Vec<_>,_>>()?;
        let prefix=Self::Prefix{target:cache.target.capture_prefix(id)?,predictors,tails:request.tails.clone()};self.validate_prefix(&prefix)?;Ok(prefix)
    }
    fn prefix_position(&self,prefix:&Self::Prefix)->usize {prefix.target.position()}
    fn prefix_bytes(&self,prefix:&Self::Prefix)->usize {
        prefix.predictors.iter().flatten().fold(prefix.target.bytes(),|bytes,p|bytes.saturating_add(p.bytes()))
            .saturating_add(prefix.tails.iter().flatten().map(|t|t.dims().iter().product::<usize>()*t.dtype().size()).sum::<usize>())
    }
    fn restore_prefix(&self,cache:&mut Self::Cache,id:RequestId,table:&[KvPageId],prefix:&Self::Prefix)->Result<(),GenerationError> {
        self.validate_prefix(prefix)?;
        self.target.restore_prefix(&mut cache.target,id,table,&prefix.target)?;
        for (state,prefix) in cache.predictors.iter_mut().zip(&prefix.predictors) {
            if let Some(prefix)=prefix {state.restore_prefix(id,&table[..prefix.position().div_ceil(state.config().block_size)],prefix)?;}
        }
        cache.requests.insert(id,Request{tails:prefix.tails.clone(),verified:None});Ok(())
    }
    fn export_prefix_tensors(&self,prefix:&Self::Prefix,shard:crate::PrefixShard)->Result<crate::PrefixTensors<DeviceBackend<R,F,I,BT>>,GenerationError> {
        self.validate_prefix(prefix)?;let mut tensors=crate::PrefixTensors::new("deepseek-mtp-v1",prefix.target.position());
        tensors.add_child("target",self.target.export_prefix_tensors(&prefix.target,shard)?)?;
        for (depth,prefix) in prefix.predictors.iter().enumerate() {
            if let Some(prefix)=prefix {tensors.add_child(format!("predictor/{depth}"),prefix.export_tensors(shard)?)?;}
        }
        for (depth,tail) in prefix.tails.iter().enumerate() {
            if let Some(tail)=tail {tensors.push(format!("tail/{depth}"),tail.clone().unsqueeze_dim::<4>(1),(1,0..1))?;}
        }
        Ok(tensors)
    }
    fn import_prefix_tensors(&self,tensors:&crate::PrefixTensors<DeviceBackend<R,F,I,BT>>,shard:crate::PrefixShard,device:&R::Device)->Result<Self::Prefix,GenerationError> {
        let position=tensors.position;let count=self.predictors.len();let width=self.target.config.hidden_size;
        tensors.validate("deepseek-mtp-v1",self.target.config.max_position_embeddings,position.min(count+1))?;
        if tensors.scalars()!=0 || tensors.children()!=1+position.saturating_sub(1).min(count) {return Err(GenerationError("MTP transfer fields differ".into()));}
        let target=self.target.import_prefix_tensors(tensors.child("target")?,shard,device)?;
        if target.position()!=position {return Err(GenerationError("MTP transferred target position differs".into()));}
        let mut predictors=Vec::with_capacity(count);let mut tails=Vec::with_capacity(count+1);
        for depth in 0..count {
            predictors.push(if position>depth+1 {Some(DenseBatchPrefix::import_tensors(tensors.child(&format!("predictor/{depth}"))?,
                &self.predictor_geometry(),self.target.config.max_position_embeddings,shard,device)?)} else {None});
        }
        for depth in 0..=count {tails.push(if position>depth {Some(tensors.tensor(&format!("tail/{depth}"),[1,1,1,width],
            self.target.embedding.dtype(),(1,0..1),device)?.reshape([1,1,width]))} else {None});}
        let prefix=Self::Prefix{target,predictors,tails};self.validate_prefix(&prefix)?;Ok(prefix)
    }
    fn encode_prefix(&self,prefix:&Self::Prefix,_model_id:&str)->Result<Vec<u8>,GenerationError> {
        self.validate_prefix(prefix)?;
        let predictors=prefix.predictors.iter().map(|p|p.as_ref().map(DenseBatchPrefix::encode).transpose()).collect::<Result<Vec<_>,_>>()?;
        let tails=prefix.tails.iter().map(|t|t.as_ref().map(|t|t.clone().try_into_data().map_err(|e|GenerationError(e.to_string()))).transpose()).collect::<Result<Vec<_>,_>>()?;
        rmp_serde::to_vec(&(1u32,prefix.target.encode()?,predictors,tails)).map_err(|e|GenerationError(e.to_string()))
    }
    fn decode_prefix(&self,bytes:&[u8],_model_id:&str,device:&R::Device)->Result<Self::Prefix,GenerationError> {
        let (version,target,predictors,tails):(u32,Vec<u8>,Vec<Option<Vec<u8>>>,Vec<Option<TensorData>>)=rmp_serde::from_slice(bytes).map_err(|e|GenerationError(e.to_string()))?;
        if version!=1 || predictors.len()!=self.predictors.len() || tails.len()!=predictors.len()+1 {return Err(GenerationError("invalid MTP prefix record".into()));}
        let target=self.target.decode_prefix(&target,"",device)?;
        check_tails(&tails,target.position(),self.target.config.hidden_size).map_err(|e|GenerationError(e.to_string()))?;
        let predictors=predictors.iter().map(|p|p.as_ref().map(|bytes|DenseBatchPrefix::decode(bytes,&self.predictor_geometry(),self.target.config.max_position_embeddings,device)).transpose()).collect::<Result<Vec<_>,_>>()?;
        let tails=tails.into_iter().map(|t|t.map(|data|{let dtype=data.dtype;Tensor::from_data(data,(device,dtype))})).collect();
        let prefix=Self::Prefix{target,predictors,tails};self.validate_prefix(&prefix)?;Ok(prefix)
    }
}
