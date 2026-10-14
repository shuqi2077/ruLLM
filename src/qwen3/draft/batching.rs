use super::*;
use crate::{CausalModelLimits,ContinuousBatchScheduler,DenseBatchCache,DenseBatchPrefix,DeviceBatchModel,
    PrefixBatchModel,PagedKvCacheConfig,PagedKvAppendReservation,RequestId,KvPageId,ScheduledBatch,ScheduledBatchKind};
use crate::batch_kv::DenseBatchDelta;
use ruda::runtime::server::ComputeServer;
use ruda_tensor::DeviceOps;
use ruda_tensor_device::{BoolElement,DeviceBackend,DeviceRuntime,FloatElement,IntElement};
use std::collections::BTreeMap;

#[derive(Clone)]
struct BatchVerified<B:Backend> {
    tokens:Vec<i32>,logits:Tensor<B,3>,features:Tensor<B,3>,delta:DenseBatchDelta<B>,next:usize,
}
#[derive(Clone)]
struct Request<B:Backend> {pending:Option<Tensor<B,3>>,verified:Option<BatchVerified<B>>}
impl<B:Backend> Default for Request<B> {fn default()->Self {Self{pending:None,verified:None}}}

#[derive(Clone)]
pub struct Qwen3DraftBatchCache<B:Backend> {
    target:DenseBatchCache<B>,draft:DenseBatchCache<B>,requests:BTreeMap<RequestId,Request<B>>,
}
#[derive(Clone)]
pub struct Qwen3DraftBatchPrefix<B:Backend> {
    target:DenseBatchPrefix<B>,draft:Option<DenseBatchPrefix<B>>,pending:Option<Tensor<B,3>>,
}

fn greedy_rows<B:Backend>(logits:Tensor<B,3>)->Result<Vec<i32>,GenerationError> {
    let [batch,one,vocab]=logits.dims();
    if one!=1 || vocab==0 {return Err(GenerationError("invalid batched draft logits".into()));}
    let values=logits.cast(DType::F32).try_into_data().map_err(|e|GenerationError(e.to_string()))?
        .to_vec::<f32>().map_err(|e|GenerationError(e.to_string()))?;
    let mut tokens=Vec::with_capacity(batch);
    for row in values.chunks_exact(vocab) {
        let mut best=None;
        for (index,&value) in row.iter().enumerate() {if !value.is_nan() && best.is_none_or(|(_,maximum)|value>maximum) {best=Some((index,value));}}
        tokens.push(best.ok_or_else(||GenerationError("all batched draft logits are NaN".into()))?.0 as i32);
    }
    Ok(tokens)
}

impl<R,F,I,BT> Draft<DeviceBackend<R,F,I,BT>>
where R:DeviceRuntime,R::Server:ComputeServer,R::Device:DeviceOps,F:FloatElement,I:IntElement,BT:BoolElement {
    fn propose_batch(&self,target:&Qwen3Model<DeviceBackend<R,F,I,BT>>,batch:&ScheduledBatch,
        lookahead:&BTreeMap<RequestId,PagedKvAppendReservation>,cache:&mut Qwen3DraftBatchCache<DeviceBackend<R,F,I,BT>>,
        context_len:usize,device:&R::Device)->Result<Vec<Vec<i32>>,GenerationError> {
        let c=&self.config;let b=batch.batch_size();let s=c.block_size;let dtype=target.embedding.dtype();
        let features=batch.sequences.iter().map(|row|cache.requests[&row.request_id].pending.as_ref().unwrap().clone()).collect();
        let context=norm(self.fc.forward(Tensor::cat(features,0)),&self.hidden_norm,c.rms_norm_eps);
        let mut queries=batch.clone();let mut writes=batch.clone();
        let mut tokens=Vec::with_capacity(b*s);
        for (number,row) in batch.sequences.iter().enumerate() {
            let old=cache.draft.sequence_length(row.request_id).unwrap_or(0);
            if old.checked_add(context_len)!=Some(row.start_position) {return Err(GenerationError("batched draft context position differs".into()));}
            let end=row.start_position+s;
            if end>c.max_position_embeddings {return Err(GenerationError("batched draft exceeds its context limit".into()));}
            let table=lookahead[&row.request_id].block_table[..end.div_ceil(cache.draft.config().block_size)].to_vec();
            let q=&mut queries.sequences[number];q.token_ids=vec![self.mask;s];q.token_ids[0]=row.token_ids[0];q.context_length=end;q.block_table=table.clone();
            let w=&mut writes.sequences[number];w.start_position=old;w.context_length=end;w.token_ids=vec![self.mask;context_len+s];w.block_table=table;
            tokens.extend_from_slice(&q.token_ids);
        }
        cache.draft.validate(&writes,CausalModelLimits{vocab_size:c.vocab_size,max_sequence_length:c.max_position_embeddings},self.layers.len())?;
        let ids=Tensor::<DeviceBackend<R,F,I,BT>,1,Int>::from_data(TensorData::new(tokens,[b*s]),device);
        let mut hidden=target.embedding.clone().select(0,ids).reshape([b,s,c.hidden_size]);
        for (index,layer) in self.layers.iter().enumerate() {
            let mut qcos=Vec::with_capacity(b);let mut qsin=Vec::with_capacity(b);
            let mut kcos=Vec::with_capacity(b);let mut ksin=Vec::with_capacity(b);let mut rotary_updates=Vec::with_capacity(b);
            for (query,write) in queries.sequences.iter().zip(&writes.sequences) {
                let state=cache.draft.rotary_state(write.request_id,index,write.start_position,&self.rotary[index])?;
                let (cos,sin)=state.cos_sin((query.start_position..query.context_length).map(|p|p as f32).collect(),
                    1,s,write.context_length,dtype,device);qcos.push(cos);qsin.push(sin);
                let (cos,sin)=state.cos_sin((write.start_position..write.context_length).map(|p|p as f32).collect(),
                    1,context_len+s,write.context_length,dtype,device);kcos.push(cos);ksin.push(sin);
                rotary_updates.push((write.request_id,write.context_length,state));
            }
            let qcos=Tensor::cat(qcos,0);let qsin=Tensor::cat(qsin,0);let kcos=Tensor::cat(kcos,0);let ksin=Tensor::cat(ksin,0);
            let normalized=norm(hidden.clone(),&layer.input_norm,c.rms_norm_eps);
            let q=norm(layer.q.forward(normalized.clone()).reshape([b,s,c.num_attention_heads,c.head_dim]),&layer.q_norm,c.rms_norm_eps).swap_dims(1,2);
            let tail=Tensor::cat(vec![context.clone(),normalized],1);
            let k=norm(layer.k.forward(tail.clone()).reshape([b,context_len+s,c.num_key_value_heads,c.head_dim]),&layer.k_norm,c.rms_norm_eps).swap_dims(1,2);
            let v=layer.v.forward(tail).reshape([b,context_len+s,c.num_key_value_heads,c.head_dim]).swap_dims(1,2);
            let q=rotate(q,qcos.clone(),qsin.clone());let k=rotate(k,kcos.clone(),ksin.clone());
            let attended=cache.draft.attention_queries(index,&writes,&queries,q,k,v,(c.head_dim as f32).sqrt().recip(),false)?
                .swap_dims(1,2).reshape([b,s,c.num_attention_heads*c.head_dim]);
            for (id,end,state) in rotary_updates {cache.draft.commit_rotary(id,index,end,&state)?;}
            hidden=hidden+layer.o.forward(attended);
            let normalized=norm(hidden.clone(),&layer.post_norm,c.rms_norm_eps);
            hidden=hidden+layer.mlp.forward(normalized)?;
        }
        cache.draft.commit(&writes);
        for row in &batch.sequences {cache.draft.truncate(row.request_id,row.start_position)?;cache.requests.get_mut(&row.request_id).unwrap().pending=None;}
        let logits=target.head.forward(norm(hidden,&self.norm,c.rms_norm_eps));
        let mut candidates=vec![Vec::with_capacity(s);b];let mut previous=batch.sequences.iter().map(|row|row.token_ids[0]).collect::<Vec<_>>();
        for step in (if self.anchor_first {0} else {1})..s {
            let mut row=logits.clone().slice([0..b,step..step+1,0..c.vocab_size]);
            if let Some((w1,w2))=&self.markov {
                let ids=Tensor::<DeviceBackend<R,F,I,BT>,1,Int>::from_data(TensorData::new(previous.clone(),[b]),device);
                let bias=w1.clone().select(0,ids).cast(DType::F32).matmul(w2.clone().cast(DType::F32).transpose()).reshape([b,1,c.vocab_size]);
                row=row.cast(DType::F32)+bias;
            }
            previous=greedy_rows(row)?;
            for (candidate,&token) in candidates.iter_mut().zip(&previous) {candidate.push(token);}
        }
        Ok(candidates)
    }
}

impl<R,F,I,BT> DeviceBatchModel<DeviceBackend<R,F,I,BT>> for Qwen3Drafted<DeviceBackend<R,F,I,BT>>
where R:DeviceRuntime,R::Server:ComputeServer,R::Device:DeviceOps,F:FloatElement,I:IntElement,BT:BoolElement {
    type Cache=Qwen3DraftBatchCache<DeviceBackend<R,F,I,BT>>;
    fn batch_limits(&self)->CausalModelLimits {self.target.batch_limits()}
    fn new_batch_cache(&self,config:PagedKvCacheConfig)->Self::Cache {
        Self::Cache{target:self.target.new_batch_cache(config),draft:DenseBatchCache::new(config,self.draft.layers.len()),requests:BTreeMap::new()}
    }
    fn fork_batch_cache(&self,cache:&Self::Cache)->Self::Cache {cache.clone()}
    fn retain_batch_cache(&self,cache:&mut Self::Cache,scheduler:&ContinuousBatchScheduler) {
        cache.target.retain(scheduler);cache.draft.retain(scheduler);cache.requests.retain(|id,_|scheduler.kv_cache().contains_sequence(*id));
    }
    fn lookahead_tokens(&self)->usize {self.draft.config.block_size-usize::from(!self.draft.anchor_first)}
    fn forward_batch(&self,batch:&ScheduledBatch,cache:&mut Self::Cache,device:&R::Device)
        ->Result<Tensor<DeviceBackend<R,F,I,BT>,3>,GenerationError> {
        self.forward_batch_lookahead(batch,&BTreeMap::new(),cache,device)
    }
    fn forward_batch_lookahead(&self,batch:&ScheduledBatch,lookahead:&BTreeMap<RequestId,PagedKvAppendReservation>,
        cache:&mut Self::Cache,device:&R::Device)->Result<Tensor<DeviceBackend<R,F,I,BT>,3>,GenerationError> {
        cache.target.validate(batch,self.batch_limits(),self.target.layers.len())?;
        let vocab=self.target.config.vocab_size;let width=self.draft.taps.len()*self.target.config.hidden_size;
        for row in &batch.sequences {cache.requests.entry(row.request_id).or_default();}
        if batch.kind==ScheduledBatchKind::Prefill {
            let (logits,features)=self.target.forward_batch_features(batch,&mut cache.target,device,true,&self.draft.taps)?;
            let features=features.ok_or_else(||GenerationError("batched prefill omitted draft features".into()))?;
            for (index,row) in batch.sequences.iter().enumerate() {
                let state=cache.requests.get_mut(&row.request_id).unwrap();state.verified=None;
                append_features(&mut state.pending,features.clone().slice([index..index+1,0..batch.next_n(),0..width]));
            }
            return Ok(logits);
        }
        if batch.next_n()!=1 {return Err(GenerationError("draft decode requires one committed anchor per row".into()));}
        let mut output=(0..batch.batch_size()).map(|_|None).collect::<Vec<_>>();
        let mut ordinary=Vec::new();let mut proposals=BTreeMap::<usize,Vec<usize>>::new();
        for (index,row) in batch.sequences.iter().enumerate() {
            let state=cache.requests.get_mut(&row.request_id).unwrap();
            if let Some(mut verified)=state.verified.take() {
                if verified.tokens.get(verified.next)==row.token_ids.first() {
                    let step=verified.next;
                    cache.target.append_delta(row.request_id,&row.block_table,&verified.delta,step-1)?;
                    output[index]=Some(verified.logits.clone().slice([0..1,step..step+1,0..vocab]));
                    append_features(&mut state.pending,verified.features.clone().slice([0..1,step..step+1,0..width]));
                    verified.next+=1;if verified.next<verified.tokens.len() {state.verified=Some(verified);}
                    continue;
                }
            }
            let candidate_count=self.lookahead_tokens();
            let can_draft=lookahead.get(&row.request_id).is_some_and(|r|r.token_count>=candidate_count+1)
                && row.start_position>0 && state.pending.as_ref().is_some_and(|p|p.dims()[1]>0)
                && row.start_position.checked_add(self.draft.config.block_size).is_some_and(|p|p<=self.draft.config.max_position_embeddings);
            if can_draft {proposals.entry(state.pending.as_ref().unwrap().dims()[1]).or_default().push(index);}
            else {ordinary.push(index);}
        }
        if !ordinary.is_empty() {
            let work=ScheduledBatch{id:batch.id,kind:batch.kind,sequences:ordinary.iter().map(|&i|batch.sequences[i].clone()).collect()};
            let (logits,features)=self.target.forward_batch_features(&work,&mut cache.target,device,true,&self.draft.taps)?;
            let features=features.ok_or_else(||GenerationError("batched decode omitted draft features".into()))?;
            for (row,&index) in ordinary.iter().enumerate() {
                output[index]=Some(logits.clone().slice([row..row+1,0..1,0..vocab]));
                append_features(&mut cache.requests.get_mut(&batch.sequences[index].request_id).unwrap().pending,
                    features.clone().slice([row..row+1,0..1,0..width]));
            }
        }
        for (context_len,indices) in proposals {
            let work=ScheduledBatch{id:batch.id,kind:batch.kind,sequences:indices.iter().map(|&i|batch.sequences[i].clone()).collect()};
            let candidates=self.draft.propose_batch(&self.target,&work,lookahead,cache,context_len,device)?;
            let mut verify=work.clone();
            for (row,candidates) in verify.sequences.iter_mut().zip(candidates) {
                row.token_ids.extend(candidates);row.context_length=row.start_position+row.token_ids.len();
                row.block_table=lookahead[&row.request_id].block_table[..row.context_length.div_ceil(cache.target.config().block_size)].to_vec();
            }
            let (logits,features)=self.target.forward_batch_features(&verify,&mut cache.target,device,false,&self.draft.taps)?;
            let features=features.ok_or_else(||GenerationError("batched verification omitted draft features".into()))?;
            for (row,&index) in indices.iter().enumerate() {
                let sequence=&verify.sequences[row];let s=sequence.token_ids.len();
                let delta=cache.target.capture_delta(sequence.request_id,sequence.start_position+1,sequence.context_length)?;
                cache.target.truncate(sequence.request_id,sequence.start_position+1)?;
                let row_logits=logits.clone().slice([row..row+1,0..s,0..vocab]);let row_features=features.clone().slice([row..row+1,0..s,0..width]);
                output[index]=Some(row_logits.clone().slice([0..1,0..1,0..vocab]));
                let state=cache.requests.get_mut(&sequence.request_id).unwrap();
                append_features(&mut state.pending,row_features.clone().slice([0..1,0..1,0..width]));
                state.verified=Some(BatchVerified{tokens:sequence.token_ids.clone(),logits:row_logits,features:row_features,delta,next:1});
            }
        }
        Ok(Tensor::cat(output.into_iter().map(|value|value.ok_or_else(||GenerationError("missing batched speculative row".into()))).collect::<Result<Vec<_>,_>>()?,0))
    }
}

impl<R,F,I,BT> Qwen3Drafted<DeviceBackend<R,F,I,BT>>
where R:DeviceRuntime,R::Server:ComputeServer,R::Device:DeviceOps,F:FloatElement,I:IntElement,BT:BoolElement {
    fn draft_geometry(&self)->Vec<(usize,usize,usize,DType)> {
        let c=&self.draft.config;vec![(c.num_key_value_heads,c.head_dim,c.head_dim,self.target.embedding.dtype());self.draft.layers.len()]
    }
    fn validate_batch_prefix(&self,prefix:&Qwen3DraftBatchPrefix<DeviceBackend<R,F,I,BT>>)->Result<(),GenerationError> {
        let position=prefix.target.position();let old=prefix.draft.as_ref().map_or(0,DenseBatchPrefix::position);
        if old>position || position>self.batch_limits().max_sequence_length {return Err(GenerationError("draft batch prefix positions differ".into()));}
        if let Some(draft)=&prefix.draft {draft.validate(&self.draft_geometry(),self.draft.config.max_position_embeddings)?;}
        match &prefix.pending {
            None if old==position=>(),
            Some(pending) if position>old && pending.dims()==[1,position-old,self.draft.taps.len()*self.target.config.hidden_size]
                && pending.dtype()==self.target.embedding.dtype()=>(),
            _=>return Err(GenerationError("draft batch prefix feature history differs".into())),
        }
        Ok(())
    }
}

impl<R,F,I,BT> PrefixBatchModel<DeviceBackend<R,F,I,BT>> for Qwen3Drafted<DeviceBackend<R,F,I,BT>>
where R:DeviceRuntime,R::Server:ComputeServer,R::Device:DeviceOps,F:FloatElement,I:IntElement,BT:BoolElement {
    type Prefix=Qwen3DraftBatchPrefix<DeviceBackend<R,F,I,BT>>;
    fn prefix_tensor_transfer(&self)->bool {true}
    fn export_prefix_tensors(&self,prefix:&Self::Prefix,shard:crate::PrefixShard)
        ->Result<crate::PrefixTensors<DeviceBackend<R,F,I,BT>>,GenerationError> {
        self.validate_batch_prefix(prefix)?;
        let mut tensors=crate::PrefixTensors::new("qwen3-feature-draft-v1",prefix.target.position());
        tensors.add_child("target",self.target.export_prefix_tensors(&prefix.target,shard)?)?;
        if let Some(draft)=&prefix.draft {tensors.add_child("draft",draft.export_tensors(Default::default())?)?;}
        if let Some(pending)=&prefix.pending {tensors.push("pending",pending.clone().unsqueeze_dim::<4>(1),(1,0..1))?;}
        Ok(tensors)
    }
    fn import_prefix_tensors(&self,tensors:&crate::PrefixTensors<DeviceBackend<R,F,I,BT>>,shard:crate::PrefixShard,device:&R::Device)
        ->Result<Self::Prefix,GenerationError> {
        let target=self.target.import_prefix_tensors(tensors.child("target")?,shard,device)?;
        let draft=if tensors.has_child("draft") {Some(DenseBatchPrefix::import_tensors(tensors.child("draft")?,
            &self.draft_geometry(),self.draft.config.max_position_embeddings,Default::default(),device)?)} else {None};
        let old=draft.as_ref().map_or(0,DenseBatchPrefix::position);
        if old>target.position() || target.position()!=tensors.position || tensors.children()!=1+usize::from(draft.is_some()) {
            return Err(GenerationError("draft transfer child positions/count differ".into()));
        }
        let count=target.position()-old;let width=self.draft.taps.len()*self.target.config.hidden_size;
        tensors.validate("qwen3-feature-draft-v1",self.batch_limits().max_sequence_length,usize::from(count>0))?;
        let pending=if count>0 {Some(tensors.tensor("pending",[1,1,count,width],self.target.embedding.dtype(),(1,0..1),device)?.reshape([1,count,width]))} else {None};
        let prefix=Self::Prefix{target,draft,pending};self.validate_batch_prefix(&prefix)?;Ok(prefix)
    }
    fn capture_prefix(&self,cache:&<Self as DeviceBatchModel<DeviceBackend<R,F,I,BT>>>::Cache,id:RequestId)->Result<Self::Prefix,GenerationError> {
        let request=cache.requests.get(&id).ok_or_else(||GenerationError("missing draft batch prefix request".into()))?;
        let prefix=Self::Prefix{target:cache.target.capture_prefix(id)?,draft:if cache.draft.sequence_length(id).unwrap_or(0)>0 {Some(cache.draft.capture_prefix(id)?)} else {None},pending:request.pending.clone()};
        self.validate_batch_prefix(&prefix)?;Ok(prefix)
    }
    fn prefix_position(&self,prefix:&Self::Prefix)->usize {prefix.target.position()}
    fn prefix_bytes(&self,prefix:&Self::Prefix)->usize {
        prefix.target.bytes().saturating_add(prefix.draft.as_ref().map_or(0,DenseBatchPrefix::bytes))
            .saturating_add(prefix.pending.as_ref().map_or(0,|p|p.dims().iter().product::<usize>().saturating_mul(p.dtype().size())))
    }
    fn restore_prefix(&self,cache:&mut <Self as DeviceBatchModel<DeviceBackend<R,F,I,BT>>>::Cache,id:RequestId,table:&[KvPageId],prefix:&Self::Prefix)->Result<(),GenerationError> {
        self.validate_batch_prefix(prefix)?;
        if cache.requests.contains_key(&id) {return Err(GenerationError("duplicate draft prefix request".into()));}
        self.target.restore_prefix(&mut cache.target,id,table,&prefix.target)?;
        if let Some(draft)=&prefix.draft {cache.draft.restore_prefix(id,&table[..draft.position().div_ceil(cache.draft.config().block_size)],draft)?;}
        cache.requests.insert(id,Request{pending:prefix.pending.clone(),verified:None});Ok(())
    }
    fn encode_prefix(&self,prefix:&Self::Prefix,model_id:&str)->Result<Vec<u8>,GenerationError> {
        self.validate_batch_prefix(prefix)?;
        let target=self.target.encode_prefix(&prefix.target,model_id)?;
        let draft=prefix.draft.as_ref().map(DenseBatchPrefix::encode).transpose()?;
        let pending=prefix.pending.clone().map(|p|p.try_into_data().map_err(|e|GenerationError(e.to_string()))).transpose()?;
        rmp_serde::to_vec(&(1u32,model_id,target,draft,pending)).map_err(|e|GenerationError(e.to_string()))
    }
    fn decode_prefix(&self,bytes:&[u8],model_id:&str,device:&R::Device)->Result<Self::Prefix,GenerationError> {
        let (version,identity,target,draft,pending):(u32,String,Vec<u8>,Option<Vec<u8>>,Option<TensorData>)=rmp_serde::from_slice(bytes).map_err(|e|GenerationError(e.to_string()))?;
        if version!=1 || identity!=model_id {return Err(GenerationError("draft batch prefix identity differs".into()));}
        let target=self.target.decode_prefix(&target,model_id,device)?;
        let draft=draft.map(|bytes|DenseBatchPrefix::decode(&bytes,&self.draft_geometry(),self.draft.config.max_position_embeddings,device)).transpose()?;
        let old=draft.as_ref().map_or(0,DenseBatchPrefix::position);
        if old>target.position() {return Err(GenerationError("draft prefix exceeds target position".into()));}
        check_pending(pending.as_ref(),target.position()-old,self.draft.taps.len()*self.target.config.hidden_size).map_err(|e|GenerationError(e.to_string()))?;
        let pending=pending.map(|data|{let dtype=data.dtype;Tensor::from_data(data,(device,dtype))});
        let prefix=Self::Prefix{target,draft,pending};self.validate_batch_prefix(&prefix)?;Ok(prefix)
    }
}
