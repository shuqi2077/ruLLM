use super::*;
use crate::PrefixBatchModel;
use ruda_model::record::{FullPrecisionSettings,Record};

impl<R,F,I,BT> Qwen35TextModel<DeviceBackend<R,F,I,BT>>
where R:DeviceRuntime,R::Server:ComputeServer,R::Device:DeviceOps,F:FloatElement,I:IntElement,BT:BoolElement {
    fn validate_batch_prefix(&self,prefix:&Qwen35Cache<DeviceBackend<R,F,I,BT>>)->Result<(),GenerationError> {
        let c=&self.config;let dtype=self.embedding.weight.val().dtype();
        if prefix.position==0 || prefix.position>c.max_position_embeddings || prefix.batch!=Some(1) || prefix.layers.len()!=self.layers.len() {
            return Err(GenerationError("invalid Qwen batch prefix position/layer count".into()));
        }
        self.rotary_for(prefix)?;
        for (state,kind) in prefix.layers.iter().zip(&c.layer_types) {
            let valid=match (state,kind) {
                (LayerCache::Full{key:Some(k),value:Some(v)},LayerType::FullAttention)=>
                    k.dims()==[1,c.num_key_value_heads,prefix.position,c.head_dim] && k.dims()==v.dims() && k.dtype()==dtype && v.dtype()==dtype,
                (LayerCache::Delta{convolution:Some(conv),state:Some(state)},LayerType::LinearAttention)=>{
                    let channels=2*c.linear_num_key_heads*c.linear_key_head_dim+c.linear_num_value_heads*c.linear_value_head_dim;
                    conv.dims()==[1,channels,c.linear_conv_kernel_dim] && conv.dtype()==dtype
                        && state.dims()==[1,c.linear_num_value_heads,c.linear_key_head_dim,c.linear_value_head_dim] && state.dtype()==DType::F32
                },
                _=>false,
            };
            if !valid {return Err(GenerationError("Qwen batch prefix native full/recurrent geometry differs".into()));}
        }
        Ok(())
    }
}

impl<R,F,I,BT> PrefixBatchModel<DeviceBackend<R,F,I,BT>> for Qwen35TextModel<DeviceBackend<R,F,I,BT>>
where R:DeviceRuntime,R::Server:ComputeServer,R::Device:DeviceOps,F:FloatElement,I:IntElement,BT:BoolElement {
    type Prefix=Qwen35Cache<DeviceBackend<R,F,I,BT>>;
    fn prefix_tensor_transfer(&self)->bool {true}
    fn export_prefix_tensors(&self,prefix:&Self::Prefix,shard:crate::PrefixShard)
        ->Result<crate::PrefixTensors<DeviceBackend<R,F,I,BT>>,GenerationError> {
        self.validate_batch_prefix(prefix)?;
        let c=&self.config;
        let schema=if prefix.rotary_length==0 {"qwen35-hybrid-v1"}else{"qwen35-hybrid-v2"};
        let mut tensors=crate::PrefixTensors::new(schema,prefix.position);
        if prefix.rotary_length!=0 {tensors.set_scalar("rotary-cached-length",prefix.rotary_length)?;}
        for (index,layer) in prefix.layers.iter().enumerate() {
            match layer {
                LayerCache::Full{key:Some(key),value:Some(value)}=>{
                    let span=shard.heads(shard.kv_heads,c.num_key_value_heads)?;
                    tensors.push(format!("{index}/key"),key.clone(),span.clone())?;
                    tensors.push(format!("{index}/value"),value.clone(),span)?;
                },
                LayerCache::Delta{convolution:Some(conv),state:Some(state)}=>{
                    let key_channels=c.linear_num_key_heads*c.linear_key_head_dim;
                    let value_channels=c.linear_num_value_heads*c.linear_value_head_dim;
                    for (name,start,n,heads,dim,global) in [
                        ("conv-q",0,key_channels,c.linear_num_key_heads,c.linear_key_head_dim,shard.linear_key_heads),
                        ("conv-k",key_channels,key_channels,c.linear_num_key_heads,c.linear_key_head_dim,shard.linear_key_heads),
                        ("conv-v",2*key_channels,value_channels,c.linear_num_value_heads,c.linear_value_head_dim,shard.linear_value_heads)] {
                        let tensor=conv.clone().slice([0..1,start..start+n,0..c.linear_conv_kernel_dim]).reshape([1,heads,dim,c.linear_conv_kernel_dim]);
                        tensors.push(format!("{index}/{name}"),tensor,shard.heads(global,heads)?)?;
                    }
                    tensors.push(format!("{index}/state"),state.clone(),shard.heads(shard.linear_value_heads,c.linear_num_value_heads)?)?;
                },
                _=>return Err(GenerationError("incomplete Qwen3.5 transfer state".into())),
            }
        }
        Ok(tensors)
    }
    fn import_prefix_tensors(&self,tensors:&crate::PrefixTensors<DeviceBackend<R,F,I,BT>>,shard:crate::PrefixShard,device:&R::Device)
        ->Result<Self::Prefix,GenerationError> {
        let c=&self.config;let dtype=self.embedding.weight.val().dtype();
        let count=c.layer_types.iter().map(|kind|if matches!(kind,LayerType::FullAttention) {2}else{4}).sum();
        let rotary_length=if self.rotary.initial_cached_length()==0 {
            tensors.validate("qwen35-hybrid-v1",c.max_position_embeddings,count)?;
            if tensors.scalars()!=0 {return Err(GenerationError("unexpected Qwen3.5 static rotary state".into()));}
            0
        } else {
            tensors.validate("qwen35-hybrid-v2",c.max_position_embeddings,count)?;
            if tensors.scalars()!=1 {return Err(GenerationError("missing Qwen3.5 dynamic rotary state".into()));}
            let length=tensors.scalar("rotary-cached-length")
                .ok_or_else(||GenerationError("missing Qwen3.5 dynamic rotary length".into()))?;
            if self.rotary.state_at(length).is_none() {return Err(GenerationError("Qwen3.5 dynamic rotary policy differs".into()));}
            length
        };
        if tensors.children()!=0 {return Err(GenerationError("Qwen3.5 prefix has unexpected child state".into()));}
        let mut prefix=self.new_cache();prefix.position=tensors.position;prefix.batch=Some(1);
        prefix.rotary_length=rotary_length;
        for (index,kind) in c.layer_types.iter().enumerate() {
            prefix.layers[index]=match kind {
                LayerType::FullAttention=>{
                    let shape=[1,c.num_key_value_heads,tensors.position,c.head_dim];let span=shard.heads(shard.kv_heads,c.num_key_value_heads)?;
                    LayerCache::Full{key:Some(tensors.tensor(&format!("{index}/key"),shape,dtype,span.clone(),device)?),
                        value:Some(tensors.tensor(&format!("{index}/value"),shape,dtype,span,device)?)}
                },
                LayerType::LinearAttention=>{
                    let mut conv=Vec::with_capacity(3);
                    for (name,heads,dim,global) in [("conv-q",c.linear_num_key_heads,c.linear_key_head_dim,shard.linear_key_heads),
                        ("conv-k",c.linear_num_key_heads,c.linear_key_head_dim,shard.linear_key_heads),
                        ("conv-v",c.linear_num_value_heads,c.linear_value_head_dim,shard.linear_value_heads)] {
                        conv.push(tensors.tensor(&format!("{index}/{name}"),[1,heads,dim,c.linear_conv_kernel_dim],dtype,
                            shard.heads(global,heads)?,device)?.reshape([1,heads*dim,c.linear_conv_kernel_dim]));
                    }
                    let state=tensors.tensor(&format!("{index}/state"),[1,c.linear_num_value_heads,c.linear_key_head_dim,c.linear_value_head_dim],
                        DType::F32,shard.heads(shard.linear_value_heads,c.linear_num_value_heads)?,device)?;
                    LayerCache::Delta{convolution:Some(Tensor::cat(conv,1)),state:Some(state)}
                },
            };
        }
        self.validate_batch_prefix(&prefix)?;Ok(prefix)
    }
    fn capture_prefix(&self,cache:&Self::Cache,id:RequestId)->Result<Self::Prefix,GenerationError> {
        let mut prefix=cache.requests.get(&id).ok_or_else(||GenerationError("unknown Qwen batch prefix request".into()))?.clone();
        let table=cache.tables.get(&id).ok_or_else(||GenerationError("missing Qwen batch prefix pages".into()))?;
        for index in 0..prefix.layers.len() {
            if !matches!(&prefix.layers[index],LayerCache::Full{..}) {continue;}
            let (key,value)=if cache.fused_paged {
                let arena=cache.arenas[index].as_ref().ok_or_else(||GenerationError("missing Qwen prefix arena".into()))?;
                let ids=table.iter().map(|p|i32::try_from(p.0).map_err(|e|GenerationError(e.to_string()))).collect::<Result<Vec<_>,_>>()?;
                let extract=|tensor:Tensor<DeviceBackend<R,F,I,BT>,4>| {
                    let [_,block,h,d]=tensor.dims();
                    let ids=Tensor::<DeviceBackend<R,F,I,BT>,1,Int>::from_data(TensorData::new(ids.clone(),[ids.len()]),&tensor.device());
                    tensor.select(0,ids).reshape([table.len()*block,h,d]).slice([0..prefix.position,0..h,0..d])
                        .swap_dims(0,1).unsqueeze_dim::<4>(0)
                };
                (extract(arena.key.clone()),extract(arena.value.clone()))
            } else {
                let mut keys=Vec::new();let mut values=Vec::new();
                for (number,id) in table.iter().enumerate() {
                    let page=cache.pages[index].get(id).ok_or_else(||GenerationError("missing Qwen prefix physical page".into()))?;
                    let [_,h,block,d]=page.key.dims();let count=block.min(prefix.position-number*block);
                    keys.push(page.key.clone().slice([0..1,0..h,0..count,0..d]));
                    values.push(page.value.clone().slice([0..1,0..h,0..count,0..d]));
                }
                (Tensor::cat(keys,2),Tensor::cat(values,2))
            };
            prefix.layers[index]=LayerCache::Full{key:Some(key),value:Some(value)};
        }
        self.validate_batch_prefix(&prefix)?;Ok(prefix)
    }
    fn prefix_position(&self,prefix:&Self::Prefix)->usize {prefix.position}
    fn prefix_bytes(&self,prefix:&Self::Prefix)->usize {
        fn bytes<B:Backend,const D:usize>(t:&Tensor<B,D>)->usize {t.dims().iter().product::<usize>().saturating_mul(t.dtype().size())}
        prefix.layers.iter().fold(0usize,|sum,layer|sum.saturating_add(match layer {
            LayerCache::Full{key,value}=>key.as_ref().map(bytes).unwrap_or(0).saturating_add(value.as_ref().map(bytes).unwrap_or(0)),
            LayerCache::Delta{convolution,state}=>convolution.as_ref().map(bytes).unwrap_or(0).saturating_add(state.as_ref().map(bytes).unwrap_or(0)),
        }))
    }
    fn restore_prefix(&self,cache:&mut Self::Cache,id:RequestId,table:&[KvPageId],prefix:&Self::Prefix)->Result<(),GenerationError> {
        self.validate_batch_prefix(prefix)?;
        let block=cache.config.block_size;
        if cache.requests.contains_key(&id) || table.len()!=prefix.position.div_ceil(block)
            || prefix.position>cache.config.max_sequence_length || table.iter().any(|p|p.0 as usize>=cache.config.num_pages)
            || table.iter().collect::<BTreeSet<_>>().len()!=table.len()
            || cache.tables.values().any(|other|other.iter().any(|p|table.contains(p))) {
            return Err(GenerationError("invalid Qwen prefix destination pages".into()));
        }
        let mut request=prefix.clone();
        for (index,layer) in prefix.layers.iter().enumerate() {
            let LayerCache::Full{key:Some(key),value:Some(value)}=layer else {continue;};
            let [_,h,_,d]=key.dims();
            if cache.fused_paged {
                let mut arena=cache.arenas[index].take().unwrap_or_else(||KvArena {
                    key:Tensor::empty([cache.config.num_pages,block,h,d],(&key.device(),key.dtype())),
                    value:Tensor::empty([cache.config.num_pages,block,h,d],(&value.device(),value.dtype())),
                    positions:BTreeMap::new(),resident:BTreeSet::new(),
                });
                for (number,page) in table.iter().enumerate() {
                    let start=number*block;let count=block.min(prefix.position-start);let p=page.0 as usize;
                    arena.key=arena.key.slice_assign([p..p+1,0..count,0..h,0..d],key.clone().slice([0..1,0..h,start..start+count,0..d]).swap_dims(1,2));
                    arena.value=arena.value.slice_assign([p..p+1,0..count,0..h,0..d],value.clone().slice([0..1,0..h,start..start+count,0..d]).swap_dims(1,2));
                }
                arena.positions.insert(id,prefix.position);arena.resident.extend(table.iter().copied());cache.arenas[index]=Some(arena);
            } else {
                for (number,&page) in table.iter().enumerate() {
                    let start=number*block;let count=block.min(prefix.position-start);
                    let k=Tensor::zeros([1,h,block,d],(&key.device(),key.dtype())).slice_assign([0..1,0..h,0..count,0..d],key.clone().slice([0..1,0..h,start..start+count,0..d]));
                    let v=Tensor::zeros([1,h,block,d],(&value.device(),value.dtype())).slice_assign([0..1,0..h,0..count,0..d],value.clone().slice([0..1,0..h,start..start+count,0..d]));
                    cache.pages[index].insert(page,KvPage{key:k,value:v});
                }
            }
            request.layers[index]=LayerCache::Full{key:None,value:None};
        }
        cache.requests.insert(id,request);cache.tables.insert(id,table.to_vec());Ok(())
    }
    fn encode_prefix(&self,prefix:&Self::Prefix,model_id:&str)->Result<Vec<u8>,GenerationError> {
        self.validate_batch_prefix(prefix)?;
        let item=prefix.record(model_id).map_err(|e|GenerationError(e.to_string()))?.into_item::<FullPrecisionSettings>();
        rmp_serde::to_vec(&item).map_err(|e|GenerationError(e.to_string()))
    }
    fn decode_prefix(&self,bytes:&[u8],model_id:&str,device:&R::Device)->Result<Self::Prefix,GenerationError> {
        type Native<R,F,I,BT>=DeviceBackend<R,F,I,BT>;
        let item=rmp_serde::from_slice(bytes).map_err(|e|GenerationError(e.to_string()))?;
        let record=Qwen35CacheRecord::<Native<R,F,I,BT>>::from_item::<FullPrecisionSettings>(item,device);
        let prefix=record.restore(model_id,&self.new_cache(),device).map_err(|e|GenerationError(e.to_string()))?;
        self.validate_batch_prefix(&prefix)?;Ok(prefix)
    }
}
