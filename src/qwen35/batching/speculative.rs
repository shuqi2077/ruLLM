use super::*;
use crate::SpeculativeBatchModel;

#[derive(Clone)]
pub(super) struct RecurrentStep<B:Backend> {
    pub(super) convolution:Tensor<B,3>,
    pub(super) state:Tensor<B,4>,
}
#[derive(Clone)]
pub(super) struct Trajectory<B:Backend> {
    pub(super) start:usize,
    pub(super) end:usize,
    pub(super) initial:Qwen35Cache<B>,
    pub(super) layers:Vec<Vec<RecurrentStep<B>>>,
    pub(super) rotary_lengths:Vec<usize>,
}
#[derive(Clone)]
enum LayerDelta<B:Backend> {
    Full {key:Tensor<B,4>,value:Tensor<B,4>},
    Recurrent(Vec<RecurrentStep<B>>),
}
#[derive(Clone)]
pub struct Qwen35BatchDelta<B:Backend> {start:usize,count:usize,layers:Vec<LayerDelta<B>>,rotary_lengths:Vec<usize>}

fn error(message:impl ToString)->GenerationError {GenerationError(message.to_string())}

fn kv_span<B:Backend>(cache:&Qwen35BatchCache<B>,id:RequestId,layer:usize,start:usize,end:usize)
    ->Result<(Tensor<B,4>,Tensor<B,4>),GenerationError> {
    let block=cache.config.block_size;
    let table=cache.tables.get(&id).ok_or_else(||error("missing Qwen delta page table"))?;
    let first=start/block;let last=end.div_ceil(block);
    if start>=end || last>table.len() {return Err(error("invalid Qwen delta KV span"));}
    if cache.fused_paged {
        let arena=cache.arenas[layer].as_ref().ok_or_else(||error("missing Qwen delta arena"))?;
        let pages=table[first..last].iter().map(|p|i32::try_from(p.0).map_err(error)).collect::<Result<Vec<_>,_>>()?;
        let extract=|tensor:Tensor<B,4>| {
            let [_,block,h,d]=tensor.dims();let offset=start%block;
            let ids=Tensor::<B,1,Int>::from_data(TensorData::new(pages.clone(),[pages.len()]),&tensor.device());
            tensor.select(0,ids).reshape([pages.len()*block,h,d]).slice([offset..offset+end-start,0..h,0..d])
                .swap_dims(0,1).unsqueeze_dim::<4>(0)
        };
        Ok((extract(arena.key.clone()),extract(arena.value.clone())))
    } else {
        let mut keys=Vec::new();let mut values=Vec::new();
        for page in first..last {
            let cached=cache.pages[layer].get(&table[page]).ok_or_else(||error("missing Qwen delta physical page"))?;
            let [_,h,_,d]=cached.key.dims();let from=start.max(page*block)-page*block;let to=end.min((page+1)*block)-page*block;
            keys.push(cached.key.clone().slice([0..1,0..h,from..to,0..d]));
            values.push(cached.value.clone().slice([0..1,0..h,from..to,0..d]));
        }
        Ok((Tensor::cat(keys,2),Tensor::cat(values,2)))
    }
}

impl<R,F,I,BT> SpeculativeBatchModel<DeviceBackend<R,F,I,BT>> for Qwen35TextModel<DeviceBackend<R,F,I,BT>>
where R:DeviceRuntime,R::Server:ComputeServer,R::Device:DeviceOps,F:FloatElement,I:IntElement,BT:BoolElement {
    type Delta=Qwen35BatchDelta<DeviceBackend<R,F,I,BT>>;
    fn begin_batch_speculation(&self,cache:&mut Self::Cache,batch:&ScheduledBatch)->Result<(),GenerationError> {
        for row in &batch.sequences {cache.trajectories.remove(&row.request_id);}
        Ok(())
    }
    fn forward_batch_block(&self,batch:&ScheduledBatch,cache:&mut Self::Cache,device:&R::Device)
        ->Result<Tensor<DeviceBackend<R,F,I,BT>,3>,GenerationError> {
        self.forward_batch_impl(batch,cache,device,true)
    }
    fn capture_batch_delta(&self,cache:&Self::Cache,id:RequestId,start:usize,end:usize)->Result<Self::Delta,GenerationError> {
        let history=cache.trajectories.get(&id).ok_or_else(||error("missing Qwen speculative trajectory"))?;
        if start>=end || start<history.start || end>history.end || cache.sequence_length(id)!=Some(history.end)
            || history.layers.len()!=self.layers.len() || history.rotary_lengths.len()!=history.end-history.start {
            return Err(error("Qwen speculative trajectory span differs"));
        }
        let mut layers=Vec::with_capacity(self.layers.len());
        for (index,kind) in self.config.layer_types.iter().enumerate() {
            layers.push(match kind {
                LayerType::FullAttention=>{let (key,value)=kv_span(cache,id,index,start,end)?;LayerDelta::Full{key,value}},
                LayerType::LinearAttention=>{
                    let states=&history.layers[index];
                    if states.len()!=history.end-history.start {return Err(error("incomplete Qwen recurrent trajectory"));}
                    LayerDelta::Recurrent(states[start-history.start..end-history.start].to_vec())
                },
            });
        }
        Ok(Self::Delta{start,count:end-start,layers,
            rotary_lengths:history.rotary_lengths[start-history.start..end-history.start].to_vec()})
    }
    fn truncate_batch_cache(&self,cache:&mut Self::Cache,id:RequestId,position:usize)->Result<(),GenerationError> {
        let history=cache.trajectories.get(&id).ok_or_else(||error("missing Qwen truncate trajectory"))?;
        if position<history.start || position>history.end || cache.sequence_length(id)!=Some(history.end) {
            return Err(error("Qwen truncate position exceeds trajectory"));
        }
        let mut request=history.initial.clone();request.position=position;
        if position>history.start {
            request.batch=Some(1);
            request.rotary_length=*history.rotary_lengths.get(position-history.start-1)
                .ok_or_else(||error("missing Qwen truncate rotary state"))?;
            for (index,kind) in self.config.layer_types.iter().enumerate() {
                if *kind==LayerType::LinearAttention {
                    let step=history.layers[index].get(position-history.start-1).ok_or_else(||error("missing Qwen truncate recurrent state"))?;
                    request.layers[index]=LayerCache::Delta {convolution:Some(step.convolution.clone()),state:Some(step.state.clone())};
                }
            }
        }
        let table=cache.tables.get_mut(&id).ok_or_else(||error("missing Qwen truncate pages"))?;
        let keep=position.div_ceil(cache.config.block_size);
        if keep>table.len() {return Err(error("Qwen truncate page count differs"));}
        let released=table.split_off(keep);
        for pages in &mut cache.pages {for page in &released {pages.remove(page);}}
        for arena in cache.arenas.iter_mut().flatten() {
            arena.positions.insert(id,position);
            for page in &released {arena.resident.remove(page);}
        }
        cache.requests.insert(id,request);cache.trajectories.remove(&id);Ok(())
    }
    fn append_batch_delta(&self,cache:&mut Self::Cache,id:RequestId,table:&[KvPageId],delta:&Self::Delta,offset:usize)
        ->Result<(),GenerationError> {
        let c=&self.config;let position=delta.start.checked_add(offset).ok_or_else(||error("Qwen delta position overflow"))?;
        let block=cache.config.block_size;
        if offset>=delta.count || cache.sequence_length(id)!=Some(position) || position>=cache.config.max_sequence_length
            || position>=c.max_position_embeddings || delta.layers.len()!=self.layers.len() || table.len()!=(position+1).div_ceil(block)
            || delta.rotary_lengths.len()!=delta.count
            || delta.rotary_lengths.get(offset).is_none_or(|&length|self.rotary.state_at(length).is_none())
            || table.iter().any(|p|p.0 as usize>=cache.config.num_pages) || table.iter().collect::<BTreeSet<_>>().len()!=table.len()
            || cache.tables.get(&id).is_none_or(|old|!table.starts_with(old))
            || cache.tables.iter().any(|(&other,pages)|other!=id && pages.iter().any(|p|table.contains(p))) {
            return Err(error("invalid Qwen delta append destination"));
        }
        let dtype=self.embedding.weight.val().dtype();let device=self.embedding.weight.val().device();
        let channels=2*c.linear_num_key_heads*c.linear_key_head_dim+c.linear_num_value_heads*c.linear_value_head_dim;
        for (layer,kind) in delta.layers.iter().zip(&c.layer_types) {
            let valid=match (layer,kind) {
                (LayerDelta::Full{key,value},LayerType::FullAttention)=>
                    key.dims()==[1,c.num_key_value_heads,delta.count,c.head_dim] && key.dims()==value.dims()
                    && key.dtype()==dtype && value.dtype()==dtype && key.device()==device && value.device()==device,
                (LayerDelta::Recurrent(steps),LayerType::LinearAttention)=>steps.len()==delta.count && steps.get(offset).is_some_and(|step|
                    step.convolution.dims()==[1,channels,c.linear_conv_kernel_dim] && step.convolution.dtype()==dtype && step.convolution.device()==device
                    && step.state.dims()==[1,c.linear_num_value_heads,c.linear_key_head_dim,c.linear_value_head_dim]
                    && step.state.dtype()==DType::F32 && step.state.device()==device),
                _=>false,
            };
            if !valid {return Err(error("Qwen delta layer geometry/device differs"));}
        }
        let page=table[position/block];let slot=position%block;
        for (index,layer) in delta.layers.iter().enumerate() {
            match layer {
                LayerDelta::Full{key,value}=>{
                    let (h,d)=(c.num_key_value_heads,c.head_dim);
                    let key=key.clone().slice([0..1,0..h,offset..offset+1,0..d]);
                    let value=value.clone().slice([0..1,0..h,offset..offset+1,0..d]);
                    if cache.fused_paged {
                        let mut arena=cache.arenas[index].take().ok_or_else(||error("missing Qwen delta destination arena"))?;
                        let p=page.0 as usize;
                        arena.key=arena.key.slice_assign([p..p+1,slot..slot+1,0..h,0..d],key.swap_dims(1,2));
                        arena.value=arena.value.slice_assign([p..p+1,slot..slot+1,0..h,0..d],value.swap_dims(1,2));
                        arena.positions.insert(id,position+1);arena.resident.insert(page);cache.arenas[index]=Some(arena);
                    } else {
                        let old=cache.pages[index].remove(&page).unwrap_or_else(||KvPage {
                            key:Tensor::zeros([1,h,block,d],(&device,dtype)),value:Tensor::zeros([1,h,block,d],(&device,dtype)),
                        });
                        cache.pages[index].insert(page,KvPage {
                            key:old.key.slice_assign([0..1,0..h,slot..slot+1,0..d],key),
                            value:old.value.slice_assign([0..1,0..h,slot..slot+1,0..d],value),
                        });
                    }
                },
                LayerDelta::Recurrent(steps)=>{
                    let step=&steps[offset];cache.requests.get_mut(&id).unwrap().layers[index]=LayerCache::Delta {
                        convolution:Some(step.convolution.clone()),state:Some(step.state.clone()),
                    };
                },
            }
        }
        let request=cache.requests.get_mut(&id).unwrap();request.position=position+1;request.batch=Some(1);
        request.rotary_length=delta.rotary_lengths[offset];
        cache.tables.insert(id,table.to_vec());cache.trajectories.remove(&id);Ok(())
    }
}
