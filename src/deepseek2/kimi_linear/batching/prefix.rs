use super::*;

impl<R, F, I, BT> PrefixBatchModel<DeviceBackend<R, F, I, BT>> for KimiLinearModel<DeviceBackend<R, F, I, BT>>
where R: DeviceRuntime, R::Server: ComputeServer, R::Device: DeviceOps, F: FloatElement, I: IntElement, BT: BoolElement {
    type Prefix = KimiLinearBatchPrefix<DeviceBackend<R, F, I, BT>>;
    fn prefix_tensor_transfer(&self)->bool {true}
    fn export_prefix_tensors(&self,prefix:&Self::Prefix,shard:crate::PrefixShard)
        ->Result<crate::PrefixTensors<DeviceBackend<R,F,I,BT>>,GenerationError> {
        if shard.kv_heads.is_some() || shard.linear_key_heads.is_some() || shard.linear_value_heads.is_some() {
            return Err(error("Kimi Linear native caches are EP-replicated, not TP-sharded"));
        }
        prefix.dense.validate(&self.dense_geometry(),self.config.model_max_length)?;
        self.validate_recurrent(&prefix.recurrent,true)?;
        let mut tensors=crate::PrefixTensors::new("kimi-linear-v1",prefix.dense.position());
        tensors.add_child("dense",prefix.dense.export_tensors(Default::default())?)?;
        if let Some(c)=&self.config.linear_attn_config {
            for (index,state) in prefix.recurrent.iter().enumerate() {
                for (which,conv) in state.conv.as_ref().unwrap().iter().enumerate() {
                    tensors.push(format!("{index}/conv-{which}"),conv.clone().reshape([1,c.num_heads,c.head_dim,c.short_conv_kernel_size]),(c.num_heads,0..c.num_heads))?;
                }
                tensors.push(format!("{index}/state"),state.recurrent.as_ref().unwrap().clone(),(c.num_heads,0..c.num_heads))?;
            }
        }
        Ok(tensors)
    }
    fn import_prefix_tensors(&self,tensors:&crate::PrefixTensors<DeviceBackend<R,F,I,BT>>,shard:crate::PrefixShard,device:&R::Device)
        ->Result<Self::Prefix,GenerationError> {
        if shard.kv_heads.is_some() || shard.linear_key_heads.is_some() || shard.linear_value_heads.is_some() {
            return Err(error("Kimi Linear native caches are EP-replicated, not TP-sharded"));
        }
        let count=self.config.num_hidden_layers-self.config.mla_layers();
        tensors.validate("kimi-linear-v1",self.config.model_max_length,count*4)?;
        if tensors.children()!=1 {return Err(error("Kimi Linear prefix children differ"));}
        let dense=DenseBatchPrefix::import_tensors(tensors.child("dense")?,&self.dense_geometry(),self.config.model_max_length,Default::default(),device)?;
        if dense.position()!=tensors.position {return Err(error("Kimi Linear dense/recurrent positions differ"));}
        let mut recurrent=Vec::with_capacity(count);
        if let Some(c)=&self.config.linear_attn_config {
            for index in 0..count {
                let mut conv=Vec::with_capacity(3);
                for which in 0..3 {
                    conv.push(tensors.tensor(&format!("{index}/conv-{which}"),[1,c.num_heads,c.head_dim,c.short_conv_kernel_size],
                        self.embedding.dtype(),(c.num_heads,0..c.num_heads),device)?.reshape([1,c.num_heads*c.head_dim,c.short_conv_kernel_size]));
                }
                let state=tensors.tensor(&format!("{index}/state"),[1,c.num_heads,c.head_dim,c.head_dim],DType::F32,(c.num_heads,0..c.num_heads),device)?;
                recurrent.push(kda::State{conv:Some(conv.try_into().map_err(|_|error("Kimi convolution count"))?),recurrent:Some(state)});
            }
        }
        self.validate_recurrent(&recurrent,true)?;Ok(Self::Prefix{dense,recurrent})
    }
    fn capture_prefix(&self, cache: &Self::Cache, id: RequestId) -> Result<Self::Prefix, GenerationError> {
        let recurrent = cache.recurrent.get(&id).ok_or_else(|| error("unknown Kimi Linear prefix request"))?.clone();
        self.validate_recurrent(&recurrent, true)?;
        let dense = cache.dense.capture_prefix(id)?;
        dense.validate(&self.dense_geometry(), self.config.model_max_length)?;
        Ok(Self::Prefix {dense, recurrent})
    }
    fn prefix_position(&self, prefix: &Self::Prefix) -> usize { prefix.dense.position() }
    fn prefix_bytes(&self, prefix: &Self::Prefix) -> usize {
        prefix.recurrent.iter().fold(prefix.dense.bytes(), |bytes, state| {
            let conv = state.conv.as_ref().map_or(0, |conv| conv.iter().fold(0usize, |n, x|
                n.saturating_add(x.dims().iter().product::<usize>().saturating_mul(x.dtype().size()))));
            let recurrent = state.recurrent.as_ref().map_or(0, |x| x.dims().iter().product::<usize>().saturating_mul(x.dtype().size()));
            bytes.saturating_add(conv).saturating_add(recurrent)
        })
    }
    fn restore_prefix(&self, cache: &mut Self::Cache, id: RequestId, table: &[KvPageId], prefix: &Self::Prefix) -> Result<(), GenerationError> {
        prefix.dense.validate(&self.dense_geometry(), self.config.model_max_length)?;
        self.validate_recurrent(&prefix.recurrent, true)?;
        if cache.recurrent.contains_key(&id) { return Err(error("Kimi Linear prefix destination already exists")); }
        cache.dense.restore_prefix(id, table, &prefix.dense)?;
        cache.recurrent.insert(id, prefix.recurrent.clone()); Ok(())
    }
    fn encode_prefix(&self, prefix: &Self::Prefix, model_id: &str) -> Result<Vec<u8>, GenerationError> {
        if model_id.is_empty() { return Err(error("Kimi Linear prefix requires model identity")); }
        prefix.dense.validate(&self.dense_geometry(), self.config.model_max_length)?;
        self.validate_recurrent(&prefix.recurrent, true)?;
        let mut states = Vec::with_capacity(prefix.recurrent.len());
        for state in &prefix.recurrent {
            let mut data = state.conv.as_ref().unwrap().iter().map(|x| x.clone().try_into_data().map_err(error)).collect::<Result<Vec<_>, _>>()?;
            data.push(state.recurrent.as_ref().unwrap().clone().try_into_data().map_err(error)?); states.push(data);
        }
        rmp_serde::to_vec(&(1u32, model_id, prefix.dense.encode()?, states)).map_err(error)
    }
    fn decode_prefix(&self, bytes: &[u8], model_id: &str, device: &R::Device) -> Result<Self::Prefix, GenerationError> {
        let (version, identity, dense, states): (u32, String, Vec<u8>, Vec<Vec<TensorData>>) = rmp_serde::from_slice(bytes).map_err(error)?;
        if version != 1 || model_id.is_empty() || identity != model_id || *device != self.embedding.device()
            || states.len() != self.config.num_hidden_layers - self.config.mla_layers() { return Err(error("Kimi Linear prefix identity/device/layout differs")); }
        if let Some(c) = &self.config.linear_attn_config {
            let conv = [1, c.num_heads * c.head_dim, c.short_conv_kernel_size]; let state = [1, c.num_heads, c.head_dim, c.head_dim];
            for data in &states {
                if data.len() != 4 { return Err(error("incomplete Kimi Linear prefix recurrent state")); }
                for (i, data) in data.iter().enumerate() {
                    let (shape, dtype) = if i == 3 {(&state[..], DType::F32)} else {(&conv[..], self.embedding.dtype())};
                    if data.shape.as_slice() != shape || data.dtype != dtype
                        || shape.iter().try_fold(dtype.size(), |n, &d| n.checked_mul(d)) != Some(data.as_bytes().len()) {
                        return Err(error("invalid Kimi Linear prefix native tensor payload"));
                    }
                }
            }
        }
        let dense = DenseBatchPrefix::decode(&dense, &self.dense_geometry(), self.config.model_max_length, device)?;
        let recurrent = states.into_iter().map(|data| {
            let mut data = data.into_iter();
            let conv = std::array::from_fn(|_| {let value = data.next().unwrap(); let dtype = value.dtype; Tensor::from_data(value, (device, dtype))});
            kda::State {conv: Some(conv), recurrent: Some(Tensor::from_data(data.next().unwrap(), (device, DType::F32)))}
        }).collect();
        Ok(Self::Prefix {dense, recurrent})
    }
}
