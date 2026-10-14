use super::{Backend,DType,LlamaKvCache,LlamaLayerCache,Tensor,TensorData};
use crate::GenerationCacheContinuation;
use ruda_model::record::{PrecisionSettings,Record,Recorder,RecorderError};
use super::rope::RotaryHistory;
use serde::{Deserialize,Deserializer,Serialize,Serializer,de::{Error,SeqAccess,Visitor},ser::SerializeTuple};
use std::marker::PhantomData;

/// Exact retained Llama KV history, original capacity limit and completed position.
/// Recorder float precision does not narrow the original native tensor storage.
pub struct LlamaKvCacheRecord<B: Backend> {
    version: u32,
    model_id: String,
    position: usize,
    capacity: usize,
    layers: Vec<Option<(TensorData,TensorData)>>,
    rotary: Vec<RotaryHistory>,
    backend: PhantomData<B>,
}

#[derive(Clone)]
pub struct LlamaKvCacheRecordItem {
    version: u32,
    model_id: String,
    position: usize,
    capacity: usize,
    layers: Vec<Option<(TensorData,TensorData)>>,
    rotary: Vec<RotaryHistory>,
}

impl Serialize for LlamaKvCacheRecordItem {
    fn serialize<S: Serializer>(&self,serializer: S) -> Result<S::Ok,S::Error> {
        let mut tuple = serializer.serialize_tuple(if self.version == 1 {5} else {6})?;
        tuple.serialize_element(&self.version)?;
        tuple.serialize_element(&self.model_id)?;
        tuple.serialize_element(&self.position)?;
        tuple.serialize_element(&self.capacity)?;
        tuple.serialize_element(&self.layers)?;
        if self.version != 1 {tuple.serialize_element(&self.rotary)?;}
        tuple.end()
    }
}

impl<'de> Deserialize<'de> for LlamaKvCacheRecordItem {
    fn deserialize<D: Deserializer<'de>>(deserializer: D) -> Result<Self,D::Error> {
        struct CacheVisitor;
        impl<'de> Visitor<'de> for CacheVisitor {
            type Value = LlamaKvCacheRecordItem;
            fn expecting(&self,f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
                f.write_str("a versioned Llama KV continuation tuple")
            }
            fn visit_seq<A: SeqAccess<'de>>(self,mut fields: A) -> Result<Self::Value,A::Error> {
                let version = fields.next_element()?.ok_or_else(||A::Error::missing_field("version"))?;
                if version != 1 && version != 2 {return Err(A::Error::custom("unsupported Llama KV continuation version"));}
                let model_id = fields.next_element()?.ok_or_else(||A::Error::missing_field("model_id"))?;
                let position = fields.next_element()?.ok_or_else(||A::Error::missing_field("position"))?;
                let capacity = fields.next_element()?.ok_or_else(||A::Error::missing_field("capacity"))?;
                let layers = fields.next_element()?.ok_or_else(||A::Error::missing_field("layers"))?;
                let rotary = if version == 2 {
                    fields.next_element()?.ok_or_else(||A::Error::missing_field("rotary"))?
                } else {Vec::new()};
                Ok(LlamaKvCacheRecordItem {version,model_id,position,capacity,layers,rotary})
            }
        }
        deserializer.deserialize_tuple(6,CacheVisitor)
    }
}

impl<B: Backend> Record<B> for LlamaKvCacheRecord<B> {
    type Item<S: PrecisionSettings> = LlamaKvCacheRecordItem;
    fn into_item<S: PrecisionSettings>(self) -> Self::Item<S> {
        LlamaKvCacheRecordItem {version:self.version,model_id:self.model_id,position:self.position,capacity:self.capacity,layers:self.layers,rotary:self.rotary}
    }
    fn from_item<S: PrecisionSettings>(item: Self::Item<S>,_device: &B::Device) -> Self {
        Self {version:item.version,model_id:item.model_id,position:item.position,capacity:item.capacity,layers:item.layers,rotary:item.rotary,backend:PhantomData}
    }
}

fn invalid(reason: &str) -> RecorderError {RecorderError::Unknown(format!("invalid Llama KV continuation: {reason}"))}

fn data_shape(key: &TensorData,value: &TensorData,position: usize) -> Result<(),RecorderError> {
    if key.rank() != 4 || value.rank() != 4 || key.shape != value.shape || key.dtype != value.dtype
        || !matches!(key.dtype,DType::F16|DType::BF16|DType::F32|DType::Flex32|DType::F64) {
        return Err(invalid("matching native floating rank-four K/V storage is required"));
    }
    if key.shape[0] == 0 || key.shape[1] == 0 || key.shape[3] == 0 || key.shape[2] != position {
        return Err(invalid("retained cache geometry differs from the completed position"));
    }
    for data in [key,value] {
        let count = data.shape.iter().try_fold(1usize,|count,&dimension|count.checked_mul(dimension))
            .and_then(|count|count.checked_mul(data.dtype.size())).ok_or_else(||invalid("cache byte count overflow"))?;
        if data.as_bytes().len() != count {return Err(invalid("cache payload byte count differs from native shape/dtype"));}
    }
    Ok(())
}

impl<B: Backend> LlamaKvCacheRecord<B> {
    pub(crate) fn position(&self) -> usize {self.position}

    /// Capture actual completed history with fallible native readback.
    /// Only logical prefixes are read; unused reserved GPU slots are never saved.
    /// model_id must identify the actual frozen weights/configuration/adapters.
    pub fn capture(cache: &LlamaKvCache<B>,model_id: &str) -> Result<Self,RecorderError> {
        if model_id.is_empty() || cache.layers.is_empty() || cache.position > cache.max_sequence_length {
            return Err(invalid("complete cache and explicit exact model identity are required"));
        }
        for layer in &cache.layers {layer.rotary.validate(cache.position).map_err(|e|invalid(&e.0))?;}
        let rotary = if cache.layers.iter().any(|layer|layer.rotary.is_dynamic()) {
            cache.layers.iter().map(|layer|layer.rotary.clone()).collect()
        } else {Vec::new()};
        let mut layers = Vec::with_capacity(cache.layers.len());
        for layer in &cache.layers {
            if layer.sequence_length != cache.position || layer.max_sequence_length != cache.max_sequence_length {
                return Err(invalid("layers have incomplete or inconsistent completed positions/capacities"));
            }
            let payload = match (&layer.key,&layer.value) {
                (None,None) if cache.position == 0 => None,
                (Some(key),Some(value)) => {
                    let [batch,heads,stored,width] = key.dims();
                    if key.dims() != value.dims() || key.dtype() != value.dtype() || key.device() != value.device()
                        || stored != layer.storage_capacity || stored < cache.position || stored > cache.max_sequence_length {
                        return Err(invalid("physical layer storage differs from actual retained K/V geometry"));
                    }
                    let ranges = [0..batch,0..heads,0..cache.position,0..width];
                    let key = key.clone().slice(ranges.clone()).try_into_data().map_err(|error|invalid(&error.to_string()))?;
                    let value = value.clone().slice(ranges).try_into_data().map_err(|error|invalid(&error.to_string()))?;
                    data_shape(&key,&value,cache.position)?;
                    Some((key,value))
                }
                _ => return Err(invalid("layer contains incomplete K/V state")),
            };
            layers.push(payload);
        }
        Ok(Self {version:if rotary.is_empty() {1} else {2},model_id:model_id.into(),position:cache.position,capacity:cache.max_sequence_length,layers,rotary,backend:PhantomData})
    }

    /// Save actual frozen host payloads through any existing native recorder.
    pub fn save<R: Recorder<B>>(self,recorder: &R,args: R::RecordArgs) -> Result<R::RecordOutput,RecorderError> {recorder.record(self,args)}
    /// Load raw exact-storage history; checked restore performs device allocation.
    pub fn load<R: Recorder<B>>(recorder: &R,args: R::LoadArgs,device: &B::Device) -> Result<Self,RecorderError> {recorder.load(args,device)}

    /// Restore on an explicit device with unchanged layer count, model and limit.
    /// Physical reserve starts at retained length and grows by the existing cache policy.
    pub fn restore(self,model_id: &str,layers: usize,capacity: usize,device: &B::Device) -> Result<LlamaKvCache<B>,RecorderError> {
        if !matches!(self.version,1|2) || model_id.is_empty() || self.model_id != model_id || self.layers.len() != layers
            || layers == 0 || capacity == 0 || self.capacity != capacity || self.position > capacity {
            return Err(invalid("record version, exact model identity, layers or original capacity differ"));
        }
        if (self.version == 1 && !self.rotary.is_empty()) || (self.version == 2 && self.rotary.len() != layers) {
            return Err(invalid("rotary history count differs from the record version/layers"));
        }
        for history in &self.rotary {history.validate(self.position).map_err(|e|invalid(&e.0))?;}
        for layer in &self.layers {
            match layer {
                Some((key,value)) => data_shape(key,value,self.position)?,
                None if self.position == 0 => (),
                None => return Err(invalid("completed layer is missing its actual KV history")),
            }
        }
        let position = self.position;
        let mut rotary = self.rotary.into_iter();
        let layers = self.layers.into_iter().map(|payload| {
            let mut layer = LlamaLayerCache::new(capacity);
            layer.rotary = rotary.next().unwrap_or_default();
            if let Some((key,value)) = payload {
                let dtype = key.dtype;
                layer.key = Some(Tensor::from_data(key,(device,dtype)));
                layer.value = Some(Tensor::from_data(value,(device,dtype)));
                layer.sequence_length = position;
                layer.storage_capacity = position;
            }
            layer
        }).collect();
        Ok(LlamaKvCache {layers,position,max_sequence_length:capacity})
    }
}

impl<B: Backend> LlamaKvCache<B> {
    /// Exact native continuation record, including only retained physical slots.
    pub fn record(&self,model_id: &str) -> Result<LlamaKvCacheRecord<B>,RecorderError> {LlamaKvCacheRecord::capture(self,model_id)}
}

impl<B: Backend> GenerationCacheContinuation<B> for LlamaKvCache<B> {
    type Record = LlamaKvCacheRecord<B>;
    fn generation_position(&self) -> usize {self.position}
    fn capture_generation(&self,model_id: &str,_device: &B::Device) -> Result<Self::Record,RecorderError> {self.record(model_id)}
    fn restore_generation(record: Self::Record,model_id: &str,template: &Self,position: usize,device: &B::Device)
        -> Result<Self,RecorderError> {
        if record.position != position {return Err(invalid("cache and generated-request input counts differ"));}
        record.restore(model_id,template.layers.len(),template.max_sequence_length,device)
    }
}
