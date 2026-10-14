use super::*;
use crate::GenerationCacheContinuation;
use ruda_model::record::{PrecisionSettings, Record, Recorder, RecorderError};
use std::marker::PhantomData;

#[derive(Clone)]
pub(super) enum LayerCache<B: Backend> {
    Mla {key: Option<Tensor<B, 4>>, value: Option<Tensor<B, 4>>}, Kda(kda::State<B>),
}
#[derive(Clone)]
pub struct KimiLinearCache<B: Backend> {
    pub(super) position: usize, pub(super) batch: Option<usize>, pub(super) layers: Vec<LayerCache<B>>,
    layout: Vec<(u8, usize, usize, usize)>, dtype: DType, limit: usize,
}
impl<B: Backend> KimiLinearCache<B> {
    pub fn position(&self) -> usize { self.position }
    pub fn batch_size(&self) -> Option<usize> { self.batch }
    pub(super) fn new(c: &KimiLinearConfig, dtype: DType) -> Self {
        let mut layers = Vec::new(); let mut layout = Vec::new();
        for i in 0..c.num_hidden_layers {
            if c.is_kda_layer(i) {
                let a = c.linear_attn_config.as_ref().unwrap();
                layout.push((1, a.num_heads, a.head_dim, a.short_conv_kernel_size));
                layers.push(LayerCache::Kda(kda::State::empty()));
            } else {
                layout.push((0, c.num_attention_heads, c.qk_nope_head_dim + c.qk_rope_head_dim, c.v_head_dim));
                layers.push(LayerCache::Mla {key: None, value: None});
            }
        }
        Self {position: 0, batch: None, layers, layout, dtype, limit: c.model_max_length}
    }
    pub(super) fn validate(&self, c: &KimiLinearConfig, dtype: DType) -> Result<(), GenerationError> {
        let template = Self::new(c, dtype);
        if self.layout != template.layout || self.dtype != dtype || self.limit != template.limit {
            return Err(error("Kimi Linear cache model geometry differs"));
        }
        self.validate_state()
    }
    fn validate_state(&self) -> Result<(), GenerationError> {
        if self.layers.len() != self.layout.len() || self.position > self.limit
            || (self.position == 0) != self.batch.is_none() || self.batch == Some(0) {
            return Err(error("invalid Kimi Linear cache position or batch"));
        }
        for (layer, &(kind, h, d, w)) in self.layers.iter().zip(&self.layout) {
            let valid = match (layer, self.batch) {
                (LayerCache::Mla {key: None, value: None}, None) => kind == 0,
                (LayerCache::Kda(kda::State {conv: None, recurrent: None}), None) => kind == 1,
                (LayerCache::Mla {key: Some(k), value: Some(v)}, Some(b)) => kind == 0
                    && k.dims() == [b, h, self.position, d] && v.dims() == [b, h, self.position, w]
                    && k.dtype() == self.dtype && v.dtype() == self.dtype && k.device() == v.device(),
                (LayerCache::Kda(kda::State {conv: Some(conv), recurrent: Some(state)}), Some(b)) => kind == 1
                    && conv.iter().all(|v| v.dims() == [b, h * d, w] && v.dtype() == self.dtype && v.device() == state.device())
                    && state.dims() == [b, h, d, d] && state.dtype() == DType::F32,
                _ => false,
            };
            if !valid { return Err(error("Kimi Linear cache native layer layout differs")); }
        }
        Ok(())
    }
    pub fn record(&self, model_id: &str) -> Result<KimiLinearCacheRecord<B>, RecorderError> { KimiLinearCacheRecord::capture(self, model_id) }
}

type LayerData = (u8, Vec<TensorData>);
pub struct KimiLinearCacheRecord<B: Backend> {
    version: u32, model_id: String, position: usize, batch: Option<usize>, layers: Vec<LayerData>, backend: PhantomData<B>,
}
impl<B: Backend> Record<B> for KimiLinearCacheRecord<B> {
    type Item<S: PrecisionSettings> = (u32, String, usize, Option<usize>, Vec<(u8, Vec<TensorData>)>);
    fn into_item<S: PrecisionSettings>(self) -> Self::Item<S> { (self.version, self.model_id, self.position, self.batch, self.layers) }
    fn from_item<S: PrecisionSettings>(item: Self::Item<S>, _: &B::Device) -> Self {
        Self {version: item.0, model_id: item.1, position: item.2, batch: item.3, layers: item.4, backend: PhantomData}
    }
}
fn invalid(message: impl ToString) -> RecorderError { RecorderError::Unknown(message.to_string()) }
impl<B: Backend> KimiLinearCacheRecord<B> {
    pub fn capture(cache: &KimiLinearCache<B>, model_id: &str) -> Result<Self, RecorderError> {
        cache.validate_state().map_err(invalid)?;
        if model_id.is_empty() { return Err(invalid("Kimi Linear continuation requires model identity")); }
        let mut layers = Vec::with_capacity(cache.layers.len());
        for layer in &cache.layers {
            layers.push(match layer {
                LayerCache::Mla {key, value} => (0, match (key, value) {
                    (Some(k), Some(v)) => vec![k.clone().try_into_data().map_err(invalid)?, v.clone().try_into_data().map_err(invalid)?],
                    _ => Vec::new(),
                }),
                LayerCache::Kda(state) => (1, match (&state.conv, &state.recurrent) {
                    (Some(conv), Some(state)) => {
                        let mut tensors = conv.iter().map(|x| x.clone().try_into_data().map_err(invalid)).collect::<Result<Vec<_>, _>>()?;
                        tensors.push(state.clone().try_into_data().map_err(invalid)?); tensors
                    },
                    _ => Vec::new(),
                }),
            });
        }
        Ok(Self {version: 1, model_id: model_id.into(), position: cache.position, batch: cache.batch, layers, backend: PhantomData})
    }
    pub fn save<R: Recorder<B>>(self, recorder: &R, args: R::RecordArgs) -> Result<R::RecordOutput, RecorderError> { recorder.record(self, args) }
    pub fn load<R: Recorder<B>>(recorder: &R, args: R::LoadArgs, device: &B::Device) -> Result<Self, RecorderError> { recorder.load(args, device) }
    pub fn restore(self, model_id: &str, template: &KimiLinearCache<B>, device: &B::Device) -> Result<KimiLinearCache<B>, RecorderError> {
        if self.version != 1 || model_id.is_empty() || self.model_id != model_id || template.position != 0 || template.batch.is_some()
            || self.layers.len() != template.layout.len() || self.position > template.limit
            || (self.position == 0) != self.batch.is_none() || self.batch == Some(0) {
            return Err(invalid("Kimi Linear continuation identity, position or layer count differs"));
        }
        for ((kind, data), &(expected, h, d, w)) in self.layers.iter().zip(&template.layout) {
            if *kind != expected { return Err(invalid("Kimi Linear continuation layer order differs")); }
            let specs = match self.batch {
                None => Vec::new(),
                Some(b) if expected == 0 => vec![(vec![b, h, self.position, d], template.dtype), (vec![b, h, self.position, w], template.dtype)],
                Some(b) => vec![(vec![b, h * d, w], template.dtype), (vec![b, h * d, w], template.dtype),
                    (vec![b, h * d, w], template.dtype), (vec![b, h, d, d], DType::F32)],
            };
            if data.len() != specs.len() { return Err(invalid("incomplete Kimi Linear continuation layer")); }
            for (tensor, (shape, dtype)) in data.iter().zip(specs) {
                if tensor.shape.as_slice() != shape.as_slice() || tensor.dtype != dtype
                    || shape.iter().try_fold(dtype.size(), |n, &d| n.checked_mul(d)) != Some(tensor.as_bytes().len()) {
                    return Err(invalid("Kimi Linear continuation native tensor payload differs"));
                }
            }
        }
        let mut cache = template.clone(); cache.position = self.position; cache.batch = self.batch;
        for (state, (kind, data)) in cache.layers.iter_mut().zip(self.layers) {
            if data.is_empty() { continue; }
            let mut data = data.into_iter();
            let mut next = || data.next().unwrap();
            *state = if kind == 0 {
                let k = next(); let v = next(); let dtype = k.dtype;
                LayerCache::Mla {key: Some(Tensor::from_data(k, (device, dtype))), value: Some(Tensor::from_data(v, (device, dtype)))}
            } else {
                let conv = std::array::from_fn(|_| {let d = next(); let dtype = d.dtype; Tensor::from_data(d, (device, dtype))});
                LayerCache::Kda(kda::State {conv: Some(conv), recurrent: Some(Tensor::from_data(next(), (device, DType::F32)))})
            };
        }
        Ok(cache)
    }
}
impl<B: Backend> GenerationCacheContinuation<B> for KimiLinearCache<B> {
    type Record = KimiLinearCacheRecord<B>;
    fn generation_position(&self) -> usize { self.position }
    fn capture_generation(&self, model_id: &str, _: &B::Device) -> Result<Self::Record, RecorderError> { self.record(model_id) }
    fn restore_generation(record: Self::Record, model_id: &str, template: &Self, position: usize, device: &B::Device) -> Result<Self, RecorderError> {
        if record.position != position { return Err(invalid("Kimi Linear generation/cache position differs")); }
        record.restore(model_id, template, device)
    }
}
