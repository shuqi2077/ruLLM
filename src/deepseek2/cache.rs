use crate::{GenerationCacheContinuation, GenerationError};
use ruda_model::record::{PrecisionSettings, Record, RecorderError};
use ruda_tensor::api::{DType, Tensor, TensorData, backend::Backend};
use std::marker::PhantomData;

#[derive(Clone, Debug)]
pub(super) struct LayerCache<B: Backend> {
    storage: Option<Tensor<B, 4>>,
    position: usize,
    capacity: usize,
}

impl<B: Backend> LayerCache<B> {
    pub(super) fn append(
        &mut self,
        input: Tensor<B, 4>,
        limit: usize,
    ) -> Result<Tensor<B, 4>, GenerationError> {
        let [batch, heads, count, width] = input.dims();
        let end = self
            .position
            .checked_add(count)
            .filter(|&end| end <= limit)
            .ok_or_else(|| GenerationError("MLA cache capacity exceeded".into()))?;
        if batch != 1 || heads != 1 || count == 0 || width == 0 {
            return Err(GenerationError(
                "invalid shared compressed MLA input".into(),
            ));
        }
        if self.capacity < end {
            let mut capacity = self.capacity.max(128).min(limit);
            while capacity < end {
                capacity = capacity.checked_mul(2).unwrap_or(limit).min(limit);
            }
            let mut storage =
                Tensor::empty([1, 1, capacity, width], (&input.device(), input.dtype()));
            if let Some(previous) = &self.storage {
                storage = storage.slice_assign(
                    [0..1, 0..1, 0..self.position, 0..width],
                    previous
                        .clone()
                        .slice([0..1, 0..1, 0..self.position, 0..width]),
                );
            }
            self.storage = Some(storage);
            self.capacity = capacity;
        }
        let storage = self
            .storage
            .take()
            .ok_or_else(|| GenerationError("MLA physical buffer missing".into()))?;
        let storage = storage.slice_assign([0..1, 0..1, self.position..end, 0..width], input);
        let prefix = storage.clone().slice([0..1, 0..1, 0..end, 0..width]);
        self.storage = Some(storage);
        self.position = end;
        Ok(prefix)
    }
}

#[derive(Clone, Debug)]
pub struct DeepseekV2Cache<B: Backend> {
    pub(super) layers: Vec<LayerCache<B>>,
    pub(super) position: usize,
    limit: usize,
    width: usize,
}

impl<B: Backend> DeepseekV2Cache<B> {
    pub(super) fn new(layers: usize, limit: usize, width: usize) -> Self {
        Self {
            layers: (0..layers)
                .map(|_| LayerCache {
                    storage: None,
                    position: 0,
                    capacity: 0,
                })
                .collect(),
            position: 0,
            limit,
            width,
        }
    }
    pub fn position(&self) -> usize {
        self.position
    }
    pub fn capacity(&self) -> usize {
        self.limit
    }
    pub(super) fn truncate(&mut self, position: usize) -> Result<(), GenerationError> {
        if position > self.position || self.layers.iter().any(|layer| layer.position != self.position) {
            return Err(GenerationError("invalid compressed MLA rollback boundary".into()));
        }
        for layer in &mut self.layers { layer.position = position; }
        self.position = position;
        Ok(())
    }
    pub(super) fn compatible(
        &self,
        layers: usize,
        limit: usize,
        width: usize,
        device: &B::Device,
        dtype: DType,
    ) -> bool {
        self.layers.len() == layers
            && self.limit == limit
            && self.width == width
            && self.position <= limit
            && self.layers.iter().all(|layer| {
                layer.position == self.position
                    && layer.capacity <= limit
                    && match &layer.storage {
                        Some(storage) => {
                            storage.dims() == [1, 1, layer.capacity, width]
                                && storage.device() == *device
                                && storage.dtype() == dtype
                                && layer.capacity >= self.position
                        }
                        None => self.position == 0 && layer.capacity == 0,
                    }
            })
    }
}

pub struct DeepseekV2CacheRecord<B: Backend> {
    version: u32,
    model_id: String,
    position: usize,
    limit: usize,
    width: usize,
    layers: Vec<Option<TensorData>>,
    backend: PhantomData<B>,
}
impl<B: Backend> Record<B> for DeepseekV2CacheRecord<B> {
    type Item<S: PrecisionSettings> = (u32, String, usize, usize, usize, Vec<Option<TensorData>>);
    fn into_item<S: PrecisionSettings>(self) -> Self::Item<S> {
        (
            self.version,
            self.model_id,
            self.position,
            self.limit,
            self.width,
            self.layers,
        )
    }
    fn from_item<S: PrecisionSettings>(item: Self::Item<S>, _device: &B::Device) -> Self {
        Self {
            version: item.0,
            model_id: item.1,
            position: item.2,
            limit: item.3,
            width: item.4,
            layers: item.5,
            backend: PhantomData,
        }
    }
}
fn invalid(reason: impl std::fmt::Display) -> RecorderError {
    RecorderError::Unknown(format!("invalid MLA continuation: {reason}"))
}
fn validate_payload(data: &TensorData, position: usize, width: usize) -> Result<(), RecorderError> {
    if data.shape.as_slice() != [1, 1, position, width]
        || !matches!(data.dtype, DType::F32 | DType::BF16 | DType::F16)
        || position
            .checked_mul(width)
            .and_then(|n| n.checked_mul(data.dtype.size()))
            != Some(data.as_bytes().len())
    {
        return Err(invalid(
            "native compressed history shape, dtype or bytes differ",
        ));
    }
    Ok(())
}

impl<B: Backend> GenerationCacheContinuation<B> for DeepseekV2Cache<B> {
    type Record = DeepseekV2CacheRecord<B>;
    fn generation_position(&self) -> usize {
        self.position
    }
    fn capture_generation(
        &self,
        model_id: &str,
        device: &B::Device,
    ) -> Result<Self::Record, RecorderError> {
        if model_id.is_empty()
            || self.layers.is_empty()
            || self.width == 0
            || self.limit == 0
            || self.position > self.limit
        {
            return Err(invalid("incomplete cache or model identity"));
        }
        B::sync(device).map_err(invalid)?;
        let layers = self
            .layers
            .iter()
            .map(|layer| {
                if layer.position != self.position
                    || layer.capacity < self.position
                    || layer.capacity > self.limit
                {
                    return Err(invalid("layer positions or capacities differ"));
                }
                match &layer.storage {
                    Some(storage) => {
                        if storage.device() != *device
                            || storage.dims() != [1, 1, layer.capacity, self.width]
                        {
                            return Err(invalid("native storage differs from layer metadata"));
                        }
                        let data = storage
                            .clone()
                            .slice([0..1, 0..1, 0..self.position, 0..self.width])
                            .try_into_data()
                            .map_err(invalid)?;
                        validate_payload(&data, self.position, self.width)?;
                        Ok(Some(data))
                    }
                    None if self.position == 0 && layer.capacity == 0 => Ok(None),
                    None => Err(invalid("layer history is missing")),
                }
            })
            .collect::<Result<Vec<_>, _>>()?;
        Ok(DeepseekV2CacheRecord {
            version: 1,
            model_id: model_id.into(),
            position: self.position,
            limit: self.limit,
            width: self.width,
            layers,
            backend: PhantomData,
        })
    }
    fn restore_generation(
        record: Self::Record,
        model_id: &str,
        template: &Self,
        position: usize,
        device: &B::Device,
    ) -> Result<Self, RecorderError> {
        if record.version != 1
            || model_id.is_empty()
            || record.model_id != model_id
            || record.position != position
            || record.limit != template.limit
            || record.width != template.width
            || position > record.limit
            || record.layers.len() != template.layers.len()
            || record.layers.is_empty()
        {
            return Err(invalid(
                "model identity, version, input count or geometry differ",
            ));
        }
        let mut dtype = None;
        for data in &record.layers {
            match data {
                Some(data) => {
                    validate_payload(data, position, record.width)?;
                    if dtype.is_some_and(|dtype| dtype != data.dtype) {
                        return Err(invalid("layer storage dtypes differ"));
                    }
                    dtype = Some(data.dtype);
                }
                None if position == 0 => (),
                None => return Err(invalid("completed history missing")),
            }
        }
        let layers = record
            .layers
            .into_iter()
            .map(|data| match data {
                Some(data) => {
                    let dtype = data.dtype;
                    LayerCache {
                        storage: Some(Tensor::from_data(data, (device, dtype))),
                        position,
                        capacity: position,
                    }
                }
                None => LayerCache {
                    storage: None,
                    position: 0,
                    capacity: 0,
                },
            })
            .collect();
        Ok(Self {
            layers,
            position,
            limit: record.limit,
            width: record.width,
        })
    }
}
