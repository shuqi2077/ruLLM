use super::*;

#[derive(Clone, Debug, Default, PartialEq, Eq, serde::Serialize, serde::Deserialize)]
pub(crate) struct RotaryHistory {
    initial: usize,
    spans: Vec<(usize, usize)>,
}

#[derive(Clone)]
pub(crate) struct RotaryDelta { initial: usize, lengths: Vec<usize> }

impl RotaryHistory {
    pub(crate) fn is_dynamic(&self) -> bool { self.initial != 0 }
    pub(crate) fn validate(&self, position: usize) -> Result<(), crate::GenerationError> {
        let mut begin = 0;
        for &(end, length) in &self.spans {
            if self.initial == 0 || end <= begin || (length != 0 && (length < self.initial || length < end))
                || (length == 0 && begin < self.initial) {
                return Err(crate::GenerationError("invalid rotary history interval".into()));
            }
            begin = end;
        }
        if self.initial == 0 && !self.spans.is_empty() || self.initial != 0 && begin != position {
            return Err(crate::GenerationError("rotary history and completed token position differ".into()));
        }
        Ok(())
    }
    fn at(&self, position: usize) -> Option<usize> {
        if self.initial == 0 { return Some(0); }
        if position == 0 { return Some(self.initial); }
        self.spans.iter().find(|&&(end, _)| end >= position)
            .map(|&(_, length)| if length == 0 { position } else { length })
    }
    fn append_length(&mut self, initial: usize, end: usize, length: usize) -> Result<(), crate::GenerationError> {
        if initial == 0 {
            if self.initial != 0 || length != 0 { return Err(crate::GenerationError("rotary cache policy differs".into())); }
            return Ok(());
        }
        let begin = self.spans.last().map_or(0, |s| s.0);
        if (self.initial != 0 && self.initial != initial) || end <= begin || length < initial || length < end {
            return Err(crate::GenerationError("rotary history append is inconsistent".into()));
        }
        self.initial = initial;
        let stored = if begin >= initial && end == begin + 1 && length == end { 0 } else { length };
        if let Some(last) = self.spans.last_mut().filter(|last| last.1 == stored) { last.0 = end; }
        else { self.spans.push((end, stored)); }
        Ok(())
    }
    pub(crate) fn commit(&mut self, state: &RotaryState, end: usize) -> Result<(), crate::GenerationError> {
        self.append_length(state.parameters.initial_cached_length(), end, state.cached_length())
    }
    pub(crate) fn truncate(&mut self, position: usize) -> Result<(), crate::GenerationError> {
        if self.initial == 0 { return Ok(()); }
        if position == 0 { self.spans.clear(); return Ok(()); }
        let index = self.spans.partition_point(|&(end, _)| end < position);
        if index == self.spans.len() { return Err(crate::GenerationError("rotary rollback exceeds retained history".into())); }
        self.spans.truncate(index + 1);
        self.spans[index].0 = position;
        Ok(())
    }
    pub(crate) fn delta(&self, start: usize, end: usize) -> Result<RotaryDelta, crate::GenerationError> {
        let lengths = (start + 1..=end).map(|p| self.at(p)
            .ok_or_else(|| crate::GenerationError("rotary delta exceeds retained history".into())))
            .collect::<Result<Vec<_>, _>>()?;
        Ok(RotaryDelta { initial: self.initial, lengths })
    }
    pub(crate) fn export<B: Backend>(&self, tensors: &mut crate::PrefixTensors<B>, layer: usize) -> Result<(), crate::GenerationError> {
        tensors.set_scalar(format!("rotary/{layer}/initial"), self.initial)?;
        tensors.set_scalar(format!("rotary/{layer}/spans"), self.spans.len())?;
        for (index, &(end, length)) in self.spans.iter().enumerate() {
            tensors.set_scalar(format!("rotary/{layer}/{index}/end"), end)?;
            tensors.set_scalar(format!("rotary/{layer}/{index}/length"), length)?;
        }
        Ok(())
    }
    pub(crate) fn import<B: Backend>(tensors: &crate::PrefixTensors<B>, layer: usize) -> Result<(Self, usize), crate::GenerationError> {
        let scalar = |suffix: &str| tensors.scalar(&format!("rotary/{layer}/{suffix}"))
            .ok_or_else(|| crate::GenerationError("missing transferred rotary metadata".into()));
        let initial = scalar("initial")?;
        let count = scalar("spans")?;
        if count > tensors.position || count > tensors.scalars() / 2 {
            return Err(crate::GenerationError("transferred rotary interval count differs".into()));
        }
        let mut spans = Vec::with_capacity(count);
        for index in 0..count { spans.push((scalar(&format!("{index}/end"))?, scalar(&format!("{index}/length"))?)); }
        let history = Self { initial, spans };
        history.validate(tensors.position)?;
        Ok((history, 2 + 2 * count))
    }
}

impl RotaryDelta {
    pub(crate) fn append(&self, history: &mut RotaryHistory, position: usize, offset: usize) -> Result<(), crate::GenerationError> {
        let length = *self.lengths.get(offset).ok_or_else(|| crate::GenerationError("missing rotary delta token".into()))?;
        history.validate(position)?;
        if position != 0 && history.initial != self.initial {
            return Err(crate::GenerationError("rotary delta destination policy differs".into()));
        }
        history.append_length(self.initial, position + 1, length)
    }
}

#[derive(Clone, Debug, PartialEq)]
pub(crate) struct RopeParameters {
    pub(crate) theta: f32,
    pub(crate) inverse_frequencies: Vec<f32>,
    pub(crate) attention_factor: f64,
    pub(crate) scaling: RopeScaling,
    pub(crate) per_layer: Vec<RopeParameters>,
}

#[derive(Clone, Debug, PartialEq)]
pub(crate) enum RopeScaling {
    Static,
    Dynamic { factor: f64, original_length: usize },
    Long { original_length: usize, inverse_long: Vec<f32> },
}

#[derive(Clone, Debug)]
pub(crate) struct RotaryState {
    parameters: RopeParameters,
    cached_length: std::sync::Arc<std::sync::Mutex<usize>>,
}

impl RopeParameters {
    pub(crate) fn for_layer(&self, index: usize) -> &Self {
        if self.per_layer.is_empty() { self } else { &self.per_layer[index] }
    }
    pub(crate) fn state(&self) -> RotaryState {
        self.state_at(self.initial_cached_length()).expect("initial rotary state")
    }
    pub(crate) fn initial_cached_length(&self) -> usize {
        match self.scaling { RopeScaling::Dynamic { original_length, .. } => original_length, _ => 0 }
    }
    pub(crate) fn state_at(&self, cached_length: usize) -> Option<RotaryState> {
        let valid = match self.scaling {
            RopeScaling::Dynamic { original_length, .. } => cached_length >= original_length,
            _ => cached_length == 0,
        };
        valid.then(|| RotaryState { parameters: self.clone(),
            cached_length: std::sync::Arc::new(std::sync::Mutex::new(cached_length)) })
    }
    pub(crate) fn runtime_state(&self) -> Option<RotaryState> {
        if matches!(self.scaling, RopeScaling::Static) { None } else { Some(self.state()) }
    }
    pub(crate) fn tables<B: Backend>(&self, config: &LlamaConfig, device: &B::Device, dtype: DType) -> Vec<RotaryEncoding<B>> {
        let mut unique: Vec<(&Self, RotaryEncoding<B>)> = Vec::new();
        (0..config.num_hidden_layers).map(|index| {
            let parameters = self.for_layer(index);
            if let Some((_, table)) = unique.iter().find(|(p, _)| *p == parameters) { return table.clone(); }
            let table = parameters.init(config, device, dtype);
            unique.push((parameters, table.clone()));
            table
        }).collect()
    }
    pub(crate) fn default_for(config: &LlamaConfig) -> Self {
        let half = config.head_dimension() / 2;
        Self {theta: config.rope_theta, inverse_frequencies: (0..half)
            .map(|i| 1.0f32 / config.rope_theta.powf(i as f32 / half as f32)).collect(), attention_factor: 1.0,
            scaling: RopeScaling::Static, per_layer: Vec::new()}
    }
    pub(crate) fn init<B: Backend>(&self, config: &LlamaConfig, device: &B::Device, dtype: DType) -> RotaryEncoding<B> {
        let half = self.inverse_frequencies.len();
        let dimension = 2 * half;
        assert!(dimension > 0 && dimension <= config.head_dimension());
        let theta = Tensor::<B, 1>::from_data(TensorData::new(self.inverse_frequencies.clone(), [half]), (device, DType::F32));
        let mut rope = RotaryEncodingConfig::new(1, dimension).with_theta(self.theta).init(device);
        rope.theta = theta.clone();
        let length = if matches!(self.scaling, RopeScaling::Static) { config.max_sequence_length } else { 1 };
        let positions = Tensor::<B, 2>::from_data(TensorData::new(
            (0..length).map(|i| i as f32).collect::<Vec<_>>(), [length, 1]), (device, DType::F32));
        let frequencies = positions * theta.unsqueeze::<2>();
        let cos = frequencies.clone().cos() * self.attention_factor;
        let sin = frequencies.sin() * self.attention_factor;
        rope.freq_complex = Tensor::cat(vec![cos, sin], 1).reshape([length, 2, half])
            .transpose().unsqueeze_dim::<4>(2).repeat_dim(2, 2)
            .reshape([length, dimension, 2]).cast(dtype);
        rope
    }
}

impl RotaryState {
    pub(crate) fn fresh(&self) -> Self { self.parameters.state() }
    pub(crate) fn for_cache(&self, history: &RotaryHistory, position: usize) -> Result<Self, crate::GenerationError> {
        history.validate(position)?;
        let initial = self.parameters.initial_cached_length();
        if initial == 0 && history.initial == 0 { return Ok(self.fresh()); }
        if position == 0 && history.initial == 0 { return Ok(self.fresh()); }
        if history.initial != initial { return Err(crate::GenerationError("cache rotary history is missing or belongs to another policy".into())); }
        history.at(position).and_then(|length| self.parameters.state_at(length))
            .ok_or_else(|| crate::GenerationError("invalid rotary continuation length".into()))
    }
    pub(crate) fn cached_length(&self) -> usize {
        *self.cached_length.lock().expect("rotary state lock poisoned")
    }
    pub(crate) fn attention_factor(&self) -> f64 { self.parameters.attention_factor }

    pub(crate) fn frequencies(&self, sequence_length: usize) -> Vec<f32> {
        let p = &self.parameters;
        match &p.scaling {
            RopeScaling::Static => p.inverse_frequencies.clone(),
            RopeScaling::Long { original_length, inverse_long } => {
                if sequence_length > *original_length { inverse_long.clone() } else { p.inverse_frequencies.clone() }
            }
            RopeScaling::Dynamic { factor, original_length } => {
                let mut cached = self.cached_length.lock().expect("rotary state lock poisoned");
                if sequence_length > *cached { *cached = sequence_length; }
                if sequence_length < *original_length && *cached > *original_length { *cached = *original_length; }
                if *cached == *original_length { return p.inverse_frequencies.clone(); }
                let dim = (2 * p.inverse_frequencies.len()) as f64;
                let base = (p.theta as f64 * (factor * *cached as f64 / *original_length as f64 - (factor - 1.0))
                    .powf(dim / (dim - 2.0))) as f32;
                (0..p.inverse_frequencies.len()).map(|i| 1.0f32 / base.powf((2 * i) as f32 / dim as f32)).collect()
            }
        }
    }

    pub(crate) fn cos_sin<B: Backend>(&self, positions: Vec<f32>, batch: usize, sequence: usize,
        sequence_length: usize, dtype: DType, device: &B::Device) -> (Tensor<B, 4>, Tensor<B, 4>) {
        let inverse = self.frequencies(sequence_length);
        let half = inverse.len();
        let angles = Tensor::<B, 2>::from_data(TensorData::new(positions, [batch * sequence, 1]), (device, DType::F32))
            * Tensor::<B, 2>::from_data(TensorData::new(inverse, [1, half]), (device, DType::F32));
        let angles = Tensor::cat(vec![angles.clone(), angles], 1).reshape([batch, 1, sequence, 2 * half]);
        ((angles.clone().cos() * self.parameters.attention_factor).cast(dtype),
         (angles.sin() * self.parameters.attention_factor).cast(dtype))
    }

    pub(crate) fn apply<B: Backend>(&self, input: Tensor<B, 4>, start: usize, sequence_length: usize) -> Tensor<B, 4> {
        let sequence = input.dims()[2];
        let (cos, sin) = self.cos_sin((start..start + sequence).map(|p| p as f32).collect(), 1, sequence,
            sequence_length, input.dtype(), &input.device());
        apply_partial(input, cos, sin)
    }
}

pub(crate) fn apply_partial<B: Backend>(input: Tensor<B, 4>, cos: Tensor<B, 4>, sin: Tensor<B, 4>) -> Tensor<B, 4> {
    let [batch, heads, sequence, width] = input.dims();
    let dim = cos.dims()[3];
    assert!(dim > 0 && dim <= width && dim % 2 == 0);
    let prefix = input.clone().slice([0..batch, 0..heads, 0..sequence, 0..dim]);
    let first = prefix.clone().slice([0..batch, 0..heads, 0..sequence, 0..dim / 2]);
    let second = prefix.clone().slice([0..batch, 0..heads, 0..sequence, dim / 2..dim]);
    let output = prefix * cos + Tensor::cat(vec![-second, first], 3) * sin;
    if dim == width { output } else {
        Tensor::cat(vec![output, input.slice([0..batch, 0..heads, 0..sequence, dim..width])], 3)
    }
}

pub(super) fn qwen2_rope<B: Backend>(
    config: &LlamaConfig,
    device: &B::Device,
) -> RotaryEncoding<B> {
    let dtype = Tensor::<B, 1>::zeros([1], device).dtype();
    qwen2_rope_with_dtype(config, device, dtype)
}

pub(crate) fn qwen2_rope_with_dtype<B: Backend>(
    config: &LlamaConfig,
    device: &B::Device,
    dtype: DType,
) -> RotaryEncoding<B> {
    RopeParameters::default_for(config).init(config, device, dtype)
}
