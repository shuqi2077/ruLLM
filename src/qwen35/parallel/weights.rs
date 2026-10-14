use super::*;
use crate::huggingface::{AwqQuantizationConfig, checkpoint::Checkpoint};
use ruda_model::module::Param;
use std::ops::Range;

#[derive(Clone)]
pub(super) struct Part {
    pub input: Range<usize>,
    pub output: Vec<Range<usize>>,
    pub bias: bool,
}

impl Part {
    pub fn full(input: usize, output: usize) -> Self { Self { input: 0..input, output: vec![0..output], bias: true } }
    pub fn outputs(&self) -> usize { self.output.iter().map(Range::len).sum() }
    pub fn validate(&self, input: usize, output: usize) -> Result<(), HuggingFaceLoadError> {
        if self.input.is_empty() || self.input.end > input || self.output.is_empty()
            || self.output.iter().any(|range| range.is_empty() || range.end > output)
            || self.output.iter().try_fold(0usize, |n, r| n.checked_add(r.len()))
                .and_then(|n| n.checked_mul(self.input.len())).is_none_or(|n| n > u32::MAX as usize) {
            return Err(HuggingFaceLoadError("invalid Qwen3.5 TP projection selection".into()));
        }
        Ok(())
    }
}

pub(super) type PackedPartLoader<B: Backend> = fn(&mut Checkpoint, &str, usize, usize,
    &AwqQuantizationConfig, DType, bool, &Part, &<B as ruda_tensor::BackendTypes>::Device)
    -> Result<Projection<B>, HuggingFaceLoadError>;

pub(super) struct Weights<'a, B: Backend> {
    pub checkpoint: &'a mut Checkpoint,
    pub device: &'a B::Device,
    pub quantization: Option<AwqQuantizationConfig>,
    pub dtype: DType,
    pub packed: Option<PackedPartLoader<B>>,
    pub fp8: Option<super::super::fp8_loading::Weights<B>>,
    pub dense_awq: usize,
    pub packed_awq: usize,
}

pub(super) fn floating_data<const D: usize>(checkpoint: &Checkpoint, name: &str, shape: [usize; D])
    -> Result<TensorData, HuggingFaceLoadError> {
    let value = checkpoint.tensors.get(name).ok_or_else(|| HuggingFaceLoadError(format!("missing tensor {name}")))?;
    if value.shape != shape.into() || !matches!(value.dtype, DType::F16 | DType::BF16 | DType::F32) {
        return Err(HuggingFaceLoadError(format!("{name}: TP floating tensor shape/dtype differs")));
    }
    value.to_data().map_err(|e| HuggingFaceLoadError(e.to_string()))
}

pub(super) fn bias_data(checkpoint: &mut Checkpoint, prefix: &str, output: usize, part: &Part, dtype: DType)
    -> Result<TensorData, HuggingFaceLoadError> {
    let name = format!("{prefix}.bias");
    let source = floating_data(checkpoint, &name, [output])?;
    let size = source.dtype.size();
    let mut bytes = Vec::with_capacity(part.outputs() * size);
    for range in &part.output { bytes.extend_from_slice(&source.as_bytes()[range.start * size..range.end * size]); }
    checkpoint.consumed.insert(name);
    Ok(TensorData::from_bytes_vec(bytes, [part.outputs()], source.dtype).convert_dtype(dtype))
}

impl<B: Backend> Weights<'_, B> {
    pub fn tensor<const D: usize>(&mut self, name: &str, shape: [usize; D]) -> Result<Tensor<B, D>, HuggingFaceLoadError> {
        self.checkpoint.tensor(name, shape, self.device)
    }
    pub fn rows<const D: usize>(&mut self, name: &str, shape: [usize; D], ranges: &[Range<usize>])
        -> Result<Tensor<B, D>, HuggingFaceLoadError> {
        let source = floating_data(self.checkpoint, name, shape)?;
        if ranges.is_empty() || ranges.iter().any(|r| r.is_empty() || r.end > shape[0]) {
            return Err(HuggingFaceLoadError(format!("{name}: invalid TP row selection")));
        }
        let stride = shape[1..].iter().product::<usize>() * source.dtype.size();
        let mut local = shape;
        local[0] = ranges.iter().map(Range::len).sum();
        let mut bytes = Vec::with_capacity(local[0] * stride);
        for range in ranges { bytes.extend_from_slice(&source.as_bytes()[range.start * stride..range.end * stride]); }
        self.checkpoint.consumed.insert(name.into());
        Ok(Tensor::from_data(TensorData::from_bytes_vec(bytes, local, source.dtype), (self.device, source.dtype)))
    }
    pub fn norm(&mut self, prefix: &str, width: usize, epsilon: f64) -> Result<Norm<B>, HuggingFaceLoadError> {
        Ok(Norm { weight: self.tensor(&format!("{prefix}.weight"), [width])?, epsilon })
    }
    pub fn projection(&mut self, prefix: &str, input: usize, output: usize, bias: bool, part: Part)
        -> Result<Projection<B>, HuggingFaceLoadError> {
        part.validate(input, output)?;
        if let Some(fp8) = &mut self.fp8 {
            if fp8.is_encoded(&format!("{prefix}.weight")) {
                let mut projections = Vec::with_capacity(part.output.len());
                for rows in &part.output {
                    projections.push(fp8.projection(self.checkpoint, prefix, input, output, part.input.clone(), rows.clone(),
                        bias, part.bias, self.dtype, self.device)?);
                }
                if projections.len() == 1 { return Ok(projections.remove(0)); }
                return Ok(Projection::Packed(std::sync::Arc::new(move |input| {
                    let input = Tensor::<B, 3>::from_primitive(TensorPrimitive::Float(input));
                    Tensor::cat(projections.iter().map(|projection| projection.forward(input.clone())).collect(), 2)
                        .into_primitive().tensor()
                })));
            }
        }
        let weight = if self.checkpoint.tensors.contains_key(&format!("{prefix}.qweight")) {
            let config = self.quantization.as_ref().ok_or_else(|| HuggingFaceLoadError(format!("{prefix}: missing AWQ configuration")))?;
            if config.modules_to_not_convert.as_ref().is_some_and(|names| names.iter().any(
                |name| prefix == name || prefix.ends_with(&format!(".{name}")) || prefix.split('.').any(|part| part == name))) {
                return Err(HuggingFaceLoadError(format!("{prefix}: packed weights in an excluded module")));
            }
            if let Some(load) = self.packed {
                let value = load(self.checkpoint, prefix, input, output, config, self.dtype, bias, &part, self.device)?;
                self.packed_awq += 1;
                return Ok(value);
            }
            let value = super::awq::dense::<B>(self.checkpoint, prefix, input, output, config, self.dtype, &part, self.device)?;
            self.dense_awq += 1;
            value
        } else {
            let name = format!("{prefix}.weight");
            let source = floating_data(self.checkpoint, &name, [output, input])?;
            let size = source.dtype.size();
            let mut bytes = Vec::with_capacity(part.outputs() * part.input.len() * size);
            for range in &part.output {
                for row in range.clone() {
                    bytes.extend_from_slice(&source.as_bytes()[(row * input + part.input.start) * size..(row * input + part.input.end) * size]);
                }
            }
            self.checkpoint.consumed.insert(name);
            Tensor::from_data(TensorData::from_bytes_vec(bytes, [part.outputs(), part.input.len()], source.dtype), (self.device, source.dtype))
        };
        let bias = if bias {
            let data = bias_data(self.checkpoint, prefix, output, &part, weight.dtype())?;
            part.bias.then(|| Param::from_tensor(Tensor::from_data(data, (self.device, weight.dtype()))))
        } else { None };
        Ok(Projection::Dense(Linear { weight: Param::from_tensor(weight.transpose()), bias }))
    }
}

impl<B: Backend> super::super::moe::Weights<B> for Weights<'_, B> {
    fn checkpoint(&mut self) -> &mut Checkpoint { self.checkpoint }
    fn device(&self) -> B::Device { self.device.clone() }
    fn fp8_block(&self) -> Option<[usize; 2]> { self.fp8.as_ref().map(|weights| weights.config.block) }
    fn fp8_projection(&mut self, weight: TensorData, scales: TensorData) -> Result<Projection<B>, HuggingFaceLoadError> {
        self.fp8.as_mut().ok_or_else(|| HuggingFaceLoadError("missing FP8 expert backend".into()))?
            .from_data(weight, scales, None, self.dtype, self.device)
    }
    fn mlp_projection(&mut self, prefix: &str, input: usize, output: usize,
        selection: Option<(usize, Range<usize>)>) -> Result<Projection<B>, HuggingFaceLoadError> {
        let part = match selection {
            None => Part::full(input, output),
            Some((0, range)) => Part { input: 0..input, output: vec![range], bias: true },
            Some((1, range)) => Part { input: range, output: vec![0..output], bias: true },
            _ => return Err(HuggingFaceLoadError("invalid Qwen3.5 MLP shard axis".into())),
        };
        self.projection(prefix, input, output, false, part)
    }
}
