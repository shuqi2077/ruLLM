use super::*;
use crate::{HuggingFaceLoadError, huggingface::checkpoint::{Checkpoint, StackedMatrixSource}};
use std::{collections::BTreeSet, ops::Range, path::Path, sync::Arc};

pub(super) type Loader<B: Backend> = fn(TensorData, TensorData, Option<TensorData>, [usize; 2], DType,
    &<B as ruda_tensor::BackendTypes>::Device) -> Result<Projection<B>, HuggingFaceLoadError>;

#[derive(Clone)]
pub(super) struct Configuration {
    pub block: [usize; 2],
    excluded: Vec<String>,
}

pub(super) struct Weights<B: Backend> {
    pub config: Configuration,
    pub encoded: BTreeSet<String>,
    pub loader: Loader<B>,
    pub projections: usize,
}

pub(super) fn configuration(raw: &serde_json::Value) -> Result<Option<Configuration>, HuggingFaceLoadError> {
    let top = raw.get("quantization_config").filter(|v| !v.is_null());
    let nested = raw.get("text_config").and_then(|v| v.get("quantization_config")).filter(|v| !v.is_null());
    if top.is_some() && nested.is_some() && top != nested { return Err(HuggingFaceLoadError("conflicting Qwen3.5 quantization configs".into())); }
    let Some(value) = top.or(nested).filter(|v| v.get("quant_method").and_then(|v| v.as_str()) == Some("fp8")) else { return Ok(None); };
    let fail = || HuggingFaceLoadError("Qwen3.5 FP8 requires dynamic E4M3FN weights, FP32 block scales and explicit block dimensions".into());
    if value.get("activation_scheme").and_then(|v| v.as_str()) != Some("dynamic")
        || value.get("fmt").is_some_and(|v| !v.is_null() && v.as_str() != Some("e4m3"))
        || value.get("scale_fmt").is_some_and(|v| !v.is_null() && v.as_str() != Some("float32"))
        || ["weight_per_tensor", "act_per_tensor"].iter().any(|&key| value.get(key).is_some_and(|v| v.as_bool() != Some(false))) {
        return Err(fail());
    }
    let block: [usize; 2] = serde_json::from_value(value.get("weight_block_size").cloned().ok_or_else(fail)?)
        .map_err(|_| fail())?;
    if block[0] == 0 || block[1] == 0 || !block[0].is_multiple_of(16) || !block[1].is_multiple_of(32)
        || block.iter().any(|&n| n > u32::MAX as usize) { return Err(fail()); }
    let excluded = value.get("modules_to_not_convert").filter(|v| !v.is_null()).cloned()
        .map(serde_json::from_value).transpose().map_err(|_| fail())?.unwrap_or_default();
    Ok(Some(Configuration { block, excluded }))
}

pub(super) fn open<B: Backend>(directory: &Path, loader: Option<Loader<B>>)
    -> Result<(Checkpoint, Option<Weights<B>>), HuggingFaceLoadError> {
    let raw = serde_json::from_slice(&std::fs::read(directory.join("config.json")).map_err(|e| HuggingFaceLoadError(e.to_string()))?)
        .map_err(|e| HuggingFaceLoadError(e.to_string()))?;
    let Some(config) = configuration(&raw)? else { return Ok((Checkpoint::open(directory)?, None)); };
    let loader = loader.ok_or_else(|| HuggingFaceLoadError("FP8 Qwen3.5 requires the native packed loader".into()))?;
    let (checkpoint, encoded) = Checkpoint::open_with_fp8(directory)?;
    for name in &encoded {
        let prefix = name.strip_suffix(".weight").unwrap_or(name);
        if config.excluded.iter().any(|p| prefix == p || prefix.ends_with(&format!(".{p}")) || prefix.split('.').any(|part| part == p)) {
            return Err(HuggingFaceLoadError(format!("encoded FP8 weight in excluded module {name}")));
        }
    }
    Ok((checkpoint, Some(Weights { config, encoded, loader, projections: 0 })))
}

fn data(checkpoint: &Checkpoint, name: &str, shape: [usize; 2], dtype: DType) -> Result<TensorData, HuggingFaceLoadError> {
    let tensor = checkpoint.tensors.get(name).ok_or_else(|| HuggingFaceLoadError(format!("missing {name}")))?;
    if tensor.shape != shape.into() || tensor.dtype != dtype { return Err(HuggingFaceLoadError(format!("FP8 tensor shape/dtype differs: {name}"))); }
    tensor.to_data().map_err(|e| HuggingFaceLoadError(e.to_string()))
}

fn slice(data: TensorData, rows: Range<usize>, columns: Range<usize>) -> TensorData {
    let size = data.dtype.size(); let width = data.shape[1];
    let mut bytes = Vec::with_capacity(rows.len() * columns.len() * size);
    for row in rows.clone() { bytes.extend_from_slice(&data.as_bytes()[(row * width + columns.start) * size..(row * width + columns.end) * size]); }
    TensorData::from_bytes_vec(bytes, [rows.len(), columns.len()], data.dtype)
}

impl<B: Backend> Weights<B> {
    pub fn is_encoded(&self, name: &str) -> bool { self.encoded.contains(name) }
    pub fn projection(&mut self, checkpoint: &mut Checkpoint, prefix: &str, input: usize, output: usize,
        columns: Range<usize>, rows: Range<usize>, bias: bool, include_bias: bool, dtype: DType, device: &B::Device)
        -> Result<Projection<B>, HuggingFaceLoadError> {
        let [bn, bk] = self.config.block;
        check_selection(&rows, &columns, output, input, [bn, bk])?;
        let name = format!("{prefix}.weight"); let scale_name = format!("{prefix}.weight_scale_inv");
        if !self.is_encoded(&name) { return Err(HuggingFaceLoadError(format!("{name}: expected encoded E4M3FN storage"))); }
        let weight = slice(data(checkpoint, &name, [output, input], DType::U8)?, rows.clone(), columns.clone());
        let scales = slice(data(checkpoint, &scale_name, [output.div_ceil(bn), input.div_ceil(bk)], DType::F32)?,
            rows.start / bn..rows.end.div_ceil(bn), columns.start / bk..columns.end.div_ceil(bk));
        let bias_name = format!("{prefix}.bias");
        let bias = if bias {
            let value = checkpoint.tensors.get(&bias_name).ok_or_else(|| HuggingFaceLoadError(format!("missing {bias_name}")))?;
            if value.shape != [output].into() || value.dtype != dtype { return Err(HuggingFaceLoadError(format!("invalid {bias_name}"))); }
            let source = value.to_data().map_err(|e| HuggingFaceLoadError(e.to_string()))?;
            checkpoint.consumed.insert(bias_name);
            include_bias.then(|| TensorData::from_bytes_vec(source.as_bytes()[rows.start * dtype.size()..rows.end * dtype.size()].to_vec(), [rows.len()], dtype))
        } else {
            if checkpoint.tensors.contains_key(&bias_name) { return Err(HuggingFaceLoadError(format!("unexpected {bias_name}"))); }
            None
        };
        let result = self.from_data(weight, scales, bias, dtype, device)?;
        checkpoint.consumed.extend([name, scale_name]);
        Ok(result)
    }
    pub fn from_data(&mut self, weight: TensorData, scales: TensorData, bias: Option<TensorData>, dtype: DType, device: &B::Device)
        -> Result<Projection<B>, HuggingFaceLoadError> {
        if scales.dtype != DType::F32 || scales.iter::<f32>().any(|v| !v.is_finite() || v < 0.0) {
            return Err(HuggingFaceLoadError("invalid FP8 block scales".into()));
        }
        let result = (self.loader)(weight, scales, bias, self.config.block, dtype, device)?;
        self.projections += 1;
        Ok(result)
    }
}

fn check_selection(rows: &Range<usize>, columns: &Range<usize>, output: usize, input: usize, block: [usize; 2])
    -> Result<(), HuggingFaceLoadError> {
    if rows.is_empty() || columns.is_empty() || rows.end > output || columns.end > input
        || rows.start % block[0] != 0 || columns.start % block[1] != 0
        || (columns.end != input && columns.end % block[1] != 0) {
        return Err(HuggingFaceLoadError("FP8 shard must start on source scale-block boundaries".into()));
    }
    Ok(())
}

pub(super) struct Stacked {
    weight: StackedMatrixSource,
    scales: StackedMatrixSource,
    shape: [usize; 3],
    block: [usize; 2],
}
impl Stacked {
    pub fn open(checkpoint: &mut Checkpoint, name: &str, shape: [usize; 3], block: [usize; 2]) -> Result<Self, HuggingFaceLoadError> {
        let weight = checkpoint.stacked_fp8_source(name, shape)?;
        let scales = checkpoint.stacked_matrix_source(&format!("{name}_scale_inv"),
            [shape[0], shape[1].div_ceil(block[0]), shape[2].div_ceil(block[1])])?;
        Ok(Self { weight, scales, shape, block })
    }
    pub fn matrix(&mut self, expert: usize, rows: Range<usize>, columns: Range<usize>) -> Result<(TensorData, TensorData), HuggingFaceLoadError> {
        check_selection(&rows, &columns, self.shape[1], self.shape[2], self.block)?;
        let scales = self.scales.matrix(expert, rows.start / self.block[0]..rows.end.div_ceil(self.block[0]),
            columns.start / self.block[1]..columns.end.div_ceil(self.block[1]))?;
        Ok((self.weight.matrix(expert, rows, columns)?, scales))
    }
}

pub(super) fn native<R, F, I, BT>(weight: TensorData, scales: TensorData, bias: Option<TensorData>,
    block: [usize; 2], dtype: DType, device: &R::Device) -> Result<Projection<DeviceBackend<R,F,I,BT>>, HuggingFaceLoadError>
where R: DeviceRuntime, R::Server: ComputeServer, R::Device: DeviceOps,
    F: FloatElement, I: IntElement, BT: BoolElement {
    use ruda_kernel::{tensor::transfer::from_data, quantization::float8_block};
    let output = weight.shape[0];
    let weight = from_data::<R>(weight, device); let scales = from_data::<R>(scales, device);
    let bias = bias.map(|data| Tensor::<DeviceBackend<R,F,I,BT>, 1>::from_data(data, (device, dtype)));
    Ok(Projection::Packed(Arc::new(move |input: ruda_kernel::tensor::RudaTensor<R>| {
        let mut shape = input.meta.shape().clone();
        let last = shape.len() - 1; shape[last] = output;
        let value = float8_block::linear(input, &weight, &scales, block);
        let value = Tensor::<DeviceBackend<R,F,I,BT>, 2>::from_primitive(TensorPrimitive::Float(value)).cast(dtype);
        let value = match &bias { Some(bias) => value + bias.clone().unsqueeze::<2>(), None => value };
        ruda_kernel::tensor::reshape::reshape(value.into_primitive().tensor(), shape)
    })))
}
