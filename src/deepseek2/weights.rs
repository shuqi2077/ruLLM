use super::*;
use std::ops::{Deref, DerefMut};

pub struct MlaFp8Projection<B: Backend> {
    forward: Arc<dyn Fn(Tensor<B, 3>) -> Tensor<B, 3> + Send + Sync>,
    materialize: Arc<dyn Fn() -> Tensor<B, 2> + Send + Sync>,
}

pub trait MlaFp8Backend: Backend {
    fn mla_fp8_projection(weight: TensorData, scales: TensorData, bias: Option<TensorData>,
        block: [usize; 2], dtype: DType, device: &Self::Device) -> Result<MlaFp8Projection<Self>, HuggingFaceLoadError>;
}

pub(super) type Loader<B: Backend> = fn(TensorData, TensorData, Option<TensorData>, [usize; 2], DType,
    &<B as ruda_tensor::BackendTypes>::Device) -> Result<Projection<B>, HuggingFaceLoadError>;

pub(super) fn native_projection<B: MlaFp8Backend>(weight: TensorData, scales: TensorData, bias: Option<TensorData>,
    block: [usize; 2], dtype: DType, device: &B::Device) -> Result<Projection<B>, HuggingFaceLoadError> {
    let layer = B::mla_fp8_projection(weight, scales, bias, block, dtype, device)?;
    Ok(Projection::Packed { forward: layer.forward, materialize: layer.materialize, dtype })
}

pub fn load_huggingface_mla<B: MlaFp8Backend>(directory: impl AsRef<Path>, device: &B::Device)
    -> Result<LoadedDeepseekV2<B>, HuggingFaceLoadError> {
    load_rank_with(directory.as_ref(), device, None, None, Some(native_projection::<B>))
}

pub(super) fn block_config(value: Option<&serde_json::Value>) -> Result<Option<[usize; 2]>, HuggingFaceLoadError> {
    let Some(value) = value.filter(|value| !value.is_null()) else { return Ok(None); };
    let fail = || HuggingFaceLoadError("MLA FP8 requires dynamic E4M3 and explicit block dimensions".into());
    if value.get("quant_method").and_then(|v| v.as_str()) != Some("fp8")
        || value.get("fmt").and_then(|v| v.as_str()) != Some("e4m3")
        || value.get("activation_scheme").and_then(|v| v.as_str()) != Some("dynamic")
        || value.get("scale_fmt").is_some_and(|v| !v.is_null() && v.as_str() != Some("float32")) {
        return Err(fail());
    }
    let array = value.get("weight_block_size").and_then(|v| v.as_array()).filter(|v| v.len() == 2).ok_or_else(fail)?;
    let first = array[0].as_u64().and_then(|n| usize::try_from(n).ok()).ok_or_else(fail)?;
    let second = array[1].as_u64().and_then(|n| usize::try_from(n).ok()).ok_or_else(fail)?;
    if first == 0 || second == 0 || !first.is_multiple_of(16) || !second.is_multiple_of(32)
        || first > u32::MAX as usize || second > u32::MAX as usize { return Err(fail()); }
    Ok(Some([first, second]))
}

pub(super) struct Weights<B: Backend> {
    checkpoint: Checkpoint,
    encoded: BTreeSet<String>,
    block: Option<[usize; 2]>,
    loader: Option<Loader<B>>,
}
impl<B: Backend> Deref for Weights<B> { type Target = Checkpoint; fn deref(&self) -> &Checkpoint { &self.checkpoint } }
impl<B: Backend> DerefMut for Weights<B> { fn deref_mut(&mut self) -> &mut Checkpoint { &mut self.checkpoint } }

impl<B: Backend> Weights<B> {
    pub(super) fn open(directory: &Path, config: &DeepseekV2Config, loader: Option<Loader<B>>) -> Result<Self, HuggingFaceLoadError> {
        let block = block_config(config.quantization_config.as_ref())?;
        let (checkpoint, encoded) = if block.is_some() { Checkpoint::open_with_fp8(directory)? }
            else { (Checkpoint::open(directory)?, BTreeSet::new()) };
        if !encoded.is_empty() && loader.is_none() {
            return Err(HuggingFaceLoadError("encoded FP8 MLA weights require load_huggingface_mla with a packed backend".into()));
        }
        Ok(Self { checkpoint, encoded, block, loader })
    }
    pub(super) fn is_fp8(&self, prefix: &str) -> bool { self.encoded.contains(&format!("{prefix}.weight")) }

    fn shapes(&self, prefix: &str, shape: [usize; 2], dtype: DType) -> Result<Vec<String>, HuggingFaceLoadError> {
        let name = format!("{prefix}.weight");
        let weight = self.tensors.get(&name).ok_or_else(|| HuggingFaceLoadError(format!("missing {name}")))?;
        let encoded = self.is_fp8(prefix);
        if weight.shape != shape.into() || weight.dtype != if encoded { DType::U8 } else { dtype } {
            return Err(HuggingFaceLoadError(format!("incompatible projection {name}")));
        }
        let scale_name = format!("{prefix}.weight_scale_inv");
        if !encoded {
            if self.tensors.contains_key(&scale_name) { return Err(HuggingFaceLoadError(format!("orphan FP8 scales {scale_name}"))); }
            return Ok(vec![name]);
        }
        let [bn, bk] = self.block.ok_or_else(|| HuggingFaceLoadError("missing FP8 configuration".into()))?;
        let scale = self.tensors.get(&scale_name).ok_or_else(|| HuggingFaceLoadError(format!("missing {scale_name}")))?;
        if scale.dtype != DType::F32 || scale.shape != [shape[0].div_ceil(bn), shape[1].div_ceil(bk)].into() {
            return Err(HuggingFaceLoadError(format!("incompatible FP8 scale geometry {scale_name}")));
        }
        Ok(vec![name, scale_name])
    }
    pub(super) fn exclude_projection(&self, prefix: &str, shape: [usize; 2], dtype: DType, excluded: &mut BTreeSet<String>)
        -> Result<(), HuggingFaceLoadError> {
        for name in self.shapes(prefix, shape, dtype)? { excluded.insert(name); }
        Ok(())
    }
    pub(super) fn packed_projection(&mut self, prefix: &str, input: usize, output: usize, bias: bool,
        device: &B::Device, dtype: DType) -> Result<Projection<B>, HuggingFaceLoadError> {
        let names = self.shapes(prefix, [output, input], dtype)?;
        if names.len() != 2 { return Err(HuggingFaceLoadError("expected encoded FP8 projection".into())); }
        let read = |name: &str| self.tensors[name].to_data().map_err(|e| HuggingFaceLoadError(format!("{name}: {e}")));
        let weight = read(&names[0])?;
        let scales = read(&names[1])?;
        if scales.iter::<f32>().any(|v| !v.is_finite() || v < 0.0) { return Err(HuggingFaceLoadError("invalid FP8 weight scales".into())); }
        let bias_name = format!("{prefix}.bias");
        let bias = if bias {
            let snapshot = self.tensors.get(&bias_name).ok_or_else(|| HuggingFaceLoadError(format!("missing {bias_name}")))?;
            if snapshot.shape != [output].into() || snapshot.dtype != dtype { return Err(HuggingFaceLoadError(format!("invalid {bias_name}"))); }
            Some(read(&bias_name)?)
        } else {
            if self.tensors.contains_key(&bias_name) { return Err(HuggingFaceLoadError(format!("unexpected {bias_name}"))); }
            None
        };
        let has_bias = bias.is_some();
        let loader = self.loader.ok_or_else(|| HuggingFaceLoadError("no packed FP8 backend".into()))?;
        let layer = loader(weight, scales, bias, self.block.expect("validated FP8 config"), dtype, device)?;
        for name in names { self.consumed.insert(name); }
        if has_bias { self.consumed.insert(bias_name); }
        Ok(layer)
    }
}

impl<R, F, I, BT> MlaFp8Backend for ruda_tensor_device::DeviceBackend<R, F, I, BT>
where R: ruda_tensor_device::DeviceRuntime, R::Server: ruda::runtime::server::ComputeServer,
    R::Device: ruda_tensor::DeviceOps, F: ruda_tensor_device::FloatElement,
    I: ruda_tensor_device::IntElement, BT: ruda_tensor_device::BoolElement {
    fn mla_fp8_projection(weight: TensorData, scales: TensorData, bias: Option<TensorData>,
        block: [usize; 2], dtype: DType, device: &Self::Device) -> Result<MlaFp8Projection<Self>, HuggingFaceLoadError> {
        use ruda_kernel::{tensor::transfer::from_data, quantization::float8_block};
        use ruda_tensor::api::TensorPrimitive;
        let weight = from_data::<R>(weight, device);
        let scales = from_data::<R>(scales, device);
        let bias = bias.map(|data| Tensor::<Self, 1>::from_data(data, (device, dtype)));
        let packed = weight.clone(); let affine = scales.clone();
        let materialize = Arc::new(move || Tensor::<Self, 2>::from_primitive(TensorPrimitive::Float(
            float8_block::dequantize(&packed, &affine, block))).cast(dtype));
        let forward = Arc::new(move |input: Tensor<Self, 3>| {
            let [batch, sequence, _] = input.dims(); let output = weight.meta.shape()[0];
            let value = float8_block::linear(input.into_primitive().tensor(), &weight, &scales, block);
            let value = Tensor::<Self, 2>::from_primitive(TensorPrimitive::Float(value)).cast(dtype).reshape([batch, sequence, output]);
            match &bias { Some(bias) => value + bias.clone().unsqueeze::<3>(), None => value }
        });
        Ok(MlaFp8Projection { forward, materialize })
    }
}

#[cfg(feature = "libtorch")]
impl<E: ruda_tensor_tch::TchElement> MlaFp8Backend for ruda_tensor_tch::LibTorch<E> {
    fn mla_fp8_projection(_: TensorData, _: TensorData, _: Option<TensorData>, _: [usize; 2], _: DType, _: &Self::Device)
        -> Result<MlaFp8Projection<Self>, HuggingFaceLoadError> {
        Err(HuggingFaceLoadError("packed MLA FP8 requires the native backend; LibTorch loader accepts floating weights".into()))
    }
}
