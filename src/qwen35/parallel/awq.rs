use super::*;
use super::weights::{Part, bias_data};
use crate::huggingface::{AwqQuantizationConfig, checkpoint::Checkpoint};
use super::super::quantized_loading::packed_tensor;

struct Source {
    weights: Vec<i32>, zeros: Vec<i32>, scales: TensorData,
    output: usize, group: usize,
}

fn source(checkpoint: &mut Checkpoint, prefix: &str, input: usize, output: usize, config: &AwqQuantizationConfig)
    -> Result<Source, HuggingFaceLoadError> {
    let layout = config.layout(input, output)?;
    for suffix in ["weight", "g_idx"] {
        if checkpoint.tensors.contains_key(&format!("{prefix}.{suffix}")) {
            return Err(HuggingFaceLoadError(format!("{prefix}: ambiguous/unsupported AWQ .{suffix}")));
        }
    }
    let names = [format!("{prefix}.qweight"), format!("{prefix}.qzeros"), format!("{prefix}.scales")];
    let weights = packed_tensor(checkpoint, &names[0], [input, output / 8], true)?.iter::<i32>().collect();
    let zeros = packed_tensor(checkpoint, &names[1], [layout.groups(), output / 8], true)?.iter::<i32>().collect();
    let scales = packed_tensor(checkpoint, &names[2], [layout.groups(), output], false)?;
    if scales.iter::<f32>().any(|v| !v.is_finite() || v < 0.0) {
        return Err(HuggingFaceLoadError(format!("{prefix}: invalid AWQ scale")));
    }
    for name in names { checkpoint.consumed.insert(name); }
    Ok(Source { weights, zeros, scales, output, group: layout.group_size })
}

fn shift(output: usize) -> usize { ((output % 8 % 2) * 4 + (output % 8) / 2) * 4 }
fn code(words: &[i32], row: usize, output: usize, width: usize) -> u32 {
    ((words[row * (width / 8) + output / 8] as u32) >> shift(output)) & 15
}
fn gcd(mut a: usize, mut b: usize) -> usize {
    while b != 0 { let remainder = a % b; a = b; b = remainder; }
    a
}

pub(super) fn dense<B: Backend>(checkpoint: &mut Checkpoint, prefix: &str, input: usize, output: usize,
    config: &AwqQuantizationConfig, dtype: DType, part: &Part, device: &B::Device)
    -> Result<Tensor<B, 2>, HuggingFaceLoadError> {
    let source = source(checkpoint, prefix, input, output, config)?;
    let scales = source.scales.iter::<f32>().collect::<Vec<_>>();
    let mut values = Vec::with_capacity(part.outputs() * part.input.len());
    for range in &part.output {
        for column in range.clone() {
            for row in part.input.clone() {
                let q = code(&source.weights, row, column, output) as f32;
                let z = code(&source.zeros, row / source.group, column, output) as f32;
                values.push((q - z) * scales[(row / source.group) * output + column]);
            }
        }
    }
    Ok(Tensor::from_data(TensorData::new(values, [part.outputs(), part.input.len()]).convert_dtype(dtype), (device, dtype)))
}

pub(super) fn packed<R, F, I, BT>(checkpoint: &mut Checkpoint, prefix: &str, input: usize, output: usize,
    config: &AwqQuantizationConfig, dtype: DType, bias: bool, part: &Part, device: &R::Device)
    -> Result<Projection<DeviceBackend<R, F, I, BT>>, HuggingFaceLoadError>
where R: DeviceRuntime, R::Server: ComputeServer, R::Device: DeviceOps,
    F: FloatElement, I: IntElement, BT: BoolElement {
    use ruda_kernel::tensor::transfer::from_data;
    use ruda_tensor::ops::FloatTensorOps;
    if !matches!(dtype, DType::F16 | DType::BF16 | DType::F32) {
        return Err(HuggingFaceLoadError("packed AWQ TP requires floating activations".into()));
    }
    let source = source(checkpoint, prefix, input, output, config)?;
    let scales = source.scales.convert_dtype(dtype).iter::<f32>().collect::<Vec<_>>();
    if scales.iter().any(|v| !v.is_finite() || *v < 0.0) {
        return Err(HuggingFaceLoadError(format!("{prefix}: invalid AWQ scales after dtype conversion")));
    }
    let columns = part.output.iter().flat_map(|range| range.clone()).collect::<Vec<_>>();
    let logical = columns.len();
    let padded = logical.div_ceil(8) * 8;
    let local_input = part.input.len();
    let group = gcd(gcd(source.group, part.input.start), local_input);
    let groups = local_input / group;
    rublas::tensor_int4::AwqGemmLayout::new(local_input, padded, group)
        .map_err(|e| HuggingFaceLoadError(format!("{prefix}: {e}")))?;
    let mut weights = vec![0i32; local_input * (padded / 8)];
    let mut zeros = vec![0i32; groups * (padded / 8)];
    let mut local_scales = vec![0.0f32; groups * padded];
    for (column, &original) in columns.iter().enumerate() {
        for row in 0..local_input {
            let q = code(&source.weights, part.input.start + row, original, source.output);
            weights[row * (padded / 8) + column / 8] |= (q << shift(column)) as i32;
        }
        for index in 0..groups {
            let original_group = (part.input.start + index * group) / source.group;
            let q = code(&source.zeros, original_group, original, source.output);
            zeros[index * (padded / 8) + column / 8] |= (q << shift(column)) as i32;
            local_scales[index * padded + column] = scales[original_group * source.output + original];
        }
    }
    let bias = if bias {
        let data = bias_data(checkpoint, prefix, output, part, dtype)?;
        if part.bias {
            let mut values = data.iter::<f32>().collect::<Vec<_>>();
            values.resize(padded, 0.0);
            Some(TensorData::new(values, [padded]).convert_dtype(dtype))
        } else { None }
    } else {
        if checkpoint.tensors.contains_key(&format!("{prefix}.bias")) {
            return Err(HuggingFaceLoadError(format!("unexpected AWQ bias {prefix}.bias")));
        }
        None
    };
    let layer = rublas::tensor_int4::AwqGemm::<R>::new(
        from_data(TensorData::new(weights, [local_input, padded / 8]), device),
        from_data(TensorData::new(zeros, [groups, padded / 8]), device),
        from_data(TensorData::new(local_scales, [groups, padded]).convert_dtype(dtype), device),
        bias.map(|data| from_data(data, device)), group,
    ).map_err(|e| HuggingFaceLoadError(format!("{prefix}: {e}")))?;
    Ok(Projection::Packed(std::sync::Arc::new(move |input| {
        let value = layer.forward(input).expect("validated Qwen3.5 packed TP projection");
        if logical == padded { value } else {
            let mut slices = value.meta.shape().iter().map(|&size| (0..size).into())
                .collect::<Vec<ruda_core::tensor::Slice>>();
            *slices.last_mut().expect("projection rank") = (0..logical).into();
            DeviceBackend::<R, F, I, BT>::float_slice(value, &slices)
        }
    })))
}
