use super::*;
use ruda_model::module::ParamId;
use ruda_store::TensorSnapshotError;
use ruda_tensor::api::TensorData;
use std::{fs::File, io::{Read, Seek, SeekFrom}, rc::Rc};

#[derive(serde::Deserialize)]
struct Entry { dtype: String, shape: Vec<usize>, data_offsets: [u64; 2] }

impl Checkpoint {
    pub(crate) fn open_with_fp8(directory: &Path) -> Result<(Self, BTreeSet<String>), HuggingFaceLoadError> {
        let directory = canonical_directory(directory)?;
        let files = discover_weight_files(&directory)?;
        let index = if directory.join(SINGLE_WEIGHTS_FILE).exists() { None }
            else { Some(read_json::<SafetensorsIndex>(&directory.join(WEIGHTS_INDEX_FILE))?) };
        let mut tensors = BTreeMap::new();
        let mut encoded = BTreeSet::new();
        for path in &files {
            let mut file = File::open(path).map_err(io_error)?;
            let length = file.metadata().map_err(io_error)?.len();
            let mut word = [0u8; 8];
            file.read_exact(&mut word).map_err(io_error)?;
            let header_length = u64::from_le_bytes(word);
            let start = header_length.checked_add(8).filter(|&n| n <= length)
                .ok_or_else(|| HuggingFaceLoadError("invalid safetensors header length".into()))?;
            let header: BTreeMap<String, serde_json::Value> = serde_json::from_reader(file.take(header_length))
                .map_err(|e| HuggingFaceLoadError(e.to_string()))?;
            let mut intervals = Vec::new();
            for (name, value) in header {
                if name == "__metadata__" { continue; }
                let entry: Entry = serde_json::from_value(value).map_err(|e| HuggingFaceLoadError(e.to_string()))?;
                let dtype = match entry.dtype.as_str() {
                    "F32" => DType::F32, "F16" => DType::F16, "BF16" => DType::BF16,
                    "I32" => DType::I32, "I64" => DType::I64, "U8" => DType::U8,
                    "F8_E4M3" => { encoded.insert(name.clone()); DType::U8 },
                    other => return Err(HuggingFaceLoadError(format!("unsupported MLA checkpoint dtype {other}: {name}"))),
                };
                let count = entry.shape.iter().try_fold(1usize, |n, &d| n.checked_mul(d))
                    .and_then(|n| n.checked_mul(dtype.size()))
                    .ok_or_else(|| HuggingFaceLoadError(format!("tensor size overflow: {name}")))?;
                let [begin, end] = entry.data_offsets;
                if begin > end || end > length - start || end - begin != count as u64 {
                    return Err(HuggingFaceLoadError(format!("invalid safetensors extent: {name}")));
                }
                intervals.push((begin, end));
                if let Some(index) = &index {
                    let shard = index.weight_map.get(&name)
                        .ok_or_else(|| HuggingFaceLoadError(format!("unindexed tensor {name}")))?;
                    if directory.join(shard).canonicalize().map_err(io_error)? != *path {
                        return Err(HuggingFaceLoadError(format!("wrong tensor shard: {name}")));
                    }
                }
                let source = path.clone();
                let shape = entry.shape;
                let stored_shape = shape.clone();
                let snapshot = TensorSnapshot::from_closure(Rc::new(move || {
                    let mut file = File::open(&source).map_err(|e| TensorSnapshotError::IoError(e.to_string()))?;
                    file.seek(SeekFrom::Start(start + begin)).map_err(|e| TensorSnapshotError::IoError(e.to_string()))?;
                    let mut bytes = vec![0u8; count];
                    file.read_exact(&mut bytes).map_err(|e| TensorSnapshotError::IoError(e.to_string()))?;
                    Ok(TensorData::from_bytes_vec(bytes, shape.clone(), dtype))
                }), dtype, stored_shape.into(), name.split('.').map(str::to_owned).collect(), vec![], ParamId::new());
                if tensors.insert(name.clone(), snapshot).is_some() {
                    return Err(HuggingFaceLoadError(format!("duplicate tensor {name}")));
                }
            }
            intervals.sort_unstable();
            let mut cursor = 0;
            for (begin, end) in intervals {
                if begin != cursor { return Err(HuggingFaceLoadError("overlapping or incomplete safetensors payload".into())); }
                cursor = end;
            }
            if cursor != length - start { return Err(HuggingFaceLoadError("unindexed safetensors payload".into())); }
        }
        if let Some(index) = index {
            for name in index.weight_map.keys() {
                if !tensors.contains_key(name) { return Err(HuggingFaceLoadError(format!("missing indexed tensor {name}"))); }
            }
        }
        Ok((Self { files, tensors, consumed: BTreeSet::new() }, encoded))
    }
}
