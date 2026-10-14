use crate::prefix_cache::{PrefilledPrompt, PrefixCacheConfig, TieredPrefixCache};
use crate::{
    CausalModel, CausalModelLimits, GenerationCacheContinuation, GenerationError,
    GreedyGenerationConfig,
};
use ruda_tensor::api::backend::Backend;
use serde::{Deserialize, Serialize};
use sha2::{Digest, Sha256};
use std::{
    io::{Read, Write},
    net::{TcpListener, TcpStream},
};

const MAGIC: &[u8; 8] = b"RUDAKV01";

mod batch;
#[cfg(feature="gpu-kv")]
mod gpu_transport;
#[cfg(feature="gpu-kv")]
mod gpu;
#[cfg(feature="gpu-kv")]
pub use gpu_transport::GpuKvConfig;
#[cfg(feature="gpu-kv")]
pub use gpu::run_gpu_batch_prefill_service;
#[cfg(feature="gpu-kv")]
pub(crate) use gpu::GpuRemoteNetwork;
pub use batch::run_batch_prefill_service;
#[cfg(feature="serving")]
pub(crate) use batch::RemoteBatchNetwork;

#[derive(Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
struct PrefillRequest {
    version: u32,
    model_id: String,
    limits: CausalModelLimits,
    tokens: Vec<i32>,
}

#[derive(Clone, Debug)]
pub struct RemotePrefill {
    pub address: String,
}

fn error(error: impl std::fmt::Display) -> GenerationError {
    GenerationError(error.to_string())
}

fn write_frame(stream: &mut TcpStream, payload: &[u8]) -> Result<(), GenerationError> {
    stream.write_all(MAGIC).map_err(error)?;
    stream
        .write_all(&(payload.len() as u64).to_be_bytes())
        .map_err(error)?;
    stream.write_all(&Sha256::digest(payload)).map_err(error)?;
    stream.write_all(payload).map_err(error)?;
    stream.flush().map_err(error)
}

fn read_frame(stream: &mut TcpStream) -> Result<Vec<u8>, GenerationError> {
    let mut magic = [0u8; 8];
    let mut length = [0u8; 8];
    let mut digest = [0u8; 32];
    stream.read_exact(&mut magic).map_err(error)?;
    if &magic != MAGIC {
        return Err(error("incompatible native KV handoff protocol"));
    }
    stream.read_exact(&mut length).map_err(error)?;
    stream.read_exact(&mut digest).map_err(error)?;
    let length = usize::try_from(u64::from_be_bytes(length)).map_err(error)?;
    let mut payload = Vec::new();
    payload.try_reserve_exact(length).map_err(error)?;
    payload.resize(length, 0);
    stream.read_exact(&mut payload).map_err(error)?;
    if Sha256::digest(&payload)[..] != digest {
        return Err(error("native KV handoff payload digest differs"));
    }
    Ok(payload)
}

impl RemotePrefill {
    pub fn prefill<B: Backend, M: CausalModel<B>>(
        &self,
        model: &M,
        model_id: &str,
        limits: CausalModelLimits,
        tokens: &[i32],
        device: &B::Device,
    ) -> Result<PrefilledPrompt<B, M>, GenerationError>
    where
        M::Cache: Clone + GenerationCacheContinuation<B>,
    {
        crate::generation::validate_generation(
            &limits,
            tokens,
            &GreedyGenerationConfig {
                max_new_tokens: 0,
                eos_token_ids: Vec::new(),
            },
        )?;
        let request = rmp_serde::to_vec(&PrefillRequest {
            version: 1,
            model_id: model_id.into(),
            limits,
            tokens: tokens.to_vec(),
        })
        .map_err(error)?;
        let mut stream = TcpStream::connect(&self.address).map_err(error)?;
        stream.set_nodelay(true).map_err(error)?;
        write_frame(&mut stream, &request)?;
        let response = read_frame(&mut stream)?;
        match response.split_first() {
            Some((0, payload)) => {
                PrefilledPrompt::import(payload, model, model_id, limits, tokens, device)
            }
            Some((1, payload)) => Err(error(String::from_utf8_lossy(payload))),
            _ => Err(error("invalid native prefill response")),
        }
    }
}

pub fn run_prefill_service<B: Backend, M: CausalModel<B>>(
    model: M,
    device: B::Device,
    limits: CausalModelLimits,
    model_id: String,
    cache_config: PrefixCacheConfig,
    address: &str,
    on_ready: impl FnOnce() -> bool,
) -> Result<(), GenerationError>
where
    M::Cache: Clone + GenerationCacheContinuation<B>,
{
    let mut cache = TieredPrefixCache::<B, M>::new(model_id.clone(), limits, cache_config)?;
    let listener = TcpListener::bind(address).map_err(error)?;
    if !on_ready() {
        return Ok(());
    }
    for incoming in listener.incoming() {
        let mut stream = match incoming {
            Ok(stream) => stream,
            Err(e) => {
                eprintln!("prefill connection: {e}");
                continue;
            }
        };
        let result = (|| {
            let payload = read_frame(&mut stream)?;
            let request: PrefillRequest = rmp_serde::from_slice(&payload).map_err(error)?;
            if request.version != 1 || request.model_id != model_id || request.limits != limits {
                return Err(error("prefill model identity, version or limits differ"));
            }
            let prefix = cache.prefill(&model, &request.tokens, &device)?;
            prefix.export(&device)
        })();
        let mut response = Vec::new();
        match result {
            Ok(payload) => {
                response.push(0);
                response.extend_from_slice(&payload);
            }
            Err(e) => {
                response.push(1);
                response.extend_from_slice(e.to_string().as_bytes());
            }
        }
        if let Err(e) = write_frame(&mut stream, &response) {
            eprintln!("prefill handoff: {e}");
        }
    }
    Ok(())
}
