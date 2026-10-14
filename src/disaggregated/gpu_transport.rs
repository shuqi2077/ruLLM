use super::error;
use crate::GenerationError;
use ruda_driver_cuda::ucx_bridge as native;
use std::{ffi::CString,path::PathBuf,sync::Arc};

#[derive(Clone,Debug)]
pub struct GpuKvConfig {
    /// ruCCL torch-bridge built with `--features ucx`; only its native C ABI is
    /// loaded. Neither PyTorch nor LibTorch is used by this serving path.
    pub library:PathBuf,
    pub transports:String,
    pub timeout_ms:u64,
}
impl GpuKvConfig {
    pub fn validate(&self)->Result<(),GenerationError> {
        if self.library.as_os_str().is_empty() || self.transports.is_empty() || self.timeout_ms==0 {
            return Err(error("GPU KV handoff needs a bridge library, UCX transports and positive timeout"));
        }
        CString::new(self.transports.as_str()).map_err(error)?;Ok(())
    }
}

pub(super) struct UcxApi {inner:Arc<native::UcxApi>}
impl UcxApi {
    pub fn load(config:&GpuKvConfig)->Result<Arc<Self>,GenerationError> {
        config.validate()?;
        Ok(Arc::new(Self{inner:native::UcxApi::load(&config.library).map_err(error)?}))
    }
}

pub(super) struct UcxEndpoint {inner:native::UcxEndpoint}
impl UcxEndpoint {
    pub fn new(api:Arc<UcxApi>,device:i32,peers:usize,config:&GpuKvConfig)->Result<Self,GenerationError> {
        native::UcxEndpoint::new(api.inner.clone(),device,peers,&config.transports,config.timeout_ms)
            .map(|inner|Self{inner}).map_err(error)
    }
    pub fn address(&self)->Result<Vec<u8>,GenerationError> {self.inner.address().map_err(error)}
    pub fn connect(&self,peer:usize,address:&[u8])->Result<(),GenerationError> {
        self.inner.connect(peer,address).map_err(error)
    }
    pub fn transfer(&self,peer:usize,receive:bool,tag:u64,pointer:u64,bytes:usize)->Result<(),GenerationError> {
        self.inner.transfer(peer,receive,tag,pointer,bytes).map_err(error)
    }
}
