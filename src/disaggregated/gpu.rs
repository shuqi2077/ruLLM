use super::*;
use super::gpu_transport::{GpuKvConfig,UcxApi,UcxEndpoint};
use crate::{GpuPrefixBackend,PagedKvCacheConfig,PrefixBatchModel,PrefixTensors,RequestId};
use crate::batch_prefix::{Prefix,PrefixStore};
use crate::prefix_transfer::{PrefixManifest,PrefixTensorReceiver};
use ruda_tensor::api::{DType,Tensor};
use serde::de::DeserializeOwned;
use std::{collections::{BTreeMap,BTreeSet},net::Shutdown,ops::Range,
    sync::{Arc,Mutex,atomic::{AtomicBool,AtomicUsize,Ordering}},time::Duration};

#[derive(Serialize,Deserialize)]
struct Request {version:u32,model_id:String,limits:CausalModelLimits,tokens:Vec<i32>,devices:Vec<i32>,transports:String}
#[derive(Serialize,Deserialize)]
struct Hello {version:u32,model_id:String,limits:CausalModelLimits,tokens:Vec<i32>,manifest:PrefixManifest,
    addresses:Vec<(i32,Vec<u8>)>,logits_dtype:DType}
#[derive(Serialize,Deserialize)]
enum Command {Pull{fragment:usize,start:usize,end:usize,destination:usize,tag:u64},Finish}
#[derive(Serialize,Deserialize)]
enum Reply {Ready(Result<(),String>),Done(Result<(),String>)}

fn write<T:Serialize>(stream:&mut TcpStream,value:&T)->Result<(),GenerationError> {
    write_frame(stream,&rmp_serde::to_vec(value).map_err(error)?)
}
fn read<T:DeserializeOwned>(stream:&mut TcpStream)->Result<T,GenerationError> {
    rmp_serde::from_slice(&read_frame(stream)?).map_err(error)
}
fn configure(stream:&TcpStream,config:&GpuKvConfig)->Result<(),GenerationError> {
    stream.set_nodelay(true).map_err(error)?;
    stream.set_read_timeout(Some(Duration::from_millis(config.timeout_ms))).map_err(error)?;
    stream.set_write_timeout(Some(Duration::from_millis(config.timeout_ms))).map_err(error)
}
fn distinct(devices:&[i32])->bool {
    !devices.is_empty() && devices.iter().all(|&d|d>=0) && devices.iter().collect::<BTreeSet<_>>().len()==devices.len()
}

/// Native device payload handoff. The TCP connection carries only token IDs,
/// schema, UCX addresses and transfer acknowledgements; KV/state/logits use UCX.
pub fn run_gpu_batch_prefill_service<B:GpuPrefixBackend,M:PrefixBatchModel<B>>(
    model:M,device:B::Device,model_id:String,cache_config:PrefixCacheConfig,kv:PagedKvCacheConfig,
    max_batch_tokens:usize,address:&str,gpu:GpuKvConfig,on_ready:impl FnOnce()->bool,
)->Result<(),GenerationError> {
    kv.validate().map_err(error)?;
    if !model.prefix_tensor_transfer() || max_batch_tokens==0 || cache_config.block_tokens==0
        || kv.max_sequence_length!=model.batch_limits().max_sequence_length {
        return Err(error("GPU prefill requires semantic prefix tensors and matching positive batch/KV limits"));
    }
    let api=UcxApi::load(&gpu)?;
    let limits=model.batch_limits();let chunk=cache_config.block_tokens.min(max_batch_tokens);
    let mut store=PrefixStore::<B,M>::new(model_id.clone(),limits,cache_config)?;
    let listener=TcpListener::bind(address).map_err(error)?;
    if !on_ready() {return Ok(());}
    for incoming in listener.incoming() {
        let mut stream=match incoming {Ok(stream)=>stream,Err(e)=>{eprintln!("GPU prefill connection: {e}");continue;}};
        let result=(||->Result<(),GenerationError> {
            configure(&stream,&gpu)?;
            let request:Request=read(&mut stream)?;
            let prepared=(|| {
                if request.version!=1 || request.model_id!=model_id || request.limits!=limits
                    || request.transports!=gpu.transports || !distinct(&request.devices) {
                    return Err(error("GPU prefill identity, limits, transport or destination topology differs"));
                }
                let prefix=super::batch::prefill(&model,&device,&request.tokens,kv,chunk,&mut store)?;
                model.sync_batch(&device)?;
                let mut bundle=PrefixTensors::new("serving-gpu-handoff-v1",request.tokens.len());
                bundle.add_child("cache",model.export_prefix_tensors(&prefix.state,Default::default())?)?;
                let logits_dtype=prefix.logits.dtype();
                bundle.push("logits",prefix.logits.unsqueeze_dim::<4>(1),(1,0..1))?;
                let mut tensors=Vec::new();let manifest=bundle.device_manifest(&mut tensors,&B::prefix_device_ordinal)?;
                let ordinals=tensors.iter().map(|t|B::prefix_device_ordinal(&t.device())).collect::<Result<BTreeSet<_>,_>>()?;
                let endpoints=ordinals.into_iter().map(|ordinal|Ok((ordinal,UcxEndpoint::new(api.clone(),ordinal,request.devices.len(),&gpu)?)))
                    .collect::<Result<BTreeMap<_,_>,GenerationError>>()?;
                let addresses=endpoints.iter().map(|(&ordinal,endpoint)|Ok((ordinal,endpoint.address()?))).collect::<Result<_,GenerationError>>()?;
                let hello=Hello{version:1,model_id:model_id.clone(),limits,tokens:request.tokens.clone(),manifest,addresses,logits_dtype};
                Ok((hello,tensors,endpoints))
            })();
            let (hello,tensors,endpoints)=match prepared {
                Ok(value)=>value,
                Err(e)=>{write(&mut stream,&Result::<Hello,String>::Err(e.to_string()))?;return Err(e);},
            };
            write(&mut stream,&Result::<&Hello,String>::Ok(&hello))?;
            let peers:Vec<Vec<u8>>=read(&mut stream)?;
            let connected=(||->Result<(),GenerationError> {
                if peers.len()!=request.devices.len() {return Err(error("GPU KV destination endpoint count differs"));}
                for endpoint in endpoints.values() {for (peer,address) in peers.iter().enumerate() {endpoint.connect(peer,address)?;}}
                Ok(())
            })();
            write(&mut stream,&connected.as_ref().map(|_|()).map_err(ToString::to_string))?;connected?;
            let mut last_tag=0u64;
            loop {
                match read::<Command>(&mut stream)? {
                    Command::Finish=>{write(&mut stream,&Reply::Done(Ok(())))?;break;},
                    Command::Pull{fragment,start,end,destination,tag}=>{
                        let mut announced=false;
                        let result=(||->Result<(),GenerationError> {
                            let tensor=tensors.get(fragment).ok_or_else(||error("GPU prefix fragment index differs"))?;
                            let [batch,heads,rows,width]=tensor.dims();
                            if batch!=1 || start>=end || end>heads || destination>=peers.len() || tag<=last_tag {
                                return Err(error("GPU prefix pull range, destination or sequence differs"));
                            }
                            let producer=B::prefix_device_ordinal(&tensor.device())?;
                            let selected=tensor.clone().slice([0..1,start..end,0..rows,0..width]);
                            B::with_prefix_pointer(selected,|pointer,bytes| {
                                write(&mut stream,&Reply::Ready(Ok(())))?;announced=true;
                                endpoints[&producer].transfer(destination,false,tag,pointer,bytes)
                            })?;
                            last_tag=tag;Ok(())
                        })();
                        let status=result.as_ref().map(|_|()).map_err(ToString::to_string);
                        write(&mut stream,&if announced {Reply::Done(status)} else {Reply::Ready(status)})?;
                        result?;
                    },
                }
            }
            // All acknowledged UCX transfers have completed before `tensors`
            // and their model-owned GPU allocations can be released.
            Ok(())
        })();
        if let Err(e)=result {eprintln!("GPU prefill handoff: {e}");}
    }
    Ok(())
}

struct SessionState {stream:TcpStream,next_tag:u64,finished:bool}
struct Session {
    state:Mutex<SessionState>,devices:Vec<i32>,producers:Vec<i32>,endpoints:Vec<UcxEndpoint>,
    shutdown:TcpStream,
}
impl Session {
    fn finish(&self)->Result<(),GenerationError> {
        let mut state=self.state.lock().map_err(error)?;
        if state.finished {return Ok(());}
        write(&mut state.stream,&Command::Finish)?;
        match read::<Reply>(&mut state.stream)? {Reply::Done(result)=>result.map_err(error)?,_=>return Err(error("GPU prefix final acknowledgement differs"))}
        state.finished=true;Ok(())
    }
}
impl<B:GpuPrefixBackend> PrefixTensorReceiver<B> for Session {
    fn receive(&self,fragment:usize,producer:i32,heads:Range<usize>,shape:[usize;4],dtype:DType,device:&B::Device)
        ->Result<Tensor<B,4>,GenerationError> {
        let ordinal=B::prefix_device_ordinal(device)?;
        let destination=self.devices.iter().position(|&id|id==ordinal).ok_or_else(||error("GPU prefix destination device is outside its endpoint group"))?;
        let source=self.producers.iter().position(|&id|id==producer).ok_or_else(||error("GPU prefix producer is outside its endpoint group"))?;
        let output=Tensor::<B,4>::empty(shape,(device,dtype));
        B::with_prefix_pointer(output,|pointer,bytes| {
            let mut state=self.state.lock().map_err(error)?;
            if state.finished {return Err(error("GPU prefix session is already complete"));}
            let tag=state.next_tag;state.next_tag=tag.checked_add(1).ok_or_else(||error("GPU prefix transfer sequence overflow"))?;
            write(&mut state.stream,&Command::Pull{fragment,start:heads.start,end:heads.end,destination,tag})?;
            match read::<Reply>(&mut state.stream)? {Reply::Ready(result)=>result.map_err(error)?,_=>return Err(error("GPU prefix readiness acknowledgement differs"))}
            let transfer=self.endpoints[destination].transfer(source,true,tag,pointer,bytes);
            let completed=match read::<Reply>(&mut state.stream) {
                Ok(Reply::Done(result))=>result.map_err(error),Ok(_)=>Err(error("GPU prefix completion acknowledgement differs")),Err(e)=>Err(e),
            };
            transfer.and(completed)
        })
    }
}
impl Drop for Session {fn drop(&mut self) {let _=self.shutdown.shutdown(Shutdown::Both);}}

pub(crate) struct GpuHandoff<B:Backend> {tensors:PrefixTensors<B>,session:Arc<Session>,tokens:Vec<i32>,limits:CausalModelLimits,logits_dtype:DType}
impl<B:Backend> GpuHandoff<B> {
    pub fn import<M:PrefixBatchModel<B>>(self,model:&M,prompt:&[i32],device:&B::Device)->Result<Prefix<B,M>,GenerationError> {
        if self.tokens!=prompt || self.limits!=model.batch_limits() || self.tensors.position!=prompt.len() || self.tensors.children()!=1 {
            return Err(error("GPU handoff tokens, positions or model limits differ"));
        }
        self.tensors.validate("serving-gpu-handoff-v1",self.limits.max_sequence_length,1)?;
        let state=model.import_prefix_tensors(self.tensors.child("cache")?,Default::default(),device)?;
        if model.prefix_position(&state)!=prompt.len() {return Err(error("GPU handoff model prefix position differs"));}
        let logits=self.tensors.tensor("logits",[1,1,1,self.limits.vocab_size],self.logits_dtype,(1,0..1),device)?.reshape([1,1,self.limits.vocab_size]);
        model.sync_batch(device)?;self.session.finish()?;
        Ok(Prefix{tokens:self.tokens,state,logits})
    }
}

#[derive(Default)]
struct Cancellation {canceled:AtomicBool,stream:Mutex<Option<TcpStream>>}
impl Cancellation {
    fn cancel(&self) {
        self.canceled.store(true,Ordering::Release);
        if let Ok(stream)=self.stream.lock() {if let Some(stream)=&*stream {let _=stream.shutdown(Shutdown::Both);}}
    }
    fn install(&self,stream:&TcpStream)->Result<(),GenerationError> {
        let mut slot=self.stream.lock().map_err(error)?;
        if self.canceled.load(Ordering::Acquire) {return Err(error("GPU prefill request was canceled"));}
        *slot=Some(stream.try_clone().map_err(error)?);Ok(())
    }
}

fn connect<B:GpuPrefixBackend>(address:&str,model_id:&str,limits:CausalModelLimits,tokens:Vec<i32>,devices:Vec<i32>,
    config:&GpuKvConfig,api:Arc<UcxApi>,cancellation:&Cancellation)->Result<GpuHandoff<B>,GenerationError> {
    if cancellation.canceled.load(Ordering::Acquire) {return Err(error("GPU prefill request was canceled"));}
    let mut stream=TcpStream::connect(address).map_err(error)?;configure(&stream,config)?;cancellation.install(&stream)?;
    write(&mut stream,&Request{version:1,model_id:model_id.into(),limits,tokens:tokens.clone(),devices:devices.clone(),transports:config.transports.clone()})?;
    let hello:Hello=read::<Result<Hello,String>>(&mut stream)?.map_err(error)?;
    let producers=hello.addresses.iter().map(|(ordinal,_)|*ordinal).collect::<Vec<_>>();
    if hello.version!=1 || hello.model_id!=model_id || hello.tokens!=tokens || hello.limits!=limits || !distinct(&producers) {
        return Err(error("GPU prefill response identity/tokens/topology differs"));
    }
    let endpoints=devices.iter().map(|&device|UcxEndpoint::new(api.clone(),device,producers.len(),config)).collect::<Result<Vec<_>,_>>()?;
    let addresses=endpoints.iter().map(UcxEndpoint::address).collect::<Result<Vec<_>,_>>()?;
    write(&mut stream,&addresses)?;
    for endpoint in &endpoints {for (peer,(_,address)) in hello.addresses.iter().enumerate() {endpoint.connect(peer,address)?;}}
    read::<Result<(),String>>(&mut stream)?.map_err(error)?;
    let shutdown=stream.try_clone().map_err(error)?;
    let session=Arc::new(Session{state:Mutex::new(SessionState{stream,next_tag:1,finished:false}),devices,producers,endpoints,shutdown});
    let tensors=PrefixTensors::from_manifest(hello.manifest,session.clone())?;
    Ok(GpuHandoff{tensors,session,tokens,limits,logits_dtype:hello.logits_dtype})
}

type Results<B>=Arc<Mutex<BTreeMap<RequestId,Result<GpuHandoff<B>,String>>>>;
struct WorkerPermit(Arc<AtomicUsize>);
impl Drop for WorkerPermit {fn drop(&mut self) {self.0.fetch_sub(1,Ordering::AcqRel);}}
pub(crate) struct GpuRemoteNetwork<B:Backend> {
    address:String,model_id:String,limits:CausalModelLimits,devices:Vec<i32>,config:GpuKvConfig,api:Arc<UcxApi>,capacity:usize,
    requests:BTreeMap<RequestId,Arc<Cancellation>>,results:Results<B>,
    workers:Arc<AtomicUsize>,
    start:fn(String,String,CausalModelLimits,Vec<i32>,Vec<i32>,GpuKvConfig,Arc<UcxApi>,Arc<Cancellation>)->Result<GpuHandoff<B>,GenerationError>,
}
impl<B:Backend> GpuRemoteNetwork<B> {
    pub fn new(remote:RemotePrefill,model_id:String,limits:CausalModelLimits,devices:Vec<B::Device>,config:GpuKvConfig,capacity:usize)
        ->Result<Self,GenerationError> where B:GpuPrefixBackend {
        let devices=devices.iter().map(B::prefix_device_ordinal).collect::<Result<Vec<_>,_>>()?;
        if !distinct(&devices) || capacity==0 || remote.address.is_empty() {return Err(error("GPU prefill needs distinct devices, address and bounded positive queue"));}
        let api=UcxApi::load(&config)?;
        Ok(Self{address:remote.address,model_id,limits,devices,config,api,capacity,requests:BTreeMap::new(),results:Arc::new(Mutex::new(BTreeMap::new())),workers:Arc::new(AtomicUsize::new(0)),
            start:|address,identity,limits,tokens,devices,config,api,cancel|connect::<B>(&address,&identity,limits,tokens,devices,&config,api,&cancel)})
    }
    pub fn submit(&mut self,id:RequestId,tokens:Vec<i32>)->Result<(),GenerationError> {
        if self.requests.len()>=self.capacity || self.requests.contains_key(&id) {return Err(error("GPU prefill request queue is full or duplicated"));}
        self.workers.fetch_update(Ordering::AcqRel,Ordering::Acquire,|n|(n<self.capacity).then_some(n+1))
            .map_err(|_|error("GPU prefill connection workers are still draining canceled requests"))?;
        let permit=WorkerPermit(self.workers.clone());
        let cancel=Arc::new(Cancellation::default());let cancellation=cancel.clone();
        let (address,identity,limits,devices,config,api,start,results)=(self.address.clone(),self.model_id.clone(),self.limits,self.devices.clone(),
            self.config.clone(),self.api.clone(),self.start,self.results.clone());
        let owner=std::thread::current();
        std::thread::Builder::new().name(format!("ruda-gpu-prefill-{}",id.0)).spawn(move || {
            let _permit=permit;
            let result=std::panic::catch_unwind(std::panic::AssertUnwindSafe(||start(address,identity,limits,tokens,devices,config,api,cancel.clone())))
                .map_err(|_|"GPU prefill connection worker panicked".to_string()).and_then(|result|result.map_err(|e|e.to_string()));
            if let Ok(mut results)=results.lock() {
                if !cancel.canceled.load(Ordering::Acquire) {results.insert(id,result);}
            }
            owner.unpark();
        }).map_err(error)?;
        self.requests.insert(id,cancellation);Ok(())
    }
    pub fn poll(&mut self)->Option<(RequestId,Result<GpuHandoff<B>,String>)> {
        let result=self.results.lock().ok()?.pop_first()?;
        self.requests.remove(&result.0);Some(result)
    }
    pub fn cancel(&mut self,id:RequestId) {
        if let Some(cancel)=self.requests.remove(&id) {cancel.cancel();}
        if let Ok(mut results)=self.results.lock() {results.remove(&id);}
    }
    pub fn waiting(&self)->bool {!self.requests.is_empty()}
}
impl<B:Backend> Drop for GpuRemoteNetwork<B> {fn drop(&mut self) {for cancel in self.requests.values() {cancel.cancel();}}}
