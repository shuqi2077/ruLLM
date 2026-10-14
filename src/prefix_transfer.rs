use crate::GenerationError;
use ruda_tensor::api::{backend::Backend,DType,Tensor,TensorData};
use serde::{Deserialize,Serialize};
use std::{collections::BTreeMap,ops::Range,sync::Arc};

/// Global head geometry of one TP rank. EP attention/recurrent caches are
/// replicated and use the default layout, independently of expert ownership.
#[derive(Debug,Clone,Copy,PartialEq,Eq)]
pub struct PrefixShard {
    pub rank:usize,
    pub world:usize,
    pub kv_heads:Option<usize>,
    pub linear_key_heads:Option<usize>,
    pub linear_value_heads:Option<usize>,
}
impl Default for PrefixShard {
    fn default()->Self {Self{rank:0,world:1,kv_heads:None,linear_key_heads:None,linear_value_heads:None}}
}
impl PrefixShard {
    pub fn heads(&self,global:Option<usize>,local:usize)->Result<(usize,Range<usize>),GenerationError> {
        let Some(global)=global else {return Ok((local,0..local));};
        if self.world==0 || self.rank>=self.world || global==0 || local==0 {
            return Err(error("invalid prefix shard topology"));
        }
        let range=if global>=self.world && global.is_multiple_of(self.world) {
            let n=global/self.world;self.rank*n..(self.rank+1)*n
        } else if self.world>global && self.world.is_multiple_of(global) {
            let start=self.rank/(self.world/global);start..start+1
        } else {return Err(error("prefix heads do not partition or replicate across TP ranks"));};
        if range.len()!=local {return Err(error("local prefix head count differs from global topology"));}
        Ok((global,range))
    }
}

#[derive(Clone)]
enum Payload<B:Backend> {
    Device(Tensor<B,4>),Host(Arc<TensorData>),
    Remote{fragment:usize,producer:i32,receiver:Arc<dyn PrefixTensorReceiver<B>>},
}

pub(crate) trait PrefixTensorReceiver<B:Backend>:Send+Sync {
    fn receive(&self,fragment:usize,producer:i32,heads:Range<usize>,shape:[usize;4],dtype:DType,device:&B::Device)
        ->Result<Tensor<B,4>,GenerationError>;
}

#[derive(Clone,Serialize,Deserialize)]
pub(crate) struct PrefixManifest {
    schema:String,position:usize,fragments:Vec<ManifestFragment>,children:BTreeMap<String,Self>,
    #[serde(default,skip_serializing_if="BTreeMap::is_empty")]
    scalars:BTreeMap<String,usize>,
}
#[derive(Clone,Serialize,Deserialize)]
struct ManifestFragment {name:String,id:usize,producer:i32,global_heads:usize,start:usize,shape:[usize;4],dtype:DType}

/// The callback is synchronous: on success and error all foreign reads/writes
/// must be complete before returning. The backend retains the allocation and
/// address pin, and fences the native producer before invoking it.
pub trait GpuPrefixBackend:Backend {
    fn prefix_device_ordinal(device:&Self::Device)->Result<i32,GenerationError>;
    fn with_prefix_pointer(tensor:Tensor<Self,4>,operation:impl FnOnce(u64,usize)->Result<(),GenerationError>)
        ->Result<Tensor<Self,4>,GenerationError>;
}
#[derive(Clone)]
struct Fragment<B:Backend> {global_heads:usize,start:usize,shape:[usize;4],dtype:DType,payload:Payload<B>}
#[derive(Serialize,Deserialize)]
struct HostFragment {name:String,global_heads:usize,start:usize,data:TensorData}
#[derive(Serialize,Deserialize)]
struct HostPrefix {
    version:u32,model_id:String,schema:String,position:usize,fragments:Vec<HostFragment>,children:BTreeMap<String,HostPrefix>,
    #[serde(default,skip_serializing_if="BTreeMap::is_empty")]
    scalars:BTreeMap<String,usize>,
}

/// Named semantic tensors, not physical KV page IDs. Only head axis 1 is
/// sharded. Convolution channels are split into Q/K/V before entering here.
#[derive(Clone)]
pub struct PrefixTensors<B:Backend> {
    pub position:usize,
    schema:String,
    tensors:BTreeMap<String,Vec<Fragment<B>>>,
    children:BTreeMap<String,Self>,
    scalars:BTreeMap<String,usize>,
}
fn error(message:impl ToString)->GenerationError {GenerationError(message.to_string())}
impl<B:Backend> PrefixTensors<B> {
    pub fn new(schema:impl Into<String>,position:usize)->Self {
        Self{position,schema:schema.into(),tensors:BTreeMap::new(),children:BTreeMap::new(),scalars:BTreeMap::new()}
    }
    pub(crate) fn set_scalar(&mut self,name:impl Into<String>,value:usize)->Result<(),GenerationError> {
        let name=name.into();
        if name.is_empty() || self.scalars.contains_key(&name) {return Err(error("invalid or duplicate prefix scalar"));}
        self.scalars.insert(name,value);Ok(())
    }
    pub(crate) fn scalar(&self,name:&str)->Option<usize> {self.scalars.get(name).copied()}
    pub(crate) fn scalars(&self)->usize {self.scalars.len()}
    pub fn add_child(&mut self,name:impl Into<String>,child:Self)->Result<(),GenerationError> {
        let name=name.into();if self.children.contains_key(&name) {return Err(error("duplicate prefix child"));}
        self.children.insert(name,child);Ok(())
    }
    pub fn child(&self,name:&str)->Result<&Self,GenerationError> {
        self.children.get(name).ok_or_else(||error(format!("missing prefix child {name}")))
    }
    pub fn children(&self)->usize {self.children.len()}
    pub fn has_child(&self,name:&str)->bool {self.children.contains_key(name)}
    pub fn validate(&self,schema:&str,position_limit:usize,names:usize)->Result<(),GenerationError> {
        if self.schema!=schema || self.position==0 || self.position>position_limit || self.tensors.len()!=names {
            return Err(error("prefix tensor schema/position/component count differs"));
        }
        Ok(())
    }
    fn insert(&mut self,name:String,fragment:Fragment<B>)->Result<(),GenerationError> {
        let [batch,heads,rows,width]=fragment.shape;
        if name.is_empty() || batch!=1 || heads==0 || rows==0 || width==0
            || fragment.start.checked_add(heads).is_none_or(|end|end>fragment.global_heads) {
            return Err(error("invalid prefix tensor fragment geometry"));
        }
        let fragments=self.tensors.entry(name).or_default();
        for old in fragments.iter() {
            if old.global_heads!=fragment.global_heads || old.dtype!=fragment.dtype
                || old.shape[0]!=batch || old.shape[2]!=rows || old.shape[3]!=width {
                return Err(error("prefix tensor fragments disagree on global geometry or dtype"));
            }
            if old.start==fragment.start && old.shape[1]==heads {
                // Replicated GQA/EP state has one canonical producer, never a sum.
                return Ok(());
            }
            if old.start<fragment.start+heads && fragment.start<old.start+old.shape[1] {
                return Err(error("partially overlapping prefix tensor shards"));
            }
        }
        fragments.push(fragment);fragments.sort_by_key(|f|f.start);Ok(())
    }
    pub fn push(&mut self,name:impl Into<String>,tensor:Tensor<B,4>,heads:(usize,Range<usize>))->Result<(),GenerationError> {
        if tensor.dims()[1]!=heads.1.len() {return Err(error("prefix tensor head span differs"));}
        self.insert(name.into(),Fragment{global_heads:heads.0,start:heads.1.start,shape:tensor.dims(),dtype:tensor.dtype(),payload:Payload::Device(tensor)})
    }
    pub fn merge(&mut self,other:Self)->Result<(),GenerationError> {
        if self.position!=other.position || self.schema!=other.schema || self.scalars!=other.scalars {
            return Err(error("prefix rank schema/position/scalar state differs"));
        }
        for (name,fragments) in other.tensors {for fragment in fragments {self.insert(name.clone(),fragment)?;}}
        for (name,child) in other.children {
            if let Some(existing)=self.children.get_mut(&name) {existing.merge(child)?;} else {self.children.insert(name,child);}
        }
        Ok(())
    }
    pub fn tensor(&self,name:&str,shape:[usize;4],dtype:DType,heads:(usize,Range<usize>),device:&B::Device)
        ->Result<Tensor<B,4>,GenerationError> {
        let fragments=self.tensors.get(name).ok_or_else(||error(format!("missing prefix tensor {name}")))?;
        if shape[0]!=1 || heads.1.len()!=shape[1] || heads.1.end>heads.0 {return Err(error("invalid destination prefix shard"));}
        let mut cursor=heads.1.start;let mut pieces=Vec::new();
        for fragment in fragments {
            if fragment.global_heads!=heads.0 || fragment.dtype!=dtype || fragment.shape[2..]!=shape[2..] {
                return Err(error(format!("prefix tensor {name} geometry/dtype differs")));
            }
            let start=fragment.start.max(heads.1.start);let end=(fragment.start+fragment.shape[1]).min(heads.1.end);
            if start>=end {continue;}
            if start!=cursor {return Err(error(format!("missing head span in prefix tensor {name}")));}
            let lo=start-fragment.start;let hi=end-fragment.start;
            let piece=match &fragment.payload {
                Payload::Device(tensor)=>tensor.clone().slice([0..1,lo..hi,0..shape[2],0..shape[3]]).to_device(device),
                Payload::Host(data)=>{
                    let stride=shape[2].checked_mul(shape[3]).and_then(|n|n.checked_mul(dtype.size()))
                        .ok_or_else(||error("prefix tensor byte stride overflow"))?;
                    let data=TensorData::from_bytes_vec(data.as_bytes()[lo*stride..hi*stride].to_vec(),[1,hi-lo,shape[2],shape[3]],dtype);
                    Tensor::from_data(data,(device,dtype))
                },
                Payload::Remote{fragment,producer,receiver}=>receiver.receive(*fragment,*producer,lo..hi,[1,hi-lo,shape[2],shape[3]],dtype,device)?,
            };
            pieces.push(piece);cursor=end;
        }
        if cursor!=heads.1.end || pieces.is_empty() {return Err(error(format!("incomplete prefix tensor {name}")));}
        if pieces.len()==1 {Ok(pieces.remove(0))} else {Ok(Tensor::cat(pieces,1))}
    }
    pub fn encode_host(&self,model_id:&str)->Result<Vec<u8>,GenerationError> {
        rmp_serde::to_vec(&self.host_record(model_id)?).map_err(error)
    }
    fn host_record(&self,model_id:&str)->Result<HostPrefix,GenerationError> {
        let mut fragments=Vec::new();
        for (name,parts) in &self.tensors {for part in parts {
            let data=match &part.payload {
                Payload::Device(t)=>t.clone().try_into_data().map_err(error)?,Payload::Host(data)=>(**data).clone(),
                Payload::Remote{..}=>return Err(error("remote prefix payload must be received before host serialization")),
            };
            fragments.push(HostFragment{name:name.clone(),global_heads:part.global_heads,start:part.start,data});
        }}
        let children=self.children.iter().map(|(name,child)|Ok((name.clone(),child.host_record(model_id)?)))
            .collect::<Result<_,GenerationError>>()?;
        Ok(HostPrefix{version:if self.scalars.is_empty() {1}else{2},model_id:model_id.into(),
            schema:self.schema.clone(),position:self.position,fragments,children,scalars:self.scalars.clone()})
    }
    /// Host payload stays on host until each destination rank requests its
    /// exact head interval. No full global cache is uploaded onto the root GPU.
    pub fn decode_host(bytes:&[u8],model_id:&str)->Result<Self,GenerationError> {
        let host:HostPrefix=rmp_serde::from_slice(bytes).map_err(error)?;
        Self::from_host_record(host,model_id)
    }
    fn from_host_record(host:HostPrefix,model_id:&str)->Result<Self,GenerationError> {
        if !matches!(host.version,1|2) || (host.version==1 && !host.scalars.is_empty())
            || host.model_id!=model_id || model_id.is_empty() {return Err(error("prefix transfer identity/version differs"));}
        let mut result=Self::new(host.schema,host.position);
        for (name,value) in host.scalars {result.set_scalar(name,value)?;}
        for (name,child) in host.children {result.add_child(name,Self::from_host_record(child,model_id)?)?;}
        for fragment in host.fragments {
            let shape:[usize;4]=fragment.data.shape.as_slice().try_into().map_err(error)?;
            let dtype=fragment.data.dtype;
            if !matches!(dtype,DType::F16|DType::BF16|DType::F32|DType::F64|DType::Flex32)
                || shape.iter().try_fold(dtype.size(),|n,&d|n.checked_mul(d))!=Some(fragment.data.as_bytes().len()) {
                return Err(error("prefix transfer tensor byte length or dtype differs"));
            }
            result.insert(fragment.name,Fragment{global_heads:fragment.global_heads,start:fragment.start,shape,dtype,payload:Payload::Host(Arc::new(fragment.data))})?;
        }
        Ok(result)
    }

    pub(crate) fn device_manifest(&self,tensors:&mut Vec<Tensor<B,4>>,ordinal:&impl Fn(&B::Device)->Result<i32,GenerationError>)
        ->Result<PrefixManifest,GenerationError> {
        let mut fragments=Vec::new();
        for (name,parts) in &self.tensors {for part in parts {
            let Payload::Device(tensor)=&part.payload else {return Err(error("GPU handoff requires resident device tensors"));};
            fragments.push(ManifestFragment{name:name.clone(),id:tensors.len(),producer:ordinal(&tensor.device())?,
                global_heads:part.global_heads,start:part.start,shape:part.shape,dtype:part.dtype});
            tensors.push(tensor.clone());
        }}
        let children=self.children.iter().map(|(name,child)|Ok((name.clone(),child.device_manifest(tensors,ordinal)?)))
            .collect::<Result<_,GenerationError>>()?;
        Ok(PrefixManifest{schema:self.schema.clone(),position:self.position,fragments,children,scalars:self.scalars.clone()})
    }
    pub(crate) fn from_manifest(manifest:PrefixManifest,receiver:Arc<dyn PrefixTensorReceiver<B>>)->Result<Self,GenerationError> {
        let mut result=Self::new(manifest.schema,manifest.position);
        for (name,value) in manifest.scalars {result.set_scalar(name,value)?;}
        for part in manifest.fragments {
            if !matches!(part.dtype,DType::F16|DType::BF16|DType::F32|DType::F64|DType::Flex32)
                || part.shape.iter().try_fold(part.dtype.size(),|n,&d|n.checked_mul(d)).is_none() {
                return Err(error("GPU prefix manifest dtype/size differs"));
            }
            result.insert(part.name,Fragment{global_heads:part.global_heads,start:part.start,shape:part.shape,dtype:part.dtype,
                payload:Payload::Remote{fragment:part.id,producer:part.producer,receiver:receiver.clone()}})?;
        }
        for (name,child) in manifest.children {result.add_child(name,Self::from_manifest(child,receiver.clone())?)?;}
        Ok(result)
    }
}

#[cfg(feature="nvidia")]
impl<F:ruda_tensor_device::FloatElement,I:ruda_tensor_device::IntElement,BT:ruda_tensor_device::BoolElement> GpuPrefixBackend
    for ruda_tensor_device::DeviceBackend<ruda_driver_cuda::CudaRuntime,F,I,BT> {
    fn prefix_device_ordinal(device:&ruda_driver_cuda::CudaDevice)->Result<i32,GenerationError> {
        i32::try_from(device.index).map_err(error)
    }
    fn with_prefix_pointer(tensor:Tensor<Self,4>,operation:impl FnOnce(u64,usize)->Result<(),GenerationError>)
        ->Result<Tensor<Self,4>,GenerationError> {
        let tensor=ruda_kernel::tensor::contiguous::into_contiguous(tensor.into_primitive().tensor());
        Self::sync(&tensor.device).map_err(error)?;
        let bytes=tensor.meta.shape.iter().try_fold(tensor.dtype.size(),|n,&d|n.checked_mul(d))
            .ok_or_else(||error("GPU prefix byte count overflow"))?;
        let resource=tensor.client.get_resource(tensor.handle.clone()).map_err(error)?;
        if resource.resource().size<bytes as u64 {return Err(error("GPU prefix allocation is smaller than the payload"));}
        let _pin=resource.resource().address_pin();
        operation(resource.resource().ptr,bytes)?;
        // UCX completion (including cancellation drain) precedes returning the
        // tensor to the model; no host payload staging or pointer escape.
        Ok(Tensor::from_primitive(ruda_tensor::TensorPrimitive::Float(tensor)))
    }
}
