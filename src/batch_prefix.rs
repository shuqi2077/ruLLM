use crate::{CausalModelLimits,GenerationError,PrefixBatchModel};
use crate::prefix_cache::{PrefixCacheConfig,PrefixCacheStats};
use ruda_tensor::api::{backend::Backend,DType,Tensor,TensorData};
use serde::{Deserialize,Serialize};
use sha2::{Digest,Sha256};
use std::{collections::BTreeMap,fs::{self,File,OpenOptions},io::{Read,Write},path::{Path,PathBuf}};

const MAGIC:&[u8;8]=b"RUBPKV01";
pub(crate) struct Prefix<B:Backend,M:PrefixBatchModel<B>> {
    pub tokens:Vec<i32>,pub state:M::Prefix,pub logits:Tensor<B,3>,
}
impl<B:Backend,M:PrefixBatchModel<B>> Clone for Prefix<B,M> {
    fn clone(&self)->Self {Self{tokens:self.tokens.clone(),state:self.state.clone(),logits:self.logits.clone()}}
}
enum Tier<B:Backend,M:PrefixBatchModel<B>> {Device(Prefix<B,M>),Host(Vec<u8>),Disk(PathBuf,usize)}
struct Entry<B:Backend,M:PrefixBatchModel<B>> {tier:Tier<B,M>,used:u64}
#[derive(Serialize,Deserialize)]
struct Header {version:u32,model_id:String,limits:CausalModelLimits,tokens:Vec<i32>}
pub(crate) struct PrefixStore<B:Backend,M:PrefixBatchModel<B>> {
    model_id:String,limits:CausalModelLimits,config:PrefixCacheConfig,
    entries:BTreeMap<Vec<i32>,Entry<B,M>>,tick:u64,stats:PrefixCacheStats,
    _disk_owner:Option<File>,
}
fn error(message:impl std::fmt::Display)->GenerationError {GenerationError(message.to_string())}
impl<B:Backend,M:PrefixBatchModel<B>> PrefixStore<B,M> {
    pub fn new(model_id:String,limits:CausalModelLimits,mut config:PrefixCacheConfig)->Result<Self,GenerationError> {
        if model_id.is_empty() || config.block_tokens==0 || (config.disk_bytes>0 && config.disk_directory.is_none()) {
            return Err(error("batch prefix cache needs exact model identity, positive block size and a configured disk directory"));
        }
        let mut disk_owner=None;
        if let Some(directory)=&mut config.disk_directory {
            *directory=directory.join(format!("{:x}-batch-v1",Sha256::digest(model_id.as_bytes())));
            fs::create_dir_all(&*directory).map_err(error)?;
            let owner=OpenOptions::new().create(true).truncate(false).read(true).write(true).open(directory.join(".owner.lock")).map_err(error)?;
            owner.try_lock().map_err(|e|error(format!("batch prefix cache directory already owned or cannot be locked: {e}")))?;
            disk_owner=Some(owner);
        }
        let mut this=Self{model_id,limits,config,entries:BTreeMap::new(),tick:0,stats:PrefixCacheStats::default(),_disk_owner:disk_owner};
        if let Some(directory)=this.config.disk_directory.clone() {
            for entry in fs::read_dir(directory).map_err(error)? {
                let path=entry.map_err(error)?.path();
                let Some(name)=path.file_name().and_then(|n|n.to_str()) else {continue;};
                if !name.starts_with("ruda-batch-kv-") {continue;}
                if name.ends_with(".tmp") {fs::remove_file(&path).map_err(error)?;continue;}
                if !name.ends_with(".mpk") || !path.is_file() {continue;}
                let found=this.read_header(&path);
                match found {
                    Ok((header,size,_))=>{
                        if size>this.config.disk_bytes {fs::remove_file(&path).map_err(error)?;continue;}
                        if let Some(old)=this.entries.remove(&header.tokens) {this.discard(old)?;}
                        this.tick+=1;this.stats.disk_bytes=this.stats.disk_bytes.checked_add(size).ok_or_else(||error("prefix disk accounting overflow"))?;
                        this.entries.insert(header.tokens,Entry{tier:Tier::Disk(path,size),used:this.tick});
                    },
                    Err(e)=>{eprintln!("batch prefix cache discarded invalid cache file: {e}");fs::remove_file(&path).map_err(error)?;},
                }
            }
            while this.stats.disk_bytes>this.config.disk_bytes {this.evict_disk()?;}
        }
        Ok(this)
    }
    pub fn enabled(&self)->bool {self.config.device_entries>0 || self.config.host_bytes>0 || self.config.disk_bytes>0}
    pub fn model_id(&self)->&str {&self.model_id}
    pub fn stats(&self)->PrefixCacheStats {self.stats.clone()}
    fn header(&self,tokens:Vec<i32>)->Header {Header{version:1,model_id:self.model_id.clone(),limits:self.limits,tokens}}
    fn validate_header(&self,h:&Header)->Result<(),GenerationError> {
        if h.version!=1 || h.model_id!=self.model_id || h.limits!=self.limits || h.tokens.is_empty()
            || h.tokens.len()>self.limits.max_sequence_length
            || h.tokens.iter().any(|&t|t<0 || t as usize>=self.limits.vocab_size) {
            return Err(error("batch prefix identity, context or tokens differ"));
        }
        Ok(())
    }
    fn read_header(&self,path:&Path)->Result<(Header,usize,File),GenerationError> {
        let mut file=File::open(path).map_err(error)?;let mut frame=[0u8;20];file.read_exact(&mut frame).map_err(error)?;
        if &frame[..8]!=MAGIC {return Err(error("unknown batch prefix file format"));}
        let metadata=u32::from_le_bytes(frame[8..12].try_into().unwrap()) as usize;
        let payload=usize::try_from(u64::from_le_bytes(frame[12..20].try_into().unwrap())).map_err(error)?;
        let size=metadata.checked_add(payload).and_then(|n|n.checked_add(20)).ok_or_else(||error("batch prefix file overflow"))?;
        let max_meta=self.limits.max_sequence_length.checked_mul(12).and_then(|n|n.checked_add(self.model_id.len()+1024))
            .ok_or_else(||error("batch prefix metadata size overflow"))?;
        if metadata>max_meta || size as u64!=file.metadata().map_err(error)?.len() || size>self.config.disk_bytes {
            return Err(error("batch prefix file exceeds its metadata or disk budget"));
        }
        let mut bytes=vec![0;metadata];file.read_exact(&mut bytes).map_err(error)?;
        let header:Header=serde_json::from_slice(&bytes).map_err(error)?;self.validate_header(&header)?;
        Ok((header,size,file))
    }
    fn validate(&self,model:&M,prefix:&Prefix<B,M>,device:&B::Device)->Result<(),GenerationError> {
        self.validate_header(&self.header(prefix.tokens.clone()))?;
        if model.prefix_position(&prefix.state)!=prefix.tokens.len() || prefix.logits.dims()!=[1,1,self.limits.vocab_size]
            || prefix.logits.device()!=*device {return Err(error("batch prefix cache/logits position or geometry differs"));}
        Ok(())
    }
    fn encode(&self,model:&M,prefix:&Prefix<B,M>,device:&B::Device)->Result<Vec<u8>,GenerationError> {
        self.validate(model,prefix,device)?;B::sync(device).map_err(error)?;
        let cache=model.encode_prefix(&prefix.state,&self.model_id)?;
        let logits=prefix.logits.clone().try_into_data().map_err(error)?;
        rmp_serde::to_vec(&(cache,logits)).map_err(error)
    }
    fn decode(&self,model:&M,tokens:Vec<i32>,bytes:&[u8],device:&B::Device)->Result<Prefix<B,M>,GenerationError> {
        self.decode_payload(model,tokens,bytes,device,false)
    }
    fn decode_payload(&self,model:&M,tokens:Vec<i32>,bytes:&[u8],device:&B::Device,portable:bool)->Result<Prefix<B,M>,GenerationError> {
        let (cache,logits):(Vec<u8>,TensorData)=rmp_serde::from_slice(bytes).map_err(error)?;
        if logits.shape.as_slice()!=[1,1,self.limits.vocab_size]
            || !matches!(logits.dtype,DType::F16|DType::BF16|DType::F32|DType::F64|DType::Flex32)
            || self.limits.vocab_size.checked_mul(logits.dtype.size())!=Some(logits.as_bytes().len()) {
            return Err(error("invalid batch prefix logits storage"));
        }
        let dtype=logits.dtype;
        let state=if portable {
            let tensors=crate::PrefixTensors::decode_host(&cache,&self.model_id)?;
            model.import_prefix_tensors(&tensors,Default::default(),device)?
        } else {model.decode_prefix(&cache,&self.model_id,device)?};
        let prefix=Prefix{tokens,state,logits:Tensor::from_data(logits,(device,dtype))};
        self.validate(model,&prefix,device)?;Ok(prefix)
    }
    pub fn export_handoff(&self,model:&M,prefix:&Prefix<B,M>,device:&B::Device,portable:bool)->Result<Vec<u8>,GenerationError> {
        if portable && model.prefix_tensor_transfer() {
            self.validate(model,prefix,device)?;model.sync_batch(device)?;
            let tensors=model.export_prefix_tensors(&prefix.state,Default::default())?;
            let cache=tensors.encode_host(&self.model_id)?;
            let logits=prefix.logits.clone().try_into_data().map_err(error)?;
            let payload=rmp_serde::to_vec(&(cache,logits)).map_err(error)?;
            let mut header=self.header(prefix.tokens.clone());header.version=2;
            return rmp_serde::to_vec(&(header,payload)).map_err(error);
        }
        rmp_serde::to_vec(&(self.header(prefix.tokens.clone()),self.encode(model,prefix,device)?)).map_err(error)
    }
    pub fn import_handoff(&self,model:&M,prompt:&[i32],bytes:&[u8],device:&B::Device)->Result<Prefix<B,M>,GenerationError> {
        let (mut header,payload):(Header,Vec<u8>)=rmp_serde::from_slice(bytes).map_err(error)?;
        let portable=header.version==2;if portable {header.version=1;}
        self.validate_header(&header)?;
        if header.tokens!=prompt {return Err(error("remote batch prefix token sequence differs"));}
        self.decode_payload(model,header.tokens,&payload,device,portable)
    }
    pub fn lookup(&mut self,model:&M,prompt:&[i32],device:&B::Device)->Result<Option<Prefix<B,M>>,GenerationError> {
        self.stats.lookups+=1;self.tick+=1;
        loop {
            let key=self.entries.keys().filter(|key|prompt.starts_with(key)).max_by_key(|key|key.len()).cloned();
            let Some(key)=key else {return Ok(None);};
            let entry=self.entries.get(&key).unwrap();
            let result=match &entry.tier {
                Tier::Device(prefix)=>Ok(prefix.clone()),
                Tier::Host(bytes)=>self.decode(model,key.clone(),bytes,device),
                Tier::Disk(path,_)=>self.read_header(path).and_then(|(header,_,mut file)| {
                    if header.tokens!=key {return Err(error("batch prefix file token key changed"));}
                    let mut bytes=Vec::new();file.read_to_end(&mut bytes).map_err(error)?;
                    self.decode(model,key.clone(),&bytes,device)
                }),
            }.and_then(|prefix|{self.validate(model,&prefix,device)?;Ok(prefix)});
            match result {
                Ok(prefix)=>{
                    self.entries.get_mut(&key).unwrap().used=self.tick;
                    self.stats.hits+=1;self.stats.reused_tokens+=key.len() as u64;return Ok(Some(prefix));
                },
                Err(e)=>{
                    eprintln!("batch prefix restore failed; recomputing request prefix: {e}");
                    let entry=self.entries.remove(&key).unwrap();self.discard(entry)?;
                },
            }
        }
    }
    pub fn insert(&mut self,model:&M,prefix:Prefix<B,M>,device:&B::Device)->Result<(),GenerationError> {
        if !self.enabled() {return Ok(());}
        self.validate(model,&prefix,device)?;self.tick+=1;
        if let Some(existing)=self.entries.get_mut(&prefix.tokens) {existing.used=self.tick;return Ok(());}
        self.entries.insert(prefix.tokens.clone(),Entry{tier:Tier::Device(prefix),used:self.tick});self.stats.device_entries+=1;
        while self.stats.device_entries>self.config.device_entries {
            let key=self.entries.iter().filter(|(_,e)|matches!(&e.tier,Tier::Device(_))).min_by_key(|(_,e)|e.used).unwrap().0.clone();
            let entry=self.entries.remove(&key).unwrap();self.stats.device_entries-=1;
            let Tier::Device(prefix)=entry.tier else {unreachable!()};
            if self.config.host_bytes==0 && self.config.disk_bytes==0 {continue;}
            let raw=model.prefix_bytes(&prefix.state).saturating_add(self.limits.vocab_size.saturating_mul(prefix.logits.dtype().size()));
            if raw>self.config.host_bytes.max(self.config.disk_bytes) {continue;}
            let bytes=self.encode(model,&prefix,device)?;
            if bytes.len()<=self.config.host_bytes {
                self.stats.host_bytes+=bytes.len();self.entries.insert(key,Entry{tier:Tier::Host(bytes),used:entry.used});
            } else {self.put_disk(key,bytes,entry.used)?;}
        }
        while self.stats.host_bytes>self.config.host_bytes {
            let key=self.entries.iter().filter(|(_,e)|matches!(&e.tier,Tier::Host(_))).min_by_key(|(_,e)|e.used).unwrap().0.clone();
            let entry=self.entries.remove(&key).unwrap();let Tier::Host(bytes)=entry.tier else {unreachable!()};
            self.stats.host_bytes-=bytes.len();self.put_disk(key,bytes,entry.used)?;
        }
        Ok(())
    }
    fn discard(&mut self,entry:Entry<B,M>)->Result<(),GenerationError> {
        match entry.tier {
            Tier::Device(_)=>self.stats.device_entries-=1,
            Tier::Host(bytes)=>self.stats.host_bytes-=bytes.len(),
            Tier::Disk(path,size)=>{match fs::remove_file(path) {Ok(())=>(),Err(e) if e.kind()==std::io::ErrorKind::NotFound=>(),Err(e)=>return Err(error(e))};self.stats.disk_bytes-=size;},
        }
        Ok(())
    }
    fn evict_disk(&mut self)->Result<(),GenerationError> {
        let key=self.entries.iter().filter(|(_,e)|matches!(&e.tier,Tier::Disk(_, _))).min_by_key(|(_,e)|e.used)
            .ok_or_else(||error("batch prefix disk accounting differs from entries"))?.0.clone();
        let entry=self.entries.remove(&key).unwrap();self.discard(entry)
    }
    fn put_disk(&mut self,key:Vec<i32>,bytes:Vec<u8>,used:u64)->Result<(),GenerationError> {
        if self.config.disk_bytes==0 {return Ok(());}
        let header=serde_json::to_vec(&self.header(key.clone())).map_err(error)?;
        let metadata=u32::try_from(header.len()).map_err(error)?;
        let size=20usize.checked_add(header.len()).and_then(|n|n.checked_add(bytes.len())).ok_or_else(||error("batch prefix size overflow"))?;
        if size>self.config.disk_bytes {return Ok(());}
        while self.stats.disk_bytes>self.config.disk_bytes-size {self.evict_disk()?;}
        let directory=self.config.disk_directory.as_ref().unwrap();
        let mut serial=self.tick;
        let (temporary,path,mut file)=loop {
            let base=format!("ruda-batch-kv-{}-{serial}",std::process::id());
            let temporary=directory.join(format!("{base}.tmp"));let path=directory.join(format!("{base}.mpk"));
            if path.exists() {serial=serial.checked_add(1).ok_or_else(||error("batch prefix filename overflow"))?;continue;}
            match OpenOptions::new().create_new(true).write(true).open(&temporary) {
                Ok(file)=>break (temporary,path,file),Err(e) if e.kind()==std::io::ErrorKind::AlreadyExists=>serial=serial.checked_add(1).ok_or_else(||error("batch prefix filename overflow"))?,Err(e)=>return Err(error(e)),
            }
        };
        let result=(||->std::io::Result<()> {
            file.write_all(MAGIC)?;file.write_all(&metadata.to_le_bytes())?;
            file.write_all(&(bytes.len() as u64).to_le_bytes())?;file.write_all(&header)?;file.write_all(&bytes)?;file.sync_all()
        })();
        drop(file);
        if let Err(e)=result {let _=fs::remove_file(temporary);return Err(error(e));}
        if let Err(e)=fs::rename(&temporary,&path) {let _=fs::remove_file(temporary);return Err(error(e));}
        self.stats.disk_bytes+=size;self.entries.insert(key,Entry{tier:Tier::Disk(path,size),used});Ok(())
    }
}
