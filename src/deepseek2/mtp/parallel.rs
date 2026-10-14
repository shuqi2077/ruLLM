use super::*;
use crate::{DeviceBatchModel,PrefixBatchModel};
use ruda_tensor::api::TensorPrimitive;
use ruda_tensor::device::Device;
use ruda_tensor::distributed::{DistributedBackend,ReduceOperation};

pub struct DeepseekMtpExpertParallel<B:DistributedBackend> {
    pub config:DeepseekV2Config,ranks:Vec<DeepseekMtp<B>>,devices:Vec<B::Device>,
}
#[derive(Clone,Debug)]
pub struct DeepseekMtpExpertParallelCache<B:DistributedBackend> {
    ranks:Vec<DeepseekMtpGenerationCache<B>>,devices:Vec<B::Device>,
}
pub struct DeepseekMtpExpertParallelCacheRecord<B:DistributedBackend> {ranks:Vec<DeepseekMtpGenerationCacheRecord<B>>}
pub struct LoadedDeepseekMtpExpertParallel<B:DistributedBackend> {
    pub model:DeepseekMtpExpertParallel<B>,pub eos_token_ids:Vec<i32>,pub reports:Vec<HuggingFaceLoadReport>,
}
crate::parallel_batch::parallel_batch_model!(DeepseekMtpExpertParallel,DeepseekMtp);

pub fn load_huggingface_deepseek_mtp_expert_parallel<B:DistributedBackend>(directory:impl AsRef<Path>,devices:Vec<B::Device>)
    ->Result<LoadedDeepseekMtpExpertParallel<B>,HuggingFaceLoadError> {load_parallel(directory.as_ref(),devices,None)}
pub fn load_huggingface_mla_mtp_expert_parallel<B:DistributedBackend+MlaFp8Backend>(directory:impl AsRef<Path>,devices:Vec<B::Device>)
    ->Result<LoadedDeepseekMtpExpertParallel<B>,HuggingFaceLoadError> {load_parallel(directory.as_ref(),devices,Some(weights::native_projection::<B>))}

fn load_parallel<B:DistributedBackend>(directory:&Path,devices:Vec<B::Device>,packed:Option<weights::Loader<B>>)
    ->Result<LoadedDeepseekMtpExpertParallel<B>,HuggingFaceLoadError> {
    let ids=devices.iter().map(Device::to_id).collect::<Vec<_>>();
    if ids.is_empty() || ids.iter().enumerate().any(|(i,id)|ids[..i].contains(id)) {
        return Err(HuggingFaceLoadError("MTP EP requires distinct native devices".into()));
    }
    let config:DeepseekV2Config=serde_json::from_slice(&std::fs::read(directory.join("config.json")).map_err(|e|HuggingFaceLoadError(e.to_string()))?)
        .map_err(|e|HuggingFaceLoadError(e.to_string()))?;
    config.validate()?;
    if !config.n_routed_experts.is_multiple_of(devices.len()) {return Err(HuggingFaceLoadError("MTP EP must evenly partition checkpoint experts".into()));}
    let mut ranks=Vec::with_capacity(devices.len());let mut reports=Vec::with_capacity(devices.len());let mut eos=None;
    for (rank,device) in devices.iter().enumerate() {
        let group=ids.clone();let reduction=Arc::new(move|value:Tensor<B,3>|Tensor::from_primitive(TensorPrimitive::Float(
            B::all_reduce(value.into_primitive().tensor(),ReduceOperation::Sum,group.clone()).resolve())));
        let loaded=load(directory,device,packed,Some(ExpertShard{rank,world:devices.len()}),Some(reduction))?;
        if eos.as_ref().is_some_and(|ids|*ids!=loaded.eos_token_ids) {return Err(HuggingFaceLoadError("MTP EP EOS differs between ranks".into()));}
        eos=Some(loaded.eos_token_ids);ranks.push(loaded.model);reports.push(loaded.report);
    }
    Ok(LoadedDeepseekMtpExpertParallel{model:DeepseekMtpExpertParallel{config,ranks,devices},eos_token_ids:eos.unwrap_or_default(),reports})
}

impl<B:DistributedBackend> DeepseekMtpExpertParallel<B> {
    pub fn devices(&self)->&[B::Device] {&self.devices}
    pub fn with_attention_mode(mut self,mode:DeepseekV2AttentionMode)->Self {
        for model in &mut self.ranks {model.target.attention_mode=mode;}self
    }
}
impl<B:DistributedBackend> CausalModel<B> for DeepseekMtpExpertParallel<B> {
    type Cache=DeepseekMtpExpertParallelCache<B>;
    fn new_cache(&self)->Self::Cache {Self::Cache{ranks:self.ranks.iter().map(CausalModel::new_cache).collect(),devices:self.devices.clone()}}
    fn forward_cached_last(&self,tokens:Tensor<B,2,Int>,cache:&mut Self::Cache)->Tensor<B,3> {
        self.try_forward_cached_last(tokens,cache).expect("validated MTP EP input")
    }
    fn try_forward_cached_last(&self,tokens:Tensor<B,2,Int>,cache:&mut Self::Cache)->Result<Tensor<B,3>,GenerationError> {
        if self.devices.first()!=Some(&tokens.device()) || cache.devices!=self.devices || cache.ranks.len()!=self.ranks.len()
            || cache.ranks.iter().any(|c|c.position()!=cache.generation_position()) {
            return Err(GenerationError("MTP EP input topology or cache positions differ".into()));
        }
        let data=tokens.try_into_data().map_err(|e|GenerationError(e.to_string()))?;
        std::thread::scope(|scope| {
            let mut handles=Vec::with_capacity(self.ranks.len());
            for ((model,state),device) in self.ranks.iter().zip(&mut cache.ranks).zip(&self.devices) {
                let data=data.clone();handles.push(scope.spawn(move|| {
                    let output=model.try_forward_cached_last(Tensor::from_data(data,device),state);
                    B::sync(device).map_err(|e|GenerationError(e.to_string()))?;output
                }));
            }
            let mut root=None;let mut failure=None;
            for (rank,handle) in handles.into_iter().enumerate() {
                match handle.join() {Ok(Ok(output)) if rank==0=>root=Some(output),Ok(Ok(_))=>(),Ok(Err(e))=>failure=Some(e),
                    Err(_)=>failure=Some(GenerationError("MTP EP rank panicked".into()))}
            }
            if let Some(error)=failure {Err(error)} else {root.ok_or_else(||GenerationError("MTP EP has no root output".into()))}
        })
    }
}
impl<B:DistributedBackend> Record<B> for DeepseekMtpExpertParallelCacheRecord<B> {
    type Item<S:PrecisionSettings>=Vec<<DeepseekMtpGenerationCacheRecord<B> as Record<B>>::Item<S>>;
    fn into_item<S:PrecisionSettings>(self)->Self::Item<S> {self.ranks.into_iter().map(|r|r.into_item::<S>()).collect()}
    fn from_item<S:PrecisionSettings>(item:Self::Item<S>,device:&B::Device)->Self {
        Self{ranks:item.into_iter().map(|r|DeepseekMtpGenerationCacheRecord::from_item::<S>(r,device)).collect()}
    }
}
fn identity(model_id:&str,rank:usize,world:usize)->String {format!("{model_id}:deepseek-mtp-ep:{world}:{rank}")}
impl<B:DistributedBackend> GenerationCacheContinuation<B> for DeepseekMtpExpertParallelCache<B> {
    type Record=DeepseekMtpExpertParallelCacheRecord<B>;
    fn generation_position(&self)->usize {self.ranks.first().map_or(0,DeepseekMtpGenerationCache::position)}
    fn capture_generation(&self,model_id:&str,device:&B::Device)->Result<Self::Record,RecorderError> {
        let position=self.generation_position();
        if model_id.is_empty() || self.ranks.is_empty() || self.ranks.len()!=self.devices.len() || self.devices.first()!=Some(device)
            || self.ranks.iter().any(|r|r.position()!=position) {return Err(invalid("incomplete MTP EP cache group"));}
        let ranks=self.ranks.iter().zip(&self.devices).enumerate().map(|(rank,(cache,device))| {
            B::sync(device).map_err(invalid)?;cache.capture_generation(&identity(model_id,rank,self.ranks.len()),device)
        }).collect::<Result<Vec<_>,_>>()?;
        Ok(Self::Record{ranks})
    }
    fn restore_generation(record:Self::Record,model_id:&str,template:&Self,position:usize,device:&B::Device)->Result<Self,RecorderError> {
        let world=template.ranks.len();
        if world==0 || model_id.is_empty() || record.ranks.len()!=world || template.devices.len()!=world || template.devices.first()!=Some(device) {
            return Err(invalid("MTP EP continuation topology differs"));
        }
        let ranks=record.ranks.into_iter().zip(&template.ranks).zip(&template.devices).enumerate().map(|(rank,((record,cache),device))|
            DeepseekMtpGenerationCache::restore_generation(record,&identity(model_id,rank,world),cache,position,device)).collect::<Result<Vec<_>,_>>()?;
        Ok(Self{ranks,devices:template.devices.clone()})
    }
}
