use crate::{CausalModel, CausalModelLimits, GenerationCacheContinuation, GenerationError};
use ruda_model::record::{FullPrecisionSettings, Record};
use ruda_tensor::api::{Int, Tensor, TensorData, backend::Backend};
use serde::Deserialize;
use sha2::{Digest, Sha256};
use std::{
    collections::BTreeMap,
    fs::{self, OpenOptions},
    io::Write,
    path::PathBuf,
};

pub struct PrefilledPrompt<B: Backend, M: CausalModel<B>> {
    pub(crate) model_id: String,
    pub(crate) limits: CausalModelLimits,
    pub(crate) tokens: Vec<i32>,
    pub(crate) cache: M::Cache,
    pub(crate) logits: Tensor<B, 3>,
}

impl<B: Backend, M: CausalModel<B>> PrefilledPrompt<B, M> {
    pub fn tokens(&self) -> &[i32] {
        &self.tokens
    }

    pub(crate) fn validate(
        &self,
        model_id: &str,
        limits: CausalModelLimits,
        prompt: &[i32],
        device: &B::Device,
    ) -> Result<(), GenerationError> {
        let [batch, sequence, vocabulary] = self.logits.dims();
        if model_id.is_empty()
            || self.model_id != model_id
            || self.limits != limits
            || self.tokens.is_empty()
            || !prompt.starts_with(&self.tokens)
            || batch != 1
            || sequence != 1
            || vocabulary != limits.vocab_size
            || self.logits.device() != *device
        {
            return Err(GenerationError(
                "prefix identity, tokens, limits or native logits differ".into(),
            ));
        }
        Ok(())
    }
}

impl<B: Backend, M: CausalModel<B>> Clone for PrefilledPrompt<B, M>
where
    M::Cache: Clone,
{
    fn clone(&self) -> Self {
        Self {
            model_id: self.model_id.clone(),
            limits: self.limits,
            tokens: self.tokens.clone(),
            cache: self.cache.clone(),
            logits: self.logits.clone(),
        }
    }
}

impl<B: Backend, M: CausalModel<B>> PrefilledPrompt<B, M>
where
    M::Cache: Clone + GenerationCacheContinuation<B>,
{
    pub fn export(&self, device: &B::Device) -> Result<Vec<u8>, GenerationError> {
        self.validate(&self.model_id, self.limits, &self.tokens, device)?;
        if self.cache.generation_position() != self.tokens.len() {
            return Err(error("prefix cache position differs from exported prompt"));
        }
        TieredPrefixCache::<B, M>::capture(self, device)
    }

    pub fn import(
        bytes: &[u8],
        model: &M,
        model_id: &str,
        limits: CausalModelLimits,
        expected_prompt: &[i32],
        device: &B::Device,
    ) -> Result<Self, GenerationError> {
        let prefix = TieredPrefixCache::<B, M>::restore(bytes, model, model_id, limits, device)?;
        if prefix.tokens != expected_prompt {
            return Err(error("prefill handoff does not match the requested prompt"));
        }
        Ok(prefix)
    }
}

#[derive(Clone, Debug)]
pub struct PrefixCacheConfig {
    pub block_tokens: usize,
    pub device_entries: usize,
    pub host_bytes: usize,
    pub disk_directory: Option<PathBuf>,
    pub disk_bytes: usize,
}

#[derive(Clone, Debug, Default, serde::Serialize)]
pub struct PrefixCacheStats {
    pub lookups: u64,
    pub hits: u64,
    pub reused_tokens: u64,
    pub device_entries: usize,
    pub host_bytes: usize,
    pub disk_bytes: usize,
}

enum Tier<B: Backend, M: CausalModel<B>> {
    Device(PrefilledPrompt<B, M>),
    Host(Vec<u8>),
    Disk(PathBuf, usize),
}
struct Entry<B: Backend, M: CausalModel<B>> {
    tier: Tier<B, M>,
    used: u64,
}

pub struct TieredPrefixCache<B: Backend, M: CausalModel<B>> {
    model_id: String,
    limits: CausalModelLimits,
    config: PrefixCacheConfig,
    entries: BTreeMap<Vec<i32>, Entry<B, M>>,
    tick: u64,
    stats: PrefixCacheStats,
}

fn error(e: impl std::fmt::Display) -> GenerationError {
    GenerationError(e.to_string())
}

impl<B: Backend, M: CausalModel<B>> TieredPrefixCache<B, M>
where
    M::Cache: Clone + GenerationCacheContinuation<B>,
{
    pub fn new(
        model_id: String,
        limits: CausalModelLimits,
        mut config: PrefixCacheConfig,
    ) -> Result<Self, GenerationError> {
        if model_id.is_empty() {
            return Err(error("prefix cache requires exact frozen model identity"));
        }
        if config.block_tokens == 0 {
            return Err(error("prefix block size must be positive"));
        }
        if config.disk_bytes > 0 && config.disk_directory.is_none() {
            return Err(error("disk tier needs an explicit directory"));
        }
        if let Some(directory) = &mut config.disk_directory {
            *directory = directory.join(format!("{:x}", Sha256::digest(model_id.as_bytes())));
            fs::create_dir_all(&*directory).map_err(error)?;
        }
        let mut cache = Self {
            model_id,
            limits,
            config,
            entries: BTreeMap::new(),
            tick: 0,
            stats: PrefixCacheStats::default(),
        };
        cache.reopen_disk()?;
        Ok(cache)
    }

    fn reopen_disk(&mut self) -> Result<(), GenerationError> {
        let Some(directory) = &self.config.disk_directory else {
            return Ok(());
        };
        if self.config.disk_bytes == 0 {
            return Ok(());
        }
        for entry in fs::read_dir(directory).map_err(error)? {
            let entry = entry.map_err(error)?;
            let path = entry.path();
            if !entry.file_type().map_err(error)?.is_file()
                || !entry.file_name().to_string_lossy().starts_with("ruda-kv-")
                || path.extension().and_then(|v| v.to_str()) != Some("mpk")
            {
                continue;
            }
            let metadata = entry.metadata().map_err(error)?;
            let size = usize::try_from(metadata.len()).map_err(error)?;
            let mut decoder = rmp_serde::Deserializer::new(std::io::BufReader::new(
                fs::File::open(&path).map_err(error)?,
            ));
            let header = <(
                u32,
                String,
                CausalModelLimits,
                Vec<i32>,
                serde::de::IgnoredAny,
                serde::de::IgnoredAny,
            )>::deserialize(&mut decoder);
            let (version, id, limits, tokens, _, _) = match header {
                Ok(header) => header,
                Err(e) => {
                    eprintln!("ignored incomplete KV payload {}: {e}", path.display());
                    continue;
                }
            };
            if version != 1
                || id != self.model_id
                || limits != self.limits
                || tokens.is_empty()
                || tokens.len() > limits.max_sequence_length
                || tokens
                    .iter()
                    .any(|&t| t < 0 || t as usize >= limits.vocab_size)
            {
                eprintln!("ignored incompatible KV payload {}", path.display());
                continue;
            }
            self.tick += 1;
            if self.entries.contains_key(&tokens) {
                fs::remove_file(&path).map_err(error)?;
                continue;
            }
            self.stats.disk_bytes = self
                .stats
                .disk_bytes
                .checked_add(size)
                .ok_or_else(|| error("disk cache byte count overflow"))?;
            self.entries.insert(
                tokens,
                Entry {
                    tier: Tier::Disk(path, size),
                    used: self.tick,
                },
            );
        }
        while self.stats.disk_bytes > self.config.disk_bytes {
            self.evict_disk()?;
        }
        Ok(())
    }

    fn evict_disk(&mut self) -> Result<(), GenerationError> {
        let old = self
            .entries
            .iter()
            .filter(|(_, e)| matches!(&e.tier, Tier::Disk(..)))
            .min_by_key(|(_, e)| e.used)
            .map(|(key, _)| key.clone())
            .ok_or_else(|| error("disk budget has no evictable payload"))?;
        let Entry {
            tier: Tier::Disk(path, size),
            ..
        } = self.entries.get(&old).expect("existing disk payload")
        else {
            unreachable!()
        };
        fs::remove_file(path).map_err(error)?;
        self.stats.disk_bytes -= *size;
        self.entries.remove(&old);
        Ok(())
    }

    pub fn stats(&self) -> PrefixCacheStats {
        self.stats.clone()
    }

    pub fn prefill(
        &mut self,
        model: &M,
        prompt: &[i32],
        device: &B::Device,
    ) -> Result<PrefilledPrompt<B, M>, GenerationError> {
        if prompt.is_empty()
            || prompt.len() > self.limits.max_sequence_length
            || prompt
                .iter()
                .any(|&t| t < 0 || t as usize >= self.limits.vocab_size)
        {
            return Err(error("invalid prompt tokens for prefix prefill"));
        }
        let existing = self.lookup(model, prompt, device)?;
        let (mut cache, mut start, mut result) = match existing {
            Some(prefix) => (prefix.cache.clone(), prefix.tokens.len(), Some(prefix)),
            None => (model.new_cache(), 0, None),
        };
        while start < prompt.len() {
            let end = if self.config.device_entries == 0
                && self.config.host_bytes == 0
                && self.config.disk_bytes == 0
            {
                prompt.len()
            } else {
                (start / self.config.block_tokens + 1)
                    .saturating_mul(self.config.block_tokens)
                    .min(prompt.len())
            };
            let input = Tensor::<B, 2, Int>::from_data(
                TensorData::new(prompt[start..end].to_vec(), [1, end - start]),
                device,
            );
            let logits = model.try_forward_cached_last(input, &mut cache)?;
            B::sync(device).map_err(error)?;
            let prefix: PrefilledPrompt<B, M> = PrefilledPrompt {
                model_id: self.model_id.clone(),
                limits: self.limits,
                tokens: prompt[..end].to_vec(),
                cache: cache.clone(),
                logits,
            };
            prefix.validate(&self.model_id, self.limits, prompt, device)?;
            if prefix.cache.generation_position() != end {
                return Err(error("prefill did not complete all layer cache state"));
            }
            if let Err(e) = self.insert(prefix.clone(), device) {
                eprintln!("prefix cache tier write: {e}");
            }
            result = Some(prefix);
            start = end;
        }
        result.ok_or_else(|| error("prefix prefill produced no completed state"))
    }

    pub fn lookup(
        &mut self,
        model: &M,
        prompt: &[i32],
        device: &B::Device,
    ) -> Result<Option<PrefilledPrompt<B, M>>, GenerationError> {
        self.stats.lookups += 1;
        self.tick += 1;
        loop {
            let key = self
                .entries
                .keys()
                .filter(|key| prompt.starts_with(key))
                .max_by_key(|key| key.len())
                .cloned();
            let Some(key) = key else { return Ok(None) };
            let entry = self
                .entries
                .get_mut(&key)
                .expect("selected existing prefix");
            entry.used = self.tick;
            let result = match &entry.tier {
                Tier::Device(prefix) => Ok(prefix.clone()),
                Tier::Host(bytes) => {
                    Self::restore(bytes, model, &self.model_id, self.limits, device)
                }
                Tier::Disk(path, _) => fs::read(path).map_err(error).and_then(|bytes| {
                    Self::restore(&bytes, model, &self.model_id, self.limits, device)
                }),
            }
            .and_then(|result| {
                result.validate(&self.model_id, self.limits, prompt, device)?;
                if result.cache.generation_position() != key.len() {
                    return Err(error("prefix cache position differs from token key"));
                }
                Ok(result)
            });
            let result = match result {
                Ok(result) => result,
                Err(e) => {
                    eprintln!("prefix cache restore failed; recomputing uncached suffix: {e}");
                    let failed = self.entries.remove(&key).expect("failed prefix entry");
                    match failed.tier {
                        Tier::Device(_) => self.stats.device_entries -= 1,
                        Tier::Host(bytes) => self.stats.host_bytes -= bytes.len(),
                        Tier::Disk(_, _) => (),
                    }
                    continue;
                }
            };
            self.stats.hits += 1;
            self.stats.reused_tokens += key.len() as u64;
            return Ok(Some(result));
        }
    }

    pub fn insert(
        &mut self,
        prefix: PrefilledPrompt<B, M>,
        device: &B::Device,
    ) -> Result<(), GenerationError> {
        prefix.validate(&self.model_id, self.limits, &prefix.tokens, device)?;
        if prefix.cache.generation_position() != prefix.tokens.len() {
            return Err(error("incomplete prefix cache"));
        }
        if self.config.device_entries == 0
            && self.config.host_bytes == 0
            && self.config.disk_bytes == 0
        {
            return Ok(());
        }
        self.tick += 1;
        if let Some(entry) = self.entries.get_mut(&prefix.tokens) {
            entry.used = self.tick;
            return Ok(());
        }
        self.entries.insert(
            prefix.tokens.clone(),
            Entry {
                tier: Tier::Device(prefix),
                used: self.tick,
            },
        );
        self.stats.device_entries += 1;
        while self.stats.device_entries > self.config.device_entries {
            let key = self
                .entries
                .iter()
                .filter(|(_, e)| matches!(&e.tier, Tier::Device(_)))
                .min_by_key(|(_, e)| e.used)
                .map(|(key, _)| key.clone())
                .expect("over-budget device tier");
            let entry = self.entries.remove(&key).expect("existing device entry");
            self.stats.device_entries -= 1;
            let Tier::Device(prefix) = entry.tier else {
                unreachable!()
            };
            if self.config.host_bytes == 0 && self.config.disk_bytes == 0 {
                continue;
            }
            let bytes = match Self::capture(&prefix, device) {
                Ok(bytes) => bytes,
                Err(e) => {
                    self.entries.insert(
                        key,
                        Entry {
                            tier: Tier::Device(prefix),
                            used: entry.used,
                        },
                    );
                    self.stats.device_entries += 1;
                    return Err(e);
                }
            };
            if bytes.len() <= self.config.host_bytes {
                self.stats.host_bytes += bytes.len();
                self.entries.insert(
                    key,
                    Entry {
                        tier: Tier::Host(bytes),
                        used: entry.used,
                    },
                );
            } else {
                self.put_disk(key, bytes, entry.used)?;
            }
        }
        while self.stats.host_bytes > self.config.host_bytes {
            let key = self
                .entries
                .iter()
                .filter(|(_, e)| matches!(&e.tier, Tier::Host(_)))
                .min_by_key(|(_, e)| e.used)
                .map(|(key, _)| key.clone())
                .expect("over-budget host tier");
            let entry = self.entries.remove(&key).expect("existing host entry");
            let Tier::Host(bytes) = entry.tier else {
                unreachable!()
            };
            self.stats.host_bytes -= bytes.len();
            self.put_disk(key, bytes, entry.used)?;
        }
        Ok(())
    }

    fn put_disk(
        &mut self,
        key: Vec<i32>,
        bytes: Vec<u8>,
        used: u64,
    ) -> Result<(), GenerationError> {
        if bytes.len() > self.config.disk_bytes || self.config.disk_bytes == 0 {
            return Ok(());
        }
        while self.stats.disk_bytes.saturating_add(bytes.len()) > self.config.disk_bytes {
            self.evict_disk()?;
        }
        let directory = self
            .config
            .disk_directory
            .as_ref()
            .expect("validated disk directory");
        let mut serial = self.tick;
        let (path, mut file) = loop {
            let path = directory.join(format!("ruda-kv-{}-{serial}.mpk", std::process::id()));
            match OpenOptions::new().write(true).create_new(true).open(&path) {
                Ok(file) => break (path, file),
                Err(e) if e.kind() == std::io::ErrorKind::AlreadyExists => serial += 1,
                Err(e) => return Err(error(e)),
            }
        };
        if let Err(e) = file.write_all(&bytes).and_then(|()| file.sync_all()) {
            drop(file);
            let _ = fs::remove_file(&path);
            return Err(error(e));
        }
        self.stats.disk_bytes += bytes.len();
        self.entries.insert(
            key,
            Entry {
                tier: Tier::Disk(path, bytes.len()),
                used,
            },
        );
        Ok(())
    }

    fn capture(
        prefix: &PrefilledPrompt<B, M>,
        device: &B::Device,
    ) -> Result<Vec<u8>, GenerationError> {
        B::sync(device).map_err(error)?;
        let record = prefix
            .cache
            .capture_generation(&prefix.model_id, device)
            .map_err(error)?;
        let item = record.into_item::<FullPrecisionSettings>();
        let logits = prefix.logits.clone().try_into_data().map_err(error)?;
        rmp_serde::to_vec(&(
            1u32,
            &prefix.model_id,
            prefix.limits,
            &prefix.tokens,
            item,
            logits,
        ))
        .map_err(error)
    }

    fn restore(
        bytes: &[u8],
        model: &M,
        model_id: &str,
        limits: CausalModelLimits,
        device: &B::Device,
    ) -> Result<PrefilledPrompt<B, M>, GenerationError> {
        let (version, id, saved_limits, tokens, item, logits): (
            u32,
            String,
            CausalModelLimits,
            Vec<i32>,
            <<M::Cache as GenerationCacheContinuation<B>>::Record as Record<B>>::Item<
                FullPrecisionSettings,
            >,
            TensorData,
        ) = rmp_serde::from_slice(bytes).map_err(error)?;
        if version != 1
            || id != model_id
            || saved_limits != limits
            || tokens.is_empty()
            || tokens.len() > limits.max_sequence_length
            || tokens
                .iter()
                .any(|&t| t < 0 || t as usize >= limits.vocab_size)
            || logits.shape.as_slice() != [1, 1, limits.vocab_size]
            || !matches!(
                logits.dtype,
                ruda_tensor::api::DType::F16
                    | ruda_tensor::api::DType::BF16
                    | ruda_tensor::api::DType::F32
                    | ruda_tensor::api::DType::F64
                    | ruda_tensor::api::DType::Flex32
            )
            || limits.vocab_size.checked_mul(logits.dtype.size()) != Some(logits.as_bytes().len())
        {
            return Err(error(
                "offloaded prefix identity, tokens or logits geometry differs",
            ));
        }
        let record = <M::Cache as GenerationCacheContinuation<B>>::Record::from_item::<
            FullPrecisionSettings,
        >(item, device);
        let cache = M::Cache::restore_generation(
            record,
            model_id,
            &model.new_cache(),
            tokens.len(),
            device,
        )
        .map_err(error)?;
        let dtype = logits.dtype;
        let prefix = PrefilledPrompt {
            model_id: id,
            limits,
            tokens,
            cache,
            logits: Tensor::from_data(logits, (device, dtype)),
        };
        prefix.validate(model_id, limits, &prefix.tokens, device)?;
        Ok(prefix)
    }
}
