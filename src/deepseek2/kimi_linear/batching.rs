use super::*;
use crate::{ContinuousBatchScheduler, DenseBatchCache, DenseBatchPrefix, DenseBatchDelta, DeviceBatchModel,
    KvPageId, PagedKvCacheConfig, PrefixBatchModel, RequestId, ScheduledBatch, SpeculativeBatchModel};
use std::collections::BTreeMap;
mod prefix;
mod speculative;
pub use speculative::KimiLinearBatchDelta;
use speculative::Trajectory;

#[derive(Clone)]
pub struct KimiLinearBatchCache<B: Backend> {
    dense: DenseBatchCache<B>, recurrent: BTreeMap<RequestId, Vec<kda::State<B>>>, trajectories: BTreeMap<RequestId, Trajectory<B>>,
}
impl<B: Backend> KimiLinearBatchCache<B> {
    pub fn resident_requests(&self) -> usize { self.dense.resident_requests() }
    pub fn sequence_length(&self, id: RequestId) -> Option<usize> { self.dense.sequence_length(id) }
}
#[derive(Clone)]
pub struct KimiLinearBatchPrefix<B: Backend> {dense: DenseBatchPrefix<B>, recurrent: Vec<kda::State<B>>}

impl<B: Backend> KimiLinearModel<B> {
    fn dense_geometry(&self) -> Vec<(usize, usize, usize, DType)> {
        let c = &self.config;
        vec![(c.num_attention_heads, c.qk_nope_head_dim + c.qk_rope_head_dim, c.v_head_dim, self.embedding.dtype()); c.mla_layers()]
    }
    fn empty_recurrent(&self) -> Vec<kda::State<B>> {
        (0..self.config.num_hidden_layers - self.config.mla_layers()).map(|_| kda::State::empty()).collect()
    }
    fn validate_recurrent(&self, states: &[kda::State<B>], completed: bool) -> Result<(), GenerationError> {
        if states.len() != self.config.num_hidden_layers - self.config.mla_layers() { return Err(error("Kimi Linear recurrent layer count differs")); }
        if states.is_empty() { return Ok(()); }
        let c = self.config.linear_attn_config.as_ref().unwrap(); let dtype = self.embedding.dtype(); let device = self.embedding.device();
        for state in states {
            let valid = match (&state.conv, &state.recurrent) {
                (None, None) => !completed,
                (Some(conv), Some(state)) => completed && state.dims() == [1, c.num_heads, c.head_dim, c.head_dim]
                    && state.dtype() == DType::F32 && state.device() == device
                    && conv.iter().all(|v| v.dims() == [1, c.num_heads * c.head_dim, c.short_conv_kernel_size] && v.dtype() == dtype && v.device() == device),
                _ => false,
            };
            if !valid { return Err(error("Kimi Linear recurrent state geometry/device differs")); }
        }
        Ok(())
    }
}
impl<R, F, I, BT> DeviceBatchModel<DeviceBackend<R, F, I, BT>> for KimiLinearModel<DeviceBackend<R, F, I, BT>>
where R: DeviceRuntime, R::Server: ComputeServer, R::Device: DeviceOps, F: FloatElement, I: IntElement, BT: BoolElement {
    type Cache = KimiLinearBatchCache<DeviceBackend<R, F, I, BT>>;
    fn batch_limits(&self) -> CausalModelLimits { self.limits() }
    fn new_batch_cache(&self, config: PagedKvCacheConfig) -> Self::Cache {
        Self::Cache {dense: DenseBatchCache::new(config, self.config.mla_layers()), recurrent: BTreeMap::new(), trajectories: BTreeMap::new()}
    }
    fn fork_batch_cache(&self, cache: &Self::Cache) -> Self::Cache { cache.clone() }
    fn retain_batch_cache(&self, cache: &mut Self::Cache, scheduler: &ContinuousBatchScheduler) {
        cache.dense.retain(scheduler);
        cache.recurrent.retain(|id, _| scheduler.kv_cache().contains_sequence(*id));
        cache.trajectories.retain(|id, _| cache.recurrent.contains_key(id));
    }
    fn forward_batch(&self, batch: &ScheduledBatch, cache: &mut Self::Cache, device: &R::Device)
        -> Result<Tensor<DeviceBackend<R, F, I, BT>, 3>, GenerationError> { self.forward_batch_impl(batch, cache, device, false) }
}
impl<R, F, I, BT> KimiLinearModel<DeviceBackend<R, F, I, BT>>
where R: DeviceRuntime, R::Server: ComputeServer, R::Device: DeviceOps, F: FloatElement, I: IntElement, BT: BoolElement {
    fn forward_batch_impl(&self, batch: &ScheduledBatch, cache: &mut KimiLinearBatchCache<DeviceBackend<R, F, I, BT>>,
        device: &R::Device, speculative: bool) -> Result<Tensor<DeviceBackend<R, F, I, BT>, 3>, GenerationError> {
        let c = &self.config; let (b, s) = (batch.batch_size(), batch.next_n());
        cache.dense.validate(batch, self.limits(), c.mla_layers())?;
        if *device != self.embedding.device() { return Err(error("Kimi Linear batch device differs")); }
        for row in &batch.sequences {
            let states = cache.recurrent.entry(row.request_id).or_insert_with(|| self.empty_recurrent());
            self.validate_recurrent(states, row.start_position > 0)?;
            if speculative {
                let history = cache.trajectories.entry(row.request_id).or_insert_with(|| Trajectory {
                    start: row.start_position, end: row.start_position, initial: states.clone(), steps: Vec::new(),
                });
                if history.end != row.start_position { return Err(error("Kimi Linear speculative history is not contiguous")); }
                history.steps.extend((0..s).map(|_| self.empty_recurrent()));
            } else { cache.trajectories.remove(&row.request_id); }
        }
        let ids = Tensor::<DeviceBackend<R, F, I, BT>, 1, Int>::from_data(TensorData::new(batch.token_matrix(), [b * s]), device);
        let mut hidden = self.embedding.clone().select(0, ids).reshape([b, s, c.hidden_size]);
        let dtype = hidden.dtype(); let mut dense_index = 0; let mut kda_index = 0;
        for layer in &self.layers {
            let input = norm(hidden.clone(), &layer.input_norm, c.rms_norm_eps);
            let update = match &layer.mixer {
                Mixer::Mla(m) => {
                    let (q, k, v) = m.project(input, c);
                    let output = cache.dense.attention(dense_index, batch, q, k, v, ((c.qk_nope_head_dim + c.qk_rope_head_dim) as f32).sqrt().recip())?;
                    dense_index += 1;
                    m.out.forward(output.swap_dims(1, 2).reshape([b, s, c.num_attention_heads * c.v_head_dim]))
                },
                Mixer::Kda(m) => {
                    let a = c.linear_attn_config.as_ref().unwrap(); let (h, d, w) = (a.num_heads, a.head_dim, a.short_conv_kernel_size);
                    let conv = std::array::from_fn(|which| Tensor::cat(batch.sequences.iter().map(|row| {
                        cache.recurrent[&row.request_id][kda_index].conv.as_ref().map(|conv| conv[which].clone())
                            .unwrap_or_else(|| Tensor::zeros([1, h * d, w], (device, dtype)))
                    }).collect(), 0));
                    let recurrent = Tensor::cat(batch.sequences.iter().map(|row| {
                        cache.recurrent[&row.request_id][kda_index].recurrent.clone()
                            .unwrap_or_else(|| Tensor::zeros([1, h, d, d], (device, DType::F32)))
                    }).collect(), 0);
                    let mut state = kda::State {conv: Some(conv), recurrent: Some(recurrent)};
                    let (output, history) = m.forward(input, c, &mut state, speculative)?;
                    for (row_index, row) in batch.sequences.iter().enumerate() {
                        let split = |conv: &[Tensor<DeviceBackend<R, F, I, BT>, 3>; 3], offset: usize| {
                            std::array::from_fn(|j| conv[j].clone().slice([row_index..row_index + 1, 0..h * d, offset..offset + w]))
                        };
                        cache.recurrent.get_mut(&row.request_id).unwrap()[kda_index] = kda::State {
                            conv: Some(split(state.conv.as_ref().unwrap(), 0)),
                            recurrent: Some(state.recurrent.as_ref().unwrap().clone().slice([row_index..row_index + 1, 0..h, 0..d, 0..d])),
                        };
                        if let Some(history) = &history {
                            let trajectory = cache.trajectories.get_mut(&row.request_id).unwrap();
                            let start = row.start_position - trajectory.start;
                            for token in 0..s {
                                trajectory.steps[start + token][kda_index] = kda::State {
                                    conv: Some(split(&history.conv, token + 1)),
                                    recurrent: Some(history.states[token].clone().slice([row_index..row_index + 1, 0..h, 0..d, 0..d])),
                                };
                            }
                        }
                    }
                    kda_index += 1; output
                },
            };
            hidden = hidden + update;
            let input = norm(hidden.clone(), &layer.post_norm, c.rms_norm_eps);
            hidden = hidden + self.feed_forward(layer, input)?;
        }
        if !speculative { hidden = hidden.slice([0..b, s - 1..s, 0..c.hidden_size]); }
        let logits = self.head.forward(norm(hidden, &self.norm, c.rms_norm_eps));
        cache.dense.commit(batch);
        if speculative { for row in &batch.sequences { cache.trajectories.get_mut(&row.request_id).unwrap().end = row.context_length; } }
        Ok(logits)
    }
}
