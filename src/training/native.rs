use super::{LoraConfig, LoraPair, SftModel, TrainError, TrainingBackend, linear_cross_entropy};
use crate::{DeepseekV2Config, KimiLinearConfig, Qwen3Config, Qwen3MoeConfig, Qwen35TextConfig};
use ruda_model::module::{Module, Param};
use ruda_tensor::api::{backend::{AutodiffBackend, Backend}, DType, Int, Tensor, TensorData};
use std::{collections::BTreeMap, path::Path};

mod attention;
mod recurrent;
mod feed_forward;

#[derive(Clone, Debug)]
pub(crate) enum Architecture {
    Qwen3(Qwen3Config, Option<Qwen3MoeConfig>),
    Qwen35(Qwen35TextConfig),
    Mla(DeepseekV2Config),
    KimiLinear(KimiLinearConfig),
}

impl Architecture {
    fn geometry(&self) -> (usize, usize, usize, usize, f64, bool) {
        match self {
            Self::Qwen3(c, _) => (c.vocab_size, c.hidden_size, c.num_hidden_layers, c.max_position_embeddings, c.rms_norm_eps, c.tie_word_embeddings),
            Self::Qwen35(c) => (c.vocab_size, c.hidden_size, c.num_hidden_layers, c.max_position_embeddings, c.rms_norm_eps, c.tie_word_embeddings),
            Self::Mla(c) => (c.vocab_size, c.hidden_size, c.num_hidden_layers, c.max_position_embeddings, c.rms_norm_eps, c.tie_word_embeddings),
            Self::KimiLinear(c) => (c.vocab_size, c.hidden_size, c.num_hidden_layers, c.model_max_length, c.rms_norm_eps, c.tie_word_embeddings),
        }
    }
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(crate) enum ProjectionTarget {
    None,
    AttentionInput,
    AttentionOutput,
    FeedForwardInput,
    FeedForwardOutput,
}

/// Trainable native weights in Hugging Face-compatible layouts. Inference
/// caches and packed-kernel closures are deliberately not part of this module.
#[derive(Module, Debug)]
pub struct NativeSftModel<B: Backend> {
    vectors: Vec<Param<Tensor<B, 1>>>,
    matrices: Vec<Param<Tensor<B, 2>>>,
    convolutions: Vec<Param<Tensor<B, 3>>>,
    states: Vec<Param<Tensor<B, 4>>>,
    #[module(skip)]
    layout: Layout,
}

#[derive(Clone, Debug)]
struct Layout {
    architecture: Architecture,
    vectors: BTreeMap<String, usize>,
    matrices: BTreeMap<String, usize>,
    convolutions: BTreeMap<String, usize>,
    states: BTreeMap<String, usize>,
    targets: Vec<ProjectionTarget>,
    base: String,
}

/// One adapter slot per registered matrix; routers, embeddings and norms are
/// excluded. Expert and recurrent-mixer projections retain independent pairs.
#[derive(Module, Debug)]
pub struct NativeLora<B: Backend> {
    pairs: Vec<Option<LoraPair<B>>>,
    #[module(skip)]
    names: BTreeMap<String, usize>,
    rank: usize,
    alpha: usize,
}

impl<B: Backend> NativeLora<B> {
    pub fn scale(&self) -> f64 { self.alpha as f64 / self.rank as f64 }
    pub fn rank(&self) -> usize { self.rank }
    pub fn alpha(&self) -> usize { self.alpha }
}

impl<B: Backend> NativeSftModel<B> {
    pub(crate) fn new(architecture: Architecture, base: String) -> Self {
        Self {
            vectors: Vec::new(), matrices: Vec::new(), convolutions: Vec::new(), states: Vec::new(),
            layout: Layout { architecture, base, vectors: BTreeMap::new(), matrices: BTreeMap::new(),
                convolutions: BTreeMap::new(), states: BTreeMap::new(), targets: Vec::new() },
        }
    }

    pub(crate) fn add_vector(&mut self, name: String, tensor: Tensor<B, 1>) {
        assert!(!self.layout.vectors.contains_key(&name), "duplicate parameter {name}");
        self.layout.vectors.insert(name, self.vectors.len());
        self.vectors.push(Param::from_tensor(tensor.detach()));
    }

    pub(crate) fn add_constant(&mut self, name: String, tensor: Tensor<B, 1>) {
        self.add_vector(name, tensor);
        let last = self.vectors.pop().expect("registered constant");
        self.vectors.push(last.set_require_grad(false));
    }

    pub(crate) fn add_matrix(&mut self, name: String, tensor: Tensor<B, 2>, target: ProjectionTarget) {
        assert!(!self.layout.matrices.contains_key(&name), "duplicate parameter {name}");
        self.layout.matrices.insert(name, self.matrices.len());
        self.matrices.push(Param::from_tensor(tensor.detach()));
        self.layout.targets.push(target);
    }

    pub(crate) fn add_convolution(&mut self, name: String, tensor: Tensor<B, 3>) {
        self.layout.convolutions.insert(name, self.convolutions.len());
        self.convolutions.push(Param::from_tensor(tensor.detach()));
    }

    pub(crate) fn add_state(&mut self, name: String, tensor: Tensor<B, 4>) {
        self.layout.states.insert(name, self.states.len());
        self.states.push(Param::from_tensor(tensor.detach()));
    }

    pub fn vocabulary_size(&self) -> usize { self.layout.architecture.geometry().0 }
    pub fn max_sequence_length(&self) -> usize { self.layout.architecture.geometry().3 }

    pub fn lora(&self, config: LoraConfig, device: &B::Device) -> NativeLora<B> {
        assert!(config.rank > 0, "LoRA rank must be positive");
        B::seed(device, config.seed);
        let pairs = self.matrices.iter().zip(&self.layout.targets).map(|(weight, target)| {
            let enabled = match target {
                ProjectionTarget::None => false,
                ProjectionTarget::AttentionInput => config.targets.qkv,
                ProjectionTarget::AttentionOutput => config.targets.output,
                ProjectionTarget::FeedForwardInput => config.targets.gate_up,
                ProjectionTarget::FeedForwardOutput => config.targets.down,
            };
            let [output, input] = weight.dims();
            enabled.then(|| LoraPair::new(input, output, config.rank, device))
        }).collect();
        NativeLora { pairs, names: self.layout.matrices.clone(), rank: config.rank, alpha: config.alpha }
    }

    fn check_adapters(&self, adapters: Option<&NativeLora<B>>) -> Result<(), TrainError> {
        if let Some(adapters) = adapters {
            if adapters.names != self.layout.matrices || adapters.pairs.len() != self.matrices.len() {
                return Err(TrainError("LoRA adapters do not match the model's projections".into()));
            }
            for (weight, pair) in self.matrices.iter().zip(&adapters.pairs) {
                if let Some(pair) = pair {
                    let [output, input] = weight.dims();
                    if pair.a.dims() != [input, adapters.rank] || pair.b.dims() != [adapters.rank, output] {
                        return Err(TrainError("LoRA projection dimensions differ from the base model".into()));
                    }
                }
            }
        }
        Ok(())
    }

    fn vector(&self, name: &str) -> Tensor<B, 1> { self.vectors[self.layout.vectors[name]].val() }
    fn matrix(&self, name: &str) -> Tensor<B, 2> { self.matrices[self.layout.matrices[name]].val() }
    fn convolution(&self, name: &str) -> Tensor<B, 3> { self.convolutions[self.layout.convolutions[name]].val() }

    fn project(&self, prefix: &str, input: Tensor<B, 3>, adapters: Option<&NativeLora<B>>) -> Tensor<B, 3> {
        let index = self.layout.matrices[&format!("{prefix}.weight")];
        let weight = self.matrices[index].val();
        let dtype = input.dtype();
        let mut output = input.clone().matmul(weight.cast(dtype).transpose().unsqueeze::<3>());
        if let Some(&index) = self.layout.vectors.get(&format!("{prefix}.bias")) {
            output = output + self.vectors[index].val().cast(dtype).unsqueeze::<3>();
        }
        if let Some(adapters) = adapters {
            if let Some(pair) = &adapters.pairs[index] {
                output = output + pair.delta(input, adapters.scale());
            }
        }
        output
    }

    fn normalized<const D: usize>(&self, input: Tensor<B, D>, prefix: &str, epsilon: f64, offset: bool) -> Tensor<B, D> {
        let weight = self.vector(&format!("{prefix}.weight"));
        if offset {
            let dtype = input.dtype();
            let input = input.cast(DType::F32);
            let variance = input.clone().square().mean_dim(D - 1);
            ((input * (variance + epsilon).rsqrt()) * (weight.cast(DType::F32) + 1.0).unsqueeze::<D>()).cast(dtype)
        } else {
            rms_norm(input, weight, epsilon)
        }
    }

    pub fn merge_lora(mut self, adapters: &NativeLora<B>) -> Result<Self, TrainError> {
        self.check_adapters(Some(adapters))?;
        for (weight, pair) in self.matrices.iter_mut().zip(&adapters.pairs) {
            if let Some(pair) = pair {
                let value = weight.val();
                let merged = value.clone() + pair.merged(adapters.scale(), value.dtype()).transpose();
                *weight = Param::initialized(weight.id, merged.detach()).set_require_grad(weight.is_require_grad());
            }
        }
        Ok(self)
    }

    /// Floating-point weights under Hugging Face-compatible tensor names.
    pub fn huggingface_tensors(&self) -> Vec<(String, TensorData)> {
        let mut tensors = Vec::new();
        for (name, &index) in &self.layout.vectors { tensors.push((name.clone(), self.vectors[index].val().into_data())); }
        for (name, &index) in &self.layout.matrices { tensors.push((name.clone(), self.matrices[index].val().into_data())); }
        for (name, &index) in &self.layout.convolutions { tensors.push((name.clone(), self.convolutions[index].val().into_data())); }
        for (name, &index) in &self.layout.states { tensors.push((name.clone(), self.states[index].val().into_data())); }
        tensors
    }

    pub fn export_huggingface_safetensors(&self, path: impl AsRef<Path>) -> Result<(), String> {
        super::write_safetensors(path, &self.huggingface_tensors(), &[("format", "pt")])
    }
}

fn rms_norm<B: Backend, const D: usize>(input: Tensor<B, D>, weight: Tensor<B, 1>, epsilon: f64) -> Tensor<B, D> {
    let dtype = input.dtype();
    let input = input.cast(DType::F32);
    let variance = input.clone().square().mean_dim(D - 1);
    (input * (variance + epsilon).rsqrt()).cast(dtype) * weight.cast(dtype).unsqueeze::<D>()
}

impl<B: TrainingBackend> NativeSftModel<B> {
    /// A fresh full-sequence graph for every batch; recurrent states never leak
    /// between sequences or micro-batches and remain attached during backward.
    pub fn forward_train_hidden(&self, tokens: Tensor<B, 2, Int>, adapters: Option<&NativeLora<B>>)
        -> Result<Tensor<B, 3>, TrainError> {
        self.check_adapters(adapters)?;
        let [batch, sequence] = tokens.dims();
        let (_, width, layers, capacity, epsilon, _) = self.layout.architecture.geometry();
        if batch == 0 || sequence == 0 || sequence > capacity {
            return Err(TrainError("invalid training batch or context length".into()));
        }
        let base = &self.layout.base;
        let mut hidden = self.matrix(&format!("{base}.embed_tokens.weight"))
            .select(0, tokens.reshape([batch * sequence])).reshape([batch, sequence, width]);
        let offset = matches!(&self.layout.architecture, Architecture::Qwen35(_));
        for layer in 0..layers {
            let prefix = format!("{base}.layers.{layer}");
            let input = self.normalized(hidden.clone(), &format!("{prefix}.input_layernorm"), epsilon, offset);
            let update = match &self.layout.architecture {
                Architecture::Qwen3(c, _) => self.qwen_attention(input, &prefix, c, layer, adapters),
                Architecture::Qwen35(c) => match c.layer_types[layer] {
                    crate::Qwen35LayerType::FullAttention => self.qwen35_attention(input, &prefix, c, adapters),
                    crate::Qwen35LayerType::LinearAttention => self.gated_delta(input, &prefix, c, adapters),
                },
                Architecture::Mla(c) => self.mla_attention(input, &prefix, c, adapters),
                Architecture::KimiLinear(c) => {
                    if c.is_kda_layer(layer) { self.kda(input, &prefix, c, adapters) }
                    else { self.kimi_mla_attention(input, &prefix, c, adapters) }
                }
            };
            hidden = hidden + update;
            let input = self.normalized(hidden.clone(), &format!("{prefix}.post_attention_layernorm"), epsilon, offset);
            hidden = hidden + self.feed_forward(input, &prefix, layer, adapters)?;
        }
        Ok(self.normalized(hidden, &format!("{base}.norm"), epsilon, offset))
    }

    pub fn train_loss(&self, tokens: Tensor<B, 2, Int>, targets: Tensor<B, 1, Int>, normalizer: usize,
        adapters: Option<&NativeLora<B>>, chunk_rows: usize) -> Result<Tensor<B, 1>, TrainError> {
        let hidden = self.forward_train_hidden(tokens, adapters)?;
        let [batch, sequence, width] = hidden.dims();
        let head = if self.layout.architecture.geometry().5 {
            self.matrix(&format!("{}.embed_tokens.weight", self.layout.base))
        } else { self.matrix("lm_head.weight") };
        Ok(linear_cross_entropy(hidden.reshape([batch * sequence, width]), head.transpose(), targets, normalizer, chunk_rows))
    }
}

impl<B: AutodiffBackend + TrainingBackend> SftModel<B> for NativeSftModel<B> {
    type Adapters = NativeLora<B>;
    fn vocabulary_size(&self) -> usize { self.vocabulary_size() }
    fn sft_loss(&self, tokens: Tensor<B, 2, Int>, targets: Tensor<B, 1, Int>, normalizer: usize,
        adapters: Option<&Self::Adapters>, chunk_rows: usize) -> Result<Tensor<B, 1>, TrainError> {
        self.train_loss(tokens, targets, normalizer, adapters, chunk_rows)
    }
}
