//! Low-rank adapters for the packed Llama/Qwen2 projections.
//!
//! Each adapted projection computes `y = x W + (alpha / rank) * (x A) B` with
//! `A [in, rank]` (Gaussian, std `1/sqrt(in)`) and `B [rank, out]` (zeros), so
//! training starts exactly at the base model. Adapter parameters are kept in
//! FP32 regardless of the base dtype and are cast at the projection
//! boundary. The base model is trained frozen (`Module::no_grad`).

use ruda_model::module::{Module, Param};
use ruda_tensor::api::backend::Backend;
use ruda_tensor::api::module::linear;
use ruda_tensor::Distribution;
use ruda_tensor::api::{DType, Tensor};

/// Which packed projections receive an adapter.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct LoraTargets {
    pub qkv: bool,
    pub output: bool,
    pub gate_up: bool,
    pub down: bool,
}

impl LoraTargets {
    pub const ALL: Self = Self {
        qkv: true,
        output: true,
        gate_up: true,
        down: true,
    };
    pub const ATTENTION: Self = Self {
        qkv: true,
        output: true,
        gate_up: false,
        down: false,
    };
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct LoraConfig {
    pub rank: usize,
    pub alpha: usize,
    pub targets: LoraTargets,
    pub seed: u64,
}

impl Default for LoraConfig {
    fn default() -> Self {
        Self {
            rank: 16,
            alpha: 32,
            targets: LoraTargets::ALL,
            seed: 0,
        }
    }
}

/// One low-rank update `A [in, rank] @ B [rank, out]`.
#[derive(Module, Debug)]
pub struct LoraPair<B: Backend> {
    pub a: Param<Tensor<B, 2>>,
    pub b: Param<Tensor<B, 2>>,
}

impl<B: Backend> LoraPair<B> {
    fn new(input: usize, output: usize, rank: usize, device: &B::Device) -> Self {
        let std = 1.0 / (input as f64).sqrt();
        Self {
            a: Param::from_tensor(
                Tensor::<B, 2>::random([input, rank], Distribution::Normal(0.0, std), device)
                    .cast(DType::F32)
                    .detach(),
            ),
            b: Param::from_tensor(Tensor::<B, 2>::zeros([rank, output], (device, DType::F32)).detach()),
        }
    }

    /// `scale * (x A) B`, computed in FP32 and returned in `x`'s dtype.
    pub fn delta<const D: usize>(&self, input: Tensor<B, D>, scale: f64) -> Tensor<B, D> {
        let dtype = input.dtype();
        let hidden = linear(input.cast(DType::F32), self.a.val(), None);
        linear(hidden, self.b.val(), None)
            .mul_scalar(scale)
            .cast(dtype)
    }

    /// Dense `scale * A B` in `dtype`, for merging into a base weight.
    pub fn merged(&self, scale: f64, dtype: DType) -> Tensor<B, 2> {
        self.a.val().matmul(self.b.val()).mul_scalar(scale).cast(dtype)
    }
}

/// Adapters of one decoder layer.
#[derive(Module, Debug)]
pub struct LoraLayer<B: Backend> {
    pub qkv: Option<LoraPair<B>>,
    pub output: Option<LoraPair<B>>,
    pub gate_up: Option<LoraPair<B>>,
    pub down: Option<LoraPair<B>>,
}

/// Adapters for every decoder layer of a packed model.
#[derive(Module, Debug)]
pub struct LlamaLora<B: Backend> {
    pub layers: Vec<LoraLayer<B>>,
    rank: usize,
    alpha: usize,
}

/// Projection widths needed to size adapters.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct LoraShapes {
    pub layers: usize,
    pub hidden: usize,
    pub qkv_out: usize,
    pub attention_out: usize,
    pub intermediate: usize,
}

impl<B: Backend> LlamaLora<B> {
    pub fn new(shapes: LoraShapes, config: LoraConfig, device: &B::Device) -> Self {
        assert!(config.rank > 0, "LoRA rank must be positive");
        B::seed(device, config.seed);
        let targets = config.targets;
        let pair = |enabled: bool, input: usize, output: usize| {
            enabled.then(|| LoraPair::new(input, output, config.rank, device))
        };
        let layers = (0..shapes.layers)
            .map(|_| LoraLayer {
                qkv: pair(targets.qkv, shapes.hidden, shapes.qkv_out),
                output: pair(targets.output, shapes.attention_out, shapes.hidden),
                gate_up: pair(targets.gate_up, shapes.hidden, 2 * shapes.intermediate),
                down: pair(targets.down, shapes.intermediate, shapes.hidden),
            })
            .collect();
        Self {
            layers,
            rank: config.rank,
            alpha: config.alpha,
        }
    }

    /// `alpha / rank`.
    pub fn scale(&self) -> f64 {
        self.alpha as f64 / self.rank as f64
    }

    pub const fn rank(&self) -> usize {
        self.rank
    }

    pub const fn alpha(&self) -> usize {
        self.alpha
    }
}

/// Adds an optional adapter's delta to a base projection output.
pub(crate) fn with_lora<B: Backend, const D: usize>(
    base: Tensor<B, D>,
    input: Tensor<B, D>,
    pair: Option<&LoraPair<B>>,
    scale: f64,
) -> Tensor<B, D> {
    match pair {
        Some(pair) => base + pair.delta(input, scale),
        None => base,
    }
}
