use super::{LlamaLora, TrainError, TrainingBackend};
use crate::PackedLlamaForCausalLm;
use ruda_model::module::AutodiffModule;
use ruda_tensor::api::{backend::AutodiffBackend, Int, Tensor};

/// A causal model and its architecture-specific adapters accepted by the SFT driver.
pub trait SftModel<B: AutodiffBackend + TrainingBackend>: AutodiffModule<B> {
    type Adapters: AutodiffModule<B>;

    fn vocabulary_size(&self) -> usize;

    fn sft_loss(
        &self,
        tokens: Tensor<B, 2, Int>,
        targets: Tensor<B, 1, Int>,
        normalizer: usize,
        adapters: Option<&Self::Adapters>,
        chunk_rows: usize,
    ) -> Result<Tensor<B, 1>, TrainError>;
}

impl<B: AutodiffBackend + TrainingBackend> SftModel<B> for PackedLlamaForCausalLm<B> {
    type Adapters = LlamaLora<B>;

    fn vocabulary_size(&self) -> usize {
        self.geometry().vocab_size
    }

    fn sft_loss(
        &self,
        tokens: Tensor<B, 2, Int>,
        targets: Tensor<B, 1, Int>,
        normalizer: usize,
        adapters: Option<&Self::Adapters>,
        chunk_rows: usize,
    ) -> Result<Tensor<B, 1>, TrainError> {
        Ok(self.train_loss(tokens, targets, normalizer, adapters, chunk_rows))
    }
}
