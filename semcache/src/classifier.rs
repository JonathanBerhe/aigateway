//! A fine-tuned MiniLM classifier of one text: BERT, pooler and one logit.
//!
//! It is the network of the cross-encoder (semcache::rerank), fed one text
//! instead of a pair. The classifier router and the guardrail use it. A
//! folder holds the model in the Hugging Face layout (config.json,
//! model.safetensors, tokenizer.json), as train-burn saves it.

use crate::embed::{configure_tokenizer, tokenize};
use crate::rerank::CrossEncoder;
use burn::tensor::Device;
use minilm_burn::MiniLmConfig;
use std::path::Path;
use tokenizers::Tokenizer;

/// A classifier loaded for inference on the default (CPU) device.
pub struct TextClassifier {
    model: CrossEncoder,
    tokenizer: Tokenizer,
    device: Device,
}

impl TextClassifier {
    /// Load a model folder. Texts are cut at `max_tokens` tokens.
    pub fn load(dir: &Path, max_tokens: usize) -> Result<Self, String> {
        let device = Device::default();
        let config = MiniLmConfig::load_from_hf(dir.join("config.json")).map_err(|e| e.to_string())?;
        let mut model = CrossEncoder::init(&config, &device);
        model.load_weights(&dir.join("model.safetensors"), false)?;
        let mut tokenizer =
            Tokenizer::from_file(dir.join("tokenizer.json")).map_err(|e| format!("tokenizer.json: {e}"))?;
        configure_tokenizer(&mut tokenizer, max_tokens)?;
        Ok(Self { model, tokenizer, device })
    }

    /// Probability of the positive class for each text, from 0 to 1.
    pub fn score(&self, texts: &[&str]) -> Result<Vec<f32>, String> {
        if texts.is_empty() {
            return Ok(Vec::new());
        }
        let batch = tokenize(&self.tokenizer, texts, &self.device)?;
        let logits: Vec<f32> = self.model.forward(batch).into_data().try_to_vec().map_err(|e| format!("tensor data: {e:?}"))?;
        Ok(logits.into_iter().map(|x| 1.0 / (1.0 + (-x).exp())).collect())
    }
}
