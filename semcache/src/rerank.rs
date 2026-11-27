//! Cross-encoder check for candidate cache hits.
//!
//! The bi-encoder (embed.rs) embeds each text alone and then averages its
//! token vectors. Word order across the two texts is lost in that average:
//! "convert dollars to euros" and "convert euros to dollars" get almost the
//! same vector. A cross-encoder reads both texts together in one forward
//! pass, as `[CLS] a [SEP] b [SEP]`, so attention compares each word of one
//! text with each word of the other. It costs one forward pass per pair, not
//! one per text, so the cache runs it only on the few candidates that the
//! vector search returns.
//!
//! The model is a BERT classifier in the Hugging Face layout of
//! BertForSequenceClassification with one label: the MiniLM encoder, a pooler
//! (dense layer and tanh on the [CLS] token) and a linear classifier. The
//! Burn trainer (train-burn) writes it, and transformers can load the folder.

use crate::embed::{TokenBatch, configure_tokenizer, tokenize_pairs};
use burn::module::Module;
use burn::nn::{Dropout, DropoutConfig, Linear, LinearConfig};
use burn::tensor::{Device, Tensor, activation};
use burn_store::{KeyRemapper, ModuleSnapshot, PyTorchToBurnAdapter, SafetensorsStore};
use minilm_burn::{MiniLmConfig, MiniLmModel};
use std::path::Path;
use tokenizers::Tokenizer;

/// File in a cross-encoder folder that holds its thresholds.
pub const RERANK_FILE: &str = "rerank.json";

/// Token limit for a pair at serving time. A lookup is skipped when the
/// prompt alone has more than 256 tokens, so a pair has at most 512, the
/// position limit of MiniLM. Training uses shorter pairs (QQP questions).
pub const MAX_PAIR_TOKENS: usize = 512;

#[derive(Module, Debug)]
pub struct CrossEncoder {
    pub bert: MiniLmModel,
    pub pooler: Linear,
    pub dropout: Dropout,
    pub classifier: Linear,
}

impl CrossEncoder {
    /// A new cross-encoder with random weights.
    pub fn init(config: &MiniLmConfig, device: &Device) -> Self {
        let hidden = config.hidden_size;
        Self {
            bert: config.init(device),
            pooler: LinearConfig::new(hidden, hidden).init(device),
            dropout: DropoutConfig::new(config.hidden_dropout_prob).init(),
            classifier: LinearConfig::new(hidden, 1).init(device),
        }
    }

    /// One logit per pair [batch]. A positive logit means "same meaning".
    pub fn forward(&self, batch: TokenBatch) -> Tensor<1> {
        let hidden = self.bert.forward(batch.ids, batch.mask, Some(batch.types)).hidden_states;
        let [n, _, dim] = hidden.dims();
        let cls = hidden.slice([0..n, 0..1, 0..dim]).reshape([n, dim]);
        let pooled = activation::tanh(self.pooler.forward(cls));
        self.classifier.forward(self.dropout.forward(pooled)).reshape([n])
    }

    /// Load weights from a Hugging Face safetensors file. It accepts the
    /// BertForSequenceClassification layout (a trained cross-encoder) and the
    /// plain BertModel layout of a sentence-transformers model, which has no
    /// classifier. Set `allow_missing` to start from such a model. Returns
    /// the names of the tensors that the file did not have.
    pub fn load_weights(&mut self, path: &Path, allow_missing: bool) -> Result<Vec<String>, String> {
        let mut store = SafetensorsStore::from_file(path)
            .with_from_adapter(PyTorchToBurnAdapter)
            .remap(from_hf_names()?)
            .allow_partial(allow_missing);
        let result = self.load_from(&mut store).map_err(|e| format!("{}: {e}", path.display()))?;
        Ok(result.missing.into_iter().map(|(name, _)| name).collect())
    }
}

fn remapper(patterns: Vec<(&str, &str)>) -> Result<KeyRemapper, String> {
    KeyRemapper::from_patterns(patterns).map_err(|e| e.to_string())
}

/// Hugging Face BERT names to the names of `CrossEncoder`. The patterns run
/// in order on each name, like the minilm-burn loader.
fn from_hf_names() -> Result<KeyRemapper, String> {
    remapper(vec![
        // A BertModel checkpoint has no "bert." prefix.
        (r"^(embeddings|encoder|pooler)\.", "bert.$1."),
        (r"^bert\.pooler\.dense", "pooler"),
        (r"encoder\.layer\.([0-9]+)", "encoder.layers.$1"),
        (r"attention\.self\.query", "mha.query"),
        (r"attention\.self\.key", "mha.key"),
        (r"attention\.self\.value", "mha.value"),
        (r"attention\.output\.dense", "mha.output"),
        (r"attention\.output\.LayerNorm", "norm_1"),
        (r"intermediate\.dense", "pwff.linear_inner"),
        (r"(layers\.[0-9]+)\.output\.dense", "$1.pwff.linear_outer"),
        (r"(layers\.[0-9]+)\.output\.LayerNorm", "$1.norm_2"),
        (r"embeddings\.LayerNorm", "embeddings.layer_norm"),
    ])
}

/// The names of `CrossEncoder` to the BertForSequenceClassification layout,
/// for saving. It reverses `from_hf_names`.
pub fn to_hf_names() -> Result<KeyRemapper, String> {
    remapper(vec![
        (r"^bert\.encoder\.layers\.([0-9]+)\.mha\.query", "bert.encoder.layer.$1.attention.self.query"),
        (r"^bert\.encoder\.layers\.([0-9]+)\.mha\.key", "bert.encoder.layer.$1.attention.self.key"),
        (r"^bert\.encoder\.layers\.([0-9]+)\.mha\.value", "bert.encoder.layer.$1.attention.self.value"),
        (r"^bert\.encoder\.layers\.([0-9]+)\.mha\.output", "bert.encoder.layer.$1.attention.output.dense"),
        (r"^bert\.encoder\.layers\.([0-9]+)\.norm_1", "bert.encoder.layer.$1.attention.output.LayerNorm"),
        (r"^bert\.encoder\.layers\.([0-9]+)\.pwff\.linear_inner", "bert.encoder.layer.$1.intermediate.dense"),
        (r"^bert\.encoder\.layers\.([0-9]+)\.pwff\.linear_outer", "bert.encoder.layer.$1.output.dense"),
        (r"^bert\.encoder\.layers\.([0-9]+)\.norm_2", "bert.encoder.layer.$1.output.LayerNorm"),
        (r"^bert\.embeddings\.layer_norm", "bert.embeddings.LayerNorm"),
        (r"^pooler\.", "bert.pooler.dense."),
    ])
}

/// Thresholds of the two-stage check, chosen together on a labeled set.
#[derive(Debug, Clone, Copy)]
pub struct RerankSettings {
    /// Cosine similarity at which a cached entry becomes a candidate.
    pub candidate_threshold: f32,
    /// Cross-encoder probability at which a candidate is a hit.
    pub threshold: f32,
}

impl RerankSettings {
    /// Read rerank.json (written by `semcache-eval --rerank ... --save-threshold`).
    pub fn load(dir: &Path) -> Result<Self, String> {
        let path = dir.join(RERANK_FILE);
        let text = std::fs::read_to_string(&path)
            .map_err(|e| format!("{}: {e}. Run semcache-eval --rerank --save-threshold", path.display()))?;
        let value: serde_json::Value = serde_json::from_str(&text).map_err(|e| format!("{}: {e}", path.display()))?;
        let number = |key: &str| {
            value[key].as_f64().map(|v| v as f32).ok_or_else(|| format!("{}: no numeric \"{key}\"", path.display()))
        };
        Ok(Self { candidate_threshold: number("candidate_threshold")?, threshold: number("threshold")? })
    }
}

pub struct Reranker {
    model: CrossEncoder,
    tokenizer: Tokenizer,
    device: Device,
}

impl Reranker {
    /// Load a cross-encoder folder: config.json, model.safetensors, tokenizer.json.
    pub fn load(dir: &Path) -> Result<Self, String> {
        let device = Device::default();
        let config = MiniLmConfig::load_from_hf(dir.join("config.json")).map_err(|e| e.to_string())?;
        let mut model = CrossEncoder::init(&config, &device);
        model.load_weights(&dir.join("model.safetensors"), false)?;
        let mut tokenizer =
            Tokenizer::from_file(dir.join("tokenizer.json")).map_err(|e| format!("tokenizer.json: {e}"))?;
        configure_tokenizer(&mut tokenizer, MAX_PAIR_TOKENS)?;
        Ok(Self { model, tokenizer, device })
    }

    /// Probability that the two texts of each pair have the same meaning.
    pub fn score(&self, pairs: &[(&str, &str)]) -> Result<Vec<f32>, String> {
        Ok(self.logits(pairs)?.into_iter().map(|x| 1.0 / (1.0 + (-x).exp())).collect())
    }

    /// The raw classifier output of each pair, before the sigmoid.
    pub fn logits(&self, pairs: &[(&str, &str)]) -> Result<Vec<f32>, String> {
        if pairs.is_empty() {
            return Ok(Vec::new());
        }
        let batch = tokenize_pairs(&self.tokenizer, pairs, &self.device)?;
        self.model.forward(batch).into_data().try_to_vec().map_err(|e| format!("tensor data: {e:?}"))
    }
}
