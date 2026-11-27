//! Sentence embeddings with MiniLM on Burn.
//!
//! `Embedder` wraps the minilm-burn model and its tokenizer. It returns one
//! L2-normalized vector per input text, so a dot product of two vectors is
//! their cosine similarity.

use burn::tensor::{Device, Int, Tensor};
use minilm_burn::{MiniLmConfig, MiniLmModel, MiniLmVariant, load_pretrained, mean_pooling, normalize_l2};
use std::path::PathBuf;
use tokenizers::{EncodeInput, Tokenizer, TruncationParams};

/// sentence-transformers uses 256 tokens for the all-MiniLM models.
/// Longer input is cut, so the embedding no longer covers the whole text.
pub const MAX_TOKENS: usize = 256;

/// Embedding size of all-MiniLM-L6-v2 and all-MiniLM-L12-v2.
pub const DIM: usize = 384;

#[derive(Debug, Clone)]
pub struct Embedding {
    pub vector: Vec<f32>,
    /// Tokens before truncation, including [CLS] and [SEP].
    pub tokens: usize,
    /// True when the text was longer than MAX_TOKENS and was cut.
    pub truncated: bool,
}

pub struct Embedder {
    model: MiniLmModel,
    tokenizer: Tokenizer,
    device: Device,
}

/// Where the model comes from.
#[derive(Debug, Clone)]
pub enum ModelSource {
    /// A pretrained sentence-transformers model from the Hugging Face Hub.
    Hub(MiniLmVariant),
    /// A local folder with config.json, model.safetensors and tokenizer.json,
    /// for example the output of train/finetune_cache.py.
    Dir(PathBuf),
}

/// Threshold for the pretrained Hub models. On the QQP dev set, MiniLM L6
/// has a precision of 0.94 at 0.97.
pub const HUB_THRESHOLD: f32 = 0.97;

/// File in a model folder that holds the threshold chosen for that model.
pub const THRESHOLD_FILE: &str = "semcache.json";

impl ModelSource {
    pub fn label(&self) -> String {
        match self {
            ModelSource::Hub(v) => format!("MiniLM {v:?}"),
            ModelSource::Dir(p) => p.display().to_string(),
        }
    }

    /// The similarity threshold that belongs to this model. A threshold is
    /// only valid for the model it was measured on, so a model folder carries
    /// its own in semcache.json (written by `semcache-eval --save-threshold`).
    pub fn default_threshold(&self) -> Result<f32, String> {
        let ModelSource::Dir(dir) = self else {
            return Ok(HUB_THRESHOLD);
        };
        let path = dir.join(THRESHOLD_FILE);
        let text = std::fs::read_to_string(&path).map_err(|e| {
            format!("{}: {e}. Run semcache-eval --save-threshold, or pass --threshold", path.display())
        })?;
        let value: serde_json::Value = serde_json::from_str(&text).map_err(|e| format!("{}: {e}", path.display()))?;
        value["threshold"]
            .as_f64()
            .map(|t| t as f32)
            .ok_or_else(|| format!("{}: no numeric \"threshold\"", path.display()))
    }
}

/// "l6", "l12", or the path of a local model folder.
pub fn parse_model(name: &str) -> Result<ModelSource, String> {
    match name.to_ascii_lowercase().as_str() {
        "l6" => Ok(ModelSource::Hub(MiniLmVariant::L6)),
        "l12" => Ok(ModelSource::Hub(MiniLmVariant::L12)),
        _ => {
            let dir = PathBuf::from(name);
            if dir.join("config.json").is_file() {
                Ok(ModelSource::Dir(dir))
            } else {
                Err(format!("'{name}' is not l6, l12 or a folder with config.json"))
            }
        }
    }
}

impl Embedder {
    /// Load the model. A Hub model is downloaded on the first call into the
    /// user cache folder (about 90 MB for L6).
    pub fn load(source: &ModelSource) -> Result<Self, String> {
        let device = Device::default();
        let (model, mut tokenizer) = match source {
            ModelSource::Hub(variant) => {
                MiniLmModel::pretrained(&device, *variant, None).map_err(|e| e.to_string())?
            }
            ModelSource::Dir(dir) => {
                let config = MiniLmConfig::load_from_hf(dir.join("config.json")).map_err(|e| e.to_string())?;
                let mut model = config.init(&device);
                load_pretrained(&mut model, dir.join("model.safetensors")).map_err(|e| e.to_string())?;
                let tokenizer = Tokenizer::from_file(dir.join("tokenizer.json"))
                    .map_err(|e| format!("tokenizer.json: {e}"))?;
                (model, tokenizer)
            }
        };

        configure_tokenizer(&mut tokenizer, MAX_TOKENS)?;
        Ok(Self { model, tokenizer, device })
    }

    /// Embed a batch of texts. The batch is padded to its longest text.
    pub fn embed(&self, texts: &[&str]) -> Result<Vec<Embedding>, String> {
        if texts.is_empty() {
            return Ok(Vec::new());
        }
        let batch = tokenize(&self.tokenizer, texts, &self.device)?;
        let (tokens, truncated) = (batch.tokens.clone(), batch.truncated.clone());
        let flat: Vec<f32> = sentence_embeddings(&self.model, batch)
            .into_data()
            .try_to_vec()
            .map_err(|e| format!("tensor data: {e:?}"))?;

        if flat.len() != texts.len() * DIM {
            return Err(format!("expected {} values, got {}", texts.len() * DIM, flat.len()));
        }
        Ok(flat
            .chunks_exact(DIM)
            .zip(tokens.into_iter().zip(truncated))
            .map(|(v, (tokens, truncated))| Embedding { vector: v.to_vec(), tokens, truncated })
            .collect())
    }
}

/// Pad in `tokenize` to the longest text of the batch, not in the tokenizer,
/// and cut texts at `max_tokens`.
pub fn configure_tokenizer(tokenizer: &mut Tokenizer, max_tokens: usize) -> Result<(), String> {
    tokenizer.with_padding(None);
    tokenizer
        .with_truncation(Some(TruncationParams { max_length: max_tokens, ..Default::default() }))
        .map_err(|e| e.to_string())?;
    Ok(())
}

/// Model input for a batch of texts or text pairs, padded to the longest one.
pub struct TokenBatch {
    pub ids: Tensor<2, Int>,
    /// Segment of each token: 0 for the first text, 1 for the second text of a pair.
    pub types: Tensor<2, Int>,
    pub mask: Tensor<2>,
    /// Tokens of each input before truncation, including [CLS] and [SEP].
    pub tokens: Vec<usize>,
    /// True for each input that was longer than the tokenizer limit.
    pub truncated: Vec<bool>,
}

/// Tokenize `texts` on `device`. The sidecar and the Burn trainer share this
/// function, so training and serving see the same input.
pub fn tokenize(tokenizer: &Tokenizer, texts: &[&str], device: &Device) -> Result<TokenBatch, String> {
    encode(tokenizer, texts.to_vec(), device)
}

/// Tokenize text pairs as `[CLS] a [SEP] b [SEP]`, for the cross-encoder.
/// When a pair is too long, the tokenizer cuts the longer text first.
pub fn tokenize_pairs(tokenizer: &Tokenizer, pairs: &[(&str, &str)], device: &Device) -> Result<TokenBatch, String> {
    encode(tokenizer, pairs.to_vec(), device)
}

fn encode<'s, E: Into<EncodeInput<'s>> + Send>(
    tokenizer: &Tokenizer,
    inputs: Vec<E>,
    device: &Device,
) -> Result<TokenBatch, String> {
    let encodings = tokenizer.encode_batch(inputs, true).map_err(|e| format!("tokenizer: {e}"))?;

    let batch = encodings.len();
    let seq = encodings.iter().map(|e| e.get_ids().len()).max().unwrap_or(1).max(1);
    let mut ids = vec![0i64; batch * seq];
    let mut types = vec![0i64; batch * seq];
    let mut mask = vec![0f32; batch * seq];
    let mut tokens = Vec::with_capacity(batch);
    let mut truncated = Vec::with_capacity(batch);

    for (i, enc) in encodings.iter().enumerate() {
        let rows = enc.get_ids().iter().zip(enc.get_type_ids()).zip(enc.get_attention_mask());
        for (j, ((&id, &kind), &m)) in rows.enumerate() {
            ids[i * seq + j] = id as i64;
            types[i * seq + j] = kind as i64;
            mask[i * seq + j] = m as f32;
        }
        let overflow: usize = enc.get_overflowing().iter().map(|o| o.get_ids().len()).sum();
        tokens.push(enc.get_ids().len() + overflow);
        truncated.push(!enc.get_overflowing().is_empty());
    }

    Ok(TokenBatch {
        ids: Tensor::<1, Int>::from_data(ids.as_slice(), device).reshape([batch, seq]),
        types: Tensor::<1, Int>::from_data(types.as_slice(), device).reshape([batch, seq]),
        mask: Tensor::<1>::from_data(mask.as_slice(), device).reshape([batch, seq]),
        tokens,
        truncated,
    })
}

/// Sentence embeddings [batch, DIM]: mean pooling over real tokens, then L2
/// normalization, as sentence-transformers does for all-MiniLM.
pub fn sentence_embeddings(model: &MiniLmModel, batch: TokenBatch) -> Tensor<2> {
    let output = model.forward(batch.ids, batch.mask.clone(), None);
    normalize_l2(mean_pooling(output.hidden_states, batch.mask))
}
