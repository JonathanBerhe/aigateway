//! Save trained models in the Hugging Face layout.
//!
//! The weights go to model.safetensors in the Hugging Face BERT layout, so
//! both the sidecar (Burn) and PyTorch can load the result.
//! BurnToPyTorchAdapter transposes Linear weights and renames LayerNorm
//! gamma/beta to weight/bias. The key remapper reverses the mapping that
//! the loader applies to Hugging Face weights.
//!
//! - `save`: the bi-encoder, as a sentence-transformers folder.
//! - `save_cross_encoder`: the cross-encoder, as a transformers
//!   BertForSequenceClassification folder with one label.

use burn_store::{BurnToPyTorchAdapter, KeyRemapper, ModuleSnapshot, SafetensorsStore};
use minilm_burn::{HfModelFiles, MiniLmModel};
use semcache::rerank::{CrossEncoder, to_hf_names};
use std::path::Path;

pub const BASE_MODEL: &str = "sentence-transformers/all-MiniLM-L6-v2";

/// Tokenizer files of the base repository, for transformers AutoTokenizer.
const TOKENIZER_FILES: &[&str] = &["tokenizer_config.json", "special_tokens_map.json", "vocab.txt"];

/// Files that make a folder a complete sentence-transformers model.
const SENTENCE_FILES: &[&str] = &["modules.json", "sentence_bert_config.json", "1_Pooling/config.json"];

fn hf_names() -> Result<KeyRemapper, String> {
    KeyRemapper::from_patterns(vec![
        (r"^encoder\.layers\.([0-9]+)\.mha\.query", "encoder.layer.$1.attention.self.query"),
        (r"^encoder\.layers\.([0-9]+)\.mha\.key", "encoder.layer.$1.attention.self.key"),
        (r"^encoder\.layers\.([0-9]+)\.mha\.value", "encoder.layer.$1.attention.self.value"),
        (r"^encoder\.layers\.([0-9]+)\.mha\.output", "encoder.layer.$1.attention.output.dense"),
        (r"^encoder\.layers\.([0-9]+)\.norm_1", "encoder.layer.$1.attention.output.LayerNorm"),
        (r"^encoder\.layers\.([0-9]+)\.pwff\.linear_inner", "encoder.layer.$1.intermediate.dense"),
        (r"^encoder\.layers\.([0-9]+)\.pwff\.linear_outer", "encoder.layer.$1.output.dense"),
        (r"^encoder\.layers\.([0-9]+)\.norm_2", "encoder.layer.$1.output.LayerNorm"),
        (r"^embeddings\.layer_norm", "embeddings.LayerNorm"),
    ])
    .map_err(|e| e.to_string())
}

fn copy(from: &Path, out: &Path, name: &str) -> Result<(), String> {
    let to = out.join(name);
    if let Some(dir) = to.parent() {
        std::fs::create_dir_all(dir).map_err(|e| e.to_string())?;
    }
    std::fs::copy(from, &to).map(|_| ()).map_err(|e| format!("copy {name}: {e}"))
}

/// Copy small files of the base repository (cached by hf-hub) into `out`.
fn download(base_model: &str, files: &[&str], out: &Path) -> Result<(), String> {
    let client = hf_hub::HFClient::builder().build_sync().map_err(|e| e.to_string())?;
    let (owner, name) = hf_hub::split_id(base_model);
    let repo = client.model(owner, name);
    for file in files {
        let path = repo.download_file().filename(*file).send().map_err(|e| format!("download {file}: {e}"))?;
        copy(&path, out, file)?;
    }
    Ok(())
}

pub fn save(model: &MiniLmModel, base: &HfModelFiles, out: &Path) -> Result<(), String> {
    std::fs::create_dir_all(out.join("2_Normalize")).map_err(|e| e.to_string())?;

    // A second run into the same folder, or a better epoch, replaces the file.
    let mut store = SafetensorsStore::from_file(out.join("model.safetensors"))
        .overwrite(true)
        .with_to_adapter(BurnToPyTorchAdapter)
        .remap(hf_names()?);
    model.save_into(&mut store).map_err(|e| format!("save weights: {e}"))?;

    copy(&base.config_path, out, "config.json")?;
    copy(&base.tokenizer_path, out, "tokenizer.json")?;
    download(BASE_MODEL, TOKENIZER_FILES, out)?;
    download(BASE_MODEL, SENTENCE_FILES, out)
}

/// Save the cross-encoder. config.json is the base config with the
/// classifier architecture and one label, so transformers loads the
/// classifier weights too.
pub fn save_cross_encoder(model: &CrossEncoder, base_model: &str, base: &HfModelFiles, out: &Path) -> Result<(), String> {
    std::fs::create_dir_all(out).map_err(|e| e.to_string())?;

    let mut store = SafetensorsStore::from_file(out.join("model.safetensors"))
        .overwrite(true)
        .with_to_adapter(BurnToPyTorchAdapter)
        .remap(to_hf_names()?);
    model.save_into(&mut store).map_err(|e| format!("save weights: {e}"))?;

    let text = std::fs::read_to_string(&base.config_path).map_err(|e| e.to_string())?;
    let mut config: serde_json::Value = serde_json::from_str(&text).map_err(|e| e.to_string())?;
    config["architectures"] = serde_json::json!(["BertForSequenceClassification"]);
    config["id2label"] = serde_json::json!({"0": "LABEL_0"});
    config["label2id"] = serde_json::json!({"LABEL_0": 0});
    // The model is trained for a sigmoid. An MS MARCO base sets Identity for
    // sentence-transformers, which then returns logits, not probabilities.
    if let Some(fields) = config.as_object_mut() {
        fields.remove("sbert_ce_default_activation_function");
        fields.remove("_name_or_path");
    }
    config["sentence_transformers"] = serde_json::json!({"activation_fn": "torch.nn.modules.activation.Sigmoid"});
    let pretty = serde_json::to_string_pretty(&config).map_err(|e| e.to_string())? + "\n";
    std::fs::write(out.join("config.json"), pretty).map_err(|e| e.to_string())?;

    copy(&base.tokenizer_path, out, "tokenizer.json")?;
    download(base_model, TOKENIZER_FILES, out)
}
