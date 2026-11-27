//! Training of a single-text classifier: the network of the cross-encoder
//! (BERT, pooler, one logit) on one text. finetune-router and finetune-guard
//! use it. Each binary keeps its own data, dev metrics and saving.
//!
//! The recipe:
//! - start from a sentence-transformers model with a new classifier head
//! - binary cross-entropy on targets in [0, 1] (0 and 1, or soft targets)
//! - AdamW with weight decay, linear warmup then linear decay to 0,
//!   gradient clipping by norm
//! - after each epoch, the dev scores of the model without autodiff, so
//!   dropout is off

use burn::grad_clipping::GradientClippingConfig;
use burn::module::Module;
use burn::optim::{AdamWConfig, GradientsParams};
use burn::tensor::Device;
use minilm_burn::{HfModelFiles, MiniLmConfig, download_hf_model};
use semcache::embed::{configure_tokenizer, tokenize};
use semcache::rerank::CrossEncoder;
use semcache::router::sigmoid;
use tokenizers::Tokenizer;

use crate::{Progress, data, learning_rate, loss};

#[derive(clap::Args, Debug)]
pub struct ClassifierArgs {
    /// Hugging Face model to start from (BERT layout with model.safetensors)
    #[arg(long, default_value = "sentence-transformers/all-MiniLM-L6-v2")]
    pub base: String,

    #[arg(long, default_value_t = 32)]
    pub batch_size: usize,

    #[arg(long, default_value_t = 3e-5)]
    pub lr: f64,

    #[arg(long, default_value_t = 0.01)]
    pub weight_decay: f32,

    /// Token limit of a prompt, at training and at serving time
    #[arg(long, default_value_t = 256)]
    pub max_seq_length: usize,

    /// Share of the steps used for linear warmup
    #[arg(long, default_value_t = 0.1)]
    pub warmup: f64,

    /// Gradient norm clipping (0 turns it off)
    #[arg(long, default_value_t = 1.0)]
    pub grad_clip: f32,

    #[arg(long, default_value_t = 42)]
    pub seed: u64,
}

/// What `train` hands to the binary after each epoch.
pub struct Epoch<'a> {
    /// 1-based.
    pub number: usize,
    /// The model without autodiff, for saving.
    pub model: &'a CrossEncoder,
    /// Score of each dev text, from 0 to 1.
    pub dev_scores: Vec<f32>,
    /// Files of the base model, for export::save_cross_encoder.
    pub base: &'a HfModelFiles,
}

/// Scores of `texts`, in batches of similar length.
fn score(model: &CrossEncoder, tokenizer: &Tokenizer, device: &Device, texts: &[String]) -> Result<Vec<f32>, String> {
    let mut order: Vec<usize> = (0..texts.len()).collect();
    order.sort_by_key(|&i| texts[i].len());
    let mut scores = vec![0f32; texts.len()];
    for chunk in order.chunks(64) {
        let batch: Vec<&str> = chunk.iter().map(|&i| texts[i].as_str()).collect();
        let logits: Vec<f32> =
            model.forward(tokenize(tokenizer, &batch, device)?).into_data().try_to_vec().map_err(|e| format!("{e:?}"))?;
        for (&i, x) in chunk.iter().zip(logits) {
            scores[i] = sigmoid(x);
        }
    }
    Ok(scores)
}

/// Train for `epochs` epochs on (text, target) pairs and call `on_epoch`
/// after each epoch with the dev scores. The default device is Metal.
pub fn train(
    args: &ClassifierArgs,
    epochs: usize,
    log_every: usize,
    mut train: Vec<(String, f32)>,
    dev: &[String],
    mut on_epoch: impl FnMut(Epoch) -> Result<(), String>,
) -> Result<(), Box<dyn std::error::Error>> {
    data::shuffle(&mut train, args.seed);
    let device = Device::default().autodiff();
    device.seed(args.seed);
    let eval_device = device.clone().inner();
    println!("device: {device:?}");

    let base = download_hf_model(&args.base, None)?;
    let config = MiniLmConfig::load_from_hf(&base.config_path)?;
    let mut model = CrossEncoder::init(&config, &device);
    let missing = model.load_weights(&base.weights_path, true)?;
    if missing.iter().any(|name| !name.starts_with("classifier.")) {
        return Err(format!("the base model misses more than the classifier: {missing:?}").into());
    }
    println!("loaded {}, new: {}", args.base, missing.join(", "));
    let mut tokenizer = Tokenizer::from_file(&base.tokenizer_path).map_err(|e| e.to_string())?;
    configure_tokenizer(&mut tokenizer, args.max_seq_length)?;

    let clipping = (args.grad_clip > 0.0).then_some(GradientClippingConfig::Norm(args.grad_clip));
    let mut optim = AdamWConfig::new()
        .with_beta_1(0.9)
        .with_beta_2(0.999)
        .with_epsilon(1e-8)
        .with_weight_decay(args.weight_decay)
        .with_grad_clipping(clipping)
        .init();

    let total = train.len().div_ceil(args.batch_size) * epochs;
    let warmup = (total as f64 * args.warmup).ceil() as usize;
    println!("{total} steps, {warmup} warmup steps, batch size {}", args.batch_size);

    let mut step = 0;
    let mut progress = Progress::new(log_every, total);
    for epoch in 0..epochs {
        if epoch > 0 {
            data::shuffle(&mut train, args.seed + epoch as u64);
        }
        for batch in train.chunks(args.batch_size) {
            let texts: Vec<&str> = batch.iter().map(|(t, _)| t.as_str()).collect();
            let targets: Vec<f32> = batch.iter().map(|(_, y)| *y).collect();
            let logits = model.forward(tokenize(&tokenizer, &texts, &device)?);
            let loss = loss::bce_with_logits_soft(logits, &targets);
            let loss_value = f64::from(loss.clone().into_scalar::<f32>());
            let grads = GradientsParams::from_grads(loss.backward(), &model);
            model = optim.step(learning_rate(step, total, warmup, args.lr), model, grads);
            step += 1;
            progress.step(step, loss_value, learning_rate(step.min(total - 1), total, warmup, args.lr));
        }
        let valid = model.valid();
        let dev_scores = score(&valid, &tokenizer, &eval_device, dev)?;
        on_epoch(Epoch { number: epoch + 1, model: &valid, dev_scores, base: &base })?;
    }
    Ok(())
}
