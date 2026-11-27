//! Fine-tune all-MiniLM-L6-v2 for the semantic cache with Burn.
//!
//! It follows the recipe of train/finetune_cache.py, the PyTorch reference,
//! so the two results can be compared on the same evaluation:
//! - QQP train pairs, minus each pair that shares a question with dev or test
//! - OnlineContrastiveLoss with margin 0.5
//! - AdamW: lr 2e-5, betas 0.9 and 0.999, eps 1e-8, no weight decay
//! - linear warmup over 10% of the steps, then linear decay to 0
//! - gradient clipping at norm 1, batch size 64, texts cut at 128 tokens
//!
//! One known difference: Burn clips the norm of each parameter gradient,
//! while the Hugging Face trainer clips the norm of all gradients together.
//!
//! Run from the repository root:
//!   cargo run --release --manifest-path train-burn/Cargo.toml -- --out models/minilm-l6-qqp-burn
//! The default device is Metal. BURN_DEVICE=flex trains on the CPU (slow).

use burn::grad_clipping::GradientClippingConfig;
use burn::module::Module;
use burn::optim::{AdamWConfig, GradientsParams};
use burn::tensor::Device;
use clap::Parser;
use minilm_burn::{MiniLmConfig, download_hf_model, load_pretrained};
use semcache::embed::{DIM, configure_tokenizer, sentence_embeddings, tokenize};
use std::path::PathBuf;
use std::time::Instant;
use tokenizers::Tokenizer;
use train_burn::{Progress, data, export, learning_rate, loss};

#[derive(Parser, Debug)]
#[command(about = "Fine-tune MiniLM L6 for the semantic cache with Burn")]
struct Args {
    /// Raw QQP train split (train/export_qqp.py)
    #[arg(long, default_value = "data/qqp-train-raw.tsv")]
    train: PathBuf,

    /// Pair files to keep out of training (dev and test)
    #[arg(long, num_args = 1.., default_values = ["data/qqp-dev.tsv", "data/qqp-test.tsv"])]
    holdout: Vec<PathBuf>,

    /// Output model folder
    #[arg(long, default_value = "models/minilm-l6-qqp-burn")]
    out: PathBuf,

    /// Cap on training pairs, 0 = all
    #[arg(long, default_value_t = 0)]
    max_train: usize,

    #[arg(long, default_value_t = 1)]
    epochs: usize,

    #[arg(long, default_value_t = 64)]
    batch_size: usize,

    #[arg(long, default_value_t = 2e-5)]
    lr: f64,

    /// Cosine distance margin for negatives
    #[arg(long, default_value_t = 0.5)]
    margin: f32,

    #[arg(long, default_value_t = 128)]
    max_seq_length: usize,

    /// Share of the steps used for linear warmup
    #[arg(long, default_value_t = 0.1)]
    warmup: f64,

    /// Gradient norm clipping, per parameter (0 turns it off)
    #[arg(long, default_value_t = 1.0)]
    grad_clip: f32,

    #[arg(long, default_value_t = 42)]
    seed: u64,

    #[arg(long, default_value_t = 250)]
    log_every: usize,
}

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let args = Args::parse();
    let started = Instant::now();

    // Data.
    let raw = data::read_pairs(&args.train)?;
    let holdout = args.holdout.iter().map(|p| data::read_pairs(p)).collect::<Result<Vec<_>, _>>()?;
    let total_raw = raw.len();
    let (mut train, removed) = data::leak_free(raw, &holdout);
    println!("train pairs: {} of {total_raw} ({removed} removed because they share a question with a holdout set)", train.len());
    data::shuffle(&mut train, args.seed);
    if args.max_train > 0 {
        train.truncate(args.max_train);
    }
    let positives = train.iter().filter(|p| p.same).count();
    println!("training on {} pairs ({positives} paraphrases)", train.len());

    // Model, tokenizer and optimizer.
    let device = Device::default().autodiff();
    device.seed(args.seed);
    println!("device: {device:?}");

    let base = download_hf_model(export::BASE_MODEL, None)?;
    let config = MiniLmConfig::load_from_hf(&base.config_path)?;
    let mut model = config.init(&device);
    load_pretrained(&mut model, &base.weights_path)?;
    let mut tokenizer = Tokenizer::from_file(&base.tokenizer_path).map_err(|e| e.to_string())?;
    configure_tokenizer(&mut tokenizer, args.max_seq_length)?;

    let clipping = (args.grad_clip > 0.0).then_some(GradientClippingConfig::Norm(args.grad_clip));
    let mut optim = AdamWConfig::new()
        .with_beta_1(0.9)
        .with_beta_2(0.999)
        .with_epsilon(1e-8)
        .with_weight_decay(0.0)
        .with_grad_clipping(clipping)
        .init();

    let steps_per_epoch = train.len().div_ceil(args.batch_size);
    let total = steps_per_epoch * args.epochs;
    let warmup = (total as f64 * args.warmup).ceil() as usize;
    println!("{total} steps, {warmup} warmup steps, batch size {}", args.batch_size);

    // Training loop.
    let mut step = 0;
    let mut progress = Progress::new(args.log_every, total);
    for epoch in 0..args.epochs {
        if epoch > 0 {
            data::shuffle(&mut train, args.seed + epoch as u64);
        }
        for batch in train.chunks(args.batch_size) {
            let n = batch.len();
            let texts: Vec<&str> = batch.iter().map(|p| p.a.as_str()).chain(batch.iter().map(|p| p.b.as_str())).collect();
            let same: Vec<bool> = batch.iter().map(|p| p.same).collect();

            // One forward pass over both sides of the pairs.
            let embeddings = sentence_embeddings(&model, tokenize(&tokenizer, &texts, &device)?);
            let a = embeddings.clone().slice([0..n, 0..DIM]);
            let b = embeddings.slice([n..2 * n, 0..DIM]);
            let loss = loss::online_contrastive(a, b, &same, args.margin);
            let loss_value = f64::from(loss.clone().into_scalar::<f32>());

            let grads = GradientsParams::from_grads(loss.backward(), &model);
            model = optim.step(learning_rate(step, total, warmup, args.lr), model, grads);
            step += 1;
            progress.step(step, loss_value, learning_rate(step.min(total - 1), total, warmup, args.lr));
        }
    }
    println!("trained in {:.0}s", started.elapsed().as_secs_f64());

    // The model without autodiff, saved in the Hugging Face layout.
    export::save(&model.valid(), &base, &args.out)?;
    println!("saved to {}", args.out.display());
    let mut data: Vec<&std::path::Path> = vec![args.train.as_path()];
    data.extend(args.holdout.iter().map(|p| p.as_path()));
    let metrics = serde_json::json!({"train_pairs": train.len(), "steps": total});
    semcache::version::write_run_record(&args.out, "finetune-cache", &format!("{args:?}"), &data, metrics, None, started)?;
    Ok(())
}
