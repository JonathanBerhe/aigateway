//! Train the cross-encoder of the cache check with Burn.
//!
//! The recipe follows the sentence-transformers CrossEncoder training on QQP:
//! - start from cross-encoder/ms-marco-MiniLM-L6-v2: a MiniLM L6 cross-encoder
//!   for search relevance, with the same vocabulary as the cache model. It
//!   already compares two texts, so it learns QQP faster than a model that
//!   embeds one text at a time (all-MiniLM-L6-v2 reached a dev AP of 0.867,
//!   this start 0.881). --base all-MiniLM-L6-v2 adds a new classifier head.
//! - QQP train pairs, minus each pair that shares a question with dev or test
//! - binary cross-entropy on the logit
//! - AdamW: lr 2e-5, weight decay 0.01, linear warmup over 10% of the steps,
//!   then linear decay to 0, gradient clipping at norm 1
//! - batch size 32, pairs cut at 128 tokens
//! - the two questions of a pair in random order, so the model does not
//!   learn a meaning for the position of a text
//! - hard negatives (--hard-negatives, train_burn::hard): small edits of
//!   training questions that change the meaning (swaps, antonyms, other
//!   names and numbers), with the label "different". They widen the margin
//!   on such edits. The threshold chosen on dev is then stricter, so the
//!   cache gets fewer hits (see the README). --hard-negatives 0 turns them off.
//!
//! Run from the repository root:
//!   cargo run --release --manifest-path train-burn/Cargo.toml --bin finetune-rerank -- --out models/minilm-l6-qqp-rerank
//! The default device is Metal. BURN_DEVICE=flex trains on the CPU (slow).

use burn::grad_clipping::GradientClippingConfig;
use burn::module::Module;
use burn::optim::{AdamWConfig, GradientsParams};
use burn::tensor::Device;
use clap::Parser;
use minilm_burn::{MiniLmConfig, download_hf_model};
use semcache::embed::{configure_tokenizer, tokenize_pairs};
use semcache::rerank::CrossEncoder;
use std::collections::HashSet;
use std::path::PathBuf;
use std::time::Instant;
use tokenizers::Tokenizer;
use train_burn::data::XorShift;
use train_burn::{Progress, data, export, hard, learning_rate, loss};

#[derive(Parser, Debug)]
#[command(about = "Train the cross-encoder of the semantic cache check with Burn")]
struct Args {
    /// Raw QQP train split (train/export_qqp.py)
    #[arg(long, default_value = "data/qqp-train-raw.tsv")]
    train: PathBuf,

    /// Pair files to keep out of training (dev and test)
    #[arg(long, num_args = 1.., default_values = ["data/qqp-dev.tsv", "data/qqp-test.tsv"])]
    holdout: Vec<PathBuf>,

    /// Hugging Face model to start from (BERT layout with model.safetensors)
    #[arg(long, default_value = "cross-encoder/ms-marco-MiniLM-L6-v2")]
    base: String,

    /// Output model folder
    #[arg(long, default_value = "models/minilm-l6-qqp-rerank")]
    out: PathBuf,

    /// Cap on training pairs, 0 = all
    #[arg(long, default_value_t = 0)]
    max_train: usize,

    /// Hard negatives to add, made from the training questions (0 = none).
    /// 30000 gives about 24000, because swaps are rare.
    #[arg(long, default_value_t = 30_000)]
    hard_negatives: usize,

    /// Print this many random hard negatives and stop, to check the rules
    #[arg(long, default_value_t = 0)]
    show_hard: usize,

    #[arg(long, default_value_t = 1)]
    epochs: usize,

    #[arg(long, default_value_t = 32)]
    batch_size: usize,

    #[arg(long, default_value_t = 2e-5)]
    lr: f64,

    #[arg(long, default_value_t = 0.01)]
    weight_decay: f32,

    /// Token limit for a pair, both questions together
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

    #[arg(long, default_value_t = 500)]
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
    if args.hard_negatives > 0 {
        let mut seen = HashSet::new();
        let questions: Vec<&str> =
            train.iter().flat_map(|p| [p.a.as_str(), p.b.as_str()]).filter(|q| seen.insert(*q)).collect();
        let (hard, counts) = hard::make(&questions, args.hard_negatives, args.seed);
        let mut kinds: Vec<_> = counts.by_kind.iter().collect();
        kinds.sort_by_key(|(k, _)| format!("{k:?}"));
        println!("hard negatives: {} from {} questions {kinds:?}", hard.len(), questions.len());
        for (kind, a, b) in &counts.examples {
            println!("  {kind:?}: {a}  ->  {b}");
        }
        if args.show_hard > 0 {
            let mut sample: Vec<&data::Pair> = hard.iter().collect();
            data::shuffle(&mut sample, args.seed + 2);
            for p in sample.iter().take(args.show_hard) {
                println!("  {}  ->  {}", p.a, p.b);
            }
            return Ok(());
        }
        train.extend(hard);
        data::shuffle(&mut train, args.seed + 1);
    }
    let positives = train.iter().filter(|p| p.same).count();
    println!("training on {} pairs ({positives} paraphrases)", train.len());

    // Model, tokenizer and optimizer.
    let device = Device::default().autodiff();
    device.seed(args.seed);
    println!("device: {device:?}");

    let base = download_hf_model(&args.base, None)?;
    let config = MiniLmConfig::load_from_hf(&base.config_path)?;
    let mut model = CrossEncoder::init(&config, &device);
    // A cross-encoder base has every weight. A sentence-transformers base
    // has the encoder and the pooler, and the classifier starts random.
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

    let steps_per_epoch = train.len().div_ceil(args.batch_size);
    let total = steps_per_epoch * args.epochs;
    let warmup = (total as f64 * args.warmup).ceil() as usize;
    println!("{total} steps, {warmup} warmup steps, batch size {}", args.batch_size);

    // Training loop.
    let mut step = 0;
    let mut progress = Progress::new(args.log_every, total);
    let mut coin = XorShift::new(args.seed ^ 0x9E37_79B9_7F4A_7C15);
    for epoch in 0..args.epochs {
        if epoch > 0 {
            data::shuffle(&mut train, args.seed + epoch as u64);
        }
        for batch in train.chunks(args.batch_size) {
            let pairs: Vec<(&str, &str)> = batch
                .iter()
                .map(|p| if coin.next_u64() & 1 == 0 { (p.a.as_str(), p.b.as_str()) } else { (p.b.as_str(), p.a.as_str()) })
                .collect();
            let same: Vec<bool> = batch.iter().map(|p| p.same).collect();

            let logits = model.forward(tokenize_pairs(&tokenizer, &pairs, &device)?);
            let loss = loss::bce_with_logits(logits, &same);
            let loss_value = f64::from(loss.clone().into_scalar::<f32>());

            let grads = GradientsParams::from_grads(loss.backward(), &model);
            model = optim.step(learning_rate(step, total, warmup, args.lr), model, grads);
            step += 1;
            progress.step(step, loss_value, learning_rate(step.min(total - 1), total, warmup, args.lr));
        }
    }
    println!("trained in {:.0}s", started.elapsed().as_secs_f64());

    // The model without autodiff, saved in the Hugging Face layout.
    export::save_cross_encoder(&model.valid(), &args.base, &base, &args.out)?;
    println!("saved to {}", args.out.display());
    let mut data: Vec<&std::path::Path> = vec![args.train.as_path()];
    data.extend(args.holdout.iter().map(|p| p.as_path()));
    let metrics = serde_json::json!({"train_pairs": train.len(), "steps": total});
    semcache::version::write_run_record(&args.out, "finetune-rerank", &format!("{args:?}"), &data, metrics, None, started)?;
    Ok(())
}
