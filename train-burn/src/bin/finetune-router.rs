//! Train the classifier router with Burn.
//!
//! The router reads one prompt and predicts how much the strong model gains
//! over the cheap one on it. The data is SPROUT (train/export_sprout.py):
//! each prompt has the judge score of the cheap and of the strong model.
//!
//! The training is train_burn::classifier on the soft target (gain + 1) / 2,
//! so the score orders the prompts by the expected gain (see
//! semcache::router). 2 epochs. After each epoch, the APGR on dev. The best
//! epoch is saved, with the threshold that recovers the target share of the
//! quality gap on dev.
//!
//! Run from the repository root:
//!   cargo run --release --manifest-path train-burn/Cargo.toml --bin finetune-router
//! The default device is Metal. BURN_DEVICE=flex trains on the CPU (slow).

use clap::Parser;
use semcache::router::{ROUTER_FILE, RouterFile, RouterKind};
use semcache::routing::{Curve, load_sprout, threshold_for_share};
use semcache::version::{model_version, write_run_record};
use std::path::PathBuf;
use std::time::Instant;
use train_burn::classifier::{self, ClassifierArgs};
use train_burn::export;

#[derive(Parser, Debug)]
#[command(about = "Train the classifier router with Burn")]
struct Args {
    #[arg(long, default_value = "data/sprout-train.jsonl")]
    train: PathBuf,

    #[arg(long, default_value = "data/sprout-dev.jsonl")]
    dev: PathBuf,

    #[arg(long, default_value = "wxai-llama-3-1-8b-instruct")]
    cheap: String,

    #[arg(long, default_value = "openai-gpt-4o")]
    strong: String,

    #[arg(long, default_value = "models/router-classifier")]
    out: PathBuf,

    #[arg(long, default_value_t = 2)]
    epochs: usize,

    /// Choose the threshold that recovers this share of the quality gap on dev
    #[arg(long, default_value_t = 0.5)]
    target_pgr: f64,

    #[arg(long, default_value_t = 200)]
    log_every: usize,

    #[command(flatten)]
    train_args: ClassifierArgs,
}

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let args = Args::parse();
    let started = Instant::now();

    let train = load_sprout(&args.train, &args.cheap, &args.strong)?;
    let dev = load_sprout(&args.dev, &args.cheap, &args.strong)?;
    println!("{} train and {} dev prompts, cheap {}, strong {}", train.len(), dev.len(), args.cheap, args.strong);
    let examples = train.iter().map(|e| (e.prompt.clone(), (e.gain() + 1.0) / 2.0)).collect();
    let dev_texts: Vec<String> = dev.iter().map(|e| e.prompt.clone()).collect();

    let mut best: Option<(f64, serde_json::Value)> = None;
    classifier::train(&args.train_args, args.epochs, args.log_every, examples, &dev_texts, |epoch| {
        let curve = Curve::new(&epoch.dev_scores, &dev);
        let (apgr, cpt) = (curve.apgr(), curve.cpt(args.target_pgr));
        let cpt_text = cpt.map_or("never".into(), |s| format!("{:.1}%", 100.0 * s));
        println!("epoch {}: dev APGR {apgr:.3}, CPT({:.0}%) {cpt_text}", epoch.number, 100.0 * args.target_pgr);
        if best.as_ref().is_some_and(|(b, _)| apgr <= *b) {
            return Ok(());
        }
        let share = cpt.ok_or("the target PGR is never reached on dev")?;
        let threshold = threshold_for_share(&epoch.dev_scores, share);
        let n = epoch.dev_scores.len() as f64;
        let dev_strong_share = epoch.dev_scores.iter().filter(|&&s| s >= threshold).count() as f64 / n;
        export::save_cross_encoder(epoch.model, &args.train_args.base, epoch.base, &args.out)?;
        RouterFile {
            cheap: args.cheap.clone(),
            strong: args.strong.clone(),
            threshold,
            dev_strong_share,
            kind: RouterKind::Classifier { max_tokens: args.train_args.max_seq_length },
        }
        .save(&args.out)?;
        println!(
            "saved epoch {} to {}, threshold {threshold:.4} ({:.1}% strong calls on dev)",
            epoch.number,
            args.out.display(),
            100.0 * dev_strong_share
        );
        let metrics = serde_json::json!({
            "epoch": epoch.number, "dev_apgr": apgr, "threshold": threshold, "dev_strong_share": dev_strong_share,
        });
        best = Some((apgr, metrics));
        Ok(())
    })?;

    let (apgr, metrics) = best.ok_or("no epoch")?;
    let version = model_version(&args.out, ROUTER_FILE).ok();
    write_run_record(&args.out, "finetune-router", &format!("{args:?}"), &[&args.train, &args.dev], metrics, version, started)?;
    println!("trained in {:.0}s, best dev APGR {apgr:.3}", started.elapsed().as_secs_f64());
    Ok(())
}
