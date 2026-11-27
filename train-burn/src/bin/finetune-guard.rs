//! Train the guardrail classifier with Burn.
//!
//! The classifier reads one prompt and gives the probability that it is a
//! prompt injection or a jailbreak. The data comes from train/export_guard.py
//! (xTRam1/safe-guard-prompt-injection, split into train and dev). Pass
//! --train twice to add data/guard-extra.jsonl, the train splits of deepset
//! and the jailbreak set.
//!
//! The training is train_burn::classifier on the labels 0 and 1, 3 epochs.
//! After each epoch, the dev threshold and the recall there. The threshold
//! is 0.5, the decision point of the loss, or higher when 0.5 gives more
//! false positives on dev than the target (1%). The lowest threshold that
//! meets the target would sit just above a benign score, and on an easy dev
//! set that is far below the injections. The epoch with the best recall,
//! then the best AUC, is saved with guard.json. On a tie the later epoch
//! wins, because the learning rate has decayed further.
//!
//! Run from the repository root:
//!   cargo run --release --manifest-path train-burn/Cargo.toml --bin finetune-guard
//! The default device is Metal. BURN_DEVICE=flex trains on the CPU (slow).

use clap::Parser;
use semcache::guard::{Confusion, GUARD_FILE, GuardFile, load_guard, roc_auc, threshold_for_fpr};
use semcache::version::{model_version, write_run_record};
use std::path::PathBuf;
use std::time::Instant;
use train_burn::classifier::{self, ClassifierArgs};
use train_burn::export;

#[derive(Parser, Debug)]
#[command(about = "Train the guardrail classifier with Burn")]
struct Args {
    /// Train files. Can repeat.
    #[arg(long, default_values = ["data/guard-train.jsonl"])]
    train: Vec<PathBuf>,

    #[arg(long, default_value = "data/guard-dev.jsonl")]
    dev: PathBuf,

    #[arg(long, default_value = "models/guard-classifier")]
    out: PathBuf,

    #[arg(long, default_value_t = 3)]
    epochs: usize,

    /// False positive rate on dev that the threshold must not exceed
    #[arg(long, default_value_t = 0.01)]
    target_fpr: f64,

    #[arg(long, default_value_t = 50)]
    log_every: usize,

    #[command(flatten)]
    train_args: ClassifierArgs,
}

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let args = Args::parse();
    let started = Instant::now();

    let mut train = Vec::new();
    for path in &args.train {
        train.extend(load_guard(path)?);
    }
    let dev = load_guard(&args.dev)?;
    let injections = train.iter().filter(|e| e.injection).count();
    println!("{} train prompts ({injections} injections), {} dev prompts", train.len(), dev.len());
    let examples = train.iter().map(|e| (e.text.clone(), if e.injection { 1.0 } else { 0.0 })).collect();
    let dev_texts: Vec<String> = dev.iter().map(|e| e.text.clone()).collect();

    let mut best: Option<(f64, f64, serde_json::Value)> = None;
    classifier::train(&args.train_args, args.epochs, args.log_every, examples, &dev_texts, |epoch| {
        let scores = &epoch.dev_scores;
        let auc = roc_auc(scores, &dev).unwrap_or(0.5);
        let threshold = threshold_for_fpr(scores, &dev, args.target_fpr).max(0.5);
        let flagged: Vec<bool> = scores.iter().map(|&s| s >= threshold).collect();
        let c = Confusion::new(&flagged, &dev);
        let (recall, fpr) = (c.recall().unwrap_or(0.0), c.fpr().unwrap_or(0.0));
        println!(
            "epoch {}: dev AUC {auc:.4}, threshold {threshold:.4}: recall {:.1}%, FPR {:.2}%",
            epoch.number,
            100.0 * recall,
            100.0 * fpr
        );
        if best.as_ref().is_some_and(|(r, a, _)| (recall, auc) < (*r, *a)) {
            return Ok(());
        }
        export::save_cross_encoder(epoch.model, &args.train_args.base, epoch.base, &args.out)?;
        GuardFile { threshold, max_tokens: args.train_args.max_seq_length, dev_fpr: fpr, dev_recall: recall }
            .save(&args.out)?;
        println!("saved epoch {} to {}", epoch.number, args.out.display());
        let metrics = serde_json::json!({
            "epoch": epoch.number, "dev_auc": auc, "threshold": threshold, "dev_recall": recall, "dev_fpr": fpr,
        });
        best = Some((recall, auc, metrics));
        Ok(())
    })?;

    let (recall, auc, metrics) = best.ok_or("no epoch")?;
    let version = model_version(&args.out, GUARD_FILE).ok();
    let mut data: Vec<&std::path::Path> = args.train.iter().map(|p| p.as_path()).collect();
    data.push(&args.dev);
    write_run_record(&args.out, "finetune-guard", &format!("{args:?}"), &data, metrics, version, started)?;
    println!(
        "trained in {:.0}s, dev recall {:.1}% at the target FPR, AUC {auc:.4}",
        started.elapsed().as_secs_f64(),
        100.0 * recall
    );
    Ok(())
}
