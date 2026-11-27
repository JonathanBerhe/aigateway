//! Compare the guardrails on labeled prompts.
//!
//! Input: JSONL files of train/export_guard.py. For each file, the tool
//! reports the regex deny list of the Rust core (aigw_core::guard), the
//! classifier at the threshold of guard.json (--guard), and both together
//! (a prompt is flagged when either one flags it):
//!   recall     share of injections that are flagged
//!   FPR        share of benign prompts that are flagged (false positives)
//!   precision  share of flagged prompts that are injections
//!   AUC        area under the ROC curve of the classifier score

use aigw_core::guard;
use clap::Parser;
use semcache::classifier::TextClassifier;
use semcache::guard::{Confusion, GuardExample, GuardFile, load_guard, roc_auc};
use semcache::routing::map_parallel;
use std::path::PathBuf;

#[derive(Parser, Debug)]
#[command(about = "Compare the regex deny list and the guardrail classifier")]
struct Args {
    /// JSONL files of train/export_guard.py
    #[arg(required = true)]
    data: Vec<PathBuf>,

    /// Guardrail classifier folder (guard.json of finetune-guard)
    #[arg(long)]
    guard: Option<PathBuf>,

    /// Use this threshold instead of the one in guard.json
    #[arg(long)]
    threshold: Option<f32>,

    /// Parallel workers for the classifier
    #[arg(long, default_value_t = 8)]
    threads: usize,

    /// Print this many missed injections and false positives per file
    #[arg(long, default_value_t = 0)]
    show_errors: usize,
}

fn pct(x: Option<f64>) -> String {
    x.map_or("n/a".into(), |v| format!("{:.1}%", 100.0 * v))
}

fn row(name: &str, method: &str, c: Confusion, auc: Option<f64>) {
    let auc = auc.map_or("".into(), |a| format!("{a:.3}"));
    println!("| {name} | {method} | {} | {} | {} | {auc} |", pct(c.recall()), pct(c.fpr()), pct(c.precision()));
}

fn short(text: &str) -> String {
    let one_line: String = text.split_whitespace().collect::<Vec<_>>().join(" ");
    let cut: String = one_line.chars().take(110).collect();
    if cut.len() < one_line.len() { format!("{cut}...") } else { cut }
}

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let args = Args::parse();
    let model = match &args.guard {
        Some(dir) => {
            let file = GuardFile::load(dir)?;
            let threshold = args.threshold.unwrap_or(file.threshold);
            println!(
                "classifier {}: threshold {threshold:.4} (dev FPR {}, dev recall {})",
                dir.display(),
                pct(Some(file.dev_fpr)),
                pct(Some(file.dev_recall))
            );
            Some((dir.clone(), file, threshold))
        }
        None => None,
    };

    println!("\n| Data | Method | Recall | FPR | Precision | AUC |");
    println!("|---|---|---|---|---|---|");
    let mut errors = Vec::new();
    let (mut regex_time, mut regex_prompts) = (std::time::Duration::ZERO, 0usize);
    guard::scan(""); // compile the patterns before the timing
    for path in &args.data {
        let examples: Vec<GuardExample> = load_guard(path)?;
        let injections = examples.iter().filter(|e| e.injection).count();
        let name = format!(
            "{} ({}, {injections} inj.)",
            path.file_stem().map_or(String::new(), |s| s.to_string_lossy().into_owned()),
            examples.len()
        );
        let started = std::time::Instant::now();
        let regex: Vec<bool> = examples.iter().map(|e| guard::scan(&e.text).is_some()).collect();
        regex_time += started.elapsed();
        regex_prompts += examples.len();
        row(&name, "regex", Confusion::new(&regex, &examples), None);

        if let Some((dir, file, threshold)) = &model {
            let texts: Vec<&str> = examples.iter().map(|e| e.text.as_str()).collect();
            let scores = map_parallel(&texts, args.threads, || TextClassifier::load(dir, file.max_tokens), |m, t| {
                Ok(m.score(&[t])?[0])
            })?;
            let flagged: Vec<bool> = scores.iter().map(|&s| s >= *threshold).collect();
            row(&name, "classifier", Confusion::new(&flagged, &examples), roc_auc(&scores, &examples));
            let either: Vec<bool> = flagged.iter().zip(&regex).map(|(a, b)| *a || *b).collect();
            row(&name, "regex or classifier", Confusion::new(&either, &examples), None);

            if args.show_errors > 0 {
                let mut missed: Vec<(f32, &str)> =
                    examples.iter().zip(&scores).filter(|(e, s)| e.injection && **s < *threshold).map(|(e, s)| (*s, e.text.as_str())).collect();
                let mut false_pos: Vec<(f32, &str)> =
                    examples.iter().zip(&scores).filter(|(e, s)| !e.injection && **s >= *threshold).map(|(e, s)| (*s, e.text.as_str())).collect();
                missed.sort_by(|a, b| a.0.total_cmp(&b.0));
                false_pos.sort_by(|a, b| b.0.total_cmp(&a.0));
                errors.push((name.clone(), missed.iter().take(args.show_errors).map(|(s, t)| format!("{s:.3}  {}", short(t))).collect::<Vec<_>>(),
                    false_pos.iter().take(args.show_errors).map(|(s, t)| format!("{s:.3}  {}", short(t))).collect::<Vec<_>>()));
            }
        }
    }
    println!(
        "\nregex: {:.1} µs per prompt on average ({regex_prompts} prompts)",
        regex_time.as_secs_f64() * 1e6 / regex_prompts.max(1) as f64
    );
    for (name, missed, false_pos) in errors {
        println!("\n{name}: missed injections (lowest scores first)");
        missed.iter().for_each(|l| println!("  {l}"));
        println!("{name}: false positives (highest scores first)");
        false_pos.iter().for_each(|l| println!("  {l}"));
    }
    Ok(())
}
