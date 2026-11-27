//! Train the router head: a logistic regression on the cache embedding.
//!
//! 1. Embed the train and dev prompts with the cache model (cached in data/).
//! 2. Fit one head for each L2 value on train, on the soft target
//!    (gain + 1) / 2 (see semcache::router).
//! 3. Keep the head with the best APGR on dev, and choose its threshold on
//!    dev: the smallest share of strong calls that reaches the target PGR.
//! 4. Save router.json in the output folder.
//!
//! The test set stays for router-eval.

use clap::Parser;
use semcache::embed::parse_model;
use semcache::router::{RouterFile, RouterKind, dot, fit_logistic, sigmoid};
use semcache::routing::{Curve, embed_all, embedding_cache, load_sprout, threshold_for_share};
use semcache::version::{fingerprint, write_run_record};
use std::path::PathBuf;
use std::time::Instant;

#[derive(Parser, Debug)]
#[command(about = "Train the router head on the cache embedding")]
struct Args {
    #[arg(long, default_value = "data/sprout-train.jsonl")]
    train: PathBuf,

    #[arg(long, default_value = "data/sprout-dev.jsonl")]
    dev: PathBuf,

    /// The cache embedding model (a --model value of the sidecar)
    #[arg(long, default_value = "models/minilm-l6-qqp-burn")]
    model: String,

    #[arg(long, default_value = "wxai-llama-3-1-8b-instruct")]
    cheap: String,

    #[arg(long, default_value = "openai-gpt-4o")]
    strong: String,

    #[arg(long, default_value = "models/router-head")]
    out: PathBuf,

    /// Parallel embedders for the prompts that are not in the cache yet
    #[arg(long, default_value_t = 4)]
    threads: usize,

    /// L2 values to try. The best one on dev wins.
    #[arg(long, value_delimiter = ',', default_value = "0,0.0001,0.001,0.01")]
    l2: Vec<f32>,

    /// Gradient steps (full batch, Adam). 600 steps reach the same dev
    /// APGR, 3000 give the head that the README evaluates.
    #[arg(long, default_value_t = 3000)]
    steps: usize,

    #[arg(long, default_value_t = 0.05)]
    lr: f32,

    /// Choose the threshold that recovers this share of the quality gap on dev
    #[arg(long, default_value_t = 0.5)]
    target_pgr: f64,

    /// Choose the threshold that sends this share of dev prompts to the
    /// strong model instead
    #[arg(long)]
    strong_share: Option<f64>,
}

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let args = Args::parse();
    let source = parse_model(&args.model)?;
    let train = load_sprout(&args.train, &args.cheap, &args.strong)?;
    let dev = load_sprout(&args.dev, &args.cheap, &args.strong)?;

    let run_started = Instant::now();
    let started = Instant::now();
    let embed = |data: &PathBuf, examples: &[semcache::routing::RouteExample]| {
        let texts: Vec<&str> = examples.iter().map(|e| e.prompt.as_str()).collect();
        embed_all(&source, &texts, args.threads, Some(&embedding_cache(data, &args.model)))
    };
    let x_train = embed(&args.train, &train)?;
    let x_dev = embed(&args.dev, &dev)?;
    eprintln!("router-head: {} train and {} dev embeddings in {:.0}s", x_train.len(), x_dev.len(), started.elapsed().as_secs_f64());

    let y: Vec<f32> = train.iter().map(|e| (e.gain() + 1.0) / 2.0).collect();
    println!("| L2 | Train APGR | Dev APGR | Dev CPT(50%) |");
    println!("|---|---|---|---|");
    let mut best: Option<(f64, Vec<f32>, f32, Vec<f32>, f32)> = None;
    for &l2 in &args.l2 {
        let started = Instant::now();
        let (w, b) = fit_logistic(&x_train, &y, l2, args.steps, args.lr);
        let head = |x: &[Vec<f32>]| -> Vec<f32> { x.iter().map(|v| sigmoid(dot(&w, v) + b)).collect() };
        let (s_train, s_dev) = (head(&x_train), head(&x_dev));
        let (c_train, c_dev) = (Curve::new(&s_train, &train), Curve::new(&s_dev, &dev));
        println!(
            "| {l2} | {:.3} | {:.3} | {} |",
            c_train.apgr(),
            c_dev.apgr(),
            c_dev.cpt(0.5).map_or("never".into(), |s| format!("{:.1}%", 100.0 * s))
        );
        eprintln!("router-head: L2 {l2} fitted in {:.1}s", started.elapsed().as_secs_f64());
        if best.as_ref().is_none_or(|b| c_dev.apgr() > b.0) {
            best = Some((c_dev.apgr(), w, b, s_dev, l2));
        }
    }
    let (apgr, weights, bias, s_dev, l2) = best.ok_or("no L2 values")?;
    let curve = Curve::new(&s_dev, &dev);
    let share = match args.strong_share {
        Some(s) => s,
        None => curve.cpt(args.target_pgr).ok_or("the target PGR is never reached on dev")?,
    };
    let threshold = threshold_for_share(&s_dev, share);
    let dev_strong_share = s_dev.iter().filter(|&&s| s >= threshold).count() as f64 / s_dev.len() as f64;

    let file = RouterFile {
        cheap: args.cheap.clone(),
        strong: args.strong.clone(),
        threshold,
        dev_strong_share,
        kind: RouterKind::Head { embed_model: args.model.clone(), weights, bias },
    };
    file.save(&args.out)?;
    let metrics = serde_json::json!({"dev_apgr": apgr, "l2": l2, "threshold": threshold, "dev_strong_share": dev_strong_share});
    let version = fingerprint(&args.out, &[semcache::router::ROUTER_FILE]).ok();
    write_run_record(&args.out, "router-head", &format!("{args:?}"), &[&args.train, &args.dev], metrics, version, run_started)?;
    println!(
        "\nrouter-head: L2 {l2}, dev APGR {apgr:.3}. Threshold {threshold:.4} sends {:.1}% of dev prompts to the strong model. Saved {}",
        100.0 * dev_strong_share,
        args.out.join(semcache::router::ROUTER_FILE).display()
    );
    Ok(())
}
