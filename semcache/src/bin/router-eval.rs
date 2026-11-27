//! Compare routers offline on prompts with known model scores.
//!
//! Input: a JSONL file of train/export_sprout.py. Each prompt has the score
//! of the cheap and the strong model's answer. The tool computes the quality
//! curve of each router (see semcache::routing) and compares the routers at
//! the operating point of the heuristic router of the gateway (core/src/route.rs).
//!
//! Routers:
//!   random     every prompt gets the same score
//!   heuristic  the score of core/src/route.rs, as the gateway computes it
//!   oracle     the real gain of the strong model: the upper bound
//!   source     with --source-prior: the mean gain of the prompt's source
//!              dataset on train. It knows where a prompt comes from, which a
//!              router does not, so it shows how much a router gains by
//!              recognizing the benchmark of a prompt.
//!   trained    with --router: a head or a classifier (see semcache::router)

use aigw_core::route::{DEFAULT_THRESHOLD, route};
use clap::Parser;
use semcache::embed::parse_model;
use semcache::classifier::TextClassifier;
use semcache::router::{RouterFile, RouterKind};
use semcache::routing::{Curve, Prices, RouteExample, embed_all, embedding_cache, load_sprout, map_parallel};
use std::collections::BTreeMap;
use std::path::PathBuf;

#[derive(Parser, Debug)]
#[command(about = "Compare routers on prompts with known cheap and strong model scores")]
struct Args {
    /// JSONL file of train/export_sprout.py, for example data/sprout-dev.jsonl
    data: PathBuf,

    /// Model of the cheap tier (a SPROUT model name)
    #[arg(long, default_value = "wxai-llama-3-1-8b-instruct")]
    cheap: String,

    /// Model of the strong tier (a SPROUT model name)
    #[arg(long, default_value = "openai-gpt-4o")]
    strong: String,

    /// Threshold of the heuristic router. The gateway default is 0.5.
    #[arg(long, default_value_t = DEFAULT_THRESHOLD)]
    heuristic_threshold: f32,

    /// Use at most this many prompts
    #[arg(long)]
    limit: Option<usize>,

    /// Also print, for each source dataset, the gain of the strong model and
    /// the share of strong calls of the heuristic
    #[arg(long)]
    by_dataset: bool,

    /// Trained router folder (router.json of router-head or finetune-router).
    /// Can repeat.
    #[arg(long)]
    router: Vec<PathBuf>,

    /// Train file for the source prior router (data/sprout-train.jsonl)
    #[arg(long)]
    source_prior: Option<PathBuf>,

    /// Parallel workers for the trained routers
    #[arg(long, default_value_t = 8)]
    threads: usize,

    /// Dollars per million tokens: cheap input, cheap output, strong input,
    /// strong output. The cost table uses them. The defaults are list prices
    /// of GPT-4o and a typical price of Llama 3.1 8B at an inference host.
    #[arg(long, value_delimiter = ',', default_value = "0.10,0.10,2.50,10.00")]
    prices: Vec<f64>,
}

/// The source of a prompt: the first two parts of its SPROUT dataset name.
fn source(e: &RouteExample) -> String {
    e.dataset.split('/').take(2).collect::<Vec<_>>().join("/")
}

/// Name of a router folder for the tables: its last path part.
fn folder_name(dir: &std::path::Path) -> String {
    dir.file_name().map_or_else(|| dir.display().to_string(), |n| n.to_string_lossy().into_owned())
}

fn pct(x: f64) -> String {
    format!("{:.1}%", 100.0 * x)
}

fn opt_pct(x: Option<f64>) -> String {
    x.map_or("never".into(), pct)
}

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let args = Args::parse();
    let mut examples = load_sprout(&args.data, &args.cheap, &args.strong)?;
    if let Some(n) = args.limit {
        examples.truncate(n);
    }
    let n = examples.len() as f64;
    if examples.is_empty() {
        return Err("no prompts".into());
    }

    let better = examples.iter().filter(|e| e.gain() > 0.0).count() as f64 / n;
    let worse = examples.iter().filter(|e| e.gain() < 0.0).count() as f64 / n;
    let heuristic: Vec<f32> = examples.iter().map(|e| route(&e.prompt, args.heuristic_threshold).score).collect();
    let mut routers: Vec<(String, Vec<f32>)> = vec![
        ("random".into(), vec![0.0; examples.len()]),
        ("heuristic (route.rs)".into(), heuristic.clone()),
        ("oracle".into(), examples.iter().map(RouteExample::gain).collect()),
    ];
    if let Some(path) = &args.source_prior {
        let train = load_sprout(path, &args.cheap, &args.strong)?;
        let mut sums: BTreeMap<String, (f64, usize)> = BTreeMap::new();
        for e in &train {
            let s = sums.entry(source(e)).or_default();
            s.0 += e.gain() as f64;
            s.1 += 1;
        }
        let prior: Vec<f32> =
            examples.iter().map(|e| sums.get(&source(e)).map_or(0.0, |(sum, n)| (sum / *n as f64) as f32)).collect();
        routers.push(("source prior (train)".into(), prior));
    }
    let mut trained = Vec::new();
    let texts: Vec<&str> = examples.iter().map(|e| e.prompt.as_str()).collect();
    for dir in &args.router {
        let file = RouterFile::load(dir)?;
        let scores: Vec<f32> = match &file.kind {
            RouterKind::Head { embed_model, .. } => {
                let source = parse_model(embed_model)?;
                let cache = args.limit.is_none().then(|| embedding_cache(&args.data, embed_model));
                let vectors = embed_all(&source, &texts, args.threads, cache.as_deref())?;
                vectors.iter().map(|v| file.head_score(v).unwrap_or(0.5)).collect()
            }
            RouterKind::Classifier { max_tokens } => {
                map_parallel(&texts, args.threads, || TextClassifier::load(dir, *max_tokens), |m, text| Ok(m.score(&[text])?[0]))?
            }
        };
        let kind = match file.kind {
            RouterKind::Head { .. } => "head",
            RouterKind::Classifier { .. } => "classifier",
        };
        routers.push((format!("{kind} ({})", folder_name(dir)), scores.clone()));
        trained.push((folder_name(dir), file, scores));
    }
    let curves: Vec<(String, Curve)> = routers.iter().map(|(name, s)| (name.clone(), Curve::new(s, &examples))).collect();
    let base = &curves[0].1;

    println!("router-eval: {}, {} prompts", args.data.display(), examples.len());
    println!("cheap   {:<45} quality {:.3}", args.cheap, base.cheap);
    println!("strong  {:<45} quality {:.3}", args.strong, base.strong);
    println!(
        "The strong model is better on {} of the prompts, equal on {}, worse on {}.",
        pct(better),
        pct(1.0 - better - worse),
        pct(worse)
    );

    // The operating point of the gateway: score >= threshold goes strong.
    let t = args.heuristic_threshold;
    let strong_share = heuristic.iter().filter(|&&s| s >= t).count() as f64 / n;
    let quality: f64 = examples
        .iter()
        .zip(&heuristic)
        .map(|(e, &s)| if s >= t { e.strong } else { e.cheap } as f64)
        .sum::<f64>()
        / n;
    println!(
        "\nHeuristic at threshold {t:.2}: {} strong calls, quality {quality:.3}, PGR {:.2}",
        pct(strong_share),
        base.pgr(quality)
    );

    println!("\n| Router | APGR | CPT(50%) | CPT(80%) |");
    println!("|---|---|---|---|");
    for (name, c) in &curves {
        println!("| {name} | {:.3} | {} | {} |", c.apgr(), opt_pct(c.cpt(0.5)), opt_pct(c.cpt(0.8)));
    }

    // The same routers, with the extra dollar cost of the strong model as
    // the x axis instead of the share of strong calls.
    let [cheap_in, cheap_out, strong_in, strong_out] = args.prices[..] else {
        return Err("--prices needs four numbers".into());
    };
    let prices = Prices { cheap_in, cheap_out, strong_in, strong_out };
    let costs: Vec<(f64, f64)> = examples.iter().map(|e| prices.cost(e)).collect();
    let extra: Vec<f64> = costs.iter().map(|(c, s)| s - c).collect();
    let (all_cheap, all_strong): (f64, f64) = costs.iter().fold((0.0, 0.0), |a, c| (a.0 + c.0, a.1 + c.1));
    println!(
        "\nCost per 1000 prompts: ${:.3} with the cheap model only, ${:.3} with the strong model only (prices {:?} $ per 1M tokens).",
        1000.0 * all_cheap / n,
        1000.0 * all_strong / n,
        args.prices
    );
    let per_dollar: Vec<f32> = examples.iter().zip(&extra).map(|(e, x)| (e.gain() as f64 / x.max(1e-12)) as f32).collect();
    println!("Share of the extra cost of the strong model, instead of the share of strong calls:");
    println!("| Router | APGR | Cost for 50% of the gap | Cost for 80% |");
    println!("|---|---|---|---|");
    let rows = routers.iter().map(|(name, sc)| (name.as_str(), sc)).chain([("oracle per dollar", &per_dollar)]);
    for (name, scores) in rows {
        let c = Curve::weighted(scores, &examples, &extra);
        println!("| {name} | {:.3} | {} | {} |", c.apgr(), opt_pct(c.cpt(0.5)), opt_pct(c.cpt(0.8)));
    }

    let heuristic_curve = &curves[1].1;
    for (name, head, scores) in &trained {
        let sent: Vec<bool> = scores.iter().map(|&s| s >= head.threshold).collect();
        let share = sent.iter().filter(|&&up| up).count() as f64 / n;
        let q: f64 = examples.iter().zip(&sent).map(|(e, &up)| if up { e.strong } else { e.cheap } as f64).sum::<f64>() / n;
        let hq = heuristic_curve.quality_at(share);
        println!(
            "\nRouter {name} at its threshold {:.4}: {} strong calls ({} on dev), quality {q:.3}, PGR {:.2}.",
            head.threshold,
            pct(share),
            pct(head.dev_strong_share),
            base.pgr(q)
        );
        println!(
            "The heuristic at the same share of strong calls: quality {hq:.3}, PGR {:.2}. It needs {} strong calls for quality {q:.3}.",
            base.pgr(hq),
            opt_pct(heuristic_curve.share_for_quality(q))
        );
    }

    println!("\nHeuristic thresholds:");
    println!("| Threshold | Strong calls | Quality | PGR |");
    println!("|---|---|---|---|");
    for t in [0.05f32, 0.1, 0.15, 0.2, 0.3, 0.4, 0.5, 0.6] {
        let sent: Vec<bool> = heuristic.iter().map(|&s| s >= t).collect();
        let q: f64 = examples.iter().zip(&sent).map(|(e, &up)| if up { e.strong } else { e.cheap } as f64).sum::<f64>() / n;
        let share = sent.iter().filter(|&&up| up).count() as f64 / n;
        println!("| {t:.2} | {} | {q:.3} | {:.2} |", pct(share), base.pgr(q));
    }

    if args.by_dataset {
        let mut groups: BTreeMap<String, Vec<usize>> = BTreeMap::new();
        for (i, e) in examples.iter().enumerate() {
            groups.entry(source(e)).or_default().push(i);
        }
        let names: String = trained.iter().map(|(name, _, _)| format!(" {name} |")).collect();
        let rule: String = trained.iter().map(|_| "---|").collect();
        println!("\nShare of strong calls by source, each router at its threshold:");
        println!("| Source | Prompts | Cheap | Strong | Strong better | heuristic |{names}");
        println!("|---|---|---|---|---|---|{rule}");
        for (source, idx) in groups {
            let m = idx.len() as f64;
            let mean = |f: &dyn Fn(&RouteExample) -> f32| idx.iter().map(|&i| f(&examples[i]) as f64).sum::<f64>() / m;
            let up = idx.iter().filter(|&&i| examples[i].gain() > 0.0).count() as f64 / m;
            let sent = idx.iter().filter(|&&i| heuristic[i] >= t).count() as f64 / m;
            let routed: String = trained
                .iter()
                .map(|(_, file, scores)| {
                    let share = idx.iter().filter(|&&i| scores[i] >= file.threshold).count() as f64 / m;
                    format!(" {} |", pct(share))
                })
                .collect();
            println!(
                "| {source} | {} | {:.3} | {:.3} | {} | {} |{routed}",
                idx.len(),
                mean(&|e| e.cheap),
                mean(&|e| e.strong),
                pct(up),
                pct(sent)
            );
        }
    }
    Ok(())
}
