//! Choose the semantic cache threshold from labeled paraphrase pairs.
//!
//! Input: a TSV file with `question1 <TAB> question2 <TAB> label`, where
//! label 1 means "same meaning". scripts/fetch-qqp.sh writes one from the
//! Quora Question Pairs validation split.
//!
//! For each threshold t, a pair counts as a cache hit when the cosine
//! similarity of its two questions is at least t. Then:
//!   precision  = share of hits that are real paraphrases (1 - wrong answers)
//!   recall     = share of real paraphrases that hit
//!   false hits = share of different questions that hit anyway
//!
//! It also measures the embedding latency at batch size 1, which is the cost
//! that a lookup adds to a request when the gateway is idle.
//!
//! With --rerank, it also evaluates the two-stage check: a pair is a
//! candidate when its cosine similarity reaches the candidate floor, and a
//! hit when the cross-encoder probability of the candidate reaches P.

use clap::Parser;
use semcache::embed::{DIM, Embedder, ModelSource, THRESHOLD_FILE, parse_model};
use semcache::index::{CachedResponse, Index, dot};
use semcache::rerank::{RERANK_FILE, RerankSettings, Reranker};
use std::collections::HashMap;
use std::path::PathBuf;
use std::time::{Duration, Instant};

#[derive(Parser, Debug)]
#[command(about = "Evaluate semantic cache thresholds on paraphrase pairs")]
struct Args {
    /// TSV file: question1, question2, label (0 or 1)
    pairs: PathBuf,

    /// l6, l12, or a local model folder (see train/)
    #[arg(long, default_value = "l6")]
    model: String,

    /// Use at most this many pairs
    #[arg(long)]
    limit: Option<usize>,

    /// Batch size for the throughput run. On the Flex CPU backend, 1 is fastest.
    #[arg(long, default_value_t = 1)]
    batch: usize,

    /// Entries in the synthetic index for the search-time run (0 skips it)
    #[arg(long, default_value_t = 10_000)]
    index_size: usize,

    /// Number of single-text embeddings for the latency run
    #[arg(long, default_value_t = 200)]
    latency_samples: usize,

    /// Precision targets for the recommended thresholds
    #[arg(long, value_delimiter = ',', default_value = "0.90,0.95,0.99")]
    precision: Vec<f64>,

    /// Also print these thresholds, for example the one chosen on another set
    #[arg(long, value_delimiter = ',')]
    at: Vec<f32>,

    /// Write the lowest threshold that reaches this precision into the model
    /// folder (semcache.json). The sidecar then uses it by default. With
    /// --rerank, write the cascade thresholds into the cross-encoder folder
    /// (rerank.json) instead.
    #[arg(long)]
    save_threshold: Option<f64>,

    /// Cross-encoder folder: also evaluate the two-stage check
    #[arg(long)]
    rerank: Option<PathBuf>,

    /// Cosine similarity at which a pair becomes a candidate for the
    /// cross-encoder. Default: rerank.json of the folder, or 0.80.
    #[arg(long)]
    candidate: Option<f32>,

    /// Also print these cross-encoder thresholds, for example the one chosen on dev
    #[arg(long, value_delimiter = ',')]
    rerank_at: Vec<f32>,

    /// Print every pair with its scores, for small hand-made sets
    #[arg(long)]
    list: bool,

    /// Print the N most similar "different" pairs (the worst false hits),
    /// for error analysis: real errors or label noise?
    #[arg(long, default_value_t = 0)]
    show_errors: usize,
}

struct Pair {
    a: usize,
    b: usize,
    same: bool,
}

#[derive(Clone, Copy)]
struct Row {
    t: f32,
    precision: f64,
    recall: f64,
    false_hits: f64,
    f1: f64,
}

/// Precision, recall, false hits and F1 when a pair is a hit at score >= t.
fn sweep(scored: &[(f32, bool)], thresholds: impl Iterator<Item = f32>) -> Vec<Row> {
    thresholds
        .map(|t| {
            let (mut tp, mut fp, mut fneg, mut tn) = (0f64, 0f64, 0f64, 0f64);
            for &(s, same) in scored {
                match (s >= t, same) {
                    (true, true) => tp += 1.0,
                    (true, false) => fp += 1.0,
                    (false, true) => fneg += 1.0,
                    (false, false) => tn += 1.0,
                }
            }
            let precision = if tp + fp > 0.0 { tp / (tp + fp) } else { 1.0 };
            let recall = tp / (tp + fneg).max(1.0);
            let f1 = if precision + recall > 0.0 { 2.0 * precision * recall / (precision + recall) } else { 0.0 };
            Row { t, precision, recall, false_hits: fp / (fp + tn).max(1.0), f1 }
        })
        .collect()
}

/// Average precision: the mean of the precision at the rank of each
/// paraphrase, with pairs sorted by score. It summarizes the whole
/// precision-recall curve in one number, independent of a threshold.
fn average_precision(scored: &[(f32, bool)]) -> f64 {
    let mut ranked = scored.to_vec();
    ranked.sort_by(|a, b| b.0.total_cmp(&a.0));
    let (mut hits, mut sum) = (0f64, 0f64);
    for (k, (_, same)) in ranked.iter().enumerate() {
        if *same {
            hits += 1.0;
            sum += hits / (k + 1) as f64;
        }
    }
    sum / hits.max(1.0)
}

fn centi(t: f32) -> i32 {
    (t * 100.0).round() as i32
}

/// Print the rows at each 0.05, every row from 0.90, and the asked rows.
fn print_table(name: &str, rows: &[Row], at: &[f32]) {
    let asked: Vec<i32> = at.iter().map(|t| centi(*t)).collect();
    println!("| {name} | precision | recall | false hits | F1 |");
    println!("|---|---|---|---|---|");
    for r in rows.iter().filter(|r| centi(r.t) % 5 == 0 || r.t >= 0.9 || asked.contains(&centi(r.t))) {
        let mark = if asked.contains(&centi(r.t)) { " <- asked" } else { "" };
        println!("| {:.2} | {:.3} | {:.3} | {:.4} | {:.3} |{mark}", r.t, r.precision, r.recall, r.false_hits, r.f1);
    }
}

fn percentile(sorted: &[f64], p: f64) -> f64 {
    let i = ((sorted.len() as f64 - 1.0) * p).round() as usize;
    sorted[i.min(sorted.len() - 1)]
}

/// Time exact lookups in one namespace of `size` random unit vectors.
fn search_time(size: usize) {
    let mut seed = 0x2545_F491_4F6C_DD1Du64; // xorshift64, fixed seed
    let mut random_unit = || {
        let mut v: Vec<f32> = (0..DIM)
            .map(|_| {
                seed ^= seed << 13;
                seed ^= seed >> 7;
                seed ^= seed << 17;
                (seed >> 40) as f32 / (1u64 << 24) as f32 - 0.5
            })
            .collect();
        let n = v.iter().map(|x| x * x).sum::<f32>().sqrt();
        v.iter_mut().for_each(|x| *x /= n);
        v
    };
    let now = Instant::now();
    let ttl = Duration::from_secs(3600);
    let mut index = Index::new(size);
    let answer = CachedResponse { content: String::new(), model: String::new(), finish_reason: "stop".into(), usage: None };
    for i in 0..size {
        index.insert("ns", &i.to_string(), random_unit(), answer.clone(), ttl, now);
    }
    let mut ms: Vec<f64> = (0..200)
        .map(|_| {
            let q = random_unit();
            let t0 = Instant::now();
            std::hint::black_box(index.lookup("ns", &q, 0.97, now));
            t0.elapsed().as_secs_f64() * 1000.0
        })
        .collect();
    ms.sort_by(f64::total_cmp);
    println!("exact search, {} entries: p50 {:.2} ms, p99 {:.2} ms", index.len(), percentile(&ms, 0.5), percentile(&ms, 0.99));
}

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let args = Args::parse();
    let source = parse_model(&args.model)?;

    // Read the pairs and deduplicate the texts.
    let data = std::fs::read_to_string(&args.pairs)?;
    let mut texts: Vec<String> = Vec::new();
    let mut ids: HashMap<String, usize> = HashMap::new();
    let mut pairs = Vec::new();
    for line in data.lines() {
        let cols: Vec<&str> = line.split('\t').collect();
        if cols.len() < 3 || cols[0] == "question1" {
            continue;
        }
        let mut id = |s: &str| {
            *ids.entry(s.to_owned()).or_insert_with(|| {
                texts.push(s.to_owned());
                texts.len() - 1
            })
        };
        let (a, b) = (id(cols[0]), id(cols[1]));
        pairs.push(Pair { a, b, same: cols[2].trim() == "1" });
        if Some(pairs.len()) == args.limit {
            break;
        }
    }
    let positives = pairs.iter().filter(|p| p.same).count();
    println!("pairs: {} ({} paraphrases, {} different), unique texts: {}", pairs.len(), positives,
        pairs.len() - positives, texts.len());

    let started = Instant::now();
    let embedder = Embedder::load(&source)?;
    println!("model: {}, loaded in {:.1}s", source.label(), started.elapsed().as_secs_f64());

    // Latency at batch size 1.
    let n = args.latency_samples.min(texts.len());
    let mut ms = Vec::with_capacity(n);
    embedder.embed(&["warm up"])?;
    for text in texts.iter().take(n) {
        let t0 = Instant::now();
        embedder.embed(&[text.as_str()])?;
        ms.push(t0.elapsed().as_secs_f64() * 1000.0);
    }
    ms.sort_by(f64::total_cmp);
    println!("latency, batch 1, {n} texts: p50 {:.1} ms, p95 {:.1} ms, p99 {:.1} ms, max {:.1} ms",
        percentile(&ms, 0.50), percentile(&ms, 0.95), percentile(&ms, 0.99), ms[ms.len() - 1]);

    // Embed every text in batches and measure throughput.
    let t0 = Instant::now();
    let mut vectors = Vec::with_capacity(texts.len());
    let mut truncated = 0;
    for chunk in texts.chunks(args.batch.max(1)) {
        let refs: Vec<&str> = chunk.iter().map(String::as_str).collect();
        for e in embedder.embed(&refs)? {
            truncated += e.truncated as usize;
            vectors.push(e.vector);
        }
    }
    let secs = t0.elapsed().as_secs_f64();
    println!("throughput, batch {}: {:.0} texts/s ({} texts in {:.1}s, {} truncated)",
        args.batch, texts.len() as f64 / secs, texts.len(), secs, truncated);
    if args.index_size > 0 {
        search_time(args.index_size);
    }

    // Similarity of each pair.
    let scored: Vec<(f32, bool)> = pairs.iter().map(|p| (dot(&vectors[p.a], &vectors[p.b]), p.same)).collect();
    let mean = |same: bool| {
        let v: Vec<f32> = scored.iter().filter(|s| s.1 == same).map(|s| s.0).collect();
        v.iter().sum::<f32>() / v.len().max(1) as f32
    };
    println!("mean similarity: paraphrases {:.3}, different {:.3}\n", mean(true), mean(false));

    if args.show_errors > 0 {
        let mut worst: Vec<(f32, &Pair)> = pairs.iter().zip(&scored).filter(|(p, _)| !p.same).map(|(p, s)| (s.0, p)).collect();
        worst.sort_by(|a, b| b.0.total_cmp(&a.0));
        println!("most similar pairs labeled \"different\":");
        for (score, p) in worst.iter().take(args.show_errors) {
            println!("  {score:.3}  {}  |  {}", texts[p.a], texts[p.b]);
        }
        println!();
    }

    println!("average precision: {:.3}\n", average_precision(&scored));

    // Threshold sweep.
    let rows = sweep(&scored, (50..=99).map(|i| i as f32 / 100.0));
    print_table("threshold", &rows, &args.at);

    println!();
    for target in &args.precision {
        match rows.iter().find(|r| r.precision >= *target) {
            Some(r) => println!("lowest threshold with precision >= {target}: {:.2} (recall {:.3}, false hits {:.4})",
                r.t, r.recall, r.false_hits),
            None => println!("no threshold reaches precision {target}"),
        }
    }
    let best = rows.iter().max_by(|a, b| a.f1.total_cmp(&b.f1)).expect("rows");
    println!("best F1: threshold {:.2} (precision {:.3}, recall {:.3})", best.t, best.precision, best.recall);

    if let (Some(target), None) = (args.save_threshold, &args.rerank) {
        let ModelSource::Dir(dir) = &source else {
            return Err("--save-threshold needs a model folder, not a Hub model".into());
        };
        let Some(r) = rows.iter().find(|r| r.precision >= target) else {
            return Err(format!("no threshold reaches precision {target}; nothing written").into());
        };
        let record = serde_json::json!({
            "threshold": (f64::from(r.t) * 100.0).round() / 100.0,
            "target_precision": target,
            "precision": r.precision,
            "recall": r.recall,
            "false_hits": r.false_hits,
            "pairs": args.pairs.display().to_string(),
        });
        let path = dir.join(THRESHOLD_FILE);
        std::fs::write(&path, serde_json::to_string_pretty(&record)? + "\n")?;
        println!("wrote threshold {:.2} to {}", r.t, path.display());
    }
    if let Some(dir) = &args.rerank {
        rerank(&args, dir, &texts, &pairs, &scored)?;
    }
    Ok(())
}

/// Evaluate the two-stage check: cosine floor, then the cross-encoder.
fn rerank(
    args: &Args,
    dir: &PathBuf,
    texts: &[String],
    pairs: &[Pair],
    cosine: &[(f32, bool)],
) -> Result<(), Box<dyn std::error::Error>> {
    let started = Instant::now();
    let reranker = Reranker::load(dir)?;
    println!("\ncross-encoder: {}, loaded in {:.1}s", dir.display(), started.elapsed().as_secs_f64());
    let saved = RerankSettings::load(dir).ok();
    let candidate = args.candidate.or(saved.map(|s| s.candidate_threshold)).unwrap_or(0.80);

    // Score every pair at batch 1, the way the sidecar checks one candidate.
    reranker.score(&[("warm", "up")])?;
    let mut ms = Vec::with_capacity(pairs.len());
    let mut prob = Vec::with_capacity(pairs.len());
    for p in pairs {
        let t0 = Instant::now();
        prob.push(reranker.score(&[(texts[p.a].as_str(), texts[p.b].as_str())])?[0]);
        ms.push(t0.elapsed().as_secs_f64() * 1000.0);
    }
    ms.sort_by(f64::total_cmp);
    println!("cross-encoder latency, batch 1, {} pairs: p50 {:.1} ms, p95 {:.1} ms, p99 {:.1} ms",
        ms.len(), percentile(&ms, 0.50), percentile(&ms, 0.95), percentile(&ms, 0.99));
    if args.list {
        println!("\nlabel  cosine  probability  pair");
        for (p, (c, pr)) in pairs.iter().zip(cosine.iter().zip(&prob)) {
            let label = if p.same { "same" } else { "diff" };
            println!("{label}   {:.3}   {pr:.3}        {}  |  {}", c.0, texts[p.a], texts[p.b]);
        }
        println!();
    }
    let alone: Vec<(f32, bool)> = prob.iter().zip(cosine).map(|(p, c)| (*p, c.1)).collect();
    println!("cross-encoder alone: average precision {:.3}", average_precision(&alone));

    // A pair that is not a candidate never reaches the cross-encoder.
    let cascade = |floor: f32| -> Vec<(f32, bool)> {
        cosine.iter().zip(&prob).map(|(c, p)| (if c.0 >= floor { *p } else { -1.0 }, c.1)).collect()
    };
    let probabilities = || (1..=99).map(|i| i as f32 / 100.0);
    let target = args.save_threshold.unwrap_or(0.95);

    println!("\ncandidate floor vs. recall at precision >= {target}:");
    println!("| candidate floor | pairs checked | recall ceiling | P | precision | recall | false hits |");
    println!("|---|---|---|---|---|---|---|");
    let mut floors = vec![0.60, 0.70, 0.80, 0.85, 0.90, candidate];
    floors.sort_by(f32::total_cmp);
    floors.dedup_by(|a, b| centi(*a) == centi(*b));
    for floor in floors {
        let checked = cosine.iter().filter(|c| c.0 >= floor).count() as f64 / cosine.len().max(1) as f64;
        let ceiling = sweep(cosine, std::iter::once(floor))[0].recall;
        let rows = sweep(&cascade(floor), probabilities());
        let mark = if centi(floor) == centi(candidate) { " <- candidate" } else { "" };
        match rows.iter().find(|r| r.precision >= target) {
            Some(r) => println!("| {floor:.2} | {:.1}% | {ceiling:.3} | {:.2} | {:.3} | {:.3} | {:.4} |{mark}",
                checked * 100.0, r.t, r.precision, r.recall, r.false_hits),
            None => println!("| {floor:.2} | {:.1}% | {ceiling:.3} | none | | | |{mark}", checked * 100.0),
        }
    }

    let scored = cascade(candidate);
    println!("\ncascade at candidate floor {candidate:.2}, hit when the cross-encoder probability >= P:");
    let rows = sweep(&scored, probabilities());
    print_table("P", &rows, &args.rerank_at);
    println!();
    for t in &args.precision {
        match rows.iter().find(|r| r.precision >= *t) {
            Some(r) => println!("lowest P with precision >= {t}: {:.2} (recall {:.3}, false hits {:.4})", r.t, r.recall, r.false_hits),
            None => println!("no P reaches precision {t}"),
        }
    }

    if args.show_errors > 0 {
        let mut worst: Vec<(f32, f32, &Pair)> = pairs
            .iter()
            .zip(cosine.iter().zip(&prob))
            .filter(|(p, _)| !p.same)
            .map(|(p, (c, pr))| (c.0, *pr, p))
            .collect();
        worst.sort_by(|a, b| b.0.total_cmp(&a.0));
        println!("\nmost similar \"different\" pairs by cosine, with the cross-encoder probability:");
        for (c, pr, p) in worst.iter().take(args.show_errors) {
            println!("  cos {c:.3}  p {pr:.3}  {}  |  {}", texts[p.a], texts[p.b]);
        }
        worst.retain(|w| w.0 >= candidate);
        worst.sort_by(|a, b| b.1.total_cmp(&a.1));
        println!("\nworst false hits of the cascade (candidates with the highest probability):");
        for (c, pr, p) in worst.iter().take(args.show_errors) {
            println!("  p {pr:.3}  cos {c:.3}  {}  |  {}", texts[p.a], texts[p.b]);
        }
    }

    if let Some(target) = args.save_threshold {
        let Some(r) = rows.iter().find(|r| r.precision >= target) else {
            return Err(format!("no P reaches precision {target} at floor {candidate}; nothing written").into());
        };
        let record = serde_json::json!({
            "candidate_threshold": (f64::from(candidate) * 100.0).round() / 100.0,
            "threshold": (f64::from(r.t) * 100.0).round() / 100.0,
            "bi_encoder": args.model,
            "target_precision": target,
            "precision": r.precision,
            "recall": r.recall,
            "false_hits": r.false_hits,
            "pairs": args.pairs.display().to_string(),
        });
        let path = dir.join(RERANK_FILE);
        std::fs::write(&path, serde_json::to_string_pretty(&record)? + "\n")?;
        println!("wrote candidate floor {candidate:.2} and P {:.2} to {}", r.t, path.display());
    }
    Ok(())
}
