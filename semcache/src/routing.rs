//! Offline evaluation of routers on prompts with known model scores.
//!
//! Each example has the score of a cheap and a strong model's answer, from 0
//! to 1 (SPROUT, see train/export_sprout.py). A router gives each prompt a
//! score, and the prompts with the highest scores go to the strong model.
//! For each share of strong calls, the quality is the mean score of the
//! answers that the prompts get. This follows the RouteLLM paper:
//!
//! - PGR (performance gap recovered): (quality - cheap) / (strong - cheap).
//!   0 is the cheap model alone, 1 is the strong model alone.
//! - APGR: the mean PGR over all shares of strong calls from 0 to 1.
//!   A random router gets 0.5.
//! - CPT(x): the smallest share of strong calls that reaches a PGR of x.
//!
//! Prompts with the same router score are sent in random order. The curve
//! uses the expected quality of that order, so a router that gives every
//! prompt the same score is exactly the random router.
//!
//! Curve::weighted measures the cost in money instead: the x axis is the
//! share of the extra cost of the strong model, from the token counts of the
//! two answers and the prices per token (Prices).

use crate::embed::{DIM, Embedder, ModelSource};
use serde_json::Value;
use std::io::{BufRead, BufReader};
use std::path::Path;
use std::sync::atomic::{AtomicUsize, Ordering};

#[derive(Debug, Clone)]
pub struct RouteExample {
    pub prompt: String,
    pub dataset: String,
    pub cheap: f32,
    pub strong: f32,
    /// Input and output tokens of each model's answer (0 when unknown).
    pub cheap_tokens: (u32, u32),
    pub strong_tokens: (u32, u32),
}

/// Prices in dollars per million tokens.
#[derive(Debug, Clone, Copy)]
pub struct Prices {
    pub cheap_in: f64,
    pub cheap_out: f64,
    pub strong_in: f64,
    pub strong_out: f64,
}

impl Prices {
    /// Dollar cost of the cheap and of the strong answer to one prompt.
    pub fn cost(&self, e: &RouteExample) -> (f64, f64) {
        let dollars = |(i, o): (u32, u32), pi: f64, po: f64| (i as f64 * pi + o as f64 * po) / 1e6;
        (dollars(e.cheap_tokens, self.cheap_in, self.cheap_out), dollars(e.strong_tokens, self.strong_in, self.strong_out))
    }
}

impl RouteExample {
    /// What the strong model adds on this prompt. It is negative when the
    /// cheap model's answer is better.
    pub fn gain(&self) -> f32 {
        self.strong - self.cheap
    }
}

/// Read a JSONL file of train/export_sprout.py. Prompts without a score for
/// one of the two models are skipped.
pub fn load_sprout(path: &Path, cheap: &str, strong: &str) -> Result<Vec<RouteExample>, String> {
    let file = std::fs::File::open(path).map_err(|e| format!("{}: {e}", path.display()))?;
    let mut out = Vec::new();
    for (n, line) in BufReader::new(file).lines().enumerate() {
        let line = line.map_err(|e| format!("{}: {e}", path.display()))?;
        let row: Value = serde_json::from_str(&line).map_err(|e| format!("{}:{}: {e}", path.display(), n + 1))?;
        let score = |model: &str| row["score"][model].as_f64().map(|s| s as f32);
        if n == 0 {
            for model in [cheap, strong] {
                if row["score"].get(model).is_none() {
                    let known: Vec<&str> = row["score"].as_object().map(|o| o.keys().map(|k| k.as_str()).collect()).unwrap_or_default();
                    return Err(format!("{}: no model '{model}', the models are: {}", path.display(), known.join(", ")));
                }
            }
        }
        let (Some(c), Some(s)) = (score(cheap), score(strong)) else { continue };
        let tokens = |model: &str| {
            let n = |field: &str| row[field][model].as_u64().unwrap_or(0) as u32;
            (n("input_tokens"), n("output_tokens"))
        };
        out.push(RouteExample {
            prompt: row["prompt"].as_str().unwrap_or_default().to_owned(),
            dataset: row["dataset"].as_str().unwrap_or_default().to_owned(),
            cheap: c,
            strong: s,
            cheap_tokens: tokens(cheap),
            strong_tokens: tokens(strong),
        });
    }
    Ok(out)
}

/// Embed all texts with `threads` embedders in parallel, one text at a time.
/// With `cache`, the vectors are read from that file when it has the right
/// size, and written to it otherwise (raw little-endian f32, DIM per text).
pub fn embed_all(source: &ModelSource, texts: &[&str], threads: usize, cache: Option<&Path>) -> Result<Vec<Vec<f32>>, String> {
    if let Some(path) = cache {
        if let Ok(bytes) = std::fs::read(path) {
            if bytes.len() == texts.len() * DIM * 4 {
                let flat: Vec<f32> = bytes.chunks_exact(4).map(|b| f32::from_le_bytes([b[0], b[1], b[2], b[3]])).collect();
                return Ok(flat.chunks_exact(DIM).map(<[f32]>::to_vec).collect());
            }
        }
    }
    let vectors = map_parallel(texts, threads, || Embedder::load(source), |e, text| Ok(e.embed(&[text])?.remove(0).vector))?;
    if let Some(path) = cache {
        let bytes: Vec<u8> = vectors.iter().flatten().flat_map(|v| v.to_le_bytes()).collect();
        std::fs::write(path, bytes).map_err(|e| format!("{}: {e}", path.display()))?;
    }
    Ok(vectors)
}

/// Run `f` on each text with `threads` workers. Each worker loads its own
/// model with `load`, and the results keep the order of the texts.
pub fn map_parallel<M, T: Send>(
    texts: &[&str],
    threads: usize,
    load: impl Fn() -> Result<M, String> + Sync,
    f: impl Fn(&M, &str) -> Result<T, String> + Sync,
) -> Result<Vec<T>, String> {
    let threads = threads.clamp(1, texts.len().max(1));
    let chunk = texts.len().div_ceil(threads).max(1);
    let done = AtomicUsize::new(0);
    let parts: Vec<Result<Vec<T>, String>> = std::thread::scope(|scope| {
        let handles: Vec<_> = texts
            .chunks(chunk)
            .map(|part| {
                let (done, load, f) = (&done, &load, &f);
                scope.spawn(move || {
                    let model = load()?;
                    let mut out = Vec::with_capacity(part.len());
                    for text in part {
                        out.push(f(&model, text)?);
                        let n = done.fetch_add(1, Ordering::Relaxed) + 1;
                        if n % 2000 == 0 {
                            eprintln!("processed {n} / {}", texts.len());
                        }
                    }
                    Ok(out)
                })
            })
            .collect();
        handles.into_iter().map(|h| h.join().unwrap_or_else(|_| Err("worker thread panicked".into()))).collect()
    });
    let mut out = Vec::with_capacity(texts.len());
    for part in parts {
        out.extend(part?);
    }
    Ok(out)
}

/// Cache file for the embeddings of a data file under one model:
/// data/sprout-dev.jsonl and models/minilm-l6-qqp-burn give
/// data/sprout-dev.minilm-l6-qqp-burn.emb
pub fn embedding_cache(data: &Path, model: &str) -> std::path::PathBuf {
    let tag: String = model
        .trim_end_matches('/')
        .rsplit('/')
        .next()
        .unwrap_or(model)
        .chars()
        .map(|c| if c.is_ascii_alphanumeric() || c == '-' || c == '.' { c } else { '_' })
        .collect();
    data.with_extension(format!("{tag}.emb"))
}

/// The threshold that sends `share` of the prompts to the strong model:
/// the score of the last prompt that goes there.
pub fn threshold_for_share(scores: &[f32], share: f64) -> f32 {
    let mut sorted = scores.to_vec();
    sorted.sort_by(|a, b| b.total_cmp(a));
    let k = ((share * sorted.len() as f64).ceil() as usize).clamp(1, sorted.len().max(1));
    sorted.get(k - 1).copied().unwrap_or(0.5)
}

/// Quality as a function of the share of strong calls: a piecewise linear
/// curve from share 0 (all cheap) to share 1 (all strong).
#[derive(Debug, Clone)]
pub struct Curve {
    /// (share of strong calls, mean quality), at each group of equal scores.
    points: Vec<(f64, f64)>,
    pub cheap: f64,
    pub strong: f64,
}

impl Curve {
    /// The x axis is the share of prompts sent to the strong model.
    pub fn new(scores: &[f32], examples: &[RouteExample]) -> Curve {
        Curve::weighted(scores, examples, &vec![1.0; examples.len()])
    }

    /// The x axis is the share of `weights` spent, for example the extra
    /// dollar cost of the strong model on each prompt. Negative weights
    /// count as 0.
    pub fn weighted(scores: &[f32], examples: &[RouteExample], weights: &[f64]) -> Curve {
        assert_eq!(scores.len(), examples.len());
        assert_eq!(weights.len(), examples.len());
        let n = examples.len().max(1) as f64;
        let total: f64 = weights.iter().map(|w| w.max(0.0)).sum::<f64>();
        let total = if total > 0.0 { total } else { 1.0 };
        let mut order: Vec<usize> = (0..examples.len()).collect();
        order.sort_by(|&a, &b| scores[b].total_cmp(&scores[a]));

        let cheap: f64 = examples.iter().map(|e| e.cheap as f64).sum::<f64>() / n;
        let strong: f64 = examples.iter().map(|e| e.strong as f64).sum::<f64>() / n;
        let mut points = vec![(0.0, cheap)];
        let (mut spent, mut quality) = (0.0f64, cheap);
        let mut i = 0;
        while i < order.len() {
            let mut j = i;
            while j < order.len() && scores[order[j]] == scores[order[i]] {
                quality += examples[order[j]].gain() as f64 / n;
                spent += weights[order[j]].max(0.0);
                j += 1;
            }
            points.push(((spent / total).min(1.0), quality));
            i = j;
        }
        Curve { points, cheap, strong }
    }

    /// Mean quality when this share of the prompts goes to the strong model.
    pub fn quality_at(&self, share: f64) -> f64 {
        let share = share.clamp(0.0, 1.0);
        for w in self.points.windows(2) {
            let ((a0, q0), (a1, q1)) = (w[0], w[1]);
            if share <= a1 {
                return if a1 > a0 { q0 + (q1 - q0) * (share - a0) / (a1 - a0) } else { q1 };
            }
        }
        self.points.last().map_or(self.cheap, |p| p.1)
    }

    pub fn pgr(&self, quality: f64) -> f64 {
        (quality - self.cheap) / (self.strong - self.cheap)
    }

    /// Mean PGR over all shares of strong calls (exact for the piecewise curve).
    pub fn apgr(&self) -> f64 {
        let area: f64 = self.points.windows(2).map(|w| (w[1].0 - w[0].0) * (w[0].1 + w[1].1) / 2.0).sum();
        self.pgr(area)
    }

    /// The smallest share of strong calls that reaches this quality, or None.
    pub fn share_for_quality(&self, quality: f64) -> Option<f64> {
        if self.points.first()?.1 >= quality {
            return Some(0.0);
        }
        for w in self.points.windows(2) {
            let ((a0, q0), (a1, q1)) = (w[0], w[1]);
            if q1 >= quality {
                return Some(a0 + (a1 - a0) * (quality - q0) / (q1 - q0));
            }
        }
        None
    }

    /// CPT(x): the smallest share of strong calls that reaches a PGR of x.
    pub fn cpt(&self, pgr: f64) -> Option<f64> {
        self.share_for_quality(self.cheap + pgr * (self.strong - self.cheap))
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn ex(cheap: f32, strong: f32) -> RouteExample {
        RouteExample {
            prompt: String::new(),
            dataset: String::new(),
            cheap,
            strong,
            cheap_tokens: (0, 0),
            strong_tokens: (0, 0),
        }
    }

    fn examples() -> Vec<RouteExample> {
        vec![ex(0.0, 1.0), ex(1.0, 1.0), ex(0.0, 1.0), ex(1.0, 0.0)]
    }

    #[test]
    fn the_same_score_everywhere_is_the_random_router() {
        let c = Curve::new(&[0.3; 4], &examples());
        assert!((c.apgr() - 0.5).abs() < 1e-12, "apgr {}", c.apgr());
        assert!((c.quality_at(0.5) - 0.625).abs() < 1e-12);
    }

    #[test]
    fn the_curve_starts_cheap_and_ends_strong() {
        let c = Curve::new(&[0.9, 0.1, 0.5, 0.2], &examples());
        assert_eq!(c.quality_at(0.0), c.cheap);
        assert!((c.quality_at(1.0) - c.strong).abs() < 1e-12);
    }

    #[test]
    fn the_oracle_recovers_the_gap_with_the_fewest_strong_calls() {
        let e = examples();
        let oracle: Vec<f32> = e.iter().map(|e| e.gain()).collect();
        let c = Curve::new(&oracle, &e);
        // cheap 0.5, strong 0.75: two prompts gain 1, so 2 of 4 strong calls
        // reach quality 1.0, a PGR of 2.
        assert_eq!(c.cpt(1.0), Some(0.25));
        assert!((c.quality_at(0.5) - 1.0).abs() < 1e-12);
        assert!(c.apgr() > 0.5);
    }

    #[test]
    fn a_reversed_router_is_worse_than_random() {
        let e = examples();
        let reversed: Vec<f32> = e.iter().map(|e| -e.gain()).collect();
        assert!(Curve::new(&reversed, &e).apgr() < 0.5);
    }

    #[test]
    fn the_threshold_for_a_share_sends_that_share() {
        let scores = [0.9, 0.1, 0.5, 0.7, 0.3];
        assert_eq!(threshold_for_share(&scores, 0.4), 0.7);
        assert_eq!(threshold_for_share(&scores, 1.0), 0.1);
        assert_eq!(embedding_cache(Path::new("data/sprout-dev.jsonl"), "models/minilm-l6-qqp-burn/"),
            Path::new("data/sprout-dev.minilm-l6-qqp-burn.emb"));
    }

    #[test]
    fn a_cheap_prompt_moves_the_cost_curve_less() {
        // Both prompts gain 1. The first costs 1, the second 3.
        let e = vec![ex(0.0, 1.0), ex(0.0, 1.0)];
        let c = Curve::weighted(&[0.9, 0.1], &e, &[1.0, 3.0]);
        // A quarter of the cost buys half of the gap.
        assert!((c.quality_at(0.25) - 0.5).abs() < 1e-12);
        assert!((c.apgr() - (0.25 * 0.5 / 2.0 + 0.75 * 0.75) / 1.0).abs() < 1e-12, "apgr {}", c.apgr());
    }

    #[test]
    fn ties_are_averaged() {
        // Two tied prompts at the top: one gains 1, the other 0. Sending one
        // of the two is worth 0.5 of a gain on average.
        let e = vec![ex(0.0, 1.0), ex(1.0, 1.0)];
        let c = Curve::new(&[1.0, 1.0], &e);
        assert!((c.quality_at(0.5) - 0.75).abs() < 1e-12);
    }
}
