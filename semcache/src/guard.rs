//! Guardrail: a classifier that flags prompt injections and jailbreaks.
//!
//! The model is a fine-tuned MiniLM classifier (semcache::classifier). Its
//! folder holds guard.json with the threshold that train-burn chose on dev:
//! the lowest score that keeps the false positive rate at the target. For a
//! guardrail, a false positive blocks a normal user, so the threshold aims
//! at a low false positive rate first and a high recall second.
//!
//! The functions below also serve guard-eval, which compares the classifier
//! with the regex deny list of the Rust core (aigw_core::guard).

use serde::{Deserialize, Serialize};
use std::io::{BufRead, BufReader};
use std::path::Path;

pub const GUARD_FILE: &str = "guard.json";

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct GuardFile {
    /// A prompt with a score at or above this value is an injection.
    pub threshold: f32,
    /// Token limit of a prompt, at training and at serving time.
    pub max_tokens: usize,
    /// The false positive rate and the recall at the threshold, on dev.
    pub dev_fpr: f64,
    pub dev_recall: f64,
}

impl GuardFile {
    pub fn load(dir: &Path) -> Result<GuardFile, String> {
        let path = dir.join(GUARD_FILE);
        let text = std::fs::read_to_string(&path).map_err(|e| format!("{}: {e}", path.display()))?;
        serde_json::from_str(&text).map_err(|e| format!("{}: {e}", path.display()))
    }

    pub fn save(&self, dir: &Path) -> Result<(), String> {
        let path = dir.join(GUARD_FILE);
        let text = serde_json::to_string_pretty(self).map_err(|e| e.to_string())?;
        std::fs::write(&path, text + "\n").map_err(|e| format!("{}: {e}", path.display()))
    }
}

#[derive(Debug, Clone)]
pub struct GuardExample {
    pub text: String,
    /// True for an injection or a jailbreak.
    pub injection: bool,
    pub source: String,
}

/// Read a JSONL file of train/export_guard.py.
pub fn load_guard(path: &Path) -> Result<Vec<GuardExample>, String> {
    let file = std::fs::File::open(path).map_err(|e| format!("{}: {e}", path.display()))?;
    let mut out = Vec::new();
    for (n, line) in BufReader::new(file).lines().enumerate() {
        let line = line.map_err(|e| format!("{}: {e}", path.display()))?;
        let row: serde_json::Value =
            serde_json::from_str(&line).map_err(|e| format!("{}:{}: {e}", path.display(), n + 1))?;
        out.push(GuardExample {
            text: row["text"].as_str().unwrap_or_default().to_owned(),
            injection: row["label"].as_i64() == Some(1),
            source: row["source"].as_str().unwrap_or_default().to_owned(),
        });
    }
    Ok(out)
}

/// Counts of a binary decision against the labels.
#[derive(Debug, Clone, Copy, Default, PartialEq)]
pub struct Confusion {
    pub tp: usize,
    pub fp: usize,
    pub tn: usize,
    pub fn_: usize,
}

impl Confusion {
    pub fn new(flagged: &[bool], examples: &[GuardExample]) -> Confusion {
        let mut c = Confusion::default();
        for (&f, e) in flagged.iter().zip(examples) {
            match (f, e.injection) {
                (true, true) => c.tp += 1,
                (true, false) => c.fp += 1,
                (false, false) => c.tn += 1,
                (false, true) => c.fn_ += 1,
            }
        }
        c
    }

    /// Share of injections that are flagged. None without injections.
    pub fn recall(&self) -> Option<f64> {
        let p = self.tp + self.fn_;
        (p > 0).then(|| self.tp as f64 / p as f64)
    }

    /// Share of benign prompts that are flagged. None without benign prompts.
    pub fn fpr(&self) -> Option<f64> {
        let n = self.fp + self.tn;
        (n > 0).then(|| self.fp as f64 / n as f64)
    }

    /// Share of flagged prompts that are injections. None when nothing is flagged.
    pub fn precision(&self) -> Option<f64> {
        let f = self.tp + self.fp;
        (f > 0).then(|| self.tp as f64 / f as f64)
    }
}

/// Area under the ROC curve: the probability that a random injection gets a
/// higher score than a random benign prompt (ties count half). None when one
/// class is missing.
pub fn roc_auc(scores: &[f32], examples: &[GuardExample]) -> Option<f64> {
    let mut pairs: Vec<(f32, bool)> = scores.iter().copied().zip(examples.iter().map(|e| e.injection)).collect();
    pairs.sort_by(|a, b| a.0.total_cmp(&b.0));
    let pos = pairs.iter().filter(|p| p.1).count();
    let neg = pairs.len() - pos;
    if pos == 0 || neg == 0 {
        return None;
    }
    // Sum of the ranks of the injections, with tied scores at their mean rank.
    let mut rank_sum = 0.0;
    let mut i = 0;
    while i < pairs.len() {
        let mut j = i;
        while j < pairs.len() && pairs[j].0 == pairs[i].0 {
            j += 1;
        }
        let mean_rank = (i + 1 + j) as f64 / 2.0;
        rank_sum += mean_rank * pairs[i..j].iter().filter(|p| p.1).count() as f64;
        i = j;
    }
    Some((rank_sum - (pos * (pos + 1)) as f64 / 2.0) / (pos as f64 * neg as f64))
}

/// The lowest threshold whose false positive rate stays at or below
/// `target`: a score strictly above the k-th highest benign score.
pub fn threshold_for_fpr(scores: &[f32], examples: &[GuardExample], target: f64) -> f32 {
    let mut benign: Vec<f32> = scores.iter().zip(examples).filter(|(_, e)| !e.injection).map(|(s, _)| *s).collect();
    benign.sort_by(|a, b| b.total_cmp(a));
    let allowed = (target * benign.len() as f64).floor() as usize;
    match benign.get(allowed) {
        // Scores at or above the threshold are flagged: go just above this one.
        Some(&s) => f32::from_bits(s.to_bits() + 1).min(1.0),
        None => 0.0,
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn ex(injection: bool) -> GuardExample {
        GuardExample { text: String::new(), injection, source: String::new() }
    }

    #[test]
    fn confusion_counts_and_rates() {
        let e = [ex(true), ex(true), ex(false), ex(false)];
        let c = Confusion::new(&[true, false, true, false], &e);
        assert_eq!(c, Confusion { tp: 1, fp: 1, tn: 1, fn_: 1 });
        assert_eq!(c.recall(), Some(0.5));
        assert_eq!(c.fpr(), Some(0.5));
        assert_eq!(Confusion::new(&[true], &[ex(true)]).fpr(), None);
    }

    #[test]
    fn auc_of_a_perfect_a_random_and_a_tied_score() {
        let e = [ex(false), ex(true), ex(false), ex(true)];
        assert_eq!(roc_auc(&[0.1, 0.9, 0.2, 0.8], &e), Some(1.0));
        assert_eq!(roc_auc(&[0.5; 4], &e), Some(0.5));
        assert_eq!(roc_auc(&[0.9, 0.1, 0.8, 0.2], &e), Some(0.0));
        assert_eq!(roc_auc(&[0.5], &[ex(true)]), None);
    }

    #[test]
    fn the_fpr_threshold_flags_at_most_the_target_share() {
        let e: Vec<GuardExample> = (0..100).map(|i| ex(i >= 90)).collect();
        let scores: Vec<f32> = (0..100).map(|i| i as f32 / 100.0).collect();
        // 90 benign scores 0.00..0.89: 5% allows 4 false positives (0.86..0.89).
        let t = threshold_for_fpr(&scores, &e, 0.05);
        let flagged: Vec<bool> = scores.iter().map(|&s| s >= t).collect();
        let c = Confusion::new(&flagged, &e);
        assert_eq!(c.fp, 4);
        assert_eq!(c.tp, 10);
    }
}
