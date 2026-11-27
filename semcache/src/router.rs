//! Trained routers. Each one scores a prompt from 0 to 1, and a prompt goes
//! to the strong tier when its score reaches the threshold.
//!
//! Both kinds were trained on the soft target (gain + 1) / 2, where gain is
//! the strong model's score minus the cheap model's score on that prompt
//! (see semcache::routing). So a score orders the prompts by the expected
//! gain of the strong model, and 0.5 means "no expected gain".
//!
//! - head: a logistic regression on the L2-normalized cache embedding. With
//!   the cache on, the sidecar already has that embedding, so routing costs
//!   one dot product.
//! - classifier: a fine-tuned MiniLM classifier (BERT, pooler, one logit),
//!   the same network as the cross-encoder of the cache check, on one text.
//!   It costs one more forward pass.
//!
//! A router folder holds router.json, and for a classifier also the model
//! files in the Hugging Face layout.

use crate::embed::DIM;
use serde::{Deserialize, Serialize};
use std::path::Path;

pub const ROUTER_FILE: &str = "router.json";

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct RouterFile {
    /// The cheap and strong models of the training data.
    pub cheap: String,
    pub strong: String,
    /// A prompt goes to the strong tier when its score reaches this value.
    pub threshold: f32,
    /// The share of strong calls that the threshold gave on dev.
    pub dev_strong_share: f64,
    #[serde(flatten)]
    pub kind: RouterKind,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(tag = "kind", rename_all = "snake_case")]
pub enum RouterKind {
    /// A linear head on the embedding of `embed_model` (a --model value).
    Head { embed_model: String, weights: Vec<f32>, bias: f32 },
    /// A fine-tuned classifier. The model files are in the same folder.
    Classifier { max_tokens: usize },
}

impl RouterFile {
    pub fn load(dir: &Path) -> Result<RouterFile, String> {
        let path = dir.join(ROUTER_FILE);
        let text = std::fs::read_to_string(&path).map_err(|e| format!("{}: {e}", path.display()))?;
        let file: RouterFile = serde_json::from_str(&text).map_err(|e| format!("{}: {e}", path.display()))?;
        if let RouterKind::Head { weights, .. } = &file.kind {
            if weights.len() != DIM {
                return Err(format!("{}: {} weights, expected {DIM}", path.display(), weights.len()));
            }
        }
        Ok(file)
    }

    pub fn save(&self, dir: &Path) -> Result<(), String> {
        std::fs::create_dir_all(dir).map_err(|e| format!("{}: {e}", dir.display()))?;
        let path = dir.join(ROUTER_FILE);
        let text = serde_json::to_string_pretty(self).map_err(|e| e.to_string())?;
        std::fs::write(&path, text + "\n").map_err(|e| format!("{}: {e}", path.display()))
    }

    /// Score of a cache embedding with a head router. None for a classifier.
    pub fn head_score(&self, embedding: &[f32]) -> Option<f32> {
        match &self.kind {
            RouterKind::Head { weights, bias, .. } => Some(sigmoid(dot(weights, embedding) + bias)),
            RouterKind::Classifier { .. } => None,
        }
    }
}

pub fn sigmoid(x: f32) -> f32 {
    1.0 / (1.0 + (-x).exp())
}

pub fn dot(a: &[f32], b: &[f32]) -> f32 {
    a.iter().zip(b).map(|(x, y)| x * y).sum()
}

/// Fit a logistic regression with soft targets in [0, 1] by full-batch
/// gradient descent with Adam. `l2` is the weight decay on the weights.
/// Returns the weights and the bias.
pub fn fit_logistic(x: &[Vec<f32>], y: &[f32], l2: f32, steps: usize, lr: f32) -> (Vec<f32>, f32) {
    let dim = x.first().map_or(0, Vec::len);
    let n = x.len().max(1) as f32;
    // Parameters: dim weights, then the bias.
    let mut p = vec![0f32; dim + 1];
    let (mut m, mut v) = (vec![0f32; dim + 1], vec![0f32; dim + 1]);
    let (b1, b2, eps) = (0.9f32, 0.999f32, 1e-8f32);
    let mut grad = vec![0f32; dim + 1];
    for step in 1..=steps {
        grad.iter_mut().for_each(|g| *g = 0.0);
        for (xi, &yi) in x.iter().zip(y) {
            let err = sigmoid(dot(&p[..dim], xi) + p[dim]) - yi;
            for (g, &xv) in grad[..dim].iter_mut().zip(xi) {
                *g += err * xv;
            }
            grad[dim] += err;
        }
        for (j, g) in grad.iter_mut().enumerate() {
            *g /= n;
            if j < dim {
                *g += l2 * p[j];
            }
        }
        let (c1, c2) = (1.0 - b1.powi(step as i32), 1.0 - b2.powi(step as i32));
        for j in 0..=dim {
            m[j] = b1 * m[j] + (1.0 - b1) * grad[j];
            v[j] = b2 * v[j] + (1.0 - b2) * grad[j] * grad[j];
            p[j] -= lr * (m[j] / c1) / ((v[j] / c2).sqrt() + eps);
        }
    }
    let bias = p[dim];
    p.truncate(dim);
    (p, bias)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn logistic_regression_learns_a_soft_target() {
        // Target 0.9 when x0 > 0, 0.1 otherwise.
        let x: Vec<Vec<f32>> = (0..200).map(|i| vec![if i % 2 == 0 { 1.0 } else { -1.0 }, (i % 7) as f32 / 7.0]).collect();
        let y: Vec<f32> = x.iter().map(|v| if v[0] > 0.0 { 0.9 } else { 0.1 }).collect();
        let (w, b) = fit_logistic(&x, &y, 0.0, 2000, 0.05);
        let p = |v: &[f32]| sigmoid(dot(&w, v) + b);
        assert!((p(&[1.0, 0.3]) - 0.9).abs() < 0.02, "{}", p(&[1.0, 0.3]));
        assert!((p(&[-1.0, 0.3]) - 0.1).abs() < 0.02, "{}", p(&[-1.0, 0.3]));
    }

    #[test]
    fn a_saved_router_loads_again() {
        let dir = std::env::temp_dir().join(format!("aigw-router-test-{}", std::process::id()));
        let file = RouterFile {
            cheap: "a".into(),
            strong: "b".into(),
            threshold: 0.55,
            dev_strong_share: 0.3,
            kind: RouterKind::Head { embed_model: "l6".into(), weights: vec![0.01; DIM], bias: -0.2 },
        };
        file.save(&dir).unwrap();
        let text = std::fs::read_to_string(dir.join(ROUTER_FILE)).unwrap();
        assert!(text.contains("\"kind\": \"head\""), "{text}");
        let back = RouterFile::load(&dir).unwrap();
        assert_eq!(back.threshold, 0.55);
        let score = back.head_score(&vec![0.0; DIM]).unwrap();
        assert!((score - sigmoid(-0.2)).abs() < 1e-6);
        std::fs::remove_dir_all(&dir).unwrap();
    }
}
