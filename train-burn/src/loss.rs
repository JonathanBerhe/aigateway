//! Training losses.
//!
//! `online_contrastive` is OnlineContrastiveLoss, as in sentence-transformers.
//!
//! For each pair, d is the cosine distance (1 - cosine similarity). The loss
//! keeps only the hard pairs of the batch:
//! - positives farther apart than the closest negative: loss d^2
//! - negatives closer than the farthest positive: loss max(0, margin - d)^2
//!
//! The selection of hard pairs uses values only, without gradients, like the
//! boolean indexing in the PyTorch code.

use burn::tensor::Tensor;

/// `a` and `b` are L2-normalized embeddings [n, dim]. `same[i]` is the label of pair i.
pub fn online_contrastive(a: Tensor<2>, b: Tensor<2>, same: &[bool], margin: f32) -> Tensor<1> {
    let [n, _] = a.dims();
    assert_eq!(n, same.len(), "one label per pair");
    let device = a.device();

    let distance: Tensor<1> = (a * b).sum_dim(1).reshape([n]).neg().add_scalar(1.0);
    let values: Vec<f32> = distance.clone().into_data().try_to_vec().expect("f32 distances");

    let (pos, neg): (Vec<f32>, Vec<f32>) = {
        let pos = values.iter().zip(same).filter(|(_, s)| **s).map(|(d, _)| *d).collect();
        let neg = values.iter().zip(same).filter(|(_, s)| !**s).map(|(d, _)| *d).collect();
        (pos, neg)
    };
    let mean = |v: &[f32]| if v.is_empty() { 0.0 } else { v.iter().sum::<f32>() / v.len() as f32 };
    // Same fallbacks as sentence-transformers when a side has fewer than 2 pairs.
    let neg_cut = if pos.len() > 1 { pos.iter().copied().fold(f32::MIN, f32::max) } else { mean(&neg) };
    let pos_cut = if neg.len() > 1 { neg.iter().copied().fold(f32::MAX, f32::min) } else { mean(&pos) };

    let pos_mask: Vec<f32> = values.iter().zip(same).map(|(d, s)| f32::from(u8::from(*s && *d > pos_cut))).collect();
    let neg_mask: Vec<f32> = values.iter().zip(same).map(|(d, s)| f32::from(u8::from(!*s && *d < neg_cut))).collect();
    let pos_mask = Tensor::<1>::from_data(pos_mask.as_slice(), &device);
    let neg_mask = Tensor::<1>::from_data(neg_mask.as_slice(), &device);

    let positive = (distance.clone().powf_scalar(2.0) * pos_mask).sum();
    let negative = (distance.neg().add_scalar(margin).clamp_min(0.0).powf_scalar(2.0) * neg_mask).sum();
    positive + negative
}

/// Binary cross-entropy on logits, the mean over the batch, as
/// BCEWithLogitsLoss in PyTorch. The form max(x, 0) - x*y + log(1 + e^-|x|)
/// does not overflow for large logits.
pub fn bce_with_logits(logits: Tensor<1>, same: &[bool]) -> Tensor<1> {
    let labels: Vec<f32> = same.iter().map(|s| f32::from(u8::from(*s))).collect();
    bce_with_logits_soft(logits, &labels)
}

/// The same loss with targets in [0, 1]. Its minimum is at
/// sigmoid(logit) = target, so the model learns the expected target.
pub fn bce_with_logits_soft(logits: Tensor<1>, targets: &[f32]) -> Tensor<1> {
    let [n] = logits.dims();
    assert_eq!(n, targets.len(), "one target per logit");
    let targets = Tensor::<1>::from_data(targets, &logits.device());
    let softplus = logits.clone().abs().neg().exp().log1p();
    (logits.clone().clamp_min(0.0) - logits * targets + softplus).mean()
}

#[cfg(test)]
mod tests {
    use super::*;
    use burn::tensor::Device;

    fn unit_rows(rows: &[[f32; 2]], device: &Device) -> Tensor<2> {
        let flat: Vec<f32> = rows
            .iter()
            .flat_map(|r| {
                let n = (r[0] * r[0] + r[1] * r[1]).sqrt();
                [r[0] / n, r[1] / n]
            })
            .collect();
        Tensor::<1>::from_data(flat.as_slice(), device).reshape([rows.len(), 2])
    }

    #[test]
    fn matches_a_hand_computed_value() {
        let device = Device::flex();
        // Pair distances: p1 0.0, p2 1 - cos(60°) = 0.5, n1 1 - cos(90°) = 1.0, n2 1 - cos(30°) ≈ 0.134.
        let a = unit_rows(&[[1.0, 0.0], [1.0, 0.0], [1.0, 0.0], [1.0, 0.0]], &device);
        let b = unit_rows(
            &[[1.0, 0.0], [0.5, 3f32.sqrt() / 2.0], [0.0, 1.0], [3f32.sqrt() / 2.0, 0.5]],
            &device,
        );
        let same = [true, true, false, false];
        // Hard positives: d > min(neg) = 0.134 -> only p2 (0.5): 0.25.
        // Hard negatives: d < max(pos) = 0.5 -> only n2 (0.134): (0.5 - 0.134)^2.
        let d_n2 = 1.0 - 3f32.sqrt() / 2.0;
        let expected = 0.25 + (0.5 - d_n2).powi(2);
        let got: f32 = online_contrastive(a, b, &same, 0.5).into_scalar();
        assert!((got - expected).abs() < 1e-5, "got {got}, expected {expected}");
    }

    #[test]
    fn easy_pairs_give_zero_loss() {
        let device = Device::flex();
        // Positives identical, negatives orthogonal: no hard pair at all.
        let a = unit_rows(&[[1.0, 0.0], [0.0, 1.0], [1.0, 0.0], [0.0, 1.0]], &device);
        let b = unit_rows(&[[1.0, 0.0], [0.0, 1.0], [0.0, 1.0], [1.0, 0.0]], &device);
        let got: f32 = online_contrastive(a, b, &[true, true, false, false], 0.5).into_scalar();
        assert!(got.abs() < 1e-6, "got {got}");
    }

    #[test]
    fn bce_matches_the_definition() {
        let device = Device::flex();
        let logits = [2.0f32, -1.0, 0.0, 8.0];
        let same = [true, false, true, false];
        let expected: f32 = logits
            .iter()
            .zip(same)
            .map(|(x, y)| {
                let p = 1.0 / (1.0 + (-x).exp());
                if y { -p.ln() } else { -(1.0 - p).ln() }
            })
            .sum::<f32>()
            / 4.0;
        let got: f32 = bce_with_logits(Tensor::<1>::from_data(logits.as_slice(), &device), &same).into_scalar();
        assert!((got - expected).abs() < 1e-4, "got {got}, expected {expected}");
    }
}
