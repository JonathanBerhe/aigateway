//! Shared code of the Burn trainers: finetune-cache (the bi-encoder of the
//! cache), finetune-rerank (the cross-encoder of the cache check), and
//! finetune-router and finetune-guard (single-text classifiers).

pub mod classifier;
pub mod data;
pub mod export;
pub mod hard;
pub mod loss;

/// Learning rate at optimizer step `step` (0-based), like the Hugging Face
/// "linear" schedule: warmup from 0, then decay to 0 at the last step.
pub fn learning_rate(step: usize, total: usize, warmup: usize, base: f64) -> f64 {
    if step < warmup {
        base * step as f64 / warmup.max(1) as f64
    } else {
        base * (total.saturating_sub(step)) as f64 / (total - warmup).max(1) as f64
    }
}

/// Progress line every `every` steps: mean loss of the window, rate and ETA.
pub struct Progress {
    every: usize,
    total: usize,
    loss_sum: f64,
    window_start: std::time::Instant,
}

impl Progress {
    pub fn new(every: usize, total: usize) -> Self {
        Self { every: every.max(1), total, loss_sum: 0.0, window_start: std::time::Instant::now() }
    }

    /// Record the loss of step `step` (1-based, after the optimizer step).
    pub fn step(&mut self, step: usize, loss: f64, lr: f64) {
        self.loss_sum += loss;
        if step % self.every != 0 && step != self.total {
            return;
        }
        let window = match step % self.every {
            0 => self.every,
            w => w,
        };
        let rate = window as f64 / self.window_start.elapsed().as_secs_f64();
        let eta = (self.total - step) as f64 / rate;
        println!(
            "step {step}/{}  loss {:.4}  lr {lr:.2e}  {rate:.2} steps/s  eta {eta:.0}s",
            self.total,
            self.loss_sum / window as f64
        );
        self.loss_sum = 0.0;
        self.window_start = std::time::Instant::now();
    }
}
