//! Heuristic model router.
//!
//! The router gives each prompt a difficulty score from 0.0 to 1.0 and sends
//! it to the strong tier when the score reaches a threshold. The score is a
//! weighted sum of four cheap signals: length, code, reasoning words and math.
//!
//! This heuristic is the baseline. In phase 3 a trained classifier replaces
//! it, and that classifier must beat this file on the evaluation set.

use crate::tokens;

pub const TIER_CHEAP: u32 = 0;
pub const TIER_STRONG: u32 = 1;

pub const DEFAULT_THRESHOLD: f32 = 0.5;

/// Only this many bytes of the prompt are scanned for keywords and code.
/// It keeps the cost of a 100k-token prompt bounded. The token estimate
/// still reads the full text.
const SCAN_LIMIT: usize = 16 * 1024;

/// Prompts of this many estimated tokens get the full length signal.
const LONG_PROMPT_TOKENS: f32 = 1500.0;

const W_LENGTH: f32 = 0.35;
const W_CODE: f32 = 0.25;
const W_REASONING: f32 = 0.30;
const W_MATH: f32 = 0.10;

const REASONING_WORDS: &[&str] = &[
    "analyze", "analyse", "architecture", "compare", "debug", "derive", "design",
    "explain why", "optimize", "optimise", "proof", "prove", "refactor",
    "step by step", "trade-off", "tradeoff",
];

const CODE_MARKERS: &[&str] = &[
    "```", "fn ", "def ", "class ", "import ", "#include", "function ", "=> ",
    "select ", "local ", "async ", "return ",
];

const MATH_MARKERS: &[&str] = &[
    "equation", "integral", "derivative", "matrix", "probability", "theorem", "\\frac", "∑", "∫",
];

#[derive(Debug, Clone, Copy, PartialEq)]
pub struct Decision {
    pub tier: u32,
    pub score: f32,
    pub prompt_tokens: u32,
}

/// Score `prompt` and pick a tier. A score at or above `threshold` picks
/// the strong tier. A threshold that is not finite falls back to 0.5.
pub fn route(prompt: &str, threshold: f32) -> Decision {
    let threshold = if threshold.is_finite() { threshold } else { DEFAULT_THRESHOLD };
    let prompt_tokens = tokens::estimate(prompt);

    let scan = prefix_on_char_boundary(prompt, SCAN_LIMIT).to_lowercase();

    let length = (prompt_tokens as f32 / LONG_PROMPT_TOKENS).min(1.0);
    let code = saturate(count_markers(&scan, CODE_MARKERS), 2);
    let reasoning = saturate(count_markers(&scan, REASONING_WORDS), 2);
    let math = saturate(count_markers(&scan, MATH_MARKERS), 1);

    let score = (W_LENGTH * length + W_CODE * code + W_REASONING * reasoning + W_MATH * math)
        .clamp(0.0, 1.0);
    let tier = if score >= threshold { TIER_STRONG } else { TIER_CHEAP };

    Decision { tier, score, prompt_tokens }
}

fn count_markers(haystack: &str, markers: &[&str]) -> u32 {
    markers.iter().filter(|m| haystack.contains(*m)).count() as u32
}

/// Map a hit count to 0.0..=1.0, reaching 1.0 at `full` hits.
fn saturate(hits: u32, full: u32) -> f32 {
    (hits as f32 / full as f32).min(1.0)
}

fn prefix_on_char_boundary(s: &str, max: usize) -> &str {
    if s.len() <= max {
        return s;
    }
    let mut end = max;
    while !s.is_char_boundary(end) {
        end -= 1;
    }
    &s[..end]
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn small_talk_goes_to_the_cheap_tier() {
        let d = route("hi, how are you?", DEFAULT_THRESHOLD);
        assert_eq!(d.tier, TIER_CHEAP);
        assert!(d.score < 0.1, "score {}", d.score);
    }

    #[test]
    fn code_plus_reasoning_goes_to_the_strong_tier() {
        let prompt = "Refactor this and explain why, step by step:\n```rust\nfn main() { return; }\n```";
        let d = route(prompt, DEFAULT_THRESHOLD);
        assert_eq!(d.tier, TIER_STRONG, "score {}", d.score);
    }

    #[test]
    fn a_very_long_prompt_raises_the_score() {
        let short = route("summarize this", DEFAULT_THRESHOLD);
        let long = route(&"lorem ipsum dolor sit amet ".repeat(800), DEFAULT_THRESHOLD);
        assert!(long.score > short.score);
        assert!((long.score - W_LENGTH).abs() < 1e-6, "score {}", long.score);
    }

    #[test]
    fn threshold_controls_the_tier() {
        let prompt = "Compare these two designs.";
        assert_eq!(route(prompt, 0.0).tier, TIER_STRONG);
        assert_eq!(route(prompt, 1.0).tier, TIER_CHEAP);
    }

    #[test]
    fn non_finite_threshold_uses_the_default() {
        let a = route("hello", f32::NAN);
        let b = route("hello", DEFAULT_THRESHOLD);
        assert_eq!(a, b);
    }

    #[test]
    fn score_stays_in_range_for_hostile_input() {
        let prompt = "prove derive ```fn def class``` ∑ ∫ equation ".repeat(5000);
        let d = route(&prompt, DEFAULT_THRESHOLD);
        assert!((0.0..=1.0).contains(&d.score));
    }

    #[test]
    fn scan_limit_never_splits_a_character() {
        let prompt = "é".repeat(SCAN_LIMIT); // 2 bytes per char
        let _ = route(&prompt, DEFAULT_THRESHOLD);
        assert_eq!(prefix_on_char_boundary("aé", 2), "a");
    }
}
