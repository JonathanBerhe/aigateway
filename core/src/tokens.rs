//! Heuristic token counting.
//!
//! This is a placeholder for a real tokenizer (the `tokenizers` crate arrives
//! in phase 2). It gives a usable estimate when an upstream does not report
//! `usage`, and it costs a single pass over the text with no allocation.
//!
//! Rules, tuned on English BPE vocabularies such as `cl100k`:
//! - A run of letters or digits counts as one token per 4 characters, rounded up.
//! - A CJK character counts as one token.
//! - Any other non-space character (punctuation, symbols, emoji) counts as one token.
//! - Whitespace is free.

/// Estimate the number of tokens in `text`.
pub fn estimate(text: &str) -> u32 {
    let mut tokens: u32 = 0;
    let mut run: u32 = 0;

    for c in text.chars() {
        if c.is_alphanumeric() && !is_cjk(c) {
            run = run.saturating_add(1);
            continue;
        }
        tokens = tokens.saturating_add(run.div_ceil(4));
        run = 0;
        if !c.is_whitespace() {
            tokens = tokens.saturating_add(1);
        }
    }
    tokens.saturating_add(run.div_ceil(4))
}

/// True for Han, Hiragana, Katakana and Hangul characters. BPE tokenizers
/// spend about one token on each of these.
fn is_cjk(c: char) -> bool {
    matches!(c as u32,
        0x3040..=0x30FF     // Hiragana, Katakana
        | 0x3400..=0x4DBF   // CJK Extension A
        | 0x4E00..=0x9FFF   // CJK Unified Ideographs
        | 0xAC00..=0xD7AF   // Hangul syllables
        | 0xF900..=0xFAFF   // CJK Compatibility Ideographs
        | 0x20000..=0x2FA1F // CJK Extensions B to F, supplement
    )
}

#[cfg(test)]
mod tests {
    use super::estimate;

    #[test]
    fn empty_and_whitespace_cost_nothing() {
        assert_eq!(estimate(""), 0);
        assert_eq!(estimate("   \n\t "), 0);
    }

    #[test]
    fn words_cost_one_token_per_four_chars() {
        assert_eq!(estimate("hi"), 1);
        assert_eq!(estimate("gateway"), 2);
        // the(1) quick(2) brown(2) fox(1)
        assert_eq!(estimate("the quick brown fox"), 6);
    }

    #[test]
    fn punctuation_counts_separately() {
        assert_eq!(estimate("Hello, world!"), 2 + 1 + 2 + 1);
    }

    #[test]
    fn accented_letters_stay_in_the_word() {
        // "perché" is one run of 6 letters: 2 tokens.
        assert_eq!(estimate("perché"), 2);
    }

    #[test]
    fn cjk_characters_cost_one_token_each() {
        assert_eq!(estimate("你好世界"), 4);
    }

    #[test]
    fn estimate_is_in_a_sane_range_for_prose() {
        let text = "The gateway routes each prompt to a small or a large model.";
        let n = estimate(text);
        // One token per common English word and per punctuation mark gives 13.
        assert!((12..=24).contains(&n), "estimate {n} is out of range");
    }
}
