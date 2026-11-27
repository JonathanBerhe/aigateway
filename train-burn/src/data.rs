//! Training pairs: TSV reading, the leak filter and a seeded shuffle.

use std::collections::HashSet;
use std::path::Path;

pub struct Pair {
    pub a: String,
    pub b: String,
    pub same: bool,
}

/// Same normalization as train/finetune_cache.py: lower case, whitespace collapsed.
pub fn norm(text: &str) -> String {
    text.to_lowercase().split_whitespace().collect::<Vec<_>>().join(" ")
}

/// Read `question1 <TAB> question2 <TAB> label` lines. The header is skipped.
pub fn read_pairs(path: &Path) -> Result<Vec<Pair>, String> {
    let text = std::fs::read_to_string(path).map_err(|e| format!("{}: {e}", path.display()))?;
    Ok(text
        .lines()
        .filter_map(|line| {
            let mut cols = line.split('\t');
            let (a, b, label) = (cols.next()?, cols.next()?, cols.next()?);
            (a != "question1").then(|| Pair { a: a.to_owned(), b: b.to_owned(), same: label.trim() == "1" })
        })
        .collect())
}

/// Remove every training pair that shares a question with a holdout set.
/// QQP reuses questions, so without this the evaluation sees trained text.
/// Returns the kept pairs and the number removed.
pub fn leak_free(train: Vec<Pair>, holdout: &[Vec<Pair>]) -> (Vec<Pair>, usize) {
    let held: HashSet<String> = holdout.iter().flatten().flat_map(|p| [norm(&p.a), norm(&p.b)]).collect();
    let before = train.len();
    let kept: Vec<Pair> = train.into_iter().filter(|p| !held.contains(&norm(&p.a)) && !held.contains(&norm(&p.b))).collect();
    let removed = before - kept.len();
    (kept, removed)
}

/// xorshift64: a small seeded generator, for repeatable runs.
pub struct XorShift(u64);

impl XorShift {
    pub fn new(seed: u64) -> Self {
        Self(seed.max(1))
    }

    pub fn next_u64(&mut self) -> u64 {
        self.0 ^= self.0 << 13;
        self.0 ^= self.0 >> 7;
        self.0 ^= self.0 << 17;
        self.0
    }
}

/// Fisher-Yates shuffle with a fixed seed.
pub fn shuffle<T>(items: &mut [T], seed: u64) {
    let mut rng = XorShift::new(seed);
    for i in (1..items.len()).rev() {
        items.swap(i, (rng.next_u64() % (i as u64 + 1)) as usize);
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn pair(a: &str, b: &str) -> Pair {
        Pair { a: a.into(), b: b.into(), same: false }
    }

    #[test]
    fn norm_matches_the_python_rule() {
        assert_eq!(norm("  What IS\tthe   Capital?\n"), "what is the capital?");
    }

    #[test]
    fn leak_filter_removes_pairs_that_share_a_question() {
        let holdout = vec![vec![pair("Is it safe?", "Is this safe?")]];
        let train = vec![pair("is it   SAFE?", "other"), pair("x", "y"), pair("z", "Is this safe?")];
        let (kept, removed) = leak_free(train, &holdout);
        assert_eq!(removed, 2);
        assert_eq!(kept.len(), 1);
        assert_eq!(kept[0].a, "x");
    }

    #[test]
    fn shuffle_is_repeatable_and_a_permutation() {
        let mut a: Vec<u32> = (0..100).collect();
        let mut b = a.clone();
        shuffle(&mut a, 42);
        shuffle(&mut b, 42);
        assert_eq!(a, b);
        let mut sorted = a.clone();
        sorted.sort();
        assert_eq!(sorted, (0..100).collect::<Vec<_>>());
        assert_ne!(a, (0..100).collect::<Vec<_>>());
    }
}
