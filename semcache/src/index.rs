//! Exact (brute-force) vector index with namespaces, TTL and LRU eviction.
//!
//! Every lookup compares the query with each live entry of its namespace.
//! That is the baseline: it is simple, exact, and fast enough for tens of
//! thousands of entries (384 dims). An approximate index such as HNSW is
//! worth its complexity only when measurements show that search time matters.
//!
//! A namespace is an exact-match partition of the cache. The gateway builds it
//! from the tier, the model, the system prompt and the sampling parameters, so
//! an answer is only reused under the same conditions.

use serde::{Deserialize, Serialize};
use std::collections::HashMap;
use std::time::{Duration, Instant};

/// The answer that the cache stores and returns.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct CachedResponse {
    pub content: String,
    pub model: String,
    pub finish_reason: String,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub usage: Option<Usage>,
}

#[derive(Debug, Clone, Copy, PartialEq, Serialize, Deserialize)]
pub struct Usage {
    pub prompt_tokens: u64,
    pub completion_tokens: u64,
}

#[derive(Debug)]
struct Entry {
    /// The prompt text, for the cross-encoder check and for deduplication.
    text: String,
    vector: Vec<f32>,
    response: CachedResponse,
    expires_at: Instant,
    last_used: Instant,
}

/// Result of a search. `score` is the best similarity in the namespace,
/// also when it is below the threshold, so callers can log near misses.
#[derive(Debug, Clone, PartialEq)]
pub struct Lookup {
    pub score: Option<f32>,
    pub hit: Option<CachedResponse>,
}

/// A cached entry that the vector search found, for the cross-encoder check.
#[derive(Debug, Clone, PartialEq)]
pub struct Candidate {
    pub score: f32,
    pub text: String,
    pub response: CachedResponse,
}

pub struct Index {
    capacity: usize,
    spaces: HashMap<String, Vec<Entry>>,
    len: usize,
}

impl Index {
    pub fn new(capacity: usize) -> Self {
        Self { capacity: capacity.max(1), spaces: HashMap::new(), len: 0 }
    }

    pub fn len(&self) -> usize {
        self.len
    }

    pub fn is_empty(&self) -> bool {
        self.len == 0
    }

    pub fn capacity(&self) -> usize {
        self.capacity
    }

    /// Find the most similar live entry in `ns`. It is a hit when its score
    /// is at or above `threshold`. A hit refreshes the entry for LRU.
    pub fn lookup(&mut self, ns: &str, query: &[f32], threshold: f32, now: Instant) -> Lookup {
        let Some(entries) = self.spaces.get_mut(ns) else {
            return Lookup { score: None, hit: None };
        };
        let best = entries
            .iter()
            .enumerate()
            .filter(|(_, e)| e.expires_at > now)
            .map(|(i, e)| (i, dot(&e.vector, query)))
            .max_by(|a, b| a.1.total_cmp(&b.1));

        match best {
            Some((i, score)) if score >= threshold => {
                let entry = &mut entries[i];
                entry.last_used = now;
                Lookup { score: Some(score), hit: Some(entry.response.clone()) }
            }
            Some((_, score)) => Lookup { score: Some(score), hit: None },
            None => Lookup { score: None, hit: None },
        }
    }

    /// The `k` most similar live entries of `ns` with a score at or above
    /// `floor`, best first. Also returns the best score of the namespace,
    /// below the floor too, for logging.
    pub fn candidates(&self, ns: &str, query: &[f32], floor: f32, k: usize, now: Instant) -> (Option<f32>, Vec<Candidate>) {
        let Some(entries) = self.spaces.get(ns) else {
            return (None, Vec::new());
        };
        let mut scored: Vec<(f32, &Entry)> =
            entries.iter().filter(|e| e.expires_at > now).map(|e| (dot(&e.vector, query), e)).collect();
        scored.sort_by(|a, b| b.0.total_cmp(&a.0));
        let best = scored.first().map(|(score, _)| *score);
        let found = scored
            .into_iter()
            .take_while(|(score, _)| *score >= floor)
            .take(k)
            .map(|(score, e)| Candidate { score, text: e.text.clone(), response: e.response.clone() })
            .collect();
        (best, found)
    }

    /// Refresh the entry with this text for LRU, after a confirmed hit.
    pub fn touch(&mut self, ns: &str, text: &str, now: Instant) {
        if let Some(entry) = self.spaces.get_mut(ns).and_then(|es| es.iter_mut().find(|e| e.text == text)) {
            entry.last_used = now;
        }
    }

    /// Store an answer. An entry with the same text in the same namespace is
    /// replaced. A similar vector is not enough: "dollars to euros" and
    /// "euros to dollars" have almost the same vector but need two entries.
    /// When the index is full, expired entries go first, then the least
    /// recently used entry of the whole index.
    pub fn insert(&mut self, ns: &str, text: &str, vector: Vec<f32>, response: CachedResponse, ttl: Duration, now: Instant) {
        let entry = Entry { text: text.to_owned(), vector, response, expires_at: now + ttl, last_used: now };

        if let Some(entries) = self.spaces.get_mut(ns)
            && let Some(dup) = entries.iter_mut().find(|e| e.text == entry.text)
        {
            *dup = entry;
            return;
        }

        if self.len >= self.capacity {
            self.remove_expired(now);
        }
        if self.len >= self.capacity {
            self.evict_lru();
        }
        self.spaces.entry(ns.to_owned()).or_default().push(entry);
        self.len += 1;
    }

    pub fn remove_expired(&mut self, now: Instant) {
        let mut removed = 0;
        self.spaces.retain(|_, entries| {
            let before = entries.len();
            entries.retain(|e| e.expires_at > now);
            removed += before - entries.len();
            !entries.is_empty()
        });
        self.len -= removed;
    }

    fn evict_lru(&mut self) {
        let oldest = self
            .spaces
            .iter()
            .flat_map(|(ns, entries)| entries.iter().enumerate().map(move |(i, e)| (ns, i, e.last_used)))
            .min_by_key(|(_, _, used)| *used)
            .map(|(ns, i, _)| (ns.clone(), i));

        if let Some((ns, i)) = oldest {
            let entries = self.spaces.get_mut(&ns).expect("namespace exists");
            entries.swap_remove(i);
            if entries.is_empty() {
                self.spaces.remove(&ns);
            }
            self.len -= 1;
        }
    }
}

/// Dot product with 8 independent sums, so the compiler can use SIMD lanes.
/// For L2-normalized vectors it equals the cosine similarity.
pub fn dot(a: &[f32], b: &[f32]) -> f32 {
    debug_assert_eq!(a.len(), b.len());
    let mut acc = [0f32; 8];
    let chunks_a = a.chunks_exact(8);
    let chunks_b = b.chunks_exact(8);
    let tail: f32 = chunks_a.remainder().iter().zip(chunks_b.remainder()).map(|(x, y)| x * y).sum();
    for (ca, cb) in chunks_a.zip(chunks_b) {
        for k in 0..8 {
            acc[k] += ca[k] * cb[k];
        }
    }
    acc.iter().sum::<f32>() + tail
}

#[cfg(test)]
mod tests {
    use super::*;

    fn unit(v: &[f32]) -> Vec<f32> {
        let n = v.iter().map(|x| x * x).sum::<f32>().sqrt();
        v.iter().map(|x| x / n).collect()
    }

    fn answer(text: &str) -> CachedResponse {
        CachedResponse { content: text.into(), model: "m".into(), finish_reason: "stop".into(), usage: None }
    }

    const HOUR: Duration = Duration::from_secs(3600);

    #[test]
    fn hit_above_threshold_and_miss_below() {
        let now = Instant::now();
        let mut idx = Index::new(10);
        idx.insert("ns", "q", unit(&[1.0, 0.0, 0.0]), answer("a"), HOUR, now);

        let close = idx.lookup("ns", &unit(&[1.0, 0.1, 0.0]), 0.9, now);
        assert_eq!(close.hit.unwrap().content, "a");

        let far = idx.lookup("ns", &unit(&[0.0, 1.0, 0.0]), 0.9, now);
        assert!(far.hit.is_none());
        assert!(far.score.unwrap().abs() < 1e-6);
    }

    #[test]
    fn namespaces_are_isolated() {
        let now = Instant::now();
        let mut idx = Index::new(10);
        idx.insert("a", "q", unit(&[1.0, 0.0]), answer("a"), HOUR, now);
        let other = idx.lookup("b", &unit(&[1.0, 0.0]), 0.5, now);
        assert_eq!(other, Lookup { score: None, hit: None });
    }

    #[test]
    fn expired_entries_never_hit() {
        let now = Instant::now();
        let mut idx = Index::new(10);
        idx.insert("ns", "q", unit(&[1.0, 0.0]), answer("a"), Duration::from_secs(1), now);
        let later = now + Duration::from_secs(2);
        assert!(idx.lookup("ns", &unit(&[1.0, 0.0]), 0.5, later).hit.is_none());
    }

    #[test]
    fn the_same_text_replaces_the_old_answer() {
        let now = Instant::now();
        let mut idx = Index::new(10);
        idx.insert("ns", "q", unit(&[1.0, 0.0]), answer("old"), HOUR, now);
        idx.insert("ns", "q", unit(&[1.0, 0.0]), answer("new"), HOUR, now);
        assert_eq!(idx.len(), 1);
        assert_eq!(idx.lookup("ns", &unit(&[1.0, 0.0]), 0.5, now).hit.unwrap().content, "new");
    }

    #[test]
    fn different_texts_with_the_same_vector_stay_apart() {
        let now = Instant::now();
        let mut idx = Index::new(10);
        idx.insert("ns", "dollars to euros", unit(&[1.0, 0.0]), answer("d2e"), HOUR, now);
        idx.insert("ns", "euros to dollars", unit(&[1.0, 0.0]), answer("e2d"), HOUR, now);
        assert_eq!(idx.len(), 2);
    }

    #[test]
    fn candidates_are_the_best_entries_above_the_floor() {
        let now = Instant::now();
        let mut idx = Index::new(10);
        idx.insert("ns", "a", unit(&[1.0, 0.0, 0.0]), answer("a"), HOUR, now);
        idx.insert("ns", "b", unit(&[1.0, 0.3, 0.0]), answer("b"), HOUR, now);
        idx.insert("ns", "c", unit(&[1.0, 0.6, 0.0]), answer("c"), HOUR, now);
        idx.insert("ns", "d", unit(&[0.0, 1.0, 0.0]), answer("d"), HOUR, now);

        // Scores: a 1.0, b 0.96, c 0.86, d 0.0.
        let (best, found) = idx.candidates("ns", &unit(&[1.0, 0.0, 0.0]), 0.8, 2, now);
        assert!((best.unwrap() - 1.0).abs() < 1e-6);
        let texts: Vec<&str> = found.iter().map(|c| c.text.as_str()).collect();
        assert_eq!(texts, ["a", "b"]); // "c" is above the floor, but k is 2

        let (best, found) = idx.candidates("ns", &unit(&[0.0, 0.0, 1.0]), 0.9, 2, now);
        assert!(best.unwrap().abs() < 1e-6);
        assert!(found.is_empty());
    }

    #[test]
    fn full_index_evicts_the_least_recently_used_entry() {
        let t0 = Instant::now();
        let mut idx = Index::new(2);
        idx.insert("ns", "x", unit(&[1.0, 0.0, 0.0]), answer("x"), HOUR, t0);
        idx.insert("ns", "y", unit(&[0.0, 1.0, 0.0]), answer("y"), HOUR, t0 + Duration::from_secs(1));
        // Touch "x", so "y" becomes the least recently used.
        idx.lookup("ns", &unit(&[1.0, 0.0, 0.0]), 0.9, t0 + Duration::from_secs(2));
        idx.insert("ns", "z", unit(&[0.0, 0.0, 1.0]), answer("z"), HOUR, t0 + Duration::from_secs(3));

        let now = t0 + Duration::from_secs(4);
        assert_eq!(idx.len(), 2);
        assert!(idx.lookup("ns", &unit(&[1.0, 0.0, 0.0]), 0.9, now).hit.is_some());
        assert!(idx.lookup("ns", &unit(&[0.0, 1.0, 0.0]), 0.9, now).hit.is_none());
    }

    #[test]
    fn dot_matches_a_plain_loop() {
        let a: Vec<f32> = (0..383).map(|i| (i as f32).sin()).collect();
        let b: Vec<f32> = (0..383).map(|i| (i as f32).cos()).collect();
        let plain: f32 = a.iter().zip(&b).map(|(x, y)| x * y).sum();
        assert!((dot(&a, &b) - plain).abs() < 1e-3);
    }
}
