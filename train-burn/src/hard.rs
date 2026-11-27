//! Hard negatives for the cross-encoder: small edits that change the meaning.
//!
//! QQP has few pairs that differ by a swap or by one word, and its labels are
//! lenient about them. These rules make such pairs from the training
//! questions, with the label "different":
//! - swap: the two sides of a direction or a comparison change places,
//!   for example "from Delhi to Mumbai" becomes "from Mumbai to Delhi"
//! - antonym: one word becomes its opposite, for example "best" becomes "worst"
//! - entity: one country, city, language or company becomes another one
//! - number: one number or year changes
//!
//! The word pairs, entities and swap verbs of semcache/eval/contrast.tsv are
//! left out (convert, translate, cast, merge, learn, evict; increase,
//! enable, gain, good, first, benefits; Austria, New York, Mac, Windows). So
//! the contrast set measures whether the model generalizes to new edits.
//! Only the generic "from X to Y" rule also covers one contrast pair.

use crate::data::{Pair, XorShift};
use regex::{Captures, Regex};
use std::collections::{HashMap, HashSet};

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum Kind {
    Swap,
    Antonym,
    Entity,
    Number,
}

const SWAP_RULES: &[&str] = &[
    r"(?i)\bfrom (?P<x>[a-z][\w.+#-]*) to (?P<y>[a-z][\w.+#-]*)",
    r"(?i)\b(?:switch|switching|migrate|migrating|transfer|transferring|change|changing|move|moving|exchange|exchanging|port|porting) (?P<x>[a-z][\w.+#-]*) (?:to|into|for) (?P<y>[a-z][\w.+#-]*)",
    r"(?i)\b(?:is|are|was|were) (?P<x>[a-z][\w.+#-]*) (?:better|worse|faster|slower|cheaper|safer|harder|easier|bigger|smaller|stronger|weaker|richer|poorer|older|younger|smarter|more popular|more powerful|more difficult|more expensive) than (?P<y>[a-z][\w.+#-]*)",
    r"(?i)\b(?:study|studying|take|taking|use|using|read|reading|watch|watching|visit|visiting) (?P<x>[a-z][\w.+#-]*) (?:before|after) (?P<y>[a-z][\w.+#-]*)",
    // Proper nouns only: without a part-of-speech tagger, a lowercase word
    // before the verb is often an adverb ("never actually invade the UK").
    // The subject must follow an auxiliary or a question word, as in "Can
    // India trust China?", not a preposition ("youngsters in India hate").
    r"\b(?i:can|could|did|does|do|will|would|should|why|how|when|where) (?P<x>[A-Z][a-z]+) (?:marry|married|kill|killed|beat|beats|defeat|defeated|love|loves|hate|hates|fire|fired|hire|hired|sue|sued|attack|attacked|invade|invaded|trust|trusts) (?P<y>[A-Z][a-z]+)",
];

/// Sides that make a swap keep the meaning ("from head to toe", "from
/// time to time"), or common verbs after "to" ("from Europe to find a job").
const IDIOM: &[&str] = &[
    "time", "day", "days", "year", "years", "head", "toe", "top", "bottom", "start", "end", "beginning", "finish",
    "scratch", "zero", "rags", "riches", "left", "right", "side", "point", "one", "another", "bad", "worse", "find",
    "get", "make", "be", "have", "go", "see", "know", "take", "become", "work", "live", "come", "give", "keep",
    "buy", "real", "till", "things", "stuff",
];

/// Words that make a swap meaningless or ungrammatical.
const STOP: &[&str] = &[
    "a", "an", "the", "i", "you", "he", "she", "it", "we", "they", "me", "him", "her", "us", "them", "my", "your",
    "his", "its", "our", "their", "this", "that", "these", "those", "one", "someone", "something", "anything",
    "everything", "nothing", "what", "which", "who", "how", "why", "when", "where", "there", "here", "and", "or",
    "not", "be", "to", "of", "in", "on", "at", "for", "with", "do", "does", "did", "can", "will", "would", "should",
    "all", "any", "some", "each", "every", "other", "own", "same", "so", "very", "just", "only", "also",
];

/// Opposite words, used in both directions.
const ANTONYMS: &[(&str, &str)] = &[
    ("best", "worst"),
    ("better", "worse"),
    ("buy", "sell"),
    ("buying", "selling"),
    ("advantages", "disadvantages"),
    ("pros", "cons"),
    ("before", "after"),
    ("start", "stop"),
    ("open", "close"),
    ("add", "remove"),
    ("install", "uninstall"),
    ("more", "less"),
    ("higher", "lower"),
    ("cheap", "expensive"),
    ("legal", "illegal"),
    ("positive", "negative"),
    ("maximum", "minimum"),
    ("import", "export"),
    ("upload", "download"),
    ("encrypt", "decrypt"),
    ("rich", "poor"),
    ("true", "false"),
    ("love", "hate"),
    ("older", "younger"),
    ("easiest", "hardest"),
    ("fastest", "slowest"),
    ("faster", "slower"),
    ("success", "failure"),
    ("successful", "unsuccessful"),
    ("happy", "sad"),
    ("strong", "weak"),
    ("strengths", "weaknesses"),
    ("male", "female"),
    ("men", "women"),
    ("boys", "girls"),
    ("husband", "wife"),
    ("father", "mother"),
    ("son", "daughter"),
    ("north", "south"),
    ("east", "west"),
    ("always", "never"),
    ("possible", "impossible"),
    ("safe", "dangerous"),
    ("healthy", "unhealthy"),
    ("agree", "disagree"),
    ("accept", "reject"),
    ("input", "output"),
    ("employer", "employee"),
    ("teacher", "student"),
    ("buyer", "seller"),
    ("winner", "loser"),
];

/// Interchangeable names. Matched with case.
const ENTITIES: &[&[&str]] = &[
    &[
        "India", "China", "Canada", "Germany", "France", "Italy", "Spain", "Japan", "Brazil", "Russia", "Pakistan",
        "Mexico", "Nigeria", "Egypt", "Turkey", "Bangladesh", "Indonesia", "Sweden", "Norway", "Kenya",
    ],
    &[
        "Mumbai", "Delhi", "Bangalore", "Chennai", "Hyderabad", "Pune", "Kolkata", "London", "Paris", "Berlin",
        "Tokyo", "Chicago", "Toronto", "Boston", "Seattle", "Dubai", "Singapore",
    ],
    &["Python", "Java", "JavaScript", "Ruby", "PHP", "Swift", "Kotlin", "Scala", "Perl", "Haskell"],
    &["Android", "iOS", "Linux", "Ubuntu"],
    &["Google", "Facebook", "Amazon", "Apple", "Microsoft", "Netflix", "Uber", "Tesla", "Samsung", "IBM"],
    &["Hindi", "English", "French", "German", "Spanish", "Chinese", "Japanese", "Arabic", "Russian"],
];

/// How the edits of one run were made.
#[derive(Debug, Default)]
pub struct Counts {
    pub by_kind: HashMap<Kind, usize>,
    pub examples: Vec<(Kind, String, String)>,
}

pub struct Rules {
    swaps: Vec<Regex>,
    word: Regex,
    number: Regex,
    stop: HashSet<&'static str>,
    antonym: HashMap<&'static str, &'static str>,
    entity: HashMap<&'static str, usize>,
}

impl Default for Rules {
    fn default() -> Self {
        Self::new()
    }
}

impl Rules {
    pub fn new() -> Self {
        let mut antonym = HashMap::new();
        for &(a, b) in ANTONYMS {
            antonym.insert(a, b);
            antonym.insert(b, a);
        }
        let entity = ENTITIES.iter().enumerate().flat_map(|(i, list)| list.iter().map(move |e| (*e, i))).collect();
        Self {
            swaps: SWAP_RULES.iter().map(|r| Regex::new(r).expect("swap rule")).collect(),
            word: Regex::new(r"[A-Za-z][A-Za-z']*").expect("word"),
            number: Regex::new(r"\b(?:19|20)\d\d\b|\b\d{1,2}\b").expect("number"),
            stop: STOP.iter().copied().collect(),
            antonym,
            entity,
        }
    }

    /// Swap the two sides of the first rule that matches with real words.
    pub fn swap(&self, text: &str) -> Option<String> {
        for rule in &self.swaps {
            for caps in rule.captures_iter(text) {
                if let Some(out) = self.swap_captures(text, &caps) {
                    return Some(out);
                }
            }
        }
        None
    }

    fn swap_captures(&self, text: &str, caps: &Captures) -> Option<String> {
        let (x, y) = (caps.name("x")?, caps.name("y")?);
        let (xl, yl) = (x.as_str().to_lowercase(), y.as_str().to_lowercase());
        let skip = |w: &str| self.stop.contains(w) || IDIOM.contains(&w) || w.ends_with("ing");
        if xl == yl || skip(&xl) || skip(&yl) {
            return None;
        }
        Some(format!("{}{}{}{}{}", &text[..x.start()], y.as_str(), &text[x.end()..y.start()], x.as_str(), &text[y.end()..]))
    }

    /// Replace one word by its antonym. Skipped when the text already has
    /// the antonym, as in "pros and cons".
    pub fn antonym(&self, text: &str, rng: &mut XorShift) -> Option<(String, &'static str)> {
        let words: HashSet<String> = self.word.find_iter(text).map(|m| m.as_str().to_lowercase()).collect();
        let found: Vec<_> = self
            .word
            .find_iter(text)
            .filter_map(|m| {
                let lower = m.as_str().to_lowercase();
                let (key, opposite) = self.antonym.get_key_value(lower.as_str())?;
                (!words.contains(*opposite)).then_some((m, *key, *opposite))
            })
            .collect();
        if found.is_empty() {
            return None;
        }
        let (m, key, opposite) = found[(rng.next_u64() % found.len() as u64) as usize];
        let replacement = if m.as_str().starts_with(char::is_uppercase) { capitalize(opposite) } else { opposite.to_string() };
        Some((format!("{}{}{}", &text[..m.start()], replacement, &text[m.end()..]), key))
    }

    /// Replace one known name by another of the same list.
    pub fn entity(&self, text: &str, rng: &mut XorShift) -> Option<String> {
        let found: Vec<_> = self.word.find_iter(text).filter_map(|m| Some((m, *self.entity.get(m.as_str())?))).collect();
        if found.is_empty() {
            return None;
        }
        let (m, list) = found[(rng.next_u64() % found.len() as u64) as usize];
        let others: Vec<&str> = ENTITIES[list].iter().copied().filter(|e| *e != m.as_str() && !text.contains(e)).collect();
        if others.is_empty() {
            return None;
        }
        let other = others[(rng.next_u64() % others.len() as u64) as usize];
        Some(format!("{}{}{}", &text[..m.start()], other, &text[m.end()..]))
    }

    /// Change one number: a year by 1 to 5, a small number by 1 to 3.
    pub fn number(&self, text: &str, rng: &mut XorShift) -> Option<String> {
        let found: Vec<_> = self.number.find_iter(text).collect();
        if found.is_empty() {
            return None;
        }
        let m = found[(rng.next_u64() % found.len() as u64) as usize];
        let n: i64 = m.as_str().parse().ok()?;
        let step = if m.as_str().len() == 4 { 1 + (rng.next_u64() % 5) as i64 } else { 1 + (rng.next_u64() % 3) as i64 };
        let new = if rng.next_u64() & 1 == 0 || n - step < 0 { n + step } else { n - step };
        Some(format!("{}{}{}", &text[..m.start()], new, &text[m.end()..]))
    }
}

fn capitalize(word: &str) -> String {
    let mut chars = word.chars();
    chars.next().map(|c| c.to_uppercase().collect::<String>() + chars.as_str()).unwrap_or_default()
}

/// Make up to `max` hard negatives from `questions`. A question gives at
/// most one edit. A first pass takes swaps, the rarest and most useful
/// kind, up to 35% of `max`. A second pass fills the rest with one of the
/// other kinds, chosen at random within shares that keep the mix varied:
/// antonyms 40% (each word pair at most 2%), entities 25%, numbers 10%.
pub fn make(questions: &[&str], max: usize, seed: u64) -> (Vec<Pair>, Counts) {
    let rules = Rules::new();
    let mut rng = XorShift::new(seed);
    let mut order: Vec<usize> = (0..questions.len()).collect();
    crate::data::shuffle(&mut order, seed);

    let share = |p: f64| (max as f64 * p).ceil() as usize;
    let caps: HashMap<Kind, usize> =
        [(Kind::Swap, share(0.35)), (Kind::Antonym, share(0.40)), (Kind::Entity, share(0.25)), (Kind::Number, share(0.10))]
            .into();
    let pair_cap = share(0.02).max(1);
    let mut per_pair: HashMap<&'static str, usize> = HashMap::new();
    let mut used = vec![false; questions.len()];
    let mut counts = Counts::default();
    let mut out = Vec::new();

    let keep = |kind: Kind, q: &str, edited: String, counts: &mut Counts, out: &mut Vec<Pair>| {
        if edited == q {
            return false;
        }
        *counts.by_kind.entry(kind).or_default() += 1;
        if counts.examples.iter().filter(|(k, _, _)| *k == kind).count() < 4 {
            counts.examples.push((kind, q.to_string(), edited.clone()));
        }
        out.push(Pair { a: q.to_string(), b: edited, same: false });
        true
    };
    let full = |kind: Kind, counts: &Counts| counts.by_kind.get(&kind).copied().unwrap_or(0) >= caps[&kind];

    for &i in &order {
        if full(Kind::Swap, &counts) || out.len() >= max {
            break;
        }
        if let Some(e) = rules.swap(questions[i]) {
            used[i] = keep(Kind::Swap, questions[i], e, &mut counts, &mut out);
        }
    }
    for &i in &order {
        if out.len() >= max {
            break;
        }
        if used[i] {
            continue;
        }
        let q = questions[i];
        let edit = match rng.next_u64() % 3 {
            0 if !full(Kind::Antonym, &counts) => rules.antonym(q, &mut rng).and_then(|(e, key)| {
                let n = per_pair.entry(key).or_default();
                (*n < pair_cap).then(|| {
                    *n += 1;
                    (Kind::Antonym, e)
                })
            }),
            1 if !full(Kind::Entity, &counts) => rules.entity(q, &mut rng).map(|e| (Kind::Entity, e)),
            2 if !full(Kind::Number, &counts) => rules.number(q, &mut rng).map(|e| (Kind::Number, e)),
            _ => None,
        };
        if let Some((kind, edited)) = edit {
            keep(kind, q, edited, &mut counts, &mut out);
        }
    }
    (out, counts)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn swaps_the_two_sides() {
        let rules = Rules::new();
        assert_eq!(rules.swap("How do I get from Delhi to Mumbai?").as_deref(), Some("How do I get from Mumbai to Delhi?"));
        assert_eq!(rules.swap("Is Python better than Java for beginners?").as_deref(), Some("Is Java better than Python for beginners?"));
        assert_eq!(rules.swap("Why did Brutus kill Caesar?").as_deref(), Some("Why did Caesar kill Brutus?"));
        // A pronoun, an adverb, an idiom or a verb form on one side: no swap.
        assert_eq!(rules.swap("How do I move it to the cloud?"), None);
        assert_eq!(rules.swap("Why do they hate me?"), None);
        assert_eq!(rules.swap("Does anyone truly love Jesus?"), None);
        assert_eq!(rules.swap("Why do youngsters in India hate Nehru family?"), None);
        assert_eq!(rules.swap("Which services should I get done from head to toe?"), None);
        assert_eq!(rules.swap("How do you talk someone down from trying to commit suicide?"), None);
    }

    #[test]
    fn antonym_keeps_case_and_skips_pairs_that_are_both_present() {
        let rules = Rules::new();
        let mut rng = XorShift::new(1);
        let (out, key) = rules.antonym("Best phones under 300?", &mut rng).unwrap();
        assert_eq!((out.as_str(), key), ("Worst phones under 300?", "best"));
        assert_eq!(rules.antonym("What are the pros and cons of Rust?", &mut rng), None);
    }

    #[test]
    fn entity_and_number_change_one_value() {
        let rules = Rules::new();
        let mut rng = XorShift::new(7);
        let out = rules.entity("Is it safe to travel to Egypt?", &mut rng).unwrap();
        assert!(out.starts_with("Is it safe to travel to ") && !out.contains("Egypt"), "{out}");
        let out = rules.number("Will Clinton run in 2016?", &mut rng).unwrap();
        assert!(out != "Will Clinton run in 2016?" && out.starts_with("Will Clinton run in 20"), "{out}");
    }

    #[test]
    fn make_respects_the_maximum() {
        let questions = ["How do I get from Delhi to Mumbai?", "Best phones?", "Is India safe?", "Top 10 books?"];
        let (pairs, counts) = make(&questions, 2, 42);
        assert!(pairs.len() <= 2);
        assert!(pairs.iter().all(|p| !p.same && p.a != p.b));
        assert_eq!(counts.by_kind.values().sum::<usize>(), pairs.len());
    }
}
