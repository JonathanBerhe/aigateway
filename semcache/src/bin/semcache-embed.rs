//! Print the embedding of each input line as a JSON array, one per line.
//! With --rerank, read `text_a <TAB> text_b` lines and print the logit of
//! the cross-encoder for each pair. With --classifier, print the score of a
//! single-text classifier (a classifier router or the guardrail) for each line.
//!
//! train/parity.py uses it to make sure that Burn and PyTorch give the same
//! output for the same model:
//!   printf 'hello\nworld\n' | semcache-embed --model models/minilm-l6-qqp
//!   printf 'a\tb\n' | semcache-embed --rerank models/minilm-l6-qqp-rerank
//!   printf 'hello\n' | semcache-embed --classifier models/router-classifier --max-tokens 256

use clap::Parser;
use semcache::embed::{Embedder, parse_model};
use semcache::rerank::Reranker;
use semcache::classifier::TextClassifier;
use std::io::{BufRead, Write};
use std::path::PathBuf;

#[derive(Parser, Debug)]
#[command(about = "Embed lines from stdin and print JSON vectors")]
struct Args {
    /// l6, l12, or a local model folder
    #[arg(long, default_value = "l6")]
    model: String,

    /// Cross-encoder folder: score tab-separated pairs instead
    #[arg(long)]
    rerank: Option<PathBuf>,

    /// Single-text classifier folder: print the score of each line instead
    #[arg(long)]
    classifier: Option<PathBuf>,

    /// Token limit of the classifier (max_tokens in router.json or guard.json)
    #[arg(long, default_value_t = 256)]
    max_tokens: usize,
}

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let args = Args::parse();
    if let Some(dir) = &args.rerank {
        let reranker = Reranker::load(dir)?;
        for line in std::io::stdin().lock().lines() {
            let line = line?;
            let (a, b) = line.split_once('\t').ok_or("a --rerank line needs a tab")?;
            println!("{}", reranker.logits(&[(a, b)])?[0]);
        }
        return Ok(());
    }
    if let Some(dir) = &args.classifier {
        let classifier = TextClassifier::load(dir, args.max_tokens)?;
        for line in std::io::stdin().lock().lines() {
            println!("{}", classifier.score(&[line?.as_str()])?[0]);
        }
        return Ok(());
    }
    let embedder = Embedder::load(&parse_model(&args.model)?)?;
    let stdout = std::io::stdout();
    let mut out = stdout.lock();
    for line in std::io::stdin().lock().lines() {
        let line = line?;
        let embedding = embedder.embed(&[line.as_str()])?.remove(0);
        serde_json::to_writer(&mut out, &embedding.vector)?;
        out.write_all(b"\n")?;
    }
    Ok(())
}
