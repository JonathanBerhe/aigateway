//! aigw-semcache: the semantic cache sidecar of aigateway.

use clap::Parser;
use semcache::embed::{ModelSource, THRESHOLD_FILE, parse_model};
use semcache::rerank::{RERANK_FILE, RerankSettings};
use semcache::router::{ROUTER_FILE, RouterFile, RouterKind};
use semcache::guard::{GUARD_FILE, GuardFile};
use semcache::server::{EmbedModel, EmbedQueue, Guard, Listen, Rerank, Router, Shared, classifier_thread, rerank_thread, run};
use semcache::version::{fingerprint, model_version};
use std::path::PathBuf;
use std::sync::Arc;
use std::time::Instant;

#[derive(Parser, Debug)]
#[command(version, about = "Semantic cache sidecar for aigateway")]
struct Args {
    /// unix:/path/to.sock or host:port
    #[arg(long, default_value = "unix:/tmp/aigw-semcache.sock")]
    listen: Listen,

    /// l6 (faster), l12, or a local model folder (see train/)
    #[arg(long, default_value = "l6")]
    model: String,

    /// Similarity for a hit when the gateway sends none. Default: semcache.json
    /// of a model folder, or 0.97 for the Hub models. With --rerank, the
    /// similarity for a candidate. Default: rerank.json.
    #[arg(long)]
    threshold: Option<f32>,

    /// Cross-encoder folder: check each candidate hit with it
    #[arg(long)]
    rerank: Option<PathBuf>,

    /// Cross-encoder probability for a hit. Default: rerank.json.
    #[arg(long)]
    rerank_threshold: Option<f32>,

    /// Most candidates per lookup that the cross-encoder reads
    #[arg(long, default_value_t = 3)]
    candidates: usize,

    /// Trained router folder (router.json): answers the "route" request.
    /// A head router must belong to the --model embedding.
    #[arg(long)]
    router: Option<PathBuf>,

    /// Router score for the strong tier. Default: router.json.
    #[arg(long)]
    router_threshold: Option<f32>,

    /// Guardrail classifier folder (guard.json): answers the "guard" request
    #[arg(long)]
    guard: Option<PathBuf>,

    /// Guard score for an injection. Default: guard.json.
    #[arg(long)]
    guard_threshold: Option<f32>,

    /// Maximum number of cached answers, all namespaces together
    #[arg(long, default_value_t = 10_000)]
    capacity: usize,

    /// Maximum number of queued texts embedded in one forward pass. On the
    /// Flex CPU backend, batch 1 had the highest throughput (see README).
    #[arg(long, default_value_t = 1)]
    max_batch: usize,

    /// Embedding threads, each with its own copy of the model
    #[arg(long, default_value_t = 1)]
    embed_threads: usize,
}

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let args = Args::parse();
    args.listen.check_free().map_err(|e| e.to_string())?;
    let source = parse_model(&args.model)?;
    let label = source.label();
    let saved = args.rerank.as_deref().map(RerankSettings::load).transpose();
    let saved = match (saved, args.threshold, args.rerank_threshold) {
        // Both flags given: rerank.json is not needed.
        (Err(_), Some(_), Some(_)) => None,
        (saved, _, _) => saved?,
    };
    let threshold = match (args.threshold, saved) {
        (Some(t), _) => t,
        (None, Some(s)) => s.candidate_threshold,
        (None, None) => source.default_threshold()?,
    };

    let version = match &source {
        ModelSource::Dir(dir) => model_version(dir, THRESHOLD_FILE)?,
        ModelSource::Hub(variant) => format!("hub-{variant:?}").to_lowercase(),
    };
    let started = Instant::now();
    eprintln!("semcache: loading {label} (a Hub model is downloaded on the first run)");
    let queue = EmbedQueue::start(source, args.max_batch, args.embed_threads)?;
    let meaning = if args.rerank.is_some() { "candidate threshold" } else { "default threshold" };
    eprintln!("semcache: model ready in {:.1}s, version {version}, {meaning} {threshold}", started.elapsed().as_secs_f64());
    let rerank = match &args.rerank {
        Some(dir) => {
            let p = args.rerank_threshold.or(saved.map(|s| s.threshold)).ok_or("no cross-encoder threshold")?;
            let queue = rerank_thread(dir.clone())?;
            let version = model_version(dir, RERANK_FILE)?;
            eprintln!(
                "semcache: cross-encoder {} (version {version}) ready, hit at probability {p}, up to {} candidates",
                dir.display(),
                args.candidates
            );
            Some(Rerank { queue, model: dir.display().to_string(), version, threshold: p, candidates: args.candidates.max(1) })
        }
        None => None,
    };

    let router = match &args.router {
        Some(dir) => Some(load_router(dir, &args.model, args.router_threshold)?),
        None => None,
    };

    let guard = match &args.guard {
        Some(dir) => {
            let file = GuardFile::load(dir)?;
            let threshold = args.guard_threshold.unwrap_or(file.threshold);
            let queue = classifier_thread("guard", dir.clone(), file.max_tokens)?;
            let version = model_version(dir, GUARD_FILE)?;
            eprintln!("semcache: guard {} (version {version}) ready, injection at score {threshold}", dir.display());
            Some(Guard { queue, path: dir.display().to_string(), version, threshold })
        }
        None => None,
    };

    let runtime = tokio::runtime::Runtime::new()?;
    runtime.block_on(async {
        // The embedding threads warm up their models when they start.
        if let Some(r) = &rerank {
            r.queue.call(("warm up".into(), vec!["warm up".into()])).await?;
        }
        if let Some(q) = router.as_ref().and_then(|r| r.classifier.as_ref()) {
            q.call("warm up".into()).await?;
        }
        if let Some(g) = &guard {
            g.queue.call("warm up".into()).await?;
        }
        let model = EmbedModel { label, version, threshold };
        let shared = Arc::new(Shared::new(queue, args.capacity, model, rerank, router, guard));
        run(args.listen, shared).await.map_err(|e| e.to_string())
    })?;
    Ok(())
}

/// Two model names are the same model when their last path parts match, so
/// models/x and /models/x (a Docker mount) are one model. "l6" stays "l6".
fn same_model(a: &str, b: &str) -> bool {
    let name = |s: &str| s.trim_end_matches('/').rsplit('/').next().unwrap_or_default().to_owned();
    name(a) == name(b)
}

fn load_router(dir: &std::path::Path, model: &str, threshold: Option<f32>) -> Result<Router, String> {
    let file = RouterFile::load(dir)?;
    let threshold = threshold.unwrap_or(file.threshold);
    let classifier = match &file.kind {
        RouterKind::Head { embed_model, .. } => {
            if !same_model(embed_model, model) {
                return Err(format!(
                    "the router head {} belongs to the embedding model {embed_model}, but the sidecar runs --model {model}",
                    dir.display()
                ));
            }
            None
        }
        RouterKind::Classifier { max_tokens } => Some(classifier_thread("router", dir.to_path_buf(), *max_tokens)?),
    };
    let kind = if classifier.is_some() { "classifier" } else { "head" };
    // A head lives in router.json. A classifier also has its model files.
    let version = match kind {
        "head" => fingerprint(dir, &[ROUTER_FILE])?,
        _ => model_version(dir, ROUTER_FILE)?,
    };
    eprintln!("semcache: router {} ({kind}, version {version}) ready, strong tier at score {threshold}", dir.display());
    Ok(Router { file, path: dir.display().to_string(), version, threshold, classifier })
}
