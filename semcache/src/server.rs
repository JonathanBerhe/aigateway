//! Sidecar server: listener, connection loop and the embedding thread.
//!
//! Threads:
//! - The tokio runtime accepts connections and handles the line protocol.
//! - "embed" threads (--embed-threads, default 1) each own a copy of the model
//!   and share one queue. A thread takes jobs in micro-batches: it waits for
//!   one job, then also takes every job that is already queued, up to
//!   `max_batch`. Under load, one forward pass serves several requests. With
//!   no load, a job runs alone with no added wait.
//! - With --rerank, one "rerank" thread owns the cross-encoder. A job is one
//!   query with its candidates, scored in one forward pass.
//! - With a classifier router (--router), one "router" thread owns it. A head
//!   router needs no thread: it reads the embedding of the "embed" thread.
//! - With --guard, one "guard" thread owns the guardrail classifier.

use crate::embed::{Embedder, Embedding, ModelSource};
use crate::index::{Index, Lookup};
use crate::protocol::{Reply, Request};
use crate::rerank::Reranker;
use crate::classifier::TextClassifier;
use crate::router::RouterFile;
use serde::Serialize;
use std::collections::HashMap;
use std::hash::{DefaultHasher, Hash, Hasher};
use std::panic::{AssertUnwindSafe, catch_unwind};
use std::path::{Path, PathBuf};
use std::sync::{Arc, Mutex, MutexGuard};
use std::time::{Duration, Instant};
use tokio::io::{AsyncBufReadExt, AsyncRead, AsyncReadExt, AsyncWrite, AsyncWriteExt, BufReader};
use tokio::net::{TcpListener, UnixListener};
use tokio::sync::{mpsc, oneshot};

/// Largest request line. The gateway accepts bodies up to 8 MiB.
pub const MAX_LINE: usize = 16 * 1024 * 1024;

/// How long a query embedding waits for the store that follows a miss.
const PENDING_TTL: Duration = Duration::from_secs(300);
const MAX_PENDING: usize = 10_000;

/// How long the embedding of a route request waits for the cache lookup.
const ROUTED_TTL: Duration = Duration::from_secs(30);

// ---------------------------------------------------------------- embedding

struct EmbedJob {
    text: String,
    reply: oneshot::Sender<Result<Embedding, String>>,
}

#[derive(Clone)]
pub struct EmbedQueue {
    tx: mpsc::Sender<EmbedJob>,
}

impl EmbedQueue {
    /// Start `threads` embedding threads, each with its own copy of the model
    /// (about 90 MB for L6). Returns when every model is ready, or with the
    /// first load error.
    pub fn start(source: ModelSource, max_batch: usize, threads: usize) -> Result<Self, String> {
        let (tx, rx) = mpsc::channel::<EmbedJob>(1024);
        // The threads share the receiver. A thread holds the lock only while
        // it takes its jobs, not while it embeds them.
        let rx = Arc::new(Mutex::new(rx));
        let (ready_tx, ready_rx) = std::sync::mpsc::channel::<Result<(), String>>();
        let max_batch = max_batch.max(1);
        let threads = threads.max(1);

        for i in 0..threads {
            let (rx, ready_tx, source) = (rx.clone(), ready_tx.clone(), source.clone());
            std::thread::Builder::new()
                .name(format!("embed-{i}"))
                .spawn(move || {
                    // One embedding before the first request, in every thread,
                    // so no request pays the warm-up cost of a model copy.
                    let embedder = match Embedder::load(&source).and_then(|e| e.embed(&["warm up"]).map(|_| e)) {
                        Ok(e) => {
                            let _ = ready_tx.send(Ok(()));
                            e
                        }
                        Err(e) => {
                            let _ = ready_tx.send(Err(e));
                            return;
                        }
                    };
                    loop {
                        let jobs = {
                            let mut rx = rx.lock().unwrap_or_else(|p| p.into_inner());
                            let Some(first) = rx.blocking_recv() else { break };
                            let mut jobs = vec![first];
                            while jobs.len() < max_batch {
                                match rx.try_recv() {
                                    Ok(job) => jobs.push(job),
                                    Err(_) => break,
                                }
                            }
                            jobs
                        };
                        let result = {
                            let texts: Vec<&str> = jobs.iter().map(|j| j.text.as_str()).collect();
                            catch_unwind(AssertUnwindSafe(|| embedder.embed(&texts)))
                                .unwrap_or_else(|_| Err("the embedding code panicked".into()))
                        };
                        match result {
                            Ok(embeddings) => {
                                for (job, e) in jobs.into_iter().zip(embeddings) {
                                    let _ = job.reply.send(Ok(e));
                                }
                            }
                            Err(err) => {
                                for job in jobs {
                                    let _ = job.reply.send(Err(err.clone()));
                                }
                            }
                        }
                    }
                })
                .map_err(|e| e.to_string())?;
        }

        for _ in 0..threads {
            ready_rx.recv().map_err(|_| "an embedding thread stopped".to_string())??;
        }
        Ok(Self { tx })
    }

    pub async fn embed(&self, text: String) -> Result<Embedding, String> {
        let (reply, rx) = oneshot::channel();
        self.tx
            .send(EmbedJob { text, reply })
            .await
            .map_err(|_| "the embedding thread stopped".to_string())?;
        rx.await.map_err(|_| "the embedding thread dropped the job".to_string())?
    }
}

// ---------------------------------------------------------------- model threads

type Job<J, R> = (J, oneshot::Sender<Result<R, String>>);

/// A model on its own thread: the cross-encoder, the classifier router or
/// the guard. Jobs wait in a queue and run one at a time.
pub struct ModelThread<J, R> {
    tx: mpsc::Sender<Job<J, R>>,
    name: &'static str,
}

/// Query and candidate texts in, one "same meaning" probability per candidate out.
pub type RerankThread = ModelThread<(String, Vec<String>), Vec<f32>>;
/// One text in, one score out: the classifier router or the guard.
pub type ClassifierThread = ModelThread<String, f32>;

impl<J: Send + 'static, R: Send + 'static> ModelThread<J, R> {
    /// Start a thread named `name`, load its model with `load` and answer
    /// each job with `run`. Returns when the model is ready, or with the
    /// load error.
    pub fn start<M>(
        name: &'static str,
        load: impl FnOnce() -> Result<M, String> + Send + 'static,
        run: impl Fn(&M, J) -> Result<R, String> + Send + 'static,
    ) -> Result<Self, String> {
        let (tx, mut rx) = mpsc::channel::<Job<J, R>>(1024);
        let (ready_tx, ready_rx) = std::sync::mpsc::channel::<Result<(), String>>();
        std::thread::Builder::new()
            .name(name.into())
            .spawn(move || {
                let model = match load() {
                    Ok(m) => {
                        let _ = ready_tx.send(Ok(()));
                        m
                    }
                    Err(e) => {
                        let _ = ready_tx.send(Err(e));
                        return;
                    }
                };
                while let Some((job, reply)) = rx.blocking_recv() {
                    let result = catch_unwind(AssertUnwindSafe(|| run(&model, job)))
                        .unwrap_or_else(|_| Err(format!("the {name} code panicked")));
                    let _ = reply.send(result);
                }
            })
            .map_err(|e| e.to_string())?;
        ready_rx.recv().map_err(|_| format!("the {name} thread stopped"))??;
        Ok(Self { tx, name })
    }

    pub async fn call(&self, job: J) -> Result<R, String> {
        let name = self.name;
        let (reply, rx) = oneshot::channel();
        self.tx.send((job, reply)).await.map_err(|_| format!("the {name} thread stopped"))?;
        rx.await.map_err(|_| format!("the {name} thread dropped the job"))?
    }
}

/// The cross-encoder thread of the cache check.
pub fn rerank_thread(dir: PathBuf) -> Result<RerankThread, String> {
    ModelThread::start("rerank", move || Reranker::load(&dir), |model, (query, candidates): (String, Vec<String>)| {
        let pairs: Vec<(&str, &str)> = candidates.iter().map(|c| (query.as_str(), c.as_str())).collect();
        model.score(&pairs)
    })
}

/// The thread of a single-text classifier, named `name`.
pub fn classifier_thread(name: &'static str, dir: PathBuf, max_tokens: usize) -> Result<ClassifierThread, String> {
    ModelThread::start(name, move || TextClassifier::load(&dir, max_tokens), move |model, text: String| {
        model.score(&[text.as_str()])?.first().copied().ok_or_else(|| format!("no {name} score"))
    })
}

/// The trained router of the "route" request.
pub struct Router {
    pub file: RouterFile,
    /// The router folder, for the stats.
    pub path: String,
    /// Content fingerprint of the router files (semcache::version).
    pub version: String,
    /// A prompt goes to the strong tier when its score reaches this value.
    pub threshold: f32,
    /// The classifier thread. None for a head router.
    pub classifier: Option<ClassifierThread>,
}

/// The guardrail classifier of the "guard" request.
pub struct Guard {
    pub queue: ClassifierThread,
    /// The guard folder, for the stats.
    pub path: String,
    /// Content fingerprint of the guard files (semcache::version).
    pub version: String,
    /// A prompt with a score at or above this value is an injection.
    pub threshold: f32,
}

/// The cross-encoder check. With it, the lookup threshold is the candidate
/// floor, and a candidate is a hit only when the cross-encoder agrees.
pub struct Rerank {
    pub queue: RerankThread,
    pub model: String,
    /// Content fingerprint of the cross-encoder files (semcache::version).
    pub version: String,
    /// Cross-encoder probability for a hit.
    pub threshold: f32,
    /// Most candidates checked per lookup, best cosine first.
    pub candidates: usize,
}

// ---------------------------------------------------------------- state

struct Pending {
    ns: String,
    vector: Vec<f32>,
    at: Instant,
}

/// The embedding of a route request, for the lookup of the same request.
struct Routed {
    text_hash: u64,
    embedding: Embedding,
    at: Instant,
}

fn text_hash(text: &str) -> u64 {
    let mut h = DefaultHasher::new();
    text.hash(&mut h);
    h.finish()
}

#[derive(Debug, Default, Serialize)]
struct Counters {
    lookups: u64,
    hits: u64,
    misses: u64,
    skips: u64,
    stores: u64,
    reembeds: u64,
    errors: u64,
    /// Lookups that had candidates for the cross-encoder.
    rerank_checks: u64,
    /// Checks where the cross-encoder refused every candidate.
    rerank_rejects: u64,
    /// Route requests, and the ones that picked the strong tier.
    routes: u64,
    routes_strong: u64,
    /// Lookups that reused the embedding of a route request.
    route_reuses: u64,
    /// Guard requests, and the ones that found an injection.
    guards: u64,
    guard_flags: u64,
}

struct State {
    index: Index,
    pending: HashMap<String, Pending>,
    routed: HashMap<String, Routed>,
    counters: Counters,
}

/// The embedding model of the cache.
pub struct EmbedModel {
    pub label: String,
    /// Content fingerprint of the model folder, or "hub-..." for a Hub model.
    pub version: String,
    /// Used when a lookup does not send its own threshold. With the
    /// cross-encoder check, it is the candidate floor.
    pub threshold: f32,
}

pub struct Shared {
    state: Mutex<State>,
    embed: EmbedQueue,
    model: EmbedModel,
    rerank: Option<Rerank>,
    router: Option<Router>,
    guard: Option<Guard>,
}

impl Shared {
    pub fn new(
        embed: EmbedQueue,
        capacity: usize,
        model: EmbedModel,
        rerank: Option<Rerank>,
        router: Option<Router>,
        guard: Option<Guard>,
    ) -> Self {
        let state = State {
            index: Index::new(capacity),
            pending: HashMap::new(),
            routed: HashMap::new(),
            counters: Counters::default(),
        };
        Self { state: Mutex::new(state), embed, model, rerank, router, guard }
    }

    fn lock(&self) -> MutexGuard<'_, State> {
        // A panic while the lock is held leaves valid data, so ignore poisoning.
        self.state.lock().unwrap_or_else(|p| p.into_inner())
    }
}

impl State {
    fn remember(&mut self, id: String, ns: String, vector: Vec<f32>, now: Instant) {
        if self.pending.len() >= MAX_PENDING {
            self.pending.retain(|_, p| now.duration_since(p.at) < PENDING_TTL);
        }
        if self.pending.len() < MAX_PENDING {
            self.pending.insert(id, Pending { ns, vector, at: now });
        }
    }

    fn take_pending(&mut self, id: &str, ns: &str, now: Instant) -> Option<Vec<f32>> {
        let p = self.pending.remove(id)?;
        (p.ns == ns && now.duration_since(p.at) < PENDING_TTL).then_some(p.vector)
    }

    fn remember_routed(&mut self, id: String, text: &str, embedding: Embedding, now: Instant) {
        if self.routed.len() >= MAX_PENDING {
            self.routed.retain(|_, r| now.duration_since(r.at) < ROUTED_TTL);
        }
        if self.routed.len() < MAX_PENDING {
            self.routed.insert(id, Routed { text_hash: text_hash(text), embedding, at: now });
        }
    }

    fn take_routed(&mut self, id: &str, text: &str, now: Instant) -> Option<Embedding> {
        let r = self.routed.remove(id)?;
        (r.text_hash == text_hash(text) && now.duration_since(r.at) < ROUTED_TTL).then_some(r.embedding)
    }
}

// ---------------------------------------------------------------- requests

async fn handle(shared: &Shared, req: Request) -> Reply {
    match req {
        Request::Ping => Reply::ok(),

        Request::Stats => {
            let st = shared.lock();
            let rerank = shared.rerank.as_ref().map(|r| {
                serde_json::json!({"model": r.model, "version": r.version, "threshold": r.threshold, "candidates": r.candidates})
            });
            let router = shared.router.as_ref().map(|r| {
                let kind = if r.classifier.is_some() { "classifier" } else { "head" };
                serde_json::json!({"model": r.path, "version": r.version, "kind": kind, "threshold": r.threshold})
            });
            let guard = shared
                .guard
                .as_ref()
                .map(|g| serde_json::json!({"model": g.path, "version": g.version, "threshold": g.threshold}));
            let stats = serde_json::json!({
                "model": shared.model.label,
                "version": shared.model.version,
                "threshold": shared.model.threshold,
                "rerank": rerank,
                "router": router,
                "guard": guard,
                "entries": st.index.len(),
                "capacity": st.index.capacity(),
                "pending": st.pending.len(),
                "counters": st.counters,
            });
            Reply { stats: Some(stats), ..Reply::ok() }
        }

        Request::Route { id, text } => {
            let Some(router) = &shared.router else {
                return Reply::error("no router: start the sidecar with --router");
            };
            let started = Instant::now();
            let (score, embed_ms) = match &router.classifier {
                Some(queue) => match queue.call(text).await {
                    Ok(s) => (s, None),
                    Err(e) => {
                        shared.lock().counters.errors += 1;
                        return Reply::error(e);
                    }
                },
                None => {
                    let emb = match shared.embed.embed(text.clone()).await {
                        Ok(e) => e,
                        Err(e) => {
                            shared.lock().counters.errors += 1;
                            return Reply::error(e);
                        }
                    };
                    let embed_ms = Some(started.elapsed().as_secs_f64() * 1000.0);
                    let score = router.file.head_score(&emb.vector).unwrap_or(0.5);
                    if let Some(id) = id {
                        shared.lock().remember_routed(id, &text, emb, Instant::now());
                    }
                    (score, embed_ms)
                }
            };
            let strong = score >= router.threshold;
            let mut st = shared.lock();
            st.counters.routes += 1;
            st.counters.routes_strong += u64::from(strong);
            let route_ms = Some(started.elapsed().as_secs_f64() * 1000.0);
            let version = Some(router.version.clone());
            Reply { score: Some(score), strong: Some(strong), embed_ms, route_ms, version, ..Reply::ok() }
        }

        Request::Guard { text } => {
            let Some(guard) = &shared.guard else {
                return Reply::error("no guard: start the sidecar with --guard");
            };
            let started = Instant::now();
            let score = match guard.queue.call(text).await {
                Ok(s) => s,
                Err(e) => {
                    shared.lock().counters.errors += 1;
                    return Reply::error(e);
                }
            };
            let injection = score >= guard.threshold;
            let mut st = shared.lock();
            st.counters.guards += 1;
            st.counters.guard_flags += u64::from(injection);
            let guard_ms = Some(started.elapsed().as_secs_f64() * 1000.0);
            let version = Some(guard.version.clone());
            Reply { score: Some(score), injection: Some(injection), guard_ms, version, ..Reply::ok() }
        }

        Request::Lookup { id, ns, text, threshold } => {
            let started = Instant::now();
            let query = shared.rerank.is_some().then(|| text.clone());
            let reused = shared.lock().take_routed(&id, &text, started);
            let emb = match reused {
                Some(e) => {
                    shared.lock().counters.route_reuses += 1;
                    e
                }
                None => match shared.embed.embed(text).await {
                    Ok(e) => e,
                    Err(e) => {
                        shared.lock().counters.errors += 1;
                        return Reply::error(e);
                    }
                },
            };
            let embed_ms = Some(started.elapsed().as_secs_f64() * 1000.0);
            let threshold = threshold.unwrap_or(shared.model.threshold);

            // The lock must not live across the await below, so this block
            // ends before the cross-encoder runs.
            let (best, candidates, rerank, query) = {
                let now = Instant::now();
                let mut st = shared.lock();
                st.counters.lookups += 1;
                if emb.truncated {
                    // The embedding misses the end of the text. Two prompts that
                    // share their first 256 tokens would look identical.
                    st.counters.skips += 1;
                    return Reply { hit: Some(false), skip: Some("too_long"), embed_ms, ..Reply::ok() };
                }
                let (Some(rerank), Some(query)) = (&shared.rerank, query) else {
                    let Lookup { score, hit } = st.index.lookup(&ns, &emb.vector, threshold, now);
                    if hit.is_some() {
                        st.counters.hits += 1;
                    } else {
                        st.counters.misses += 1;
                        st.remember(id, ns, emb.vector, now);
                    }
                    return Reply { hit: Some(hit.is_some()), score, response: hit, embed_ms, ..Reply::ok() };
                };

                // Two stages: the vector search finds candidates, then the
                // cross-encoder reads the query with each candidate text.
                let (best, candidates) = st.index.candidates(&ns, &emb.vector, threshold, rerank.candidates, now);
                if candidates.is_empty() {
                    st.counters.misses += 1;
                    st.remember(id, ns, emb.vector, now);
                    return Reply { hit: Some(false), score: best, embed_ms, ..Reply::ok() };
                }
                // The same text has the same meaning: no check needed.
                if let Some(c) = candidates.iter().find(|c| c.text == query) {
                    st.counters.hits += 1;
                    st.index.touch(&ns, &query, now);
                    return Reply {
                        hit: Some(true),
                        score: Some(c.score),
                        response: Some(c.response.clone()),
                        embed_ms,
                        ..Reply::ok()
                    };
                }
                st.counters.rerank_checks += 1;
                (best, candidates, rerank, query)
            };

            let t0 = Instant::now();
            let texts = candidates.iter().map(|c| c.text.clone()).collect();
            let probs = match rerank.queue.call((query, texts)).await {
                Ok(p) => p,
                Err(e) => {
                    shared.lock().counters.errors += 1;
                    return Reply::error(e);
                }
            };
            let rerank_ms = Some(t0.elapsed().as_secs_f64() * 1000.0);
            // No score at all counts as a refusal (probability 0).
            let (i, p) = probs.iter().copied().enumerate().max_by(|a, b| a.1.total_cmp(&b.1)).unwrap_or((0, 0.0));

            let now = Instant::now();
            let mut st = shared.lock();
            if p >= rerank.threshold {
                st.counters.hits += 1;
                st.index.touch(&ns, &candidates[i].text, now);
                let c = &candidates[i];
                Reply {
                    hit: Some(true),
                    score: Some(c.score),
                    rerank: Some(p),
                    response: Some(c.response.clone()),
                    embed_ms,
                    rerank_ms,
                    ..Reply::ok()
                }
            } else {
                st.counters.misses += 1;
                st.counters.rerank_rejects += 1;
                st.remember(id, ns, emb.vector, now);
                Reply { hit: Some(false), score: best, rerank: Some(p), embed_ms, rerank_ms, ..Reply::ok() }
            }
        }

        Request::Store { id, ns, text, ttl, response } => {
            let pending = shared.lock().take_pending(&id, &ns, Instant::now());
            let vector = match pending {
                Some(v) => v,
                None => match shared.embed.embed(text.clone()).await {
                    Ok(e) if e.truncated => {
                        return Reply { stored: Some(false), skip: Some("too_long"), ..Reply::ok() };
                    }
                    Ok(e) => {
                        shared.lock().counters.reembeds += 1;
                        e.vector
                    }
                    Err(e) => {
                        shared.lock().counters.errors += 1;
                        return Reply::error(e);
                    }
                },
            };
            let mut st = shared.lock();
            st.index.insert(&ns, &text, vector, response, Duration::from_secs(ttl.max(1)), Instant::now());
            st.counters.stores += 1;
            Reply { stored: Some(true), ..Reply::ok() }
        }
    }
}

async fn serve_connection<S>(stream: S, shared: Arc<Shared>)
where
    S: AsyncRead + AsyncWrite + Unpin,
{
    let (read_half, mut write_half) = tokio::io::split(stream);
    let mut reader = BufReader::new(read_half);
    let mut line = Vec::new();

    loop {
        line.clear();
        let n = match (&mut reader).take(MAX_LINE as u64 + 1).read_until(b'\n', &mut line).await {
            Ok(n) => n,
            Err(_) => return,
        };
        if n == 0 {
            return; // the gateway closed the connection
        }
        let too_long = line.len() > MAX_LINE;
        let reply = if too_long {
            Reply::error("request line is too long")
        } else {
            match serde_json::from_slice::<Request>(line.trim_ascii()) {
                Ok(req) => handle(&shared, req).await,
                Err(e) => Reply::error(format!("bad request: {e}")),
            }
        };
        let mut out = serde_json::to_vec(&reply).unwrap_or_else(|_| br#"{"ok":false,"error":"encode"}"#.to_vec());
        out.push(b'\n');
        if write_half.write_all(&out).await.is_err() || too_long {
            return;
        }
    }
}

// ---------------------------------------------------------------- listener

#[derive(Debug, Clone)]
pub enum Listen {
    Unix(PathBuf),
    Tcp(String),
}

impl Listen {
    /// Fail early when the address is taken, before the models load. A
    /// first run with a Hub model downloads it, so a late error wastes that
    /// time. `run` checks again when it binds.
    pub fn check_free(&self) -> std::io::Result<()> {
        match self {
            Listen::Unix(path) => unix_socket_state(path).map(|_| ()),
            // Binding and dropping the listener frees the port again.
            Listen::Tcp(addr) => std::net::TcpListener::bind(addr).map(drop),
        }
    }
}

impl std::str::FromStr for Listen {
    type Err = String;
    fn from_str(s: &str) -> Result<Self, String> {
        match s.strip_prefix("unix:") {
            Some(path) if !path.is_empty() => Ok(Listen::Unix(PathBuf::from(path))),
            Some(_) => Err("unix: needs a path, for example unix:/tmp/aigw-semcache.sock".into()),
            None if s.contains(':') => Ok(Listen::Tcp(s.to_owned())),
            None => Err(format!("'{s}' is not unix:/path or host:port")),
        }
    }
}

async fn shutdown_signal() {
    let ctrl_c = tokio::signal::ctrl_c();
    let mut term = tokio::signal::unix::signal(tokio::signal::unix::SignalKind::terminate())
        .expect("install SIGTERM handler");
    tokio::select! {
        _ = ctrl_c => {}
        _ = term.recv() => {}
    }
}

/// The unix socket file that this process created. On drop, it deletes the
/// file, but only if the file at the path is still the one it created.
struct SocketFile {
    path: PathBuf,
    id: (u64, u64),
}

/// What is at a unix socket path, when it does not block a new sidecar.
#[derive(Debug, PartialEq)]
enum SocketState {
    Free,
    /// A socket that no process listens on: left by a crash.
    Stale,
}

/// A live socket at `path` means that another sidecar runs: that is an
/// error. Any file there that is not a socket is an error too.
fn unix_socket_state(path: &Path) -> std::io::Result<SocketState> {
    use std::io::{Error, ErrorKind};
    use std::os::unix::fs::FileTypeExt;
    let Ok(meta) = std::fs::symlink_metadata(path) else { return Ok(SocketState::Free) };
    if !meta.file_type().is_socket() {
        let msg = format!("{} exists and is not a socket", path.display());
        return Err(Error::new(ErrorKind::AlreadyExists, msg));
    }
    if std::os::unix::net::UnixStream::connect(path).is_ok() {
        let msg = format!("another sidecar listens on unix:{}, stop it first", path.display());
        return Err(Error::new(ErrorKind::AddrInUse, msg));
    }
    Ok(SocketState::Stale)
}

impl SocketFile {
    /// Bind a socket at `path`. A socket that nobody listens on is left by
    /// a crash, and is replaced. See `unix_socket_state` for the errors.
    fn bind(path: PathBuf) -> std::io::Result<(std::os::unix::net::UnixListener, SocketFile)> {
        if unix_socket_state(&path)? == SocketState::Stale {
            std::fs::remove_file(&path)?;
        }
        let listener = std::os::unix::net::UnixListener::bind(&path)?;
        let id = file_id(&path)?;
        Ok((listener, SocketFile { path, id }))
    }
}

impl Drop for SocketFile {
    fn drop(&mut self) {
        if file_id(&self.path).ok() == Some(self.id) {
            let _ = std::fs::remove_file(&self.path);
        }
    }
}

/// Device and inode: they tell two files at the same path apart.
fn file_id(path: &Path) -> std::io::Result<(u64, u64)> {
    use std::os::unix::fs::MetadataExt;
    let meta = std::fs::symlink_metadata(path)?;
    Ok((meta.dev(), meta.ino()))
}

/// Accept connections until SIGINT or SIGTERM.
pub async fn run(listen: Listen, shared: Arc<Shared>) -> std::io::Result<()> {
    let shutdown = shutdown_signal();
    tokio::pin!(shutdown);

    match listen {
        Listen::Unix(path) => {
            let (listener, _file) = SocketFile::bind(path.clone())?;
            listener.set_nonblocking(true)?;
            let listener = UnixListener::from_std(listener)?;
            eprintln!("semcache: listening on unix:{}", path.display());
            loop {
                tokio::select! {
                    accepted = listener.accept() => {
                        let (stream, _) = accepted?;
                        tokio::spawn(serve_connection(stream, shared.clone()));
                    }
                    _ = &mut shutdown => break,
                }
            }
        }
        Listen::Tcp(addr) => {
            let listener = TcpListener::bind(&addr).await?;
            eprintln!("semcache: listening on {addr}");
            loop {
                tokio::select! {
                    accepted = listener.accept() => {
                        let (stream, _) = accepted?;
                        let _ = stream.set_nodelay(true);
                        tokio::spawn(serve_connection(stream, shared.clone()));
                    }
                    _ = &mut shutdown => break,
                }
            }
        }
    }
    eprintln!("semcache: stopped");
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::{Listen, SocketFile};
    use std::os::unix::net::UnixListener;
    use std::path::PathBuf;

    /// A short path: macOS limits a socket path to 104 bytes.
    fn socket_path(name: &str) -> PathBuf {
        let path = PathBuf::from(format!("/tmp/aigw-test-{}-{name}.sock", std::process::id()));
        let _ = std::fs::remove_file(&path);
        path
    }

    #[test]
    fn a_live_socket_is_refused() {
        let path = socket_path("live");
        let (_listener, _file) = SocketFile::bind(path.clone()).unwrap();
        let err = SocketFile::bind(path.clone()).err().unwrap();
        assert_eq!(err.kind(), std::io::ErrorKind::AddrInUse);
        assert!(path.exists());
    }

    #[test]
    fn a_socket_left_by_a_crash_is_replaced() {
        let path = socket_path("crash");
        drop(UnixListener::bind(&path).unwrap()); // closed, but the file stays
        let (_listener, file) = SocketFile::bind(path.clone()).unwrap();
        drop(file);
        assert!(!path.exists());
    }

    #[test]
    fn exit_keeps_a_socket_file_that_another_process_made() {
        let path = socket_path("replaced");
        let (old_listener, old_file) = SocketFile::bind(path.clone()).unwrap();
        drop(old_listener);
        let (_listener, new_file) = SocketFile::bind(path.clone()).unwrap();
        drop(old_file);
        assert!(path.exists(), "the old process deleted the new socket");
        drop(new_file);
        assert!(!path.exists());
    }

    #[test]
    fn the_early_check_finds_a_live_socket_and_keeps_a_stale_one() {
        let path = socket_path("early");
        let listen = Listen::Unix(path.clone());
        assert!(listen.check_free().is_ok(), "no file yet");
        let (listener, file) = SocketFile::bind(path.clone()).unwrap();
        assert_eq!(listen.check_free().err().unwrap().kind(), std::io::ErrorKind::AddrInUse);
        drop(listener); // the file stays, nobody listens
        assert!(listen.check_free().is_ok());
        assert!(path.exists(), "the early check deletes nothing");
        drop(file);
    }

    #[test]
    fn the_early_check_finds_a_taken_tcp_port() {
        let taken = std::net::TcpListener::bind("127.0.0.1:0").unwrap();
        let addr = taken.local_addr().unwrap().to_string();
        assert!(Listen::Tcp(addr.clone()).check_free().is_err());
        drop(taken);
        assert!(Listen::Tcp(addr).check_free().is_ok());
    }

    #[test]
    fn a_file_that_is_not_a_socket_is_kept() {
        let path = socket_path("regular");
        std::fs::write(&path, "data").unwrap();
        let err = SocketFile::bind(path.clone()).err().unwrap();
        assert_eq!(err.kind(), std::io::ErrorKind::AlreadyExists);
        assert_eq!(std::fs::read_to_string(&path).unwrap(), "data");
        std::fs::remove_file(&path).unwrap();
    }
}
