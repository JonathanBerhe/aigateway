//! Load test of a running sidecar.
//!
//! For each concurrency level, the tool opens that many connections, like
//! that many NGINX workers, and sends requests over them as fast as the
//! sidecar answers. It reports the throughput and the latency of a request
//! (send to reply) at p50, p95 and p99.
//!
//! Input: a text file with one prompt per line. A lookup with a new text is
//! a miss, so the test measures the embedding, which is the cost of a lookup.
//!
//!   semcache-bench --listen unix:/tmp/aigw-semcache.sock --texts data/bench-short.txt

use clap::Parser;
use semcache::server::Listen;
use std::io::{BufRead, BufReader, Write};
use std::path::PathBuf;
use std::sync::atomic::{AtomicUsize, Ordering};
use std::time::{Duration, Instant};

#[derive(Parser, Debug)]
#[command(about = "Load test of a running semcache sidecar")]
struct Args {
    /// Sidecar address: unix:/path or host:port
    #[arg(long, default_value = "unix:/tmp/aigw-semcache.sock")]
    listen: Listen,

    /// One prompt per line
    #[arg(long)]
    texts: PathBuf,

    /// lookup (embedding and search), guard or route
    #[arg(long, default_value = "lookup")]
    op: String,

    /// Concurrent connections, one run per value
    #[arg(long, value_delimiter = ',', default_value = "1,2,4,8,16")]
    concurrency: Vec<usize>,

    /// Requests per run
    #[arg(long, default_value_t = 400)]
    requests: usize,
}

enum Conn {
    Unix(BufReader<std::os::unix::net::UnixStream>),
    Tcp(BufReader<std::net::TcpStream>),
}

impl Conn {
    fn open(listen: &Listen) -> std::io::Result<Conn> {
        Ok(match listen {
            Listen::Unix(path) => Conn::Unix(BufReader::new(std::os::unix::net::UnixStream::connect(path)?)),
            Listen::Tcp(addr) => {
                let s = std::net::TcpStream::connect(addr)?;
                s.set_nodelay(true)?;
                Conn::Tcp(BufReader::new(s))
            }
        })
    }

    /// Send one request line and read the reply line.
    fn call(&mut self, line: &[u8], reply: &mut String) -> std::io::Result<()> {
        reply.clear();
        match self {
            Conn::Unix(r) => {
                r.get_mut().write_all(line)?;
                r.read_line(reply)?;
            }
            Conn::Tcp(r) => {
                r.get_mut().write_all(line)?;
                r.read_line(reply)?;
            }
        }
        Ok(())
    }
}

fn request(op: &str, n: usize, text: &str) -> Vec<u8> {
    let body = match op {
        "lookup" => serde_json::json!({"op": "lookup", "id": format!("bench-{n}"), "ns": "bench", "text": text}),
        other => serde_json::json!({"op": other, "text": text}),
    };
    let mut line = serde_json::to_vec(&body).expect("json");
    line.push(b'\n');
    line
}

fn percentile(sorted: &[Duration], p: f64) -> f64 {
    if sorted.is_empty() {
        return 0.0;
    }
    let i = ((p * sorted.len() as f64).ceil() as usize).clamp(1, sorted.len()) - 1;
    sorted[i].as_secs_f64() * 1000.0
}

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let args = Args::parse();
    let texts: Vec<String> = std::fs::read_to_string(&args.texts)?
        .lines()
        .filter(|l| !l.trim().is_empty())
        .map(str::to_owned)
        .collect();
    if texts.is_empty() {
        return Err("no texts".into());
    }
    // A short warm-up, so the first run does not pay for cold caches.
    let mut warm = Conn::open(&args.listen)?;
    let mut reply = String::new();
    for (n, text) in texts.iter().take(20).enumerate() {
        warm.call(&request(&args.op, n, text), &mut reply)?;
    }

    println!("| Connections | Requests/s | p50 ms | p95 ms | p99 ms | Errors |");
    println!("|---|---|---|---|---|---|");
    let mut offset = 0;
    for &c in &args.concurrency {
        let next = AtomicUsize::new(0);
        let started = Instant::now();
        let results: Vec<(Vec<Duration>, usize)> = std::thread::scope(|scope| {
            let handles: Vec<_> = (0..c.max(1))
                .map(|_| {
                    let (next, texts, args) = (&next, &texts, &args);
                    scope.spawn(move || {
                        let (mut times, mut errors) = (Vec::new(), 0usize);
                        let Ok(mut conn) = Conn::open(&args.listen) else { return (times, args.requests) };
                        let mut reply = String::new();
                        loop {
                            let i = next.fetch_add(1, Ordering::Relaxed);
                            if i >= args.requests {
                                break;
                            }
                            let n = offset + i;
                            let line = request(&args.op, n, &texts[n % texts.len()]);
                            let t0 = Instant::now();
                            match conn.call(&line, &mut reply) {
                                Ok(()) if reply.contains("\"ok\":true") => times.push(t0.elapsed()),
                                _ => errors += 1,
                            }
                        }
                        (times, errors)
                    })
                })
                .collect();
            handles.into_iter().map(|h| h.join().unwrap_or_default()).collect()
        });
        let wall = started.elapsed().as_secs_f64();
        offset += args.requests;
        let errors: usize = results.iter().map(|r| r.1).sum();
        let mut times: Vec<Duration> = results.into_iter().flat_map(|r| r.0).collect();
        times.sort();
        println!(
            "| {c} | {:.0} | {:.1} | {:.1} | {:.1} | {errors} |",
            times.len() as f64 / wall,
            percentile(&times, 0.50),
            percentile(&times, 0.95),
            percentile(&times, 0.99)
        );
    }
    Ok(())
}
