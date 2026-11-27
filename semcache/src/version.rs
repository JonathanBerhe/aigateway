//! Model versions and run records.
//!
//! A model version is a content fingerprint: FNV-1a (64 bit) over the bytes
//! of the model files and the settings file next to them, as 12 hex digits.
//! Two folders with the same files have the same version, and a retrained
//! model gets a new one. The sidecar reports the versions in its stats and
//! in its replies, so a log line names the exact model behind a decision.
//! FNV is not a cryptographic hash. It tells versions apart, it does not
//! protect against a forged file.
//!
//! A run record (train.json in a model folder) keeps what produced a model:
//! the trainer, its arguments, the data files with their fingerprints, the
//! dev metrics, the git commit and the time it took.

use std::io::Read;
use std::path::Path;
use std::time::{Instant, SystemTime, UNIX_EPOCH};

const FNV_OFFSET: u64 = 0xcbf2_9ce4_8422_2325;
const FNV_PRIME: u64 = 0x0000_0100_0000_01b3;

pub const RUN_FILE: &str = "train.json";

#[derive(Debug, Clone, Copy)]
struct Fnv(u64);

impl Fnv {
    fn new() -> Self {
        Fnv(FNV_OFFSET)
    }

    fn write(&mut self, bytes: &[u8]) {
        for &b in bytes {
            self.0 ^= u64::from(b);
            self.0 = self.0.wrapping_mul(FNV_PRIME);
        }
    }

    fn hex(self) -> String {
        format!("{:012x}", self.0 >> 16)
    }
}

/// Fingerprint of the files of `dir` that exist, in the given order. A file
/// name goes into the hash too, so moving bytes between files changes it.
pub fn fingerprint(dir: &Path, files: &[&str]) -> Result<String, String> {
    let mut h = Fnv::new();
    let mut buf = vec![0u8; 1 << 20];
    for name in files {
        let path = dir.join(name);
        let Ok(mut f) = std::fs::File::open(&path) else { continue };
        h.write(name.as_bytes());
        loop {
            let n = f.read(&mut buf).map_err(|e| format!("{}: {e}", path.display()))?;
            if n == 0 {
                break;
            }
            h.write(&buf[..n]);
        }
    }
    Ok(h.hex())
}

/// Fingerprint of one file.
pub fn file_fingerprint(path: &Path) -> Result<String, String> {
    let dir = path.parent().unwrap_or(Path::new("."));
    let name = path.file_name().and_then(|n| n.to_str()).ok_or_else(|| format!("{}: no file name", path.display()))?;
    fingerprint(dir, &[name])
}

/// The model files of a folder in the Hugging Face layout, plus its settings file.
pub fn model_version(dir: &Path, settings: &str) -> Result<String, String> {
    fingerprint(dir, &["model.safetensors", "config.json", settings])
}

/// The current git commit, with "-dirty" when the work tree has changes.
fn git_commit() -> Option<String> {
    let run = |args: &[&str]| std::process::Command::new("git").args(args).output().ok();
    let head = run(&["rev-parse", "--short", "HEAD"]).filter(|o| o.status.success())?;
    let commit = String::from_utf8_lossy(&head.stdout).trim().to_owned();
    let dirty = run(&["status", "--porcelain", "--untracked-files=no"]).is_some_and(|o| !o.stdout.is_empty());
    Some(if dirty { format!("{commit}-dirty") } else { commit })
}

/// Write train.json into `out`. `args` is the Debug form of the trainer
/// arguments, `data` the input files, and `metrics` the dev results.
/// `version` is the model version, when the trainer writes every file that
/// goes into it. The cache and cross-encoder trainers do not: semcache-eval
/// writes their threshold file later.
pub fn write_run_record(
    out: &Path,
    trainer: &str,
    args: &str,
    data: &[&Path],
    metrics: serde_json::Value,
    version: Option<String>,
    started: Instant,
) -> Result<(), String> {
    let data: Vec<serde_json::Value> = data
        .iter()
        .map(|p| serde_json::json!({"path": p.display().to_string(), "fingerprint": file_fingerprint(p).ok()}))
        .collect();
    let finished = SystemTime::now().duration_since(UNIX_EPOCH).map(|d| d.as_secs()).unwrap_or(0);
    let record = serde_json::json!({
        "trainer": trainer,
        "finished_unix": finished,
        "seconds": started.elapsed().as_secs(),
        "git_commit": git_commit(),
        "version": version,
        "args": args,
        "data": data,
        "metrics": metrics,
    });
    let path = out.join(RUN_FILE);
    let text = serde_json::to_string_pretty(&record).map_err(|e| e.to_string())?;
    std::fs::write(&path, text + "\n").map_err(|e| format!("{}: {e}", path.display()))
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn fnv_matches_the_reference_values() {
        // FNV-1a 64 of "" and "a", from the reference implementation.
        assert_eq!(Fnv::new().0, 0xcbf29ce484222325);
        let mut h = Fnv::new();
        h.write(b"a");
        assert_eq!(h.0, 0xaf63dc4c8601ec8c);
    }

    #[test]
    fn the_fingerprint_follows_the_content() {
        let dir = std::env::temp_dir().join(format!("aigw-version-test-{}", std::process::id()));
        std::fs::create_dir_all(&dir).unwrap();
        std::fs::write(dir.join("model.safetensors"), b"weights v1").unwrap();
        let v1 = model_version(&dir, "router.json").unwrap();
        assert_eq!(v1.len(), 12);
        assert_eq!(model_version(&dir, "router.json").unwrap(), v1, "same files, same version");
        std::fs::write(dir.join("router.json"), b"{}").unwrap();
        let v2 = model_version(&dir, "router.json").unwrap();
        assert_ne!(v1, v2, "a new settings file changes the version");
        std::fs::write(dir.join("model.safetensors"), b"weights v2").unwrap();
        assert_ne!(model_version(&dir, "router.json").unwrap(), v2);
        std::fs::remove_dir_all(&dir).unwrap();
    }
}
