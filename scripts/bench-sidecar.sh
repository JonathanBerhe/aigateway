#!/usr/bin/env bash
# Load test of the sidecar with several embedding thread counts.
#
# For each setup, it starts a sidecar on a test socket, records its memory
# (resident set size), runs semcache-bench with short prompts (QQP questions)
# and long prompts (SPROUT), records the memory again and stops the sidecar.
#
# Usage: scripts/bench-sidecar.sh [model]   (default models/minilm-l6-qqp-burn)
# Needs data/bench-short.txt and data/bench-long.txt (make bench-data).

set -euo pipefail

ROOT=$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)
MODEL=${1:-models/minilm-l6-qqp-burn}
BIN=$ROOT/semcache/target/release
SOCK=/tmp/aigw-bench-$$.sock
LOG=$(mktemp)
trap 'rm -f "$LOG"' EXIT

rss_mb() { echo $(( $(ps -o rss= -p "$1") / 1024 )); }

# setup: "<embed threads> <max batch>"
for setup in "1 1" "2 1" "4 1" "6 1" "1 8"; do
  read -r threads batch <<<"$setup"
  "$BIN/aigw-semcache" --listen "unix:$SOCK" --model "$MODEL" --embed-threads "$threads" \
    --max-batch "$batch" >"$LOG" 2>&1 &
  pid=$!
  for _ in $(seq 1 300); do grep -q listening "$LOG" && break; sleep 0.1; done
  grep -q listening "$LOG" || { cat "$LOG"; kill "$pid"; exit 1; }
  idle=$(rss_mb "$pid")

  echo "## $threads embedding thread(s), max batch $batch: $idle MB at start"
  echo "short prompts (QQP):"
  "$BIN/semcache-bench" --listen "unix:$SOCK" --texts "$ROOT/data/bench-short.txt" --requests 800
  echo "long prompts (SPROUT):"
  "$BIN/semcache-bench" --listen "unix:$SOCK" --texts "$ROOT/data/bench-long.txt" --requests 300
  echo "memory after the load: $(rss_mb "$pid") MB"
  echo
  kill "$pid"
  wait "$pid" 2>/dev/null || true
done
