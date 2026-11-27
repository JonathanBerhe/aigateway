#!/usr/bin/env bash
# Download labeled paraphrase pairs from the Quora Question Pairs (QQP)
# validation split, through the Hugging Face datasets-server API.
#
# Usage: scripts/fetch-qqp.sh [pairs] [output] [offset]
#   pairs   number of pairs (default 4000, the split has 40430)
#   output  TSV path (default data/qqp-dev.tsv)
#   offset  first row of the split (default 0). Use another range for a test set.
#
# Output columns: question1, question2, label (1 = same meaning).
# QQP labels are noisy, and its "different" pairs are often on the same
# topic. That makes it a strict test for a cache threshold.

set -euo pipefail

PAIRS=${1:-4000}
OUT=${2:-data/qqp-dev.tsv}
START=${3:-0}
PAGE=100
API="https://datasets-server.huggingface.co/rows?dataset=nyu-mll/glue&config=qqp&split=validation"

command -v jq >/dev/null || { echo "fetch-qqp.sh: jq is required" >&2; exit 1; }
mkdir -p "$(dirname "$OUT")"
tmp=$(mktemp)
trap 'rm -f "$tmp"' EXIT

printf 'question1\tquestion2\tlabel\n' >"$tmp"
for ((offset = 0; offset < PAIRS; offset += PAGE)); do
  length=$((PAIRS - offset < PAGE ? PAIRS - offset : PAGE))
  curl -fsS --retry 3 "$API&offset=$((START + offset))&length=$length" |
    jq -r '.rows[].row | [(.question1 | gsub("[\t\n\r]"; " ")), (.question2 | gsub("[\t\n\r]"; " ")), (.label | tostring)] | @tsv' >>"$tmp"
  printf '\rfetch-qqp.sh: %d / %d pairs' "$((offset + length))" "$PAIRS" >&2
done
echo >&2
mv "$tmp" "$OUT"
trap - EXIT
echo "fetch-qqp.sh: wrote $(($(wc -l <"$OUT") - 1)) pairs to $OUT" >&2
