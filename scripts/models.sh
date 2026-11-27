#!/usr/bin/env bash
# List the model folders with their run records (train.json).
#
# Usage: scripts/models.sh [folder]   (default models)

set -euo pipefail
DIR=${1:-models}
command -v jq >/dev/null || { echo "models.sh: jq is required" >&2; exit 1; }

printf '| Model | Trainer | Version | Finished | Commit | Dev metrics |\n|---|---|---|---|---|---|\n'
for d in "$DIR"/*/; do
  name=$(basename "$d")
  if [[ -f $d/train.json ]]; then
    jq -r --arg name "$name" '
      "| \($name) | \(.trainer) | \(.version // "-") | \(.finished_unix | todate | .[0:16]) | \(.git_commit // "-") | " +
      (.metrics | to_entries | map("\(.key) \(.value | if type == "number" then (. * 1000 | round / 1000) else . end)") | join(", ")) + " |"
    ' "$d/train.json"
  else
    echo "| $name | no run record | - | - | - | - |"
  fi
done
