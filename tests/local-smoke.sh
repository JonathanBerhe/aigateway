#!/usr/bin/env bash
# Run tests/smoke.sh against a local gateway and the local mock upstream.
# It starts both in the background, runs the checks, and stops both.

set -euo pipefail

ROOT=$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)
OPENRESTY=${OPENRESTY:-openresty}
MOCK=$ROOT/tests/mock

# A fixed test setup. AIGW_ENV_FILE=/dev/null ignores a developer .env file.
export AIGW_ENV_FILE=/dev/null
export AIGW_API_KEYS=test-key
export AIGW_CHEAP_BASE_URL=http://127.0.0.1:19000/cheap/v1
export AIGW_CHEAP_MODEL=mock-cheap-model
export AIGW_STRONG_BASE_URL=http://127.0.0.1:19000/strong/v1
export AIGW_STRONG_MODEL=mock-strong-model
export AIGW_STRONG_API_KEY=mock-strong-key
# The regex prompt guard needs no sidecar, so the smoke test checks it.
export AIGW_GUARD=regex AIGW_GUARD_ACTION=block
# A model shadow router with no sidecar behind it: every shadow call fails,
# and the test checks that it never touches the response.
export AIGW_ROUTER_SHADOW=model AIGW_ROUTER_ADDR=unix:/tmp/aigw-no-sidecar.sock
unset AIGW_CHEAP_API_KEY AIGW_ROUTER_THRESHOLD

stop_all() {
  "$ROOT/scripts/gateway.sh" stop || true
  AIGW_PREFIX=$MOCK "$ROOT/scripts/gateway.sh" stop || true
}
trap stop_all EXIT

wait_for() {
  local url=$1 i
  for i in $(seq 1 50); do
    curl -s -o /dev/null "$url" && return 0
    sleep 0.1
  done
  echo "local-smoke.sh: $url did not come up" >&2
  return 1
}

stop_all
mkdir -p "$MOCK/logs"
"$OPENRESTY" -p "$MOCK/" -c conf/nginx.conf
"$ROOT/scripts/gateway.sh" daemon

wait_for http://127.0.0.1:19000/
wait_for http://127.0.0.1:8001/healthz

AIGW_ACCESS_LOG=$ROOT/gateway/logs/access.log "$ROOT/tests/smoke.sh"
