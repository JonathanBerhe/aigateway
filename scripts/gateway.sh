#!/usr/bin/env bash
# Start, stop or check the aigateway data plane with OpenResty.
#
# Usage: scripts/gateway.sh [start|daemon|stop|reload|test]
#   start   run in the foreground (default; also the Docker entrypoint)
#   daemon  run in the background
#   stop    stop a background gateway
#   reload  reload nginx.conf and the Lua code without dropping connections
#   test    validate the configuration and exit
#
# Configuration comes from AIGW_* environment variables (see .env.example).
# The script also reads .env in the repository root, or the file in
# AIGW_ENV_FILE. A variable that is already set in the environment wins.

set -euo pipefail

ROOT=$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)
PREFIX=${AIGW_PREFIX:-$ROOT/gateway}
OPENRESTY=${OPENRESTY:-openresty}

die() { echo "gateway.sh: $*" >&2; exit 1; }

load_env_file() {
  local file=$1 line key value
  [[ -f $file ]] || return 0
  while IFS= read -r line || [[ -n $line ]]; do
    [[ $line =~ ^[[:space:]]*([A-Z_][A-Z0-9_]*)=(.*)$ ]] || continue
    key=${BASH_REMATCH[1]}
    value=${BASH_REMATCH[2]}
    if [[ -z ${!key+x} ]]; then
      export "$key=$value"
    fi
  done <"$file"
}

find_core_lib() {
  local candidate
  for candidate in \
    "$ROOT/core/target/release/libaigw_core.dylib" \
    "$ROOT/core/target/release/libaigw_core.so" \
    "$PREFIX/lib/libaigw_core.so"; do
    if [[ -f $candidate ]]; then
      echo "$candidate"
      return 0
    fi
  done
  return 1
}

# First IPv4 nameserver of this host. NGINX needs it for proxy_pass with
# variables. Set AIGW_RESOLVER to override it.
find_resolver() {
  if [[ -n ${AIGW_RESOLVER:-} ]]; then
    echo "$AIGW_RESOLVER"
    return 0
  fi
  awk '$1 == "nameserver" && $2 ~ /^[0-9.]+$/ { print $2; exit }' /etc/resolv.conf 2>/dev/null
}

find_ca_bundle() {
  local candidate
  for candidate in \
    "${AIGW_CA_BUNDLE:-}" \
    /etc/ssl/certs/ca-certificates.crt \
    /etc/pki/tls/certs/ca-bundle.crt \
    /etc/ssl/cert.pem \
    /opt/homebrew/etc/ca-certificates/cert.pem; do
    if [[ -n $candidate && -f $candidate ]]; then
      echo "$candidate"
      return 0
    fi
  done
  return 1
}

write_generated_conf() {
  local resolver ca
  resolver=$(find_resolver) || true
  [[ -n $resolver ]] || die "no IPv4 nameserver in /etc/resolv.conf. Set AIGW_RESOLVER."
  ca=$(find_ca_bundle) || die "no CA bundle found. Set AIGW_CA_BUNDLE."
  cat >"$PREFIX/conf/generated.conf" <<EOF
# Written by scripts/gateway.sh at startup. Do not edit.
resolver $resolver ipv6=off valid=30s;
resolver_timeout 5s;
proxy_ssl_trusted_certificate $ca;
EOF
}

prepare() {
  load_env_file "${AIGW_ENV_FILE:-$ROOT/.env}"
  if [[ -z ${AIGW_CORE_LIB:-} ]]; then
    AIGW_CORE_LIB=$(find_core_lib) || die "libaigw_core not found. Run 'make core' or set AIGW_CORE_LIB."
    export AIGW_CORE_LIB
  fi
  command -v "$OPENRESTY" >/dev/null || die "'$OPENRESTY' is not on PATH. Install OpenResty or set OPENRESTY."
  mkdir -p "$PREFIX/logs"
  write_generated_conf
}

nginx() {
  "$OPENRESTY" -p "$PREFIX/" -c conf/nginx.conf "$@"
}

# Stop through the PID file, so that stop works even when the configuration
# or the environment is broken. Wait until the master process is gone.
stop() {
  local pidfile=$PREFIX/logs/nginx.pid pid i
  [[ -f $pidfile ]] || return 0
  pid=$(<"$pidfile")
  kill -TERM "$pid" 2>/dev/null || return 0
  for i in $(seq 1 50); do
    kill -0 "$pid" 2>/dev/null || return 0
    sleep 0.1
  done
  die "process $pid did not stop"
}

cmd=${1:-start}
case $cmd in
  start)  prepare; exec "$OPENRESTY" -p "$PREFIX/" -c conf/nginx.conf -g "daemon off;" ;;
  daemon) prepare; nginx ;;
  test)   prepare; nginx -t ;;
  reload) prepare; nginx -s reload ;;
  stop)   stop ;;
  *)      die "unknown command '$cmd'. Use start, daemon, stop, reload or test." ;;
esac
