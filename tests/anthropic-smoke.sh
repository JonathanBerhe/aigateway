#!/usr/bin/env bash
# End-to-end check of the format conversion: the strong tier speaks the
# Anthropic Messages API (the mock's /strong/anthropic/v1/messages), the
# cheap tier stays OpenAI. Starts and stops the mock and the gateway.

set -uo pipefail

ROOT=$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)
OPENRESTY=${OPENRESTY:-openresty}
MOCK=$ROOT/tests/mock
URL=http://127.0.0.1:8000
TMP=$(mktemp -d)

export AIGW_ENV_FILE=/dev/null
export AIGW_API_KEYS=test-key
export AIGW_CHEAP_BASE_URL=http://127.0.0.1:19000/cheap/v1 AIGW_CHEAP_MODEL=mock-cheap-model
export AIGW_STRONG_BASE_URL=http://127.0.0.1:19000/strong/anthropic/v1 AIGW_STRONG_MODEL=claude-mock
export AIGW_STRONG_FORMAT=anthropic AIGW_STRONG_API_KEY=mock-anthropic-key
unset AIGW_CHEAP_API_KEY AIGW_CACHE_ADDR AIGW_GUARD AIGW_ROUTER AIGW_ROUTER_SHADOW AIGW_STRONG_MAX_TOKENS

stop_all() {
  "$ROOT/scripts/gateway.sh" stop || true
  AIGW_PREFIX=$MOCK "$ROOT/scripts/gateway.sh" stop || true
}
trap 'stop_all; rm -rf "$TMP"' EXIT

PASS=0 FAIL=0
expect() {
  if [[ $2 == "$3" ]]; then PASS=$((PASS + 1)); printf '  ok    %s\n' "$1"
  else FAIL=$((FAIL + 1)); printf '  FAIL  %s\n        expected %s, got %s\n' "$1" "$3" "$2"; fi
}

# post BODY [header]: prints the status; body in $TMP/body, headers in $TMP/headers.
post() {
  curl -sS -o "$TMP/body" -D "$TMP/headers" -w '%{http_code}' -H "Authorization: Bearer test-key" \
    -H 'Content-Type: application/json' ${2:+-H "$2"} --data "$1" "$URL/v1/chat/completions"
}
header() { awk -v name="$1:" 'tolower($0) ~ "^" tolower(name) { sub(/^[^:]*:[ \t]*/, ""); sub(/\r$/, ""); print; exit }' "$TMP/headers"; }
metric() { curl -sS http://127.0.0.1:8001/metrics | awk -v s="$1" '$1 == s { print $2 }'; }
# Text of an OpenAI SSE stream in $TMP/body.
stream_text() { sed -n 's/^data: //p' "$TMP/body" | grep -v '^\[DONE\]' | jq -rj '.choices[0].delta.content // empty'; }

stop_all 2>/dev/null
mkdir -p "$MOCK/logs"
"$OPENRESTY" -p "$MOCK/" -c conf/nginx.conf
for _ in $(seq 1 50); do curl -s -o /dev/null http://127.0.0.1:19000/ && break; sleep 0.1; done
"$ROOT/scripts/gateway.sh" daemon
for _ in $(seq 1 50); do curl -s -o /dev/null http://127.0.0.1:8001/healthz && break; sleep 0.1; done

REPLY="This is a mock reply from the strong tier."
SYS='{"role":"system","content":"Be terse."}'

echo "format conversion smoke test"
before=$(metric 'aigw_tokens_total{kind="prompt",source="upstream",tier="strong"}')
# A body in a variable: bash 3.2 (macOS) splits escaped quotes inside $( ).
WITH_SYSTEM='{"model":"aigw-strong","messages":['"$SYS"',{"role":"user","content":"hi"}]}'
expect "JSON: status 200" "$(post "$WITH_SYSTEM")" 200
expect "the request reached the Messages API" "$(header X-Mock-Format)" anthropic
expect "the upstream key goes in X-Api-Key" "$(header X-Mock-Api-Key)" mock-anthropic-key
expect "no Authorization header goes upstream" "$(header X-Mock-Authorization)" none
expect "the API version header is set" "$(header X-Mock-Anthropic-Version)" 2023-06-01
expect "the system message is the system field" "$(header X-Mock-System)" "Be terse."
expect "only the user message stays in messages" "$(header X-Mock-Roles)" user
expect "max_tokens gets the tier default" "$(header X-Mock-Max-Tokens)" 1024
expect "the answer is a chat completion" "$(jq -r .object "$TMP/body")" chat.completion
expect "the answer has the reply text" "$(jq -r '.choices[0].message.content' "$TMP/body")" "$REPLY"
expect "end_turn maps to stop" "$(jq -r '.choices[0].finish_reason' "$TMP/body")" stop
expect "usage maps to prompt_tokens" "$(jq -r .usage.prompt_tokens "$TMP/body")" 42
after=$(metric 'aigw_tokens_total{kind="prompt",source="upstream",tier="strong"}')
expect "the tokens count as upstream usage" "$(( ${after:-0} - ${before:-0} ))" 42

post '{"model":"aigw-strong","max_tokens":50,"stop":["END","STOP"],"messages":[{"role":"user","content":"hi"}]}' >/dev/null
expect "max_tokens of the client goes upstream" "$(header X-Mock-Max-Tokens)" 50
expect "stop becomes stop_sequences" "$(header X-Mock-Stop)" END,STOP

STREAM='{"model":"aigw-strong","stream":true,"messages":[{"role":"user","content":"hi"}]}'
expect "stream: status 200" "$(post "$STREAM")" 200
expect "stream: the text adds up to the reply" "$(stream_text)" "$REPLY"
expect "stream: the chunks are chat.completion.chunk" \
  "$(sed -n 's/^data: //p' "$TMP/body" | head -1 | jq -r .object)" chat.completion.chunk
expect "stream: a usage chunk comes before [DONE]" "$(grep -c '"prompt_tokens":42' "$TMP/body")" 1
expect "stream: it ends with [DONE]" "$(tail -c 14 "$TMP/body" | grep -c '\[DONE\]')" 1
post "$STREAM" "X-Mock-Split: true" >/dev/null
expect "split stream: events that span chunks still translate" "$(stream_text)" "$REPLY"

expect "an upstream error keeps its status" \
  "$(post '{"model":"aigw-strong","messages":[{"role":"user","content":"hi"}]}' 'X-Mock-Status: 529')" 529
expect "the error body is in the OpenAI format" "$(jq -r .error.type "$TMP/body")" overloaded_error

expect "tools are refused before the upstream call" \
  "$(post '{"model":"aigw-strong","tools":[{"type":"function","function":{"name":"f"}}],"messages":[{"role":"user","content":"hi"}]}')" 400
expect "the refusal has its own error code" "$(jq -r .error.code "$TMP/body")" unsupported_for_model

expect "the OpenAI tier still works" "$(post '{"model":"aigw-cheap","messages":[{"role":"user","content":"hi"}]}')" 200
expect "the OpenAI tier is not translated" "$(header X-Mock-Format)" ""

echo; echo "passed: $PASS, failed: $FAIL"
[[ $FAIL -eq 0 ]]
