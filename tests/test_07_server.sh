#!/usr/bin/env bash
# simple_server.py (uses only Python stdlib http.server) must respond to
# /health, /, and /generate. /generate spawns the C++ binary internally,
# so we configure BITMAMBA_BINARY and BITMAMBA_MODEL_PATH env vars before
# starting the server.

set -uo pipefail
source "$(dirname "${BASH_SOURCE[0]}")/lib.sh"

if ! command -v python3 >/dev/null; then
    skip_test "python3 not available"
fi
if ! command -v curl >/dev/null; then
    skip_test "curl not available"
fi

# The /generate endpoint shells out to the C++ binary, so a model is required.
MODEL="$(find_test_model)" || skip_test "no test model. Set BITMAMBA_TEST_MODEL=<path>"
require_binary
ensure_tokenizer_next_to_bin
info "using model:  $MODEL"
info "using binary: $BITMAMBA_BIN"

# Random high port to avoid collisions with other test runs or manual servers.
# $RANDOM ∈ [0, 32767], so PORT ∈ [20000, 52767].
PORT=$((20000 + RANDOM % 30000))
HOST=127.0.0.1
info "using port $PORT"

info "starting simple_server on $HOST:$PORT"
BITMAMBA_BINARY="$BITMAMBA_BIN" \
BITMAMBA_MODEL_PATH="$MODEL" \
python3 "$REPO_ROOT/python/simple_server.py" --host "$HOST" --port "$PORT" \
    >/tmp/bm_simple_server.log 2>&1 &
SERVER_PID=$!

# Ensure the server is killed on exit whether the test passes or fails.
trap 'kill "$SERVER_PID" 2>/dev/null; wait "$SERVER_PID" 2>/dev/null; true' EXIT

# Wait up to 5s for the port to become reachable.
ready=0
for _ in 1 2 3 4 5 6 7 8 9 10; do
    if curl -s -m 1 "http://$HOST:$PORT/health" >/dev/null 2>&1; then
        ready=1; break
    fi
    sleep 0.5
done

if [ "$ready" -ne 1 ]; then
    fail "simple_server did not become reachable within 5s"
    echo "─── server log ───"
    tail -20 /tmp/bm_simple_server.log
    TEST_FAILS=$((TEST_FAILS+1))
    finalize
fi
pass "simple_server is reachable on $HOST:$PORT"

HEALTH="$(curl -s -m 3 "http://$HOST:$PORT/health")"
assert_contains "$HEALTH" "healthy" "/health responds with healthy status"

ROOT="$(curl -s -m 3 "http://$HOST:$PORT/")"
assert_contains "$ROOT" "endpoints" "/ responds with an endpoints listing"

# /generate end-to-end. The simple_server invokes the C++ binary as a subprocess,
# so this also validates the env-var-configured paths. Short max_tokens to keep
# it fast; the timeout has headroom for cold-cache prefill of the prompt.
info "POST /generate (this will run the model — expect ~5-30s)"
GEN_BODY='{"prompt":"Hello","max_tokens":3,"temperature":0.0}'
GEN="$(curl -s -m 120 -X POST -H 'Content-Type: application/json' \
        --data "$GEN_BODY" "http://$HOST:$PORT/generate")"

assert_contains "$GEN" "response" "/generate returns a response field"

# Pull the text out of the JSON and make sure it's non-empty.
GEN_TEXT="$(echo "$GEN" | python3 -c \
    'import sys, json; d=json.load(sys.stdin); print(d.get("response",""))' 2>/dev/null || true)"
if [ -n "$GEN_TEXT" ]; then
    pass "/generate returns non-empty text: \"$(echo "$GEN_TEXT" | head -c 60)\""
else
    fail "/generate returned empty response text. Raw body: $(echo "$GEN" | head -c 200)"
    TEST_FAILS=$((TEST_FAILS+1))
fi

finalize
