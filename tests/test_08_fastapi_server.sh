#!/usr/bin/env bash
# FastAPI server (python/server.py) — OpenAI-compatible endpoints:
#   /v1/models, /v1/completions, /v1/chat/completions
# Auto-skips if FastAPI / uvicorn aren't importable, so users with only the
# stdlib server (test_07) still get a clean run.

set -uo pipefail
source "$(dirname "${BASH_SOURCE[0]}")/lib.sh"

if ! command -v python3 >/dev/null; then skip_test "python3 not available"; fi
if ! command -v curl    >/dev/null; then skip_test "curl not available";    fi

# FastAPI + uvicorn aren't in the stdlib — auto-skip if not installed.
if ! python3 -c "import fastapi, uvicorn, pydantic" 2>/dev/null; then
    skip_test "FastAPI/uvicorn not installed. Run: pip install -r python/requirements.txt"
fi

MODEL="$(find_test_model)" || skip_test "no test model. Set BITMAMBA_TEST_MODEL=<path>"
require_binary
ensure_tokenizer_next_to_bin
info "using model:  $MODEL"
info "using binary: $BITMAMBA_BIN"

PORT=$((20000 + RANDOM % 30000))
HOST=127.0.0.1
info "using port $PORT"

# server.py reads --model from CLI and the binary path from BITMAMBA_BINARY env.
info "starting FastAPI server on $HOST:$PORT"
BITMAMBA_BINARY="$BITMAMBA_BIN" \
python3 "$REPO_ROOT/python/server.py" --model "$MODEL" --host "$HOST" --port "$PORT" \
    >/tmp/bm_fastapi_server.log 2>&1 &
SERVER_PID=$!

# Make sure we kill the server no matter how the test ends.
trap 'kill "$SERVER_PID" 2>/dev/null; wait "$SERVER_PID" 2>/dev/null; true' EXIT

# Wait up to 10s for readiness (FastAPI startup is heavier than http.server).
ready=0
for _ in 1 2 3 4 5 6 7 8 9 10 11 12 13 14 15 16 17 18 19 20; do
    if curl -s -m 1 "http://$HOST:$PORT/health" >/dev/null 2>&1; then
        ready=1; break
    fi
    sleep 0.5
done

if [ "$ready" -ne 1 ]; then
    fail "FastAPI server did not become reachable within 10s"
    echo "─── server log ───"; tail -30 /tmp/bm_fastapi_server.log
    TEST_FAILS=$((TEST_FAILS+1))
    finalize
fi
pass "FastAPI server is reachable on $HOST:$PORT"

# ── Discovery endpoint ─────────────────────────────────────────────────────
MODELS="$(curl -s -m 3 "http://$HOST:$PORT/v1/models")"
assert_contains "$MODELS" "bitmamba" "/v1/models lists the bitmamba model"

# ── /v1/completions (OpenAI text completion shape) ─────────────────────────
info "POST /v1/completions (this will run the model — expect ~5-30s)"
COMP_BODY='{"model":"bitmamba","prompt":"Hello","max_tokens":3,"temperature":0.0}'
COMP="$(curl -s -m 120 -X POST -H 'Content-Type: application/json' \
        --data "$COMP_BODY" "http://$HOST:$PORT/v1/completions")"
assert_contains "$COMP" "choices"   "/v1/completions returns an OpenAI-shaped response with 'choices'"
assert_contains "$COMP" "text"      "/v1/completions response contains 'text' field"

# Extract the generated text and verify it's non-empty.
COMP_TEXT="$(echo "$COMP" | python3 -c \
    'import sys, json; d=json.load(sys.stdin); print(d["choices"][0].get("text",""))' \
    2>/dev/null || true)"
if [ -n "$COMP_TEXT" ]; then
    pass "/v1/completions text is non-empty: \"$(echo "$COMP_TEXT" | head -c 60)\""
else
    fail "/v1/completions returned empty text. Raw: $(echo "$COMP" | head -c 200)"
    TEST_FAILS=$((TEST_FAILS+1))
fi

# ── /v1/chat/completions ───────────────────────────────────────────────────
info "POST /v1/chat/completions (this will run the model — expect ~5-30s)"
CHAT_BODY='{"model":"bitmamba","messages":[{"role":"user","content":"Hi"}],"max_tokens":3,"temperature":0.0}'
CHAT="$(curl -s -m 120 -X POST -H 'Content-Type: application/json' \
        --data "$CHAT_BODY" "http://$HOST:$PORT/v1/chat/completions")"
assert_contains "$CHAT" "choices" "/v1/chat/completions returns 'choices'"
assert_contains "$CHAT" "message" "/v1/chat/completions response contains 'message' field"

CHAT_TEXT="$(echo "$CHAT" | python3 -c \
    'import sys, json; d=json.load(sys.stdin); print(d["choices"][0]["message"].get("content",""))' \
    2>/dev/null || true)"
if [ -n "$CHAT_TEXT" ]; then
    pass "/v1/chat/completions content is non-empty: \"$(echo "$CHAT_TEXT" | head -c 60)\""
else
    fail "/v1/chat/completions returned empty content. Raw: $(echo "$CHAT" | head -c 200)"
    TEST_FAILS=$((TEST_FAILS+1))
fi

finalize
