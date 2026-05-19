#!/usr/bin/env bash
# simple_server.py (uses only Python stdlib http.server) must respond to
# /health and / without errors. We don't exercise /generate here because
# that would require the C++ binary plus a model file; the inference path
# is covered by test_02_legacy_cli.sh.

set -uo pipefail
source "$(dirname "${BASH_SOURCE[0]}")/lib.sh"

if ! command -v python3 >/dev/null; then
    skip_test "python3 not available"
fi
if ! command -v curl >/dev/null; then
    skip_test "curl not available"
fi

# Pick an unlikely-to-be-used local port. Avoid privileged range.
# Random high port to avoid collisions with other test runs or manual servers.
# $RANDOM ∈ [0, 32767], so PORT ∈ [20000, 52767].
PORT=$((20000 + RANDOM % 30000))
HOST=127.0.0.1
info "using port $PORT"

info "starting simple_server on $HOST:$PORT"
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

finalize
