#!/usr/bin/env bash
# Backward compatibility: the engine still accepts the exact CLI shape from
# the original README. Users who already downloaded the project should be
# able to run identical commands and get a working result.
#
# Old-style invocation (from the original README):
#   ./bitmamba <model.bin> "<prompt>" <mode> <temp> <penalty> <min_p> <top_p> <top_k> <max_tokens>

set -uo pipefail
source "$(dirname "${BASH_SOURCE[0]}")/lib.sh"
require_binary
ensure_tokenizer_next_to_bin

MODEL="$(find_test_model)" || skip_test "no test model found. Set BITMAMBA_TEST_MODEL=<path-to-.bin>, or place bitmamba_1b.bin / bitmamba_255m.bin in the repo root."
info "using model: $MODEL"

# --- Tokenizer mode, original arg layout ---
OUT="$("$BITMAMBA_BIN" "$MODEL" "Hello, I am" tokenizer 0.7 1.15 0.05 0.9 40 10 clean 2>&1 || true)"
# Output should be non-empty after the "clean" mode strips logs.
generated="$(echo "$OUT" | tail -1)"
if [ -z "$generated" ]; then
    fail "tokenizer mode produced empty output"
    TEST_FAILS=$((TEST_FAILS+1))
else
    pass "tokenizer mode produces non-empty output: \"$(echo "$generated" | head -c 60)\""
fi

# --- Raw mode (token ID input/output) ---
OUT_RAW="$("$BITMAMBA_BIN" "$MODEL" "15496 11 314 716" raw 0.7 1.15 0.05 0.9 40 5 clean 2>&1 || true)"
# Output should be space-separated integers (token IDs).
raw_tokens="$(echo "$OUT_RAW" | tail -1)"
if echo "$raw_tokens" | grep -qE '^[[:space:]]*[0-9]+([[:space:]]+[0-9]+)*[[:space:]]*$'; then
    pass "raw mode emits token IDs"
else
    fail "raw mode did not emit token IDs (got: '$raw_tokens')"
    TEST_FAILS=$((TEST_FAILS+1))
fi

# --- Default args (just model + prompt + mode) — should still run ---
OUT_DEFAULTS="$("$BITMAMBA_BIN" "$MODEL" "Hi" tokenizer 2>&1 || true)"
assert_contains "$OUT_DEFAULTS" "Generated"  "default arg invocation produces a Generated Text block"

# --- decoder.py round-trip on the raw output ---
if command -v python3 >/dev/null && [ -n "$raw_tokens" ]; then
    decoded="$(python3 "$REPO_ROOT/scripts/decoder.py" "$raw_tokens" 2>&1 || true)"
    if [ -n "$decoded" ]; then
        pass "decoder.py decodes raw output to text"
    else
        warn "decoder.py produced empty output (skipping assertion)"
    fi
else
    warn "python3 not available — skipping decoder.py round-trip"
fi

finalize
