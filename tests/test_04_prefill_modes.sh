#!/usr/bin/env bash
# The batched prefill (default) must produce the SAME tokens as the legacy
# sequential prefill, under greedy decoding. This is the correctness gate
# for the layer-major batched optimization.

set -uo pipefail
source "$(dirname "${BASH_SOURCE[0]}")/lib.sh"
require_binary
ensure_tokenizer_next_to_bin

MODEL="$(find_test_model)" || skip_test "no test model. Set BITMAMBA_TEST_MODEL=<path>"
info "using model: $MODEL"

PROMPT="The capital of France is"

# Greedy decode (temp=0) makes outputs reproducible across runs.
OUT_DEFAULT="$("$BITMAMBA_BIN" "$MODEL" "$PROMPT" tokenizer 0.0 1.0 0.0 1.0 0 6 clean 2>/dev/null | tail -1)"
OUT_SEQ="$("$BITMAMBA_BIN" --sequential-prefill "$MODEL" "$PROMPT" tokenizer 0.0 1.0 0.0 1.0 0 6 clean 2>/dev/null | tail -1)"

info "default  (batched):    '$OUT_DEFAULT'"
info "sequential:            '$OUT_SEQ'"

if [ -z "$OUT_DEFAULT" ]; then
    fail "batched (default) prefill produced empty output"
    TEST_FAILS=$((TEST_FAILS+1))
elif [ -z "$OUT_SEQ" ]; then
    fail "sequential prefill produced empty output"
    TEST_FAILS=$((TEST_FAILS+1))
elif [ "$OUT_DEFAULT" = "$OUT_SEQ" ]; then
    pass "batched and sequential prefill produce identical greedy output"
else
    fail "outputs differ between batched and sequential prefill"
    TEST_FAILS=$((TEST_FAILS+1))
fi

finalize
