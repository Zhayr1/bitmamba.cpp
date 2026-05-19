#!/usr/bin/env bash
# RYS (LLM Neuroanatomy): --repeat-start / --repeat-end execute the chosen
# layer slice extra times. Verify the engine logs the expanded execution
# path and that inference still produces output.

set -uo pipefail
source "$(dirname "${BASH_SOURCE[0]}")/lib.sh"
require_binary
ensure_tokenizer_next_to_bin

MODEL="$(find_test_model)" || skip_test "no test model. Set BITMAMBA_TEST_MODEL=<path>"
info "using model: $MODEL"

# RYS-enabled run: repeat layers 5..8 (small slice that should work on both 1B and 255M).
OUT="$("$BITMAMBA_BIN" --repeat-start 5 --repeat-end 8 "$MODEL" "Hello" tokenizer 0.0 1.0 0.0 1.0 0 3 2>&1 || true)"
assert_contains "$OUT" "[RYS]"                  "[RYS] log line is printed when flags are set"
assert_contains "$OUT" "Execution path size"   "RYS log reports the expanded execution path size"
assert_contains "$OUT" "Generated"             "RYS run still produces a Generated Text block"

# Without RYS flags, no [RYS] log line should appear.
OUT_NO_RYS="$("$BITMAMBA_BIN" "$MODEL" "Hello" tokenizer 0.0 1.0 0.0 1.0 0 3 2>&1 || true)"
assert_not_contains "$OUT_NO_RYS" "[RYS]" "no [RYS] log without the flag"

# Argument validation: end < start should be rejected.
ERR="$("$BITMAMBA_BIN" --repeat-start 10 --repeat-end 5 "$MODEL" "Hi" tokenizer 0.0 1.0 0.0 1.0 0 1 2>&1 || true)"
assert_contains "$ERR" "Error" "invalid --repeat-start/end is rejected with an Error message"

finalize
