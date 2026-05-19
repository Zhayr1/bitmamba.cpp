#!/usr/bin/env bash
# LoRA runtime: --lora <path.lora.bin> loads the adapter, validates shapes,
# and produces output. Compatible with both batched (default) and sequential
# prefill.

set -uo pipefail
source "$(dirname "${BASH_SOURCE[0]}")/lib.sh"
require_binary
ensure_tokenizer_next_to_bin

MODEL="$(find_test_model)" || skip_test "no test model. Set BITMAMBA_TEST_MODEL=<path>"
LORA="$(find_test_lora)"   || skip_test "no .lora.bin found. Set BITMAMBA_TEST_LORA=<path>, or place one at internal-docs/bitmamba_2_1b_lora.bin"
info "using model: $MODEL"
info "using lora:  $LORA"

# The 1B-trained LoRA is incompatible with a 255M model. If the model is the
# smaller one, skip — the user can re-run with a matching adapter.
case "$MODEL" in
    *255m*) skip_test "test LoRA was trained for the 1B model; skip when running against 255M" ;;
esac

OUT="$("$BITMAMBA_BIN" --lora "$LORA" "$MODEL" "Hello, I am" tokenizer 0.0 1.0 0.0 1.0 0 5 2>&1 || true)"
assert_contains "$OUT" "[LoRA] Loaded"              "LoRA file is loaded"
assert_contains "$OUT" "[LoRA] Shape check passed"  "LoRA shapes match base model"
assert_contains "$OUT" "Generated"                  "LoRA-enabled inference produces output"

# Sequential prefill + LoRA must produce the same tokens as default + LoRA.
OUT_DEFAULT="$("$BITMAMBA_BIN" --lora "$LORA"                    "$MODEL" "Hello, I am" tokenizer 0.0 1.0 0.0 1.0 0 5 clean 2>/dev/null | tail -1)"
OUT_SEQ="$("$BITMAMBA_BIN" --lora "$LORA" --sequential-prefill  "$MODEL" "Hello, I am" tokenizer 0.0 1.0 0.0 1.0 0 5 clean 2>/dev/null | tail -1)"

if [ "$OUT_DEFAULT" = "$OUT_SEQ" ] && [ -n "$OUT_DEFAULT" ]; then
    pass "LoRA outputs are identical between batched and sequential prefill"
else
    fail "LoRA outputs differ between prefill modes ('$OUT_DEFAULT' vs '$OUT_SEQ')"
    TEST_FAILS=$((TEST_FAILS+1))
fi

# Corrupt LoRA file should be rejected with a clear error.
ERR="$("$BITMAMBA_BIN" --lora "$REPO_ROOT/README.md" "$MODEL" "Hi" tokenizer 0.0 1.0 0.0 1.0 0 1 2>&1 || true)"
assert_contains "$ERR" "LoRA" "invalid .lora.bin produces a LoRA-prefixed error message"

finalize
