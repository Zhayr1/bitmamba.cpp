#!/usr/bin/env bash
# Verify the usage/help text mentions both legacy positional args and the
# new optional flags introduced by RYS / LoRA / batched-prefill work.
# Running the binary with no args prints the help and exits non-zero.

set -uo pipefail
source "$(dirname "${BASH_SOURCE[0]}")/lib.sh"
require_binary

# Capture help by running the binary with no positional args.
HELP="$("$BITMAMBA_BIN" 2>&1 || true)"

# Legacy / always-present arguments
assert_contains "$HELP" "<model.bin>"  "help mentions <model.bin>"
assert_contains "$HELP" "tokenizer"    "help mentions tokenizer mode"
assert_contains "$HELP" "raw"          "help mentions raw mode"
assert_contains "$HELP" "temp"         "help mentions temp param"
assert_contains "$HELP" "max_tokens"   "help mentions max_tokens param"

# New flags from the rys-and-api-server work
assert_contains "$HELP" "--repeat-start"       "help mentions --repeat-start"
assert_contains "$HELP" "--repeat-end"         "help mentions --repeat-end"
assert_contains "$HELP" "--repeat-count"       "help mentions --repeat-count"
assert_contains "$HELP" "--chat"               "help mentions --chat"
assert_contains "$HELP" "--lora"               "help mentions --lora"
assert_contains "$HELP" "--sequential-prefill" "help mentions --sequential-prefill"

finalize
