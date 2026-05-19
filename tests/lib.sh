#!/usr/bin/env bash
# Shared helpers for the test suite. Sourced by each test_*.sh.
#
# Exit-code convention used by run_tests.sh:
#   0   pass
#   77  skip  (e.g. model not found)
#   any other non-zero  fail

# Colours when stdout is a terminal.
if [ -t 1 ]; then
    RED='\033[0;31m'
    GREEN='\033[0;32m'
    YELLOW='\033[0;33m'
    BLUE='\033[0;34m'
    BOLD='\033[1m'
    NC='\033[0m'
else
    RED='' GREEN='' YELLOW='' BLUE='' BOLD='' NC=''
fi

# Paths computed from this file's own location so they survive being sourced.
TESTS_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$TESTS_DIR/.." && pwd)"
BITMAMBA_BIN="$REPO_ROOT/build/bitmamba"

# ── Logging ────────────────────────────────────────────────────────────────
pass() { echo -e "  ${GREEN}[PASS]${NC} $*"; }
fail() { echo -e "  ${RED}[FAIL]${NC} $*"; }
info() { echo -e "  ${BLUE}[INFO]${NC} $*"; }
warn() { echo -e "  ${YELLOW}[WARN]${NC} $*"; }

# Skip the current test file with a reason. Uses exit 77 so the driver
# counts it as skipped rather than failed.
skip_test() {
    echo -e "  ${YELLOW}[SKIP]${NC} $*"
    exit 77
}

# ── Assertions ─────────────────────────────────────────────────────────────
# Each assertion prints a PASS/FAIL line and increments TEST_FAILS on failure.
TEST_FAILS=0

assert_contains() {
    local haystack="$1" needle="$2" msg="${3:-output contains '$2'}"
    if echo "$haystack" | grep -qF -- "$needle"; then
        pass "$msg"
    else
        fail "$msg (missing '$needle')"
        TEST_FAILS=$((TEST_FAILS+1))
    fi
}

assert_not_contains() {
    local haystack="$1" needle="$2" msg="${3:-output does not contain '$2'}"
    if echo "$haystack" | grep -qF -- "$needle"; then
        fail "$msg (unexpectedly contains '$needle')"
        TEST_FAILS=$((TEST_FAILS+1))
    else
        pass "$msg"
    fi
}

assert_eq() {
    local a="$1" b="$2" msg="${3:-values are equal}"
    if [ "$a" = "$b" ]; then
        pass "$msg ('$a' == '$b')"
    else
        fail "$msg (got '$a', expected '$b')"
        TEST_FAILS=$((TEST_FAILS+1))
    fi
}

assert_exit_zero() {
    local rc="$1" msg="${2:-exit code is 0}"
    if [ "$rc" -eq 0 ]; then
        pass "$msg"
    else
        fail "$msg (got rc=$rc)"
        TEST_FAILS=$((TEST_FAILS+1))
    fi
}

assert_file_exists() {
    local path="$1" msg="${2:-file exists: $1}"
    if [ -f "$path" ]; then
        pass "$msg"
    else
        fail "$msg (not found)"
        TEST_FAILS=$((TEST_FAILS+1))
    fi
}

# Final hook for tests to call at the end. Exits non-zero if any assertion failed.
finalize() {
    if [ "$TEST_FAILS" -gt 0 ]; then
        exit 1
    fi
    exit 0
}

# ── Model / tokenizer discovery ────────────────────────────────────────────
# Returns 0 and echoes the path of a usable .bin model, or returns 1.
# Priority:
#   1. $BITMAMBA_TEST_MODEL env var (explicit override)
#   2. bitmamba_1b.bin / bitmamba_255m.bin in repo root, then up to 3 parents.
find_test_model() {
    if [ -n "${BITMAMBA_TEST_MODEL:-}" ]; then
        if [ -f "$BITMAMBA_TEST_MODEL" ]; then
            echo "$BITMAMBA_TEST_MODEL"
            return 0
        fi
        return 1
    fi
    local dirs=("$REPO_ROOT" "$REPO_ROOT/.." "$REPO_ROOT/../.." "$REPO_ROOT/../../..")
    for d in "${dirs[@]}"; do
        for f in bitmamba_1b.bin bitmamba_255m.bin; do
            if [ -f "$d/$f" ]; then
                echo "$(cd "$d" && pwd -P)/$f"
                return 0
            fi
        done
    done
    return 1
}

# Returns 0 and echoes the path of a usable .lora.bin, or returns 1.
# Priority: $BITMAMBA_TEST_LORA env var, then internal-docs/bitmamba_2_1b_lora.bin.
find_test_lora() {
    if [ -n "${BITMAMBA_TEST_LORA:-}" ]; then
        [ -f "$BITMAMBA_TEST_LORA" ] && { echo "$BITMAMBA_TEST_LORA"; return 0; }
        return 1
    fi
    local default="$REPO_ROOT/internal-docs/bitmamba_2_1b_lora.bin"
    [ -f "$default" ] && { echo "$default"; return 0; }
    return 1
}

# Ensure the binary exists; if not, the test should be skipped by the caller.
require_binary() {
    if [ ! -x "$BITMAMBA_BIN" ]; then
        skip_test "binary not built — run tests/run_tests.sh which builds first, or 'cmake --build build -j' manually"
    fi
}

# Ensure tokenizer.bin is next to the binary (the engine looks for it there).
ensure_tokenizer_next_to_bin() {
    local tk_src="$REPO_ROOT/tokenizer.bin"
    local tk_dst="$REPO_ROOT/build/tokenizer.bin"
    if [ ! -f "$tk_dst" ] && [ -f "$tk_src" ]; then
        cp "$tk_src" "$tk_dst"
    fi
}
