#!/usr/bin/env bash
# Syntax-check every Python script we ship in scripts/. This catches typos
# and unresolved imports without needing the heavy ML deps installed.

set -uo pipefail
source "$(dirname "${BASH_SOURCE[0]}")/lib.sh"

if ! command -v python3 >/dev/null; then
    skip_test "python3 not available"
fi

shopt -s nullglob
scripts=( "$REPO_ROOT/scripts"/*.py )

if [ "${#scripts[@]}" -eq 0 ]; then
    fail "no Python scripts found under scripts/"
    TEST_FAILS=$((TEST_FAILS+1))
fi

for s in "${scripts[@]}"; do
    name="$(basename "$s")"
    if python3 -m py_compile "$s" 2>/tmp/bm_pyc_err; then
        pass "scripts/$name compiles"
    else
        fail "scripts/$name has a syntax error"
        cat /tmp/bm_pyc_err
        TEST_FAILS=$((TEST_FAILS+1))
    fi
done

# Also syntax-check the FastAPI server files (no fastapi install needed; py_compile
# only parses the AST).
for s in "$REPO_ROOT"/python/*.py; do
    [ -f "$s" ] || continue
    name="$(basename "$s")"
    if python3 -m py_compile "$s" 2>/tmp/bm_pyc_err; then
        pass "python/$name compiles"
    else
        fail "python/$name has a syntax error"
        cat /tmp/bm_pyc_err
        TEST_FAILS=$((TEST_FAILS+1))
    fi
done

finalize
