#!/usr/bin/env bash
# Driver for the bitmamba.cpp test suite.
# Discovers test_*.sh files in this directory and runs each in order.
# Reports pass/skip/fail counts and exits non-zero if any test failed.
#
# Usage:
#   tests/run_tests.sh                  # run all
#   tests/run_tests.sh test_03_rys      # run a single test (omit .sh)
#
# Environment:
#   BITMAMBA_TEST_MODEL  Optional. Absolute path to a .bin model file.
#                        If unset, tests look for bitmamba_1b.bin or
#                        bitmamba_255m.bin in the repo root and parent dirs.
#   BITMAMBA_TEST_LORA   Optional. Absolute path to a .lora.bin adapter.

set -uo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"
cd "$REPO_ROOT"

# Colours when terminal.
if [ -t 1 ]; then
    RED='\033[0;31m' GREEN='\033[0;32m' YELLOW='\033[0;33m'
    BOLD='\033[1m' NC='\033[0m'
else
    RED='' GREEN='' YELLOW='' BOLD='' NC=''
fi

# Build list of tests.
if [ "$#" -gt 0 ]; then
    targets=()
    for t in "$@"; do
        f="$SCRIPT_DIR/${t%.sh}.sh"
        if [ ! -f "$f" ]; then
            echo "ERROR: test file not found: $f" >&2
            exit 2
        fi
        targets+=("$f")
    done
else
    shopt -s nullglob
    targets=( "$SCRIPT_DIR"/test_*.sh )
fi

if [ "${#targets[@]}" -eq 0 ]; then
    echo "No tests found in $SCRIPT_DIR"
    exit 0
fi

PASSED=0
FAILED=0
SKIPPED=0
FAILED_NAMES=()
SKIPPED_NAMES=()

echo -e "${BOLD}Running ${#targets[@]} test file(s) from $SCRIPT_DIR${NC}"
echo

for tf in "${targets[@]}"; do
    name="$(basename "$tf" .sh)"
    echo -e "${BOLD}─── $name ───${NC}"
    bash "$tf"
    rc=$?
    case $rc in
        0)  PASSED=$((PASSED+1)) ;;
        77) SKIPPED=$((SKIPPED+1)); SKIPPED_NAMES+=("$name") ;;
        *)  FAILED=$((FAILED+1));  FAILED_NAMES+=("$name (rc=$rc)") ;;
    esac
    echo
done

# Summary
echo -e "${BOLD}═══════════════ Summary ═══════════════${NC}"
echo -e "  Passed:  ${GREEN}$PASSED${NC}"
echo -e "  Skipped: ${YELLOW}$SKIPPED${NC}"
echo -e "  Failed:  ${RED}$FAILED${NC}"

if [ "$SKIPPED" -gt 0 ]; then
    echo
    echo -e "${YELLOW}Skipped tests:${NC}"
    for n in "${SKIPPED_NAMES[@]}"; do echo "    - $n"; done
fi

if [ "$FAILED" -gt 0 ]; then
    echo
    echo -e "${RED}Failed tests:${NC}"
    for n in "${FAILED_NAMES[@]}"; do echo "    - $n"; done
    exit 1
fi

exit 0
