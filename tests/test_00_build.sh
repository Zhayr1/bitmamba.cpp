#!/usr/bin/env bash
# Verify cmake configure + build produces a working binary.
# This must run first — later tests depend on the binary existing.

set -uo pipefail
source "$(dirname "${BASH_SOURCE[0]}")/lib.sh"

cd "$REPO_ROOT"

info "cmake configure"
cmake -B build -DCMAKE_BUILD_TYPE=Release >/tmp/bm_cmake_cfg.log 2>&1
rc=$?
assert_exit_zero "$rc" "cmake configure exits 0"
[ "$rc" -ne 0 ] && { echo "─── cmake log ───"; tail -30 /tmp/bm_cmake_cfg.log; }

info "cmake build"
cmake --build build -j >/tmp/bm_cmake_build.log 2>&1
rc=$?
assert_exit_zero "$rc" "cmake --build exits 0"
[ "$rc" -ne 0 ] && { echo "─── build log (last 40) ───"; tail -40 /tmp/bm_cmake_build.log; }

assert_file_exists "$BITMAMBA_BIN" "binary built at build/bitmamba"

# Tokenizer must be next to the binary so subsequent tests can use tokenizer mode.
ensure_tokenizer_next_to_bin
assert_file_exists "$REPO_ROOT/build/tokenizer.bin" "tokenizer.bin available next to binary"

finalize
