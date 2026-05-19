# Test suite

Bash-based end-to-end tests for the C++ engine, scripts, and Python server.
No Python test framework dependencies — all assertions run through the shell.

## Run all tests

```bash
tests/run_tests.sh
```

The driver discovers every `test_*.sh` in `tests/`, runs each in order, and
prints a pass / skip / fail summary. Exit code is non-zero if any test fails.

## Run a single test

```bash
tests/run_tests.sh test_04_prefill_modes
```

(Omit the `.sh` extension.)

## Model & adapter discovery

Tests that need a model look for one in this order:

1. `$BITMAMBA_TEST_MODEL` (absolute path to a `.bin`)
2. `bitmamba_1b.bin` or `bitmamba_255m.bin` in the repo root or up to 3 parent directories

If none is found the test prints a clear `[SKIP]` message — it does not fail.

For LoRA tests:

1. `$BITMAMBA_TEST_LORA` (absolute path to a `.lora.bin`)
2. `internal-docs/bitmamba_2_1b_lora.bin`

The 1B LoRA adapter is incompatible with the 255M model; the LoRA test
auto-skips in that case.

## What each test covers

| File | Purpose |
|---|---|
| `test_00_build.sh`         | `cmake` configure + build produces `build/bitmamba`; tokenizer copied alongside |
| `test_01_help.sh`          | Usage text mentions both legacy positional args and new flags (`--chat`, `--repeat-*`, `--lora`, `--sequential-prefill`) |
| `test_02_legacy_cli.sh`    | Backward compatibility: original `<model> <prompt> tokenizer <args>` shape still works; raw mode; `decoder.py` round-trip |
| `test_03_rys.sh`           | `--repeat-start/end/count` logs `[RYS]` and produces output; argument validation rejects invalid ranges |
| `test_04_prefill_modes.sh` | Batched (default) and `--sequential-prefill` produce identical greedy output |
| `test_05_lora.sh`          | `--lora` loads adapter, shape check passes, output produced; identical between prefill modes; corrupt adapter is rejected |
| `test_06_scripts.sh`       | Every `scripts/*.py` and `python/*.py` passes `python3 -m py_compile` |
| `test_07_server.sh`        | `python/simple_server.py` starts; `/health` and `/` respond |

## Exit conventions

Each test file exits:

- `0` — all assertions passed
- `77` — test skipped (e.g., model not found, python3 unavailable)
- any other non-zero — at least one assertion failed
