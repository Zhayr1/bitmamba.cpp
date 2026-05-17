#!/usr/bin/env python3
"""
brain_scanner.py — BitMamba-2 LLM Neuroanatomy Brain Scanner (v2)
=================================================================
Grid-searches all (repeat_start, repeat_end) layer combinations using
log-probability scoring on a diverse dataset of BoolQ + ARC-Easy problems.

Improvement over v1:
  - v1 used only 5 arithmetic problems (not representative of model capabilities)
  - v2 uses ~30 diverse problems from BoolQ and ARC-Easy
  - v2 pre-filters problems where baseline is correct (measures improvement, not noise)
  - Same fast single-token scoring via 'score' mode

Metric: average log-probability of the correct answer token (higher = better).
        Gives continuous signal even when argmax is wrong.

Usage:
    python3 brain_scanner.py [options]
    python3 brain_scanner.py --dataset-size 40 --min-span 2
"""

import argparse
import csv
import re
import subprocess
import sys
import time
from typing import Dict, List, Optional, Tuple

import tiktoken
from datasets import load_dataset

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------
LOGPROB_RE = re.compile(
    r"\[SCORE\] Target: \d+ \| Rank: (\d+) \| LogProb: ([-\d.e+]+)"
)


# ---------------------------------------------------------------------------
# Problem types
# ---------------------------------------------------------------------------
class Problem:
    """A single evaluation problem with context and correct token ID."""
    def __init__(self, context: str, target_token_id: int, source: str, label: str):
        self.context = context
        self.target_token_id = target_token_id
        self.source = source  # "boolq" or "arc_easy"
        self.label = label    # human-readable problem label


def build_boolq_problems(enc: tiktoken.Encoding, n: int = 20) -> List[Problem]:
    """Build BoolQ problems: passage + question → score ' Yes' vs ' No'."""
    ds = load_dataset("google/boolq", split=f"validation[:{n * 3}]")
    yes_id = enc.encode(" Yes")[0]
    no_id = enc.encode(" No")[0]

    problems = []
    for item in ds:
        passage = item["passage"]
        question = item["question"]
        answer = item["answer"]
        context = f"{passage}\nQuestion: {question}?\nAnswer:"
        target_id = yes_id if answer else no_id
        problems.append(Problem(
            context=context,
            target_token_id=target_id,
            source="boolq",
            label=f"BoolQ: {question[:60]}",
        ))
        if len(problems) >= n:
            break
    return problems


def build_arc_problems(enc: tiktoken.Encoding, n: int = 20) -> List[Problem]:
    """Build ARC-Easy problems: question + options → score correct letter token."""
    ds = load_dataset("ai2_arc", "ARC-Easy", split=f"test[:{n * 3}]")

    # Map letter labels to token IDs (with leading space)
    letter_token_ids = {}
    for letter in ["A", "B", "C", "D", "E", "1", "2", "3", "4", "5"]:
        tokens = enc.encode(f" {letter}")
        if len(tokens) == 1:
            letter_token_ids[letter] = tokens[0]

    problems = []
    for item in ds:
        question = item["question"]
        choices = item["choices"]["text"]
        labels = item["choices"]["label"]
        answer_key = item["answerKey"]

        if answer_key not in letter_token_ids:
            continue

        context = f"Question: {question}\n"
        for label, text in zip(labels, choices):
            context += f"{label}. {text}\n"
        context += "Answer:"

        target_id = letter_token_ids[answer_key]
        problems.append(Problem(
            context=context,
            target_token_id=target_id,
            source="arc_easy",
            label=f"ARC: {question[:60]}",
        ))
        if len(problems) >= n:
            break
    return problems


# ---------------------------------------------------------------------------
# Scoring
# ---------------------------------------------------------------------------
def score_problem(binary: str, model: str, problem: Problem,
                  repeat_start: int = -1, repeat_end: int = -1,
                  repeat_count: int = 1,
                  enc: tiktoken.Encoding = None) -> Optional[Tuple[int, float]]:
    """
    Run the binary in score mode for one problem.
    Returns (rank, log_prob) or None on failure.
    """
    ctx_tokens = enc.encode(problem.context)
    token_str = " ".join(str(t) for t in ctx_tokens)

    cmd = [binary]
    if repeat_start >= 0:
        cmd.extend([
            "--repeat-start", str(repeat_start),
            "--repeat-end", str(repeat_end),
            "--repeat-count", str(repeat_count),
        ])
    cmd.extend([
        model, token_str, "raw",
        "0", "1", "0", "1", "0", "1",
        "score", str(problem.target_token_id),
    ])

    try:
        result = subprocess.run(cmd, capture_output=True, text=True, timeout=120)
        m = LOGPROB_RE.search(result.stdout)
        if m:
            return int(m.group(1)), float(m.group(2))
    except subprocess.TimeoutExpired:
        pass
    except Exception as e:
        print(f"\n  [ERR] {e}", flush=True)
    return None


def evaluate_combo(binary: str, model: str, problems: List[Problem],
                   repeat_start: int, repeat_end: int, repeat_count: int,
                   enc: tiktoken.Encoding) -> Tuple[float, float, int]:
    """Evaluate one (start, end) combo over all problems. Returns (avg_lp, avg_rank, n_ok)."""
    total_lp = 0.0
    total_rank = 0
    n = 0
    for p in problems:
        result = score_problem(binary, model, p,
                               repeat_start, repeat_end, repeat_count, enc)
        if result is not None:
            rank, lp = result
            total_lp += lp
            total_rank += rank
            n += 1
    avg_lp = total_lp / n if n else -999.0
    avg_rank = total_rank / n if n else 99999.0
    return avg_lp, avg_rank, n


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main():
    parser = argparse.ArgumentParser(description="BitMamba-2 Brain Scanner v2")
    parser.add_argument("--binary", default="./build/bitmamba")
    parser.add_argument("--model", default="bitmamba_1b.bin")
    parser.add_argument("--layers", type=int, default=32)
    parser.add_argument("--min-span", type=int, default=2)
    parser.add_argument("--top-n", type=int, default=10)
    parser.add_argument("--dataset-size", type=int, default=30,
                        help="Total problems (split between BoolQ + ARC)")
    parser.add_argument("--range-start", type=int, default=None,
                        help="Only scan layers starting from this index")
    parser.add_argument("--range-end", type=int, default=None,
                        help="Only scan layers up to this index")
    parser.add_argument("--from-csv", type=str, default=None,
                        help="Re-test top N combos from a previous scan CSV")
    parser.add_argument("--filter-baseline", action="store_true", default=True,
                        help="Only keep problems where baseline predicts correctly")
    parser.add_argument("--no-filter", action="store_true",
                        help="Keep all problems regardless of baseline performance")
    parser.add_argument("--log", default="brain_scan_v2.csv")
    args = parser.parse_args()

    enc = tiktoken.get_encoding("gpt2")
    n_per_source = args.dataset_size // 2

    print("=" * 72)
    print("  BitMamba-2 Brain Scanner v2 — LLM Neuroanatomy RYS")
    print("=" * 72)
    print(f"  Binary  : {args.binary}")
    print(f"  Model   : {args.model}")
    print(f"  Metric  : avg log-probability of correct token (higher = better)")
    print(f"  Log     : {args.log}")

    # --- Build diverse dataset ---
    print(f"\n[1/4] Building dataset ({args.dataset_size} problems)...")
    boolq_problems = build_boolq_problems(enc, n_per_source)
    arc_problems = build_arc_problems(enc, n_per_source)
    all_problems = boolq_problems + arc_problems
    print(f"  Loaded: {len(boolq_problems)} BoolQ + {len(arc_problems)} ARC-Easy = {len(all_problems)} total")

    # --- Baseline scoring ---
    print("\n[2/4] Scoring baseline (no layer repetition)...")
    baseline_results = []
    for i, p in enumerate(all_problems):
        result = score_problem(args.binary, args.model, p, enc=enc)
        baseline_results.append(result)
        if result:
            rank, lp = result
            status = "correct" if rank == 0 else f"rank={rank}"
        else:
            status = "FAIL"
        print(f"\r  {i+1}/{len(all_problems)} [{status:>10}] {p.label[:50]}", end="", flush=True)
    print()

    # Compute baseline average
    valid_baseline = [(r, p) for r, p in zip(baseline_results, all_problems) if r is not None]
    base_lp = sum(r[1] for r, _ in valid_baseline) / len(valid_baseline) if valid_baseline else -999
    base_rank = sum(r[0] for r, _ in valid_baseline) / len(valid_baseline) if valid_baseline else 999
    n_correct = sum(1 for r, _ in valid_baseline if r[0] == 0)
    print(f"  Baseline: avg_lp={base_lp:.4f}  avg_rank={base_rank:.1f}  "
          f"correct={n_correct}/{len(valid_baseline)}")

    # --- Filter to problems where baseline is correct (optional) ---
    if not args.no_filter:
        filtered = [p for r, p in zip(baseline_results, all_problems)
                    if r is not None and r[0] == 0]
        if len(filtered) < 5:
            print(f"  [WARN] Only {len(filtered)} correct predictions. Using all problems instead.")
            filtered = [p for r, p in zip(baseline_results, all_problems) if r is not None]
        print(f"  Filtered to {len(filtered)} problems (baseline correct)")
        problems = filtered
    else:
        problems = [p for r, p in zip(baseline_results, all_problems) if r is not None]
        print(f"  Using all {len(problems)} valid problems")

    # Recompute baseline on filtered set
    base_lp_filtered, base_rank_filtered, base_n = evaluate_combo(
        args.binary, args.model, problems, -1, -1, 1, enc
    )
    print(f"  Filtered baseline: avg_lp={base_lp_filtered:.4f}  avg_rank={base_rank_filtered:.1f}")

    # --- Build combo list ---
    combos = []
    if args.from_csv:
        # Re-test top N combos from a previous scan
        import csv as csv_mod
        with open(args.from_csv, "r") as f:
            reader = csv_mod.DictReader(f)
            rows = []
            for row in reader:
                if row["repeat_start"].startswith("#"):
                    continue
                rows.append((int(row["repeat_start"]), int(row["repeat_end"]),
                             float(row["delta_log_prob"])))
        rows.sort(key=lambda x: x[2], reverse=True)
        combos = [(s, e) for s, e, _ in rows[:args.top_n]]
        print(f"  Re-testing top {len(combos)} combos from {args.from_csv}")
    else:
        lo = args.range_start if args.range_start is not None else 0
        hi = args.range_end if args.range_end is not None else args.layers - 1
        for start in range(lo, hi + 1):
            for end in range(start + args.min_span, min(hi + 1, args.layers)):
                combos.append((start, end))
        if args.range_start is not None:
            print(f"  Scanning layer range [{lo}, {hi}] only")

    total_jobs = len(combos)

    print(f"\n[3/4] Grid search: {total_jobs} combos x {len(problems)} problems "
          f"= {total_jobs * len(problems)} calls\n")

    results = {}
    completed = 0
    best_lp = -999.0
    best_combo = None
    t_start = time.time()

    log_file = open(args.log, "w", newline="", encoding="utf-8")
    log_writer = csv.writer(log_file)
    log_writer.writerow(["repeat_start", "repeat_end", "span",
                         "avg_log_prob", "delta_log_prob",
                         "avg_rank", "n_ok"])
    log_file.flush()

    for start, end in combos:
        avg_lp, avg_rank, n_ok = evaluate_combo(
            args.binary, args.model, problems, start, end, 1, enc
        )
        results[(start, end)] = (avg_lp, avg_rank, n_ok)
        completed += 1

        if avg_lp > best_lp:
            best_lp = avg_lp
            best_combo = (start, end)

        # Progress
        elapsed = time.time() - t_start
        avg_s = elapsed / completed
        remain = avg_s * (total_jobs - completed)
        filled = int(30 * completed / total_jobs)
        bar = "#" * filled + "." * (30 - filled)
        eta_str = f"{int(remain // 60)}m{int(remain % 60):02d}s"
        delta = avg_lp - base_lp_filtered
        best_str = (f"layers {best_combo[0]}-{best_combo[1]} lp={best_lp:.3f}"
                    if best_combo else "---")

        sys.stdout.write(
            f"\r  [{bar}] {completed:>4}/{total_jobs}  ETA {eta_str}  "
            f"last: {start:>2}-{end:>2} lp={avg_lp:.3f}(d{delta:+.3f})  "
            f"best: {best_str}"
        )
        sys.stdout.flush()

        # Write CSV immediately
        delta_lp = avg_lp - base_lp_filtered
        log_writer.writerow([start, end, end - start + 1,
                             f"{avg_lp:.6f}", f"{delta_lp:.6f}",
                             f"{avg_rank:.2f}", n_ok])
        log_file.flush()

    log_file.close()

    # --- Results ---
    print(f"\n\n[4/4] Top {args.top_n} layer combinations:\n")
    print(f"  {'Rank':>4}  {'Start':>6}  {'End':>4}  {'Avg LogProb':>12}  "
          f"{'Delta':>14}  {'Avg Rank':>9}  Span")
    print("  " + "-" * 65)

    ranked = sorted(results.items(), key=lambda x: x[1][0], reverse=True)

    for rank, ((start, end), (avg_lp, avg_rank, n_ok)) in enumerate(ranked[:args.top_n], 1):
        delta = avg_lp - base_lp_filtered
        span = end - start + 1
        marker = " <--" if rank == 1 else ""
        print(f"  {rank:>4}.  {start:>6}  {end:>4}  {avg_lp:>12.4f}  "
              f"{delta:>+14.4f}  {avg_rank:>9.1f}  [{span}]{marker}")

    if ranked:
        (bs, be), (bl, br, _) = ranked[0]
        print(f"\n  Best: --repeat-start {bs} --repeat-end {be}")
        print(f"  LogProb: {bl:.4f}  Delta={bl - base_lp_filtered:+.4f}  AvgRank: {br:.1f}")

    # Summary in log
    with open(args.log, "a", encoding="utf-8") as lf:
        lf.write(f"\n# --- SUMMARY ---\n")
        lf.write(f"# Baseline avg_log_prob: {base_lp_filtered:.6f}  avg_rank: {base_rank_filtered:.2f}\n")
        lf.write(f"# Dataset: {len(problems)} problems ({len([p for p in problems if p.source=='boolq'])} BoolQ + "
                 f"{len([p for p in problems if p.source=='arc_easy'])} ARC)\n")
        lf.write(f"# Top results:\n")
        for rank, ((s, e), (lp, rk, _)) in enumerate(ranked[:args.top_n], 1):
            lf.write(f"#  {rank:>3}. layers {s:>2}-{e:>2}  "
                     f"lp={lp:.4f}  delta={lp - base_lp_filtered:+.4f}  "
                     f"rank={rk:.1f}  span={e - s + 1}\n")
        if ranked:
            lf.write(f"# Best: --repeat-start {bs} --repeat-end {be}\n")

    print(f"\n  Full results saved to: {args.log}")
    print("  Done.\n")


if __name__ == "__main__":
    main()
