#!/usr/bin/env python3
"""
Minimal analysis on `results/eval_runs.csv`:
- IQM with percentile bootstrap CIs per algo (optionally diff CI between two algos)
- Run-score performance profile (fraction of runs > tau) with bootstrap bands
No plotting dependencies; writes CSVs and prints summaries.
"""
from __future__ import annotations

import argparse
import csv
from pathlib import Path
from typing import Any, List, Tuple

import numpy as np


def read_csv(path: Path) -> List[dict[str, Any]]:
    rows: List[dict[str, Any]] = []
    with path.open("r", newline="") as f:
        reader = csv.DictReader(f)
        for r in reader:
            rows.append(r)
    return rows


def filter_scores(rows: List[dict[str, Any]], env: str | None, algo: str) -> np.ndarray:
    vals: List[float] = []
    for r in rows:
        if r.get("algo") != algo:
            continue
        if env is not None and r.get("env") != env:
            continue
        v = r.get("last_eval_mean")
        try:
            vals.append(float(v))
        except Exception:
            pass
    return np.asarray(vals, dtype=float)


def filter_success_rates(rows: List[dict[str, Any]], env: str | None, algo: str) -> np.ndarray:
    vals: List[float] = []
    for r in rows:
        if r.get("algo") != algo:
            continue
        if env is not None and r.get("env") != env:
            continue
        v = r.get("last_success_rate")
        if v and v != "":
            try:
                vals.append(float(v))
            except Exception:
                pass
    return np.asarray(vals, dtype=float)


def iqm(x: np.ndarray) -> float:
    if x.size == 0:
        return np.nan
    x_sorted = np.sort(x)
    k = x_sorted.size
    lo, hi = int(0.25 * k), int(0.75 * k)
    if hi <= lo:
        return float(np.mean(x_sorted))  # fallback when too few values
    return float(np.mean(x_sorted[lo:hi]))


def bootstrap_iqm(scores: np.ndarray, B: int = 5000) -> Tuple[float, float, float]:
    if scores.size == 0:
        return (np.nan, np.nan, np.nan)
    K = scores.size
    samples = np.empty(B, dtype=float)
    for b in range(B):
        res = np.random.choice(scores, size=K, replace=True)
        samples[b] = iqm(res)
    return tuple(np.percentile(samples, [2.5, 50, 97.5]))


def bootstrap_diff_iqm(a: np.ndarray, b: np.ndarray, B: int = 5000) -> Tuple[float, float, float]:
    if a.size == 0 or b.size == 0:
        return (np.nan, np.nan, np.nan)
    sa = np.empty(B, dtype=float)
    sb = np.empty(B, dtype=float)
    Ka, Kb = a.size, b.size
    for i in range(B):
        ra = np.random.choice(a, size=Ka, replace=True)
        rb = np.random.choice(b, size=Kb, replace=True)
        sa[i] = iqm(ra)
        sb[i] = iqm(rb)
    diff = sa - sb
    return tuple(np.percentile(diff, [2.5, 50, 97.5]))


def bootstrap_mean(values: np.ndarray, B: int = 5000) -> Tuple[float, float, float]:
    """Bootstrap confidence interval for mean (used for success rates)."""
    if values.size == 0:
        return (np.nan, np.nan, np.nan)
    K = values.size
    samples = np.empty(B, dtype=float)
    for b in range(B):
        res = np.random.choice(values, size=K, replace=True)
        samples[b] = np.mean(res)
    return tuple(np.percentile(samples, [2.5, 50, 97.5]))


def bootstrap_diff_mean(a: np.ndarray, b: np.ndarray, B: int = 5000) -> Tuple[float, float, float]:
    """Bootstrap CI for difference in means."""
    if a.size == 0 or b.size == 0:
        return (np.nan, np.nan, np.nan)
    sa = np.empty(B, dtype=float)
    sb = np.empty(B, dtype=float)
    Ka, Kb = a.size, b.size
    for i in range(B):
        ra = np.random.choice(a, size=Ka, replace=True)
        rb = np.random.choice(b, size=Kb, replace=True)
        sa[i] = np.mean(ra)
        sb[i] = np.mean(rb)
    diff = sa - sb
    return tuple(np.percentile(diff, [2.5, 50, 97.5]))


def run_score_profile(scores: np.ndarray, taus: np.ndarray) -> np.ndarray:
    return np.array([np.mean(scores > t) if scores.size > 0 else np.nan for t in taus], dtype=float)


def bootstrap_profile(scores: np.ndarray, taus: np.ndarray, B: int = 2000) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    if scores.size == 0:
        nan = np.full_like(taus, np.nan, dtype=float)
        return nan, nan, nan
    K = scores.size
    curves = np.empty((B, taus.size), dtype=float)
    for i in range(B):
        res = np.random.choice(scores, size=K, replace=True)
        curves[i] = run_score_profile(res, taus)
    low, med, high = np.percentile(curves, [2.5, 50, 97.5], axis=0)
    return low, med, high


def write_profile_csv(out_path: Path, taus: np.ndarray, low: np.ndarray, med: np.ndarray, high: np.ndarray) -> None:
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with out_path.open("w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["tau", "low", "median", "high"])  # 95% CI band + median
        for t, l, m, h in zip(taus, low, med, high):
            w.writerow([float(t), float(l), float(m), float(h)])


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--results", type=Path, default=Path("results/eval_runs.csv"))
    parser.add_argument("--env", type=str, default=None, help="Filter by env (optional)")
    parser.add_argument("--algo", type=str, required=True)
    parser.add_argument("--algo2", type=str, default=None, help="Optional second algo for diff CI")
    parser.add_argument("--B", type=int, default=5000, help="Bootstrap iterations")
    parser.add_argument("--profile_points", type=int, default=50)
    parser.add_argument("--outdir", type=Path, default=Path("results"))
    args = parser.parse_args()

    rows = read_csv(args.results)
    s1 = filter_scores(rows, args.env, args.algo)
    ci1 = bootstrap_iqm(s1, B=args.B)
    print(f"{args.algo}: IQM CI [2.5,50,97.5] = {ci1}")

    if args.algo2:
        s2 = filter_scores(rows, args.env, args.algo2)
        ci2 = bootstrap_iqm(s2, B=args.B)
        print(f"{args.algo2}: IQM CI [2.5,50,97.5] = {ci2}")
        diff_ci = bootstrap_diff_iqm(s1, s2, B=args.B)
        print(f"Diff IQM ({args.algo} - {args.algo2}) CI [2.5,50,97.5] = {diff_ci}")

    # Success Rate analysis (if available)
    sr1 = filter_success_rates(rows, args.env, args.algo)
    if sr1.size > 0:
        sr_ci1 = bootstrap_mean(sr1, B=args.B)
        print(f"{args.algo}: Success Rate CI [2.5,50,97.5] = {sr_ci1}")
    
    if args.algo2:
        sr2 = filter_success_rates(rows, args.env, args.algo2)
        if sr2.size > 0:
            sr_ci2 = bootstrap_mean(sr2, B=args.B)
            print(f"{args.algo2}: Success Rate CI [2.5,50,97.5] = {sr_ci2}")
        if sr1.size > 0 and sr2.size > 0:
            sr_diff = bootstrap_diff_mean(sr1, sr2, B=args.B)
            print(f"Diff Success Rate ({args.algo} - {args.algo2}) CI [2.5,50,97.5] = {sr_diff}")

    # Performance profile for algo1
    if s1.size > 0:
        taus = np.linspace(np.min(s1), np.max(s1), args.profile_points)
        low, med, high = bootstrap_profile(s1, taus, B=min(2000, args.B))
        out_csv = args.outdir / f"perf_profile_{args.env or 'all'}_{args.algo}.csv"
        write_profile_csv(out_csv, taus, low, med, high)
        print(f"Wrote performance profile CSV to {out_csv}")

    # Performance profile for algo2 (if provided)
    if args.algo2:
        if s2.size > 0:
            taus = np.linspace(np.min(s2), np.max(s2), args.profile_points)
            low, med, high = bootstrap_profile(s2, taus, B=min(2000, args.B))
            out_csv = args.outdir / f"perf_profile_{args.env or 'all'}_{args.algo2}.csv"
            write_profile_csv(out_csv, taus, low, med, high)
            print(f"Wrote performance profile CSV to {out_csv}")


if __name__ == "__main__":
    main()
