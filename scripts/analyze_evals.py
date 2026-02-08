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


def filter_scores_with_seeds(rows: List[dict[str, Any]], env: str | None, algo: str, metric: str = "last_eval_mean") -> dict[str, float]:
    vals: dict[str, float] = {}
    for r in rows:
        if r.get("algo") != algo:
            continue
        if env is not None and r.get("env") != env:
            continue
        seed = r.get("seed", "").strip()
        v = r.get(metric)
        try:
            val = float(v)
            # If duplicates exist, later ones overwrite earlier ones (usually okay for collected logs)
            if seed:
                vals[seed] = val
        except Exception:
            pass
    return vals

def filter_success_rates_with_seeds(rows: List[dict[str, Any]], env: str | None, algo: str) -> dict[str, float]:
    vals: dict[str, float] = {}
    for r in rows:
        if r.get("algo") != algo:
            continue
        if env is not None and r.get("env") != env:
            continue
        seed = r.get("seed", "").strip()
        v = r.get("last_success_rate")
        if v and v != "":
            try:
                val = float(v)
                if seed:
                    vals[seed] = val
            except Exception:
                pass
    return vals

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


def bootstrap_diff_iqm_paired(d1: dict[str, float], d2: dict[str, float], B: int = 5000) -> Tuple[float, float, float]:
    # Find common seeds
    common_seeds = sorted(list(set(d1.keys()) & set(d2.keys())))
    if len(common_seeds) == 0:
        print("Warning: No common seeds found for paired comparison!")
        return (np.nan, np.nan, np.nan)
    
    # DEBUG: Zeige die tatsaechlichen Paare an, um die "Gleichheit" zu pruefen
    print(f"\n--- Paired Runs Check (n={len(common_seeds)}) ---")
    print(f"{'Seed':<10} | {'Algo1':<10} | {'Algo2':<10} | {'Diff':<10}")
    print("-" * 46)
    
    # Pre-sort values by seed to ensure strict alignment
    a_vals = []
    b_vals = []
    
    for s in common_seeds:
        v1 = d1[s]
        v2 = d2[s]
        a_vals.append(v1)
        b_vals.append(v2)
        print(f"{s:<10} | {v1:<10.2f} | {v2:<10.2f} | {v1-v2:<10.2f}")
    
    a_vals = np.array(a_vals)
    b_vals = np.array(b_vals)
    print("-" * 46 + "\n")
    
    K = len(common_seeds)
    diffs = np.empty(B, dtype=float)
    
    # Paired Bootstrap: Resample indices (seeds)
    indices = np.arange(K)
    for i in range(B):
        # Sample K indices with replacement
        idx = np.random.choice(indices, size=K, replace=True)
        # Construct resampled sets
        a_sample = a_vals[idx]
        b_sample = b_vals[idx]
        # Calculate diff of statistics
        diffs[i] = iqm(a_sample) - iqm(b_sample)
        
    return tuple(np.percentile(diffs, [2.5, 50, 97.5]))

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


def bootstrap_diff_mean_paired(d1: dict[str, float], d2: dict[str, float], B: int = 5000) -> Tuple[float, float, float]:
    common_seeds = sorted(list(set(d1.keys()) & set(d2.keys())))
    if len(common_seeds) == 0:
        return (np.nan, np.nan, np.nan)
        
    a_vals = np.array([d1[s] for s in common_seeds])
    b_vals = np.array([d2[s] for s in common_seeds])
    
    K = len(common_seeds)
    diffs = np.empty(B, dtype=float)
    
    indices = np.arange(K)
    for i in range(B):
        idx = np.random.choice(indices, size=K, replace=True)
        a_sample = a_vals[idx]
        b_sample = b_vals[idx]
        diffs[i] = np.mean(a_sample) - np.mean(b_sample)
        
    return tuple(np.percentile(diffs, [2.5, 50, 97.5]))



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


def write_iqm_csv(out_path: Path, low: float, med: float, high: float, n_runs: int) -> None:
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with out_path.open("w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["low", "iqm", "high", "n_runs"])
        w.writerow([float(low), float(med), float(high), int(n_runs)])


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
    
    metrics = ["last_eval_mean", "mean_eval_mean"]
    for metric in metrics:
        print(f"\n--- Analyzing metric: {metric} ---")
        
        # Extract data with seeds for paired analysis
        dict_s1 = filter_scores_with_seeds(rows, args.env, args.algo, metric=metric)
        s1 = np.array(list(dict_s1.values()))
        
        # Skip if no data found for this metric
        if s1.size == 0:
            print(f"No data found for {args.algo} with metric {metric}")
            continue

        ci1 = bootstrap_iqm(s1, B=args.B)
        print(f"{args.algo}: IQM CI [2.5,50,97.5] = {ci1}")
        
        # Save IQM stats
        out_iqm = args.outdir / f"iqm_stats_{args.env or 'all'}_{args.algo}_{metric}.csv"
        write_iqm_csv(out_iqm, ci1[0], ci1[1], ci1[2], s1.size)
        print(f"Wrote IQM stats to {out_iqm}")

        s2 = np.array([])
        if args.algo2:
            dict_s2 = filter_scores_with_seeds(rows, args.env, args.algo2, metric=metric)
            s2 = np.array(list(dict_s2.values()))
            
            if s2.size > 0:
                ci2 = bootstrap_iqm(s2, B=args.B)
                print(f"{args.algo2}: IQM CI [2.5,50,97.5] = {ci2}")
                # Save IQM stats for algo2
                out_iqm2 = args.outdir / f"iqm_stats_{args.env or 'all'}_{args.algo2}_{metric}.csv"
                write_iqm_csv(out_iqm2, ci2[0], ci2[1], ci2[2], s2.size)
                print(f"Wrote IQM stats to {out_iqm2}")
                
                # Paired Diff
                diff_ci = bootstrap_diff_iqm_paired(dict_s1, dict_s2, B=args.B)
                print(f"Diff IQM ({args.algo} - {args.algo2}) CI [2.5,50,97.5] = {diff_ci}")

        # Performance profile for algo1
        if s1.size > 0:
            taus = np.linspace(np.min(s1), np.max(s1), args.profile_points)
            low, med, high = bootstrap_profile(s1, taus, B=min(2000, args.B))
            out_csv = args.outdir / f"perf_profile_{args.env or 'all'}_{args.algo}_{metric}.csv"
            write_profile_csv(out_csv, taus, low, med, high)
            print(f"Wrote performance profile CSV to {out_csv}")

        # Performance profile for algo2 (if provided)
        if args.algo2 and s2.size > 0:
            taus = np.linspace(np.min(s2), np.max(s2), args.profile_points)
            low, med, high = bootstrap_profile(s2, taus, B=min(2000, args.B))
            out_csv = args.outdir / f"perf_profile_{args.env or 'all'}_{args.algo2}_{metric}.csv"
            write_profile_csv(out_csv, taus, low, med, high)
            print(f"Wrote performance profile CSV to {out_csv}")

        # Write IQM CI to CSV
        if s1.size > 0:
            out_csv = args.outdir / f"iqm_ci_{args.env or 'all'}_{args.algo}_{metric}.csv"
            write_iqm_csv(out_csv, *ci1, n_runs=s1.size)
            print(f"Wrote IQM CI CSV to {out_csv}")
        if s2.size > 0 and args.algo2:
            out_csv = args.outdir / f"iqm_ci_{args.env or 'all'}_{args.algo2}_{metric}.csv"
            write_iqm_csv(out_csv, *ci2, n_runs=s2.size)
            print(f"Wrote IQM CI CSV to {out_csv}")

    # Success Rate analysis (if available) - only done once as it's separate from score metric
    print("\n--- Analyzing Success Rate ---")
    dict_sr1 = filter_success_rates_with_seeds(rows, args.env, args.algo)
    sr1 = np.array(list(dict_sr1.values()))
    
    if sr1.size > 0:
        sr_ci1 = bootstrap_mean(sr1, B=args.B)
        print(f"{args.algo}: Success Rate CI [2.5,50,97.5] = {sr_ci1}")
    
    if args.algo2:
        dict_sr2 = filter_success_rates_with_seeds(rows, args.env, args.algo2)
        sr2 = np.array(list(dict_sr2.values()))
        
        if sr2.size > 0:
            sr_ci2 = bootstrap_mean(sr2, B=args.B)
            print(f"{args.algo2}: Success Rate CI [2.5,50,97.5] = {sr_ci2}")
        if sr1.size > 0 and sr2.size > 0:
            sr_diff = bootstrap_diff_mean_paired(dict_sr1, dict_sr2, B=args.B)
            print(f"Diff Success Rate ({args.algo} - {args.algo2}) CI [2.5,50,97.5] = {sr_diff}")


if __name__ == "__main__":
    main()
