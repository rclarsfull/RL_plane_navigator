#!/usr/bin/env python3
"""
Plot IQM CIs and performance profiles from eval results.
Requires matplotlib (optional dependency).
"""
from __future__ import annotations

import argparse
import csv
from pathlib import Path
from typing import Any, List

import numpy as np

try:
    import matplotlib.pyplot as plt
except ImportError:
    print("matplotlib not installed; install with: pip install matplotlib")
    exit(1)


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


def iqm(x: np.ndarray) -> float:
    if x.size == 0:
        return np.nan
    x_sorted = np.sort(x)
    k = x_sorted.size
    lo, hi = int(0.25 * k), int(0.75 * k)
    if hi <= lo:
        return float(np.mean(x_sorted))
    return float(np.mean(x_sorted[lo:hi]))


def bootstrap_iqm(scores: np.ndarray, B: int = 5000) -> tuple[float, float, float]:
    if scores.size == 0:
        return (np.nan, np.nan, np.nan)
    K = scores.size
    samples = np.empty(B, dtype=float)
    for b in range(B):
        res = np.random.choice(scores, size=K, replace=True)
        samples[b] = iqm(res)
    return tuple(np.percentile(samples, [2.5, 50, 97.5]))


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--results", type=Path, default=Path("results/eval_runs.csv"))
    parser.add_argument("--env", type=str, default=None, help="Filter by env (optional)")
    parser.add_argument("--algos", nargs="+", required=True, help="Algos to plot (e.g., masked_ppo ppo)")
    parser.add_argument("--B", type=int, default=5000, help="Bootstrap iterations")
    parser.add_argument("--out", type=Path, default=Path("results/plots"))
    parser.add_argument("--dpi", type=int, default=100)
    args = parser.parse_args()

    args.out.mkdir(parents=True, exist_ok=True)

    # Read data
    rows = read_csv(args.results)

    # Plot 1: IQM CIs for all algos
    fig, ax = plt.subplots(figsize=(10, 6))
    results = []
    for algo in args.algos:
        scores = filter_scores(rows, args.env, algo)
        ci_low, ci_med, ci_high = bootstrap_iqm(scores, B=args.B)
        results.append({
            "algo": algo,
            "n_runs": scores.size,
            "iqm": ci_med,
            "low": ci_low,
            "high": ci_high,
        })

    # Bar plot
    algos = [r["algo"] for r in results]
    iqms = [r["iqm"] for r in results]
    lows = [r["iqm"] - r["low"] for r in results]
    highs = [r["high"] - r["iqm"] for r in results]
    
    ax.bar(algos, iqms, yerr=[lows, highs], capsize=5, alpha=0.7, color=["blue", "orange", "green"][:len(algos)])
    ax.set_ylabel("IQM Score", fontsize=12)
    ax.set_title(f"IQM with 95% CI ({args.env or 'All Envs'})", fontsize=14)
    ax.grid(axis="y", alpha=0.3)
    
    # Add count labels
    for i, r in enumerate(results):
        ax.text(i, r["iqm"], f"n={r['n_runs']}", ha="center", va="bottom", fontsize=10)
    
    fig.tight_layout()
    out_path = args.out / f"iqm_ci_{args.env or 'all'}.png"
    fig.savefig(out_path, dpi=args.dpi)
    print(f"Saved: {out_path}")
    
    # Print summary
    print("\nIQM Summary:")
    for r in results:
        ci_str = f"[{r['low']:.3f}, {r['iqm']:.3f}, {r['high']:.3f}]"
        print(f"  {r['algo']:20s} IQM CI = {ci_str}  (n={r['n_runs']})")

    # Plot 2: Performance profiles for all algos
    fig, ax = plt.subplots(figsize=(10, 6))
    found_any = False
    for algo in args.algos:
        profile_csv = args.out.parent / f"perf_profile_{args.env or 'all'}_{algo}.csv"
        if profile_csv.exists():
            found_any = True
            taus, lows, meds, highs = [], [], [], []
            with profile_csv.open("r") as f:
                reader = csv.DictReader(f)
                for row in reader:
                    taus.append(float(row["tau"]))
                    meds.append(float(row["median"]))
                    lows.append(float(row["low"]))
                    highs.append(float(row["high"]))
            taus, lows, meds, highs = np.array(taus), np.array(lows), np.array(meds), np.array(highs)
            ax.plot(taus, meds, label=algo, linewidth=2)
            ax.fill_between(taus, lows, highs, alpha=0.2)
    
    if found_any:
        ax.set_xlabel("Score Threshold (τ)", fontsize=12)
        ax.set_ylabel("Fraction of Runs > τ", fontsize=12)
        ax.set_title(f"Performance Profiles ({args.env or 'All Envs'})", fontsize=14)
        ax.legend(fontsize=11)
        ax.grid(alpha=0.3)
        ax.set_ylim([0, 1.05])
        
        fig.tight_layout()
        out_path = args.out / f"perf_profile_{args.env or 'all'}.png"
        fig.savefig(out_path, dpi=args.dpi)
        print(f"Saved: {out_path}")
    else:
        print("No performance profile CSVs found. Run analyze_evals.py first for each algo.")


if __name__ == "__main__":
    main()
