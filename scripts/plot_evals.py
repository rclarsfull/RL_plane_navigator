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


def filter_scores(rows: List[dict[str, Any]], env: str | None, algo: str, metric: str = "last_eval_mean") -> np.ndarray:
    vals: List[float] = []
    for r in rows:
        if r.get("algo") != algo:
            continue
        if env is not None and r.get("env") != env:
            continue
        v = r.get(metric)
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


def load_iqm_stats(path: Path) -> dict[str, float] | None:
    if not path.exists():
        return None
    try:
        with path.open("r", newline="") as f:
            reader = csv.DictReader(f)
            # Expecting one row
            for r in reader:
                return {
                    "low": float(r["low"]),
                    "iqm": float(r["iqm"]),
                    "high": float(r["high"]),
                    "n_runs": int(float(r["n_runs"])), # float cast for safety
                }
    except Exception:
        pass
    return None


def main() -> None:
# ...existing code...
    # Plot 1: IQM CIs for all algos (loop over list of metrics)
    metrics_to_plot = ["last_eval_mean", "mean_eval_mean"]
    
    for metric in metrics_to_plot:
        fig, ax = plt.subplots(figsize=(10, 6))
        results = []
        for algo in args.algos:
            # Try to load pre-calculated IQM stats
            iqm_csv = args.out.parent / f"iqm_stats_{args.env or 'all'}_{algo}_{metric}.csv"
            stats = load_iqm_stats(iqm_csv)
            
            if stats:
                print(f"Loaded IQM stats for {algo} ({metric}) from {iqm_csv}")
                results.append({
                    "algo": algo,
                    "n_runs": stats["n_runs"],
                    "iqm": stats["iqm"],
                    "low": stats["low"],
                    "high": stats["high"],
                })
            else:
                # Fallback to calculating from raw rows
                print(f"Calculating IQM stats for {algo} ({metric})... (file {iqm_csv} not found)")
                scores = filter_scores(rows, args.env, algo, metric=metric)
                ci_low, ci_med, ci_high = bootstrap_iqm(scores, B=args.B)
                results.append({
                    "algo": algo,
                    "n_runs": scores.size,
                    "iqm": ci_med,
                    "low": ci_low,
                    "high": ci_high,
                })

        # Bar plot
        algos_list = [r["algo"] for r in results]
# ...existing code...
        iqms = [r["iqm"] for r in results]
        lows = [r["iqm"] - r["low"] for r in results]
        highs = [r["high"] - r["iqm"] for r in results]
        
        ax.bar(algos_list, iqms, yerr=[lows, highs], capsize=5, alpha=0.7, color=["blue", "orange", "green"][:len(algos_list)])
        ax.set_ylabel(f"IQM Score ({metric})", fontsize=12)
        ax.set_title(f"IQM with 95% CI ({args.env or 'All Envs'}) - {metric}", fontsize=14)
        ax.grid(axis="y", alpha=0.3)
        
        # Add count labels
        for i, r in enumerate(results):
            ax.text(i, r["iqm"], f"n={r['n_runs']}", ha="center", va="bottom", fontsize=10)
        
        fig.tight_layout()
        out_path = args.out / f"iqm_ci_{args.env or 'all'}_{metric}.png"
        fig.savefig(out_path, dpi=args.dpi)
        print(f"Saved: {out_path}")
        
        # Print summary
        print(f"\nIQM Summary for {metric}:")
        for r in results:
            ci_str = f"[{r['low']:.3f}, {r['iqm']:.3f}, {r['high']:.3f}]"
            print(f"  {r['algo']:20s} IQM CI = {ci_str}  (n={r['n_runs']})")

        # Plot 2: Performance profiles for all algos (for this metric)
        fig, ax = plt.subplots(figsize=(10, 6))
        found_any = False
        for algo in args.algos:
            # Look for perf_profile_{env}_{algo}_{metric}.csv
            profile_csv = args.out.parent / f"perf_profile_{args.env or 'all'}_{algo}_{metric}.csv"
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
            ax.set_ylabel(f"Fraction of Runs > τ ({metric})", fontsize=12)
            ax.set_title(f"Performance Profiles ({args.env or 'All Envs'}) - {metric}", fontsize=14)
            ax.legend(fontsize=11)
            ax.grid(alpha=0.3)
            ax.set_ylim([0, 1.05])
            
            fig.tight_layout()
            out_path = args.out / f"perf_profile_{args.env or 'all'}_{metric}.png"
            fig.savefig(out_path, dpi=args.dpi)
            print(f"Saved: {out_path}")
        else:
            print(f"No performance profile CSVs found for {metric}. Run analyze_evals.py first.")


if __name__ == "__main__":
    main()
