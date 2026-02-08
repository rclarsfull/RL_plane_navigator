#!/usr/bin/env python3
"""
Complete Evaluation Pipeline: Collect, Analyze, and Plot.
Merges functionality from collect_evals.py, analyze_evals.py, and plot_evals.py.
"""
from __future__ import annotations

import argparse
import csv
import json
import os
import sys
from pathlib import Path
from typing import Any, List, Tuple, Dict, Optional, Union

import numpy as np

try:
    import matplotlib.pyplot as plt
    HAS_MATPLOTLIB = True
except ImportError:
    HAS_MATPLOTLIB = False
    print("Warning: matplotlib not installed. Plotting will be skipped.")


# =============================================================================
# 1. Collection Logic (from collect_evals.py)
# =============================================================================

def find_runs(log_root: Path) -> List[Tuple[str, Path]]:
    """Scan directory for evaluations.npz files."""
    runs: List[Tuple[str, Path]] = []
    if not log_root.exists():
        return runs
    for algo_dir in sorted(p for p in log_root.iterdir() if p.is_dir()):
        algo = algo_dir.name
        for run_dir in sorted(p for p in algo_dir.iterdir() if p.is_dir()):
            if (run_dir / "evaluations.npz").exists():
                runs.append((algo, run_dir))
    return runs

def read_seed_from_args(params_dir: Path, run_name: str = "") -> Union[int, str]:
    """Extract seed from saved args.yml."""
    args_yml = params_dir / "args.yml"
    if args_yml.exists():
        try:
            text = args_yml.read_text()
            lines = text.splitlines()
            for i, line in enumerate(lines):
                if "seed" in line and "- " in line:
                    if i + 1 < len(lines):
                        next_line = lines[i + 1]
                        stripped = next_line.strip()
                        if stripped.startswith("- "):
                            val = stripped[2:].strip()
                            try:
                                return int(val)
                            except ValueError:
                                return val
        except Exception:
            pass
    return ""

def load_npz_metrics(npz_path: Path) -> Dict[str, Any]:
    """Load evaluation results from .npz file."""
    metrics: Dict[str, Any] = {
        "last_timestep": -1,
        "last_eval_mean": np.nan,
        "last_eval_median": np.nan,
        "mean_eval_mean": np.nan,
        "best_eval_mean": np.nan,
        "num_evals": 0,
        "last_success_rate": "",
        "best_success_rate": "",
    }
    try:
        data = np.load(npz_path, allow_pickle=True)
        results = data.get("results")
        timesteps = data.get("timesteps")
        successes = data.get("successes")
        if results is not None and results.size > 0:
            res_arr = np.asarray(results, dtype=float)
            mean_per_eval = res_arr.mean(axis=1)
            median_per_eval = np.median(res_arr, axis=1)
            metrics["num_evals"] = int(mean_per_eval.shape[0])
            metrics["mean_eval_mean"] = float(mean_per_eval.mean())
            metrics["last_eval_mean"] = float(mean_per_eval[-1])
            metrics["last_eval_median"] = float(median_per_eval[-1])
            metrics["best_eval_mean"] = float(mean_per_eval.max())
        if successes is not None and successes.size > 0:
            succ_arr = np.asarray(successes, dtype=float)
            success_rate_per_eval = succ_arr.mean(axis=1)
            metrics["last_success_rate"] = float(success_rate_per_eval[-1])
            metrics["best_success_rate"] = float(success_rate_per_eval.max())
        if timesteps is not None and timesteps.size > 0:
            metrics["last_timestep"] = int(timesteps[-1])
    except Exception:
        pass
    return metrics

def run_collection(args: argparse.Namespace) -> None:
    """Main collection routine."""
    print(f"Collecting evaluations from {args.logs}...")
    runs = find_runs(args.logs)
    if not runs:
        print(f"No runs with evaluations.npz under {args.logs}")
        return

    csv_path = args.results
    csv_path.parent.mkdir(parents=True, exist_ok=True)
    
    essential_fields = [
        "timestamp", "env", "algo", "seed", "run_dir",
        "last_timestep", "last_eval_mean", "last_eval_median", 
        "mean_eval_mean", "best_eval_mean", "num_evals",
        "last_success_rate", "best_success_rate",
    ]
    
    # Overwrite mode for simplicity as requested by pipeline logic
    rows_to_write = []
    
    for algo, run_dir in runs:
        subdirs = [d for d in run_dir.iterdir() if d.is_dir()]
        if subdirs:
            env_hint = subdirs[0].name
            params_dir = subdirs[0]
        else:
            env_hint = run_dir.name.rsplit("_", 1)[0]
            params_dir = run_dir / env_hint
        
        seed = read_seed_from_args(params_dir, run_dir.name)
        metrics = load_npz_metrics(run_dir / "evaluations.npz")
        row = {
            "timestamp": os.environ.get("SOURCE_DATE_EPOCH") or "",
            "env": env_hint,
            "algo": algo,
            "seed": seed,
            "run_dir": str(run_dir),
            **metrics,
        }
        rows_to_write.append(row)
        print(f"Processed {algo}/{env_hint} (seed={seed}): last_mean={metrics['last_eval_mean']:.2f}")

    with csv_path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=essential_fields)
        writer.writeheader()
        for r in rows_to_write:
            writer.writerow(r)
    print(f"Saved {len(rows_to_write)} rows to {csv_path}")


# =============================================================================
# 2. Analysis Logic (from analyze_evals.py)
# =============================================================================

def read_results_csv(path: Path) -> List[Dict[str, Any]]:
    rows: List[Dict[str, Any]] = []
    if not path.exists():
        return rows
    with path.open("r", newline="") as f:
        reader = csv.DictReader(f)
        for r in reader:
            rows.append(r)
    return rows

def filter_scores_with_seeds(rows: List[Dict[str, Any]], env: Optional[str], algo: str, metric: str = "last_eval_mean") -> Dict[str, float]:
    vals: Dict[str, float] = {}
    for r in rows:
        if r.get("algo") != algo:
            continue
        if env is not None and r.get("env") != env:
            continue
        seed = r.get("seed", "").strip()
        v = r.get(metric)
        try:
            val = float(v)
            if seed:
                vals[seed] = val
        except Exception:
            pass
    return vals

def filter_success_rates_with_seeds(rows: List[Dict[str, Any]], env: Optional[str], algo: str) -> Dict[str, float]:
    vals: Dict[str, float] = {}
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
        return float(np.mean(x_sorted))
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

def bootstrap_diff_iqm_paired(d1: Dict[str, float], d2: Dict[str, float], B: int = 5000) -> Tuple[float, float, float]:
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
        diffs[i] = iqm(a_vals[idx]) - iqm(b_vals[idx])
    return tuple(np.percentile(diffs, [2.5, 50, 97.5]))

def bootstrap_mean(values: np.ndarray, B: int = 5000) -> Tuple[float, float, float]:
    if values.size == 0:
        return (np.nan, np.nan, np.nan)
    K = values.size
    samples = np.empty(B, dtype=float)
    for b in range(B):
        res = np.random.choice(values, size=K, replace=True)
        samples[b] = np.mean(res)
    return tuple(np.percentile(samples, [2.5, 50, 97.5]))

def bootstrap_diff_mean_paired(d1: Dict[str, float], d2: Dict[str, float], B: int = 5000) -> Tuple[float, float, float]:
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
        diffs[i] = np.mean(a_vals[idx]) - np.mean(b_vals[idx])
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
        w.writerow(["tau", "low", "median", "high"])
        for t, l, m, h in zip(taus, low, med, high):
            w.writerow([float(t), float(l), float(m), float(h)])

def write_iqm_csv(out_path: Path, low: float, med: float, high: float, n_runs: int) -> None:
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with out_path.open("w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["low", "iqm", "high", "n_runs"])
        w.writerow([float(low), float(med), float(high), int(n_runs)])

def run_analysis(args: argparse.Namespace) -> None:
    rows = read_results_csv(args.results)
    if not rows:
        print("No results found to analyze.")
        return

    metrics = ["last_eval_mean", "mean_eval_mean"]
    
    # 1. Analyze Scores (IQM + Profiles)
    for metric in metrics:
        print(f"\n--- [Analysis] Metric: {metric} ---")
        
        # We'll gather all algos for unified processing if needed, 
        # but following old script we focus on args.algos list
        
        # Primary Algo
        if not args.algos:
            print("No algorithms specified via --algos")
            continue
            
        algo1 = args.algos[0]
        dict_s1 = filter_scores_with_seeds(rows, args.env, algo1, metric=metric)
        s1 = np.array(list(dict_s1.values()))
        
        if s1.size > 0:
            ci1 = bootstrap_iqm(s1, B=args.B)
            print(f"{algo1}: IQM CI = {ci1}")
            out_iqm = args.out / f"iqm_stats_{args.env or 'all'}_{algo1}_{metric}.csv"
            write_iqm_csv(out_iqm, ci1[0], ci1[1], ci1[2], s1.size)
            
            taus = np.linspace(np.min(s1), np.max(s1), args.profile_points)
            low, med, high = bootstrap_profile(s1, taus, B=min(2000, args.B))
            out_prof = args.out / f"perf_profile_{args.env or 'all'}_{algo1}_{metric}.csv"
            write_profile_csv(out_prof, taus, low, med, high)

        # Other Algos (Comparison)
        for i in range(1, len(args.algos)):
            algo2 = args.algos[i]
            dict_s2 = filter_scores_with_seeds(rows, args.env, algo2, metric=metric)
            s2 = np.array(list(dict_s2.values()))
            
            if s2.size > 0:
                ci2 = bootstrap_iqm(s2, B=args.B)
                print(f"{algo2}: IQM CI = {ci2}")
                out_iqm = args.out / f"iqm_stats_{args.env or 'all'}_{algo2}_{metric}.csv"
                write_iqm_csv(out_iqm, ci2[0], ci2[1], ci2[2], s2.size)

                taus = np.linspace(np.min(s2), np.max(s2), args.profile_points)
                low, med, high = bootstrap_profile(s2, taus, B=min(2000, args.B))
                out_prof = args.out / f"perf_profile_{args.env or 'all'}_{algo2}_{metric}.csv"
                write_profile_csv(out_prof, taus, low, med, high)
                
                # Paired Diff (Algo2 vs Algo1) - Just printing for now
                if s1.size > 0:
                    diff_ci = bootstrap_diff_iqm_paired(dict_s2, dict_s1, B=args.B)
                    print(f"  Diff IQM ({algo2} - {algo1}) CI = {diff_ci}")

    # 2. Analyze Success Rates
    print("\n--- [Analysis] Success Rates ---")
    algo1 = args.algos[0]
    dict_sr1 = filter_success_rates_with_seeds(rows, args.env, algo1)
    sr1 = np.array(list(dict_sr1.values()))
    
    if sr1.size > 0:
        ci_sr1 = bootstrap_mean(sr1, B=args.B)
        print(f"{algo1}: Success Rate CI = {ci_sr1}")

    for i in range(1, len(args.algos)):
        algo2 = args.algos[i]
        dict_sr2 = filter_success_rates_with_seeds(rows, args.env, algo2)
        sr2 = np.array(list(dict_sr2.values()))
        if sr2.size > 0:
            ci_sr2 = bootstrap_mean(sr2, B=args.B)
            print(f"{algo2}: Success Rate CI = {ci_sr2}")
            if sr1.size > 0:
                diff_sr = bootstrap_diff_mean_paired(dict_sr2, dict_sr1, B=args.B)
                print(f"  Diff Success Rate ({algo2} - {algo1}) CI = {diff_sr}")


# =============================================================================
# 3. Plotting Logic (from plot_evals.py)
# =============================================================================

def get_tb_log_name(algo: str) -> str:
    """
    Try to read the tb_log_name from the algorithm's Python file in rl_zoo3.
    Falls back to the algo name if not found.
    """
    try:
        # Construct path to the algo file, assuming it's in rl_zoo3
        # This might need adjustment if the project structure is different
        # e.g. from rl_zoo3/ppo.py for algo 'ppo'
        repo_root = Path(__file__).parent.parent
        algo_file = repo_root / "rl_zoo3" / f"{algo}.py"

        if not algo_file.exists():
            return algo

        with algo_file.open("r") as f:
            for line in f:
                if "tb_log_name" in line and "=" in line:
                    # This is a simple parser, might need to be more robust
                    # e.g. tb_log_name: str = "Masked_MO_PPO_split_net",
                    parts = line.split("=")
                    if len(parts) > 1:
                        tb_name = parts[1].strip()
                        # Remove quotes and commas
                        tb_name = tb_name.replace('"', '').replace("'", "").replace(",", "")
                        return tb_name
    except Exception:
        # In case of any error, just return the original algo name
        pass
    return algo

def load_iqm_stats(path: Path) -> Optional[Dict[str, float]]:
    if not path.exists():
        return None
    try:
        with path.open("r", newline="") as f:
            reader = csv.DictReader(f)
            for r in reader:
                return {
                    "low": float(r["low"]),
                    "iqm": float(r["iqm"]),
                    "high": float(r["high"]),
                    "n_runs": int(float(r["n_runs"])),
                }
    except Exception:
        pass
    return None

def run_plotting(args: argparse.Namespace) -> None:
    if not HAS_MATPLOTLIB:
        return
    
    args.out.mkdir(parents=True, exist_ok=True)
    # We may re-read CSV to fallback or just rely on cached stats
    rows = read_results_csv(args.results)
    
    # Create a mapping from algo name to tb_log_name for prettier plots
    if args.labels:
        if len(args.labels) != len(args.algos):
            print(f"Warning: Provided {len(args.labels)} labels for {len(args.algos)} algorithms. Ignoring labels.")
            algo_to_tb_name = {algo: get_tb_log_name(algo) for algo in args.algos}
        else:
            algo_to_tb_name = dict(zip(args.algos, args.labels))
    else:
        algo_to_tb_name = {algo: get_tb_log_name(algo) for algo in args.algos}
    
    metrics = ["last_eval_mean", "mean_eval_mean"]
    for metric in metrics:
        fig, ax = plt.subplots(figsize=(10, 5))
        results = []
        for algo in args.algos:
            iqm_csv = args.out / f"iqm_stats_{args.env or 'all'}_{algo}_{metric}.csv"
            stats = load_iqm_stats(iqm_csv)
            
            if stats:
                results.append({
                    "algo": algo,
                    "n_runs": stats["n_runs"],
                    "iqm": stats["iqm"],
                    "low": stats["low"],
                    "high": stats["high"],
                })
            else:
                # Fallback calculation if analysis wasn't run separately
                scores_dict = filter_scores_with_seeds(rows, args.env, algo, metric=metric)
                scores = np.array(list(scores_dict.values()))
                if scores.size > 0:
                    ci_low, ci_med, ci_high = bootstrap_iqm(scores, B=args.B)
                    results.append({
                        "algo": algo,
                        "n_runs": scores.size,
                        "iqm": ci_med,
                        "low": ci_low,
                        "high": ci_high,
                    })

        if not results:
            print(f"No results to plot for {metric}")
            plt.close(fig)
            continue
            
        # Bar plot
        algos_list = [algo_to_tb_name[r["algo"]] for r in results]

        iqms = [r["iqm"] for r in results]
        lows = [r["iqm"] - r["low"] for r in results]
        highs = [r["high"] - r["iqm"] for r in results]
        
        # Horizontal bar plot (swapped axes)
        ax.barh(algos_list, iqms, xerr=[lows, highs], capsize=10, alpha=0.7, color=["blue", "orange", "green", "red"][:len(algos_list)])
        ax.invert_yaxis()  # Reverse y-axis to match input order (top-to-bottom)
        
        # Scaling adjustment: Orient layout so data makes up about 50% of the horizontal space
        all_lows_val = [r["low"] for r in results]
        all_highs_val = [r["high"] for r in results]
        if all_lows_val and all_highs_val:
            min_val = min(all_lows_val)
            max_val = max(all_highs_val)
            diff = max_val - min_val
            if diff == 0:
                diff = 1.0
            
            # Add padding so the data range is roughly 50% of the plot width
            padding = diff * 0.5
            
            ax.set_xlim([min_val - padding, max_val + padding])

        ax.set_xlabel("IQM Score", fontsize=24)
        # ax.set_title(f"IQM with 95% CI ({args.env or 'All Envs'}) - {metric}", fontsize=28)
        ax.tick_params(axis='both', which='major', labelsize=20)
        ax.grid(axis="x", alpha=0.3)
        
        for i, r in enumerate(results):
            # Place text to the right of the error bar
            x_pos = r["high"]
            ax.text(x_pos, i, f" n={r['n_runs']}", ha="left", va="center", fontsize=20)
        
        fig.tight_layout()
        out_path = args.out / f"iqm_ci_{args.env or 'all'}_{metric}.png"
        fig.savefig(out_path, dpi=args.dpi)
        print(f"Saved plot: {out_path}")
        plt.close(fig)

        # Performance Profiles
        fig, ax = plt.subplots(figsize=(10, 10))
        found_any = False
        for algo in args.algos:
            profile_csv = args.out / f"perf_profile_{args.env or 'all'}_{algo}_{metric}.csv"
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
                label = algo_to_tb_name.get(algo, algo)
                ax.plot(taus, meds, label=label, linewidth=2)
                ax.fill_between(taus, lows, highs, alpha=0.2)
        
        if found_any:
            ax.set_xlabel(r"Score $(\tau)$", fontsize=24)
            ax.set_ylabel(r"Fraction of runs $(\geq \tau)$", fontsize=24)
            # ax.set_title(f"Performance Profiles ({args.env or 'All Envs'}) - {metric}", fontsize=28)
            # Legend below the plot
            ax.legend(fontsize=20, loc='upper center', bbox_to_anchor=(0.5, -0.20), ncol=2)
            ax.tick_params(axis='both', which='major', labelsize=20)
            ax.grid(alpha=0.3)
            ax.set_ylim([0, 1.05])
            fig.tight_layout()
            out_path = args.out / f"perf_profile_{args.env or 'all'}_{metric}.png"
            fig.savefig(out_path, dpi=args.dpi, bbox_inches='tight')
            print(f"Saved plot: {out_path}")
        plt.close(fig)


# =============================================================================
# Main
# =============================================================================

def main():
    parser = argparse.ArgumentParser(description="Collect, Analyze and Plot RL Evaluation Results")
    
    # Common arguments
    parser.add_argument("--logs", type=Path, default=Path("logs"), help="Root directory for logs")
    parser.add_argument("--results", type=Path, default=Path("results/eval_runs.csv"), help="Path to aggregated CSV")
    parser.add_argument("--out", type=Path, default=Path("results/plots"), help="Directory for plots and stats")
    
    # Filter arguments
    parser.add_argument("--env", type=str, default=None, help="Environment ID filter")
    parser.add_argument("--algos", nargs="+", default=[], help="List of algorithms to analyze/plot")
    
    # Analysis arguments
    parser.add_argument("--B", type=int, default=5000, help="Bootstrap iterations")
    parser.add_argument("--profile_points", type=int, default=50, help="Points for profile curves")
    
    # Plotting arguments
    parser.add_argument("--dpi", type=int, default=100, help="DPI for saved plots")
    parser.add_argument("--labels", nargs="+", default=[], help="Display names for algorithms in plots (must match order of --algos)")
    
    # Actions
    parser.add_argument("--skip-collect", action="store_true", help="Skip collection step")
    parser.add_argument("--skip-analyze", action="store_true", help="Skip analysis step")
    parser.add_argument("--skip-plot", action="store_true", help="Skip plotting step")
    
    args = parser.parse_args()

    # 1. Collection
    if not args.skip_collect:
        run_collection(args)
    
    # Ensure algos are set if not provided, try to infer from CSV
    if not args.algos and args.results.exists():
        rows = read_results_csv(args.results)
        found_algos = sorted(list(set(r["algo"] for r in rows)))
        if found_algos:
            print(f"Auto-detected algorithms: {found_algos}")
            args.algos = found_algos

    # 2. Analysis
    if not args.skip_analyze:
        run_analysis(args)

    # 3. Plotting
    if not args.skip_plot:
        run_plotting(args)

if __name__ == "__main__":
    main()
