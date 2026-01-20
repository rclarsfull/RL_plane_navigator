#!/usr/bin/env python3
"""
Minimal collector: read stable-baselines3/rl_zoo3 `evaluations.npz` files
under `logs/<algo>/<env>_*` and append compact rows to `results/eval_runs.csv`.
No TensorBoard required.
"""
from __future__ import annotations

import argparse
import csv
import json
import os
from pathlib import Path
from typing import Any

import numpy as np


def find_runs(log_root: Path) -> list[tuple[str, Path]]:
    runs: list[tuple[str, Path]] = []
    if not log_root.exists():
        return runs
    for algo_dir in sorted(p for p in log_root.iterdir() if p.is_dir()):
        algo = algo_dir.name
        for run_dir in sorted(p for p in algo_dir.iterdir() if p.is_dir()):
            if (run_dir / "evaluations.npz").exists():
                runs.append((algo, run_dir))
    return runs


essential_fields = [
    "timestamp",
    "env",
    "algo",
    "seed",
    "run_dir",
    "last_timestep",
    "last_eval_mean",
    "last_eval_median",
    "best_eval_mean",
    "num_evals",
    "last_success_rate",
    "best_success_rate",
]


def append_row(csv_path: Path, row: dict[str, Any]) -> None:
    csv_path.parent.mkdir(parents=True, exist_ok=True)
    is_new = not csv_path.exists()
    with csv_path.open("a", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=essential_fields)
        if is_new:
            writer.writeheader()
        writer.writerow(row)


def read_seed_from_args(params_dir: Path, run_name: str = "") -> int | str:
    # Read seed from args.yml saved by exp_manager
    # Format is YAML OrderedDict: 
    #   - - seed
    #     - 666
    args_yml = params_dir / "args.yml"
    if args_yml.exists():
        try:
            text = args_yml.read_text()
            lines = text.splitlines()
            for i, line in enumerate(lines):
                # Look for the line containing "- - seed" or "- seed"
                if "seed" in line and "- " in line:
                    # Next line should contain the value with format "    - VALUE"
                    if i + 1 < len(lines):
                        next_line = lines[i + 1]
                        # Extract the value after "- "
                        stripped = next_line.strip()
                        if stripped.startswith("- "):
                            val = stripped[2:].strip()
                            try:
                                return int(val)
                            except ValueError:
                                return val
        except Exception:
            pass
    # Fallback: return empty string if extraction fails
    return ""


def load_npz_metrics(npz_path: Path) -> dict[str, Any]:
    metrics: dict[str, Any] = {
        "last_timestep": -1,
        "last_eval_mean": np.nan,
        "last_eval_median": np.nan,
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
            metrics["last_eval_mean"] = float(mean_per_eval[-1])
            metrics["last_eval_median"] = float(median_per_eval[-1])
            metrics["best_eval_mean"] = float(mean_per_eval.max())
        if successes is not None and successes.size > 0:
            succ_arr = np.asarray(successes, dtype=float)
            # Compute success rate per evaluation (mean across episodes)
            success_rate_per_eval = succ_arr.mean(axis=1)
            metrics["last_success_rate"] = float(success_rate_per_eval[-1])
            metrics["best_success_rate"] = float(success_rate_per_eval.max())
        if timesteps is not None and timesteps.size > 0:
            metrics["last_timestep"] = int(timesteps[-1])
    except Exception:
        # keep defaults
        pass
    return metrics


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--logs", type=Path, default=Path("logs"))
    parser.add_argument("--out", type=Path, default=Path("results/eval_runs.csv"))
    args = parser.parse_args()

    runs = find_runs(args.logs)
    if not runs:
        print(f"No runs with evaluations.npz under {args.logs}")
        return

    for algo, run_dir in runs:
        # env hint: try to find the actual env name from subdirectories
        # rl-zoo3 creates: logs/<algo>/<env>_<id>/<env>/args.yml
        subdirs = [d for d in run_dir.iterdir() if d.is_dir()]
        if subdirs:
            # Use the first subdirectory name as the env name
            env_hint = subdirs[0].name
            params_dir = subdirs[0]
        else:
            # Fallback: strip numeric suffix from run_dir name
            env_hint = run_dir.name.rsplit("_", 1)[0]
            params_dir = run_dir / env_hint
        
        # seed: read from saved args if present
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
        append_row(args.out, row)
        print(json.dumps({"env": env_hint, "algo": algo, "seed": seed, "last_mean": metrics["last_eval_mean"]}))


if __name__ == "__main__":
    main()
