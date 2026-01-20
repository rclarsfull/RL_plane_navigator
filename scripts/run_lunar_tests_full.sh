#!/bin/bash

set -euo pipefail

# Seeds and algos to test
SEEDS=(666 1337 42 2024 99)
ALGOS=(masked_ppo ppo)

# Env and logging
ENV_ID="LunarLander-v3"
LOG_ROOT="logs"

# Tunables (override via env vars when calling)
N_TIMESTEPS="${N_TIMESTEPS:-200000}"
EVAL_FREQ="${EVAL_FREQ:-10000}"
EVAL_EPISODES="${EVAL_EPISODES:-30}"
N_EVAL_ENVS="${N_EVAL_ENVS:-4}"

ITERATIONS=${#SEEDS[@]}
RUN_IDX=0

for SEED in "${SEEDS[@]}"; do
  RUN_IDX=$((RUN_IDX + 1))
  echo "----------------------------------------------------------"
  echo "LunarLander: Lauf ${RUN_IDX} von ${ITERATIONS} mit Seed: ${SEED}"
  echo "----------------------------------------------------------"

  for ALGO in "${ALGOS[@]}"; do
    echo "Starte ${ALGO}..."
    python train.py \
      --algo "${ALGO}" \
      --env "${ENV_ID}" \
      -n "${N_TIMESTEPS}" \
      --eval-freq "${EVAL_FREQ}" \
      --eval-episodes "${EVAL_EPISODES}" \
      --n-eval-envs "${N_EVAL_ENVS}" \
      -P \
      --tensorboard-log "${LOG_ROOT}/tensorboard" \
      --seed "${SEED}"
  done

done

# Collect evaluations into a CSV for analysis
echo "Collecting evaluations..."
python scripts/collect_evals.py --logs "${LOG_ROOT}" --out results/eval_runs.csv

# Analyze and generate profiles + plots
echo "Analyzing results and generating plots..."
python scripts/analyze_evals.py --results results/eval_runs.csv --env "${ENV_ID}" --algo "${ALGOS[0]}" --algo2 "${ALGOS[1]}" --B 5000
python scripts/plot_evals.py --results results/eval_runs.csv --env "${ENV_ID}" --algos "${ALGOS[@]}"

echo "Done! Plots saved to results/plots/"
ls -lh results/plots/
