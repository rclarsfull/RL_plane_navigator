#!/bin/bash

set -euo pipefail

# Seeds and algos to test
SEEDS=(666 1337 42 2024 99)
ALGOS=(multioutputppo masked_ppo)

# Env and logging
ENV_ID="crossing_planes_multiHead"
LOG_ROOT="logs"

# Tunables (override via env vars when calling)
N_TIMESTEPS="${N_TIMESTEPS:-1000000}"
EVAL_FREQ="${EVAL_FREQ:-50000}"
EVAL_EPISODES="${EVAL_EPISODES:-60}"
N_EVAL_ENVS="${N_EVAL_ENVS:-6}"

ITERATIONS=${#SEEDS[@]}
RUN_IDX=0

for SEED in "${SEEDS[@]}"; do
  RUN_IDX=$((RUN_IDX + 1))
  echo "----------------------------------------------------------"
  echo "Crossing: Lauf ${RUN_IDX} von ${ITERATIONS} mit Seed: ${SEED}"
  echo "----------------------------------------------------------"

  for ALGO in "${ALGOS[@]}"; do
    echo "Starte ${ALGO}..."
    python train.py \
      --algo "${ALGO}" \
      --verbose 1 \
      --env "${ENV_ID}" \
      --vec-env subproc \
      --device cpu \
      --num-threads 4 \
      --tensorboard-log "${LOG_ROOT}/tensorboard" \
      -P --eval-episodes "${EVAL_EPISODES}" --n-eval-envs "${N_EVAL_ENVS}" \
      -n "${N_TIMESTEPS}" --eval-freq "${EVAL_FREQ}" \
      --seed "${SEED}"
  done

done

# Collect evaluations into a CSV for analysis
python scripts/collect_evals.py --logs "${LOG_ROOT}" --out results/eval_runs.csv

echo "Fertig. Ergebnisse in results/eval_runs.csv."
