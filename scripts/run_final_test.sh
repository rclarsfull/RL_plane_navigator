#!/bin/bash

set -euo pipefail

#66 133 42 202 7 1234 999
SEEDS=(66 133 42 202 7 1234 999 2021) #66 133 42 202 7 1234 999 2021 31415 2718 1618 8675 11235 3141 16180
ALGOS=(ppo masked_ppo_split_net_with_shared) # ppo masked_ppo_split_net_with_shared

ENV_ID="crossing_planes_multiHead" #crossing_planes_multiHead
LOG_ROOT="logs"

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
    # PPO verwendet immer crossing_planes, andere Algos verwenden ENV_ID
    if [ "${ALGO}" = "ppo" ]; then
      CURRENT_ENV="crossing_planes"
    else
      CURRENT_ENV="${ENV_ID}"
    fi
    
    echo "Starte ${ALGO} mit Environment ${CURRENT_ENV}..."
    python train.py \
      --algo "${ALGO}" \
      --verbose 0 \
      --env "${CURRENT_ENV}" \
      --vec-env subproc \
      --device cpu \
      --num-threads 4 \
      --tensorboard-log "${LOG_ROOT}/tensorboard" \
      -P --eval-episodes "${EVAL_EPISODES}" --n-eval-envs "${N_EVAL_ENVS}" \
      --eval-freq "${EVAL_FREQ}" \
      --seed "${SEED}"
  done

done

python scripts/evaluate_results.py --logs "${LOG_ROOT}" --env "${ENV_ID}" --algos "${ALGOS[@]}" --labels "PPO" "Masked shared & split_net PPO" 
