#!/bin/bash

set -euo pipefail

#66 133 42 202 7 1234
SEEDS=(999) #66 133 42 202 7 1234 999 2021 31415 2718 1618 8675 11235 3141 16180
ALGOS=(multioutputppo masked_ppo masked_ppo_split_net masked_ppo_split_net_with_shared) #ppo multioutputppo masked_hybrid_ppo masked_hybrid_ppo_split_net masked_hybrid_ppo_split_net_full masked_hybrid_ppo_shared_split    #multioutputppo masked_ppo masked_ppo_split_net

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
    echo "Starte ${ALGO}..."
    python train.py \
      --algo "${ALGO}" \
      --verbose 0 \
      --env "${ENV_ID}" \
      --vec-env subproc \
      --device cpu \
      --num-threads 4 \
      --tensorboard-log "${LOG_ROOT}/tensorboard" \
      -P --eval-episodes "${EVAL_EPISODES}" --n-eval-envs "${N_EVAL_ENVS}" \
      --eval-freq "${EVAL_FREQ}" \
      --seed "${SEED}"
  done

done

python scripts/evaluate_results.py --logs "${LOG_ROOT}" --env "${ENV_ID}" --algos "${ALGOS[@]}" --labels "PPO" "Masked PPO" "Masked split_net PPO" "Masked shared & split_net PPO" 

#python scripts/evaluate_results.py --logs logs --env crossing_planes_multiHead --algos multioutputppo masked_ppo masked_ppo_split_net masked_ppo_split_net_with_shared --labels "PPO" "Masked PPO" "Masked split_net PPO" "Masked shared & split_net PPO" 