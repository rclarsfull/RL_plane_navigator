#!/bin/bash


SEEDS=(666 1337 42 2024 99)

for SEED in "${SEEDS[@]}"
do
  # Generiere einen zufälligen Seed
  # $RANDOM erzeugt eine Zahl zwischen 0 und 32767
  CURRENT_SEED=$SEED
  
  echo "----------------------------------------------------------"
  echo "Lauf $i von $ITERATIONS mit Seed: $CURRENT_SEED"
  echo "----------------------------------------------------------"

  # Erster Command: multioutputppo
  echo "Starte multioutputppo..."
  python train.py \
    --algo multioutputppo \
    --verbose 1 \
    --env crossing_planes_multiHead \
    --vec-env subproc \
    --device cpu \
    --num-threads 4 \
    --tensorboard-log logs/tensorboard \
    -P --eval-episodes 60 --n-eval-envs 6 \
    --seed $CURRENT_SEED

  # Zweiter Command: masked_ppo
  echo "Starte masked_ppo..."
  python train.py \
    --algo masked_ppo \
    --verbose 1 \
    --env crossing_planes_multiHead \
    --vec-env subproc \
    --device cpu \
    --num-threads 4 \
    --tensorboard-log logs/tensorboard \
    -P --eval-episodes 60 --n-eval-envs 6 \
    --seed $CURRENT_SEED

done

echo "Alle $ITERATIONS Durchläufe abgeschlossen."