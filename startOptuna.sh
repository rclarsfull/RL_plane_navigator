#/bin/bash
python train.py  \
  --algo ppo \
  --verbose 1 \
  --env crossing_planes \
  --vec-env subproc \
  --device cpu \
  --num-threads 4 \
  --optimize \
  --n-trials 50 \
  --n-evaluations 50 \
  --storage sqlite:///logs/mergeenv_optuna.db \
  -f logs/ \
  --study-name Study1 \