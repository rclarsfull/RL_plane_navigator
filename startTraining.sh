#/bin/bash
python train_sbx.py  \
  --algo ppo \
  --verbose 1 \
  --env crossing_planes \
  --vec-env subproc \
  --device cpu \
  --num-threads 4 \
  --tensorboard-log logs/tensorboard  \
  -P