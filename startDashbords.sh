#/bin/bash
tensorboard --logdir logs/tensorboard/ &
optuna-dashboard sqlite:///logs/mergeenv_optuna.db &