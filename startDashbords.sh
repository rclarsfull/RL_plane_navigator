#!/bin/bash

# Trap SIGINT and SIGTERM to kill background processes
trap 'kill $(jobs -p) 2>/dev/null' EXIT INT TERM

tensorboard --logdir logs/tensorboard/ &
optuna-dashboard sqlite:///logs/mergeenv_optuna.db &

# Wait for all background processes
wait