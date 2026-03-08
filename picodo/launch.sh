#!/bin/bash
# Launch a wandb sweep with one agent per GPU.
# Usage:
#   1. Create the sweep:  wandb sweep -p <project> sweeps/chess.yaml
#   2. Set SWEEP_ID below to the returned sweep ID
#   3. Run this script

cd ~/sophistication/picodo

export SWEEP_ID=<your-sweep-id>
ALLOWED_GPUs="0 1 2 3 4 5"
for gpu in $ALLOWED_GPUs; do
  CUDA_VISIBLE_DEVICES=$gpu \
  wandb agent $SWEEP_ID &
  sleep 30
done
