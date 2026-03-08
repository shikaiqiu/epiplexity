#!/bin/bash
# Quick test run with a tiny model.
# Usage: CUDA_VISIBLE_DEVICES=0 bash run.sh

cd ~/sophistication/picodo

CUDA_VISIBLE_DEVICES=5 python main.py -cn chess \
  wandb_mode=disabled \
  train_student=false \
  train_teacher=true \
  model.N=3 \
  model.P=1 \
  ds_path=chess \
  opt.lr=2 \
  B=256 \
  model.L=512 \
  opt.schedule=const \
  opt.warmup_tokens=16384000 \
  T=5000000000 \
  T_eval=1000000 \
  num_evals=50 \
  seed=0 \
  save=false
