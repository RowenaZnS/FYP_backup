#!/usr/bin/env bash
# MT-MIM 年龄生成 + Gender / CelebA / Expression + MI（无年龄监督 loss）
cd "$(dirname "$0")"
NPROC=${NPROC:-1}
export CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-0}
# 四卡示例: NPROC=4 CUDA_VISIBLE_DEVICES=0,1,2,3 bash run_age_gen_multitask.sh

torchrun --nproc_per_node="${NPROC}" --master_port=29522 \
  train_age_gen_multitask.py configs/config_age_gen_multitask.py
