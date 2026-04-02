#!/usr/bin/env bash
# 不必固定 4 卡：1 卡即可训练；多卡仅加快训练、总 batch = age_gender_bz × GPU 数（见 train_age_generation 里 total_batch_size）。
# 控制台输出会写入 config.output/train.out（与 training.log 并存）
cd "$(dirname "$0")"

# 单卡（默认）
NPROC=${NPROC:-1}
export CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-0}

# 与之前 MT-MIM 一样用 4 卡时，例如：
#   NPROC=4 CUDA_VISIBLE_DEVICES=0,1,2,3 bash run_age_gen.sh

torchrun --nproc_per_node="${NPROC}" --master_port=29521 \
  train_age_generation.py configs/config_age_gen.py
