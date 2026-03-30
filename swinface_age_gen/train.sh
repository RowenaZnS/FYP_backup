#!/bin/bash
pip install torch
pip install tensorboard

# 安装必要的依赖（MXNet已移除，因为使用ImageFolder格式不需要它）
pip install timm
pip install torchvision
pip install easydict
pip install scipy  # 用于读取IMDB的.mat文件
pip install scikit-learn  # 用于验证功能（sklearn）

# 创建output目录（如果不存在）
OUTPUT_DIR="output"
mkdir -p $OUTPUT_DIR

# 训练输出文件名（可以修改），保存到output文件夹
OUTPUT_FILE=${1:-"$OUTPUT_DIR/train_$(date +%Y%m%d_%H%M%S).out"}

# 如果用户只提供了文件名（不包含路径），则添加到output目录
if [[ "$OUTPUT_FILE" != *"/"* ]]; then
    OUTPUT_FILE="$OUTPUT_DIR/$OUTPUT_FILE"
fi

# 使用4个GPU进行分布式训练，输出保存到.out文件
CUDA_VISIBLE_DEVICES=0,1,2,3 python -m torch.distributed.launch \
    --nproc_per_node=4 \
    --nnodes=1 \
    --node_rank=0 \
    --master_addr="127.0.0.1" \
    --master_port=12581 \
    train.py configs/config_train 2>&1 | tee $OUTPUT_FILE