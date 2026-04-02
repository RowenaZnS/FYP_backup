#!/bin/bash

# SwinFace 训练脚本
# 使用方法: bash run_train.sh [GPU数量] [输出文件名]
# 例如: bash run_train.sh 1 train.out

# 设置默认值
NUM_GPUS=${1:-1}  # 默认使用1个GPU
OUTPUT_FILE=${2:-"train_$(date +%Y%m%d_%H%M%S).out"}  # 默认输出文件名

# 配置文件路径
CONFIG_FILE="configs/config_train.py"

# 检查配置文件是否存在
if [ ! -f "$CONFIG_FILE" ]; then
    echo "错误: 配置文件不存在: $CONFIG_FILE"
    exit 1
fi

# 设置输出目录（在配置文件中也需要设置）
OUTPUT_DIR="/workspace/SwinFace/swinface_project/output"
mkdir -p $OUTPUT_DIR

echo "=========================================="
echo "开始训练 SwinFace"
echo "=========================================="
echo "GPU数量: $NUM_GPUS"
echo "配置文件: $CONFIG_FILE"
echo "输出文件: $OUTPUT_FILE"
echo "模型输出目录: $OUTPUT_DIR"
echo "=========================================="

# 根据GPU数量选择运行方式
if [ $NUM_GPUS -eq 1 ]; then
    # 单GPU训练
    echo "使用单GPU训练..."
    CUDA_VISIBLE_DEVICES=0 python train.py $CONFIG_FILE --local_rank=0 2>&1 | tee $OUTPUT_FILE
else
    # 多GPU分布式训练
    echo "使用 $NUM_GPUS 个GPU进行分布式训练..."
    CUDA_VISIBLE_DEVICES=$(seq -s, 0 $((NUM_GPUS-1))) \
    python -m torch.distributed.launch \
        --nproc_per_node=$NUM_GPUS \
        --nnodes=1 \
        --node_rank=0 \
        --master_addr="127.0.0.1" \
        --master_port=12581 \
        train.py $CONFIG_FILE 2>&1 | tee $OUTPUT_FILE
fi

echo "=========================================="
echo "训练完成！"
echo "输出已保存到: $OUTPUT_FILE"
echo "=========================================="

