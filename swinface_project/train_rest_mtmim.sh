#!/bin/bash
pip install torch
pip install tensorboard

# 安装必要的依赖
pip install timm
pip install torchvision
pip install easydict
pip install scipy
pip install scikit-learn

# 创建output目录（如果不存在）
OUTPUT_DIR="output_rest"
mkdir -p $OUTPUT_DIR

# 训练输出文件名（可以修改），保存到output_rest文件夹
OUTPUT_FILE=${1:-"$OUTPUT_DIR/train_rest_mtmim_$(date +%Y%m%d_%H%M%S).out"}

# 如果用户只提供了文件名（不包含路径），则添加到output_rest目录
if [[ "$OUTPUT_FILE" != *"/"* ]]; then
    OUTPUT_FILE="$OUTPUT_DIR/$OUTPUT_FILE"
fi

# 使用4个GPU进行分布式训练，输出保存到.out文件
# 注意：使用12589端口，避免与train.py（12584）和train_rest.py（12585）冲突
CUDA_VISIBLE_DEVICES=0,1,2,3 python -m torch.distributed.launch \
    --nproc_per_node=4 \
    --nnodes=1 \
    --node_rank=0 \
    --master_addr="127.0.0.1" \
    --master_port=12589 \
    train_rest_mtmim.py configs/config_train.py 2>&1 | tee $OUTPUT_FILE

