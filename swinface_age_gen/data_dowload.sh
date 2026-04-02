#!/bin/bash
set -e  # 出错时停止

echo "=== SwinFace数据集下载脚本 ==="

# 配置数据集列表
DATASETS=(
    "raf-db:shuvoalok/raf-db-dataset:raf-db-dataset.zip"
    "morph:chiragsaipanuganti/morph:morph.zip"
    "celeba:jessicali9530/celeba-dataset:celeba-dataset.zip"
    "affectnet:mstjebashazida/affectnet:affectnet.zip"
    "imdb-wiki:abhikjha/imdb-wiki-faces-dataset:imdb-wiki-faces-dataset.zip"
    "adiencegender:alfredhhw/adiencegender:adiencegender.zip"
    "ms1m-arcface:yakhyokhuja/ms1m-arcface-dataset:ms1m-arcface-dataset.zip"
)

# 安装kaggle
echo "安装kaggle API..."
pip install kaggle --upgrade

# 设置Kaggle API凭证
export KAGGLE_USERNAME="rowenaliu"
export KAGGLE_KEY="64b1abff7c5f70f2d7bd7e965aae363b"

# 创建主目录
echo "创建主目录..."
mkdir -p dataset
cd dataset

# 下载每个数据集
for item in "${DATASETS[@]}"; do
    IFS=':' read -r folder_name dataset_id expected_zip <<< "$item"
    
    echo "处理数据集: $folder_name"
    
    # 检查是否已存在
    if [ -d "$folder_name" ] && [ -n "$(ls -A "$folder_name" 2>/dev/null)" ]; then
        echo "  目录已存在且非空，跳过..."
        continue
    fi
    
    # 创建目录
    mkdir -p "$folder_name"
    cd "$folder_name"
    
    # 下载
    echo "  正在下载..."
    if kaggle datasets download -d "$dataset_id"; then
        echo "  下载成功"
    else
        echo "  下载失败，请检查:"
        echo "  1. Kaggle API Key是否正确"
        echo "  2. 数据集是否需要接受条款"
        echo "  3. 网络连接"
        cd ..
        continue
    fi
    
    # 解压（使用通配符）
    echo "  正在解压..."
    for zipfile in *.zip; do
        if [ -f "$zipfile" ]; then
            echo "    解压: $zipfile"
            unzip -q "$zipfile"
        fi
    done
    
    cd ..
    echo "  完成: $folder_name"
    echo ""
done

# 验证结构
echo "=== 下载完成 ==="
echo "目录结构验证:"
pwd
echo ""
echo "所有平行目录:"
ls -d */
echo ""
echo "各目录大小:"
du -sh */