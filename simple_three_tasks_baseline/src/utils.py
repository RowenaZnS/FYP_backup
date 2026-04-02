import os
import random
from dataclasses import dataclass
from typing import Optional

import numpy as np
import torch

try:
    from torch.utils.tensorboard import SummaryWriter
except ImportError:  # pragma: no cover
    SummaryWriter = None  # type: ignore[misc, assignment]


def set_seed(seed: int = 42) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


@dataclass
class TrainCfg:
    # data
    train_csv: Optional[str] = None
    train_root: Optional[str] = None
    image_size: int = 224

    # model
    backbone: str = "resnet50"  # resnet50 / efficientnet_b0 / vit_base_patch16_224
    embed_dim: int = 512
    num_classes: int = 1000  # identity classes
    age_bins: int = 101  # if using classification; 0..100

    # optimization
    seed: int = 42
    batch_size: int = 64
    num_workers: int = 4
    lr: float = 1e-3
    wd: float = 1e-4
    epochs: int = 10
    amp: bool = True
    device: str = "cuda"

    # losses
    lambda_id: float = 1.0
    lambda_age: float = 1.0
    lambda_mix: float = 1.0
    lambda_sep: float = 0.1

    # misc
    out_dir: str = "outputs"


def ensure_dir(path: str) -> None:
    os.makedirs(path, exist_ok=True)


def make_tb_writer(out_dir: str, flush_secs: int = 10) -> "SummaryWriter":
    """
    TensorBoard 日志目录：<out_dir>/tensorboard/
    查看：tensorboard --logdir <out_dir>/tensorboard
    需安装：pip install tensorboard
    """
    log_dir = os.path.join(out_dir, "tensorboard")
    ensure_dir(log_dir)
    if SummaryWriter is None:
        raise ImportError(
            "需要 TensorBoard 才能写入训练曲线。请执行: pip install tensorboard"
        )
    return SummaryWriter(log_dir=log_dir, flush_secs=flush_secs)


def accuracy_top1(logits: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    pred = logits.argmax(dim=1)
    return (pred == target).float().mean()

