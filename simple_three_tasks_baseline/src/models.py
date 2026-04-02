from dataclasses import dataclass
from typing import Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F

try:
    import timm
except Exception:  # pragma: no cover
    timm = None

from .losses import ArcFace, ArcFaceCfg


@dataclass
class ModelCfg:
    backbone: str = "resnet50"
    embed_dim: int = 512
    num_classes: int = 1000
    age_mode: str = "regression"  # regression | classification
    age_bins: int = 101
    arcface_s: float = 64.0
    arcface_m: float = 0.5


def _build_backbone(name: str) -> Tuple[nn.Module, int]:
    """
    返回: (backbone, feat_dim)
    backbone 输出一个 [B, feat_dim] 的全局特征
    """
    if timm is None:
        raise ImportError("Please install timm: pip install timm")

    m = timm.create_model(name, pretrained=True, num_classes=0, global_pool="avg")
    feat_dim = m.num_features
    return m, feat_dim


class ThreeTaskModel(nn.Module):
    """
    共享 backbone + 三分支:
      - z_id -> ArcFace 分类
      - z_age -> age predictor
      - z_mix = proj_mix([z_id, z_age]) -> ArcFace 分类
    """

    def __init__(self, cfg: ModelCfg):
        super().__init__()
        self.cfg = cfg
        self.backbone, feat_dim = _build_backbone(cfg.backbone)

        self.proj_id = nn.Sequential(
            nn.Linear(feat_dim, cfg.embed_dim),
            nn.BatchNorm1d(cfg.embed_dim),
        )
        self.proj_age = nn.Sequential(
            nn.Linear(feat_dim, cfg.embed_dim),
            nn.BatchNorm1d(cfg.embed_dim),
        )
        self.proj_mix = nn.Sequential(
            nn.Linear(cfg.embed_dim * 2, cfg.embed_dim),
            nn.ReLU(inplace=True),
            nn.BatchNorm1d(cfg.embed_dim),
        )

        arc_cfg = ArcFaceCfg(s=cfg.arcface_s, m=cfg.arcface_m, easy_margin=False)
        self.arc_id = ArcFace(cfg.embed_dim, cfg.num_classes, arc_cfg)
        self.arc_mix = ArcFace(cfg.embed_dim, cfg.num_classes, arc_cfg)

        if cfg.age_mode == "regression":
            self.age_head = nn.Linear(cfg.embed_dim, 1)
        else:
            self.age_head = nn.Linear(cfg.embed_dim, cfg.age_bins)

    def forward_embeddings(self, x: torch.Tensor):
        f = self.backbone(x)  # [B, feat_dim]
        z_id = self.proj_id(f)
        z_age = self.proj_age(f)

        z_mix = self.proj_mix(torch.cat([z_id, z_age], dim=1))

        age_pred = self.age_head(z_age)
        if self.cfg.age_mode == "regression":
            age_pred = age_pred.view(-1)

        return {"z_id": z_id, "z_age": z_age, "z_mix": z_mix, "age_pred": age_pred}

    def id_logits(self, z_id: torch.Tensor, y_id: torch.Tensor) -> torch.Tensor:
        return self.arc_id(z_id, y_id)

    def mix_logits(self, z_mix: torch.Tensor, y_id: torch.Tensor) -> torch.Tensor:
        return self.arc_mix(z_mix, y_id)

    def forward(self, x: torch.Tensor, y_id: Optional[torch.Tensor] = None):
        out = self.forward_embeddings(x)
        z_id, z_age, z_mix, age_pred = out["z_id"], out["z_age"], out["z_mix"], out["age_pred"]

        if y_id is None:
            return {
                "z_id": F.normalize(z_id, dim=1),
                "z_age": z_age,
                "z_mix": F.normalize(z_mix, dim=1),
                "age_pred": age_pred,
            }

        id_logits = self.id_logits(z_id, y_id)
        mix_logits = self.mix_logits(z_mix, y_id)
        return {
            "z_id": z_id,
            "z_age": z_age,
            "z_mix": z_mix,
            "id_logits": id_logits,
            "mix_logits": mix_logits,
            "age_pred": age_pred,
        }

