from dataclasses import dataclass
from typing import Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F


class CosineDecorrelationLoss(nn.Module):
    """
    让 z_id 与 z_age 的 cosine 相似度接近 0。
    z_id: [B, D], z_age: [B, D]（不强制同 D，但建议相同）
    """

    def forward(self, z_id: torch.Tensor, z_age: torch.Tensor) -> torch.Tensor:
        z_id = F.normalize(z_id, dim=1)
        z_age = F.normalize(z_age, dim=1)
        cos = (z_id * z_age).sum(dim=1).abs()  # |cos|
        return cos.mean()


@dataclass
class ArcFaceCfg:
    s: float = 64.0
    m: float = 0.5
    easy_margin: bool = False


class ArcFace(nn.Module):
    """
    最小 ArcFace head（用于分类训练）。
    输入 embedding 必须已归一化更稳（这里内部会 normalize 一次）。
    """

    def __init__(self, in_features: int, num_classes: int, cfg: ArcFaceCfg = ArcFaceCfg()):
        super().__init__()
        self.in_features = in_features
        self.num_classes = num_classes
        self.s = cfg.s
        self.m = cfg.m
        self.easy_margin = cfg.easy_margin

        self.weight = nn.Parameter(torch.empty(num_classes, in_features))
        nn.init.xavier_uniform_(self.weight)

        self.cos_m = torch.cos(torch.tensor(self.m))
        self.sin_m = torch.sin(torch.tensor(self.m))
        self.th = torch.cos(torch.tensor(torch.pi) - torch.tensor(self.m))
        self.mm = torch.sin(torch.tensor(torch.pi) - torch.tensor(self.m)) * torch.tensor(self.m)

    def forward(self, emb: torch.Tensor, label: torch.Tensor) -> torch.Tensor:
        emb = F.normalize(emb, dim=1)
        W = F.normalize(self.weight, dim=1)
        cosine = F.linear(emb, W).clamp(-1.0, 1.0)  # [B, C]
        sine = torch.sqrt(torch.clamp(1.0 - cosine * cosine, min=1e-9))
        phi = cosine * self.cos_m - sine * self.sin_m

        if self.easy_margin:
            phi = torch.where(cosine > 0, phi, cosine)
        else:
            phi = torch.where(cosine > self.th.to(cosine.device), phi, cosine - self.mm.to(cosine.device))

        one_hot = torch.zeros_like(cosine)
        one_hot.scatter_(1, label.view(-1, 1), 1.0)

        logits = (one_hot * phi) + ((1.0 - one_hot) * cosine)
        logits *= self.s
        return logits


class AgeLoss(nn.Module):
    """
    两种模式：
      - regression: L1
      - classification: CE（预测 0..age_bins-1）
    """

    def __init__(self, mode: str = "regression", age_bins: int = 101):
        super().__init__()
        assert mode in ("regression", "classification")
        self.mode = mode
        self.age_bins = age_bins
        self.ce = nn.CrossEntropyLoss()

    def forward(self, pred, target_age: torch.Tensor) -> torch.Tensor:
        if self.mode == "regression":
            return F.l1_loss(pred.view(-1), target_age.view(-1))
        # classification: target_age 需要是 long 的 bin index
        return self.ce(pred, target_age.long())


class MultiTaskLoss(nn.Module):
    """
    汇总：L_id + L_age + L_mix + L_sep
    """

    def __init__(
        self,
        lambda_id: float = 1.0,
        lambda_age: float = 1.0,
        lambda_mix: float = 1.0,
        lambda_sep: float = 0.1,
        age_mode: str = "regression",
        age_bins: int = 101,
    ):
        super().__init__()
        self.lambda_id = lambda_id
        self.lambda_age = lambda_age
        self.lambda_mix = lambda_mix
        self.lambda_sep = lambda_sep

        self.age_loss = AgeLoss(mode=age_mode, age_bins=age_bins)
        self.sep_loss = CosineDecorrelationLoss()
        self.ce = nn.CrossEntropyLoss()

    def forward(
        self,
        id_logits: torch.Tensor,
        mix_logits: torch.Tensor,
        age_pred: torch.Tensor,
        z_id: torch.Tensor,
        z_age: torch.Tensor,
        y_id: torch.Tensor,
        y_age: torch.Tensor,
    ) -> Tuple[torch.Tensor, dict]:
        lid = self.ce(id_logits, y_id)
        lmix = self.ce(mix_logits, y_id)
        lage = self.age_loss(age_pred, y_age)
        lsep = self.sep_loss(z_id, z_age)
        total = self.lambda_id * lid + self.lambda_mix * lmix + self.lambda_age * lage + self.lambda_sep * lsep
        return total, {"lid": lid.detach(), "lmix": lmix.detach(), "lage": lage.detach(), "lsep": lsep.detach()}

