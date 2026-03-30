"""
MT-MIM + target-age conditional generation.

Reuses Swin backbone + FAM + TSS; adds age-factor extractor, fusion, generator, D, age head.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F

from analysis.age_extractor import AgeFactorExtractor
from age_gen.modules import (
    TargetAgeEncoder,
    AttentionFusion,
    FaceGenerator,
    PatchDiscriminator,
    VGGPerceptualLoss,
)
from model import build_model


def _pool_features(x, feature_mode: str):
    if feature_mode == "all":
        return torch.cat([x[0], x[1]], dim=1)
    if feature_mode == "global":
        return x[1]
    return x[0]


class MTMIMAgeGenerationModel(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        base = build_model(cfg)
        self.backbone = base.backbone
        self.fam = base.fam
        self.tss = base.tss
        self.feature = cfg.fam_feature
        self.om = base.om if getattr(cfg, "use_analysis_heads", False) else None
        if self.om is not None:
            self.om.set_output_type("List")
        dim = cfg.embedding_size

        self.age_extractor = AgeFactorExtractor(
            input_dim=dim,
            hidden_dim=getattr(cfg, "age_extractor_hidden_dim", 256),
            output_dim=dim,
        )
        self.target_age_encoder = TargetAgeEncoder(
            hidden_dim=getattr(cfg, "target_age_hidden_dim", 256),
            out_dim=getattr(cfg, "age_code_dim", 128),
        )
        dim_age = getattr(cfg, "age_code_dim", 128)
        self.fusion = AttentionFusion(
            dim_id=dim,
            dim_age=dim_age,
            latent_hw=getattr(cfg, "gen_latent_hw", 7),
            num_heads=getattr(cfg, "fusion_num_heads", 8),
        )
        self.generator = FaceGenerator(in_ch=dim, out_size=cfg.img_size)

        self.age_head = nn.Sequential(
            nn.Linear(dim, dim // 2),
            nn.ReLU(inplace=True),
            nn.Linear(dim // 2, 1),
        )
        self.discriminator = PatchDiscriminator()

        self.use_vgg = getattr(cfg, "use_vgg_perceptual", True)
        if self.use_vgg:
            self.vgg_loss = VGGPerceptualLoss()
        else:
            self.vgg_loss = None

        self.max_age = float(getattr(cfg, "age_norm_max", 100.0))

    def _feats(self, x: torch.Tensor):
        local_features, global_features, embedding = self.backbone(x)
        feat = _pool_features((local_features, global_features), self.feature)
        tss_out = self.tss(self.fam(feat))
        mixed = tss_out.mean(dim=0)
        return mixed, tss_out, embedding

    def encode_mixed(self, x: torch.Tensor) -> torch.Tensor:
        mixed, _, _ = self._feats(x)
        return mixed

    def forward_mtmim(self, x: torch.Tensor):
        mixed, _, _ = self._feats(x)
        x_age = self.age_extractor(mixed)
        x_id = mixed - x_age
        return mixed, x_age, x_id

    def forward(self, x: torch.Tensor, target_age: torch.Tensor, encode_only: bool = False):
        """
        target_age: years (same scale as dataset labels), shape (B,)
        encode_only: if True, only encode x (for x_hat path); must still go through DDP forward.
        """
        if encode_only:
            mixed, x_age, x_id = self.forward_mtmim(x)
            age_pred = self.age_predict(mixed)
            return {
                "mixed": mixed,
                "x_age": x_age,
                "x_id": x_id,
                "age_pred": age_pred,
            }
        mixed, tss_out, embedding = self._feats(x)
        x_age = self.age_extractor(mixed)
        x_id = mixed - x_age
        age_norm = (target_age / self.max_age).clamp(0.0, 1.0)
        z_age = self.target_age_encoder(age_norm)
        gen_in, attn_mask = self.fusion(x_id, z_age)
        x_hat = self.generator(gen_in)
        out = {
            "x_hat": x_hat,
            "mixed": mixed,
            "x_age": x_age,
            "x_id": x_id,
            "attn_mask": attn_mask,
        }
        if self.om is not None:
            self.om.set_output_type("List")
            out["standard_list"] = self.om(tss_out, embedding)
        return out

    def discriminate(self, x: torch.Tensor) -> torch.Tensor:
        return self.discriminator(x)

    def age_predict(self, mixed: torch.Tensor) -> torch.Tensor:
        return self.age_head(mixed).squeeze(-1)

    def perceptual_loss(self, x: torch.Tensor, x_hat: torch.Tensor):
        if self.vgg_loss is None:
            return x_hat.new_tensor(0.0)
        return self.vgg_loss(x_hat, x)


def identity_cosine_loss(x_id_a: torch.Tensor, x_id_b: torch.Tensor) -> torch.Tensor:
    a = F.normalize(x_id_a, dim=-1)
    b = F.normalize(x_id_b, dim=-1)
    return (1.0 - (a * b).sum(dim=-1)).mean()
