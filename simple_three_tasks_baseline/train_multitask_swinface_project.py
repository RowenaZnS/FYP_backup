import argparse
import os
import sys
from itertools import cycle

import torch
import torch.distributed as dist


def _init_dist():
    if dist.is_available() and dist.is_initialized():
        return
    backend = "nccl" if torch.cuda.is_available() else "gloo"
    dist.init_process_group(
        backend=backend,
        init_method="tcp://127.0.0.1:29543",
        rank=0,
        world_size=1,
    )


def _normalize_config_path(config_arg: str, proj_root: str) -> str:
    if config_arg.startswith("configs/"):
        return config_arg
    abs_p = os.path.abspath(config_arg)
    abs_configs = os.path.abspath(os.path.join(proj_root, "configs")) + os.sep
    if abs_p.startswith(abs_configs):
        rel = os.path.relpath(abs_p, proj_root).replace(os.sep, "/")
        if rel.startswith("configs/"):
            return rel
    raise ValueError(f"Config must be like configs/xxx.py (got: {config_arg})")


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("config", type=str, help="swinface_project config py, e.g. configs/config_train.py")
    p.add_argument("--backbone", type=str, default="resnet50", help="timm backbone name")
    p.add_argument("--embed_dim", type=int, default=512)
    p.add_argument("--age_mode", type=str, default="regression", choices=["regression", "classification"])
    p.add_argument("--age_bins", type=int, default=101)
    p.add_argument("--epochs", type=int, default=1)
    p.add_argument("--lr", type=float, default=5e-4)
    p.add_argument("--wd", type=float, default=0.05)
    p.add_argument("--out_dir", type=str, default="outputs_multitask_swinface_project")
    p.add_argument("--no_amp", action="store_true")

    p.add_argument("--lambda_id", type=float, default=1.0)
    p.add_argument("--lambda_age", type=float, default=1.0)
    p.add_argument("--lambda_mix", type=float, default=1.0)
    p.add_argument("--lambda_sep", type=float, default=0.1)
    return p.parse_args()


def main():
    args = parse_args()

    proj_root = os.path.join(os.path.dirname(__file__), "..", "swinface_project")
    proj_root = os.path.abspath(proj_root)
    sys.path.insert(0, proj_root)

    from utils.utils_config import get_config
    from analysis import get_analysis_train_dataloader

    from src.losses import AgeLoss, CosineDecorrelationLoss
    from src.models import ThreeTaskModel, ModelCfg
    from src.utils import accuracy_top1, ensure_dir, make_tb_writer, set_seed

    _init_dist()
    tb = None
    try:
        cfg = get_config(_normalize_config_path(args.config, proj_root))
        set_seed(int(getattr(cfg, "seed", 42)))

        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        ensure_dir(args.out_dir)
        tb = make_tb_writer(args.out_dir)

        local_rank = 0
        dl_rec = get_analysis_train_dataloader("recognition", cfg, local_rank)
        dl_age = get_analysis_train_dataloader("age_gender", cfg, local_rank)

        mcfg = ModelCfg(
            backbone=args.backbone,
            embed_dim=args.embed_dim,
            num_classes=int(cfg.num_classes),
            age_mode=args.age_mode,
            age_bins=args.age_bins,
        )
        model = ThreeTaskModel(mcfg).to(device)

        ce = torch.nn.CrossEntropyLoss()
        age_loss_fn = AgeLoss(mode=args.age_mode, age_bins=args.age_bins)
        sep_loss_fn = CosineDecorrelationLoss()

        opt = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.wd)
        scaler = torch.cuda.amp.GradScaler(enabled=((not args.no_amp) and device.type == "cuda"))

        step = 0
        model.train()
        for epoch in range(args.epochs):
            if hasattr(dl_rec, "sampler") and hasattr(dl_rec.sampler, "set_epoch"):
                dl_rec.sampler.set_epoch(epoch)
            if hasattr(dl_age, "sampler") and hasattr(dl_age.sampler, "set_epoch"):
                dl_age.sampler.set_epoch(epoch)

            epoch_loss_sum = 0.0
            epoch_batches = 0
            max_len = max(len(dl_rec), len(dl_age))
            it_rec = cycle(dl_rec) if len(dl_rec) < max_len else iter(dl_rec)
            it_age = cycle(dl_age) if len(dl_age) < max_len else iter(dl_age)

            for _ in range(max_len):
                img_r, y_id = next(it_rec)
                img_a, label_a = next(it_age)  # label_a = [age, gender]

                img_r = img_r.to(device, non_blocking=True)
                y_id = y_id.to(device, non_blocking=True)
                img_a = img_a.to(device, non_blocking=True)
                y_age = label_a[0].to(device, non_blocking=True).view(-1).float()

                if args.age_mode == "classification":
                    y_age = y_age.clamp(0, args.age_bins - 1).long()

                img = torch.cat([img_r, img_a], dim=0)
                n_rec = img_r.size(0)

                opt.zero_grad(set_to_none=True)
                with torch.cuda.amp.autocast(enabled=scaler.is_enabled()):
                    out = model.forward_embeddings(img)

                    # --- losses on recognition part only ---
                    z_id_r = out["z_id"][:n_rec]
                    z_mix_r = out["z_mix"][:n_rec]
                    id_logits = model.id_logits(z_id_r, y_id)
                    mix_logits = model.mix_logits(z_mix_r, y_id)
                    lid = ce(id_logits, y_id)
                    lmix = ce(mix_logits, y_id)

                    # --- age loss on age part only ---
                    age_pred_a = out["age_pred"][n_rec:]
                    lage = age_loss_fn(age_pred_a, y_age)

                    # --- separation on whole batch (simple) ---
                    lsep = sep_loss_fn(out["z_id"], out["z_age"])

                    loss = (
                        args.lambda_id * lid
                        + args.lambda_mix * lmix
                        + args.lambda_age * lage
                        + args.lambda_sep * lsep
                    )

                scaler.scale(loss).backward()
                scaler.step(opt)
                scaler.update()

                li = loss.item()
                epoch_loss_sum += li
                epoch_batches += 1
                if step % 50 == 0:
                    acc = accuracy_top1(id_logits.detach(), y_id).item()
                    acc_mix = accuracy_top1(mix_logits.detach(), y_id).item()
                    print(
                        f"epoch {epoch} step {step} loss {li:.4f} "
                        f"lid {lid.item():.4f} lage {lage.item():.4f} "
                        f"lmix {lmix.item():.4f} lsep {lsep.item():.4f} "
                        f"acc {acc:.3f} acc_mix {acc_mix:.3f}",
                        flush=True,
                    )
                    tb.add_scalar("train/loss", li, step)
                    tb.add_scalar("train/lid", lid.item(), step)
                    tb.add_scalar("train/lage", lage.item(), step)
                    tb.add_scalar("train/lmix", lmix.item(), step)
                    tb.add_scalar("train/lsep", lsep.item(), step)
                    tb.add_scalar("train/acc_top1", acc, step)
                    tb.add_scalar("train/acc_mix_top1", acc_mix, step)

                step += 1

            if epoch_batches > 0:
                tb.add_scalar("train/loss_epoch", epoch_loss_sum / epoch_batches, epoch)

            torch.save(
                {"epoch": epoch, "model": model.state_dict(), "opt": opt.state_dict(), "cfg": vars(args)},
                os.path.join(args.out_dir, f"ckpt_epoch_{epoch}.pt"),
            )
    finally:
        if tb is not None:
            tb.close()
        if dist.is_available() and dist.is_initialized():
            dist.destroy_process_group()


if __name__ == "__main__":
    main()

