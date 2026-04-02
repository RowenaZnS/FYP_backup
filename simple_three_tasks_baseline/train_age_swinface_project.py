import argparse
import os
import sys

import torch
import torch.distributed as dist


def _init_dist():
    if dist.is_available() and dist.is_initialized():
        return
    backend = "nccl" if torch.cuda.is_available() else "gloo"
    dist.init_process_group(
        backend=backend,
        init_method="tcp://127.0.0.1:29542",
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
    p.add_argument("--out_dir", type=str, default="outputs_age_swinface_project")
    p.add_argument("--no_amp", action="store_true")
    return p.parse_args()


def main():
    args = parse_args()

    proj_root = os.path.join(os.path.dirname(__file__), "..", "swinface_project")
    proj_root = os.path.abspath(proj_root)
    sys.path.insert(0, proj_root)

    from utils.utils_config import get_config
    from analysis import get_analysis_train_dataloader

    from src.losses import AgeLoss
    from src.models import ThreeTaskModel, ModelCfg
    from src.utils import ensure_dir, make_tb_writer, set_seed

    _init_dist()
    tb = None
    try:
        cfg = get_config(_normalize_config_path(args.config, proj_root))
        set_seed(int(getattr(cfg, "seed", 42)))

        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        ensure_dir(args.out_dir)
        tb = make_tb_writer(args.out_dir)

        local_rank = 0
        dl = get_analysis_train_dataloader("age_gender", cfg, local_rank)

        mcfg = ModelCfg(
            backbone=args.backbone,
            embed_dim=args.embed_dim,
            num_classes=1,
            age_mode=args.age_mode,
            age_bins=args.age_bins,
        )
        model = ThreeTaskModel(mcfg).to(device)
        loss_fn = AgeLoss(mode=args.age_mode, age_bins=args.age_bins)

        opt = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.wd)
        scaler = torch.cuda.amp.GradScaler(enabled=((not args.no_amp) and device.type == "cuda"))

        step = 0
        model.train()
        for epoch in range(args.epochs):
            if hasattr(dl, "sampler") and hasattr(dl.sampler, "set_epoch"):
                dl.sampler.set_epoch(epoch)

            epoch_loss_sum = 0.0
            epoch_batches = 0
            for img, label in dl:
                # label = [age_label, gender_label]
                age = label[0].to(device, non_blocking=True).view(-1).float()
                img = img.to(device, non_blocking=True)

                if args.age_mode == "classification":
                    age = age.clamp(0, args.age_bins - 1).long()

                opt.zero_grad(set_to_none=True)
                with torch.cuda.amp.autocast(enabled=scaler.is_enabled()):
                    out = model(img, y_id=None)
                    pred = out["age_pred"]
                    loss = loss_fn(pred, age)

                scaler.scale(loss).backward()
                scaler.step(opt)
                scaler.update()

                li = loss.item()
                epoch_loss_sum += li
                epoch_batches += 1
                if step % 50 == 0:
                    print(f"epoch {epoch} step {step} loss {li:.4f}", flush=True)
                    tb.add_scalar("train/loss", li, step)
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

