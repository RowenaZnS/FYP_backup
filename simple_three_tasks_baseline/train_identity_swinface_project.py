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
        init_method="tcp://127.0.0.1:29541",
        rank=0,
        world_size=1,
    )


def _normalize_config_path(config_arg: str, proj_root: str) -> str:
    # swinface_project/utils/utils_config.py 要求以 "configs/" 开头
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
    p.add_argument("--epochs", type=int, default=1)
    p.add_argument("--lr", type=float, default=5e-4)
    p.add_argument("--wd", type=float, default=0.05)
    p.add_argument("--out_dir", type=str, default="outputs_identity_swinface_project")
    p.add_argument("--no_amp", action="store_true")
    return p.parse_args()


def main():
    args = parse_args()

    # allow importing swinface_project modules
    proj_root = os.path.join(os.path.dirname(__file__), "..", "swinface_project")
    proj_root = os.path.abspath(proj_root)
    sys.path.insert(0, proj_root)

    from utils.utils_config import get_config
    from analysis import get_analysis_train_dataloader

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
        dl = get_analysis_train_dataloader("recognition", cfg, local_rank)

        mcfg = ModelCfg(
            backbone=args.backbone,
            embed_dim=args.embed_dim,
            num_classes=int(cfg.num_classes),
            age_mode="regression",
        )
        model = ThreeTaskModel(mcfg).to(device)
        ce = torch.nn.CrossEntropyLoss()
        opt = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.wd)
        scaler = torch.cuda.amp.GradScaler(enabled=((not args.no_amp) and device.type == "cuda"))

        step = 0
        model.train()
        for epoch in range(args.epochs):
            if hasattr(dl, "sampler") and hasattr(dl.sampler, "set_epoch"):
                dl.sampler.set_epoch(epoch)

            epoch_loss_sum = 0.0
            epoch_batches = 0
            for img, y_id in dl:
                img = img.to(device, non_blocking=True)
                y_id = y_id.to(device, non_blocking=True)

                opt.zero_grad(set_to_none=True)
                with torch.cuda.amp.autocast(enabled=scaler.is_enabled()):
                    out = model(img, y_id=y_id)
                    loss = ce(out["id_logits"], y_id)

                scaler.scale(loss).backward()
                scaler.step(opt)
                scaler.update()

                li = loss.item()
                epoch_loss_sum += li
                epoch_batches += 1
                if step % 50 == 0:
                    acc = accuracy_top1(out["id_logits"].detach(), y_id).item()
                    print(f"epoch {epoch} step {step} loss {li:.4f} acc {acc:.3f}", flush=True)
                    tb.add_scalar("train/loss", li, step)
                    tb.add_scalar("train/acc_top1", acc, step)
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

