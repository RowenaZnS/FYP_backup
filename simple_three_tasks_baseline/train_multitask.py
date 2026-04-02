import argparse
import os
from pathlib import Path

import torch
from torch.utils.data import DataLoader

from src.datasets import FaceIdAgeCsvDataset
from src.losses import MultiTaskLoss
from src.models import ThreeTaskModel, ModelCfg
from src.utils import TrainCfg, accuracy_top1, ensure_dir, make_tb_writer, set_seed


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--train_csv", type=str, required=True, help="CSV with columns: path,id,age")
    p.add_argument("--train_root", type=str, default=None, help="prefix root for relative paths")
    p.add_argument("--backbone", type=str, default="resnet50")
    p.add_argument("--num_classes", type=int, required=True)
    p.add_argument("--age_mode", type=str, default="regression", choices=["regression", "classification"])
    p.add_argument("--age_bins", type=int, default=101)
    p.add_argument("--image_size", type=int, default=224)
    p.add_argument("--embed_dim", type=int, default=512)
    p.add_argument("--batch_size", type=int, default=64)
    p.add_argument("--epochs", type=int, default=10)
    p.add_argument("--lr", type=float, default=1e-3)
    p.add_argument("--wd", type=float, default=1e-4)
    p.add_argument("--num_workers", type=int, default=4)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--out_dir", type=str, default="outputs_multitask")
    p.add_argument("--no_amp", action="store_true")

    p.add_argument("--lambda_id", type=float, default=1.0)
    p.add_argument("--lambda_age", type=float, default=1.0)
    p.add_argument("--lambda_mix", type=float, default=1.0)
    p.add_argument("--lambda_sep", type=float, default=0.1)
    return p.parse_args()


def main():
    args = parse_args()
    cfg = TrainCfg(
        train_csv=args.train_csv,
        train_root=args.train_root,
        image_size=args.image_size,
        backbone=args.backbone,
        embed_dim=args.embed_dim,
        num_classes=args.num_classes,
        age_bins=args.age_bins,
        seed=args.seed,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        lr=args.lr,
        wd=args.wd,
        epochs=args.epochs,
        amp=not args.no_amp,
        out_dir=args.out_dir,
        lambda_id=args.lambda_id,
        lambda_age=args.lambda_age,
        lambda_mix=args.lambda_mix,
        lambda_sep=args.lambda_sep,
    )

    set_seed(cfg.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    ensure_dir(cfg.out_dir)
    tb = make_tb_writer(cfg.out_dir)

    ds = FaceIdAgeCsvDataset(cfg.train_csv, train_root=cfg.train_root, image_size=cfg.image_size, train=True)
    dl = DataLoader(ds, batch_size=cfg.batch_size, shuffle=True, num_workers=cfg.num_workers, pin_memory=True, drop_last=True)

    mcfg = ModelCfg(
        backbone=cfg.backbone,
        embed_dim=cfg.embed_dim,
        num_classes=cfg.num_classes,
        age_mode=args.age_mode,
        age_bins=args.age_bins,
    )
    model = ThreeTaskModel(mcfg).to(device)

    loss_fn = MultiTaskLoss(
        lambda_id=cfg.lambda_id,
        lambda_age=cfg.lambda_age,
        lambda_mix=cfg.lambda_mix,
        lambda_sep=cfg.lambda_sep,
        age_mode=args.age_mode,
        age_bins=args.age_bins,
    )

    opt = torch.optim.AdamW(model.parameters(), lr=cfg.lr, weight_decay=cfg.wd)
    scaler = torch.cuda.amp.GradScaler(enabled=(cfg.amp and device.type == "cuda"))

    step = 0
    model.train()
    for epoch in range(cfg.epochs):
        epoch_loss_sum = 0.0
        epoch_batches = 0
        for x, y_id, y_age in dl:
            x = x.to(device, non_blocking=True)
            y_id = y_id.to(device, non_blocking=True)
            y_age = y_age.to(device, non_blocking=True)

            opt.zero_grad(set_to_none=True)
            with torch.cuda.amp.autocast(enabled=scaler.is_enabled()):
                out = model(x, y_id=y_id)
                if args.age_mode == "classification":
                    y_age = y_age.clamp(0, args.age_bins - 1).long()
                loss, parts = loss_fn(
                    id_logits=out["id_logits"],
                    mix_logits=out["mix_logits"],
                    age_pred=out["age_pred"],
                    z_id=out["z_id"],
                    z_age=out["z_age"],
                    y_id=y_id,
                    y_age=y_age,
                )

            scaler.scale(loss).backward()
            scaler.step(opt)
            scaler.update()

            li = loss.item()
            epoch_loss_sum += li
            epoch_batches += 1
            if step % 50 == 0:
                acc = accuracy_top1(out["id_logits"].detach(), y_id).item()
                acc_mix = accuracy_top1(out["mix_logits"].detach(), y_id).item()
                print(
                    f"epoch {epoch} step {step} "
                    f"loss {li:.4f} "
                    f"lid {parts['lid'].item():.4f} "
                    f"lage {parts['lage'].item():.4f} "
                    f"lmix {parts['lmix'].item():.4f} "
                    f"lsep {parts['lsep'].item():.4f} "
                    f"acc {acc:.3f} acc_mix {acc_mix:.3f}"
                )
                tb.add_scalar("train/loss", li, step)
                tb.add_scalar("train/lid", parts["lid"].item(), step)
                tb.add_scalar("train/lage", parts["lage"].item(), step)
                tb.add_scalar("train/lmix", parts["lmix"].item(), step)
                tb.add_scalar("train/lsep", parts["lsep"].item(), step)
                tb.add_scalar("train/acc_top1", acc, step)
                tb.add_scalar("train/acc_mix_top1", acc_mix, step)

            step += 1

        if epoch_batches > 0:
            tb.add_scalar("train/loss_epoch", epoch_loss_sum / epoch_batches, epoch)

        ckpt = {
            "epoch": epoch,
            "model": model.state_dict(),
            "opt": opt.state_dict(),
            "cfg": vars(args),
        }
        torch.save(ckpt, os.path.join(cfg.out_dir, f"ckpt_epoch_{epoch}.pt"))

    tb.close()


if __name__ == "__main__":
    main()

