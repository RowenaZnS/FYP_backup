import argparse
import os

import torch
from torch.utils.data import DataLoader

from src.datasets import FaceIdAgeCsvDataset
from src.losses import ArcFace, ArcFaceCfg
from src.models import ThreeTaskModel, ModelCfg
from src.utils import accuracy_top1, ensure_dir, make_tb_writer, set_seed


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--train_csv", type=str, required=True, help="CSV with columns: path,id,age (age ignored here)")
    p.add_argument("--train_root", type=str, default=None)
    p.add_argument("--backbone", type=str, default="resnet50")
    p.add_argument("--num_classes", type=int, required=True)
    p.add_argument("--image_size", type=int, default=224)
    p.add_argument("--embed_dim", type=int, default=512)
    p.add_argument("--batch_size", type=int, default=64)
    p.add_argument("--epochs", type=int, default=10)
    p.add_argument("--lr", type=float, default=1e-3)
    p.add_argument("--wd", type=float, default=1e-4)
    p.add_argument("--num_workers", type=int, default=4)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--out_dir", type=str, default="outputs_identity")
    p.add_argument("--no_amp", action="store_true")
    return p.parse_args()


def main():
    args = parse_args()
    set_seed(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    ensure_dir(args.out_dir)
    tb = make_tb_writer(args.out_dir)

    ds = FaceIdAgeCsvDataset(args.train_csv, train_root=args.train_root, image_size=args.image_size, train=True)
    dl = DataLoader(ds, batch_size=args.batch_size, shuffle=True, num_workers=args.num_workers, pin_memory=True, drop_last=True)

    mcfg = ModelCfg(
        backbone=args.backbone,
        embed_dim=args.embed_dim,
        num_classes=args.num_classes,
        age_mode="regression",
    )
    model = ThreeTaskModel(mcfg).to(device)

    # 只用 id 分支：ArcFace + CE
    ce = torch.nn.CrossEntropyLoss()
    opt = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.wd)
    scaler = torch.cuda.amp.GradScaler(enabled=((not args.no_amp) and device.type == "cuda"))

    step = 0
    model.train()
    for epoch in range(args.epochs):
        epoch_loss_sum = 0.0
        epoch_batches = 0
        for x, y_id, _y_age in dl:
            x = x.to(device, non_blocking=True)
            y_id = y_id.to(device, non_blocking=True)
            opt.zero_grad(set_to_none=True)
            with torch.cuda.amp.autocast(enabled=scaler.is_enabled()):
                out = model(x, y_id=y_id)
                loss = ce(out["id_logits"], y_id)

            scaler.scale(loss).backward()
            scaler.step(opt)
            scaler.update()

            li = loss.item()
            epoch_loss_sum += li
            epoch_batches += 1
            if step % 50 == 0:
                acc = accuracy_top1(out["id_logits"].detach(), y_id).item()
                print(f"epoch {epoch} step {step} loss {li:.4f} acc {acc:.3f}")
                tb.add_scalar("train/loss", li, step)
                tb.add_scalar("train/acc_top1", acc, step)
            step += 1

        if epoch_batches > 0:
            tb.add_scalar("train/loss_epoch", epoch_loss_sum / epoch_batches, epoch)

        torch.save(
            {"epoch": epoch, "model": model.state_dict(), "opt": opt.state_dict(), "cfg": vars(args)},
            os.path.join(args.out_dir, f"ckpt_epoch_{epoch}.pt"),
        )

    tb.close()


if __name__ == "__main__":
    main()

