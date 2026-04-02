"""
MT-MIM + 目标年龄条件人脸生成训练脚本。

数据路径见 configs/config_age_gen.py（复用 swinface_project/dataset）。
"""

import argparse
import logging
import math
import os
import sys
import multiprocessing

import torch
import torch.nn.functional as F
from torch import distributed
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter

from lr_scheduler import build_scheduler
from analysis import get_analysis_train_dataloader
from analysis.mi_loss import MILoss, shuffle_batch
from model_age_gen import MTMIMAgeGenerationModel, identity_cosine_loss
from utils.utils_config import get_config
from utils.utils_logging import AverageMeter, init_logging
from utils.utils_distributed_sampler import setup_seed

try:
    multiprocessing.set_start_method("spawn", force=True)
except RuntimeError:
    pass

assert torch.__version__ >= "1.9.0"

try:
    world_size = int(os.environ["WORLD_SIZE"])
    rank = int(os.environ["RANK"])
    distributed.init_process_group("nccl")
except KeyError:
    world_size = 1
    rank = 0
    distributed.init_process_group(
        backend="nccl",
        init_method="tcp://127.0.0.1:12620",
        rank=rank,
        world_size=world_size,
    )


def hinge_d_loss(real: torch.Tensor, fake: torch.Tensor) -> torch.Tensor:
    real = real.mean(dim=(1, 2, 3))
    fake = fake.mean(dim=(1, 2, 3))
    return F.relu(1.0 - real).mean() + F.relu(1.0 + fake).mean()


def hinge_g_loss(fake: torch.Tensor) -> torch.Tensor:
    fake = fake.mean(dim=(1, 2, 3))
    return F.relu(1.0 - fake).mean()


def set_backbone_grad(model: MTMIMAgeGenerationModel, trainable: bool):
    for p in model.backbone.parameters():
        p.requires_grad = trainable


# Keep file handle alive for the whole run (rank 0 only)
_console_out_f = None


class _TeeStream:
    """Write to both the original stream and a log file (console mirror)."""

    def __init__(self, *streams):
        self.streams = streams

    def write(self, data):
        for s in self.streams:
            s.write(data)
            s.flush()

    def flush(self):
        for s in self.streams:
            s.flush()

    def fileno(self):
        return self.streams[0].fileno()

    def isatty(self):
        return self.streams[0].isatty()


def main(args):
    global _console_out_f

    cfg = get_config(args.config)
    setup_seed(seed=cfg.seed, cuda_deterministic=False)

    torch.cuda.set_device(args.local_rank)
    device = torch.device("cuda", args.local_rank)

    os.makedirs(cfg.output, exist_ok=True)
    if rank == 0:
        out_path = os.path.join(cfg.output, "train.out")
        _console_out_f = open(out_path, "w", encoding="utf-8", buffering=1)
        sys.stdout = _TeeStream(sys.__stdout__, _console_out_f)
        sys.stderr = _TeeStream(sys.__stderr__, _console_out_f)

    init_logging(rank, cfg.output)

    tensorboard_dir = os.path.join(cfg.output, "tensorboard_age_gen")
    os.makedirs(tensorboard_dir, exist_ok=True)
    summary_writer = SummaryWriter(log_dir=tensorboard_dir) if rank == 0 else None

    train_loader = get_analysis_train_dataloader("age_gender", cfg, args.local_rank)

    model = MTMIMAgeGenerationModel(cfg).to(device)
    model = torch.nn.parallel.DistributedDataParallel(
        module=model,
        broadcast_buffers=False,
        device_ids=[args.local_rank],
        bucket_cap_mb=16,
        find_unused_parameters=True,
    )
    model.train()
    if model.module.vgg_loss is not None:
        model.module.vgg_loss.eval()

    mi_loss_fn = MILoss(temperature=getattr(cfg, "mi_temperature", 0.1))

    cfg.total_batch_size = world_size * cfg.age_gender_bz
    cfg.epoch_step = max(len(train_loader), 1)
    cfg.num_epoch = math.ceil(cfg.total_step / cfg.epoch_step)

    cfg.lr = cfg.lr * cfg.total_batch_size / 512.0
    cfg.warmup_lr = cfg.warmup_lr * cfg.total_batch_size / 512.0
    cfg.min_lr = cfg.min_lr * cfg.total_batch_size / 512.0

    m = model.module
    g_params = (
        list(m.backbone.parameters())
        + list(m.fam.parameters())
        + list(m.tss.parameters())
        + list(m.age_extractor.parameters())
        + list(m.target_age_encoder.parameters())
        + list(m.fusion.parameters())
        + list(m.generator.parameters())
        + list(m.age_head.parameters())
    )

    opt_g = torch.optim.AdamW(g_params, lr=cfg.lr, weight_decay=cfg.weight_decay, foreach=False)
    opt_d = torch.optim.AdamW(m.discriminator.parameters(), lr=cfg.lr_d, weight_decay=cfg.weight_decay, foreach=False)

    lr_scheduler = build_scheduler(
        optimizer=opt_g,
        lr_name=cfg.lr_name,
        warmup_lr=cfg.warmup_lr,
        min_lr=cfg.min_lr,
        num_steps=cfg.total_step,
        warmup_steps=cfg.warmup_step,
    )

    amp = torch.cuda.amp.grad_scaler.GradScaler(growth_interval=100)

    start_epoch = 0
    global_step = 0

    if cfg.init:
        init_model_path = os.path.join(cfg.init_model, f"start_{rank}.pt")
        if os.path.exists(init_model_path):
            ck = torch.load(init_model_path, map_location=device)
            m.backbone.load_state_dict(ck["state_dict_backbone"], strict=False)
            if rank == 0:
                logging.info("Loaded backbone from %s", init_model_path)
            del ck

    freeze_until = int(getattr(cfg, "freeze_backbone_steps", 0))
    set_backbone_grad(m, freeze_until == 0)

    for key, value in cfg.items():
        num_space = max(0, 25 - len(str(key)))
        logging.info(": %s%s%s", key, " " * num_space, str(value))

    loss_am = AverageMeter()
    id_am = AverageMeter()
    age_am = AverageMeter()
    adv_am = AverageMeter()
    rec_am = AverageMeter()
    vgg_am = AverageMeter()
    mi_am = AverageMeter()
    d_am = AverageMeter()

    for epoch in range(start_epoch, cfg.num_epoch):
        if isinstance(train_loader, DataLoader):
            train_loader.sampler.set_epoch(epoch)

        for idx, batch in enumerate(train_loader):
            if global_step >= cfg.total_step:
                break

            imgs, [age_label, _gender] = batch
            imgs = imgs.cuda(non_blocking=True)
            age_src = age_label.float().cuda(non_blocking=True).view(-1)

            b = age_src.size(0)
            recon_mask = torch.rand(b, device=device) < float(cfg.recon_prob)
            target_age = torch.empty_like(age_src)
            target_age[recon_mask] = age_src[recon_mask]
            rand = torch.rand(b, device=device)
            lo = float(cfg.target_age_min)
            hi = float(cfg.target_age_max)
            target_age[~recon_mask] = lo + rand[~recon_mask] * (hi - lo)

            frozen = global_step < freeze_until
            set_backbone_grad(m, not frozen)

            opt_g.zero_grad(set_to_none=True)

            with torch.cuda.amp.autocast(enabled=cfg.fp16):
                out = model(imgs, target_age)
                x_hat = out["x_hat"]
                x_id = out["x_id"]
                x_age = out["x_age"]

                enc = model(x_hat, target_age, True)
                x_id_hat = enc["x_id"]
                loss_id = identity_cosine_loss(x_id, x_id_hat)
                loss_age = F.smooth_l1_loss(enc["age_pred"], target_age)

                x_age_shuf = shuffle_batch(x_age)
                loss_mi = mi_loss_fn(x_id, x_age, x_age_shuf)

                if recon_mask.any():
                    loss_rec = F.l1_loss(x_hat[recon_mask], imgs[recon_mask])
                else:
                    loss_rec = x_hat.new_tensor(0.0)

                loss_vgg = m.perceptual_loss(imgs, x_hat) if m.vgg_loss is not None else x_hat.new_tensor(0.0)

                for p in m.discriminator.parameters():
                    p.requires_grad_(False)
                d_fake = m.discriminate(x_hat)
                loss_g_adv = hinge_g_loss(d_fake)

                loss_g_non_age = (
                    cfg.lambda_id * loss_id
                    + cfg.lambda_mi * loss_mi
                    + cfg.lambda_rec * loss_rec
                    + cfg.lambda_vgg * loss_vgg
                    + cfg.lambda_adv * loss_g_adv
                )
                loss_g_age_w = cfg.lambda_age * loss_age
                loss_g = loss_g_non_age + loss_g_age_w

            g_trainable = [p for p in g_params if p.requires_grad]

            def _g_step(loss_tensor):
                if cfg.fp16:
                    amp.scale(loss_tensor).backward()
                    amp.unscale_(opt_g)
                    if g_trainable:
                        torch.nn.utils.clip_grad_norm_(g_trainable, 5.0)
                    amp.step(opt_g)
                    amp.update()
                else:
                    loss_tensor.backward()
                    if g_trainable:
                        torch.nn.utils.clip_grad_norm_(g_trainable, 5.0)
                    opt_g.step()

            if getattr(cfg, "separate_age_g_step", False):
                _g_step(loss_g_non_age)
                opt_g.zero_grad(set_to_none=True)
                with torch.cuda.amp.autocast(enabled=cfg.fp16):
                    out2 = model(imgs, target_age)
                    enc2 = model(out2["x_hat"], target_age, True)
                    loss_age2 = F.smooth_l1_loss(enc2["age_pred"], target_age)
                _g_step(cfg.lambda_age * loss_age2)
            else:
                _g_step(loss_g)

            for p in m.discriminator.parameters():
                p.requires_grad_(True)

            lr_scheduler.step_update(global_step)

            opt_d.zero_grad(set_to_none=True)
            with torch.cuda.amp.autocast(enabled=cfg.fp16):
                d_real = m.discriminate(imgs)
                d_fake_d = m.discriminate(x_hat.detach())
                loss_d = hinge_d_loss(d_real, d_fake_d)

            if cfg.fp16:
                amp.scale(loss_d).backward()
                amp.unscale_(opt_d)
                torch.nn.utils.clip_grad_norm_(m.discriminator.parameters(), 5.0)
                amp.step(opt_d)
                amp.update()
            else:
                loss_d.backward()
                torch.nn.utils.clip_grad_norm_(m.discriminator.parameters(), 5.0)
                opt_d.step()

            with torch.no_grad():
                loss_am.update(loss_g.item(), 1)
                id_am.update(loss_id.item(), 1)
                age_am.update(loss_age.item(), 1)
                adv_am.update(loss_g_adv.item(), 1)
                rec_am.update(loss_rec.item() if recon_mask.any() else 0.0, 1)
                vgg_am.update(loss_vgg.item(), 1)
                mi_am.update(loss_mi.item(), 1)
                d_am.update(loss_d.item(), 1)

            if rank == 0 and summary_writer is not None and (global_step + 1) % cfg.frequent == 0:
                summary_writer.add_scalar("loss/total_g", loss_am.avg, global_step)
                summary_writer.add_scalar("loss/id", id_am.avg, global_step)
                summary_writer.add_scalar("loss/age", age_am.avg, global_step)
                summary_writer.add_scalar("loss/adv_g", adv_am.avg, global_step)
                summary_writer.add_scalar("loss/rec", rec_am.avg, global_step)
                summary_writer.add_scalar("loss/vgg", vgg_am.avg, global_step)
                summary_writer.add_scalar("loss/mi", mi_am.avg, global_step)
                summary_writer.add_scalar("loss/d", d_am.avg, global_step)
                summary_writer.add_scalar("meta/recon_frac", recon_mask.float().mean().item(), global_step)
                loss_am.reset()
                id_am.reset()
                age_am.reset()
                adv_am.reset()
                rec_am.reset()
                vgg_am.reset()
                mi_am.reset()
                d_am.reset()

            if rank == 0 and (global_step + 1) % cfg.frequent == 0:
                logging.info(
                    "step %d  id %.4f  age %.4f  mi %.4f  rec %.4f  vgg %.4f  g_adv %.4f  d %.4f  frozen_backbone %s",
                    global_step,
                    loss_id.item(),
                    loss_age.item(),
                    loss_mi.item(),
                    loss_rec.item() if recon_mask.any() else 0.0,
                    loss_vgg.item(),
                    loss_g_adv.item(),
                    loss_d.item(),
                    str(frozen),
                )

            if cfg.save_all_states and rank == 0 and (global_step + 1) % cfg.save_verbose == 0:
                ck = {
                    "epoch": epoch,
                    "global_step": global_step,
                    "state_dict_backbone": m.backbone.state_dict(),
                    "state_dict_fam": m.fam.state_dict(),
                    "state_dict_tss": m.tss.state_dict(),
                    "state_dict_age_extractor": m.age_extractor.state_dict(),
                    "state_dict_target_age_encoder": m.target_age_encoder.state_dict(),
                    "state_dict_fusion": m.fusion.state_dict(),
                    "state_dict_generator": m.generator.state_dict(),
                    "state_dict_age_head": m.age_head.state_dict(),
                    "state_dict_discriminator": m.discriminator.state_dict(),
                    "state_optimizer_g": opt_g.state_dict(),
                    "state_optimizer_d": opt_d.state_dict(),
                    "state_lr_scheduler": lr_scheduler.state_dict(),
                }
                path = os.path.join(cfg.output, f"checkpoint_age_gen_step_{global_step}.pt")
                torch.save(ck, path)
                logging.info("Saved %s", path)

            global_step += 1

        if global_step >= cfg.total_step:
            break

    distributed.destroy_process_group()


if __name__ == "__main__":
    torch.backends.cudnn.benchmark = True
    parser = argparse.ArgumentParser(description="MT-MIM age-conditioned face generation")
    parser.add_argument("config", type=str)
    parser.add_argument("--local_rank", "--local-rank", type=int, default=None)
    args = parser.parse_args()
    if args.local_rank is None:
        args.local_rank = int(os.environ.get("LOCAL_RANK", 0))
    main(args)
