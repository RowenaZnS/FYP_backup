"""
MT-MIM + 目标年龄条件生成 + 多任务分析（Gender / CelebA / Expression + MI），
不包含年龄预测 loss（与 train_rest_mtmim 一致：年龄标签仅用于 MI）。

三路数据：age_gender + CelebA + expression，需配置 config.use_analysis_heads = True。
"""

import argparse
import logging
import math
import os
import sys
import multiprocessing
from itertools import cycle

import torch
import torch.nn.functional as F
from torch import distributed
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter

from lr_scheduler import build_scheduler
from analysis import get_analysis_train_dataloader
from analysis.mi_loss import MILoss, shuffle_batch
from analysis.task_name import ANALYSIS_TASKS
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
        init_method="tcp://127.0.0.1:12621",
        rank=rank,
        world_size=world_size,
    )

_console_out_f = None


class _TeeStream:
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


def _celeba_tb_tag(j: int) -> str:
    """TensorBoard 子 tag：CelebA 第 j 个属性（对应 ANALYSIS_TASKS[j+1]）。"""
    raw = ANALYSIS_TASKS[j + 1]
    slug = raw.replace(" ", "_").replace("'", "").replace(".", "").replace("/", "_")
    return f"loss/celeba/{j:02d}_{slug}"


def main(args):
    global _console_out_f

    cfg = get_config(args.config)
    assert getattr(cfg, "use_analysis_heads", False), "config 需设置 use_analysis_heads = True"

    setup_seed(seed=cfg.seed, cuda_deterministic=False)

    torch.cuda.set_device(args.local_rank)
    device = torch.device("cuda", args.local_rank)

    os.makedirs(cfg.output, exist_ok=True)
    if rank == 0:
        _console_out_f = open(os.path.join(cfg.output, "train.out"), "w", encoding="utf-8", buffering=1)
        sys.stdout = _TeeStream(sys.__stdout__, _console_out_f)
        sys.stderr = _TeeStream(sys.__stderr__, _console_out_f)

    init_logging(rank, cfg.output)

    tensorboard_dir = os.path.join(cfg.output, "tensorboard_age_gen_mt")
    os.makedirs(tensorboard_dir, exist_ok=True)
    summary_writer = SummaryWriter(log_dir=tensorboard_dir) if rank == 0 else None

    age_gender_train_loader = get_analysis_train_dataloader("age_gender", cfg, args.local_rank)
    CelebA_train_loader = get_analysis_train_dataloader("CelebA", cfg, args.local_rank)
    Expression_train_loader = get_analysis_train_dataloader("expression", cfg, args.local_rank)

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
    gender_loss = torch.nn.CrossEntropyLoss()
    celebA_losses = [torch.nn.CrossEntropyLoss() for _ in range(40)]
    expression_loss = torch.nn.CrossEntropyLoss()

    cfg.total_batch_size = world_size * (cfg.age_gender_bz + cfg.CelebA_bz + cfg.expression_bz)
    cfg.epoch_step = max(
        len(age_gender_train_loader), len(CelebA_train_loader), len(Expression_train_loader), 1
    )
    cfg.num_epoch = math.ceil(cfg.total_step / cfg.epoch_step)

    cfg.lr = cfg.lr * cfg.total_batch_size / 512.0
    cfg.warmup_lr = cfg.warmup_lr * cfg.total_batch_size / 512.0
    cfg.min_lr = cfg.min_lr * cfg.total_batch_size / 512.0

    m = model.module
    g_params = (
        list(m.backbone.parameters())
        + list(m.fam.parameters())
        + list(m.tss.parameters())
        + list(m.om.parameters())
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

    bzs = [cfg.age_gender_bz, cfg.CelebA_bz, cfg.expression_bz]
    features_cut = [0]
    for i in range(1, len(bzs) + 1):
        features_cut.append(features_cut[i - 1] + bzs[i - 1])
    ag0, ag1 = features_cut[0], features_cut[1]
    ce0, ce1 = features_cut[1], features_cut[2]
    ex0, ex1 = features_cut[2], features_cut[3]

    loss_am = AverageMeter()
    id_am = AverageMeter()
    gender_am = AverageMeter()
    celebA_am = AverageMeter()
    celebA_attr_ams = [AverageMeter() for _ in range(n_celeba)]
    expr_am = AverageMeter()
    ana_am = AverageMeter()
    adv_am = AverageMeter()
    rec_am = AverageMeter()
    vgg_am = AverageMeter()
    mi_am = AverageMeter()
    d_am = AverageMeter()

    for epoch in range(start_epoch, cfg.num_epoch):
        if isinstance(age_gender_train_loader, DataLoader):
            age_gender_train_loader.sampler.set_epoch(epoch)
        if isinstance(CelebA_train_loader, DataLoader):
            CelebA_train_loader.sampler.set_epoch(epoch)
        if isinstance(Expression_train_loader, DataLoader):
            Expression_train_loader.sampler.set_epoch(epoch)

        max_len = max(len(age_gender_train_loader), len(CelebA_train_loader), len(Expression_train_loader))
        age_gender_iter = cycle(age_gender_train_loader) if len(age_gender_train_loader) < max_len else iter(age_gender_train_loader)
        CelebA_iter = cycle(CelebA_train_loader) if len(CelebA_train_loader) < max_len else iter(CelebA_train_loader)
        Expression_iter = cycle(Expression_train_loader) if len(Expression_train_loader) < max_len else iter(Expression_train_loader)

        for idx in range(max_len):
            if global_step >= cfg.total_step:
                break

            age_gender = next(age_gender_iter)
            CelebA = next(CelebA_iter)
            expr = next(Expression_iter)

            age_gender_img, [age_label, gender_label_1] = age_gender
            CelebA_img, CelebA_label = CelebA
            expression_img, expression_label = expr

            imgs = torch.cat([age_gender_img, CelebA_img, expression_img], dim=0).cuda(non_blocking=True)
            b = imgs.size(0)
            age_src = age_label.float().cuda(non_blocking=True).view(-1)
            gender_label = gender_label_1.cuda(non_blocking=True)
            expression_label = expression_label.cuda(non_blocking=True)

            lo = float(cfg.target_age_min)
            hi = float(cfg.target_age_max)
            target_age = torch.empty(b, device=device)
            ag_len = ag1 - ag0
            recon_mask = torch.rand(ag_len, device=device) < float(cfg.recon_prob)
            ta_ag = torch.empty(ag_len, device=device)
            rand_ag = torch.rand(ag_len, device=device)
            ta_ag[recon_mask] = age_src[recon_mask]
            ta_ag[~recon_mask] = lo + rand_ag[~recon_mask] * (hi - lo)
            target_age[ag0:ag1] = ta_ag
            cl = ce1 - ce0
            target_age[ce0:ce1] = lo + torch.rand(cl, device=device) * (hi - lo)
            el = ex1 - ex0
            target_age[ex0:ex1] = lo + torch.rand(el, device=device) * (hi - lo)

            rec_full = torch.zeros(b, dtype=torch.bool, device=device)
            rec_full[ag0:ag1] = recon_mask

            frozen = global_step < freeze_until
            set_backbone_grad(m, not frozen)

            opt_g.zero_grad(set_to_none=True)

            with torch.cuda.amp.autocast(enabled=cfg.fp16):
                out = model(imgs, target_age)
                x_hat = out["x_hat"]
                x_id = out["x_id"]
                x_age = out["x_age"]
                std = out["standard_list"]

                enc = model(x_hat, target_age, True)
                x_id_hat = enc["x_id"]
                loss_id = identity_cosine_loss(x_id, x_id_hat)

                gender_output = std[5][ag0:ag1]
                gender_loss_val = gender_loss(gender_output, gender_label)
                celebA_losses_val = []
                for j in range(n_celeba):
                    attr_output = std[j + 1][ce0:ce1]
                    attr_label = CelebA_label[j].cuda(non_blocking=True)
                    celebA_losses_val.append(celebA_losses[j](attr_output, attr_label))
                # 与 OutputModule List 顺序一致：Expression 固定为第 41 号（0-based）
                expression_output = std[41][ex0:ex1]
                expression_loss_val = expression_loss(expression_output, expression_label)

                w_g = float(cfg.analysis_loss_weights[5])
                loss_gender_w = w_g * gender_loss_val
                loss_celebA_w = sum(
                    float(cfg.analysis_loss_weights[j + 1]) * celebA_losses_val[j] for j in range(n_celeba)
                )
                loss_expr_w = float(cfg.analysis_loss_weights[41]) * expression_loss_val
                loss_analysis = loss_gender_w + loss_celebA_w + loss_expr_w

                age_x_id = x_id[ag0:ag1]
                age_x_age = x_age[ag0:ag1]
                loss_mi = mi_loss_fn(age_x_id, age_x_age, shuffle_batch(age_x_age))
                mi_w = float(getattr(cfg, "mi_loss_weight", 0.1))

                if rec_full.any():
                    loss_rec = F.l1_loss(x_hat[rec_full], imgs[rec_full])
                else:
                    loss_rec = x_hat.new_tensor(0.0)

                loss_vgg = m.perceptual_loss(imgs, x_hat) if m.vgg_loss is not None else x_hat.new_tensor(0.0)

                for p in m.discriminator.parameters():
                    p.requires_grad_(False)
                d_fake = m.discriminate(x_hat)
                loss_g_adv = hinge_g_loss(d_fake)

                loss_g = (
                    cfg.lambda_id * loss_id
                    + loss_analysis
                    + mi_w * loss_mi
                    + cfg.lambda_rec * loss_rec
                    + cfg.lambda_vgg * loss_vgg
                    + cfg.lambda_adv * loss_g_adv
                )

            g_trainable = [p for p in g_params if p.requires_grad]
            if cfg.fp16:
                amp.scale(loss_g).backward()
                amp.unscale_(opt_g)
                if g_trainable:
                    torch.nn.utils.clip_grad_norm_(g_trainable, 5.0)
                amp.step(opt_g)
                amp.update()
            else:
                loss_g.backward()
                if g_trainable:
                    torch.nn.utils.clip_grad_norm_(g_trainable, 5.0)
                opt_g.step()

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
                gender_am.update(loss_gender_w.item(), 1)
                celebA_am.update(loss_celebA_w.item(), 1)
                for j in range(len(celebA_losses_val)):
                    wj = float(cfg.analysis_loss_weights[j + 1])
                    celebA_attr_ams[j].update((wj * celebA_losses_val[j]).item(), 1)
                expr_am.update(loss_expr_w.item(), 1)
                ana_am.update(loss_analysis.item(), 1)
                adv_am.update(loss_g_adv.item(), 1)
                rec_am.update(loss_rec.item() if rec_full.any() else 0.0, 1)
                vgg_am.update(loss_vgg.item(), 1)
                mi_am.update(loss_mi.item(), 1)
                d_am.update(loss_d.item(), 1)

            if rank == 0 and summary_writer is not None and (global_step + 1) % cfg.frequent == 0:
                summary_writer.add_scalar("loss/total_g", loss_am.avg, global_step)
                summary_writer.add_scalar("loss/id", id_am.avg, global_step)
                summary_writer.add_scalar("loss/analysis_total", ana_am.avg, global_step)
                summary_writer.add_scalar("loss/gender", gender_am.avg, global_step)
                summary_writer.add_scalar("loss/celeba", celebA_am.avg, global_step)
                if getattr(cfg, "log_celeba_each_attr", True):
                    for j in range(n_celeba):
                        summary_writer.add_scalar(_celeba_tb_tag(j), celebA_attr_ams[j].avg, global_step)
                summary_writer.add_scalar("loss/expression", expr_am.avg, global_step)
                summary_writer.add_scalar("loss/adv_g", adv_am.avg, global_step)
                summary_writer.add_scalar("loss/rec", rec_am.avg, global_step)
                summary_writer.add_scalar("loss/vgg", vgg_am.avg, global_step)
                summary_writer.add_scalar("loss/mi", mi_am.avg, global_step)
                summary_writer.add_scalar("loss/d", d_am.avg, global_step)
                loss_am.reset()
                id_am.reset()
                gender_am.reset()
                celebA_am.reset()
                for _am in celebA_attr_ams:
                    _am.reset()
                expr_am.reset()
                ana_am.reset()
                adv_am.reset()
                rec_am.reset()
                vgg_am.reset()
                mi_am.reset()
                d_am.reset()

            if rank == 0 and (global_step + 1) % cfg.frequent == 0:
                logging.info(
                    "step %d  id %.4f  gender %.4f  celebA %.4f  expr %.4f  analysis_sum %.4f  "
                    "mi %.4f  rec %.4f  vgg %.4f  g_adv %.4f  d %.4f  frozen_backbone %s",
                    global_step,
                    loss_id.item(),
                    loss_gender_w.item(),
                    loss_celebA_w.item(),
                    loss_expr_w.item(),
                    loss_analysis.item(),
                    loss_mi.item(),
                    loss_rec.item() if rec_full.any() else 0.0,
                    loss_vgg.item(),
                    loss_g_adv.item(),
                    loss_d.item(),
                    str(frozen),
                )
                if getattr(cfg, "log_celeba_each_attr_to_console", False):
                    for j in range(len(celebA_losses_val)):
                        wj = float(cfg.analysis_loss_weights[j + 1])
                        v = (wj * celebA_losses_val[j]).item()
                        logging.info(
                            "  celeba[%02d] %-28s %.4f",
                            j,
                            _celeba_attr_name(j),
                            v,
                        )

            if cfg.save_all_states and rank == 0 and (global_step + 1) % cfg.save_verbose == 0:
                ck = {
                    "epoch": epoch,
                    "global_step": global_step,
                    "state_dict_backbone": m.backbone.state_dict(),
                    "state_dict_fam": m.fam.state_dict(),
                    "state_dict_tss": m.tss.state_dict(),
                    "state_dict_om": m.om.state_dict(),
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
                path = os.path.join(cfg.output, f"checkpoint_age_gen_mt_step_{global_step}.pt")
                torch.save(ck, path)
                logging.info("Saved %s", path)

            global_step += 1

        if global_step >= cfg.total_step:
            break

    distributed.destroy_process_group()


if __name__ == "__main__":
    torch.backends.cudnn.benchmark = True
    parser = argparse.ArgumentParser(description="MT-MIM age-gen + multitask (no age loss)")
    parser.add_argument("config", type=str)
    parser.add_argument("--local_rank", "--local-rank", type=int, default=None)
    args = parser.parse_args()
    if args.local_rank is None:
        args.local_rank = int(os.environ.get("LOCAL_RANK", 0))
    main(args)
