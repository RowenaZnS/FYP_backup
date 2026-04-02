import argparse
import logging
import os
import multiprocessing
import math

import torch
from torch import distributed
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter

from dataset import get_dataloader
from losses import CombinedMarginLoss
from lr_scheduler import build_scheduler
from partial_fc import PartialFC, PartialFCAdamW
from analysis import get_analysis_train_dataloader
from model import build_model

from utils.utils_callbacks import CallBackLogging, CallBackVerification
from utils.utils_config import get_config
from utils.utils_logging import AverageMeter, init_logging
from utils.utils_distributed_sampler import setup_seed

# Set multiprocessing start method to 'spawn' to avoid CUDA re-initialization errors
# This must be done before any CUDA operations or DataLoader creation
try:
    multiprocessing.set_start_method('spawn', force=True)
except RuntimeError:
    # Start method can only be set once per program, ignore if already set
    pass

assert torch.__version__ >= "1.9.0", "In order to enjoy the features of the new torch, \
we have upgraded the torch to 1.9.0. torch before than 1.9.0 may not work in the future."

try:
    world_size = int(os.environ["WORLD_SIZE"])
    rank = int(os.environ["RANK"])
    distributed.init_process_group("nccl")
except KeyError:
    world_size = 1
    rank = 0
    distributed.init_process_group(
        backend="nccl",
        init_method="tcp://127.0.0.1:12587",
        rank=rank,
        world_size=world_size,
    )


def main(args):
    # get config
    cfg = get_config(args.config)
    # global control random seed
    setup_seed(seed=cfg.seed, cuda_deterministic=False)

    torch.cuda.set_device(args.local_rank)
    device = torch.device("cuda", args.local_rank)

    os.makedirs(cfg.output, exist_ok=True)
    init_logging(rank, cfg.output)

    summary_writer = (
        SummaryWriter(log_dir=os.path.join(cfg.output, "tensorboard_recognition"))
        if rank == 0
        else None
    )

    # Recognition dataloader
    train_loader = get_analysis_train_dataloader("recognition", cfg, args.local_rank)

    model = build_model(cfg).to(device)

    model = torch.nn.parallel.DistributedDataParallel(
        module=model, broadcast_buffers=False, device_ids=[args.local_rank], bucket_cap_mb=16,
        find_unused_parameters=True)
    model.train()
    # FIXME using gradient checkpoint if there are some unused parameters will cause error
    # model._set_static_graph()

    # Total batch size: only recognition
    cfg.total_batch_size = world_size * cfg.recognition_bz
    cfg.epoch_step = len(train_loader)

    cfg.num_epoch = math.ceil(cfg.total_step / cfg.epoch_step)

    cfg.lr = cfg.lr * cfg.total_batch_size / 512.0
    cfg.warmup_lr = cfg.warmup_lr * cfg.total_batch_size / 512.0
    cfg.min_lr = cfg.min_lr * cfg.total_batch_size / 512.0

    # Recognition loss
    margin_loss = CombinedMarginLoss(
        64,
        cfg.margin_list[0],
        cfg.margin_list[1],
        cfg.margin_list[2],
        cfg.interclass_filtering_threshold
    )

    if cfg.optimizer == "sgd":
        module_partial_fc = PartialFC(
            margin_loss, cfg.embedding_size, cfg.num_classes,
            cfg.sample_rate, cfg.fp16)
        module_partial_fc.train().to(device)
        # TODO the params of partial fc must be last in the params list
        opt = torch.optim.SGD(
            params=[{"params": model.module.backbone.parameters(), 'lr': cfg.lr / 10},
                    {"params": module_partial_fc.parameters()},
                    {"params": model.module.fam.parameters()},
                    {"params": model.module.tss.parameters()},
                    {"params": model.module.om.parameters()},
                    ],
            lr=cfg.lr, momentum=0.9, weight_decay=cfg.weight_decay)

    elif cfg.optimizer == "adamw":
        module_partial_fc = PartialFCAdamW(
            margin_loss, cfg.embedding_size, cfg.num_classes,
            cfg.sample_rate, cfg.fp16)
        module_partial_fc.train().to(device)
        
        # Collect all parameters and ensure they are on CUDA
        all_params = []
        all_params.append({"params": model.module.backbone.parameters(), 'lr': cfg.lr / 10})
        all_params.append({"params": module_partial_fc.parameters()})
        all_params.append({"params": model.module.fam.parameters()})
        all_params.append({"params": model.module.tss.parameters()})
        all_params.append({"params": model.module.om.parameters()})
        
        opt = torch.optim.AdamW(
            params=all_params,
            lr=cfg.lr, weight_decay=cfg.weight_decay,
            foreach=False)
    else:
        raise

    lr_scheduler = build_scheduler(
        optimizer=opt,
        lr_name=cfg.lr_name,
        warmup_lr=cfg.warmup_lr,
        min_lr=cfg.min_lr,
        num_steps=cfg.total_step,
        warmup_steps=cfg.warmup_step)

    start_epoch = 0
    global_step = 0

    if cfg.init:
        init_model_path = os.path.join(cfg.init_model, f"start_{rank}.pt")
        if os.path.exists(init_model_path):
            dict_checkpoint = torch.load(init_model_path)
            model.module.backbone.load_state_dict(dict_checkpoint["state_dict_backbone"],
                                                  strict=False)  # only load backbone!
            del dict_checkpoint
            if rank == 0:
                print(f"✓ Loaded pretrained model from {init_model_path}")
        else:
            if rank == 0:
                print(f"⚠️  Warning: Pretrained model not found at {init_model_path}, starting from random initialization")

    if cfg.resume:
        dict_checkpoint = torch.load(os.path.join(cfg.output, f"checkpoint_recognition_step_{cfg.resume_step}_gpu_{rank}.pt"))
        start_epoch = dict_checkpoint["epoch"]
        global_step = dict_checkpoint["global_step"]
        local_step = dict_checkpoint["local_step"]

        if local_step == cfg.epoch_step - 1:
            start_epoch = start_epoch+1
            local_step = 0
        else:
            local_step += 1

        global_step += 1

        model.module.backbone.load_state_dict(dict_checkpoint["state_dict_backbone"])
        module_partial_fc.load_state_dict(dict_checkpoint["state_dict_softmax_fc"])
        model.module.fam.load_state_dict(dict_checkpoint["state_dict_fam"])
        model.module.tss.load_state_dict(dict_checkpoint["state_dict_tss"])
        model.module.om.load_state_dict(dict_checkpoint["state_dict_om"])
        opt.load_state_dict(dict_checkpoint["state_optimizer"])
        for state in opt.state.values():
            for k, v in state.items():
                if torch.is_tensor(v):
                    state[k] = v.to(device)
        lr_scheduler.load_state_dict(dict_checkpoint["state_lr_scheduler"])
        del dict_checkpoint

    for key, value in cfg.items():
        num_space = 25 - len(key)
        logging.info(": " + key + " " * num_space + str(value))

    # Recognition verification
    callback_verification = CallBackVerification(
        val_targets=cfg.val_targets, rec_prefix=cfg.rec, summary_writer=summary_writer
    )

    callback_logging = CallBackLogging(
        frequent=cfg.frequent,
        total_step=cfg.total_step,
        batch_size=cfg.batch_size,
        start_step=global_step,
        writer=summary_writer
    )

    loss_am = AverageMeter()
    recognition_loss_am = AverageMeter()
    analysis_loss_ams = [AverageMeter() for j in range(42)]  # Dummy for CallBackLogging interface

    amp = torch.cuda.amp.grad_scaler.GradScaler(growth_interval=100)

    for epoch in range(start_epoch, cfg.num_epoch):

        if isinstance(train_loader, DataLoader):
            train_loader.sampler.set_epoch(epoch)

        for idx, data in enumerate(train_loader):

            # skip
            if cfg.resume:
                if idx < local_step:
                    continue

            recognition = data
            recognition_img, recognition_label = recognition

            # Move to CUDA
            recognition_label = recognition_label.cuda(non_blocking=True)

            # Forward
            img = recognition_img.cuda(non_blocking=True)
            model.module.set_output_type("List")
            outputs = model(img)
            
            # -------------------------
            # 1) Recognition loss (ArcFace / PartialFC)
            # -------------------------
            # embedding: outputs[-1]
            local_embeddings = outputs[-1]
            recognition_loss = module_partial_fc(local_embeddings, recognition_label, opt)
            
            # -------------------------
            # 2) Total loss
            # -------------------------
            loss = cfg.recognition_loss_weight * recognition_loss
            
            # -------------------------
            # 3) Backward + Optim step
            # -------------------------
            if cfg.fp16:
                amp.scale(loss).backward()
                amp.unscale_(opt)
                torch.nn.utils.clip_grad_norm_(model.module.backbone.parameters(), 5)
                amp.step(opt)
                amp.update()
            else:
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.module.backbone.parameters(), 5)
                opt.step()
            
            opt.zero_grad()
            lr_scheduler.step_update(global_step)
            
            # -------------------------
            # 4) Logging
            # -------------------------
            with torch.no_grad():
                loss_am.update(loss.item(), 1)
                recognition_loss_am.update(recognition_loss.item(), 1)
                # Keep analysis_loss_ams for interface compatibility, but don't update them
            
                callback_logging(
                    global_step,
                    loss_am,
                    recognition_loss_am,
                    analysis_loss_ams,
                    epoch,
                    cfg.fp16,
                    opt.param_groups[0]["lr"],
                    amp
                )
            
                if (global_step + 1) % cfg.verbose == 0:
                    model.module.set_output_type("Recognition")
                    callback_verification(global_step, model)

            if cfg.save_all_states and (global_step+1) % cfg.save_verbose == 0:
                checkpoint = {
                    "epoch": epoch,
                    "global_step": global_step,
                    "local_step": idx,

                    "state_dict_backbone": model.module.backbone.state_dict(),
                    "state_dict_softmax_fc": module_partial_fc.state_dict(),
                    "state_dict_fam": model.module.fam.state_dict(),
                    "state_dict_tss": model.module.tss.state_dict(),
                    "state_dict_om": model.module.om.state_dict(),
                    "state_optimizer": opt.state_dict(),
                    "state_lr_scheduler": lr_scheduler.state_dict()
                }
                torch.save(checkpoint, os.path.join(cfg.output, f"checkpoint_recognition_step_{global_step}_gpu_{rank}.pt"))

            # update
            if global_step >= cfg.total_step - 1:
                break  # end
            else:
                global_step += 1

        if global_step >= cfg.total_step - 1:
            break
        if cfg.dali:
            train_loader.reset()

    with torch.no_grad():
        model.module.set_output_type("Recognition")
        callback_verification(global_step, model)

    distributed.destroy_process_group()


if __name__ == "__main__":
    torch.backends.cudnn.benchmark = True
    parser = argparse.ArgumentParser(
        description="Distributed Arcface Training for Recognition Only")
    parser.add_argument("config", type=str, help="py config file")
    # Support both --local_rank and --local-rank (for torch.distributed.launch compatibility)
    parser.add_argument("--local_rank", "--local-rank", type=int, default=None, help="local_rank")
    args = parser.parse_args()
    
    # If local_rank is not provided, try to get it from environment variable
    # This is the recommended way for newer PyTorch versions
    if args.local_rank is None:
        args.local_rank = int(os.environ.get('LOCAL_RANK', 0))
    
    main(args)

