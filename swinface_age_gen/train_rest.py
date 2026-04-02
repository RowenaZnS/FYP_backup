import argparse
import logging
import os
import multiprocessing
from itertools import cycle
import math

import torch
from torch import distributed
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter

from lr_scheduler import build_scheduler
from analysis import *
from analysis import subnets
from model import build_model

from utils.utils_callbacks import CallBackLogging
from utils.utils_logging import AverageMeter
import time
from torch import distributed as dist
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
        init_method="tcp://127.0.0.1:12585",
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
    # Set default device to cuda (but generator should be on CPU)
    # This must be set after cuda.set_device to ensure correct device

    os.makedirs(cfg.output, exist_ok=True)
    init_logging(rank, cfg.output)

    summary_writer = (
        SummaryWriter(log_dir=os.path.join(cfg.output, "tensorboard_rest"))
        if rank == 0
        else None
    )

    # Analysis dataloaders for rest tasks (Gender, CelebA, Expression)
    # Gender from age_gender dataset (we only use gender label)
    age_gender_train_loader = get_analysis_train_dataloader("age_gender", cfg, args.local_rank)
    CelebA_train_loader = get_analysis_train_dataloader("CelebA", cfg, args.local_rank)
    Expression_train_loader = get_analysis_train_dataloader("expression", cfg, args.local_rank)

    device = torch.device("cuda", args.local_rank)
    model = build_model(cfg).to(device)

    model = torch.nn.parallel.DistributedDataParallel(
        module=model, broadcast_buffers=False, device_ids=[args.local_rank], bucket_cap_mb=16,
        find_unused_parameters=True)
    model.train()
    # FIXME using gradient checkpoint if there are some unused parameters will cause error
    # model._set_static_graph()

    # Total batch size: gender + CelebA + expression
    cfg.total_batch_size = world_size * (cfg.age_gender_bz + cfg.CelebA_bz + cfg.expression_bz)
    # Use the longest dataloader for epoch calculation
    cfg.epoch_step = max(len(age_gender_train_loader), len(CelebA_train_loader), len(Expression_train_loader))

    cfg.num_epoch = math.ceil(cfg.total_step / cfg.epoch_step)

    cfg.lr = cfg.lr * cfg.total_batch_size / 512.0
    cfg.warmup_lr = cfg.warmup_lr * cfg.total_batch_size / 512.0
    cfg.min_lr = cfg.min_lr * cfg.total_batch_size / 512.0

    # Loss functions for analysis tasks
    # Gender: CrossEntropyLoss (binary classification, outputs[5])
    gender_loss = torch.nn.CrossEntropyLoss()
    # CelebA attributes: CrossEntropyLoss for each attribute (outputs[1:41], 40 attributes)
    celebA_losses = [torch.nn.CrossEntropyLoss() for _ in range(40)]  # 40 attributes
    # Expression: CrossEntropyLoss (outputs[41])
    expression_loss = torch.nn.CrossEntropyLoss()

    if cfg.optimizer == "sgd":
        opt = torch.optim.SGD(
            params=[{"params": model.module.backbone.parameters(), 'lr': cfg.lr / 10},
                    {"params": model.module.fam.parameters()},
                    {"params": model.module.tss.parameters()},
                    {"params": model.module.om.parameters()},
                    ],
            lr=cfg.lr, momentum=0.9, weight_decay=cfg.weight_decay)

    elif cfg.optimizer == "adamw":
        # Collect all parameters and ensure they are on CUDA
        all_params = []
        all_params.append({"params": model.module.backbone.parameters(), 'lr': cfg.lr / 10})
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
        dict_checkpoint = torch.load(os.path.join(cfg.output, f"checkpoint_step_{cfg.resume_step}_gpu_{rank}.pt"))
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

    # Validation dataloaders
    CelebA_loader = get_analysis_val_dataloader(data_choose="CelebA", config=cfg)
    RAF_loader = get_analysis_val_dataloader(data_choose="RAF", config=cfg)

    # Only create verification callbacks if data loaders are available
    CelebA_verification = CelebAVerification(data_loader=CelebA_loader, summary_writer=summary_writer) if CelebA_loader else None
    RAF_verification = RAFVerification(data_loader=RAF_loader, summary_writer=summary_writer) if RAF_loader else None

    # Custom logging callback without recognition loss
    class CallBackLoggingRest(object):
        def __init__(self, frequent, total_step, batch_size, start_step=0, writer=None):
            self.frequent: int = frequent
            self.rank: int = dist.get_rank()
            self.world_size: int = dist.get_world_size()
            self.time_start = time.time()
            self.total_step: int = total_step
            self.start_step: int = start_step
            self.batch_size: int = batch_size
            self.writer = writer
            self.init = False
            self.tic = 0

        def __call__(self, global_step: int, loss: AverageMeter, analysis_losses: list,
                     epoch: int, fp16: bool, learning_rate: float, grad_scaler):
            if self.rank == 0 and global_step > 0 and (global_step+1) % self.frequent == 0:
                if self.init:
                    try:
                        speed: float = self.frequent * self.batch_size / (time.time() - self.tic)
                        speed_total = speed * self.world_size
                    except ZeroDivisionError:
                        speed_total = float('inf')
                    time_now = time.time()
                    time_sec = int(time_now - self.time_start)
                    time_sec_avg = time_sec / (global_step - self.start_step + 1)
                    eta_sec = time_sec_avg * (self.total_step - global_step - 1)
                    time_for_end = eta_sec / 3600
                    if self.writer is not None:
                        self.writer.add_scalar('time_for_end', time_for_end, global_step)
                        self.writer.add_scalar('learning_rate', learning_rate, global_step)
                        self.writer.add_scalar('loss', loss.avg, global_step)
                        for j in range(42):
                            from analysis import ANALYSIS_TASKS
                            self.writer.add_scalar(ANALYSIS_TASKS[j] + ' Training Loss', analysis_losses[j].avg, global_step)
                    msg = "Speed %.2f samples/sec   Loss %.4f   " % (speed_total, loss.avg)
                    for j in range(42):
                        from analysis import ANALYSIS_TASKS
                        temp = ANALYSIS_TASKS[j] + " Loss %.4f   " % (analysis_losses[j].avg) + "   "
                        msg += temp
                    if fp16:
                        temp = "LearningRate %.6f   Epoch: %d   Global Step: %d   " \
                               "Fp16 Grad Scale: %2.f   Required: %1.f hours" % (
                                  learning_rate, epoch, global_step, grad_scaler.get_scale(), time_for_end)
                    else:
                        temp = "LearningRate %.6f   Epoch: %d   Global Step: %d   Required: %1.f hours" % (
                                   learning_rate, epoch, global_step, time_for_end)
                    msg += temp
                    msg += "\n\n"
                    logging.info(msg)
                    loss.reset()
                    for each in analysis_losses:
                        each.reset()
                    self.tic = time.time()
                else:
                    self.init = True
                    self.tic = time.time()
    
    callback_logging = CallBackLoggingRest(
        frequent=cfg.frequent,
        total_step=cfg.total_step,
        batch_size=cfg.batch_size,
        start_step=global_step,
        writer=summary_writer
    )

    loss_am = AverageMeter()
    analysis_loss_ams = [AverageMeter() for j in range(42)]

    amp = torch.cuda.amp.grad_scaler.GradScaler(growth_interval=100)

    # Batch size configuration: [gender, CelebA, expression]
    bzs = [cfg.age_gender_bz, cfg.CelebA_bz, cfg.expression_bz]
    features_cut = [0]
    for i in range(1, len(bzs) + 1):
        features_cut.append(features_cut[i - 1] + bzs[i - 1])

    for epoch in range(start_epoch, cfg.num_epoch):

        if isinstance(age_gender_train_loader, DataLoader):
            age_gender_train_loader.sampler.set_epoch(epoch)
        if isinstance(CelebA_train_loader, DataLoader):
            CelebA_train_loader.sampler.set_epoch(epoch)
        if isinstance(Expression_train_loader, DataLoader):
            Expression_train_loader.sampler.set_epoch(epoch)

        # Use zip with cycle to handle different dataloader lengths
        max_len = max(len(age_gender_train_loader), len(CelebA_train_loader), len(Expression_train_loader))
        age_gender_iter = cycle(age_gender_train_loader) if len(age_gender_train_loader) < max_len else iter(age_gender_train_loader)
        CelebA_iter = cycle(CelebA_train_loader) if len(CelebA_train_loader) < max_len else iter(CelebA_train_loader)
        Expression_iter = cycle(Expression_train_loader) if len(Expression_train_loader) < max_len else iter(Expression_train_loader)

        for idx in range(max_len):

            # skip
            if cfg.resume:
                if idx < local_step:
                    continue

            age_gender = next(age_gender_iter)
            CelebA = next(CelebA_iter)
            RAF = next(Expression_iter)

            # Extract data
            age_gender_img, [age_label, gender_label_1] = age_gender
            CelebA_img, CelebA_label = CelebA
            expression_img, expression_label = RAF

            # Move to CUDA
            gender_label = gender_label_1.cuda(non_blocking=True)
            # CelebA_label is a list/tuple of 40 tensors (one for each attribute)
            # Each tensor has shape [batch_size]
            expression_label = expression_label.cuda(non_blocking=True)

            # Concatenate images
            img = torch.cat([age_gender_img, CelebA_img, expression_img], dim=0).cuda(non_blocking=True)
            
            # Forward
            model.module.set_output_type("List")
            outputs = model(img)
            
            # -------------------------
            # 1) Gender loss (outputs[5])
            # -------------------------
            gender_output = outputs[5][features_cut[0]:features_cut[1]]
            gender_loss_val = gender_loss(gender_output, gender_label)
            
            # -------------------------
            # 2) CelebA attributes loss (outputs[1:41], 40 attributes)
            # -------------------------
            celebA_losses_val = []
            for j in range(40):  # 40 attributes
                attr_idx = j + 1  # outputs[1] to outputs[40]
                attr_output = outputs[attr_idx][features_cut[1]:features_cut[2]]
                # CelebA_label[j] is a tensor of shape [batch_size] for the j-th attribute
                attr_label = CelebA_label[j].cuda(non_blocking=True)
                attr_loss = celebA_losses[j](attr_output, attr_label)
                celebA_losses_val.append(attr_loss)
            celebA_total_loss = sum(celebA_losses_val)
            
            # -------------------------
            # 3) Expression loss (outputs[41])
            # -------------------------
            expression_output = outputs[41][features_cut[2]:features_cut[3]]
            expression_loss_val = expression_loss(expression_output, expression_label)
            
            # -------------------------
            # 4) Total loss
            # -------------------------
            loss = (cfg.analysis_loss_weights[5] * gender_loss_val + 
                   sum([cfg.analysis_loss_weights[j+1] * celebA_losses_val[j] for j in range(40)]) +
                   cfg.analysis_loss_weights[41] * expression_loss_val)
            
            # -------------------------
            # 5) Backward + Optim step
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
            # 6) Logging
            # -------------------------
            with torch.no_grad():
                loss_am.update(loss.item(), 1)
            
                # Update loss meters
                analysis_loss_ams[5].update(gender_loss_val.item(), 1)  # Gender at index 5
                for j in range(40):
                    analysis_loss_ams[j+1].update(celebA_losses_val[j].item(), 1)  # CelebA attributes at indices 1-40
                analysis_loss_ams[41].update(expression_loss_val.item(), 1)  # Expression at index 41
            
                callback_logging(
                    global_step,
                    loss_am,
                    analysis_loss_ams,
                    epoch,
                    cfg.fp16,
                    opt.param_groups[0]["lr"],
                    amp
                )
            
                if (global_step + 1) % cfg.verbose == 0:
                    model.module.set_output_type("Attribute")
                    if CelebA_verification:
                        CelebA_verification(global_step, model)
                    model.module.set_output_type("Expression")
                    if RAF_verification:
                        RAF_verification(global_step, model)

            if cfg.save_all_states and (global_step+1) % cfg.save_verbose == 0:
                checkpoint = {
                    "epoch": epoch,
                    "global_step": global_step,
                    "local_step": idx,

                    "state_dict_backbone": model.module.backbone.state_dict(),
                    "state_dict_fam": model.module.fam.state_dict(),
                    "state_dict_tss": model.module.tss.state_dict(),
                    "state_dict_om": model.module.om.state_dict(),
                    "state_optimizer": opt.state_dict(),
                    "state_lr_scheduler": lr_scheduler.state_dict()
                }
                torch.save(checkpoint, os.path.join(cfg.output, f"checkpoint_rest_step_{global_step}_gpu_{rank}.pt"))

            # update
            if global_step >= cfg.total_step - 1:
                break  # end
            else:
                global_step += 1

        if global_step >= cfg.total_step - 1:
            break
        if cfg.dali:
            if hasattr(age_gender_train_loader, 'reset'):
                age_gender_train_loader.reset()

    with torch.no_grad():
        model.module.set_output_type("Attribute")
        if CelebA_verification:
            CelebA_verification(global_step, model)
        model.module.set_output_type("Expression")
        if RAF_verification:
            RAF_verification(global_step, model)

    distributed.destroy_process_group()


if __name__ == "__main__":
    torch.backends.cudnn.benchmark = True
    parser = argparse.ArgumentParser(
        description="Distributed Training for Rest Analysis Tasks (Gender, CelebA, Expression)")
    parser.add_argument("config", type=str, help="py config file")
    # Support both --local_rank and --local-rank (for torch.distributed.launch compatibility)
    parser.add_argument("--local_rank", "--local-rank", type=int, default=None, help="local_rank")
    args = parser.parse_args()
    
    # If local_rank is not provided, try to get it from environment variable
    # This is the recommended way for newer PyTorch versions
    if args.local_rank is None:
        args.local_rank = int(os.environ.get('LOCAL_RANK', 0))
    
    main(args)

