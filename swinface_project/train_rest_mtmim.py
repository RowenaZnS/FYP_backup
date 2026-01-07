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
from analysis.mi_estimator import MIEstimator
from analysis.mi_loss import MILoss, shuffle_batch
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
        init_method="tcp://127.0.0.1:12589",
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

    # Create tensorboard directory: tensorboard_rest/new for comparison
    tensorboard_dir = os.path.join(cfg.output, "tensorboard_rest", "new")
    os.makedirs(tensorboard_dir, exist_ok=True)
    
    summary_writer = (
        SummaryWriter(log_dir=tensorboard_dir)
        if rank == 0
        else None
    )

    # Analysis dataloaders for rest tasks (Gender, CelebA, Expression)
    # Note: We use age_gender loader to get both gender labels and age labels (for MI loss)
    # But we don't train age task, only use age labels for MI minimization
    age_gender_train_loader = get_analysis_train_dataloader("age_gender", cfg, args.local_rank)
    CelebA_train_loader = get_analysis_train_dataloader("CelebA", cfg, args.local_rank)
    Expression_train_loader = get_analysis_train_dataloader("expression", cfg, args.local_rank)

    device = torch.device("cuda", args.local_rank)
    model = build_model(cfg).to(device)

    model = torch.nn.parallel.DistributedDataParallel(
        module=model, broadcast_buffers=False, device_ids=[args.local_rank], bucket_cap_mb=16,
        find_unused_parameters=True)
    model.train()

    # Total batch size: age_gender (for gender labels and age labels for MI) + CelebA + expression
    cfg.total_batch_size = world_size * (cfg.age_gender_bz + cfg.CelebA_bz + cfg.expression_bz)
    # Use the longest dataloader for epoch calculation
    cfg.epoch_step = max(len(age_gender_train_loader), len(CelebA_train_loader), len(Expression_train_loader))

    cfg.num_epoch = math.ceil(cfg.total_step / cfg.epoch_step)

    cfg.lr = cfg.lr * cfg.total_batch_size / 512.0
    cfg.warmup_lr = cfg.warmup_lr * cfg.total_batch_size / 512.0
    cfg.min_lr = cfg.min_lr * cfg.total_batch_size / 512.0

    # Loss functions for analysis tasks (no age loss - only Gender, CelebA, Expression)
    # Gender: CrossEntropyLoss (binary classification, outputs[5])
    gender_loss = torch.nn.CrossEntropyLoss()
    # CelebA attributes: CrossEntropyLoss for each attribute (outputs[1:41], 40 attributes)
    celebA_losses = [torch.nn.CrossEntropyLoss() for _ in range(40)]  # 40 attributes
    # Expression: CrossEntropyLoss (outputs[41])
    expression_loss = torch.nn.CrossEntropyLoss()

    # MT-MIM components (must be enabled)
    if not getattr(cfg, 'use_mt_mim', False):
        logging.warning("MT-MIM is not enabled in config, but train_rest_mtmim.py requires it. Enabling MT-MIM...")
        cfg.use_mt_mim = True
    
    logging.info("MT-MIM mode enabled - initializing MI Estimator and MI Loss")
    mi_estimator = MIEstimator(
        input_dim=cfg.embedding_size,
        hidden_dim=getattr(cfg, 'mi_estimator_hidden_dim', 256),
        output_dim=cfg.embedding_size
    ).to(device)
    
    mi_loss_fn = MILoss(temperature=getattr(cfg, 'mi_temperature', 0.1))
    
    # Separate optimizer for MI Estimator
    mi_opt = torch.optim.Adam(
        mi_estimator.parameters(), 
        lr=getattr(cfg, 'mi_estimator_lr', 1e-4)
    )

    if cfg.optimizer == "sgd":
        param_groups = [
            {"params": model.module.backbone.parameters(), 'lr': cfg.lr / 10},
            {"params": model.module.fam.parameters()},
            {"params": model.module.tss.parameters()},
            {"params": model.module.om.parameters()},
        ]
        
        # Add MT-MIM parameters if enabled
        if getattr(cfg, 'use_mt_mim', False):
            param_groups.append({"params": model.module.age_extractor.parameters()})
            param_groups.append({"params": model.module.mt_mim_age_head.parameters()})
            param_groups.append({"params": model.module.mt_mim_id_head.parameters()})
        
        opt = torch.optim.SGD(
            params=param_groups,
            lr=cfg.lr, momentum=0.9, weight_decay=cfg.weight_decay)

    elif cfg.optimizer == "adamw":
        # Collect all parameters and ensure they are on CUDA
        all_params = []
        all_params.append({"params": model.module.backbone.parameters(), 'lr': cfg.lr / 10})
        all_params.append({"params": model.module.fam.parameters()})
        all_params.append({"params": model.module.tss.parameters()})
        all_params.append({"params": model.module.om.parameters()})
        
        # Add MT-MIM parameters if enabled
        if getattr(cfg, 'use_mt_mim', False):
            all_params.append({"params": model.module.age_extractor.parameters()})
            all_params.append({"params": model.module.mt_mim_age_head.parameters()})
            all_params.append({"params": model.module.mt_mim_id_head.parameters()})
        
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
        checkpoint_path = os.path.join(cfg.output, f"checkpoint_rest_mtmim_step_{cfg.resume_step}_gpu_{rank}.pt")
        if os.path.exists(checkpoint_path):
            dict_checkpoint = torch.load(checkpoint_path)
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
            
            # Load MT-MIM components if enabled and available
            if getattr(cfg, 'use_mt_mim', False):
                if "state_dict_age_extractor" in dict_checkpoint:
                    model.module.age_extractor.load_state_dict(dict_checkpoint["state_dict_age_extractor"])
                    logging.info("Loaded age_extractor state")
                if "state_dict_mt_mim_age_head" in dict_checkpoint:
                    model.module.mt_mim_age_head.load_state_dict(dict_checkpoint["state_dict_mt_mim_age_head"])
                    logging.info("Loaded mt_mim_age_head state")
                if "state_dict_mt_mim_id_head" in dict_checkpoint:
                    model.module.mt_mim_id_head.load_state_dict(dict_checkpoint["state_dict_mt_mim_id_head"])
                    logging.info("Loaded mt_mim_id_head state")
                if mi_estimator is not None and "state_dict_mi_estimator" in dict_checkpoint:
                    mi_estimator.load_state_dict(dict_checkpoint["state_dict_mi_estimator"])
                    logging.info("Loaded mi_estimator state")
                if mi_opt is not None and "state_mi_optimizer" in dict_checkpoint:
                    mi_opt.load_state_dict(dict_checkpoint["state_mi_optimizer"])
                    logging.info("Loaded mi_optimizer state")
            
            del dict_checkpoint
        else:
            if rank == 0:
                logging.warning(f"Checkpoint not found at {checkpoint_path}, starting from scratch")

    for key, value in cfg.items():
        num_space = 25 - len(key)
        logging.info(": " + key + " " * num_space + str(value))

    # Validation dataloaders
    CelebA_loader = get_analysis_val_dataloader(data_choose="CelebA", config=cfg)
    RAF_loader = get_analysis_val_dataloader(data_choose="RAF", config=cfg)

    # Only create verification callbacks if data loaders are available
    CelebA_verification = CelebAVerification(data_loader=CelebA_loader, summary_writer=summary_writer) if CelebA_loader else None
    RAF_verification = RAFVerification(data_loader=RAF_loader, summary_writer=summary_writer) if RAF_loader else None

    # Custom logging callback for MT-MIM rest tasks
    class CallBackLoggingRestMTMIM(object):
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
                     mi_loss: AverageMeter, epoch: int, fp16: bool, learning_rate: float, grad_scaler):
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
                        self.writer.add_scalar('Loss/MT_MIM_MI', mi_loss.avg, global_step)
                        for j in range(42):
                            from analysis import ANALYSIS_TASKS
                            self.writer.add_scalar(ANALYSIS_TASKS[j] + ' Training Loss', analysis_losses[j].avg, global_step)
                    msg = "Speed %.2f samples/sec   Loss %.4f   MI Loss %.4f   " % (speed_total, loss.avg, mi_loss.avg)
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
                    mi_loss.reset()
                    self.tic = time.time()
                else:
                    self.init = True
                    self.tic = time.time()
    
    callback_logging = CallBackLoggingRestMTMIM(
        frequent=cfg.frequent,
        total_step=cfg.total_step,
        batch_size=cfg.total_batch_size,
        start_step=global_step,
        writer=summary_writer
    )

    loss_am = AverageMeter()
    analysis_loss_ams = [AverageMeter() for j in range(42)]
    mi_loss_am = AverageMeter()

    amp = torch.cuda.amp.grad_scaler.GradScaler(growth_interval=100)

    # Batch size configuration: [age_gender, CelebA, expression]
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
            # age_label is only used for MI loss calculation, not for age task training
            age_label = age_label.cuda(non_blocking=True)
            gender_label = gender_label_1.cuda(non_blocking=True)
            # CelebA_label is a list/tuple of 40 tensors (one for each attribute)
            expression_label = expression_label.cuda(non_blocking=True)

            # Concatenate images
            img = torch.cat([age_gender_img, CelebA_img, expression_img], dim=0).cuda(non_blocking=True)
            
            # =========================================
            # MT-MIM Forward Pass
            # =========================================
            model.module.set_output_type("MT_MIM")
            mt_mim_outputs = model(img)
            
            # Get separated features
            embedding = mt_mim_outputs['embedding']
            x_age = mt_mim_outputs['x_age']
            x_id = mt_mim_outputs['x_id']
            # age_pred = mt_mim_outputs['age_pred']  # Not used since we don't train age
            id_embedding = mt_mim_outputs['id_embedding']
            
            # Convert id_embedding to float32 for consistency
            id_embedding = id_embedding.float()
            
            # Also get standard outputs for other tasks
            standard_outputs = mt_mim_outputs['standard_output']
            
            # -------------------------
            # 1) Gender loss (outputs[5])
            # -------------------------
            gender_output = standard_outputs[5][features_cut[0]:features_cut[1]]
            gender_loss_val = gender_loss(gender_output, gender_label)
            
            # -------------------------
            # 2) CelebA attributes loss (outputs[1:41], 40 attributes)
            # -------------------------
            celebA_losses_val = []
            for j in range(40):  # 40 attributes
                attr_idx = j + 1  # outputs[1] to outputs[40]
                attr_output = standard_outputs[attr_idx][features_cut[1]:features_cut[2]]
                attr_label = CelebA_label[j].cuda(non_blocking=True)
                attr_loss = celebA_losses[j](attr_output, attr_label)
                celebA_losses_val.append(attr_loss)
            celebA_total_loss = sum(celebA_losses_val)
            
            # -------------------------
            # 3) Expression loss (outputs[41])
            # -------------------------
            expression_output = standard_outputs[41][features_cut[2]:features_cut[3]]
            expression_loss_val = expression_loss(expression_output, expression_label)
            
            # -------------------------
            # 4) MI Minimization Loss
            # -------------------------
            # Compute MI loss on age_gender samples (where we have age labels for MI calculation)
            # Note: We use age labels only for MI loss, not for age task training
            age_x_id = x_id[features_cut[0]:features_cut[1]]
            age_x_age = x_age[features_cut[0]:features_cut[1]]
            
            # Shuffle x_age to create negative pairs
            age_x_age_shuffled = shuffle_batch(age_x_age)
            
            # MI loss: minimize correlation between x_id and x_age
            mi_loss_val = mi_loss_fn(age_x_id, age_x_age, age_x_age_shuffled)
            
            # -------------------------
            # 5) Total loss (no age loss, only Gender + CelebA + Expression + MI)
            # -------------------------
            mi_weight = getattr(cfg, 'mi_loss_weight', 0.1)
            loss = (cfg.analysis_loss_weights[5] * gender_loss_val +  # Gender loss
                   sum([cfg.analysis_loss_weights[j+1] * celebA_losses_val[j] for j in range(40)]) +  # CelebA losses
                   cfg.analysis_loss_weights[41] * expression_loss_val +  # Expression loss
                   mi_weight * mi_loss_val)  # MI loss
            
            # -------------------------
            # 6) Backward + Optim step
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
            
            # Update MI Estimator separately
            if mi_opt is not None and global_step % getattr(cfg, 'update_mi_estimator_every', 1) == 0:
                mi_estimator.train()
                with torch.no_grad():
                    age_x_id_for_mi = age_x_id.detach()
                    age_x_age_for_mi = age_x_age.detach()
                
                x_age_pred = mi_estimator(age_x_id_for_mi)
                mi_est_loss = torch.nn.functional.mse_loss(x_age_pred, age_x_age_for_mi)
                mi_opt.zero_grad()
                mi_est_loss.backward()
                mi_opt.step()
            
            lr_scheduler.step_update(global_step)
            
            # -------------------------
            # 7) Logging
            # -------------------------
            with torch.no_grad():
                loss_am.update(loss.item(), 1)
            
                # Update loss meters (no age loss at index 0)
                # Note: analysis_loss_ams[0] for Age is not updated since we don't train age
                analysis_loss_ams[5].update(gender_loss_val.item(), 1)  # Gender at index 5
                for j in range(40):
                    analysis_loss_ams[j+1].update(celebA_losses_val[j].item(), 1)  # CelebA attributes at indices 1-40
                analysis_loss_ams[41].update(expression_loss_val.item(), 1)  # Expression at index 41
                mi_loss_am.update(mi_loss_val.item(), 1)
            
                callback_logging(
                    global_step,
                    loss_am,
                    analysis_loss_ams,
                    mi_loss_am,
                    epoch,
                    cfg.fp16,
                    opt.param_groups[0]["lr"],
                    amp
                )
                
                # Additional detailed logging for MT-MIM
                if rank == 0 and global_step > 0 and (global_step + 1) % cfg.frequent == 0:
                    logging.info(f"MT-MIM Rest Tasks (No Age) - Gender: {gender_loss_val.item():.4f}, "
                               f"CelebA: {celebA_total_loss.item():.4f}, "
                               f"Expression: {expression_loss_val.item():.4f}, "
                               f"MI: {mi_loss_val.item():.4f}, "
                               f"Total: {loss.item():.4f}")
                
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
                
                # Save MT-MIM components
                if getattr(cfg, 'use_mt_mim', False):
                    checkpoint["state_dict_age_extractor"] = model.module.age_extractor.state_dict()
                    checkpoint["state_dict_mt_mim_age_head"] = model.module.mt_mim_age_head.state_dict()
                    checkpoint["state_dict_mt_mim_id_head"] = model.module.mt_mim_id_head.state_dict()
                    if mi_estimator is not None:
                        checkpoint["state_dict_mi_estimator"] = mi_estimator.state_dict()
                    if mi_opt is not None:
                        checkpoint["state_mi_optimizer"] = mi_opt.state_dict()
                
                torch.save(checkpoint, os.path.join(cfg.output, f"checkpoint_rest_mtmim_step_{global_step}_gpu_{rank}.pt"))

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
        description="Distributed Training for Rest Analysis Tasks with MT-MIM (Gender, CelebA, Expression)")
    parser.add_argument("config", type=str, help="py config file")
    # Support both --local_rank and --local-rank (for torch.distributed.launch compatibility)
    parser.add_argument("--local_rank", "--local-rank", type=int, default=None, help="local_rank")
    args = parser.parse_args()
    
    # If local_rank is not provided, try to get it from environment variable
    # This is the recommended way for newer PyTorch versions
    if args.local_rank is None:
        args.local_rank = int(os.environ.get('LOCAL_RANK', 0))
    
    main(args)

