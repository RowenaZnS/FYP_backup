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
from timm.loss import LabelSmoothingCrossEntropy, SoftTargetCrossEntropy

from dataset import get_dataloader
from losses import CombinedMarginLoss
from lr_scheduler import build_scheduler
from partial_fc import PartialFC, PartialFCAdamW
from analysis import *
from analysis import subnets
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
        init_method="tcp://127.0.0.1:12584",
        rank=rank,
        world_size=world_size,
    )


def main(args):
    # get config
    cfg = get_config(args.config)
    # global control random seed
    setup_seed(seed=cfg.seed, cuda_deterministic=False)

    torch.cuda.set_device(args.local_rank)
    
    # Set default device to cuda (but generator should be on CPU)
    # This must be set after cuda.set_device to ensure correct device
    torch.set_default_device("cuda")

    os.makedirs(cfg.output, exist_ok=True)
    init_logging(rank, cfg.output)

    summary_writer = (
        SummaryWriter(log_dir=os.path.join(cfg.output, "tensorboard"))
        if rank == 0
        else None
    )

    # Recognition dataloader
    '''
    train_loader = get_dataloader(
        cfg.rec,
        args.local_rank,
        cfg.batch_size,
        cfg.dali,
        cfg.seed,
        cfg.num_workers
    )
    '''
    train_loader = get_analysis_train_dataloader("recognition", cfg, args.local_rank)
    # Analysis dataloaders
    age_gender_train_loader = get_analysis_train_dataloader("age_gender", cfg, args.local_rank)
    CelebA_train_loader = get_analysis_train_dataloader("CelebA", cfg, args.local_rank)
    Expression_train_loader = get_analysis_train_dataloader("expression", cfg, args.local_rank)

    model = build_model(cfg).cuda()

    model = torch.nn.parallel.DistributedDataParallel(
        module=model, broadcast_buffers=False, device_ids=[args.local_rank], bucket_cap_mb=16,
        find_unused_parameters=True)
    model.train()
    # FIXME using gradient checkpoint if there are some unused parameters will cause error
    model._set_static_graph()

    cfg.total_batch_size = world_size * (cfg.recognition_bz + cfg.age_gender_bz + cfg.CelebA_bz + cfg.expression_bz)
    cfg.epoch_step = len(train_loader)

    cfg.num_epoch = math.ceil(cfg.total_step / cfg.epoch_step)

    #cfg.total_recognition_bz = cfg.recognition_bz * world_size
    #cfg.warmup_step = cfg.num_image // cfg.total_recognition_bz * cfg.warmup_epoch
    #cfg.total_step = cfg.num_image // cfg.total_recognition_bz * cfg.num_epoch

    cfg.lr = cfg.lr * cfg.total_batch_size / 512.0
    cfg.warmup_lr = cfg.warmup_lr * cfg.total_batch_size / 512.0
    cfg.min_lr = cfg.min_lr * cfg.total_batch_size / 512.0

    # Helper function to print all parameter devices
    def print_parameter_devices(optimizer, step_num=None):
        """Print device information for all parameters and their optimizer states"""
        if rank != 0:  # Only print from rank 0 to avoid duplicate output
            return
        
        print(f"\n{'='*80}")
        if step_num is not None:
            print(f"Parameter Device Check at Step {step_num}")
        else:
            print(f"Parameter Device Check (Initial)")
        print(f"{'='*80}")
        
        for group_idx, param_group in enumerate(optimizer.param_groups):
            print(f"\nParameter Group {group_idx}:")
            if 'lr' in param_group:
                print(f"  Learning Rate: {param_group['lr']}")
            
            for param_idx, param in enumerate(param_group['params']):
                param_name = f"Group{group_idx}_Param{param_idx}"
                param_shape = tuple(param.shape)
                param_device = param.device
                param_dtype = param.dtype
                
                print(f"  {param_name}:")
                print(f"    Shape: {param_shape}")
                print(f"    Device: {param_device} ({'✓ GPU' if param_device.type == 'cuda' else '✗ CPU'})")
                print(f"    Dtype: {param_dtype}")
                
                if param in optimizer.state:
                    state = optimizer.state[param]
                    print(f"    Optimizer States:")
                    for state_key, state_value in state.items():
                        if isinstance(state_value, torch.Tensor):
                            state_device = state_value.device
                            state_dtype = state_value.dtype
                            state_shape = tuple(state_value.shape)
                            
                            # Determine correct device status based on state type
                            if state_key == 'step':
                                # step should be on CPU
                                if state_device.type == 'cpu':
                                    device_status = '✓ CPU (correct - step should be on CPU)'
                                else:
                                    device_status = f'✗ {state_device.type.upper()} (WRONG - step should be on CPU)'
                            else:
                                # exp_avg and exp_avg_sq should match parameter device
                                if state_device == param_device:
                                    device_status = f'✓ {state_device.type.upper()} (correct - matches param device)'
                                else:
                                    device_status = f'✗ {state_device.type.upper()} (WRONG - should be on {param_device})'
                            
                            print(f"      {state_key}:")
                            print(f"        Shape: {state_shape}")
                            print(f"        Device: {state_device} {device_status}")
                            print(f"        Dtype: {state_dtype}")
                            
                            # Check if device matches and print warning if needed
                            if state_key == 'step':
                                if state_device.type != 'cpu':
                                    print(f"        ⚠️  WARNING: step should be on CPU but is on {state_device}")
                            else:
                                if state_device != param_device:
                                    print(f"        ⚠️  WARNING: Device mismatch! Param on {param_device}, state on {state_device}")
                        else:
                            print(f"      {state_key}: {type(state_value).__name__} (not a tensor)")
                else:
                    print(f"    Optimizer States: None (parameter not yet used by optimizer)")
        
        print(f"{'='*80}\n")

    # Helper function to ensure all optimizer states are on CUDA
    def ensure_optimizer_states_on_cuda(optimizer):
        """Ensure all optimizer state tensors are on CUDA device"""
        for param_group in optimizer.param_groups:
            for param in param_group['params']:
                if param in optimizer.state:
                    state = optimizer.state[param]
                    for key, value in state.items():
                        if isinstance(value, torch.Tensor):
                            if key == 'step':
                                # step should be on CPU
                                if value.device.type != 'cpu':
                                    state[key] = value.cpu()
                            else:
                                # Other states should be on same device as parameter
                                if value.device != param.device:
                                    state[key] = value.to(device=param.device)

    # Recognition loss
    margin_loss = CombinedMarginLoss(
        64,
        cfg.margin_list[0],
        cfg.margin_list[1],
        cfg.margin_list[2],
        cfg.interclass_filtering_threshold
    )
    age_loss = AgeLoss(total_iter=cfg.total_step)

    # Analysis task_losses
    criteria = [age_loss]
    criteria.extend([torch.nn.CrossEntropyLoss() for j in range(41)])  # Total:42

    if cfg.optimizer == "sgd":
        module_partial_fc = PartialFC(
            margin_loss, cfg.embedding_size, cfg.num_classes,
            cfg.sample_rate, cfg.fp16)
        module_partial_fc.train().cuda()
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
        module_partial_fc.train().cuda()
        
        # Collect all parameters and ensure they are on CUDA
        all_params = []
        all_params.append({"params": model.module.backbone.parameters(), 'lr': cfg.lr / 10})
        all_params.append({"params": module_partial_fc.parameters()})
        all_params.append({"params": model.module.fam.parameters()})
        all_params.append({"params": model.module.tss.parameters()})
        all_params.append({"params": model.module.om.parameters()})
        
        # Ensure all parameters are on the correct device before creating optimizer
        for param_group in all_params:
            for param in param_group['params']:
                if param.device.type != 'cuda':
                    param.data = param.data.cuda(args.local_rank)
        
        opt = torch.optim.AdamW(
            params=all_params,
            lr=cfg.lr, 
            weight_decay=cfg.weight_decay,
            foreach=False)  # Disable foreach to avoid device mismatch issues with PyTorch 2.0+
        
        # Print parameter devices after optimizer creation
        print_parameter_devices(opt, step_num="Initial")
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
            # module_partial_fc.load_state_dict(dict_checkpoint["state_dict_softmax_fc"])                                   
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
        module_partial_fc.load_state_dict(dict_checkpoint["state_dict_softmax_fc"])
        model.module.fam.load_state_dict(dict_checkpoint["state_dict_fam"])
        model.module.tss.load_state_dict(dict_checkpoint["state_dict_tss"])
        model.module.om.load_state_dict(dict_checkpoint["state_dict_om"])
        opt.load_state_dict(dict_checkpoint["state_optimizer"])
        lr_scheduler.load_state_dict(dict_checkpoint["state_lr_scheduler"])
        del dict_checkpoint

    for key, value in cfg.items():
        num_space = 25 - len(key)
        logging.info(": " + key + " " * num_space + str(value))

    callback_verification = CallBackVerification(
        val_targets=cfg.val_targets, rec_prefix=cfg.rec, summary_writer=summary_writer
    )

    FGNet_loader = get_analysis_val_dataloader(data_choose="FGNet", config=cfg)
    LAP_loader = get_analysis_val_dataloader(data_choose="LAP", config=cfg)
    CelebA_loader = get_analysis_val_dataloader(data_choose="CelebA", config=cfg)
    RAF_loader = get_analysis_val_dataloader(data_choose="RAF", config=cfg)

    # Only create verification callbacks if data loaders are available
    FGNet_verification = FGNetVerification(data_loader=FGNet_loader, summary_writer=summary_writer) if FGNet_loader else None
    LAP_verification = LAPVerification(data_loader=LAP_loader, summary_writer=summary_writer) if LAP_loader else None
    CelebA_verification = CelebAVerification(data_loader=CelebA_loader, summary_writer=summary_writer) if CelebA_loader else None
    RAF_verification = RAFVerification(data_loader=RAF_loader, summary_writer=summary_writer) if RAF_loader else None

    callback_logging = CallBackLogging(
        frequent=cfg.frequent,
        total_step=cfg.total_step,
        batch_size=cfg.batch_size,
        start_step=global_step,
        writer=summary_writer
    )

    loss_am = AverageMeter()
    recognition_loss_am = AverageMeter()
    analysis_loss_ams = [AverageMeter() for j in range(42)]

    amp = torch.cuda.amp.grad_scaler.GradScaler(growth_interval=100)

    bzs = [cfg.recognition_bz, cfg.age_gender_bz, cfg.CelebA_bz, cfg.expression_bz]

    features_cut = [0 for i in range(5)]
    for i in range(1, 5):
        features_cut[i] = features_cut[i - 1] + bzs[i - 1]

    '''
    with torch.no_grad():
        model.module.set_output_type("Recognition")
        callback_verification(global_step, model)
        model.module.set_output_type("Age")
        FGNet_verification(global_step, model)
        LAP_verification(global_step, model)
        model.module.set_output_type("Attribute")
        CelebA_verification(global_step, model)
        model.module.set_output_type("Expression")
        RAF_verification(global_step, model)
    '''

    for epoch in range(start_epoch, cfg.num_epoch):

        if isinstance(train_loader, DataLoader):
            train_loader.sampler.set_epoch(epoch)
        if isinstance(age_gender_train_loader, DataLoader):
            age_gender_train_loader.sampler.set_epoch(epoch)
        if isinstance(CelebA_train_loader, DataLoader):
            CelebA_train_loader.sampler.set_epoch(epoch)
        if isinstance(Expression_train_loader, DataLoader):
            Expression_train_loader.sampler.set_epoch(epoch)

        for idx, data in enumerate(
                zip(train_loader, age_gender_train_loader, CelebA_train_loader, Expression_train_loader)):

            # skip
            if cfg.resume:
                if idx < local_step:
                    continue

            recognition = data[0]
            age_gender = data[1]
            CelebA = data[2]
            RAF = data[3]

            recognition_img, recognition_label = recognition
            age_gender_img, [age_label, gender_label_1] = age_gender
            CelebA_img, CelebA_label = CelebA
            expression_img, expression_label = RAF
            gender_label = torch.cat([gender_label_1, CelebA_label[4]])

            recognition_label = recognition_label.cuda(non_blocking=True)
            age_label = age_label.cuda(non_blocking=True)
            gender_label = gender_label.cuda(non_blocking=True)
            expression_label = expression_label.cuda(non_blocking=True)

            analysis_labels = [age_label]
            for j in range(40):
                analysis_labels.append(CelebA_label[j].cuda(non_blocking=True))
            analysis_labels[5] = gender_label
            analysis_labels.append(expression_label)

            img = torch.cat([recognition_img, age_gender_img, CelebA_img, expression_img], dim=0).cuda(
                non_blocking=True)

            # Concat images from different dataloaders
            model.module.set_output_type("List")
            outputs = model(img)

            local_embeddings = outputs[-1][features_cut[0]: features_cut[1]]
            recognition_loss = module_partial_fc(local_embeddings, recognition_label, opt)

            analysis_losses = []

            for j in range(42):
                if j == 0:  # age
                    analysis_output = outputs[j][features_cut[1]: features_cut[2]]
                    analysis_loss = criteria[j](analysis_output, analysis_labels[j], global_step)
                elif j == 5:
                    analysis_output = outputs[j][features_cut[1]: features_cut[3]]
                    analysis_loss = criteria[j](analysis_output, analysis_labels[j])
                elif j == 41:
                    analysis_output = outputs[j][features_cut[3]: features_cut[4]]
                    analysis_loss = criteria[j](analysis_output, analysis_labels[j])
                else:
                    analysis_output = outputs[j][features_cut[2]: features_cut[3]]
                    analysis_loss = criteria[j](analysis_output, analysis_labels[j])

                analysis_losses.append(analysis_loss)

            loss = cfg.recognition_loss_weight * recognition_loss

            for j in range(42):
                loss += analysis_losses[j] * cfg.analysis_loss_weights[j]

            if cfg.fp16:
                amp.scale(loss).backward()
                amp.unscale_(opt)
                torch.nn.utils.clip_grad_norm_(model.module.backbone.parameters(), 5)
                # Ensure all optimizer states are on correct device before step
                if cfg.optimizer == "adamw":
                    # Print devices before step (only at first few steps or when verbose)
                    if global_step < 3 or (global_step % cfg.verbose == 0):
                        print_parameter_devices(opt, step_num=global_step)
                    ensure_optimizer_states_on_cuda(opt)
                amp.step(opt)
                amp.update()
            else:
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.module.backbone.parameters(), 5)
                # Ensure all optimizer states are on correct device before step
                if cfg.optimizer == "adamw":
                    # Print devices before step (only at first few steps or when verbose)
                    if global_step < 3 or (global_step % cfg.verbose == 0):
                        print_parameter_devices(opt, step_num=global_step)
                    ensure_optimizer_states_on_cuda(opt)
                opt.step()

            opt.zero_grad()
            lr_scheduler.step_update(global_step)

            with torch.no_grad():
                loss_am.update(loss.item(), 1)
                recognition_loss_am.update(recognition_loss.item(), 1)
                for j in range(42):
                    analysis_loss_ams[j].update(analysis_losses[j].item(), 1)

                # Get current learning rate from optimizer (scheduler has already updated it via step_update)
                current_lr = opt.param_groups[0]['lr']
                callback_logging(global_step, loss_am, recognition_loss_am, analysis_loss_ams, epoch, cfg.fp16,
                                 current_lr, amp)

                if (global_step+1) % cfg.verbose == 0:
                    model.module.set_output_type("Recognition")
                    callback_verification(global_step, model)
                    model.module.set_output_type("Age")
                    if FGNet_verification:
                        FGNet_verification(global_step, model)
                    if LAP_verification:
                        LAP_verification(global_step, model)
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
                    "state_dict_softmax_fc": module_partial_fc.state_dict(),
                    "state_dict_fam": model.module.fam.state_dict(),
                    "state_dict_tss": model.module.tss.state_dict(),
                    "state_dict_om": model.module.om.state_dict(),
                    "state_optimizer": opt.state_dict(),
                    "state_lr_scheduler": lr_scheduler.state_dict()
                }
                torch.save(checkpoint, os.path.join(cfg.output, f"checkpoint_step_{global_step}_gpu_{rank}.pt"))

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
        model.module.set_output_type("Age")
        if FGNet_verification:
            FGNet_verification(global_step, model)
        if LAP_verification:
            LAP_verification(global_step, model)
        model.module.set_output_type("Attribute")
        if CelebA_verification:
            CelebA_verification(global_step, model)
        model.module.set_output_type("Expression")
        if RAF_verification:
            RAF_verification(global_step, model)

    # if rank == 0:
    # path_module = os.path.join(cfg.output, "model.pt")
    # torch.save(backbone.module.state_dict(), path_module)

    # from torch2onnx import convert_onnx
    # convert_onnx(backbone.module.cpu().eval(), path_module, os.path.join(cfg.output, "model.onnx"))

    distributed.destroy_process_group()


if __name__ == "__main__":
    torch.backends.cudnn.benchmark = True
    parser = argparse.ArgumentParser(
        description="Distributed Arcface Training in Pytorch")
    parser.add_argument("config", type=str, help="py config file")
    # Support both --local_rank and --local-rank (for torch.distributed.launch compatibility)
    parser.add_argument("--local_rank", "--local-rank", type=int, default=None, help="local_rank")
    args = parser.parse_args()
    
    # If local_rank is not provided, try to get it from environment variable
    # This is the recommended way for newer PyTorch versions
    if args.local_rank is None:
        args.local_rank = int(os.environ.get('LOCAL_RANK', 0))
    
    main(args)
