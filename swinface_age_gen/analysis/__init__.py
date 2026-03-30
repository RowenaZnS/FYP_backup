
from .losses import AgeLoss
from .verification import FGNetVerification, CelebAVerification, RAFVerification, LAPVerification
from .task_name import ANALYSIS_TASKS
from timm.data.constants import IMAGENET_DEFAULT_MEAN, IMAGENET_DEFAULT_STD
from timm.data import Mixup
from timm.data import create_transform

import torch
import numpy as np
import torch.distributed as dist
import os


from typing import Iterable
from functools import partial
from torchvision import transforms
from utils.utils_distributed_sampler import DistributedSampler
from utils.utils_distributed_sampler import get_dist_info, worker_init_fn

from .datasets import AgeGenderDataset, CelebADataset, RAFDataset, FGnetDataset, ExpressionDataset, LAPDataset
from .samplers import SubsetRandomSampler
from torchvision.datasets import ImageFolder

# MXFaceDataset is only needed for MXNet RecordIO format
# Import it conditionally to avoid MXNet import errors when not using RecordIO
try:
    from dataset import MXFaceDataset
    MXFACEDATASET_AVAILABLE = True
except (ImportError, AttributeError):
    MXFACEDATASET_AVAILABLE = False
    MXFaceDataset = None

# Global cache for datasets to avoid reloading on each call
_dataset_cache = {}
_dataloader_cache = {}


def get_analysis_train_dataloader(data_choose, config, local_rank) -> Iterable:
    # Create cache key based on dataset type and key config parameters
    # This allows reusing datasets when config hasn't changed
    cache_key = f"{data_choose}_{local_rank}"
    
    # For recognition, include root_dir in cache key
    if data_choose == "recognition":
        cache_key = f"{data_choose}_{config.rec}_{local_rank}"
    elif data_choose == "age_gender":
        cache_key = f"{data_choose}_{config.age_gender_data_path}_{str(config.age_gender_data_list)}_{local_rank}"
    elif data_choose == "CelebA":
        cache_key = f"{data_choose}_{config.CelebA_train_data}_{local_rank}"
    elif data_choose == "expression":
        cache_key = f"{data_choose}_{config.AffectNet_data}_{local_rank}"
    
    # Check if dataloader is already cached
    if cache_key in _dataloader_cache:
        if dist.get_rank() == 0:
            print(f"Using cached dataloader for {data_choose} (rank {local_rank})", flush=True)
        # Update sampler epoch for new epoch
        cached_loader = _dataloader_cache[cache_key]
        if isinstance(cached_loader, torch.utils.data.DataLoader) and hasattr(cached_loader.sampler, 'set_epoch'):
            # Note: epoch will be set in train.py loop
            pass
        return cached_loader
    
    # Dataset not cached, create it
    if data_choose == "recognition":
        batch_size = config.recognition_bz
        root_dir = config.rec
        
        # Check if dataset is cached
        dataset_cache_key = f"dataset_{cache_key}"
        if dataset_cache_key in _dataset_cache:
            dataset_train = _dataset_cache[dataset_cache_key]
            if dist.get_rank() == 0:
                print(f"Using cached Recognition dataset: {len(dataset_train)} images, {len(dataset_train.classes)} classes", flush=True)
        else:
            # Check if MXNet RecordIO format exists
            rec_file = os.path.join(root_dir, 'train.rec')
            idx_file = os.path.join(root_dir, 'train.idx')
            
            if os.path.exists(rec_file) and os.path.exists(idx_file):
                # Use MXNet RecordIO format
                if not MXFACEDATASET_AVAILABLE:
                    raise ImportError("MXNet RecordIO format detected but MXNet is not available. "
                                    "Install MXNet with: pip install mxnet, or use ImageFolder format instead.")
                dataset_train = MXFaceDataset(root_dir=root_dir, local_rank=local_rank)
            else:
                # Use ImageFolder format (each folder is a person/class)
                if dist.get_rank() == 0:
                    print(f"Loading Recognition data from ImageFolder: {root_dir}", flush=True)
                transform = transforms.Compose([
                    transforms.Resize([config.img_size, config.img_size]),
                    transforms.RandomHorizontalFlip(),
                    transforms.ToTensor(),
                    transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
                ])
                dataset_train = ImageFolder(root_dir, transform=transform)
                if dist.get_rank() == 0:
                    print(f"Recognition dataset loaded: {len(dataset_train)} images, {len(dataset_train.classes)} classes", flush=True)
            # Cache the dataset
            _dataset_cache[dataset_cache_key] = dataset_train

    elif data_choose == "age_gender":
        # Check if dataset is cached
        dataset_cache_key = f"dataset_{cache_key}"
        if dataset_cache_key in _dataset_cache:
            dataset_train = _dataset_cache[dataset_cache_key]
            if dist.get_rank() == 0:
                print(f"Using cached Age/Gender dataset: {len(dataset_train)} samples", flush=True)
        else:
            if dist.get_rank() == 0:
                print(f"Loading Age/Gender dataset: {config.age_gender_data_list}", flush=True)
            batch_size = config.age_gender_bz
            transform = create_transform(
                input_size=config.img_size,
                scale=config.AUG_SCALE_SCALE if config.AUG_SCALE_SET else None,
                ratio=config.AUG_SCALE_RATIO if config.AUG_SCALE_SET else None,
                is_training=True,
                color_jitter=config.AUG_COLOR_JITTER if config.AUG_COLOR_JITTER > 0 else None,
                auto_augment=config.AUG_AUTO_AUGMENT if config.AUG_AUTO_AUGMENT != 'none' else None,
                re_prob=config.AUG_REPROB,
                re_mode=config.AUG_REMODE,
                re_count=config.AUG_RECOUNT,
                interpolation=config.INTERPOLATION,
                mean=[0.5, 0.5, 0.5],
                std=[0.5, 0.5, 0.5],
            )
            dataset_train = AgeGenderDataset(config=config, dataset=config.age_gender_data_list, transform=transform)
            if dist.get_rank() == 0:
                print(f"Age/Gender dataset loaded: {len(dataset_train)} samples", flush=True)
            # Cache the dataset
            _dataset_cache[dataset_cache_key] = dataset_train

    elif data_choose == "CelebA":
        # Check if dataset is cached
        dataset_cache_key = f"dataset_{cache_key}"
        if dataset_cache_key in _dataset_cache:
            dataset_train = _dataset_cache[dataset_cache_key]
            if dist.get_rank() == 0:
                print(f"Using cached CelebA dataset: {len(dataset_train)} samples", flush=True)
        else:
            if dist.get_rank() == 0:
                print("Loading CelebA dataset...", flush=True)
            batch_size = config.CelebA_bz
            transform = create_transform(
                input_size=config.img_size,
                scale=config.AUG_SCALE_SCALE if config.AUG_SCALE_SET else None,
                ratio=config.AUG_SCALE_RATIO if config.AUG_SCALE_SET else None,
                is_training=True,
                color_jitter=config.AUG_COLOR_JITTER if config.AUG_COLOR_JITTER > 0 else None,
                auto_augment=config.AUG_AUTO_AUGMENT if config.AUG_AUTO_AUGMENT != 'none' else None,
                re_prob=config.AUG_REPROB,
                re_mode=config.AUG_REMODE,
                re_count=config.AUG_RECOUNT,
                interpolation=config.INTERPOLATION,
                mean=[0.5, 0.5, 0.5],
                std=[0.5, 0.5, 0.5],
            )
            dataset_train = CelebADataset(config=config, choose="train", transform=transform)
            if dist.get_rank() == 0:
                print(f"CelebA dataset loaded: {len(dataset_train)} samples", flush=True)
            # Cache the dataset
            _dataset_cache[dataset_cache_key] = dataset_train

    elif data_choose == "expression":
        # Check if dataset is cached
        dataset_cache_key = f"dataset_{cache_key}"
        if dataset_cache_key in _dataset_cache:
            dataset_train = _dataset_cache[dataset_cache_key]
            if dist.get_rank() == 0:
                print(f"Using cached Expression dataset: {len(dataset_train)} samples", flush=True)
        else:
            if dist.get_rank() == 0:
                print("Loading Expression (AffectNet) dataset...", flush=True)
            batch_size = config.expression_bz
            transform = create_transform(
                input_size=config.img_size,
                scale=config.AUG_SCALE_SCALE if config.AUG_SCALE_SET else None,
                ratio=config.AUG_SCALE_RATIO if config.AUG_SCALE_SET else None,
                is_training=True,
                color_jitter=config.AUG_COLOR_JITTER if config.AUG_COLOR_JITTER > 0 else None,
                auto_augment=config.AUG_AUTO_AUGMENT if config.AUG_AUTO_AUGMENT != 'none' else None,
                re_prob=config.AUG_REPROB,
                re_mode=config.AUG_REMODE,
                re_count=config.AUG_RECOUNT,
                interpolation=config.INTERPOLATION,
                mean=[0.5, 0.5, 0.5],
                std=[0.5, 0.5, 0.5],
            )
            dataset_train = ExpressionDataset(config=config, transform=transform)
            if dist.get_rank() == 0:
                print(f"Expression dataset loaded: {len(dataset_train)} samples", flush=True)
            # Cache the dataset
            _dataset_cache[dataset_cache_key] = dataset_train

    rank, world_size = get_dist_info()
    sampler_train = DistributedSampler(
            dataset_train, num_replicas=world_size, rank=rank, shuffle=True, seed=config.seed)

    data_loader_train = torch.utils.data.DataLoader(
        dataset_train, sampler=sampler_train,
        batch_size=batch_size,
        num_workers=config.train_num_workers,
        pin_memory=config.train_pin_memory,
        drop_last=True,
    )
    
    # Cache the dataloader
    _dataloader_cache[cache_key] = data_loader_train
    
    return data_loader_train



def get_mixup_fn(config):

    mixup_fn = None
    mixup_active = config.AUG_MIXUP > 0 or config.AUG_CUTMIX > 0. or config.AUG_CUTMIX_MINMAX is not None
    if mixup_active:
        mixup_fn = Mixup(
            mixup_alpha=config.AUG_MIXUP, cutmix_alpha=config.AUG_CUTMIX, cutmix_minmax=config.AUG_CUTMIX_MINMAX,
            prob=config.AUG_MIXUP_PROB, switch_prob=config.AUG_MIXUP_SWITCH_PROB, mode=config.AUG_MIXUP_MODE,
            label_smoothing=config.RAF_LABEL_SMOOTHING, num_classes=config.RAF_NUM_CLASSES)
    return mixup_fn


def get_analysis_val_dataloader(data_choose, config):
    try:
        if data_choose == "CelebA":
            dataset_val = CelebADataset(config=config, choose="test")
        elif data_choose == "LAP":
            dataset_val = LAPDataset(config=config, choose="test")
        elif data_choose == "FGNet":
            dataset_val = FGnetDataset(config=config, choose="all")
        elif data_choose == "RAF":
            dataset_val = RAFDataset(config=config, choose="test")
        else:
            return None
    except (FileNotFoundError, OSError) as e:
        if dist.get_rank() == 0:
            print(f"⚠️  Warning: {data_choose} validation dataset not found, skipping: {e}")
        return None

    if len(dataset_val) == 0:
        if dist.get_rank() == 0:
            print(f"⚠️  Warning: {data_choose} validation dataset is empty, skipping")
        return None

    indices = np.arange(dist.get_rank(), len(dataset_val), dist.get_world_size())
    sampler_val = SubsetRandomSampler(indices)

    # Disable pin_memory for validation to avoid issues when default device is CUDA
    # Validation doesn't need the performance boost from pin_memory as much as training
    data_loader_val = torch.utils.data.DataLoader(
        dataset_val, sampler=sampler_val,
        batch_size=config.val_batch_size,
        shuffle=False,
        num_workers=config.val_num_workers,
        pin_memory=False,  # Disabled to avoid CUDA tensor pinning issues
        drop_last=False
    )

    return data_loader_val

