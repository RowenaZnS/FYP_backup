from easydict import EasyDict as edict

config = edict()

# 与 analysis 数据集采样逻辑兼容（仅用 age_gender 训练时仍可能被读取）
config.num_image = 5822653
config.recognition_bz = 32

# -----------------------------------------------------------------------------
# Data — 指向原 swinface_project 下的 dataset，不复制数据
# -----------------------------------------------------------------------------
config.age_gender_data_path = "/workspace/FYP_backup/swinface_project/dataset/imdb-wiki"
config.age_gender_data_list = ["IMDB"]

config.img_size = 112
config.age_gender_bz = 16
config.train_num_workers = 2
config.train_pin_memory = True

# Augment (与主项目一致)
config.INTERPOLATION = "bicubic"
config.AUG_COLOR_JITTER = 0.4
config.AUG_AUTO_AUGMENT = "rand-m9-mstd0.5-inc1"
config.AUG_REPROB = 0.25
config.AUG_REMODE = "pixel"
config.AUG_RECOUNT = 1
config.AUG_MIXUP = 0.0
config.AUG_CUTMIX = 0.0
config.AUG_CUTMIX_MINMAX = None
config.AUG_MIXUP_PROB = 1.0
config.AUG_MIXUP_SWITCH_PROB = 0.5
config.AUG_MIXUP_MODE = "batch"
config.AUG_SCALE_SET = True
config.AUG_SCALE_SCALE = (1.0, 1.0)
config.AUG_SCALE_RATIO = (1.0, 1.0)

# -----------------------------------------------------------------------------
# Model (与 SwinFace + FAM 一致)
# -----------------------------------------------------------------------------
config.network = "swin_t"
config.fam_kernel_size = 3
config.fam_in_chans = 2112
config.fam_conv_shared = False
config.fam_conv_mode = "split"
config.fam_channel_attention = "CBAM"
config.fam_spatial_attention = None
config.fam_pooling = "max"
config.fam_la_num_list = [2 for j in range(11)]
config.fam_feature = "all"
config.embedding_size = 512

# Age generation 专用
config.age_extractor_hidden_dim = 256
config.target_age_hidden_dim = 256
config.age_code_dim = 128
config.gen_latent_hw = 7
config.fusion_num_heads = 8
config.age_norm_max = 100.0
config.use_vgg_perceptual = True

# -----------------------------------------------------------------------------
# Loss weights
# -----------------------------------------------------------------------------
config.lambda_id = 1.0
config.lambda_age = 5.0
config.lambda_adv = 0.1
config.lambda_rec = 2.0
config.lambda_vgg = 0.5
config.lambda_mi = 0.05

# True：生成器分两步更新（先非 age，再 age），避免 age 数值大时主导同一次 backward
config.separate_age_g_step = True

# 以一定概率将目标年龄设为「真实年龄」做自重建约束
config.recon_prob = 0.35

# 目标年龄采样：在 [age_min, age_max] 均匀（非重建分支）
config.target_age_min = 5.0
config.target_age_max = 90.0

# -----------------------------------------------------------------------------
# Training
# -----------------------------------------------------------------------------
config.seed = 2048
config.fp16 = True
config.optimizer = "adamw"
config.lr = 2e-4
config.weight_decay = 0.05
config.lr_name = "cosine"
config.warmup_step = 2000
config.warmup_lr = 1e-7
config.min_lr = 1e-6
config.total_step = 100000

config.lr_d = 2e-4

config.freeze_backbone_steps = 2000

config.init = False
config.init_model = "/workspace/FYP_backup/swinface_project/output/init/"
config.resume = False
config.resume_step = 0

config.output = "/workspace/FYP_backup/swinface_age_gen/output_age_gen"
config.verbose = 500
config.save_verbose = 2000
config.frequent = 50
config.save_all_states = True

config.mi_temperature = 0.1
