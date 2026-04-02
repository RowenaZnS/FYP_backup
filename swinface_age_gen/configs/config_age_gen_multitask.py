from easydict import EasyDict as edict

config = edict()

config.num_image = 5822653
config.recognition_bz = 32

# -----------------------------------------------------------------------------
# Data（与 train_rest_mtmim 相同三路：age_gender + CelebA + expression）
# 数据路径仍指向 swinface_project/dataset
# -----------------------------------------------------------------------------
config.age_gender_data_path = "/workspace/FYP_backup/swinface_project/dataset/imdb-wiki"
config.age_gender_data_list = ["IMDB"]

config.CelebA_train_data = "/workspace/FYP_backup/swinface_project/dataset/celeba/img_align_celeba/img_align_celeba"
config.CelebA_train_label = "/workspace/FYP_backup/swinface_project/dataset/celeba/list_attr_celeba.csv"
config.CelebA_val_data = "/workspace/FYP_backup/swinface_project/dataset/celeba/img_align_celeba/img_align_celeba"
config.CelebA_val_label = "/workspace/FYP_backup/swinface_project/dataset/celeba/list_attr_celeba.csv"

config.AffectNet_data = "/workspace/FYP_backup/swinface_project/dataset/affectnet/archive (3)/Train"
config.AffectNet_label = "/workspace/FYP_backup/swinface_project/dataset/affectnet/archive (3)/labels.csv"

config.img_size = 112
config.age_gender_bz = 8
config.CelebA_bz = 8
config.expression_bz = 8
config.train_num_workers = 2
config.train_pin_memory = True

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

config.RAF_NUM_CLASSES = 7

# -----------------------------------------------------------------------------
# Model
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

config.age_extractor_hidden_dim = 256
config.target_age_hidden_dim = 256
config.age_code_dim = 128
config.gen_latent_hw = 7
config.fusion_num_heads = 8
config.age_norm_max = 100.0
config.use_vgg_perceptual = True

# 打开 SwinFace OutputModule：Gender / CelebA / Expression（无年龄任务 loss）
config.use_analysis_heads = True

# -----------------------------------------------------------------------------
# Loss：无 lambda_age / 无年龄监督；多任务权重与 train_rest 类似
# -----------------------------------------------------------------------------
config.lambda_id = 1.0
config.lambda_adv = 0.1
config.lambda_rec = 2.0
config.lambda_vgg = 0.5
# 与 train_rest_mtmim 一致：MI 用单独权重（不用 lambda_age）
config.mi_loss_weight = 0.1

config.analysis_loss_weights = [1.0 for _ in range(42)]

# TensorBoard：为 CelebA 40 个属性各写一条曲线（加权后的标量）
config.log_celeba_each_attr = True


config.recon_prob = 0.35
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

config.output = "/workspace/FYP_backup/swinface_age_gen/output_age_gen_multitask"
config.verbose = 500
config.save_verbose = 2000
config.frequent = 50
config.save_all_states = True

config.mi_temperature = 0.1
