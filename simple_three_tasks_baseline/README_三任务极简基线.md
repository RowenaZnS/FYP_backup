## 三任务极简基线（Identity / Age / Mixture）

你老师的 idea 可以做成一个**很朴素、好解释、好复现**的 baseline：同一张脸图像通过一个 backbone（ResNet / EfficientNet / ViT 任一），得到共享特征后分出三条任务分支：

- **Face identity**：做人脸识别（分类/度量学习）
- **Age information**：做年龄估计（回归或分桶分类）
- **Mixture**：把 identity 与 age 两种信息融合后再做任务（可以是“同时做识别+年龄”，或只做一个“混合 embedding 的识别”）

这类 baseline 的价值是：不用复杂 attention，也能清晰验证“分离/融合”到底有没有用。

---

## 目标与设定（对应老师给的 3 types）

我们把老师的 3 types 对齐到可训练目标：

- **Type-A: face identity（跨年龄识别）**
  - 输出：identity embedding（用于识别）
  - loss：ArcFace/AM-Softmax（或普通 CE）
  - 解释：只要 backbone 学得好，embedding 对年龄变化会有一定鲁棒性，这就是“cross-age recognition”的最简单实现。

- **Type-B: age information（年龄估计）**
  - 输出：age head（回归或分类）
  - loss：L1/MSE（回归）或 CE（分桶）
  - 解释：用同一 backbone 或共享 backbone + 专门 age head。

- **Type-C: mixture（identity+age 融合）**
  - 输入：identity feature 与 age feature
  - 融合：concat + MLP（最简单可靠）
  - loss：至少一个（推荐同时做识别+年龄，形成“多任务”）

---

## 模型结构（极简但完整）

### 共享主干（Backbone）

任选其一（建议直接用 `timm`）：

- ResNet（如 `resnet50`）
- EfficientNet（如 `efficientnet_b0`）
- ViT（如 `vit_base_patch16_224`）

共享 backbone 输出一个特征向量 \(f\)。

### 三个分支（3 heads）

从共享特征 \(f\) 走三条路：

- **Identity head**：\(z_{id} = \mathrm{Proj}_{id}(f)\)，归一化后用于 ArcFace（或 CE）
- **Age head**：\(z_{age} = \mathrm{Proj}_{age}(f)\)，接 age predictor 输出年龄
- **Mixture head**：
  - **融合 1（你老师说的 “three feature -> fusion”）**：可以扩展成从 backbone 的 3 个 stage 抽 feature 再 fusion（但 baseline 先不做，先用 1 个共享 feature）
  - **融合 2（你老师说的 “concat and know the final feature”）**：
    \[
    z_{mix} = \mathrm{Proj}_{mix}([\;z_{id}, z_{age}\;])
    \]
    然后对 \(z_{mix}\) 做识别（ArcFace）或年龄估计，或者两者都做

### 可选：分离约束（强烈建议加一条，简单又好讲）

为了让 identity 与 age 更“解耦”，加一个非常朴素的约束：

- **Orthogonality / decorrelation loss**
  - 让 \(z_{id}\) 与 \(z_{age}\) 的 cosine 相似度接近 0（避免信息塌到同一个子空间）

---

## 三个 loss（至少 3 个）

对应你老师要求的 “face: three loss function at least”：

\[
\mathcal{L}=\lambda_{id}\mathcal{L}_{id}+\lambda_{age}\mathcal{L}_{age}+\lambda_{mix}\mathcal{L}_{mix}+\lambda_{sep}\mathcal{L}_{sep}
\]

- **\(\mathcal{L}_{id}\)**：ArcFace/AM-Softmax（或 CE）做 identity
- **\(\mathcal{L}_{age}\)**：年龄回归（L1/MSE）或年龄分桶分类（CE）
- **\(\mathcal{L}_{mix}\)**：对融合后的 \(z_{mix}\) 再做一个任务（推荐：再做一次 identity ArcFace，证明“融合信息”对跨年龄识别是否有帮助）
- **\(\mathcal{L}_{sep}\)**（可选但推荐）：让 \(z_{id}\) 与 \(z_{age}\) 去相关/正交

---

## 训练 flow（3 种实验设置）

你可以做 3 个最小实验（完全贴合老师的 1/2/3）：

- **实验 1：Cross-age recognition（只训 identity）**
  - loss：\(\mathcal{L}_{id}\)

- **实验 2：Age estimation（只训 age）**
  - loss：\(\mathcal{L}_{age}\)

- **实验 3：Both tasks（多任务 + mixture）**
  - loss：\(\mathcal{L}_{id} + \mathcal{L}_{age} + \mathcal{L}_{mix} (+ \mathcal{L}_{sep})\)

---

## 流程图（Mermaid）

```mermaid
flowchart TB
  A[Input Face Image] --> B[Backbone: ResNet / EfficientNet / ViT]
  B --> F[Shared feature f]

  F --> IDP[Proj_id]
  F --> AGP[Proj_age]

  IDP --> ZID[z_id (norm)]
  AGP --> ZAGE[z_age]

  ZID --> HID[Identity Head / ArcFace]
  ZAGE --> HAGE[Age Head]

  ZID --> CAT[Concat]
  ZAGE --> CAT
  CAT --> MIXP[Proj_mix]
  MIXP --> HMIX[Mixture Head]

  HID --> LID[L_id]
  HAGE --> LAGE[L_age]
  HMIX --> LMIX[L_mix]
  ZID --> LSEP[L_sep: decorrelate(z_id, z_age)]
  ZAGE --> LSEP

  LID --> LT[Total Loss]
  LAGE --> LT
  LMIX --> LT
  LSEP --> LT
  LT --> UPD[Update params]
```

---

## 代码落地（我会在新文件夹里提供）

目录：`FYP_backup/simple_three_tasks_baseline/`

- `src/models.py`：三分支模型（可选 backbone）
- `src/losses.py`：ArcFace + age loss + mixture loss + 分离约束
- `src/datasets.py`：最简单的 dataset（支持 ImageFolder + CSV 标注）
- `train_identity.py` / `train_age.py` / `train_multitask.py`：三个实验入口

> 重点是“可跑 + 可讲 + 可改”：你后面想把“3 个 stage feature -> fusion”加进去，也能在这个 baseline 上自然扩展。

---

## 运行方式（直接复用 `swinface_project` 的 dataset / config）

如果你的数据已经按 `swinface_project` 的格式放好了（例如 `configs/config_train.py` 里配置的 `rec` 与 `age_gender_data_path`），可以直接用下面这三个入口脚本，它们内部会调用：

- `swinface_project/analysis/get_analysis_train_dataloader("recognition", cfg, local_rank)`
- `swinface_project/analysis/get_analysis_train_dataloader("age_gender", cfg, local_rank)`

示例 config：

- `/workspace/FYP_backup/swinface_project/configs/config_train.py`

### 实验 1：只训 identity（cross-age recognition baseline）

```bash
python3 /workspace/FYP_backup/simple_three_tasks_baseline/train_identity_swinface_project.py \
  /workspace/FYP_backup/swinface_project/configs/config_train.py \
  --backbone resnet50
```

### 实验 2：只训 age（age estimation baseline）

```bash
python3 /workspace/FYP_backup/simple_three_tasks_baseline/train_age_swinface_project.py \
  /workspace/FYP_backup/swinface_project/configs/config_train.py \
  --backbone resnet50
```

### 实验 3：identity + age + mixture（both task + fusion）

```bash
python3 /workspace/FYP_backup/simple_three_tasks_baseline/train_multitask_swinface_project.py \
  /workspace/FYP_backup/swinface_project/configs/config_train.py \
  --backbone resnet50 \
  --lambda_sep 0.1
```

### 注意：为什么脚本里要做 “单进程分布式初始化”

`swinface_project` 的 `get_analysis_train_dataloader()` 内部会调用 `torch.distributed.get_rank()` / `get_world_size()`（用于 `DistributedSampler` 和缓存逻辑），因此即使你只用单卡/单进程，也需要初始化一个 `world_size=1` 的进程组，否则会报错。

这三个 `*_swinface_project.py` 脚本已经内置了这一步，所以你不用改原工程。

