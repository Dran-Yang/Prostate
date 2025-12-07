# 前列腺 MRI 自监督预训练工程

<div align="center">

**基于 DINOv2 的多模态前列腺 MRI 自监督学习框架**

[![Python 3.10+](https://img.shields.io/badge/python-3.10+-blue.svg)](https://www.python.org/downloads/)
[![PyTorch 2.0+](https://img.shields.io/badge/PyTorch-2.0+-ee4c2c.svg)](https://pytorch.org/)
[![CUDA 12.x](https://img.shields.io/badge/CUDA-12.x-76B900.svg)](https://developer.nvidia.com/cuda-toolkit)

</div>

---

## 📋 目录

1. [项目简介](#项目简介)
2. [主要特性](#主要特性)
3. [环境需求与安装](#环境需求与安装)
4. [数据准备](#数据准备)
5. [快速开始](#快速开始)
6. [配置说明](#配置说明)
7. [训练监控](#训练监控)
8. [Resume 训练](#resume-训练)
9. [下游任务](#下游任务)
10. [常见问题 FAQ](#常见问题-faq)
11. [项目结构](#项目结构)
12. [引用](#引用)

---

## 项目简介

本项目是一个面向**前列腺多模态 MRI**的自监督预训练框架，基于 [DINOv2](https://github.com/facebookresearch/dinov2) 和 [mm-dinov2](https://github.com/mahmoodlab/mmdino) 思想开发。通过在大规模未标注的前列腺 MRI 数据上进行自监督学习，可以为下游的**分割、分级、分类**等任务提供强大的预训练 backbone。

**适用场景：**
- 前列腺癌检测与分级
- 前列腺分割
- 多参数 MRI（mpMRI）特征提取
- 少样本医学影像学习

**核心技术：**
- **多模态输入**：同时处理 T2WI、ADC、DWI 等多个 MRI 序列
- **自监督学习**：DINO + iBOT 联合训练，无需标注数据
- **前列腺特定优化**：支持基于 ROI 的裁剪，提高病灶区域学习效率
- **稳定训练**：fp32 策略 + xFormers 加速，适合医学影像的数值稳定性要求

---

## 主要特性

### ✨ 核心功能

- 🎯 **多模态 MRI 支持**：灵活配置 T2WI、ADC、DWI 等序列组合
- 🏥 **医学影像优化**：针对前列腺 MRI 的数据增强策略
- 🚀 **高效训练**：xFormers 加速 + 可选的 FSDP 多卡支持
- 📊 **ROI 引导**：可选的基于前列腺分割 mask 的前景裁剪
- 💾 **灵活的检查点**：支持单卡和多卡的检查点保存与恢复
- 🔧 **可扩展架构**：易于添加新的 MRI 序列或下游任务

### 🛠️ 技术亮点

- **fp32 训练策略**：确保医学影像数值稳定性，避免精度损失
- **智能 DWI 选择**：自动选择最高 b 值的 DWI 序列
- **鲁棒的数据加载**：处理缺失模态、不同切片数等边界情况
- **随机轴选择**：训练时随机选择轴位/冠状位/矢状位，增强模型泛化性
- **百分比标注**：支持半监督学习，可配置使用部分标注数据

---

## 环境需求与安装

### 1. 系统要求

- **操作系统**：Linux（推荐 Ubuntu 20.04/22.04）或 WSL2
- **GPU**：NVIDIA GPU with CUDA 12.x（推荐 RTX 3090/4090 或更高）
- **显存**：至少 16GB（推荐 24GB+）
- **内存**：32GB+
- **磁盘**：SSD（NIfTI 文件 I/O 密集）

### 2. Python 环境

推荐使用 **Python 3.10** 或更高版本。

#### 使用 Conda 创建环境

```bash
# 创建新环境
conda create -n prostate-ssl python=3.10 -y
conda activate prostate-ssl

# 安装 PyTorch（CUDA 12.1 版本）
conda install pytorch torchvision pytorch-cuda=12.1 -c pytorch -c nvidia

# 验证 PyTorch 安装
python -c "import torch; print(f'PyTorch: {torch.__version__}'); print(f'CUDA available: {torch.cuda.is_available()}')"
```

#### 或使用 virtualenv

```bash
# 创建虚拟环境
python3.10 -m venv prostate-ssl-env
source prostate-ssl-env/bin/activate

# 安装 PyTorch
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu121
```

### 3. 安装依赖包

创建 `requirements.txt` 文件（如果项目中还没有）：

```bash
cat > requirements.txt << 'EOF'
# 核心依赖
torch>=2.0.0
torchvision>=0.15.0
omegaconf>=2.3.0
timm>=0.9.0

# 医学影像处理
monai>=1.2.0
nibabel>=5.0.0
SimpleITK>=2.2.0

# 分布式训练
fvcore>=0.1.5

# xFormers（加速训练）
xformers>=0.0.20

# 数据处理
pandas>=2.0.0
numpy>=1.24.0
scipy>=1.10.0

# 可视化（可选）
tensorboard>=2.13.0
matplotlib>=3.7.0

# 工具
tqdm>=4.65.0
pyyaml>=6.0
EOF

# 安装依赖
pip install -r requirements.txt
```

### 4. 安装 xFormers（重要）

xFormers 提供了高效的注意力机制实现，能显著加速训练。

```bash
# 方法 1：使用 pip（推荐）
pip install xformers

# 方法 2：从源码编译（如果上述方法失败）
pip install -v -U git+https://github.com/facebookresearch/xformers.git@main#egg=xformers

# 验证安装
python -c "import xformers; print(f'xFormers version: {xformers.__version__}')"
```

**如果无法安装 xFormers：**
- 可以设置环境变量 `XFORMERS_DISABLED=1` 继续运行（性能会下降）
- 或者使用纯 PyTorch 实现（需要修改代码，见 FAQ）

### 5. 克隆项目

```bash
git clone https://github.com/Dran-Yang/Prostate.git
cd Prostate
```

### 6. 验证安装

运行以下命令验证环境是否正确：

```bash
python -c "
import torch
import monai
import nibabel as nib
print('✓ PyTorch:', torch.__version__)
print('✓ CUDA available:', torch.cuda.is_available())
print('✓ MONAI:', monai.__version__)
print('✓ nibabel:', nib.__version__)
try:
    import xformers
    print('✓ xFormers:', xformers.__version__)
except ImportError:
    print('✗ xFormers not installed')
"
```

预期输出：
```
✓ PyTorch: 2.x.x
✓ CUDA available: True
✓ MONAI: 1.x.x
✓ nibabel: 5.x.x
✓ xFormers: 0.x.x
```

---

## 数据准备

### 1. 数据组织结构

项目期望每个病例存放在独立的文件夹中，每个文件夹包含多个 MRI 序列和可选的分割 mask。

#### 标准目录结构

```
/path/to/prostate_dataset/
├── patient_001/
│   ├── ax_t2wi.nii.gz         # T2 加权像（轴位）
│   ├── ax_adc.nii.gz          # 表观扩散系数（ADC）
│   ├── ax_dwi_b1000.nii.gz    # 扩散加权像（DWI，b=1000）
│   ├── ax_dwi_b2000.nii.gz    # DWI（b=2000，可选）
│   └── roi_Prostate.nii.gz    # 前列腺分割 mask（可选）
├── patient_002/
│   ├── ax_t2wi.nii
│   ├── ax_adc.nii
│   ├── ax_dwi_1000.nii
│   └── roi_Prostate.nii
└── patient_003/
    └── ...
```

#### 命名规范

**必需的 MRI 序列**（默认配置）：
- `ax_t2wi.nii` 或 `ax_t2wi.nii.gz`：T2 加权像
- `ax_adc.nii` 或 `ax_adc.nii.gz`：表观扩散系数
- `ax_dwi_*.nii` 或 `ax_dwi_*.nii.gz`：扩散加权像（自动选择最高 b 值）

**可选文件**：
- `roi_Prostate.nii` 或 `roi_Prostate.nii.gz`：前列腺分割 mask
  - 如果设置 `crop_from_tumor_foreground: True`，会基于此 mask 进行前景裁剪
  - 如果文件不存在，会使用整个图像的中心区域

**DWI 命名灵活性**：
代码会自动搜索 `ax_dwi*.nii*` 文件并选择 b 值最高的。支持以下命名格式：
- `ax_dwi_b1000.nii.gz` ✓
- `ax_dwi_1000.nii` ✓
- `dwi_b2000.nii.gz` ✓
- `dwi_2000.nii` ✓

### 2. DICOM 转 NIfTI

如果你的原始数据是 DICOM 格式，需要先转换为 NIfTI。

#### 方法 1：使用 dcm2niix（推荐）

```bash
# 安装 dcm2niix
sudo apt-get install dcm2niix  # Ubuntu
# 或
conda install -c conda-forge dcm2niix

# 转换单个病例
dcm2niix -o /output/patient_001 -f ax_t2wi /input/patient_001/T2_DICOM_folder

# 批量转换脚本示例
for patient_dir in /input/*/; do
    patient_id=$(basename "$patient_dir")
    dcm2niix -o "/output/$patient_id" -f ax_t2wi "$patient_dir/T2_DICOM/"
    dcm2niix -o "/output/$patient_id" -f ax_adc "$patient_dir/ADC_DICOM/"
    dcm2niix -o "/output/$patient_id" -f ax_dwi_b1000 "$patient_dir/DWI_b1000_DICOM/"
done
```

#### 方法 2：使用 SimpleITK

```python
import SimpleITK as sitk
import os

def convert_dicom_series_to_nifti(dicom_dir, output_path):
    """读取 DICOM 序列并保存为 NIfTI"""
    reader = sitk.ImageSeriesReader()
    dicom_names = reader.GetGDCMSeriesFileNames(dicom_dir)
    reader.SetFileNames(dicom_names)
    image = reader.Execute()
    sitk.WriteImage(image, output_path)

# 使用示例
convert_dicom_series_to_nifti(
    "/input/patient_001/T2_DICOM/",
    "/output/patient_001/ax_t2wi.nii.gz"
)
```

#### 方法 3：使用 MONAI

```python
from monai.transforms import LoadImage
from monai.data import write_nifti

loader = LoadImage(image_only=False)
image, meta = loader("/input/patient_001/T2_DICOM/")
write_nifti(
    image,
    "/output/patient_001/ax_t2wi.nii.gz",
    affine=meta["affine"],
)
```

### 3. 数据集分割

创建训练集/验证集/测试集的 CSV 文件。

#### 创建 split CSV

```bash
# 创建 split 目录
mkdir -p split

# 生成训练集 CSV（示例）
cat > split/train.csv << 'EOF'
patient_id
patient_001
patient_002
patient_003
patient_004
patient_005
EOF

# 生成验证集 CSV
cat > split/val.csv << 'EOF'
patient_id
patient_006
patient_007
EOF

# 生成测试集 CSV
cat > split/test.csv << 'EOF'
patient_id
patient_008
patient_009
EOF
```

**CSV 格式说明：**
- 第一行是列名，支持多种命名：`patient_id`, `case_id`, `id`, `ID`, `subject`, `name`
- 每行一个病例 ID，需要与数据目录中的文件夹名完全匹配
- 文件必须是 UTF-8 编码

#### 自动生成 split 脚本

```python
import os
import pandas as pd
from sklearn.model_selection import train_test_split

# 扫描数据目录
data_root = "/path/to/prostate_dataset"
patient_ids = sorted([d for d in os.listdir(data_root) 
                      if os.path.isdir(os.path.join(data_root, d))])

# 分割数据集（70% 训练，15% 验证，15% 测试）
train_ids, temp_ids = train_test_split(patient_ids, test_size=0.3, random_state=42)
val_ids, test_ids = train_test_split(temp_ids, test_size=0.5, random_state=42)

# 保存 CSV
os.makedirs("split", exist_ok=True)
pd.DataFrame({"patient_id": train_ids}).to_csv("split/train.csv", index=False)
pd.DataFrame({"patient_id": val_ids}).to_csv("split/val.csv", index=False)
pd.DataFrame({"patient_id": test_ids}).to_csv("split/test.csv", index=False)

print(f"✓ 训练集: {len(train_ids)} 病例")
print(f"✓ 验证集: {len(val_ids)} 病例")
print(f"✓ 测试集: {len(test_ids)} 病例")
```

### 4. 数据质量检查

在开始训练前，建议运行以下脚本检查数据完整性：

```python
import os
import nibabel as nib
from pathlib import Path

def check_dataset(data_root, split_csv):
    """检查数据集完整性"""
    import pandas as pd
    
    df = pd.read_csv(split_csv)
    patient_ids = df.iloc[:, 0].tolist()
    
    issues = []
    required_files = ["ax_t2wi", "ax_adc"]
    
    for pid in patient_ids:
        patient_dir = Path(data_root) / pid
        
        if not patient_dir.exists():
            issues.append(f"❌ {pid}: 目录不存在")
            continue
        
        # 检查必需文件
        for seq in required_files:
            found = list(patient_dir.glob(f"{seq}.nii*"))
            if not found:
                issues.append(f"⚠️ {pid}: 缺少 {seq}")
        
        # 检查 DWI
        dwi_files = list(patient_dir.glob("ax_dwi*.nii*")) + list(patient_dir.glob("dwi*.nii*"))
        if not dwi_files:
            issues.append(f"⚠️ {pid}: 缺少 DWI 序列")
        
        # 检查 ROI（可选）
        roi_files = list(patient_dir.glob("roi_Prostate.nii*"))
        if not roi_files:
            issues.append(f"ℹ️ {pid}: 没有 ROI mask（可选）")
    
    if issues:
        print("\n".join(issues))
    else:
        print(f"✓ 所有 {len(patient_ids)} 个病例检查通过！")
    
    return len(issues) == 0

# 使用示例
check_dataset("/path/to/prostate_dataset", "split/train.csv")
```

---

## 快速开始

### 1. 修改配置文件

编辑 `dinov2/configs/train/prostate_vitb14_mm-dino.yaml`：

```yaml
train:
  # 修改这里：指向你的数据根目录
  dataset_path: ProstateSSL:split=TRAIN:root=/path/to/prostate_dataset:split_csv=split/train.csv:mri_sequences=ax_t2wi,ax_adc,ax_dwi:random_axes=True:random_slices=True
  
  batch_size_per_gpu: 8  # 根据显存调整（4-12）
  num_workers: 4
  OFFICIAL_EPOCH_LENGTH: 50  # ceil(训练集病例数 / batch_size_per_gpu)
  
  # 修改这里：输出目录
  output_dir: ./output/prostate_ssl_run1

optim:
  base_lr: 3.5e-4  # 适合 batch_size=8 的学习率
  epochs: 300
  warmup_epochs: 30
```

**参数说明：**
- `root`：数据根目录的绝对路径
- `split_csv`：训练集 CSV 文件的相对路径（相对于项目根目录）
- `mri_sequences`：使用的 MRI 序列，逗号分隔
- `random_axes`：训练时随机选择轴位/冠状位/矢状位（建议 `True`）
- `random_slices`：随机选择切片（建议 `True`）
- `OFFICIAL_EPOCH_LENGTH`：每个 epoch 的迭代次数，建议设为 `ceil(病例数 / batch_size)`

### 2. 单卡训练（推荐新手）

```bash
cd Prostate/dinov2

# 方法 1：使用配置文件
python -m train.train \
  --config-file configs/train/prostate_vitb14_mm-dino.yaml \
  --output-dir /path/to/output

# 方法 2：命令行覆盖配置
python -m train.train \
  --config-file configs/train/prostate_vitb14_mm-dino.yaml \
  --output-dir /path/to/output \
  train.batch_size_per_gpu=8 \
  optim.base_lr=3.5e-4 \
  optim.epochs=300
```

### 3. 多卡训练

```bash
cd Prostate/dinov2

# 使用 torchrun（推荐）
torchrun --nproc_per_node=4 -m train.train \
  --config-file configs/train/prostate_vitb14_mm-dino.yaml \
  --output-dir /path/to/output

# 或使用 python -m torch.distributed.launch（旧版）
python -m torch.distributed.launch \
  --nproc_per_node=4 \
  --use_env \
  -m train.train \
  --config-file configs/train/prostate_vitb14_mm-dino.yaml \
  --output-dir /path/to/output
```

**注意：**
- `--nproc_per_node`：GPU 数量
- 多卡训练会自动使用 FSDP（Fully Sharded Data Parallel）
- 学习率会根据总 batch size 自动调整（使用 sqrt scaling）

### 4. 运行 Smoke Test（推荐首次运行）

在正式训练前，建议先用少量数据验证流程：

```bash
cd Prostate/dinov2

# 设置测试数据路径（2-4 个病例即可）
export PROSTATE_DATA_ROOT=/path/to/small_subset

# 运行 smoke test（5 个迭代）
python tests/test_prostate_ssl_training.py
```

预期输出：
```
[step 0] loss=X.XXXX
[step 1] loss=X.XXXX
[step 2] loss=X.XXXX
[step 3] loss=X.XXXX
[step 4] loss=X.XXXX
Smoke test completed without runtime errors.
```

如果出现错误，请检查：
- 数据路径是否正确
- 数据格式是否符合要求
- CUDA 是否可用

---

## 配置说明

### 关键配置参数详解

#### 1. 数据相关 (`train` 部分)

```yaml
train:
  dataset_path: "ProstateSSL:split=TRAIN:root=/data:split_csv=split/train.csv:mri_sequences=ax_t2wi,ax_adc,ax_dwi:random_axes=True:random_slices=True"
  batch_size_per_gpu: 8
  num_workers: 4
  OFFICIAL_EPOCH_LENGTH: 50
  img_size: 224
  percentage_labels: 1.0  # 使用多少比例的标注数据（0-1）
```

**`dataset_path` 格式：**
- 格式：`DatasetName:key1=value1:key2=value2:...`
- 必需字段：
  - `split`：TRAIN / VAL / TEST
  - `root`：数据根目录
- 可选字段：
  - `split_csv`：CSV 文件路径
  - `mri_sequences`：使用的序列（默认：`ax_t2wi,ax_adc,ax_dwi`）
  - `random_axes`：随机选择切片轴（默认：False）
  - `random_slices`：随机选择切片（默认：False）

**`percentage_labels` 说明：**
- 控制使用多少比例的分割 mask
- `1.0`：使用所有可用的 mask（全监督）
- `0.5`：随机选择 50% 的病例使用 mask
- `0.0`：完全不使用 mask（纯自监督）

#### 2. 模型相关 (`student` 部分)

```yaml
student:
  arch: glioma_vit_base  # 模型架构
  patch_size: 14         # patch 大小
  drop_path_rate: 0.1    # DropPath 比例
  use_mri_seq_embed: True      # 使用序列嵌入
  img_wise_pos_embed: True     # 使用图像级位置编码
  pretrained_weights: ""       # 预训练权重路径（可选）
```

**模型架构选项：**
- `glioma_vit_small`：小模型，~22M 参数
- `glioma_vit_base`：基础模型，~86M 参数（推荐）
- `glioma_vit_large`：大模型，~304M 参数
- `glioma_vit_giant2`：超大模型，~1.1B 参数

**MRI 特定参数：**
- `use_mri_seq_embed=True`：为每个 MRI 序列学习独立的嵌入（推荐开启）
- `img_wise_pos_embed=True`：每个序列独立的位置编码（推荐开启）

#### 3. 优化器相关 (`optim` 部分)

```yaml
optim:
  base_lr: 3.5e-4              # 基础学习率
  epochs: 300                  # 总训练轮数
  warmup_epochs: 30            # warmup 轮数
  weight_decay: 0.04           # 权重衰减
  weight_decay_end: 0.4        # 最终权重衰减
  clip_grad: 3.0               # 梯度裁剪
  freeze_backbone_epochs: 0    # 冻结 backbone 的轮数
  
  # 高级参数
  scaling_rule: sqrt_wrt_1024  # 学习率缩放规则
  patch_embed_lr_mult: 0.2     # patch embedding 学习率倍数
  layerwise_decay: 0.9         # 层级学习率衰减
```

**学习率设置建议：**
- 单卡，batch_size=4: `base_lr: 2.5e-4`
- 单卡，batch_size=8: `base_lr: 3.5e-4`
- 双卡，batch_size=8: `base_lr: 5e-4`
- 四卡，batch_size=8: `base_lr: 7e-4`

#### 4. 数据增强 (`crops` 部分)

```yaml
crops:
  global_crops_size: 224        # 全局裁剪尺寸
  local_crops_size: 112         # 局部裁剪尺寸
  global_crops_scale: [0.5, 1.0]    # 全局裁剪缩放范围
  local_crops_scale: [0.2, 0.5]     # 局部裁剪缩放范围
  crop_from_tumor_foreground: True  # 基于前列腺 ROI 裁剪
  intensity_aug: rc                 # 强度增强类型
  max_blur_radius: 1                # 最大模糊半径
  gamma_range: [0.75, 1.5]          # Gamma 变换范围
```

**`intensity_aug` 选项：**
- `rc`：RandConv（随机卷积，适合医学影像）
- `color_jittering`：颜色抖动（不推荐用于灰度医学影像）
- `none`：不使用强度增强

**`crop_from_tumor_foreground` 说明：**
- `True`：裁剪时优先包含前列腺区域（需要 ROI mask）
- `False`：随机裁剪整个图像

#### 5. 损失函数 (`dino`, `ibot` 部分)

```yaml
dino:
  head_n_prototypes: 4096      # DINO 原型数量
  head_bottleneck_dim: 256     # 瓶颈层维度
  koleo_loss_weight: 0.1       # KoLeo 正则化权重

ibot:
  head_n_prototypes: 4096      # iBOT 原型数量
  mask_per_channel: True       # 每个通道独立 mask（重要）
  mask_ratio_min_max: [0.1, 0.5]  # mask 比例范围
```

**`mask_per_channel` 说明：**
- `True`：每个 MRI 序列独立生成 mask（推荐）
- `False`：所有序列共享同一个 mask

#### 6. 评估相关 (`evaluation` 部分)

```yaml
evaluation:
  eval_period_iterations: 1000  # 每隔多少次迭代评估一次（0=禁用）
  train_dataset_path: ""        # 训练集路径（用于评估）
  val_dataset_path: ""          # 验证集路径
  metric_types: ["mcc"]         # 评估指标
```

**注意：**
- 如果设置 `eval_period_iterations > 0`，需要提供标注的验证集
- 自监督预训练通常设置为 `0`（禁用评估）

---

## 训练监控

### 1. 日志输出

训练过程中会在终端输出日志：

```
[2024-12-07 10:00:00] Training  [    0/15000]  eta: 2:30:00  lr: 0.000350  wd: 0.040  ...
[2024-12-07 10:00:10] Training  [   10/15000]  eta: 2:29:50  lr: 0.000352  wd: 0.040  ...
```

**关键指标：**
- `eta`：预计剩余时间
- `lr`：当前学习率
- `wd`：当前权重衰减
- `total_loss`：总损失
- `dino_local_crops_loss`：DINO 局部裁剪损失
- `ibot_loss`：iBOT 损失

### 2. TensorBoard 可视化

```bash
# 启动 TensorBoard
tensorboard --logdir=/path/to/output --port=6006

# 在浏览器中打开
# http://localhost:6006
```

可以查看：
- 损失曲线
- 学习率变化
- 权重衰减变化
- 梯度分布

### 3. 训练指标文件

所有指标会保存到 `output_dir/training_metrics.json`：

```json
{
  "0": {
    "lr": 0.00035,
    "wd": 0.04,
    "total_loss": 5.234,
    ...
  },
  "10": {
    "lr": 0.000352,
    "wd": 0.04,
    "total_loss": 5.123,
    ...
  }
}
```

### 4. 检查点保存

检查点保存在 `output_dir/` 下：

```
output_dir/
├── config.yaml                    # 训练配置
├── training_metrics.json          # 训练指标
├── model_0001000.rank_0.pth      # 检查点（iteration 1000）
├── model_0002000.rank_0.pth
└── last_checkpoint.rank_0         # 最新检查点路径
```

**检查点保存频率：**
- 由 `saveckp_freq` 控制（单位：epoch）
- 默认每 10 个 epoch 保存一次
- 最多保留 3 个检查点（`max_to_keep=3`）

### 5. 监控训练状态

#### 检查 GPU 使用情况

```bash
# 实时监控
watch -n 1 nvidia-smi

# 或使用 gpustat
pip install gpustat
gpustat -i 1
```

#### 检查内存使用

```bash
# 查看进程内存
ps aux | grep python | grep train

# 查看系统内存
free -h
```

#### 尾随日志文件

```bash
tail -f /path/to/output/log.txt
```

---

## Resume 训练

### 1. 自动 Resume

训练脚本会自动检测并恢复最新的检查点：

```bash
# 直接运行，会自动 resume
python -m train.train \
  --config-file configs/train/prostate_vitb14_mm-dino.yaml \
  --output-dir /path/to/output
```

### 2. 禁用 Resume

如果想从头开始训练（忽略已有检查点）：

```bash
python -m train.train \
  --config-file configs/train/prostate_vitb14_mm-dino.yaml \
  --output-dir /path/to/output \
  --no-resume
```

### 3. 从指定检查点恢复

```bash
python -m train.train \
  --config-file configs/train/prostate_vitb14_mm-dino.yaml \
  --output-dir /path/to/output \
  MODEL.WEIGHTS=/path/to/model_0001000.rank_0.pth
```

### 4. 修改训练参数后 Resume

如果需要改变学习率或其他参数后继续训练：

```bash
python -m train.train \
  --config-file configs/train/prostate_vitb14_mm-dino.yaml \
  --output-dir /path/to/output \
  optim.base_lr=1e-4 \
  optim.epochs=400  # 延长训练
```

**注意：**
- 优化器状态会恢复，学习率会从恢复的 iteration 处开始调度
- 如果大幅修改训练参数，建议新建输出目录重新训练

---

## 下游任务

### 1. 提取预训练特征

```python
import torch
from dinov2.models import build_model_from_cfg
from omegaconf import OmegaConf

# 加载配置
cfg = OmegaConf.load("configs/train/prostate_vitb14_mm-dino.yaml")

# 构建模型
model = build_model_from_cfg(cfg, only_teacher=True)

# 加载预训练权重
checkpoint = torch.load("output/eval/best/teacher_checkpoint.pth")
model.load_state_dict(checkpoint["teacher"])
model.eval()

# 推理
with torch.no_grad():
    # 输入：(batch, channels, height, width)
    # 对于 3 个序列：channels=3
    # 对于 3 个序列 + mask：channels=4
    features = model(input_tensor)  # (batch, num_patches, embed_dim)
```

### 2. 分割任务微调

```python
# 使用预训练 backbone 初始化分割模型
from dinov2.models import build_model_from_cfg

# 1. 加载预训练 backbone
backbone = build_model_from_cfg(cfg, only_teacher=True)
checkpoint = torch.load("pretrained_weights.pth")
backbone.load_state_dict(checkpoint["teacher"])

# 2. 构建分割模型（例如 UNet）
class SegmentationModel(nn.Module):
    def __init__(self, backbone, num_classes=2):
        super().__init__()
        self.backbone = backbone
        self.decoder = UNetDecoder(embed_dim=768, num_classes=num_classes)
    
    def forward(self, x):
        features = self.backbone(x)
        mask = self.decoder(features)
        return mask

# 3. 微调
model = SegmentationModel(backbone, num_classes=2)
optimizer = torch.optim.Adam(model.parameters(), lr=1e-4)
# ... 训练循环
```

### 3. 分类任务微调

```python
# 使用预训练特征进行前列腺癌分级
from dinov2.eval.log_regression import eval_log_regression_with_model

# 线性探测（Linear Probing）
val_results = eval_log_regression_with_model(
    model=model,  # 预训练模型
    train_dataset_str="ProstateSupervised:split=TRAIN:root=/data",
    val_dataset_str="ProstateSupervised:split=VAL:root=/data",
    metric_types=["accuracy", "f1", "auc"],
    num_workers=4,
)

print("Validation Accuracy:", val_results["accuracy"])
```

### 4. Few-Shot 学习

预训练模型在少样本场景下表现优异：

```python
# 使用少量标注数据微调
train_dataset = make_dataset(
    dataset_str="ProstateSupervised:split=TRAIN:root=/data",
    # 仅使用 10% 的标注数据
    percentage_labels=0.1,
)

# 冻结 backbone，仅训练分类头
for param in backbone.parameters():
    param.requires_grad = False

classifier = nn.Linear(backbone.embed_dim, num_classes)
optimizer = torch.optim.Adam(classifier.parameters(), lr=1e-3)
```

---

## 常见问题 FAQ

### Q1: 找不到数据集 / 路径不对

**问题：**
```
FileNotFoundError: [Errno 2] No such file or directory: '/path/to/data/patient_001'
```

**解决方案：**
1. 检查 `dataset_path` 中的 `root` 是否为绝对路径
2. 确认 CSV 文件中的病例 ID 与数据目录名称完全一致
3. 运行数据质量检查脚本（见"数据准备"部分）

```bash
# 确认路径
ls /path/to/prostate_dataset/patient_001/
```

### Q2: CUDA / 显存不够

**问题：**
```
RuntimeError: CUDA out of memory
```

**解决方案：**
1. **减小 batch size**：
   ```yaml
   train:
     batch_size_per_gpu: 4  # 从 8 减到 4
   ```

2. **减小模型尺寸**：
   ```yaml
   student:
     arch: glioma_vit_small  # 从 base 改为 small
   ```

3. **减小图像尺寸**：
   ```yaml
   crops:
     global_crops_size: 192  # 从 224 减到 192
   ```

4. **启用梯度检查点**（需要修改代码）：
   ```python
   # 在模型定义中
   torch.utils.checkpoint.checkpoint_sequential(...)
   ```

### Q3: 学习率设置不当导致 loss 为 NaN

**问题：**
```
AssertionError: NaN detected in loss
```

**解决方案：**
1. **降低学习率**：
   ```yaml
   optim:
     base_lr: 1e-4  # 从 3.5e-4 降低
   ```

2. **增加 warmup**：
   ```yaml
   optim:
     warmup_epochs: 50  # 从 30 增加到 50
   ```

3. **检查数据**：确认 NIfTI 文件没有异常值
   ```python
   import nibabel as nib
   img = nib.load("patient_001/ax_t2wi.nii.gz").get_fdata()
   print(f"Min: {img.min()}, Max: {img.max()}, Mean: {img.mean()}")
   ```

4. **启用梯度裁剪**（已默认启用）：
   ```yaml
   optim:
     clip_grad: 3.0
   ```

### Q4: mask/ROI 太小导致增强算子报错

**问题：**
```
RuntimeError: size mismatch in RandomResizedCrop
```

**解决方案：**
1. **禁用基于 ROI 的裁剪**：
   ```yaml
   crops:
     crop_from_tumor_foreground: False
   ```

2. **增大最小肿瘤尺寸**（需要修改 `io.py`）：
   ```python
   LoadTumorSliced(
       keys=[...],
       min_tumor_size=100,  # 从 1 增加到 100
       ...
   )
   ```

3. **过滤小 ROI 病例**：在 CSV 中移除 ROI 过小的病例

### Q5: xFormers 安装失败

**问题：**
```
ModuleNotFoundError: No module named 'xformers'
```

**解决方案：**

**方法 1：使用环境变量跳过**
```bash
export XFORMERS_DISABLED=1
python -m train.train ...
```

**方法 2：安装预编译版本**
```bash
# CUDA 12.1
pip install xformers --index-url https://download.pytorch.org/whl/cu121

# CUDA 11.8
pip install xformers --index-url https://download.pytorch.org/whl/cu118
```

**方法 3：从源码编译**
```bash
pip install -v -U git+https://github.com/facebookresearch/xformers.git@main#egg=xformers
```

### Q6: 多卡训练时速度没有提升

**可能原因：**
1. **数据加载瓶颈**：增加 `num_workers`
   ```yaml
   train:
     num_workers: 8  # 从 4 增加到 8
   ```

2. **小 batch size**：增加 `batch_size_per_gpu`
   ```yaml
   train:
     batch_size_per_gpu: 12  # 尽量提高
   ```

3. **通信瓶颈**：检查网络（InfiniBand > 10GbE > 1GbE）

4. **FSDP 策略不当**：尝试 `FULL_SHARD`
   ```yaml
   compute_precision:
     student:
       backbone:
         sharding_strategy: FULL_SHARD  # 从 SHARD_GRAD_OP 改为 FULL_SHARD
   ```

### Q7: 训练时内存持续增长

**可能原因：**
1. **未释放 cache**：添加定期清理
   ```python
   # 在训练循环中每隔 100 次迭代
   if iteration % 100 == 0:
       torch.cuda.empty_cache()
   ```

2. **数据加载器泄漏**：减少 `num_workers`
   ```yaml
   train:
     num_workers: 2  # 降低到 2
   ```

3. **日志累积**：缩短日志记录频率

### Q8: 如何判断训练是否正常

**正常训练的特征：**
1. **Loss 下降**：前 10-20 个 epoch 应该明显下降
   - 初始 loss: ~5-7
   - 100 epoch 后: ~3-4
   - 收敛时: ~2-3

2. **学习率调度**：
   - Warmup 阶段：学习率从 0 逐渐升高
   - 稳定阶段：保持在 base_lr
   - Cosine 衰减：逐渐降低到 min_lr

3. **显存使用稳定**：第 10 次迭代后显存应该稳定

4. **训练速度稳定**：it/s 在 warmup 后应该稳定

**异常情况：**
- ❌ Loss 始终不变或上升
- ❌ Loss 突然变为 NaN
- ❌ 显存持续增长
- ❌ 训练速度持续下降

### Q9: 预训练需要多久

**训练时间估算（单卡 RTX 4090）：**
- ViT-B/14, 300 epochs, 400 病例: ~3-4 小时
- ViT-L/14, 300 epochs, 400 病例: ~6-8 小时

**多卡加速比：**
- 2 卡: ~1.8x
- 4 卡: ~3.2x
- 8 卡: ~5.5x

**建议：**
- 小规模实验（<100 病例）：50-100 epochs
- 中等规模（100-500 病例）：200-300 epochs
- 大规模（>500 病例）：300-500 epochs

### Q10: 如何选择最佳检查点

**方法 1：基于验证集性能**
- 如果配置了评估，训练脚本会自动保留最佳检查点
- 最佳检查点保存在 `output/eval/best/teacher_checkpoint.pth`

**方法 2：基于训练损失**
- 查看 `training_metrics.json`
- 选择 loss 最低的 checkpoint

**方法 3：在下游任务上测试**
- 加载不同 checkpoint
- 在分割/分类任务上评估性能
- 选择下游性能最好的

---

## 项目结构

```
Prostate/
├── README.md                          # 项目说明（当前文档）
├── ENGINEERING_ASSESSMENT.md          # 工程评估报告
├── requirements.txt                   # Python 依赖（需创建）
└── dinov2/                            # 主代码目录
    ├── __init__.py
    ├── configs/                       # 配置文件
    │   ├── ssl_default_config.yaml   # 默认配置
    │   └── train/                     # 训练配置
    │       └── prostate_vitb14_mm-dino.yaml  # 前列腺配置
    ├── data/                          # 数据加载与处理
    │   ├── datasets/
    │   │   ├── prostate_ssl.py       # 前列腺 SSL 数据集
    │   │   └── medical_dataset.py     # 医学数据集基类
    │   ├── monai_transforms/          # MONAI 变换
    │   │   ├── io.py                  # 数据加载
    │   │   └── spatial.py             # 空间变换
    │   ├── augmentations.py           # 数据增强
    │   ├── loaders.py                 # 数据加载器
    │   └── transforms.py              # 变换工具
    ├── models/                        # 模型定义
    │   ├── __init__.py               # 模型构建
    │   ├── glioma_vit.py             # 多模态 ViT
    │   └── vision_transformer.py      # 基础 ViT
    ├── train/                         # 训练脚本
    │   ├── train.py                  # 主训练脚本
    │   └── ssl_meta_arch.py          # SSL 架构
    ├── fsdp/                          # FSDP 支持
    │   └── __init__.py
    ├── loss/                          # 损失函数
    │   ├── dino_clstoken_loss.py
    │   ├── ibot_patch_loss.py
    │   └── koleo_loss.py
    ├── layers/                        # 神经网络层
    │   ├── attention.py
    │   ├── block.py
    │   └── patch_embed.py
    ├── utils/                         # 工具函数
    │   ├── config.py
    │   ├── utils.py
    │   └── dtype.py
    ├── tests/                         # 测试
    │   └── test_prostate_ssl_training.py
    └── visualization/                 # 可视化
        └── train/
            └── vis_loss.py
```

---

## 引用

如果你使用本项目，请引用以下论文：

```bibtex
@article{oquab2023dinov2,
  title={DINOv2: Learning Robust Visual Features without Supervision},
  author={Oquab, Maxime and Darcet, Timoth{\'e}e and Moutakanni, Th{\'e}o and Vo, Huy and Szafraniec, Marc and Khalidov, Vasil and Fernandez, Pierre and Haziza, Daniel and Massa, Francisco and El-Nouby, Alaaeldin and others},
  journal={arXiv preprint arXiv:2304.07193},
  year={2023}
}

@article{chen2023towards,
  title={Towards a general-purpose foundation model for computational pathology},
  author={Chen, Richard J and Ding, Tong and Lu, Ming Y and Williamson, Drew FK and Jaume, Guillaume and Song, Andrew H and Chen, Bowen and Zhang, Andrew and Shao, Daniel and Shaban, Muhammad and others},
  journal={Nature Medicine},
  year={2024}
}
```

---

## 许可证

本项目基于 Meta Platforms, Inc. 的 DINOv2 项目开发，遵循相应的开源许可证。

---

## 联系方式

如有问题或建议，请通过以下方式联系：
- 提交 GitHub Issue
- 邮件：[你的邮箱]

---

**祝训练顺利！🚀**
