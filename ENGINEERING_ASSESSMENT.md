# 前列腺 MRI 自监督预训练工程 - 技术评估报告

## 目录
1. [项目结构评估](#1-项目结构评估)
2. [代码与配置问题清单](#2-代码与配置问题清单)
3. [训练与实验建议](#3-训练与实验建议)
4. [优化建议总结](#4-优化建议总结)

---

## 1. 项目结构评估

### 1.1 当前目录结构

```
Prostate/
├── README.md                          # 项目说明文档
└── dinov2/
    ├── configs/                       # 配置文件
    │   ├── ssl_default_config.yaml   # 默认配置
    │   └── train/                     # 训练配置
    │       └── prostate_vitb14_mm-dino.yaml
    ├── data/                          # 数据加载与增强
    │   ├── datasets/                  # 数据集定义
    │   │   ├── prostate_ssl.py       # 前列腺SSL数据集 ✓
    │   │   ├── glioma_ssl.py
    │   │   └── medical_dataset.py
    │   ├── monai_transforms/          # MONAI变换
    │   │   ├── io.py                  # 数据加载与切片
    │   │   ├── spatial.py
    │   │   └── rc_aug.py
    │   ├── augmentations.py           # 数据增强
    │   ├── loaders.py                 # 数据加载器
    │   └── transforms.py              # 变换工具
    ├── models/                        # 模型定义
    │   ├── __init__.py               # 模型构建函数
    │   ├── glioma_vit.py             # 多模态ViT
    │   └── vision_transformer.py      # 基础ViT
    ├── train/                         # 训练脚本
    │   ├── train.py                  # 主训练脚本 ✓
    │   └── ssl_meta_arch.py          # SSL架构
    ├── fsdp/                          # FSDP支持
    │   └── __init__.py               # FSDP包装器与检查点
    ├── loss/                          # 损失函数
    │   ├── dino_clstoken_loss.py
    │   ├── ibot_patch_loss.py
    │   └── koleo_loss.py
    ├── layers/                        # 网络层
    │   ├── attention.py
    │   ├── block.py
    │   └── patch_embed.py
    ├── utils/                         # 工具函数
    │   ├── config.py                 # 配置加载
    │   ├── utils.py                  # 通用工具
    │   └── dtype.py                  # 数据类型处理
    ├── tests/                         # 测试
    │   └── test_prostate_ssl_training.py
    └── visualization/                 # 可视化
        └── train/
            └── vis_loss.py
```

### 1.2 结构评估

**优点：**
- ✅ 模块化清晰：数据、模型、训练、工具各自独立
- ✅ 配置与代码分离：使用 YAML 配置文件
- ✅ 已有测试脚本：`test_prostate_ssl_training.py` 可以快速验证
- ✅ FSDP 支持完善：适配单卡和多卡训练
- ✅ FP32 策略完整：在 `train.py` 中有 `enforce_fp32_training` 函数

**需要改进的地方：**

1. **缺少依赖管理文件**
   - ❌ 没有 `requirements.txt` 或 `environment.yml`
   - 建议：添加 `requirements.txt` 列出所有依赖

2. **入口点不明确**
   - ⚠️ 训练需要用 `python -m train.train`，但模块路径在 `dinov2` 下
   - 建议：在项目根目录添加 `train.py` 作为入口点，或在 README 中明确说明需要 `cd dinov2 && python -m train.train`

3. **配置文件路径硬编码**
   - ⚠️ `prostate_vitb14_mm-dino.yaml` 中 `root=<PATH_TO_PROSTATE_DATASET>` 需要用户手动替换
   - 建议：在 README 中明确说明，或使用环境变量

---

## 2. 代码与配置问题清单

### 2.1 数据集模块 (`data/datasets/prostate_ssl.py`)

#### 问题 1：`split_enum` 未定义
**位置：** `prostate_ssl.py` 第 87 行
```python
super().__init__(split_enum, root, transforms, transform, target_transform)
```

**问题描述：** 
- `split_enum` 变量未定义，应该是 `split` 参数转换为枚举类型

**修复方案：**
```python
# 第 87 行之前添加
if isinstance(split, str):
    split_enum = self.Split[split.upper()]
else:
    split_enum = split

super().__init__(split_enum, root, transforms, transform, target_transform)
```

#### 问题 2：数据增强可能导致尺寸为 0
**位置：** `monai_transforms/io.py` `LoadTumorSliced` 类

**问题描述：**
- 当 ROI 非常小或不存在时，`calculate_crop_slices` 可能产生空区域
- 会导致后续 resize 操作失败

**修复方案：**
```python
# 在 calculate_crop_slices 函数中添加最小尺寸检查
def calculate_crop_slices(
    com: torch.Tensor,
    spatial_crop_size: tuple[int, int],
    spatial_img_size: Sequence[int],
) -> list[slice]:
    """Calculate crop slices for a given center of mass and crop size."""
    
    com_int = com.int()
    spatial_crop_size_torch = torch.tensor(spatial_crop_size).to(com_int.device).long()
    spatial_img_size_torch = torch.tensor(spatial_img_size).to(com_int.device).long()

    # 确保裁剪尺寸不超过图像尺寸
    spatial_crop_size_torch = torch.min(spatial_crop_size_torch, spatial_img_size_torch)
    
    crop_start = torch.clamp(com_int - spatial_crop_size_torch // 2, 0)
    crop_end = crop_start + spatial_crop_size_torch
    if (crop_end > spatial_img_size_torch).any():
        crop_end = torch.min(crop_end, spatial_img_size_torch)
        crop_start = torch.clamp(crop_end - spatial_crop_size_torch, min=0)
        crop_end = torch.min(crop_start + spatial_crop_size_torch, spatial_img_size_torch)

    slice_ = [slice(start, end) for start, end in zip(crop_start, crop_end)]
    
    return slice_
```

#### 问题 3：DWI b 值选择不够灵活
**位置：** `monai_transforms/io.py` 第 57-62 行

**当前实现：**
```python
def _score(path: Path) -> int:
    m = re.search(r"(\d+)", path.stem)
    return int(m.group(1)) if m else -1

candidates.sort(key=_score, reverse=True)
```

**问题描述：**
- 仅根据文件名中的数字排序，可能误选非 b 值数字
- 对 `ax_dwi_b1000.nii` 和 `ax_dwi_1000.nii` 等不同命名格式不够鲁棒

**改进方案：**
```python
def _score(path: Path) -> int:
    """Extract b-value from DWI filename, prioritizing b=XXXX format."""
    # 优先匹配 b=1000 或 b1000 格式
    m = re.search(r'b[=_-]?(\d+)', path.stem, re.IGNORECASE)
    if m:
        return int(m.group(1))
    # 其次匹配任意数字
    m = re.search(r'(\d+)', path.stem)
    return int(m.group(1)) if m else -1

candidates.sort(key=_score, reverse=True)
logger.info(f"Selected DWI file: {candidates[0]} (b-value: {_score(candidates[0])})")
```

### 2.2 训练脚本 (`train/train.py`)

#### 问题 4：配置合并可能覆盖关键参数
**位置：** `train.py` 第 89-102 行 `ensure_dataset_path_flags`

**当前实现：**
```python
if "append_label_mask" not in cfg.train.dataset_path:
    cfg.train.dataset_path = (
        cfg.train.dataset_path
        + f":append_label_mask={cfg.crops.crop_from_tumor_foreground}"
    )
```

**问题描述：**
- 如果用户在 dataset_path 中已经指定了 `append_label_mask=False`，这段代码不会覆盖
- 但 README 中说是 "runtime flag append"，可能造成混淆

**建议：**
- 在 README 中明确说明：用户不应该在 dataset_path 中手动添加这些标志
- 或者在函数开始时先移除已有的这些标志，再添加新的

```python
def ensure_dataset_path_flags(cfg):
    """
    Append runtime flags to the dataset string exactly once so both model
    construction and dataloading see the same channel configuration.
    """
    dataset_name = cfg.train.dataset_path.split(":")[0]
    if dataset_name not in {"GliomaSSL", "GliomaSupervised", "ProstateSSL"}:
        return
    
    # 移除已有的标志（如果存在）
    tokens = cfg.train.dataset_path.split(":")
    filtered_tokens = [tokens[0]] + [
        t for t in tokens[1:] 
        if not (t.startswith("append_label_mask=") or t.startswith("percentage_labels="))
    ]
    cfg.train.dataset_path = ":".join(filtered_tokens)
    
    # 添加新的标志
    cfg.train.dataset_path = (
        cfg.train.dataset_path
        + f":append_label_mask={cfg.crops.crop_from_tumor_foreground}"
    )
    cfg.train.dataset_path = (
        cfg.train.dataset_path
        + f":percentage_labels={cfg.train.percentage_labels}"
    )
```

#### 问题 5：评估逻辑针对 glioma 数据集
**位置：** `train.py` 第 203-219 行 `do_eval_all_sequences`

**问题描述：**
- 评估函数使用 glioma 的序列组合 `["t1", "t1c", "t2", "flair"]`
- 对于前列腺数据集应该跳过或使用 `["ax_t2wi", "ax_adc", "ax_dwi"]`

**修复方案：**
```python
def do_eval_all_sequences(cfg, model, iteration):
    # 根据数据集类型选择序列
    dataset_name = cfg.train.dataset_path.split(":")[0]
    if dataset_name == "ProstateSSL":
        logger.info("Skipping multi-sequence evaluation for ProstateSSL (not applicable).")
        return
    elif dataset_name == "GliomaSSL":
        mri_sequences = ["t1", "t1c", "t2", "flair"]
    else:
        logger.info(f"Unknown dataset {dataset_name}, skipping multi-sequence eval.")
        return
    
    mri_sequence_combinations = list(powerset(mri_sequences, min_size=2, max_size=4))
    mri_sequence_combinations += ["random"]
    
    for mri_sequence_combination in mri_sequence_combinations[::-1]:
        mri_sequence_combination_str = "-".join(mri_sequence_combination)
        mri_sequence_combination_str = f"_mri_sequences-{mri_sequence_combination_str}"
        do_eval(cfg, model, f"{iteration}{mri_sequence_combination_str}", mri_sequence_combination)
```

#### 问题 6：xFormers 依赖硬性要求
**位置：** `train/ssl_meta_arch.py` 第 27-33 行

```python
try:
    from xformers.ops import fmha
    XFORMERS_AVAILABLE = True
except ImportError:
    XFORMERS_AVAILABLE = False
assert XFORMERS_AVAILABLE, "xFormers is required for DINOv2 training"
```

**问题描述：**
- 强制要求 xFormers，但某些环境可能难以安装（如 Windows）
- README 中提到可以设置 `XFORMERS_DISABLED=1`，但这里会直接报错

**改进方案：**
```python
try:
    from xformers.ops import fmha
    XFORMERS_AVAILABLE = True
except ImportError:
    XFORMERS_AVAILABLE = False

# 允许通过环境变量禁用 xFormers 检查
import os
if not XFORMERS_AVAILABLE and os.environ.get("XFORMERS_DISABLED") != "1":
    logger.warning(
        "xFormers is not available. Training will use standard PyTorch attention. "
        "For better performance, install xFormers. To suppress this warning, set "
        "XFORMERS_DISABLED=1."
    )
```

### 2.3 配置文件 (`configs/train/prostate_vitb14_mm-dino.yaml`)

#### 问题 7：学习率可能过小
**位置：** `prostate_vitb14_mm-dino.yaml` 第 39 行

```yaml
optim:
  base_lr: 1e-4
```

**问题描述：**
- 对于 batch_size=4 的单卡训练，学习率 1e-4 可能过小
- DINOv2 原始配置是 `base_lr: 0.004`（针对 batch_size=1024）
- 使用 sqrt scaling，单卡训练的学习率约为 `0.004 * sqrt(4/1024) ≈ 2.5e-4`

**建议：**
```yaml
optim:
  base_lr: 2.5e-4  # 或者 0.00025，适合单卡 batch_size=4
  # 如果使用更大的 batch size（如 8），可以适当增加到 3.5e-4
  epochs: 200
  warmup_epochs: 20  # 建议 warmup 占总 epochs 的 10%
```

#### 问题 8：OFFICIAL_EPOCH_LENGTH 需要根据数据量调整
**位置：** `prostate_vitb14_mm-dino.yaml` 第 5 行

```yaml
train:
  OFFICIAL_EPOCH_LENGTH: 4000
```

**问题描述：**
- 这个值应该根据实际数据集大小调整
- 建议设为 `ceil(num_patients / batch_size_per_gpu)`

**建议：**
在 README 中说明：
```
OFFICIAL_EPOCH_LENGTH 应该设置为每个 epoch 希望执行的迭代次数。
推荐值：ceil(训练集病例数 / batch_size_per_gpu)

例如：
- 100 个病例，batch_size=4 → OFFICIAL_EPOCH_LENGTH: 25
- 400 个病例，batch_size=4 → OFFICIAL_EPOCH_LENGTH: 100
```

#### 问题 9：评估配置未设置
**位置：** `prostate_vitb14_mm-dino.yaml` 第 51-52 行

```yaml
evaluation:
  eval_period_iterations: 0  # disable classification eval for SSL
```

**问题描述：**
- 评估被禁用，无法在训练过程中看到模型性能
- 对于监控训练进度不利

**建议：**
```yaml
evaluation:
  eval_period_iterations: 1000  # 每 1000 次迭代评估一次（约 10 epochs）
  # 如果没有标注的验证集，可以保持为 0
  # 如果有验证集，需要同时配置：
  # train_dataset_path: ProstateSupervised:split=TRAIN:root=...
  # val_dataset_path: ProstateSupervised:split=VAL:root=...
```

### 2.4 模型构建 (`models/__init__.py`)

#### 问题 10：模型配置读取逻辑复杂
**位置：** `models/__init__.py` 第 40-67 行

**问题描述：**
- 从 dataset_path 字符串解析模态信息
- 如果 dataset_path 格式不对，可能导致错误

**当前实现：**
```python
dataset_tokens = cfg.train.dataset_path.split(":")[1:]
dataset_kwargs = {}
for token in dataset_tokens:
    if "=" in token:
        key, value = token.split("=", 1)
        dataset_kwargs[key] = value
```

**改进建议：**
在 `train.py` 的 `ensure_dataset_path_flags` 函数后，添加一个验证函数：

```python
def validate_dataset_path(cfg):
    """验证 dataset_path 格式是否正确"""
    required_fields = ["root", "split"]
    dataset_path = cfg.train.dataset_path
    
    if not dataset_path:
        raise ValueError("train.dataset_path cannot be empty")
    
    tokens = dataset_path.split(":")
    dataset_name = tokens[0]
    
    if dataset_name not in {"GliomaSSL", "GliomaSupervised", "ProstateSSL"}:
        logger.warning(f"Unknown dataset: {dataset_name}")
        return
    
    kwargs = {}
    for token in tokens[1:]:
        if "=" in token:
            key, value = token.split("=", 1)
            kwargs[key] = value
    
    for field in required_fields:
        if field not in kwargs:
            raise ValueError(
                f"Required field '{field}' missing in dataset_path: {dataset_path}"
            )
    
    logger.info(f"Dataset path validation passed: {dataset_name} with {len(kwargs)} parameters")
```

---

## 3. 训练与实验建议

### 3.1 推荐的训练参数（前列腺 MRI）

基于医学影像自监督学习的最佳实践，推荐以下参数：

```yaml
train:
  batch_size_per_gpu: 8  # 如果显存允许，建议增加到 8
  num_workers: 4
  OFFICIAL_EPOCH_LENGTH: 50  # 假设 ~400 个病例，50 = ceil(400/8)
  
student:
  arch: glioma_vit_base  # ViT-B/14 是较好的平衡点
  patch_size: 14
  use_mri_seq_embed: True  # 对多模态 MRI 很重要
  img_wise_pos_embed: True  # 对多模态 MRI 很重要
  
optim:
  base_lr: 3.5e-4  # 适合 batch_size=8 的单卡训练
  epochs: 300  # 医学影像通常需要更多 epochs
  warmup_epochs: 30  # 10% 的 epochs 用于 warmup
  weight_decay: 0.04
  weight_decay_end: 0.4
  clip_grad: 3.0
  freeze_backbone_epochs: 0  # 从头训练不需要冻结
  
crops:
  global_crops_size: 224  # 适合前列腺 MRI
  local_crops_size: 112
  global_crops_scale: [0.5, 1.0]  # 前列腺通常占据较大区域
  local_crops_scale: [0.2, 0.5]
  crop_from_tumor_foreground: True  # 利用前列腺 ROI
  intensity_aug: rc  # RandConv 适合医学影像
  
dino:
  head_n_prototypes: 4096  # 相比 ImageNet 可以减少（数据量较小）
  
ibot:
  head_n_prototypes: 4096
  mask_per_channel: True  # 对多模态很重要
  
train:
  percentage_labels: 1.0  # 充分利用所有标注（如果有）
```

### 3.2 数据增强策略

对于 3D 医学影像（特别是前列腺 MRI），推荐以下增强组合：

**当前实现（已有）：**
- ✅ `RandomResizeForegroundCrop`：根据前列腺 ROI 裁剪
- ✅ `ScaleIntensityRangePercentilesd`：强度归一化
- ✅ `RandomHorizontalFlip`：水平翻转
- ✅ `RandConvAugmentation`：随机卷积增强（适合医学影像）

**建议添加（可选）：**
1. **轻微旋转**：前列腺方向可能略有不同
   ```python
   # 在 geometric_augmentation_global 中添加
   transforms.RandomRotation(degrees=10)  # 轻微旋转，不要太大
   ```

2. **弹性形变（可选）**：模拟器官形变
   ```python
   # 使用 MONAI 的 RandAffine
   from monai.transforms import RandAffined
   RandAffined(
       keys=mri_sequences,
       prob=0.3,
       rotate_range=(0.1, 0.1, 0.1),  # 轻微旋转
       scale_range=(0.1, 0.1, 0.1),   # 轻微缩放
       mode="bilinear",
   )
   ```

**不建议的增强：**
- ❌ 强烈的颜色抖动（Color Jitter）：MRI 是灰度图，颜色无意义
- ❌ 大角度旋转：前列腺 MRI 通常是轴位，旋转 >15° 不合理
- ❌ 过强的模糊：会丢失重要的边缘信息

### 3.3 混合精度 / xFormers / FSDP 建议

#### xFormers
```python
# 当前项目已采用 fp32 策略，xFormers 在 fp32 模式下也能提供加速
# 建议保持启用，除非遇到兼容性问题

# 如果需要禁用（调试用）：
export XFORMERS_DISABLED=1
```

#### 混合精度
```python
# 当前策略：enforce_fp32_training 强制使用 fp32
# 优点：稳定性好，适合医学影像
# 缺点：显存占用大，速度稍慢

# 如果显存充足（如 RTX 4090 24GB），建议保持 fp32
# 如果显存紧张，可以尝试 fp16（需要修改 enforce_fp32_training）
```

#### FSDP
```python
# 当前实现：
# - 单卡：自动跳过 FSDP 包装
# - 多卡：使用 FSDP SHARD_GRAD_OP

# 建议：
# - 单卡训练：保持当前设置（不使用 FSDP）
# - 2-4 卡：使用 SHARD_GRAD_OP（已是默认）
# - 8 卡以上：可以考虑 FULL_SHARD（需要在 config 中修改）
```

### 3.4 硬件需求与性能估计

**推荐配置：**
- GPU: NVIDIA RTX 3090 / 4090（24GB 显存）或更高
- CPU: 8 核以上
- RAM: 32GB 以上
- 磁盘: SSD（NIfTI 文件 I/O 密集）

**性能估计（单 RTX 4090）：**
- ViT-B/14, batch_size=8, fp32:
  - 训练速度：~2-3 it/s
  - 一个 epoch（50 iterations）：~20-25 秒
  - 总训练时间（300 epochs）：~2-3 小时

**显存使用估计：**
- ViT-B/14, batch_size=8, fp32: ~18-20GB
- ViT-L/14, batch_size=4, fp32: ~22-24GB

---

## 4. 优化建议总结

### 4.1 立即修复（高优先级）

1. **修复 `split_enum` 未定义错误** (`prostate_ssl.py`)
2. **改进 DWI b 值选择逻辑** (`monai_transforms/io.py`)
3. **添加数据集路径验证** (`train.py`)
4. **修复评估逻辑** (`train.py` - `do_eval_all_sequences`)
5. **放宽 xFormers 依赖检查** (`ssl_meta_arch.py`)

### 4.2 配置优化（中优先级）

1. **调整学习率**：`base_lr: 2.5e-4` → `3.5e-4`
2. **增加 batch size**：`4` → `8`（如果显存允许）
3. **调整 OFFICIAL_EPOCH_LENGTH**：根据数据集大小设置
4. **启用评估**：`eval_period_iterations: 1000`（如果有验证集）

### 4.3 代码改进（低优先级）

1. **添加 `requirements.txt`**
2. **改进 `ensure_dataset_path_flags` 逻辑**
3. **添加更多日志输出**（如 DWI 文件选择）
4. **优化裁剪逻辑**（处理极小 ROI）

### 4.4 文档完善

1. **创建详细的 README**（见 PART 2）
2. **添加数据准备指南**
3. **添加 FAQ 部分**
4. **提供命令行示例**

---

## 附录 A：推荐的训练主脚本

如果需要一个更简洁的训练入口，可以在项目根目录创建 `train_prostate.py`：

```python
#!/usr/bin/env python3
"""
前列腺 MRI 自监督预训练 - 训练脚本
使用方法：
    python train_prostate.py --data-root /path/to/data --output-dir ./output
"""
import argparse
import sys
from pathlib import Path

# 添加 dinov2 到 Python 路径
sys.path.insert(0, str(Path(__file__).parent / "dinov2"))

from train.train import main, get_args_parser

if __name__ == "__main__":
    # 创建参数解析器
    parser = get_args_parser(add_help=True)
    parser.add_argument(
        "--data-root",
        type=str,
        default=None,
        help="数据根目录（会自动更新到配置中）"
    )
    
    args = parser.parse_args()
    
    # 如果指定了 data-root，更新配置
    if args.data_root:
        args.opts.append(f"train.dataset_path=ProstateSSL:split=TRAIN:root={args.data_root}:mri_sequences=ax_t2wi,ax_adc,ax_dwi:random_axes=True:random_slices=True")
    
    # 运行训练
    main(args)
```

---

## 附录 B：代码审查清单

在提交代码前，请检查：

### 数据相关
- [ ] 数据路径在配置文件中正确设置
- [ ] CSV 分割文件存在且格式正确
- [ ] 所有病例都包含必需的模态（T2WI, ADC, DWI）
- [ ] ROI 文件存在（如果使用 `crop_from_tumor_foreground=True`）

### 配置相关
- [ ] `OFFICIAL_EPOCH_LENGTH` 根据数据集大小设置
- [ ] 学习率适合当前的 batch size
- [ ] `output_dir` 路径存在或可创建
- [ ] GPU 可见且 CUDA 可用

### 训练相关
- [ ] 第一次运行时使用小数据集验证（smoke test）
- [ ] 监控第一个 epoch 的内存使用
- [ ] 检查损失是否在合理范围（不是 NaN 或 Inf）
- [ ] 确认 checkpoint 正常保存

### 代码质量
- [ ] 没有硬编码的路径
- [ ] 日志输出充分
- [ ] 异常处理完善
- [ ] 代码符合 PEP 8 规范

---

**最后更新：** 2024-12-07  
**文档版本：** 1.0  
**作者：** AI Engineering Assistant
