# Quick Start Guide - 前列腺 MRI SSL 预训练

## 最快 5 分钟开始训练

### 1. 环境准备（2 分钟）

```bash
# 创建环境
conda create -n prostate-ssl python=3.10 -y
conda activate prostate-ssl

# 安装依赖
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu121
pip install -r requirements.txt
pip install xformers  # 可选但强烈推荐
```

### 2. 准备数据（1 分钟）

确保数据按以下结构组织：

```
/path/to/data/
├── patient_001/
│   ├── ax_t2wi.nii.gz
│   ├── ax_adc.nii.gz
│   ├── ax_dwi_b1000.nii.gz
│   └── roi_Prostate.nii.gz  # 可选
├── patient_002/
│   └── ...
```

创建训练集 CSV：

```bash
mkdir -p split
echo "patient_id" > split/train.csv
ls /path/to/data/ >> split/train.csv
```

### 3. 修改配置（1 分钟）

编辑 `dinov2/configs/train/prostate_vitb14_mm-dino.yaml`：

```yaml
train:
  dataset_path: ProstateSSL:split=TRAIN:root=/path/to/data:split_csv=split/train.csv:mri_sequences=ax_t2wi,ax_adc,ax_dwi:random_axes=True:random_slices=True
  output_dir: ./output/run1
  batch_size_per_gpu: 8  # 根据显存调整
```

### 4. 开始训练（1 分钟）

```bash
cd dinov2
python -m train.train \
  --config-file configs/train/prostate_vitb14_mm-dino.yaml \
  --output-dir ../output/run1
```

**就是这么简单！** 🎉

---

## 完整文档

- 📘 **完整使用手册**：见 [README_COMPREHENSIVE.md](README_COMPREHENSIVE.md)
- 🔧 **技术评估报告**：见 [ENGINEERING_ASSESSMENT.md](ENGINEERING_ASSESSMENT.md)

---

## 常见问题速查

### Q: CUDA out of memory
**A:** 降低 `batch_size_per_gpu` 到 4 或更低

### Q: xFormers 安装失败
**A:** 运行时添加 `export XFORMERS_DISABLED=1`

### Q: 找不到数据
**A:** 确认 `root` 路径是绝对路径，CSV 中的 patient_id 与文件夹名完全匹配

### Q: Loss 是 NaN
**A:** 降低学习率（`base_lr: 1e-4`）或增加 warmup（`warmup_epochs: 50`）

---

## 项目特点

✅ **多模态支持**：T2WI + ADC + DWI  
✅ **医学优化**：ROI 引导裁剪  
✅ **稳定训练**：fp32 策略  
✅ **高效加速**：xFormers + FSDP  
✅ **易于使用**：一个命令开始训练  

---

## 更新日志

### 2024-12-07
- ✅ 修复了 7 个关键 bug
- ✅ 添加了完整的文档
- ✅ 改进了错误处理
- ✅ 放宽了 xFormers 依赖

---

**祝训练顺利！** 如有问题请查看 [FAQ](README_COMPREHENSIVE.md#常见问题-faq) 或提交 Issue。
