mm-dinov2 (Prostate MRI SSL fork)
=================================

This fork adds prostate MRI self-supervised pretraining to mm-dinov2, with a stable fp32 pipeline and single-GPU-friendly behavior. Use this README to configure and run training reproducibly.

Environment
-----------
- OS: WSL2 Ubuntu (Win11)
- GPU: RTX 4090 (CUDA 12.x)
- Python 3.10, PyTorch 2.x
- xFormers installed (fp32 path runs with xFormers in fp32; set `XFORMERS_DISABLED=1` to force pure PyTorch)

Key defaults
------------
- Dtype: global **float32** (no GradScaler; consistent fp32 through data, model, and xFormers).
- Single GPU: FSDP wrapping is skipped; checkpointing uses plain state_dict.
- Multi GPU: original FSDP behavior is preserved.
- Scheduler: robust cosine with warmup/freeze padding (see `utils/utils.py`).

Data layout
-----------
Each patient lives under `root/<patient_id>/` with:
- `ax_t2wi.nii(.gz)`
- `ax_adc.nii(.gz)`
- `ax_dwi_*.nii(.gz)` (highest b-value is chosen)
- `roi_Prostate.nii(.gz)` (optional segmentation)

Optional split CSV: include a column with patient ids (auto-detected from `patient_id / case_id / id / ID / subject / name`).

Dataset string (train.dataset_path)
-----------------------------------
Format: `ProstateSSL:split=TRAIN:root=/path:split_csv=split/train.csv:mri_sequences=ax_t2wi,ax_adc,ax_dwi:random_axes=True:random_slices=True`

Supported flags:
- `root`: dataset root
- `split`: TRAIN | VAL | TEST (for prostate SSL the CSV controls split)
- `split_csv`: CSV path for split
- `mri_sequences`: comma list (default `ax_t2wi,ax_adc,ax_dwi`)
- `random_axes`: sample axial/coronal/sagittal in TRAIN
- `random_slices`: random slice selection in TRAIN
- `append_label_mask`: add segmentation mask as an extra channel
- `percentage_labels`: fraction (0–1) of samples that keep the mask channel when available; others receive a zero mask

All modalities and masks are resized to `spatial_size` (default 224). Images use bilinear; masks use nearest and are binarized.

Config to start from
--------------------
- `configs/train/prostate_vitb14_mm-dino.yaml`

Edit for your run:
- `train.dataset_path` (root, split_csv, sequences, random_axes/slices)
- `train.output_dir`
- `train.batch_size_per_gpu`, `train.num_workers`, `train.OFFICIAL_EPOCH_LENGTH`
- `train.percentage_labels`
- `crops.crop_from_tumor_foreground` (also controls `append_label_mask`)
- `optim.base_lr`, `optim.epochs`, `optim.warmup_epochs`, `optim.weight_decay`
- Model: `student.arch`, `patch_size`, `drop_path_rate`, MRI embedding flags (`use_mri_seq_embed`, `img_wise_pos_embed`)

Runtime flag handling
---------------------
`train/train.py` appends at runtime (only once):
- `append_label_mask=<crops.crop_from_tumor_foreground>`
- `percentage_labels=<train.percentage_labels>`

fp32 strategy is enforced in `enforce_fp32_training`; mixed_precision param/reduce/buffer dtypes are set to fp32 and GradScaler is disabled. Inputs are cast to `torch.float32`.

Run training (single GPU)
-------------------------
```bash
python -m train.train \
  --config-file configs/train/prostate_vitb14_mm-dino.yaml \
  --output-dir /out
```

Smoke test (quick sanity)
-------------------------
```bash
export PROSTATE_DATA_ROOT=/path/to/small_prostate_subset   # 2–4 patients
python tests/test_prostate_ssl_training.py
```
Runs 5 iterations in fp32 to catch dtype/shape/checkpoint issues.

What to check before a run
--------------------------
- `train.dataset_path` points to the correct root/CSV; flags are not duplicated.
- `train.output_dir` exists or is creatable.
- `percentage_labels` matches desired mask usage.
- GPU visible (`torch.cuda.is_available()`).
- Optional: `XFORMERS_DISABLED=1` if debugging fused ops.

Files of interest in this fork
------------------------------
- `data/datasets/prostate_ssl.py`: mask alignment, percentage_labels, resizing
- `data/monai_transforms/io.py`: robust slicing, missing seg handling, nearest for masks
- `models/__init__.py`: channel counting from dataset flags, mask channel included
- `train/train.py`: fp32 strategy, dataset flag append, safe GliomaDinoViT unwrap
- `utils/utils.py`: robust cosine scheduler
- `fsdp/__init__.py`: safe checkpointing when not FSDP-wrapped
- `layers/block.py`: dtype-safe scaled_index_add path
- `tests/test_prostate_ssl_training.py`: smoke test
