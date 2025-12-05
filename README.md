# Prostate Multi-Modal DINOv2

This repository adapts the [DINOv2](https://arxiv.org/abs/2304.07193) vision transformer pipeline to self-supervised learning on prostate MRI. It includes utilities for preparing patient volumes, slicing tumor-centered 2D crops, and training a multi-modal ViT backbone that embeds each MRI sequence separately.

## Repository layout
- `data/datasets/prostate_ssl.py` – self-supervised dataset for prostate MRI patients. It builds 2D slices centered on the tumor, normalizes intensities, and supports random axial/coronal/sagittal sampling during training.
- `dinov2/data/monai_transforms/io.py` – MONAI transforms to resolve patient filepaths, pick the highest-b DWI volume, crop around the tumor center of mass, and load the corresponding slice for each modality.
- `dinov2/utils/register_prostate.py` – helper to rigid/affine register patient volumes before training.
- `dinov2/configs/train/prostate_vitb14_mm-dino.yaml` – training hyperparameters for the prostate setup, including multi-modal student/teacher ViT settings and crop augmentation defaults.

## Data preparation
1. **Patient folder structure**
   Place each subject under `<DATA_ROOT>/<PATIENT_ID>/` with (at minimum):
   - `ax_t2wi.nii` (or `.nii.gz`)
   - `ax_adc.nii`
   - `ax_dwi*.nii*` (the loader will pick the highest b-value file)
   - `roi_Prostate.nii` (segmentation mask)

2. **Optional registration**
   Align ADC/DWI volumes to T2-weighted space with SimpleITK:
   ```bash   python -m dinov2.utils.register_prostate \
     --data-root <DATA_ROOT> \
     --output-root <REGISTERED_ROOT> \
     --ref-seq ax_t2wi \
     --moving-seqs ax_adc ax_dwi \
     --seg-name roi_Prostate
   ```
   The script mirrors patient folders into `<REGISTERED_ROOT>` and resamples masks with nearest-neighbor interpolation.

3. **Splits**
   If you provide a CSV split file, include a patient identifier column such as `patient_id`, `case_id`, `id`, or `subject` (the loader uses the first matching column). Otherwise, the dataset iterates over all patient folders under `root`.

## Training
1. **Environment** – Install PyTorch (with CUDA if available) and common dependencies used by DINOv2/ MONAI (e.g., `torch`, `monai`, `fvcore`).
2. **Edit the config** – Update `dinov2/configs/train/prostate_vitb14_mm-dino.yaml` so `train.dataset_path` points to your data, split CSV, and MRI sequences. The default expects three modalities (`ax_t2wi,ax_adc,ax_dwi`) and enables random slice/axis sampling during training.
3. **Launch training** – Run the SSL trainer with your config and an output directory for checkpoints/logs:
   ```bash
   python -m dinov2.train.train \
     --config-file dinov2/configs/train/prostate_vitb14_mm-dino.yaml \
     --output-dir outputs/prostate_ssl
   ```
   The config uses `OFFICIAL_EPOCH_LENGTH` to approximate steps per epoch (set it to `ceil(num_patients / batch_size_per_gpu)`). Teacher/student weights can optionally warm-start from DINOv2 checkpoints via `student.pretrained_weights`.

## Dataset details
- **MRI sequences** – Pass a comma-separated list (e.g., `ax_t2wi,ax_adc,ax_dwi`) when building `ProstateSSL`; the loader squeezes singleton channel dimensions and stacks them into `(C, H, W)` tensors. When only one modality is present it is automatically triplicated to mimic RGB.
- **Tumor-centered slicing** – During training, random axes and slices are sampled around the tumor center of mass (subject to `random_axes`/`random_slices` flags). Validation/test default to axial slices.
- **Intensity scaling** – Intensities are percentile-clipped per modality (1st–99th percentiles) and mapped to `[0, 1]` before augmentations.

## Reproducibility tips
- Fix seeds in your launcher (`torch.manual_seed`, `numpy`, etc.) if you need deterministic slice selection.
- Monitor the console log for the resolved patient count and chosen split column when supplying a CSV.
- Verify a sample after constructing `ProstateSSL`:
  ```python
  ds = ProstateSSL(
      split="train",
      root="<DATA_ROOT>",
      mri_sequences="ax_t2wi,ax_adc,ax_dwi",
      split_csv="split/train.csv",
      random_axes=True,
      random_slices=True,
  )
  img, _ = ds[0]
  print(img.shape)  # should be torch.Size([3, 224, 224])
  ```

 
EOF
)
