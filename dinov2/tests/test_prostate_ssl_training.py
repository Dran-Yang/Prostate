"""
Minimal smoke test to verify the prostate SSL pipeline can run a few iterations
without dtype/shape/FSDP checkpoint errors. Set PROSTATE_DATA_ROOT to a tiny
dataset (2-4 patients) before running.
"""

import os
from functools import partial
from pathlib import Path

import torch
from omegaconf import OmegaConf

from dinov2.configs import dinov2_default_config
from dinov2.data import (
    DataAugmentationDINO,
    MaskingGenerator,
    SamplerType,
    collate_data_and_cast,
    make_data_loader,
    make_dataset,
)
from dinov2.train.ssl_meta_arch import SSLMetaArch
from dinov2.train.train import enforce_fp32_training, ensure_dataset_path_flags


def _load_cfg(data_root: Path) -> OmegaConf:
    base_cfg = OmegaConf.create(dinov2_default_config)
    train_cfg = OmegaConf.load("configs/train/prostate_vitb14_mm-dino.yaml")
    cfg = OmegaConf.merge(base_cfg, train_cfg)
    cfg.train.dataset_path = cfg.train.dataset_path.replace(
        "<PATH_TO_PROSTATE_DATASET>", str(data_root)
    )
    ensure_dataset_path_flags(cfg)
    cfg.train.output_dir = str(Path(cfg.train.output_dir) / "smoke_test")
    cfg.train.batch_size_per_gpu = min(cfg.train.batch_size_per_gpu, 2)
    cfg.train.num_workers = 0
    cfg.optim.epochs = 1
    cfg.optim.warmup_epochs = 0
    cfg.train.OFFICIAL_EPOCH_LENGTH = 5
    return cfg


def run_smoke_train():
    data_root = Path(os.environ.get("PROSTATE_DATA_ROOT", ""))
    if not data_root.exists():
        print("Skipping smoke test: set PROSTATE_DATA_ROOT to a small dataset.")
        return

    cfg = _load_cfg(data_root)
    enforce_fp32_training(cfg)

    model = SSLMetaArch(cfg).cuda()
    model.prepare_for_distributed_training()
    model.train()

    optimizer = torch.optim.AdamW(model.get_params_groups(), lr=1e-4)

    img_size = cfg.crops.global_crops_size
    patch_size = cfg.student.patch_size
    n_tokens = (img_size // patch_size) ** 2
    mask_generator = MaskingGenerator(
        input_size=(img_size // patch_size, img_size // patch_size),
        max_num_patches=n_tokens // 2,
    )

    data_transform = DataAugmentationDINO(
        cfg.crops.global_crops_scale,
        cfg.crops.local_crops_scale,
        cfg.crops.local_crops_number,
        global_crops_size=cfg.crops.global_crops_size,
        local_crops_size=cfg.crops.local_crops_size,
        intensity_aug_name=cfg.crops.intensity_aug,
        crop_from_tumor_foreground=cfg.crops.crop_from_tumor_foreground,
        max_blur_radius=cfg.crops.max_blur_radius,
        gamma_range=cfg.crops.gamma_range,
    )
    collate_fn = partial(
        collate_data_and_cast,
        mask_ratio_tuple=cfg.ibot.mask_ratio_min_max,
        mask_probability=cfg.ibot.mask_sample_probability,
        mask_per_channel=cfg.ibot.mask_per_channel,
        n_tokens=n_tokens,
        mask_generator=mask_generator,
        dtype=torch.float32,
    )

    dataset = make_dataset(
        dataset_str=cfg.train.dataset_path,
        transform=data_transform,
        target_transform=lambda _: (),
    )
    data_loader = make_data_loader(
        dataset=dataset,
        batch_size=cfg.train.batch_size_per_gpu,
        num_workers=cfg.train.num_workers,
        shuffle=True,
        seed=0,
        sampler_type=SamplerType.INFINITE,
        sampler_advance=0,
        drop_last=True,
        collate_fn=collate_fn,
    )

    iterator = iter(data_loader)
    for step in range(5):
        batch = next(iterator)
        optimizer.zero_grad(set_to_none=True)
        loss_dict = model.forward_backward(batch, teacher_temp=cfg.teacher.teacher_temp)
        optimizer.step()
        model.update_teacher(cfg.teacher.momentum_teacher)
        loss_value = sum(v.item() for v in loss_dict.values())
        print(f"[step {step}] loss={loss_value:.4f}")

    print("Smoke test completed without runtime errors.")


if __name__ == "__main__":
    torch.cuda.set_device(0)
    run_smoke_train()
