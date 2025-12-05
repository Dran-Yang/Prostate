import logging
from enum import Enum
from pathlib import Path
from typing import Callable, Optional, Sequence

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from monai.transforms.compose import Compose
from monai.transforms.intensity.dictionary import ScaleIntensityRangePercentilesd

from dinov2.data.datasets.medical_dataset import MedicalVisionDataset
from dinov2.data.monai_transforms.io import (
    LoadTumorSliced,
    SubjectDirToProstateFPsDict,
)

logger = logging.getLogger("dinov2")


class _Split(Enum):
    TRAIN = "train"
    VAL = "val"
    TEST = "test"

    @property
    def length(self) -> int:
        # length is unknown upfront for SSL; kept for interface compatibility.
        return 0


class ProstateSSL(MedicalVisionDataset):
    """
    Self-supervised prostate MRI dataset.

    Expects each patient to live under root/<patient_id>/ with files:
      - ax_t2wi.nii
      - ax_adc.nii
      - ax_dwi_*.nii (pick highest b value)
      - roi_Prostate.nii
    """

    Split = _Split
    spatial_size = (224, 224)

    def __init__(
        self,
        *,
        split: str,
        root: str,
        mri_sequences: str | Sequence[str] | None = None,
        split_csv: Optional[str] = None,
        transforms: Optional[Callable] = None,
        transform: Optional[Callable] = None,
        target_transform: Optional[Callable] = None,
        random_axes: bool = False,
        random_slices: bool = False,
        append_label_mask: bool = False,
        spatial_size: Optional[Sequence[int] | int | float] = None,
        percentage_labels: float = 1.0,
    ) -> None:
        self.seg_name = "roi_Prostate"
        if mri_sequences is not None and isinstance(mri_sequences, str):
            self.mri_sequences = mri_sequences.split(",")
        else:
            self.mri_sequences = list(
                mri_sequences
                if mri_sequences is not None
                else ["ax_t2wi", "ax_adc", "ax_dwi"]
            )

        self.split_csv = split_csv
        self.random_axes = random_axes
        self.random_slices = random_slices
        self.append_label_mask = append_label_mask
        self.percentage_labels = float(max(min(percentage_labels, 1.0), 0.0))
        self._mask_exists: list[bool] = []
        self._mask_usage: list[bool] = []

        if spatial_size is not None:
            if isinstance(spatial_size, (int, float)):
                self.spatial_size = (int(spatial_size), int(spatial_size))
            else:
                self.spatial_size = tuple(int(x) for x in spatial_size)

        super().__init__(split, root, transforms, transform, target_transform)

        self.class_names: Sequence[str] = []

    def _define_split_dir(self):
        # Override to avoid forcing train/val/test subfolders; we use CSV split or full root.
        self._split_dir = self._root

    def _init_images(self):
        if self.split_csv:
            df = pd.read_csv(self.split_csv)
            candidate_cols = [
                "patient_id",
                "case_id",
                "id",
                "ID",
                "subject",
                "name",
            ]
            col = next((c for c in candidate_cols if c in df.columns), df.columns[0])
            self.images = np.sort(df[col].astype(str).unique())
            logger.info(
                f"Loaded {len(self.images)} patients for split from CSV {self.split_csv} using column '{col}'."
            )
        else:
            super()._init_images()

        self._init_load_transform()
        self._init_mask_usage()

    def _init_load_transform(self):
        self.img_load_transform = Compose(
            [
                SubjectDirToProstateFPsDict(
                    keys=[*self.mri_sequences, "seg"], seg_name=self.seg_name
                ),
                LoadTumorSliced(
                    keys=[*self.mri_sequences, "seg"],
                    tumor_key="seg",
                    spatial_size=self.spatial_size,
                    axes=[0, 1, 2]
                    if self.split == self.Split.TRAIN and self.random_axes
                    else [2],
                    select_random_slices=self.random_slices
                    and self.split == self.Split.TRAIN,
                    min_tumor_size=1,
                    allow_missing_seg=True,
                ),
                ScaleIntensityRangePercentilesd(
                    keys=self.mri_sequences,
                    b_min=0.0,
                    b_max=1.0,
                    lower=1,
                    upper=99,
                    clip=True,
                ),
            ]
        )

    def _init_mask_usage(self) -> None:
        """
        Track which subjects have segmentation masks and pre-select which ones will expose
        the mask channel according to percentage_labels.
        """
        resolver = SubjectDirToProstateFPsDict(keys=["seg"], seg_name=self.seg_name)
        self._mask_exists = []
        for pid in self.images:
            seg_fp = resolver(Path(self._split_dir) / pid)["seg"]
            self._mask_exists.append(seg_fp.exists())

        available_idx = [i for i, has_seg in enumerate(self._mask_exists) if has_seg]
        if self.percentage_labels >= 1.0:
            selected_idx = available_idx
        elif self.percentage_labels <= 0.0 or len(available_idx) == 0:
            selected_idx = []
        else:
            rng = np.random.default_rng()
            target = max(int(round(len(available_idx) * self.percentage_labels)), 0)
            target = min(target, len(available_idx))
            selected_idx = (
                rng.choice(available_idx, size=target, replace=False).tolist()
                if target > 0
                else []
            )

        self._mask_usage = [False for _ in self.images]
        for idx in selected_idx:
            self._mask_usage[idx] = True

        logger.info(
            "Segmentation availability: %d/%d present; using %d (~%.2f requested).",
            sum(self._mask_exists),
            len(self._mask_exists),
            sum(self._mask_usage),
            self.percentage_labels,
        )

    def _check_size(self):
        logger.info(f"Found {len(self.images)} patients for split '{self.split}'.")

    def _resize_to_spatial_size(
        self, tensor: torch.Tensor, mode: str = "bilinear", is_mask: bool = False
    ) -> torch.Tensor:
        """
        Ensure all modalities (and optional masks) share the same spatial size.
        Bilinear for images, nearest for masks to keep them binary.
        """
        tensor = tensor.float()
        if tensor.ndim != 2:
            raise ValueError(f"Expected a 2D tensor, got shape {tensor.shape}.")

        if tuple(tensor.shape[-2:]) != tuple(self.spatial_size):
            align_corners = False if mode in ("bilinear", "bicubic") else None
            tensor = F.interpolate(
                tensor.unsqueeze(0).unsqueeze(0),
                size=self.spatial_size,
                mode=mode,
                align_corners=align_corners,
            ).squeeze(0).squeeze(0)

        if is_mask:
            tensor = (tensor > 0.5).float()

        return tensor

    def get_num_classes(self) -> int:
        return 0

    def is_3d(self) -> bool:
        return False

    def is_multilabel(self) -> bool:
        return False

    def get_image_data(self, index: int) -> torch.Tensor:
        subject_dir = Path(self._split_dir) / self.images[index]
        subject_dict = self.img_load_transform(subject_dir)

        image_channels = [
            self._resize_to_spatial_size(subject_dict[key], mode="bilinear")
            for key in self.mri_sequences
        ]
        image = torch.stack(image_channels, dim=0)

        if self.append_label_mask:
            seg_tensor = self._resize_to_spatial_size(
                subject_dict["seg"], mode="nearest", is_mask=True
            )
            use_mask = (
                index < len(self._mask_usage)
                and self._mask_usage[index]
                and self._mask_exists[index]
            )
            if not use_mask:
                seg_tensor = torch.zeros_like(seg_tensor)
            image = torch.cat([image, seg_tensor.unsqueeze(0)], dim=0)

        return image

    def get_target(self, index: int) -> torch.Tensor:
        return torch.tensor(-1, dtype=torch.long)

    def get_target_name(self, index: int) -> str:
        return "Unknown"

    def __len__(self) -> int:
        return len(self.images)

    def __getitem__(self, index: int):
        image = self.get_image_data(index)
        target = self.get_target(index)

        if self.transform is not None:
            image = self.transform(image)
        if isinstance(image, torch.Tensor):
            image[torch.isnan(image)] = 0
        return image, target
