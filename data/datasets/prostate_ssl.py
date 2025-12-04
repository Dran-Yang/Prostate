import logging
import re
from enum import Enum
from pathlib import Path
from typing import Callable, Optional, Sequence

import numpy as np
import pandas as pd
import torch
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
        split: str | _Split,
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
    ) -> None:
        if isinstance(split, self.Split):
            split_enum = split
        else:
            try:
                split_enum = self.Split(str(split).lower())
            except ValueError as exc:
                valid = ", ".join(member.value for member in self.Split)
                raise ValueError(f"Unsupported split '{split}'; expected one of: {valid}") from exc

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
        if spatial_size is not None:
            if isinstance(spatial_size, (int, float)):
                self.spatial_size = (int(spatial_size), int(spatial_size))
            else:
                self.spatial_size = tuple(int(x) for x in spatial_size)

        super().__init__(split_enum, root, transforms, transform, target_transform)

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

    def _init_load_transform(self):
        self.img_load_transform = Compose(
            [
                SubjectDirToProstateFPsDict(
                    keys=[*self.mri_sequences, "seg"], seg_name="roi_Prostate"
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

    def _check_size(self):
        logger.info(f"Found {len(self.images)} patients for split '{self.split}'.")

    def get_num_classes(self) -> int:
        return 0

    def is_3d(self) -> bool:
        return False

    def is_multilabel(self) -> bool:
        return False

    def get_image_data(self, index: int) -> torch.Tensor:
        subject_dir = Path(self._split_dir) / self.images[index]
        subject_dict = self.img_load_transform(subject_dir)

        modality_tensors: list[torch.Tensor] = []
        for key in self.mri_sequences:
            modality = subject_dict[key]
            if modality.dim() == 3 and modality.size(0) == 1:
                modality = modality.squeeze(0)
            modality_tensors.append(modality)

        image = torch.stack(modality_tensors, dim=0)

        if image.size(0) == 1:
            image = image.repeat(3, 1, 1)

        if self.append_label_mask:
            label_mask = subject_dict["seg"]
            if label_mask.dim() == 3 and label_mask.size(0) == 1:
                label_mask = label_mask.squeeze(0)
            image = torch.cat([image, label_mask.unsqueeze(0)], dim=0)

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
