# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

import logging
import os
import random
import subprocess
from urllib.parse import urlparse

import numpy as np
import torch
from PIL import Image
from torch import nn

logger = logging.getLogger("dinov2")


def load_pretrained_weights(model, pretrained_weights, checkpoint_key):
    """Loads pretrained weights into a model, with support for URL loading and checkpoint key selection."""
    if urlparse(pretrained_weights).scheme:  # If it looks like an URL
        state_dict = torch.hub.load_state_dict_from_url(
            pretrained_weights, map_location="cpu"
        )
    else:
        state_dict = torch.load(pretrained_weights, map_location="cpu")
    if checkpoint_key is not None and checkpoint_key in state_dict:
        logger.info(f"Take key {checkpoint_key} in provided checkpoint dict")
        state_dict = state_dict[checkpoint_key]
    # remove `module.` prefix
    state_dict = {k.replace("module.", ""): v for k, v in state_dict.items()}
    # remove `backbone.` prefix induced by multicrop wrapper
    state_dict = {k.replace("backbone.", ""): v for k, v in state_dict.items()}
    msg = model.load_state_dict(state_dict, strict=False)
    logger.info(
        "Pretrained weights found at {} and loaded with msg: {}".format(
            pretrained_weights, msg
        )
    )


def fix_random_seeds(seed=31):
    """
    Fix random seeds.
    """
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)


def show_image_from_tensor(image: torch.Tensor):
    """Displays a single-channel image from a PyTorch tensor."""
    image = image.permute(1, 2, 0).numpy()
    image = image[:, :, 0]
    image = Image.fromarray(np.uint8(image))
    image.show()


def get_sha():
    """Retrieves the current git commit SHA, status, and branch for logging purposes."""
    cwd = os.path.dirname(os.path.abspath(__file__))

    def _run(command):
        return subprocess.check_output(command, cwd=cwd).decode("ascii").strip()

    sha = "N/A"
    diff = "clean"
    branch = "N/A"
    try:
        sha = _run(["git", "rev-parse", "HEAD"])
        subprocess.check_output(["git", "diff"], cwd=cwd)
        diff = _run(["git", "diff-index", "HEAD"])
        diff = "has uncommitted changes" if diff else "clean"
        branch = _run(["git", "rev-parse", "--abbrev-ref", "HEAD"])
    except Exception:
        pass
    message = f"sha: {sha}, status: {diff}, branch: {branch}"
    return message


class CosineScheduler(object):
    """Cosine schedule with optional warmup and freeze periods, padded/truncated to total_iters."""

    @staticmethod
    def _to_int_iters(value, name):
        ivalue = int(value)
        if ivalue < 0:
            raise ValueError(f"{name} must be non-negative, got {value}.")
        return ivalue

    def __init__(
        self,
        base_value,
        final_value,
        total_iters,
        warmup_iters=0,
        start_warmup_value=0,
        freeze_iters=0,
    ):
        super().__init__()
        self.final_value = final_value
        self.total_iters = self._to_int_iters(total_iters, "total_iters")
        self.warmup_iters = self._to_int_iters(warmup_iters, "warmup_iters")
        self.freeze_iters = self._to_int_iters(freeze_iters, "freeze_iters")

        cosine_iters = max(self.total_iters - self.warmup_iters - self.freeze_iters, 0)

        freeze_schedule = np.zeros(self.freeze_iters, dtype=float)

        warmup_schedule = (
            np.linspace(start_warmup_value, base_value, self.warmup_iters)
            if self.warmup_iters > 0
            else np.array([], dtype=float)
        )

        if cosine_iters > 0:
            iters = np.arange(cosine_iters, dtype=float)
            cosine_schedule = final_value + 0.5 * (base_value - final_value) * (
                1 + np.cos(np.pi * iters / max(cosine_iters, 1))
            )
        else:
            cosine_schedule = np.array([], dtype=float)

        schedule = np.concatenate((freeze_schedule, warmup_schedule, cosine_schedule))

        if len(schedule) < self.total_iters:
            pad = np.full(self.total_iters - len(schedule), final_value, dtype=float)
            schedule = np.concatenate((schedule, pad))
        elif len(schedule) > self.total_iters:
            schedule = schedule[: self.total_iters]

        if len(schedule) != self.total_iters:
            raise ValueError(
                f"CosineScheduler length mismatch: expected {self.total_iters}, "
                f"got {len(schedule)} (freeze={self.freeze_iters}, warmup={self.warmup_iters}, "
                f"cosine={cosine_iters})."
            )

        self.schedule = schedule

    def __getitem__(self, it):
        if it >= self.total_iters:
            return self.final_value
        else:
            return self.schedule[it]


def has_batchnorms(model):
    """Checks if a model contains any batch normalization layers."""
    bn_types = (nn.BatchNorm1d, nn.BatchNorm2d, nn.BatchNorm3d, nn.SyncBatchNorm)
    for name, module in model.named_modules():
        if isinstance(module, bn_types):
            return True
    return False
