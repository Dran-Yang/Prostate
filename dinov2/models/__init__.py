# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

import logging

from . import vision_transformer as vits

logger = logging.getLogger("dinov2")


def build_model(args, only_teacher=False, img_size=224, **kwargs):
    args.arch = args.arch.removesuffix("_memeff")
    if "vit" in args.arch:
        vit_kwargs = dict(
            img_size=img_size,
            patch_size=args.patch_size,
            init_values=args.layerscale,
            ffn_layer=args.ffn_layer,
            block_chunks=args.block_chunks,
            qkv_bias=args.qkv_bias,
            proj_bias=args.proj_bias,
            ffn_bias=args.ffn_bias,
        )
        teacher = vits.__dict__[args.arch](**vit_kwargs, **kwargs)
        if only_teacher:
            return teacher, teacher.embed_dim
        student = vits.__dict__[args.arch](
            **vit_kwargs,
            **kwargs,
            drop_path_rate=args.drop_path_rate,
            drop_path_uniform=args.drop_path_uniform,
        )
        embed_dim = student.embed_dim
    return student, teacher, embed_dim


def build_model_from_cfg(cfg, only_teacher=False):
    kwargs = {}
    dataset_tokens = cfg.train.dataset_path.split(":")[1:]
    dataset_kwargs = {}
    for token in dataset_tokens:
        if "=" in token:
            key, value = token.split("=", 1)
            dataset_kwargs[key] = value

    append_mask_flag = dataset_kwargs.get("append_label_mask", "false").lower() == "true"
    mri_sequences = None
    if "mri_sequences" in dataset_kwargs:
        mri_sequences = dataset_kwargs["mri_sequences"].split(",")
        if append_mask_flag:
            # append a pseudo modality name for the mask channel so MedicalDinoViT can embed it
            mri_sequences = [*mri_sequences, "seg"]
        # For non-medical architectures, set in_chans to the number of sequences
        # For medical_vit/glioma_vit, in_chans will be overridden below
        if not cfg.student.arch.startswith(("medical_vit", "glioma_vit")):
            kwargs["in_chans"] = len(mri_sequences)
    elif append_mask_flag:
        kwargs["in_chans"] = kwargs.get("in_chans", 3) + 1

    if cfg.student.arch.startswith(("medical_vit", "glioma_vit")) and mri_sequences is not None:
        kwargs["mri_sequences"] = mri_sequences
        kwargs["use_mri_seq_embed"] = cfg.student.use_mri_seq_embed
        kwargs["img_wise_pos_embed"] = cfg.student.img_wise_pos_embed
        # MedicalDinoViT converts each MRI sequence to 3-channel RGB internally,
        # so the PatchEmbed should always use in_chans=3
        kwargs["in_chans"] = 3

    return build_model(
        cfg.student, only_teacher=only_teacher, img_size=cfg.train.img_size, **kwargs
    )
