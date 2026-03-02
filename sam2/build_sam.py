# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.

import logging
import os
from contextlib import contextmanager

import torch
from hydra import compose, initialize_config_dir
from hydra.utils import instantiate
from omegaconf import OmegaConf


def _resolve_config_dir():
    """
    Resolve config directory without relying on Hydra package discovery.
    Expected structure:
      <repo_root>/sam2_configs/*.yaml
    """
    here = os.path.dirname(os.path.abspath(__file__))          # .../sam2
    repo_root = os.path.abspath(os.path.join(here, ".."))      # project root
    config_dir = os.path.join(repo_root, "sam2_configs")
    if not os.path.isdir(config_dir):
        raise FileNotFoundError(
            f"sam2_configs directory not found at: {config_dir}\n"
            f"Expected: <repo_root>/sam2_configs/<config_file>.yaml"
        )
    return config_dir


@contextmanager
def _hydra_config_ctx(config_dir: str):
    """
    Hydra must be initialized before compose(). In scripts, Hydra may already be initialized;
    re-initialization will raise. This context manager avoids hard dependency on external setup.
    """
    try:
        with initialize_config_dir(config_dir=config_dir, version_base=None):
            yield
    except ValueError:
        # Hydra already initialized somewhere else (e.g., another call in the same process).
        # In that case, we just compose with the existing global hydra state.
        yield


def build_sam2(
    config_file,
    ckpt_path=None,
    device="cuda",
    mode="eval",
    hydra_overrides_extra=None,
    apply_postprocessing=True,
):
    if hydra_overrides_extra is None:
        hydra_overrides_extra = []

    if apply_postprocessing:
        hydra_overrides_extra = hydra_overrides_extra.copy()
        hydra_overrides_extra += [
            # dynamically fall back to multi-mask if the single mask is not stable
            "++model.sam_mask_decoder_extra_args.dynamic_multimask_via_stability=true",
            "++model.sam_mask_decoder_extra_args.dynamic_multimask_stability_delta=0.05",
            "++model.sam_mask_decoder_extra_args.dynamic_multimask_stability_thresh=0.98",
        ]

    config_dir = _resolve_config_dir()

    # Read config and init model (no package discovery)
    with _hydra_config_ctx(config_dir):
        cfg = compose(config_name=config_file, overrides=hydra_overrides_extra)

    OmegaConf.resolve(cfg)
    model = instantiate(cfg.model, _recursive_=True)
    _load_checkpoint(model, ckpt_path)
    model = model.to(device)
    if mode == "eval":
        model.eval()
    return model


def build_sam2_video_predictor(
    config_file,
    ckpt_path=None,
    device="cuda",
    mode="eval",
    hydra_overrides_extra=None,
    apply_postprocessing=True,
):
    if hydra_overrides_extra is None:
        hydra_overrides_extra = []

    hydra_overrides = [
        "++model._target_=sam2.sam2_video_predictor.SAM2VideoPredictor",
    ]
    if apply_postprocessing:
        hydra_overrides_extra = hydra_overrides_extra.copy()
        hydra_overrides_extra += [
            # dynamically fall back to multi-mask if the single mask is not stable
            "++model.sam_mask_decoder_extra_args.dynamic_multimask_via_stability=true",
            "++model.sam_mask_decoder_extra_args.dynamic_multimask_stability_delta=0.05",
            "++model.sam_mask_decoder_extra_args.dynamic_multimask_stability_thresh=0.98",
            # the sigmoid mask logits on interacted frames with clicks in the memory encoder so that the encoded masks are exactly as what users see from clicking
            "++model.binarize_mask_from_pts_for_mem_enc=true",
            # fill small holes in the low-res masks up to `fill_hole_area` (before resizing them to the original video resolution)
            "++model.fill_hole_area=8",
        ]
    hydra_overrides.extend(hydra_overrides_extra)

    config_dir = _resolve_config_dir()

    # Read config and init model (no package discovery)
    with _hydra_config_ctx(config_dir):
        cfg = compose(config_name=config_file, overrides=hydra_overrides)

    OmegaConf.resolve(cfg)
    model = instantiate(cfg.model, _recursive_=True)
    _load_checkpoint(model, ckpt_path)
    model = model.to(device)
    if mode == "eval":
        model.eval()
    return model


def _load_checkpoint(model, ckpt_path):
    if ckpt_path is not None:
        sd = torch.load(ckpt_path, map_location="cpu")["model"]
        missing_keys, unexpected_keys = model.load_state_dict(sd)
        if missing_keys:
            logging.error(missing_keys)
            raise RuntimeError("Missing keys when loading checkpoint.")
        if unexpected_keys:
            logging.error(unexpected_keys)
            raise RuntimeError("Unexpected keys when loading checkpoint.")
        logging.info("Loaded checkpoint successfully")