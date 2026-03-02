#!/usr/bin/env python3
"""
Test SAM2UNetCDFSSAggressive on CD-FSS benchmarks.

Example:
  python test_sam2unet_cdfss.py \
      --load logs/sam2unet_cdfss.log/best_model.pt \
      --benchmark fss --split test --nshot 1 \
      --datapath_tgt ../CDFSL --datapath_src ../VOCdevkit

You can run multiple shots by calling this script twice (nshot=1 and nshot=5).

"""

from __future__ import annotations

import argparse

import torch
import torch.nn as nn

from cdfss.sam2unet_cdfss_aggressive import SAM2CDFSSConfig, SAM2UNetCDFSSAggressive
from data.dataset import FSSDataset
from common.logger import Logger, AverageMeter
from common.evaluation import Evaluator
from common import utils


@torch.no_grad()
def test(model: nn.Module, dataloader: torch.utils.data.DataLoader) -> tuple[float, float]:
    utils.fix_randseed(0)
    average_meter = AverageMeter(dataloader.dataset)

    model.eval()
    for idx, batch in enumerate(dataloader):
        batch = utils.to_cuda(batch)

        logits = model(batch["query_img"], batch["support_imgs"], batch["support_masks"])
        pred_mask = logits.argmax(dim=1)

        area_inter, area_union = Evaluator.classify_prediction(pred_mask.clone(), batch)
        average_meter.update(area_inter, area_union, batch["class_id"], loss=None)

        average_meter.write_process(idx, len(dataloader), epoch=-1, write_batch_idx=1)

    average_meter.write_result("Test", 0)
    miou, fb_iou = average_meter.compute_iou()
    return float(miou), float(fb_iou)


def main() -> None:
    parser = argparse.ArgumentParser("SAM2UNet CD-FSS (aggressive) test")

    parser.add_argument("--load", type=str, required=True, help="Path to trained model state_dict (.pt).")
    parser.add_argument("--sam2_cfg", type=str, default="sam2_hiera_l.yaml")
    parser.add_argument("--sam2_ckpt", type=str, default="", help="SAM2 checkpoint used to build the backbone. Needed only if you did NOT save the full model weights.")
    parser.add_argument("--embed_dim", type=int, default=256)
    parser.add_argument("--attn_heads", type=int, default=8)
    parser.add_argument("--num_fg_tokens", type=int, default=32)

    parser.add_argument("--benchmark", type=str, default="fss", choices=["pascal", "fss", "deepglobe", "isic", "lung"])
    parser.add_argument("--split", type=str, default="test", choices=["val", "test"])
    parser.add_argument("--fold", type=int, default=0, choices=[0, 1, 2, 3, 4])
    parser.add_argument("--nshot", type=int, default=1)
    parser.add_argument("--img_size", type=int, default=400)
    parser.add_argument("--bsz", type=int, default=30)
    parser.add_argument("--nworker", type=int, default=0)
    parser.add_argument("--datapath_src", type=str, default="../VOCdevkit")
    parser.add_argument("--datapath_tgt", type=str, default="../CDFSL")

    parser.add_argument("--dp", action="store_true", help="Enable DataParallel if multiple GPUs")
    parser.add_argument("--logpath", type=str, default="")

    args = parser.parse_args()

    Logger.initialize(args, training=False)
    Evaluator.initialize()

    cfg = SAM2CDFSSConfig(
        sam2_model_cfg=args.sam2_cfg,
        sam2_checkpoint=None if args.sam2_ckpt == "" else args.sam2_ckpt,
        embed_dim=args.embed_dim,
        attn_heads=args.attn_heads,
        num_fg_tokens=args.num_fg_tokens,
    )
    model = SAM2UNetCDFSSAggressive(cfg)
    if torch.cuda.is_available():
        model = model.cuda()

    if args.dp and torch.cuda.device_count() > 1:
        Logger.info(f"Using DataParallel on {torch.cuda.device_count()} GPUs")
        model = nn.DataParallel(model)

    if args.load == "":
        raise ValueError("--load is required")
    state = torch.load(args.load, map_location="cpu")
    model.load_state_dict(state, strict=True)

    # Dataset
    datapath = args.datapath_src if args.benchmark == "pascal" else args.datapath_tgt
    FSSDataset.initialize(img_size=args.img_size, datapath=datapath)
    dataloader = FSSDataset.build_dataloader(args.benchmark, args.bsz, args.nworker, args.fold, args.split, shot=args.nshot)

    with torch.no_grad():
        miou, fb_iou = test(model, dataloader)

    Logger.info("mIoU: %5.2f \t FB-IoU: %5.2f" % (miou, fb_iou))
    Logger.info("==================== Finished Testing ====================")


if __name__ == "__main__":
    main()
