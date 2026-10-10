"""Evaluate full RADAR without ground-truth-derived prompts."""

from __future__ import annotations

import argparse
import logging
import math
from pathlib import Path
from typing import Dict

import torch
from PIL import Image
from tqdm import tqdm

import config
from datasets.radar_dataset import build_dataloader, load_text_annotations
from model.build_model import build_radar
from tools.checkpointing import load_best_checkpoint
from tools.metrics import compute_metrics

LOGGER = logging.getLogger("radar.test")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Test full RADAR.")
    parser.add_argument(
        "--dataset", choices=config.SUPPORTED_DATASETS, default=config.DEFAULT_DATASET
    )
    parser.add_argument("--data-root", type=Path, default=config.DEFAULT_DATA_ROOT)
    parser.add_argument(
        "--biomedclip-root", type=Path, default=config.DEFAULT_BIOMEDCLIP_ROOT
    )
    parser.add_argument(
        "--medsam2-checkpoint", type=Path, default=config.DEFAULT_MEDSAM2_CHECKPOINT
    )
    parser.add_argument("--medsam2-config", default=config.DEFAULT_MEDSAM2_CONFIG)
    parser.add_argument("--checkpoint", type=Path)
    parser.add_argument(
        "--checkpoint-dir", type=Path, default=config.DEFAULT_CHECKPOINT_DIR
    )
    parser.add_argument("--output-dir", type=Path, default=config.DEFAULT_OUTPUT_DIR)
    parser.add_argument("--device", default=config.DEFAULT_DEVICE)
    parser.add_argument("--batch-size", type=int, default=config.DEFAULT_BATCH_SIZE)
    parser.add_argument("--num-workers", type=int, default=config.DEFAULT_NUM_WORKERS)
    parser.add_argument("--validate-data", action="store_true")
    parser.add_argument("--save-predictions", action="store_true")
    return parser.parse_args()


def _save_predictions(
    probability: torch.Tensor, filenames: list[str], directory: Path
) -> None:
    directory.mkdir(parents=True, exist_ok=True)
    masks = probability.gt(0.5).to(torch.uint8).mul(255).cpu().numpy()
    for mask, filename in zip(masks, filenames):
        output_name = f"{Path(filename).stem}.png"
        Image.fromarray(mask[0], mode="L").save(directory / output_name)


def main() -> None:
    args = parse_args()
    logging.basicConfig(
        level=logging.INFO, format="%(asctime)s | %(levelname)s | %(message)s"
    )
    if args.num_workers < 0:
        raise ValueError("--num-workers must be non-negative.")
    device = torch.device(args.device)
    checkpoint = args.checkpoint or (
        args.checkpoint_dir / args.dataset / "best_checkpoint.pth"
    )

    # Inference deliberately omits the frozen SDRA teacher.
    model, tokenizer = build_radar(
        medsam2_config=args.medsam2_config,
        medsam2_checkpoint=args.medsam2_checkpoint,
        biomedclip_repository=args.biomedclip_root,
        device=device,
        enable_sdra_teacher=False,
    )
    load_best_checkpoint(checkpoint, model)
    model.eval()

    split_root = args.data_root / args.dataset / "Test_Folder"
    annotations = load_text_annotations(split_root / "Test_text.xlsx")
    loader = build_dataloader(
        split_root,
        annotations,
        tokenizer,
        is_train=False,
        max_text_length=config.text_length_for(args.dataset),
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        validate_data=args.validate_data,
    )

    totals: Dict[str, float] = {}
    counts: Dict[str, int] = {}
    LOGGER.info("Testing starts")
    with torch.no_grad():
        for batch in tqdm(loader, unit="batch"):
            image_sam = batch["image_sam"].to(device, non_blocking=True)
            image_clip = batch["image_clip"].to(device, non_blocking=True)
            tokens = batch["tokens"].to(device, non_blocking=True)
            target = batch["mask"].to(device, non_blocking=True)

            # Ground truth is used only after inference for metric computation.
            prediction, _ = model(image_sam, image_clip, tokens)
            metrics = compute_metrics(prediction, target)
            batch_size = prediction.shape[0]
            for key, value in metrics.items():
                if math.isfinite(value):
                    totals[key] = totals.get(key, 0.0) + value * batch_size
                    counts[key] = counts.get(key, 0) + batch_size
            if args.save_predictions:
                _save_predictions(
                    prediction, batch["filename"], args.output_dir / args.dataset
                )

    averaged = {key: totals[key] / counts[key] for key in totals}
    for key in (
        "Dice",
        "mIoU",
        "HD95",
        "ASSD",
        "Accuracy",
        "Sensitivity",
        "Specificity",
    ):
        LOGGER.info("%s: %.4f", key, averaged.get(key, float("nan")))


if __name__ == "__main__":
    main()
