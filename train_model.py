"""Train the full RADAR segmentation model."""

from __future__ import annotations

import argparse
import logging
import math
import random
from pathlib import Path
from typing import Dict, Optional

import numpy as np
import torch
from torch import nn
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingWarmRestarts
from torch.utils.data import DataLoader
from tqdm import tqdm

import config
from datasets.radar_dataset import build_dataloader, load_text_annotations
from model.build_model import build_radar, full_radar_config
from tools.checkpointing import (
    resume_training,
    save_best_checkpoint,
    save_last_checkpoint,
)
from tools.losses import WeightedBCEDiceLoss
from tools.metrics import compute_metrics

LOGGER = logging.getLogger("radar.train")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train full RADAR.")
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
    parser.add_argument(
        "--checkpoint-dir", type=Path, default=config.DEFAULT_CHECKPOINT_DIR
    )
    parser.add_argument("--device", default=config.DEFAULT_DEVICE)
    parser.add_argument("--batch-size", type=int, default=config.DEFAULT_BATCH_SIZE)
    parser.add_argument("--num-workers", type=int, default=config.DEFAULT_NUM_WORKERS)
    parser.add_argument("--epochs", type=int, default=config.EPOCHS)
    parser.add_argument("--learning-rate", type=float, default=config.LEARNING_RATE)
    parser.add_argument(
        "--early-stopping-patience", type=int, default=config.EARLY_STOPPING_PATIENCE
    )
    parser.add_argument("--seed", type=int, default=config.DEFAULT_SEED)
    parser.add_argument("--resume", type=Path)
    parser.add_argument("--validate-data", action="store_true")
    return parser.parse_args()


def seed_everything(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def _move_batch(batch: Dict[str, object], device: torch.device) -> Dict[str, object]:
    return {
        key: value.to(device, non_blocking=True) if torch.is_tensor(value) else value
        for key, value in batch.items()
    }


def _metric_accumulator() -> Dict[str, float]:
    return {
        key: 0.0
        for key in (
            "Loss",
            "SegmentationLoss",
            "SimilarityLoss",
            "SDRALoss",
            "Dice",
            "mIoU",
            "HD95",
            "ASSD",
            "Accuracy",
            "Sensitivity",
            "Specificity",
        )
    }


def run_epoch(
    model: nn.Module,
    loader: DataLoader,
    criterion: nn.Module,
    device: torch.device,
    optimizer: Optional[torch.optim.Optimizer],
) -> Dict[str, float]:
    training = optimizer is not None
    model.train(training)
    totals = _metric_accumulator()
    counts = {key: 0 for key in totals}

    for batch in tqdm(loader, leave=False, unit="batch"):
        batch = _move_batch(batch, device)
        with torch.set_grad_enabled(training):
            prediction, extra = model(
                batch["image_sam"],
                batch["image_clip"],
                batch["tokens"],
                sdra_teacher_crops=batch["sdra_teacher_crops"] if training else None,
                sdra_region_weights=batch["sdra_region_weights"] if training else None,
                sdra_valid_mask=batch["sdra_valid_mask"] if training else None,
            )
            segmentation_loss = criterion(prediction, batch["mask"])
            similarity_loss = criterion(extra["similarity_probability"], batch["mask"])
            sdra_loss = extra["sdra_loss"]
            loss = segmentation_loss + 0.2 * similarity_loss + 0.2 * sdra_loss
            if training:
                optimizer.zero_grad(set_to_none=True)
                loss.backward()
                optimizer.step()

        batch_size = prediction.shape[0]
        losses = {
            "Loss": float(loss.detach()),
            "SegmentationLoss": float(segmentation_loss.detach()),
            "SimilarityLoss": float(similarity_loss.detach()),
            "SDRALoss": float(sdra_loss.detach()),
        }
        metrics = compute_metrics(prediction.detach(), batch["mask"])
        for key, value in {**losses, **metrics}.items():
            if math.isfinite(value):
                totals[key] += value * batch_size
                counts[key] += batch_size
    return {key: totals[key] / max(1, counts[key]) for key in totals}


def _optimizer(model: nn.Module, learning_rate: float) -> AdamW:
    decay, no_decay = [], []
    for name, parameter in model.named_parameters():
        if not parameter.requires_grad:
            continue
        if parameter.ndim == 1 or name.endswith(".bias") or "norm" in name.lower():
            no_decay.append(parameter)
        else:
            decay.append(parameter)
    return AdamW(
        [
            {"params": decay, "weight_decay": 0.1},
            {"params": no_decay, "weight_decay": 0.0},
        ],
        lr=learning_rate,
        betas=(0.9, 0.999),
        eps=1e-8,
    )


def _split_loader(
    dataset_root: Path,
    split_name: str,
    tokenizer,
    args: argparse.Namespace,
    *,
    is_train: bool,
) -> DataLoader:
    split_root = dataset_root / f"{split_name}_Folder"
    text_map = load_text_annotations(split_root / f"{split_name}_text.xlsx")
    return build_dataloader(
        split_root,
        text_map,
        tokenizer,
        is_train=is_train,
        max_text_length=config.text_length_for(args.dataset),
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        validate_data=args.validate_data,
    )


def main() -> None:
    args = parse_args()
    logging.basicConfig(
        level=logging.INFO, format="%(asctime)s | %(levelname)s | %(message)s"
    )
    if args.num_workers < 0:
        raise ValueError("--num-workers must be non-negative.")
    seed_everything(args.seed)
    device = torch.device(args.device)

    model, tokenizer = build_radar(
        medsam2_config=args.medsam2_config,
        medsam2_checkpoint=args.medsam2_checkpoint,
        biomedclip_repository=args.biomedclip_root,
        device=device,
        enable_sdra_teacher=True,
    )
    dataset_root = args.data_root / args.dataset
    train_loader = _split_loader(dataset_root, "Train", tokenizer, args, is_train=True)
    validation_loader = _split_loader(
        dataset_root, "Val", tokenizer, args, is_train=False
    )

    criterion = WeightedBCEDiceLoss(weights=(0.5, 0.5))
    optimizer = _optimizer(model, args.learning_rate)
    scheduler = CosineAnnealingWarmRestarts(optimizer, T_0=10, T_mult=1, eta_min=1e-6)
    checkpoint_directory = args.checkpoint_dir / args.dataset
    best_path = checkpoint_directory / "best_checkpoint.pth"
    last_path = checkpoint_directory / "last_checkpoint.pth"

    start_epoch = 1
    best_dice = float("-inf")
    best_epoch = 0
    if args.resume is not None:
        resume_state = resume_training(
            args.resume, model, optimizer, scheduler, scaler=None
        )
        start_epoch = resume_state["epoch"] + 1
        best_dice = resume_state["best_metric"]
        best_epoch = int(resume_state["training_state"].get("best_epoch", 0))
        LOGGER.info("Resumed training from epoch %d", resume_state["epoch"])

    LOGGER.info("Training starts")
    for epoch in range(start_epoch, args.epochs + 1):
        train_metrics = run_epoch(model, train_loader, criterion, device, optimizer)
        LOGGER.info("Validation starts")
        with torch.no_grad():
            validation_metrics = run_epoch(
                model, validation_loader, criterion, device, optimizer=None
            )
        scheduler.step(epoch)

        LOGGER.info(
            "Epoch %d/%d | train loss %.4f Dice %.4f | validation loss %.4f Dice %.4f",
            epoch,
            args.epochs,
            train_metrics["Loss"],
            train_metrics["Dice"],
            validation_metrics["Loss"],
            validation_metrics["Dice"],
        )
        if validation_metrics["Dice"] > best_dice:
            best_dice = validation_metrics["Dice"]
            best_epoch = epoch
            # Epoch 1 is eligible; the best deployment checkpoint is trainable-only.
            save_best_checkpoint(best_path, model, full_radar_config())
            LOGGER.info("Saved best checkpoint at epoch %d", epoch)

        save_last_checkpoint(
            last_path,
            model,
            optimizer,
            scheduler,
            scaler=None,
            epoch=epoch,
            best_metric=best_dice,
            training_state={"best_epoch": best_epoch},
        )
        if epoch - best_epoch >= args.early_stopping_patience:
            LOGGER.info("Early stopping triggered at epoch %d", epoch)
            break


if __name__ == "__main__":
    main()
