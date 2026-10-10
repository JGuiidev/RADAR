"""Profile parameters, MACs, and FLOPs for full RADAR inference."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import torch

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import config
from model.build_model import build_radar
from tools.profiling import format_profile, operation_profile, parameter_profile


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Profile full RADAR at batch size one."
    )
    parser.add_argument(
        "--dataset", choices=config.SUPPORTED_DATASETS, default=config.DEFAULT_DATASET
    )
    parser.add_argument(
        "--biomedclip-root", type=Path, default=config.DEFAULT_BIOMEDCLIP_ROOT
    )
    parser.add_argument(
        "--medsam2-checkpoint", type=Path, default=config.DEFAULT_MEDSAM2_CHECKPOINT
    )
    parser.add_argument("--medsam2-config", default=config.DEFAULT_MEDSAM2_CONFIG)
    parser.add_argument("--device", default=config.DEFAULT_DEVICE)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    device = torch.device(args.device)
    model, _ = build_radar(
        medsam2_config=args.medsam2_config,
        medsam2_checkpoint=args.medsam2_checkpoint,
        biomedclip_repository=args.biomedclip_root,
        device=device,
        enable_sdra_teacher=False,
    )
    tokens = torch.zeros(
        (1, config.text_length_for(args.dataset)), dtype=torch.long, device=device
    )
    tokens[:, :2] = torch.tensor((101, 102), device=device)
    inputs = (
        torch.zeros((1, 3, 512, 512), device=device),
        torch.zeros((1, 3, 224, 224), device=device),
        tokens,
    )
    parameters = parameter_profile(model)
    operations = operation_profile(model, inputs)
    print(format_profile(parameters, operations))


if __name__ == "__main__":
    main()
