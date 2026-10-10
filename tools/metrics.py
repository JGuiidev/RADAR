"""Single authoritative metric implementation for RADAR evaluation."""

from __future__ import annotations

from typing import Optional, Tuple

import numpy as np
import torch
from scipy import ndimage


def _surface_distances(
    source: np.ndarray,
    target: np.ndarray,
    spacing: Optional[Tuple[float, float]],
) -> np.ndarray:
    structure = np.ones((3, 3), dtype=bool)
    source_surface = source & ~ndimage.binary_erosion(
        source, structure=structure, border_value=0
    )
    target_surface = target & ~ndimage.binary_erosion(
        target, structure=structure, border_value=0
    )
    distance_map = ndimage.distance_transform_edt(~target_surface, sampling=spacing)
    return distance_map[source_surface]


def _surface_metrics(
    prediction: np.ndarray,
    target: np.ndarray,
    spacing: Optional[Tuple[float, float]],
) -> Tuple[float, float]:
    if not prediction.any() and not target.any():
        return 0.0, 0.0
    if not prediction.any() or not target.any():
        return float("nan"), float("nan")
    prediction_to_target = _surface_distances(prediction, target, spacing)
    target_to_prediction = _surface_distances(target, prediction, spacing)
    if prediction_to_target.size == 0 or target_to_prediction.size == 0:
        return float("nan"), float("nan")
    hd95 = max(
        float(np.percentile(prediction_to_target, 95)),
        float(np.percentile(target_to_prediction, 95)),
    )
    assd = float((prediction_to_target.mean() + target_to_prediction.mean()) / 2.0)
    return hd95, assd


def compute_metrics(
    probability: torch.Tensor,
    target: torch.Tensor,
    *,
    threshold: float = 0.5,
    spacing: Optional[Tuple[float, float]] = None,
) -> dict:
    """Compute Dice, mIoU, HD95, ASSD, accuracy, sensitivity, and specificity."""

    if target.ndim == 3:
        target = target.unsqueeze(1)
    if probability.shape != target.shape:
        raise ValueError(
            f"Metric shape mismatch: {probability.shape} vs {target.shape}"
        )
    prediction = probability.gt(threshold).float()
    target = target.float()
    dimensions = (1, 2, 3)
    true_positive = (prediction * target).sum(dim=dimensions)
    true_negative = ((1.0 - prediction) * (1.0 - target)).sum(dim=dimensions)
    false_positive = (prediction * (1.0 - target)).sum(dim=dimensions)
    false_negative = ((1.0 - prediction) * target).sum(dim=dimensions)
    epsilon = 1e-7

    prediction_numpy = prediction[:, 0].detach().cpu().numpy().astype(bool)
    target_numpy = target[:, 0].detach().cpu().numpy().astype(bool)
    surface_values = [
        _surface_metrics(prediction_numpy[index], target_numpy[index], spacing)
        for index in range(prediction_numpy.shape[0])
    ]
    hd95_values = np.asarray([value[0] for value in surface_values], dtype=float)
    assd_values = np.asarray([value[1] for value in surface_values], dtype=float)

    return {
        "Dice": float(
            (
                2.0
                * true_positive
                / (2.0 * true_positive + false_positive + false_negative + epsilon)
            ).mean()
        ),
        "mIoU": float(
            (
                true_positive
                / (true_positive + false_positive + false_negative + epsilon)
            ).mean()
        ),
        "HD95": float(np.nanmean(hd95_values))
        if np.isfinite(hd95_values).any()
        else float("nan"),
        "ASSD": float(np.nanmean(assd_values))
        if np.isfinite(assd_values).any()
        else float("nan"),
        "Accuracy": float(
            (
                (true_positive + true_negative)
                / (
                    true_positive
                    + true_negative
                    + false_positive
                    + false_negative
                    + epsilon
                )
            ).mean()
        ),
        "Sensitivity": float(
            (true_positive / (true_positive + false_negative + epsilon)).mean()
        ),
        "Specificity": float(
            (true_negative / (true_negative + false_positive + epsilon)).mean()
        ),
    }


__all__ = ["compute_metrics"]
