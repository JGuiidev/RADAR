"""Segmentation losses used to train RADAR."""

from __future__ import annotations

import torch
import torch.nn.functional as F
from torch import nn


class WeightedBCELoss(nn.Module):
    def __init__(self, weights=(0.5, 0.5), smooth: float = 1e-12) -> None:
        super().__init__()
        self.positive_weight, self.negative_weight = map(float, weights)
        self.smooth = float(smooth)

    def forward(self, probability: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        probability = probability.reshape(-1)
        target = target.reshape(-1)
        if probability.shape != target.shape:
            raise ValueError("Prediction and target shapes do not match.")
        loss = F.binary_cross_entropy(probability, target, reduction="none")
        positive = target.gt(0.5).to(loss.dtype)
        negative = 1.0 - positive
        positive_loss = (loss * positive).sum() / positive.sum().clamp_min(self.smooth)
        negative_loss = (loss * negative).sum() / negative.sum().clamp_min(self.smooth)
        return (
            self.positive_weight * positive_loss + self.negative_weight * negative_loss
        )


class WeightedDiceLoss(nn.Module):
    def __init__(self, weights=(0.5, 0.5), smooth: float = 1e-5) -> None:
        super().__init__()
        self.foreground_weight, self.background_weight = map(float, weights)
        self.smooth = float(smooth)

    def forward(self, probability: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        batch_size = probability.shape[0]
        probability = probability.reshape(batch_size, -1)
        target = target.reshape(batch_size, -1)
        weight = (
            target * (self.foreground_weight - self.background_weight)
            + self.background_weight
        )
        prediction = weight * probability
        reference = weight * target
        intersection = (prediction * reference).sum(dim=-1)
        union = prediction.square().sum(dim=-1) + reference.square().sum(dim=-1)
        return (
            1.0 - (2.0 * intersection + self.smooth) / (union + self.smooth + 1e-7)
        ).mean()


class WeightedBCEDiceLoss(nn.Module):
    def __init__(self, weights=(0.5, 0.5)) -> None:
        super().__init__()
        self.bce = WeightedBCELoss(weights)
        self.dice = WeightedDiceLoss(weights)

    def forward(self, probability: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        return 0.5 * self.bce(probability, target) + 0.5 * self.dice(
            probability, target
        )


__all__ = ["WeightedBCEDiceLoss"]
