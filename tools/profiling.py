"""Parameter and operation profiling for full RADAR inference."""

from __future__ import annotations

from collections import OrderedDict
from typing import Dict, Tuple

import torch
from torch import nn


def parameter_profile(model: nn.Module) -> Dict[str, object]:
    total = sum(parameter.numel() for parameter in model.parameters())
    trainable = sum(
        parameter.numel() for parameter in model.parameters() if parameter.requires_grad
    )
    breakdown = OrderedDict(
        (name, 0)
        for name in (
            "LoRA",
            "GRDA",
            "SDRA",
            "MPG",
            "Text Prompt",
            "Other trainable modules",
        )
    )
    for name, parameter in model.named_parameters():
        if not parameter.requires_grad:
            continue
        lowered = name.lower()
        if "lora" in lowered:
            category = "LoRA"
        elif "grda" in lowered:
            category = "GRDA"
        elif "sdra" in lowered:
            category = "SDRA"
        elif "sim_logit_scale" in lowered:
            category = "MPG"
        elif "txt_to_prompt" in lowered:
            category = "Text Prompt"
        else:
            category = "Other trainable modules"
        breakdown[category] += parameter.numel()
    if sum(breakdown.values()) != trainable:
        raise RuntimeError("Trainable parameter breakdown is incomplete.")
    return {
        "total": total,
        "trainable": trainable,
        "frozen": total - trainable,
        "breakdown": dict(breakdown),
    }


class _InferenceWrapper(nn.Module):
    def __init__(self, model: nn.Module) -> None:
        super().__init__()
        self.model = model

    def forward(
        self,
        image_sam: torch.Tensor,
        image_clip: torch.Tensor,
        tokens: torch.Tensor,
    ) -> torch.Tensor:
        return self.model(image_sam, image_clip, tokens)[0]


def operation_profile(
    model: nn.Module,
    inputs: Tuple[torch.Tensor, torch.Tensor, torch.Tensor],
) -> Dict[str, float]:
    """Measure MACs with THOP and report FLOPs under the explicit 2x convention."""

    from thop import profile

    was_training = model.training
    model.eval()
    macs, _ = profile(_InferenceWrapper(model), inputs=inputs, verbose=False)
    model.train(was_training)
    return {"macs": float(macs), "flops_2x_macs": float(2.0 * macs)}


def format_profile(parameters: Dict[str, object], operations: Dict[str, float]) -> str:
    lines = [
        f"Total parameters: {parameters['total']:,}",
        f"Trainable parameters: {parameters['trainable']:,}",
        f"Frozen parameters: {parameters['frozen']:,}",
        f"MACs: {operations['macs'] / 1e9:.4f} G",
        f"FLOPs (2 × MACs): {operations['flops_2x_macs'] / 1e9:.4f} G",
        "Trainable parameter breakdown:",
    ]
    lines.extend(
        f"  {name}: {value:,}" for name, value in parameters["breakdown"].items()
    )
    return "\n".join(lines)


__all__ = ["format_profile", "operation_profile", "parameter_profile"]
