"""Trainable-only RADAR checkpoint save and load utilities."""

from __future__ import annotations

from pathlib import Path, PureWindowsPath
from typing import Any, Dict, Mapping, Optional

import torch
from torch import nn


def unwrap_model(model: nn.Module) -> nn.Module:
    return model.module if hasattr(model, "module") else model


def trainable_parameter_names(model: nn.Module) -> set[str]:
    model = unwrap_model(model)
    return {
        name for name, parameter in model.named_parameters() if parameter.requires_grad
    }


def trainable_state_dict(model: nn.Module) -> Dict[str, torch.Tensor]:
    """Return all and only trainable parameters; frozen buffers are excluded."""

    model = unwrap_model(model)
    state = {
        name: parameter.detach().cpu()
        for name, parameter in model.named_parameters()
        if parameter.requires_grad
    }
    expected = trainable_parameter_names(model)
    if set(state) != expected:
        raise RuntimeError(
            "Trainable checkpoint keys do not match requires_grad parameters."
        )
    return state


def _validate_model_config(model_config: Mapping[str, Any]) -> Dict[str, Any]:
    forbidden_keys = {
        "data_root",
        "output_dir",
        "output_path",
        "checkpoint_dir",
        "checkpoint_path",
        "username",
        "environment",
        "device",
    }
    for key in model_config:
        if str(key).lower() in forbidden_keys:
            raise ValueError(
                f"Private or runtime field is not allowed in model_config: {key}"
            )

    def _check_value(value: Any) -> None:
        if isinstance(value, Mapping):
            for nested in value.values():
                _check_value(nested)
        elif isinstance(value, (list, tuple)):
            for nested in value:
                _check_value(nested)
        elif isinstance(value, str) and (
            Path(value).is_absolute() or PureWindowsPath(value).is_absolute()
        ):
            raise ValueError(
                "Absolute filesystem paths are not allowed in model_config."
            )

    _check_value(model_config)
    return dict(model_config)


def save_best_checkpoint(
    path: str | Path,
    model: nn.Module,
    model_config: Mapping[str, Any],
) -> None:
    """Save the deployment checkpoint with exactly two public top-level keys."""

    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    state = trainable_state_dict(model)
    payload = {
        "trainable_state_dict": state,
        "model_config": _validate_model_config(model_config),
    }
    if set(payload) != {"trainable_state_dict", "model_config"}:
        raise RuntimeError("Best checkpoint schema is invalid.")
    torch.save(payload, path)


def save_last_checkpoint(
    path: str | Path,
    model: nn.Module,
    optimizer: torch.optim.Optimizer,
    scheduler: Any,
    scaler: Optional[Any],
    *,
    epoch: int,
    best_metric: float,
    training_state: Mapping[str, Any],
) -> None:
    """Save resume-only optimizer and training state separately from the best model."""

    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "trainable_state_dict": trainable_state_dict(model),
            "optimizer": optimizer.state_dict(),
            "scheduler": scheduler.state_dict(),
            "scaler": None if scaler is None else scaler.state_dict(),
            "epoch": int(epoch),
            "best_metric": float(best_metric),
            "training_state": dict(training_state),
        },
        path,
    )


def load_trainable_parameters(
    model: nn.Module,
    state: Mapping[str, torch.Tensor],
) -> None:
    model = unwrap_model(model)
    expected = trainable_parameter_names(model)
    received = set(state)
    if received != expected:
        missing = sorted(expected - received)
        unexpected = sorted(received - expected)
        raise RuntimeError(
            f"Trainable checkpoint mismatch; missing={missing}, unexpected={unexpected}"
        )
    parameters = dict(model.named_parameters())
    with torch.no_grad():
        for name, value in state.items():
            parameters[name].copy_(
                value.to(device=parameters[name].device, dtype=parameters[name].dtype)
            )


def load_best_checkpoint(path: str | Path, model: nn.Module) -> Dict[str, Any]:
    payload = torch.load(Path(path), map_location="cpu", weights_only=False)
    if set(payload) != {"trainable_state_dict", "model_config"}:
        raise RuntimeError("Expected a best_checkpoint.pth deployment checkpoint.")
    load_trainable_parameters(model, payload["trainable_state_dict"])
    return dict(payload["model_config"])


def resume_training(
    path: str | Path,
    model: nn.Module,
    optimizer: torch.optim.Optimizer,
    scheduler: Any,
    scaler: Optional[Any],
) -> Dict[str, Any]:
    payload = torch.load(Path(path), map_location="cpu", weights_only=False)
    required = {
        "trainable_state_dict",
        "optimizer",
        "scheduler",
        "scaler",
        "epoch",
        "best_metric",
        "training_state",
    }
    if not required.issubset(payload):
        raise RuntimeError("Expected a last_checkpoint.pth resume checkpoint.")
    load_trainable_parameters(model, payload["trainable_state_dict"])
    optimizer.load_state_dict(payload["optimizer"])
    scheduler.load_state_dict(payload["scheduler"])
    if scaler is not None and payload["scaler"] is not None:
        scaler.load_state_dict(payload["scaler"])
    return {
        "epoch": int(payload["epoch"]),
        "best_metric": float(payload["best_metric"]),
        "training_state": dict(payload["training_state"]),
    }


__all__ = [
    "load_best_checkpoint",
    "load_trainable_parameters",
    "resume_training",
    "save_best_checkpoint",
    "save_last_checkpoint",
    "trainable_parameter_names",
    "trainable_state_dict",
]
