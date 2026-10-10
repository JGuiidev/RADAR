"""Construction utilities for the full RADAR model."""

from __future__ import annotations

import json
from copy import deepcopy
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

import torch
from open_clip import create_model, get_tokenizer
from open_clip.factory import _MODEL_CONFIGS
from sam2.build_sam import build_sam2

from model.grda import inject_grda_to_radar_backbones
from model.lora_injection import (
    inject_lora_to_biomedclip,
    inject_lora_to_medsam2,
    patch_biomedclip_visual_last_block,
)
from model.radar import RADAR


def _freeze(module: torch.nn.Module) -> None:
    for parameter in module.parameters():
        parameter.requires_grad = False


def _resolve_biomedclip_paths(repository: str | Path) -> Dict[str, Path]:
    root = Path(repository)
    paths = {
        "config": root / "open_clip_config.json",
        "checkpoint": root / "open_clip_pytorch_model.bin",
        "text_encoder": root / "biomedbert",
    }
    for name, path in paths.items():
        exists = path.is_dir() if name == "text_encoder" else path.is_file()
        if not exists:
            raise FileNotFoundError(f"Missing BiomedCLIP {name}: {path}")
    return paths


def _register_local_biomedclip(
    model_name: str, config_path: Path, text_encoder: Path
) -> None:
    with config_path.open("r", encoding="utf-8") as stream:
        configuration = json.load(stream)
    if "model_cfg" not in configuration or "text_cfg" not in configuration["model_cfg"]:
        raise KeyError(f"Invalid OpenCLIP configuration: {config_path}")
    model_config = deepcopy(configuration["model_cfg"])
    model_config["text_cfg"]["hf_tokenizer_name"] = str(text_encoder)
    model_config["text_cfg"]["hf_model_name"] = str(text_encoder)
    _MODEL_CONFIGS[model_name] = model_config


def _require_injection(name: str, modules: Sequence[Any]) -> None:
    if not modules:
        raise RuntimeError(f"{name} did not match any modules.")


def _medsam2_stage_groups(sam: torch.nn.Module) -> Optional[Dict[str, List[int]]]:
    """Use native Hiera stages; the GRDA injector resolves them when this is None."""

    if not hasattr(sam.image_encoder.trunk, "blocks"):
        raise AttributeError("MedSAM2 Hiera blocks were not found.")
    return None


def full_radar_config() -> Dict[str, Any]:
    """Return only the scientific settings required to reconstruct full RADAR."""

    return {
        "architecture": "RADAR",
        "medsam2_config": "configs/sam2.1_hiera_t512.yaml",
        "biomedclip_model": "BiomedCLIP-PubMedBERT_256-vit_base_patch16_224",
        "output_size": [224, 224],
        "clip_indices": [11],
        "dense_modification": True,
        "lora_rank": 8,
        "lora_alpha": 1.0,
        "grda_medsam2_image": True,
        "grda_biomedclip_visual": True,
        "grda_biomedclip_text": True,
        "grda_text_pooling": "valid_token_mask_weighted_mean",
        "sdra": True,
        "mpg_resolution": [512, 512],
        "mpg_quantile": 0.90,
        "mpg_min_component_area": 16,
        "mpg_max_components": 2,
        "mpg_erosion_kernel": 5,
        "mpg_erosion_iterations": 2,
        "mpg_inner_dilation": 17,
        "mpg_outer_dilation": 27,
        "mpg_max_candidates": 128,
        "mpg_min_point_distance": 12,
        "mpg_positive_points": 2,
        "mpg_negative_points": 2,
    }


def build_radar(
    *,
    medsam2_config: str,
    medsam2_checkpoint: str | Path,
    biomedclip_repository: str | Path,
    device: str | torch.device = "cuda:0",
    torch_dtype: torch.dtype = torch.float32,
    enable_sdra_teacher: bool = True,
) -> Tuple[RADAR, Any]:
    """Build Full RADAR; disable the SDRA teacher for validation and inference."""

    medsam2_checkpoint = Path(medsam2_checkpoint)
    if not medsam2_checkpoint.is_file():
        raise FileNotFoundError(f"Missing MedSAM2 checkpoint: {medsam2_checkpoint}")
    device = torch.device(device)

    medsam2 = build_sam2(medsam2_config, str(medsam2_checkpoint), device=device)
    _freeze(medsam2)

    paths = _resolve_biomedclip_paths(biomedclip_repository)
    model_name = "radar_biomedclip"
    _register_local_biomedclip(model_name, paths["config"], paths["text_encoder"])
    biomedclip = create_model(
        model_name=model_name, pretrained=str(paths["checkpoint"])
    )
    tokenizer = get_tokenizer(model_name)
    _freeze(biomedclip)

    # SDRA compares the adapted student with an untouched, frozen BiomedCLIP.
    sdra_teacher = deepcopy(biomedclip) if enable_sdra_teacher else None
    if sdra_teacher is not None:
        _freeze(sdra_teacher)
        sdra_teacher.eval()

    # Dense Modification is applied to the last BiomedCLIP visual block.
    patch_biomedclip_visual_last_block(
        biomedclip,
        attn_mode="vproj",
        qq_v_from="v",
        drop_attn_residual=False,
        disable_mlp=False,
        drop_mlp_residual=False,
    )

    medsam2_lora = inject_lora_to_medsam2(
        medsam2,
        target=("encoder", "decoder"),
        encoder_layers=None,
        decoder_layers=None,
        encoder_qkv_targets="qkv",
        decoder_qkvo_targets="qkvo",
        r=8,
        alpha=1.0,
    )
    _require_injection("MedSAM2 LoRA", medsam2_lora)
    biomedclip_lora = inject_lora_to_biomedclip(
        biomedclip,
        inject_vision=True,
        inject_text=True,
        vision_layers=[8, 9, 10, 11],
        text_layers=[8, 9, 10, 11],
        vision_qkv_targets="qkv",
        text_qkv_targets="qkv",
        r=8,
        alpha=1.0,
    )
    _require_injection("BiomedCLIP LoRA", biomedclip_lora)

    grda_state = inject_grda_to_radar_backbones(
        sam=medsam2,
        clip=biomedclip,
        use_medsam2=True,
        use_biomedclip_visual=True,
        use_biomedclip_text=True,
        medsam2_groups=_medsam2_stage_groups(medsam2),
        visual_group_size=1,
        text_group_size=1,
        temperature=1.0,
        text_temperature=0.07,
        text_pool_mode="mean",
        residual_init=0.0,
        guidance_scale_init=0.0,
        visual_guidance_mode="base+guidance",
        sam_enable_transition_align=True,
        print_summary=False,
    )

    model = RADAR(
        medsam2=medsam2,
        biomedclip=biomedclip,
        sdra_teacher=sdra_teacher,
        torch_dtype=torch_dtype,
        output_size=(224, 224),
        clip_indices=(11,),
        use_grda_clip_visual=True,
    )
    model._grda_state = grda_state
    return model.to(device), tokenizer


__all__ = ["build_radar", "full_radar_config"]
