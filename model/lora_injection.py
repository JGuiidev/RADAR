"""
LoRA injection utilities for MedSAM2 and BiomedCLIP.

Supports fused QKV projections, split attention projections, layer selection,
and trainable-parameter collection.
"""

from __future__ import annotations

import types
from typing import Iterable, List, Optional, Sequence, Tuple, Union

import torch
from torch import nn

# Core LoRA wrapper


class LoRALinear(nn.Module):
    """LoRA wrapper for a frozen linear layer with optional output selection."""

    def __init__(
        self,
        base: nn.Linear,
        r: int = 8,
        alpha: float = 1.0,
        dropout: float = 0.0,
        out_indices: Optional[
            torch.Tensor
        ] = None,  # 1D LongTensor indices into [0, out_features)
    ) -> None:
        super().__init__()
        if not isinstance(base, nn.Linear):
            raise TypeError("LoRALinear only supports nn.Linear as base.")
        if r <= 0:
            raise ValueError("LoRA rank r must be > 0.")

        self.base = base
        self.in_features = base.in_features
        self.out_features = base.out_features

        # The wrapped projection remains frozen; only the low-rank update is trained.
        for p in self.base.parameters():
            p.requires_grad = False

        self.r = int(r)
        self.alpha = float(alpha)
        self.scaling = self.alpha / float(self.r)

        self.lora_A = nn.Linear(self.in_features, self.r, bias=False)

        if out_indices is not None:
            if not isinstance(out_indices, torch.Tensor):
                raise TypeError("out_indices must be a torch.Tensor if provided.")
            if out_indices.ndim != 1:
                raise ValueError(
                    f"out_indices must be 1D, got shape {tuple(out_indices.shape)}"
                )
            if out_indices.dtype != torch.long:
                out_indices = out_indices.to(dtype=torch.long)

            if out_indices.numel() <= 0:
                raise ValueError("out_indices must contain at least one index.")
            if int(out_indices.min().item()) < 0 or int(
                out_indices.max().item()
            ) >= int(self.out_features):
                raise ValueError(
                    f"out_indices must be within [0, {self.out_features}), got "
                    f"min={int(out_indices.min().item())}, max={int(out_indices.max().item())}"
                )

            # Follow device moves without duplicating fixed indices in checkpoints.
            self.register_buffer("out_indices", out_indices, persistent=False)
            selected_out = int(out_indices.numel())
            self.lora_B = nn.Linear(self.r, selected_out, bias=False)
            self._use_selective_out = True
        else:
            self.out_indices = None  # type: ignore[assignment]
            self.lora_B = nn.Linear(self.r, self.out_features, bias=False)
            self._use_selective_out = False

        self.dropout = (
            nn.Dropout(dropout) if dropout and dropout > 0.0 else nn.Identity()
        )
        self.reset_parameters()

    def reset_parameters(self) -> None:
        # Zero-initializing B preserves the wrapped layer's initial output.
        nn.init.kaiming_uniform_(self.lora_A.weight, a=5**0.5)
        nn.init.zeros_(self.lora_B.weight)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        base_out = self.base(x)
        lora_part = self.lora_B(self.dropout(self.lora_A(x))) * self.scaling

        if not self._use_selective_out:
            return base_out + lora_part

        # Expand (..., selected_out) back into (..., out_features).
        delta = torch.zeros_like(base_out)
        delta[..., self.out_indices] = lora_part
        return base_out + delta


# Helpers


def _set_module_by_name(parent: nn.Module, name: str, new_module: nn.Module) -> None:
    """Set attribute `name` of `parent` to `new_module`."""
    setattr(parent, name, new_module)


def _maybe_getattr(obj: object, path: str) -> Optional[object]:
    """Resolve a dotted attribute path, returning None when it is absent."""
    cur: object = obj
    for p in path.split("."):
        if not hasattr(cur, p):
            return None
        cur = getattr(cur, p)
    return cur


def _normalize_targets(
    targets: Union[str, Sequence[str]], *, allow_o: bool = False
) -> Tuple[bool, bool, bool, bool]:
    """Normalize compact or sequence-form Q/K/V/O target selections."""
    if isinstance(targets, str):
        s = targets.lower().replace(",", "").replace(" ", "")
        tset = set(s)
    else:
        tset = set([str(t).lower() for t in targets])

    use_q = "q" in tset
    use_k = "k" in tset
    use_v = "v" in tset
    use_o = ("o" in tset) if allow_o else False

    if not (use_q or use_k or use_v or use_o):
        raise ValueError(
            "targets must include at least one of: q/k/v" + ("/o" if allow_o else "")
        )

    if not allow_o and "o" in tset:
        raise ValueError(
            "targets includes 'o' but allow_o=False for this injection site."
        )

    return use_q, use_k, use_v, use_o


def _choose_indices(n: int, chosen: Optional[Sequence[int]]) -> List[int]:
    """Return valid indices in [0, n). If chosen is None, return all."""
    if n <= 0:
        return []
    if chosen is None:
        return list(range(n))
    out: List[int] = []
    for i in chosen:
        ii = int(i)
        if 0 <= ii < n:
            out.append(ii)
    return out


def _inject_lora_into_named_linears(
    parent: nn.Module,
    names: Sequence[str],
    *,
    r: int,
    alpha: float,
    dropout: float,
) -> List[LoRALinear]:
    """Wrap named linear projections once and return the new LoRA modules."""
    loras: List[LoRALinear] = []
    for name in names:
        if not hasattr(parent, name):
            continue
        sub = getattr(parent, name)
        if isinstance(sub, LoRALinear):
            continue
        if isinstance(sub, nn.Linear):
            wrapped = LoRALinear(sub, r=r, alpha=alpha, dropout=dropout)
            _set_module_by_name(parent, name, wrapped)
            loras.append(wrapped)
    return loras


def collect_lora_parameters(module: nn.Module) -> Iterable[nn.Parameter]:
    """Collect all trainable parameters of LoRA modules inside `module`."""
    for m in module.modules():
        if isinstance(m, LoRALinear):
            for p in m.parameters():
                if p.requires_grad:
                    yield p


def mark_only_lora_trainable(module: nn.Module) -> None:
    """Freeze a module except for parameters owned by LoRA wrappers."""
    for p in module.parameters():
        p.requires_grad = False
    for p in collect_lora_parameters(module):
        p.requires_grad = True


# Attention-specific injection helpers


def _make_qkv_out_indices(
    out_features: int, *, use_q: bool, use_k: bool, use_v: bool
) -> torch.Tensor:
    """Build indices for a fused output ordered as concatenated Q, K, and V."""
    if out_features % 3 != 0:
        raise ValueError(
            f"Expected fused qkv out_features divisible by 3, got out_features={out_features}"
        )

    dim = out_features // 3
    idx_list: List[torch.Tensor] = []
    if use_q:
        idx_list.append(torch.arange(0, dim, dtype=torch.long))
    if use_k:
        idx_list.append(torch.arange(dim, 2 * dim, dtype=torch.long))
    if use_v:
        idx_list.append(torch.arange(2 * dim, 3 * dim, dtype=torch.long))

    if len(idx_list) == 0:
        raise ValueError("At least one of use_q/use_k/use_v must be True.")

    return torch.cat(idx_list, dim=0)


def _inject_lora_into_timm_fused_qkv_param_saving(
    attn_module: nn.Module,
    *,
    r: int,
    alpha: float,
    dropout: float,
    qkv_targets: Union[str, Sequence[str]] = ("q", "k", "v"),
) -> List[LoRALinear]:
    """Inject a selective LoRA update into a timm fused QKV projection."""
    if not hasattr(attn_module, "qkv"):
        return []
    qkv = getattr(attn_module, "qkv")
    if isinstance(qkv, LoRALinear):
        return []
    if not isinstance(qkv, nn.Linear):
        return []

    use_q, use_k, use_v, _ = _normalize_targets(qkv_targets, allow_o=False)
    out_idx = _make_qkv_out_indices(
        qkv.out_features, use_q=use_q, use_k=use_k, use_v=use_v
    )

    wrapped = LoRALinear(qkv, r=r, alpha=alpha, dropout=dropout, out_indices=out_idx)
    attn_module.qkv = wrapped
    return [wrapped]


def _inject_lora_into_split_qkvo_proj(
    attn_module: nn.Module,
    *,
    r: int,
    alpha: float,
    dropout: float,
    qkvo_targets: Union[str, Sequence[str]] = ("q", "k", "v", "o"),
) -> List[LoRALinear]:
    """Inject LoRA into selected q_proj, k_proj, v_proj, or out_proj layers."""
    use_q, use_k, use_v, use_o = _normalize_targets(qkvo_targets, allow_o=True)

    names: List[str] = []
    if use_q:
        names.append("q_proj")
    if use_k:
        names.append("k_proj")
    if use_v:
        names.append("v_proj")
    if use_o:
        names.append("out_proj")

    return _inject_lora_into_named_linears(
        attn_module, names, r=r, alpha=alpha, dropout=dropout
    )


def _inject_lora_into_hf_qkv(
    attn_self: nn.Module,
    *,
    r: int,
    alpha: float,
    dropout: float,
    qkv_targets: Union[str, Sequence[str]] = ("q", "k", "v"),
) -> List[LoRALinear]:
    """Inject LoRA into selected HuggingFace query, key, and value layers."""
    use_q, use_k, use_v, _ = _normalize_targets(qkv_targets, allow_o=False)

    names: List[str] = []
    if use_q:
        names.append("query")
    if use_k:
        names.append("key")
    if use_v:
        names.append("value")

    return _inject_lora_into_named_linears(
        attn_self, names, r=r, alpha=alpha, dropout=dropout
    )


# MedSAM2 injection


def inject_lora_to_medsam2(
    sam_model: nn.Module,
    *,
    r: int = 8,
    alpha: float = 1.0,
    dropout: float = 0.0,
    target: Sequence[str] = ("encoder", "decoder"),
    encoder_layers: Optional[Sequence[int]] = None,
    decoder_layers: Optional[Sequence[int]] = None,
    memory_layers: Optional[Sequence[int]] = None,
    encoder_qkv_targets: Union[str, Sequence[str]] = ("q", "k", "v"),
    decoder_qkvo_targets: Union[str, Sequence[str]] = ("q", "k", "v", "o"),
    memory_qkvo_targets: Union[str, Sequence[str]] = ("q", "k", "v", "o"),
) -> List[nn.Module]:
    """Inject LoRA into selected MedSAM2 encoder, decoder, or memory layers.

    A None layer selection covers every available layer. Encoder attention uses
    fused QKV, while decoder and memory attention use split Q/K/V/O projections.
    """
    want = set(target)
    if "all" in want:
        want = {"encoder", "decoder", "memory"}

    loras: List[nn.Module] = []

    # The image encoder exposes a fused QKV projection per trunk block.
    if "encoder" in want:
        img_enc = getattr(sam_model, "image_encoder", None)
        if img_enc is None:
            raise ValueError("SAM model does not have `image_encoder`.")

        trunk = getattr(img_enc, "trunk", None)
        if trunk is None or not hasattr(trunk, "blocks"):
            raise ValueError("sam_model.image_encoder.trunk does not have `blocks`.")

        blocks = getattr(trunk, "blocks")
        chosen = _choose_indices(len(blocks), encoder_layers)

        for i in chosen:
            blk = blocks[i]
            attn = getattr(blk, "attn", None)
            if attn is None:
                continue
            loras += _inject_lora_into_timm_fused_qkv_param_saving(
                attn,
                r=r,
                alpha=alpha,
                dropout=dropout,
                qkv_targets=encoder_qkv_targets,
            )

    # Two-way decoder attention exposes separate Q/K/V/O projections.
    if "decoder" in want:
        dec = getattr(sam_model, "sam_mask_decoder", None)
        if dec is None:
            raise ValueError("SAM model does not have `sam_mask_decoder`.")

        twt = getattr(dec, "transformer", None)
        if twt is None or not hasattr(twt, "layers"):
            raise ValueError(
                "sam_model.sam_mask_decoder.transformer does not have `layers`."
            )

        layers = getattr(twt, "layers")
        chosen = _choose_indices(len(layers), decoder_layers)

        for i in chosen:
            layer = layers[i]
            loras += _inject_lora_into_split_qkvo_proj(
                layer.self_attn,
                r=r,
                alpha=alpha,
                dropout=dropout,
                qkvo_targets=decoder_qkvo_targets,
            )
            loras += _inject_lora_into_split_qkvo_proj(
                layer.cross_attn_token_to_image,
                r=r,
                alpha=alpha,
                dropout=dropout,
                qkvo_targets=decoder_qkvo_targets,
            )
            loras += _inject_lora_into_split_qkvo_proj(
                layer.cross_attn_image_to_token,
                r=r,
                alpha=alpha,
                dropout=dropout,
                qkvo_targets=decoder_qkvo_targets,
            )

        if hasattr(twt, "final_attn_token_to_image"):
            loras += _inject_lora_into_split_qkvo_proj(
                twt.final_attn_token_to_image,
                r=r,
                alpha=alpha,
                dropout=dropout,
                qkvo_targets=decoder_qkvo_targets,
            )

    # Memory attention is optional in MedSAM2 configurations.
    if "memory" in want:
        mem = getattr(sam_model, "memory_attention", None)
        if mem is None:
            return loras
        if not hasattr(mem, "layers"):
            return loras

        layers = getattr(mem, "layers")
        chosen = _choose_indices(len(layers), memory_layers)

        for i in chosen:
            layer = layers[i]
            if hasattr(layer, "self_attn"):
                loras += _inject_lora_into_split_qkvo_proj(
                    layer.self_attn,
                    r=r,
                    alpha=alpha,
                    dropout=dropout,
                    qkvo_targets=memory_qkvo_targets,
                )
            if hasattr(layer, "cross_attn_image"):
                loras += _inject_lora_into_split_qkvo_proj(
                    layer.cross_attn_image,
                    r=r,
                    alpha=alpha,
                    dropout=dropout,
                    qkvo_targets=memory_qkvo_targets,
                )

    return loras


# BiomedCLIP injection


def inject_lora_to_biomedclip(
    clip_model: nn.Module,
    *,
    r: int = 8,
    alpha: float = 1.0,
    dropout: float = 0.0,
    inject_vision: bool = False,
    inject_text: bool = False,
    vision_layers: Optional[Sequence[int]] = None,
    text_layers: Optional[Sequence[int]] = None,
    vision_qkv_targets: Union[str, Sequence[str]] = ("q", "k", "v"),
    text_qkv_targets: Union[str, Sequence[str]] = ("q", "k", "v"),
) -> List[nn.Module]:
    """Inject LoRA into selected BiomedCLIP vision and text layers.

    Vision attention uses fused QKV projections; BERT text attention exposes
    separate query, key, and value projections. None selects every layer.
    """
    loras: List[nn.Module] = []

    # Vision transformer blocks use timm-style fused QKV projections.
    if inject_vision:
        visual = getattr(clip_model, "visual", None)
        trunk = getattr(visual, "trunk", None) if visual is not None else None
        blocks = getattr(trunk, "blocks", None) if trunk is not None else None

        if blocks is None:
            raise ValueError(
                "Cannot find BiomedCLIP vision blocks at clip_model.visual.trunk.blocks"
            )

        chosen = _choose_indices(len(blocks), vision_layers)

        for i in chosen:
            blk = blocks[i]
            attn = getattr(blk, "attn", None)
            if attn is None:
                continue
            loras += _inject_lora_into_timm_fused_qkv_param_saving(
                attn, r=r, alpha=alpha, dropout=dropout, qkv_targets=vision_qkv_targets
            )

    # Text transformer layers use HuggingFace-style split Q/K/V projections.
    if inject_text:
        layers_obj = _maybe_getattr(clip_model, "text.transformer.encoder.layer")
        if layers_obj is None:
            raise ValueError(
                "Cannot find BiomedCLIP text layers at clip_model.text.transformer.encoder.layer"
            )

        if not isinstance(layers_obj, (list, nn.ModuleList)):
            raise TypeError(
                "Expected clip_model.text.transformer.encoder.layer to be a list/ModuleList of layers."
            )

        chosen = _choose_indices(len(layers_obj), text_layers)

        for i in chosen:
            lyr = layers_obj[i]
            attn_self = _maybe_getattr(lyr, "attention.self")
            if attn_self is None:
                continue
            loras += _inject_lora_into_hf_qkv(
                attn_self,
                r=r,
                alpha=alpha,
                dropout=dropout,
                qkv_targets=text_qkv_targets,
            )

    return loras


# Dense Modification for the final BiomedCLIP vision block


def _timm_attn_forward_qq(
    attn: nn.Module, x: torch.Tensor, *, v_from: str = "q"
) -> torch.Tensor:
    """Compute Q-Q attention for an input shaped [batch, tokens, channels].

    Q supplies both queries and keys; values come from Q or the original V.
    """
    if x.dim() != 3:
        raise ValueError(f"Expected x to be (B, N, C), got shape {tuple(x.shape)}")

    B, N, C = x.shape

    if not hasattr(attn, "qkv"):
        raise ValueError("timm Attention module does not have attribute `qkv`.")

    num_heads = getattr(attn, "num_heads", None)
    if num_heads is None:
        raise ValueError(
            "Cannot find `attn.num_heads`. Please check your timm version / Attention implementation."
        )

    if C % int(num_heads) != 0:
        raise ValueError(
            f"Embedding dim C={C} is not divisible by num_heads={num_heads}."
        )

    head_dim = C // int(num_heads)
    scale = getattr(attn, "scale", None)
    if scale is None:
        scale = head_dim**-0.5

    # [B, N, 3C] -> [3, B, H, N, D]
    qkv = attn.qkv(x)
    qkv = qkv.reshape(B, N, 3, int(num_heads), head_dim).permute(2, 0, 3, 1, 4)
    q = qkv[0]  # (B, H, N, D)

    k = q

    # Some timm attention variants normalize queries and keys per head.
    if hasattr(attn, "q_norm"):
        q = attn.q_norm(q)
    if hasattr(attn, "k_norm"):
        k = attn.k_norm(k)

    if v_from not in ("q", "v"):
        raise ValueError("v_from must be 'q' or 'v'.")
    v = q if v_from == "q" else qkv[2]

    attn_weights = (q * scale) @ k.transpose(-2, -1)
    attn_weights = attn_weights.softmax(dim=-1)

    if hasattr(attn, "attn_drop") and isinstance(attn.attn_drop, nn.Module):
        attn_weights = attn.attn_drop(attn_weights)

    out = attn_weights @ v  # (B, H, N, D)
    out = out.transpose(1, 2).reshape(B, N, C)  # (B, N, C)

    if hasattr(attn, "proj"):
        out = attn.proj(out)
    if hasattr(attn, "proj_drop") and isinstance(attn.proj_drop, nn.Module):
        out = attn.proj_drop(out)

    return out


def _timm_attn_forward_vproj(attn: nn.Module, x: torch.Tensor) -> torch.Tensor:
    """Project V tokens without attention mixing for Dense Modification.

    This helper returns only the projected branch output. The patched block
    applies the residual connection and feed-forward network.
    """
    if x.dim() != 3:
        raise ValueError(f"Expected x to be (B, N, C), got shape {tuple(x.shape)}")

    if not hasattr(attn, "qkv"):
        raise ValueError("timm Attention module does not have attribute `qkv`.")
    if not hasattr(attn, "proj"):
        raise ValueError("timm Attention module does not have attribute `proj`.")

    B, N, C = x.shape

    num_heads = getattr(attn, "num_heads", None)
    if num_heads is None:
        raise ValueError(
            "Cannot find `attn.num_heads`. Please check your timm version / Attention implementation."
        )

    if C % int(num_heads) != 0:
        raise ValueError(
            f"Embedding dim C={C} is not divisible by num_heads={num_heads}."
        )

    head_dim = C // int(num_heads)

    # [B, N, 3C] -> [3, B, H, N, D]
    qkv = attn.qkv(x)
    qkv = qkv.reshape(B, N, 3, int(num_heads), head_dim).permute(2, 0, 3, 1, 4)

    v = qkv[2]  # (B, H, N, D)

    # Merge heads: [B, H, N, D] -> [B, N, C].
    out = v.transpose(1, 2).reshape(B, N, C)

    out = attn.proj(out)

    if hasattr(attn, "proj_drop") and isinstance(attn.proj_drop, nn.Module):
        out = attn.proj_drop(out)

    return out


def patch_biomedclip_visual_last_block(
    clip_model: nn.Module,
    *,
    attn_mode: str = "qq",  # "orig" | "qq" | "none" | "vproj"
    qq_v_from: str = "q",  # "q" | "v"  (only used when attn_mode="qq")
    drop_attn_residual: bool = False,  # remove skip for attn branch
    disable_mlp: bool = False,  # remove FFN/MLP branch
    drop_mlp_residual: bool = False,  # remove skip for mlp branch
) -> None:
    """Patch the final BiomedCLIP vision block for Dense Modification.

    Modes select original attention, Q-Q attention, no attention branch, or
    direct V projection. Residual flags replace addition with branch output;
    disabling the MLP leaves that branch inactive.
    """
    visual = getattr(clip_model, "visual", None)
    trunk = getattr(visual, "trunk", None) if visual is not None else None
    blocks = getattr(trunk, "blocks", None) if trunk is not None else None
    if blocks is None or len(blocks) == 0:
        raise ValueError(
            "Cannot find BiomedCLIP vision blocks at clip_model.visual.trunk.blocks"
        )

    blk = blocks[-1]

    # Preserve the original method so the patch remains reversible.
    if not hasattr(blk, "_orig_forward"):
        blk._orig_forward = blk.forward  # type: ignore[attr-defined]

    # Identity fallbacks support timm blocks without optional wrappers.
    def _get_mod(name: str) -> nn.Module:
        m = getattr(blk, name, None)
        return m if isinstance(m, nn.Module) else nn.Identity()

    norm1 = _get_mod("norm1")
    norm2 = _get_mod("norm2")
    ls1 = _get_mod("ls1")
    ls2 = _get_mod("ls2")
    drop_path1 = _get_mod("drop_path1")
    drop_path2 = _get_mod("drop_path2")
    attn = getattr(blk, "attn", None)
    mlp = getattr(blk, "mlp", None)

    if attn is None:
        raise ValueError("Last vision block does not have attribute `attn`.")
    if (not disable_mlp) and (mlp is None):
        raise ValueError(
            "Last vision block does not have attribute `mlp` but disable_mlp=False."
        )

    if attn_mode not in ("orig", "qq", "none", "vproj"):
        raise ValueError("attn_mode must be one of: 'orig', 'qq', 'none', 'vproj'.")

    def _forward_patched(self: nn.Module, x: torch.Tensor) -> torch.Tensor:
        # Attention branch
        x1 = norm1(x)

        if attn_mode == "none":
            attn_out = torch.zeros_like(x)
        elif attn_mode == "orig":
            attn_out = attn(x1)  # type: ignore[misc]
        elif attn_mode == "qq":
            attn_out = _timm_attn_forward_qq(attn, x1, v_from=qq_v_from)  # type: ignore[arg-type]
        elif attn_mode == "vproj":
            attn_out = _timm_attn_forward_vproj(attn, x1)  # type: ignore[arg-type]
        else:
            raise RuntimeError(f"Unexpected attn_mode: {attn_mode}")

        attn_out = drop_path1(ls1(attn_out))

        if drop_attn_residual:
            # If attention branch is removed, keep x unchanged to avoid collapse.
            x = x if attn_mode == "none" else attn_out
        else:
            # Preserve the standard residual path for all active modes.
            x = x + attn_out

        # Feed-forward branch
        if disable_mlp:
            mlp_out = torch.zeros_like(x)
        else:
            mlp_out = mlp(norm2(x))  # type: ignore[misc]

        mlp_out = drop_path2(ls2(mlp_out))

        if drop_mlp_residual:
            # If mlp is disabled, keep x unchanged to avoid collapse.
            x = x if disable_mlp else mlp_out
        else:
            x = x + mlp_out

        return x

    blk.forward = types.MethodType(_forward_patched, blk)

    # Expose applied options for inspection and reproducible restoration.
    blk._patched_cfg = {  # type: ignore[attr-defined]
        "attn_mode": attn_mode,
        "qq_v_from": qq_v_from,
        "drop_attn_residual": drop_attn_residual,
        "disable_mlp": disable_mlp,
        "drop_mlp_residual": drop_mlp_residual,
    }


def restore_biomedclip_visual_last_block(clip_model: nn.Module) -> None:
    """Restore the original forward of the last vision block if it was patched."""
    visual = getattr(clip_model, "visual", None)
    trunk = getattr(visual, "trunk", None) if visual is not None else None
    blocks = getattr(trunk, "blocks", None) if trunk is not None else None
    if blocks is None or len(blocks) == 0:
        raise ValueError(
            "Cannot find BiomedCLIP vision blocks at clip_model.visual.trunk.blocks"
        )

    blk = blocks[-1]
    if hasattr(blk, "_orig_forward"):
        blk.forward = blk._orig_forward  # type: ignore[attr-defined]


# Inspection helpers


def count_trainable_parameters(module: nn.Module) -> int:
    """Count trainable parameters."""
    return sum(int(p.numel()) for p in module.parameters() if p.requires_grad)


def count_lora_modules(module: nn.Module) -> int:
    """Count LoRALinear modules."""
    return sum(1 for m in module.modules() if isinstance(m, LoRALinear))
