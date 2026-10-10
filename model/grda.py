from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import (
    Any,
    Callable,
    Dict,
    Iterable,
    List,
    MutableSequence,
    Optional,
    Sequence,
    Tuple,
)

import torch
import torch.nn as nn
import torch.nn.functional as F

_VALID_GUIDANCE_MODES = {"base", "guidance", "base+guidance", "base*guidance"}
_VALID_TEXT_POOL_MODES = {"mean"}


@dataclass
class _ReplacedModuleRef:
    stack: MutableSequence[nn.Module]
    index: int
    original: nn.Module


@dataclass
class _SharedBankRef:
    owner: nn.Module
    attr_name: str
    bank: nn.Module


@dataclass
class _FeatureBankEntry:
    key: str
    tensor: torch.Tensor


@dataclass
class GRDAInjectionState:
    """Bookkeeping object for injected GRDA wrappers."""

    branch_name: str
    injected_paths: List[str] = field(default_factory=list)
    reset_handles: List[Any] = field(default_factory=list)
    group_map: Dict[str, List[int]] = field(default_factory=dict)
    group_dims: Dict[str, int] = field(default_factory=dict)
    replaced_modules: List[_ReplacedModuleRef] = field(default_factory=list)
    shared_banks: List[_SharedBankRef] = field(default_factory=list)
    branch_states: List["BranchBlockState"] = field(default_factory=list)

    def remove(self) -> None:
        for handle in self.reset_handles:
            handle.remove()
        self.reset_handles.clear()

        for ref in reversed(self.replaced_modules):
            ref.stack[ref.index] = ref.original
        self.replaced_modules.clear()

        for ref in reversed(self.shared_banks):
            if getattr(ref.owner, ref.attr_name, None) is ref.bank:
                delattr(ref.owner, ref.attr_name)
        self.shared_banks.clear()

        for branch_state in self.branch_states:
            branch_state.clear()


class BranchBlockState:
    """Forward-local feature and text-guidance state for one injected branch."""

    def __init__(self, max_blocks: Optional[int] = None) -> None:
        self.max_blocks = max_blocks
        self.completed_blocks: List[_FeatureBankEntry] = []
        self.current_partial: Optional[torch.Tensor] = None
        self.current_group: Optional[str] = None
        self.txt_tok: Optional[torch.Tensor] = None
        self.txt_mask: Optional[torch.Tensor] = None
        self.guidance_logit_bank: Dict[Tuple[int, str], torch.Tensor] = {}
        self._next_feature_id = 0

    def clear_history(self) -> None:
        self.completed_blocks.clear()
        self.current_partial = None
        self.current_group = None
        self.guidance_logit_bank.clear()
        self._next_feature_id = 0

    def clear_guidance(self) -> None:
        self.txt_tok = None
        self.txt_mask = None
        self.guidance_logit_bank.clear()

    def clear(self) -> None:
        self.clear_history()
        self.clear_guidance()

    def set_text_guidance(self, txt_tok: torch.Tensor, txt_mask: torch.Tensor) -> None:
        if txt_tok.ndim != 3:
            raise ValueError(f"txt_tok must be [B,L,C], got {tuple(txt_tok.shape)}.")
        if txt_mask.ndim != 2:
            raise ValueError(f"txt_mask must be [B,L], got {tuple(txt_mask.shape)}.")
        if tuple(txt_mask.shape) != tuple(txt_tok.shape[:2]):
            raise ValueError(
                f"txt_mask {tuple(txt_mask.shape)} does not match txt_tok {tuple(txt_tok.shape)}."
            )
        self.txt_tok = txt_tok
        self.txt_mask = txt_mask.to(device=txt_tok.device, dtype=torch.bool)
        self.guidance_logit_bank.clear()

    def get_text_guidance(
        self,
    ) -> Tuple[Optional[torch.Tensor], Optional[torch.Tensor]]:
        return self.txt_tok, self.txt_mask

    def begin_group_if_needed(self, x: torch.Tensor, group_name: str) -> torch.Tensor:
        if self.current_group != group_name or self.current_partial is None:
            self.current_group = group_name
            self.current_partial = x
        return self.current_partial

    def update_partial(self, x: torch.Tensor) -> None:
        self.current_partial = x

    def finalize_group(self) -> None:
        if self.current_partial is None:
            self.current_group = None
            return

        key = f"hist_{self._next_feature_id}"
        self._next_feature_id += 1
        self.completed_blocks.append(
            _FeatureBankEntry(key=key, tensor=self.current_partial)
        )
        self._trim_feature_bank()
        self.current_partial = None
        self.current_group = None

    def _trim_feature_bank(self) -> None:
        if (
            self.max_blocks is None
            or self.max_blocks <= 0
            or len(self.completed_blocks) <= self.max_blocks
        ):
            return
        removed = self.completed_blocks[: -self.max_blocks]
        removed_keys = {entry.key for entry in removed}
        self.completed_blocks = self.completed_blocks[-self.max_blocks :]
        for cache_key in list(self.guidance_logit_bank.keys()):
            if cache_key[1] in removed_keys:
                del self.guidance_logit_bank[cache_key]

    def transform_history(self, fn: Callable[[torch.Tensor], torch.Tensor]) -> None:
        self.completed_blocks = [
            _FeatureBankEntry(key=entry.key, tensor=fn(entry.tensor))
            for entry in self.completed_blocks
        ]
        if self.current_partial is not None:
            self.current_partial = fn(self.current_partial)
        self.guidance_logit_bank.clear()


def _normalize_guidance_mode(mode: str) -> str:
    mode = str(mode).lower().strip()
    if mode not in _VALID_GUIDANCE_MODES:
        raise ValueError(
            f"guidance_mode must be one of {sorted(_VALID_GUIDANCE_MODES)}, got {mode}."
        )
    return mode


def _normalize_text_pool_mode(mode: str) -> str:
    mode = str(mode).lower().strip()
    if mode not in _VALID_TEXT_POOL_MODES:
        raise ValueError(
            f"text_pool_mode must be one of {sorted(_VALID_TEXT_POOL_MODES)}, got {mode}."
        )
    return mode


def _get_layernorm_dim(layernorm: nn.LayerNorm, desc: str) -> int:
    if not isinstance(layernorm, nn.LayerNorm):
        raise ValueError(desc)
    shape = layernorm.normalized_shape
    return int(shape[-1] if isinstance(shape, (tuple, list)) else shape)


def _select_tensor_output(output: Any, name: str) -> torch.Tensor:
    if torch.is_tensor(output):
        return output
    if isinstance(output, (tuple, list)) and output and torch.is_tensor(output[0]):
        return output[0]
    raise TypeError(
        f"{name}: expected tensor or tuple/list[tensor] output, got {type(output)}."
    )


def _iter_state_objects(state_obj: Any) -> List[GRDAInjectionState]:
    if isinstance(state_obj, GRDAInjectionState):
        return [state_obj]
    if isinstance(state_obj, dict):
        return [v for v in state_obj.values() if isinstance(v, GRDAInjectionState)]
    return []


def _iter_unique_grda_states(module: nn.Module) -> Iterable[GRDAInjectionState]:
    visited = set()
    for sub in module.modules():
        for attr in GRDA_STATE_ATTRS:
            for st in _iter_state_objects(getattr(sub, attr, None)):
                sid = id(st)
                if sid in visited:
                    continue
                visited.add(sid)
                yield st


def _safe_module_key(prefix: str, raw: Any, used: set[str]) -> str:
    text = str(raw)
    base = "".join(ch if ch.isalnum() or ch == "_" else "_" for ch in text).strip("_")
    base = base or prefix
    if not base[0].isalpha():
        base = f"{prefix}_{base}"

    key, count = base, 1
    while key in used:
        key = f"{base}_{count}"
        count += 1
    used.add(key)
    return key


def _same_tensor_storage(a: torch.Tensor, b: torch.Tensor) -> bool:
    return (
        tuple(a.shape) == tuple(b.shape)
        and a.data_ptr() == b.data_ptr()
        and a.storage_offset() == b.storage_offset()
    )


class GRDAResidual(nn.Module):
    """Aggregate residual histories with valid-token text guidance."""

    def __init__(
        self,
        dim: int,
        *,
        temperature: float = 1.0,
        text_temperature: float = 0.07,
        text_pool_mode: str = "mean",
        norm_eps: float = 1e-6,
        proj_init_std: float = 0.02,
        guidance_mode: str = "base",
        residual_init: float = 0.0,
        guidance_scale_init: float = 0.0,
    ) -> None:
        super().__init__()
        self.dim = int(dim)
        self.temperature = float(temperature)
        self.text_temperature = float(text_temperature)
        self.text_pool_mode = _normalize_text_pool_mode(text_pool_mode)
        self.guidance_mode = _normalize_guidance_mode(guidance_mode)

        self.norm = nn.LayerNorm(self.dim, eps=norm_eps)
        self.proj = nn.Linear(self.dim, 1, bias=False)
        self.residual_scale = nn.Parameter(torch.tensor(float(residual_init)))
        self.guidance_scale = nn.Parameter(torch.tensor(float(guidance_scale_init)))
        nn.init.normal_(self.proj.weight, mean=0.0, std=proj_init_std)

    def _is_compatible(self, x: torch.Tensor) -> bool:
        return x.ndim >= 2 and (x.shape[-1] == self.dim or x.shape[1] == self.dim)

    def _to_tokens(
        self, x: torch.Tensor
    ) -> Tuple[torch.Tensor, Tuple[str, Tuple[int, ...]]]:
        if x.ndim < 2:
            raise ValueError(f"Expected tensor with ndim >= 2, got {tuple(x.shape)}.")
        if x.shape[-1] == self.dim:
            b = x.shape[0]
            return x.reshape(b, -1, self.dim), ("last", tuple(x.shape))
        if x.shape[1] == self.dim:
            tokens = x.reshape(x.shape[0], self.dim, -1).transpose(1, 2).contiguous()
            return tokens, ("channel", (x.shape[0], self.dim, *x.shape[2:]))
        raise ValueError(
            f"Cannot infer feature dimension for shape={tuple(x.shape)}, dim={self.dim}."
        )

    def _from_tokens(
        self, tokens: torch.Tensor, meta: Tuple[str, Tuple[int, ...]]
    ) -> torch.Tensor:
        layout, shape = meta
        if layout == "last":
            return tokens.reshape(*shape)
        if layout == "channel":
            b, c, *spatial = shape
            return tokens.transpose(1, 2).contiguous().reshape(b, c, *spatial)
        raise ValueError(f"Unknown layout meta: {layout}.")

    def _compute_base_logits(self, normed_sources: torch.Tensor) -> torch.Tensor:
        return self.proj(normed_sources).squeeze(-1)

    def _compute_guidance_logits_one(
        self,
        normed_tokens: torch.Tensor,
        txt_tok: torch.Tensor,
        txt_mask: torch.Tensor,
    ) -> torch.Tensor:
        if txt_tok.ndim != 3:
            raise ValueError(f"txt_tok must be [B,L,C], got {tuple(txt_tok.shape)}.")
        if txt_mask.ndim != 2 or tuple(txt_mask.shape) != tuple(txt_tok.shape[:2]):
            raise ValueError(
                f"txt_mask {tuple(txt_mask.shape)} does not match txt_tok {tuple(txt_tok.shape)}."
            )
        if txt_tok.shape[-1] != self.dim:
            raise ValueError(
                f"Text dim {txt_tok.shape[-1]} must equal feature dim {self.dim}; projection is disabled."
            )
        if normed_tokens.shape[0] != txt_tok.shape[0]:
            raise ValueError(
                f"Batch mismatch: feature B={normed_tokens.shape[0]}, text B={txt_tok.shape[0]}."
            )

        v = F.normalize(normed_tokens.float(), dim=-1)
        t = F.normalize(
            txt_tok.to(device=normed_tokens.device, dtype=torch.float32), dim=-1
        )
        mask = txt_mask.to(device=normed_tokens.device, dtype=torch.bool)
        # Cosine similarities are pooled only across valid tokens. Padding
        # positions have exactly zero weight in the mask-weighted mean.
        sim = torch.einsum("bnc,blc->bnl", v, t)
        mask_view = mask[:, None, :]
        valid_batch = mask.any(dim=1)

        masked = sim.masked_fill(~mask_view, 0.0)
        denom = mask.sum(dim=1).clamp_min(1).to(dtype=sim.dtype).view(-1, 1)
        logits = masked.sum(dim=-1) / denom

        logits = torch.where(valid_batch.view(-1, 1), logits, torch.zeros_like(logits))
        return logits.to(dtype=normed_tokens.dtype)

    def _compute_guidance_logits(
        self,
        normed_sources: torch.Tensor,
        source_keys: Sequence[Optional[str]],
        txt_tok: Optional[torch.Tensor],
        txt_mask: Optional[torch.Tensor],
        guidance_logit_bank: Dict[Tuple[int, str], torch.Tensor],
    ) -> Optional[torch.Tensor]:
        if self.guidance_mode == "base":
            return None
        if txt_tok is None or txt_mask is None:
            raise RuntimeError(
                "Text-token guidance is required when guidance_mode is not 'base'."
            )

        logits: List[torch.Tensor] = []
        module_id = id(self)
        for idx, key in enumerate(source_keys):
            if key is None:
                current = self._compute_guidance_logits_one(
                    normed_sources[idx], txt_tok, txt_mask
                )
            else:
                cache_key = (module_id, key)
                current = guidance_logit_bank.get(cache_key)
                if current is None:
                    current = self._compute_guidance_logits_one(
                        normed_sources[idx], txt_tok, txt_mask
                    )
                    guidance_logit_bank[cache_key] = current
                else:
                    current = current.to(
                        device=normed_sources.device, dtype=normed_sources.dtype
                    )
            logits.append(current)
        return torch.stack(logits, dim=0)

    def _fuse_logits(
        self, base_logits: torch.Tensor, guidance_logits: Optional[torch.Tensor]
    ) -> torch.Tensor:
        mode = self.guidance_mode
        if mode == "base":
            return base_logits
        if guidance_logits is None:
            raise RuntimeError(
                "guidance_logits is required when guidance_mode is not 'base'."
            )
        if mode == "guidance":
            return guidance_logits
        scale = self.guidance_scale.to(
            device=base_logits.device, dtype=base_logits.dtype
        )
        if mode == "base+guidance":
            return base_logits + scale * guidance_logits
        if mode == "base*guidance":
            return base_logits * (1.0 + scale * torch.sigmoid(guidance_logits))
        raise ValueError(f"Unsupported guidance_mode: {self.guidance_mode}.")

    def forward(
        self,
        completed_blocks: Sequence[_FeatureBankEntry],
        partial_block: torch.Tensor,
        txt_tok: Optional[torch.Tensor] = None,
        txt_mask: Optional[torch.Tensor] = None,
        guidance_logit_bank: Optional[Dict[Tuple[int, str], torch.Tensor]] = None,
    ) -> torch.Tensor:
        if not self._is_compatible(partial_block):
            return partial_block
        if guidance_logit_bank is None:
            guidance_logit_bank = {}

        partial_tokens, partial_meta = self._to_tokens(partial_block)
        tokenized: List[torch.Tensor] = []
        source_keys: List[Optional[str]] = []

        for entry in completed_blocks:
            source = entry.tensor
            if _same_tensor_storage(source, partial_block) or not self._is_compatible(
                source
            ):
                continue
            source_tokens, source_meta = self._to_tokens(source)
            if source_meta != partial_meta:
                continue
            tokenized.append(source_tokens)
            source_keys.append(entry.key)

        tokenized.append(partial_tokens)
        source_keys.append(None)

        if len(tokenized) == 1:
            return partial_block

        # History is [K, B, N, C]; softmax over dimension 0 aggregates depth.
        stacked = torch.stack(tokenized, dim=0)
        normed = self.norm(stacked)
        base_logits = self._compute_base_logits(normed)
        guidance_logits = self._compute_guidance_logits(
            normed, source_keys, txt_tok, txt_mask, guidance_logit_bank
        )
        logits = self._fuse_logits(base_logits, guidance_logits) / max(
            self.temperature, 1e-6
        )
        alpha = torch.softmax(logits, dim=0).to(dtype=stacked.dtype)
        merged = (stacked * alpha.unsqueeze(-1)).sum(dim=0)
        scale = self.residual_scale.to(device=merged.device, dtype=merged.dtype)
        out_tokens = partial_tokens + scale * (merged - partial_tokens)
        return self._from_tokens(out_tokens, partial_meta)


class GRDASharedBank(nn.Module):
    """Branch-level bank. One GRDA module is shared within each group."""

    def __init__(
        self,
        *,
        branch_name: str,
        group_dims: Dict[str, int],
        guidance_mode: str = "base",
        temperature: float = 1.0,
        text_temperature: float = 0.07,
        text_pool_mode: str = "mean",
        norm_eps: float = 1e-6,
        proj_init_std: float = 0.02,
        residual_init: float = 0.0,
        guidance_scale_init: float = 0.0,
    ) -> None:
        super().__init__()
        self.branch_name = branch_name
        self.guidance_mode = _normalize_guidance_mode(guidance_mode)
        self.group_to_dim = {str(k): int(v) for k, v in group_dims.items()}
        self.group_to_key: Dict[str, str] = {}
        self.group_grda = nn.ModuleDict()

        used_group_keys: set[str] = set()
        for group_name, dim in self.group_to_dim.items():
            key = _safe_module_key("group", group_name, used_group_keys)
            self.group_to_key[group_name] = key
            self.group_grda[key] = GRDAResidual(
                dim=dim,
                temperature=temperature,
                text_temperature=text_temperature,
                text_pool_mode=text_pool_mode,
                norm_eps=norm_eps,
                proj_init_std=proj_init_std,
                guidance_mode=self.guidance_mode,
                residual_init=residual_init,
                guidance_scale_init=guidance_scale_init,
            )

    def get_grda(self, group_name: str) -> GRDAResidual:
        key = self.group_to_key[str(group_name)]
        return self.group_grda[key]


def _is_medsam2_transition_block(block: nn.Module) -> bool:
    return (
        hasattr(block, "proj")
        and isinstance(getattr(block, "proj"), nn.Linear)
        and hasattr(block, "pool")
    )


def _apply_medsam2_transition_align(block: nn.Module, x: torch.Tensor) -> torch.Tensor:
    if not _is_medsam2_transition_block(block):
        return x

    proj = getattr(block, "proj")
    pool = getattr(block, "pool", None)
    in_dim, out_dim = int(proj.in_features), int(proj.out_features)

    def _pool_last(y: torch.Tensor) -> torch.Tensor:
        if pool is None:
            return y
        y = y.permute(0, 3, 1, 2).contiguous()
        y = pool(y)
        return y.permute(0, 2, 3, 1).contiguous()

    if x.ndim == 4 and x.shape[-1] == in_dim:
        return proj(_pool_last(x))

    if x.ndim == 4 and x.shape[1] == in_dim:
        y = pool(x) if pool is not None else x
        y = y.permute(0, 2, 3, 1).contiguous()
        y = proj(y)
        return y.permute(0, 3, 1, 2).contiguous()

    if x.ndim == 3 and x.shape[-1] == in_dim:
        b, n, _ = x.shape
        s = int(round(math.sqrt(n)))
        if s * s == n:
            y = x.view(b, s, s, in_dim)
            y = proj(_pool_last(y))
            return y.view(b, -1, out_dim)
        return proj(x)

    return x


class TensorFirstArgBlockGRDAWrapper(nn.Module):
    """Wrapper for blocks whose main carrier is args[0] and output is a tensor."""

    def __init__(
        self,
        block: nn.Module,
        *,
        dim: int,
        grda_bank: GRDASharedBank,
        branch_state: BranchBlockState,
        group_name: str,
        is_group_end: bool,
        name: str,
        enable_transition_align: bool = False,
    ) -> None:
        super().__init__()
        self.block = block
        self.name = name
        self.dim = int(dim)
        self.branch_state = branch_state
        self.group_name = group_name
        self.is_group_end = is_group_end
        self.enable_transition_align = bool(enable_transition_align)
        object.__setattr__(self, "_grda_bank", grda_bank)

    @property
    def grda(self) -> GRDAResidual:
        return self._grda_bank.get_grda(self.group_name)

    def _maybe_align_history_after_block(self) -> None:
        if self.enable_transition_align and _is_medsam2_transition_block(self.block):
            self.branch_state.transform_history(
                lambda x: _apply_medsam2_transition_align(self.block, x)
            )

    def forward(self, *args: Any, **kwargs: Any) -> Any:
        if not args or not torch.is_tensor(args[0]):
            raise TypeError(f"{self.name}: expected tensor at args[0].")

        carrier = self.branch_state.begin_group_if_needed(args[0], self.group_name)
        mixed = args[0]
        if self.grda._is_compatible(carrier):
            txt_tok, txt_mask = self.branch_state.get_text_guidance()
            mixed = self.grda(
                self.branch_state.completed_blocks,
                carrier,
                txt_tok=txt_tok,
                txt_mask=txt_mask,
                guidance_logit_bank=self.branch_state.guidance_logit_bank,
            )

        output = self.block(mixed, *args[1:], **kwargs)
        if not torch.is_tensor(output):
            raise TypeError(f"{self.name}: expected tensor output, got {type(output)}.")

        self._maybe_align_history_after_block()
        self.branch_state.update_partial(output)
        if self.is_group_end:
            self.branch_state.finalize_group()
        return output


class BertLayerBlockGRDAWrapper(nn.Module):
    """Wrapper specialized for HuggingFace BertLayer."""

    def __init__(
        self,
        block: nn.Module,
        *,
        dim: int,
        grda_bank: GRDASharedBank,
        branch_state: BranchBlockState,
        group_name: str,
        is_group_end: bool,
        name: str,
        enable_transition_align: bool = False,
    ) -> None:
        super().__init__()
        self.block = block
        self.name = name
        self.dim = int(dim)
        self.branch_state = branch_state
        self.group_name = group_name
        self.is_group_end = is_group_end
        object.__setattr__(self, "_grda_bank", grda_bank)

    @property
    def grda(self) -> GRDAResidual:
        return self._grda_bank.get_grda(self.group_name)

    def _mixed_hidden(self, hidden_states: torch.Tensor) -> torch.Tensor:
        carrier = self.branch_state.begin_group_if_needed(
            hidden_states, self.group_name
        )
        txt_tok, txt_mask = self.branch_state.get_text_guidance()
        return self.grda(
            self.branch_state.completed_blocks,
            carrier,
            txt_tok=txt_tok,
            txt_mask=txt_mask,
            guidance_logit_bank=self.branch_state.guidance_logit_bank,
        )

    def forward(self, *args: Any, **kwargs: Any) -> Any:
        if args and torch.is_tensor(args[0]):
            args = (self._mixed_hidden(args[0]),) + tuple(args[1:])
        elif torch.is_tensor(kwargs.get("hidden_states")):
            kwargs = dict(kwargs)
            kwargs["hidden_states"] = self._mixed_hidden(kwargs["hidden_states"])
        else:
            raise TypeError(
                f"{self.name}: expected tensor in args[0] or kwargs['hidden_states']."
            )

        output = self.block(*args, **kwargs)
        self.branch_state.update_partial(_select_tensor_output(output, self.name))
        if self.is_group_end:
            self.branch_state.finalize_group()
        return output


_WRAPPER_TYPES = (TensorFirstArgBlockGRDAWrapper, BertLayerBlockGRDAWrapper)


def _ensure_module_list(module: Any, path: str) -> MutableSequence[nn.Module]:
    if not isinstance(module, (nn.Sequential, nn.ModuleList)):
        raise TypeError(
            f"{path} is not nn.Sequential/nn.ModuleList, got {type(module)}."
        )
    return module


def _get_medsam2_stack_root_path(
    sam: nn.Module,
) -> Tuple[MutableSequence[nn.Module], nn.Module, str]:
    return (
        _ensure_module_list(
            sam.image_encoder.trunk.blocks, "sam.image_encoder.trunk.blocks"
        ),
        sam.image_encoder.trunk,
        "image_encoder.trunk.blocks",
    )


def _get_visual_stack_root_path(
    clip: nn.Module,
) -> Tuple[MutableSequence[nn.Module], nn.Module, str]:
    return (
        _ensure_module_list(clip.visual.trunk.blocks, "clip.visual.trunk.blocks"),
        clip.visual.trunk,
        "visual.trunk.blocks",
    )


def _get_text_stack_root_path(
    clip: nn.Module,
) -> Tuple[MutableSequence[nn.Module], nn.Module, str]:
    encoder = clip.text.transformer.encoder
    return (
        _ensure_module_list(encoder.layer, "clip.text.transformer.encoder.layer"),
        encoder,
        "text.transformer.encoder.layer",
    )


def _infer_dim_medsam2_block(block: nn.Module) -> int:
    return _get_layernorm_dim(
        getattr(block, "norm1", None),
        f"MedSAM2 block {block.__class__.__name__} has no LayerNorm 'norm1'.",
    )


def _infer_dim_visual_block(block: nn.Module) -> int:
    return _get_layernorm_dim(
        getattr(block, "norm1", None),
        f"Visual block {block.__class__.__name__} has no LayerNorm 'norm1'.",
    )


def _infer_dim_text_block(block: nn.Module) -> int:
    ln = getattr(
        getattr(getattr(block, "attention", None), "output", None), "LayerNorm", None
    )
    return _get_layernorm_dim(
        ln,
        f"Text block {block.__class__.__name__} has no LayerNorm at attention.output.LayerNorm.",
    )


def _infer_dim_for_injection(block: nn.Module, dim_infer_fn) -> int:
    if isinstance(block, _WRAPPER_TYPES):
        return int(block.dim)
    return int(dim_infer_fn(block))


def _validate_no_overlaps(groups: Dict[str, Sequence[int]]) -> None:
    seen: Dict[int, str] = {}
    for group_name, indices in groups.items():
        for idx in indices:
            if idx in seen:
                raise ValueError(
                    f"Layer index {idx} appears in both group '{seen[idx]}' and '{group_name}'."
                )
            seen[idx] = group_name


def _normalize_groups(
    groups: Optional[Dict[str, Sequence[int]]],
    *,
    default_groups: Dict[str, Sequence[int]],
    target_groups: Optional[Sequence[str]] = None,
    total_layers: Optional[int] = None,
) -> Dict[str, List[int]]:
    normalized: Dict[str, List[int]] = {}
    for name, indices in dict(default_groups if groups is None else groups).items():
        idx = sorted(set(int(i) for i in indices))
        if not idx:
            continue
        if total_layers is not None:
            for i in idx:
                if i < 0 or i >= total_layers:
                    raise IndexError(
                        f"Group '{name}' has invalid layer index {i}, total_layers={total_layers}."
                    )
        normalized[str(name)] = idx

    if target_groups is not None:
        target_groups = [str(name) for name in target_groups]
        missing = [name for name in target_groups if name not in normalized]
        if missing:
            raise KeyError(
                f"Unknown target_groups: {missing}. Available groups: {list(normalized.keys())}."
            )
        normalized = {name: normalized[name] for name in target_groups}

    _validate_no_overlaps(normalized)
    return dict(sorted(normalized.items(), key=lambda kv: min(kv[1])))


def _groups_from_layers_or_group_size(
    *,
    prefix: str,
    layers: Optional[Sequence[int]],
    group_size: Optional[int],
    total_layers: int,
    groups: Optional[Dict[str, Sequence[int]]] = None,
    target_groups: Optional[Sequence[str]] = None,
) -> Dict[str, List[int]]:
    if groups is not None:
        return _normalize_groups(
            groups,
            default_groups={},
            target_groups=target_groups,
            total_layers=total_layers,
        )

    indices = (
        list(range(total_layers))
        if layers is None
        else sorted(set(int(i) for i in layers))
    )
    for i in indices:
        if i < 0 or i >= total_layers:
            raise IndexError(f"Invalid layer index {i}, total_layers={total_layers}.")
    if not indices:
        return {}

    if group_size is None or group_size <= 0:
        groups_auto: Dict[str, List[int]] = {f"{prefix}_all": indices}
    else:
        groups_auto = {
            f"{prefix}_{g}": indices[start : start + group_size]
            for g, start in enumerate(range(0, len(indices), group_size))
        }

    if target_groups is None:
        return groups_auto

    missing = [name for name in target_groups if name not in groups_auto]
    if missing:
        raise KeyError(
            f"Unknown target_groups: {missing}. Available groups: {list(groups_auto.keys())}."
        )
    return {name: groups_auto[name] for name in target_groups}


def _collect_group_dims(
    stack: MutableSequence[nn.Module],
    groups: Dict[str, Sequence[int]],
    *,
    dim_infer_fn,
    path_prefix: str,
) -> Tuple[Dict[str, int], Dict[int, int]]:
    group_dims: Dict[str, int] = {}
    index_dims: Dict[int, int] = {}

    for group_name, indices in groups.items():
        dims_by_index = {
            int(index): _infer_dim_for_injection(stack[int(index)], dim_infer_fn)
            for index in sorted(int(i) for i in indices)
        }
        unique_dims = sorted(set(dims_by_index.values()))
        if len(unique_dims) != 1:
            detail = ", ".join(
                f"{path_prefix}.{index}: {dim}" for index, dim in dims_by_index.items()
            )
            raise ValueError(
                f"Group '{group_name}' has mixed dims {{{', '.join(map(str, unique_dims))}}}. Details: {detail}."
            )
        group_dims[str(group_name)] = unique_dims[0]
        index_dims.update(dims_by_index)

    return group_dims, index_dims


def _register_reset_hook(
    root: nn.Module, branch_state: BranchBlockState, state: GRDAInjectionState
) -> None:
    def _clear_history_pre(_module: nn.Module, _inputs: Tuple[Any, ...]) -> None:
        branch_state.clear_history()

    def _clear_history_post(
        _module: nn.Module, _inputs: Tuple[Any, ...], _output: Any
    ) -> None:
        branch_state.clear()

    state.reset_handles.append(root.register_forward_pre_hook(_clear_history_pre))
    state.reset_handles.append(root.register_forward_hook(_clear_history_post))


def _print_summary(state: GRDAInjectionState) -> None:
    print(f"[GRDA][Summary][{state.branch_name}]")
    print(f"  injected_wrappers={len(state.injected_paths)}")
    print(f"  branch_states={len(state.branch_states)}")
    print(f"  reset_hook_registered={len(state.reset_handles) > 0}")
    print("  groups:")
    for group_name, indices in state.group_map.items():
        dim = state.group_dims.get(group_name, "unknown")
        print(f"    - {group_name}: {indices}, dim={dim}")
    print("  injected_paths:")
    for path in state.injected_paths:
        print(f"    - {path}")


def _apply_groups_to_stack(
    stack: MutableSequence[nn.Module],
    *,
    groups: Dict[str, Sequence[int]],
    state: GRDAInjectionState,
    temperature: float,
    text_temperature: float,
    text_pool_mode: str,
    norm_eps: float,
    proj_init_std: float,
    residual_init: float,
    guidance_scale_init: float,
    path_prefix: str,
    max_blocks: Optional[int],
    wrapper_cls: type[nn.Module],
    dim_infer_fn,
    bank_owner: nn.Module,
    bank_attr: str,
    guidance_mode: str,
    enable_transition_align: bool = False,
) -> None:
    guidance_mode = _normalize_guidance_mode(guidance_mode)
    state.group_map = {
        name: list(map(int, indices)) for name, indices in groups.items()
    }
    group_dims, index_dims = _collect_group_dims(
        stack, groups, dim_infer_fn=dim_infer_fn, path_prefix=path_prefix
    )
    state.group_dims = dict(group_dims)

    bank = GRDASharedBank(
        branch_name=state.branch_name,
        group_dims=group_dims,
        guidance_mode=guidance_mode,
        temperature=temperature,
        text_temperature=text_temperature,
        text_pool_mode=text_pool_mode,
        norm_eps=norm_eps,
        proj_init_std=proj_init_std,
        residual_init=residual_init,
        guidance_scale_init=guidance_scale_init,
    )
    if hasattr(bank_owner, bank_attr):
        delattr(bank_owner, bank_attr)
    setattr(bank_owner, bank_attr, bank)
    state.shared_banks.append(_SharedBankRef(bank_owner, bank_attr, bank))

    branch_state = BranchBlockState(max_blocks=max_blocks)
    state.branch_states.append(branch_state)

    for group_name, indices in groups.items():
        sorted_indices = sorted(int(i) for i in indices)
        last_index = sorted_indices[-1]
        for index in sorted_indices:
            block = stack[index]
            if isinstance(block, _WRAPPER_TYPES):
                continue

            wrapped = wrapper_cls(
                block,
                dim=index_dims[index],
                grda_bank=bank,
                branch_state=branch_state,
                group_name=str(group_name),
                is_group_end=index == last_index,
                name=f"{path_prefix}.{index}",
                enable_transition_align=enable_transition_align
                if wrapper_cls is TensorFirstArgBlockGRDAWrapper
                else False,
            )
            state.replaced_modules.append(
                _ReplacedModuleRef(stack=stack, index=index, original=block)
            )
            stack[index] = wrapped
            state.injected_paths.append(
                f"{path_prefix}.{index} [{group_name}, dim={index_dims[index]}, mode={guidance_mode}]"
            )


def inject_grda_to_medsam2(
    sam: nn.Module,
    *,
    groups: Optional[Dict[str, Sequence[int]]] = None,
    target_groups: Optional[Sequence[str]] = None,
    temperature: float = 1.0,
    norm_eps: float = 1e-6,
    proj_init_std: float = 0.02,
    residual_init: float = 0.0,
    max_blocks: Optional[int] = None,
    enable_transition_align: bool = True,
    print_summary: bool = True,
) -> GRDAInjectionState:
    stack, root, path = _get_medsam2_stack_root_path(sam)
    default_groups = {
        "g0": [0, 1],
        "g1": [2, 3],
        "g2": [4, 5, 6, 7, 8, 9, 10],
        "g3": [11],
    }
    use_groups = _normalize_groups(
        groups,
        default_groups=default_groups,
        target_groups=target_groups,
        total_layers=len(stack),
    )

    state = GRDAInjectionState(branch_name="medsam2")
    _apply_groups_to_stack(
        stack,
        groups=use_groups,
        state=state,
        temperature=temperature,
        text_temperature=0.07,
        text_pool_mode="mean",
        norm_eps=norm_eps,
        proj_init_std=proj_init_std,
        residual_init=residual_init,
        guidance_scale_init=0.0,
        path_prefix=path,
        max_blocks=max_blocks,
        wrapper_cls=TensorFirstArgBlockGRDAWrapper,
        dim_infer_fn=_infer_dim_medsam2_block,
        bank_owner=root,
        bank_attr="_grda_bank_medsam2",
        guidance_mode="base",
        enable_transition_align=enable_transition_align,
    )
    _register_reset_hook(root, state.branch_states[0], state)
    setattr(sam, "_grda_state_medsam2", state)
    if print_summary:
        _print_summary(state)
    return state


def inject_grda_to_biomedclip_visual(
    clip: nn.Module,
    *,
    layers: Optional[Sequence[int]] = None,
    group_size: Optional[int] = None,
    groups: Optional[Dict[str, Sequence[int]]] = None,
    target_groups: Optional[Sequence[str]] = None,
    temperature: float = 1.0,
    text_temperature: float = 0.07,
    text_pool_mode: str = "mean",
    norm_eps: float = 1e-6,
    proj_init_std: float = 0.02,
    residual_init: float = 0.0,
    guidance_scale_init: float = 0.0,
    max_blocks: Optional[int] = None,
    guidance_mode: str = "base+guidance",
    print_summary: bool = True,
) -> GRDAInjectionState:
    stack, root, path = _get_visual_stack_root_path(clip)
    use_groups = _groups_from_layers_or_group_size(
        prefix="visual",
        layers=layers,
        group_size=group_size,
        total_layers=len(stack),
        groups=groups,
        target_groups=target_groups,
    )

    state = GRDAInjectionState(branch_name="biomedclip_visual")
    _apply_groups_to_stack(
        stack,
        groups=use_groups,
        state=state,
        temperature=temperature,
        text_temperature=text_temperature,
        text_pool_mode=text_pool_mode,
        norm_eps=norm_eps,
        proj_init_std=proj_init_std,
        residual_init=residual_init,
        guidance_scale_init=guidance_scale_init,
        path_prefix=path,
        max_blocks=max_blocks,
        wrapper_cls=TensorFirstArgBlockGRDAWrapper,
        dim_infer_fn=_infer_dim_visual_block,
        bank_owner=root,
        bank_attr="_grda_bank_visual",
        guidance_mode=guidance_mode,
        enable_transition_align=False,
    )
    _register_reset_hook(root, state.branch_states[0], state)
    setattr(clip, "_grda_state_visual", state)
    if print_summary:
        _print_summary(state)
    return state


def inject_grda_to_biomedclip_text(
    clip: nn.Module,
    *,
    layers: Optional[Sequence[int]] = None,
    group_size: Optional[int] = None,
    groups: Optional[Dict[str, Sequence[int]]] = None,
    target_groups: Optional[Sequence[str]] = None,
    temperature: float = 1.0,
    norm_eps: float = 1e-6,
    proj_init_std: float = 0.02,
    residual_init: float = 0.0,
    max_blocks: Optional[int] = None,
    print_summary: bool = True,
) -> GRDAInjectionState:
    stack, root, path = _get_text_stack_root_path(clip)
    use_groups = _groups_from_layers_or_group_size(
        prefix="text",
        layers=layers,
        group_size=group_size,
        total_layers=len(stack),
        groups=groups,
        target_groups=target_groups,
    )

    state = GRDAInjectionState(branch_name="biomedclip_text")
    _apply_groups_to_stack(
        stack,
        groups=use_groups,
        state=state,
        temperature=temperature,
        text_temperature=0.07,
        text_pool_mode="mean",
        norm_eps=norm_eps,
        proj_init_std=proj_init_std,
        residual_init=residual_init,
        guidance_scale_init=0.0,
        path_prefix=path,
        max_blocks=max_blocks,
        wrapper_cls=BertLayerBlockGRDAWrapper,
        dim_infer_fn=_infer_dim_text_block,
        bank_owner=root,
        bank_attr="_grda_bank_text",
        guidance_mode="base",
        enable_transition_align=False,
    )
    _register_reset_hook(root, state.branch_states[0], state)
    setattr(clip, "_grda_state_text", state)
    if print_summary:
        _print_summary(state)
    return state


def inject_grda_to_biomedclip(
    clip: nn.Module,
    *,
    target: Sequence[str] = ("visual", "text"),
    visual_layers: Optional[Sequence[int]] = None,
    text_layers: Optional[Sequence[int]] = None,
    visual_group_size: Optional[int] = None,
    text_group_size: Optional[int] = None,
    visual_groups: Optional[Dict[str, Sequence[int]]] = None,
    text_groups: Optional[Dict[str, Sequence[int]]] = None,
    visual_target_groups: Optional[Sequence[str]] = None,
    text_target_groups: Optional[Sequence[str]] = None,
    temperature: float = 1.0,
    text_temperature: float = 0.07,
    text_pool_mode: str = "mean",
    norm_eps: float = 1e-6,
    proj_init_std: float = 0.02,
    residual_init: float = 0.0,
    guidance_scale_init: float = 0.0,
    max_blocks: Optional[int] = None,
    visual_guidance_mode: str = "base+guidance",
    print_summary: bool = True,
) -> Dict[str, GRDAInjectionState]:
    states: Dict[str, GRDAInjectionState] = {}
    target_set = set(target)

    if "visual" in target_set:
        states["visual"] = inject_grda_to_biomedclip_visual(
            clip,
            layers=visual_layers,
            group_size=visual_group_size,
            groups=visual_groups,
            target_groups=visual_target_groups,
            temperature=temperature,
            text_temperature=text_temperature,
            text_pool_mode=text_pool_mode,
            norm_eps=norm_eps,
            proj_init_std=proj_init_std,
            residual_init=residual_init,
            guidance_scale_init=guidance_scale_init,
            max_blocks=max_blocks,
            guidance_mode=visual_guidance_mode,
            print_summary=print_summary,
        )

    if "text" in target_set:
        states["text"] = inject_grda_to_biomedclip_text(
            clip,
            layers=text_layers,
            group_size=text_group_size,
            groups=text_groups,
            target_groups=text_target_groups,
            temperature=temperature,
            norm_eps=norm_eps,
            proj_init_std=proj_init_std,
            residual_init=residual_init,
            max_blocks=max_blocks,
            print_summary=print_summary,
        )

    setattr(clip, "_grda_state", states)
    return states


def inject_grda_to_radar_backbones(
    sam: Optional[nn.Module] = None,
    clip: Optional[nn.Module] = None,
    *,
    use_medsam2: bool = True,
    use_biomedclip_visual: bool = False,
    use_biomedclip_text: bool = True,
    medsam2_groups: Optional[Dict[str, Sequence[int]]] = None,
    medsam2_target_groups: Optional[Sequence[str]] = None,
    visual_layers: Optional[Sequence[int]] = None,
    text_layers: Optional[Sequence[int]] = None,
    visual_group_size: Optional[int] = None,
    text_group_size: Optional[int] = None,
    visual_groups: Optional[Dict[str, Sequence[int]]] = None,
    text_groups: Optional[Dict[str, Sequence[int]]] = None,
    visual_target_groups: Optional[Sequence[str]] = None,
    text_target_groups: Optional[Sequence[str]] = None,
    temperature: float = 1.0,
    text_temperature: float = 0.07,
    text_pool_mode: str = "mean",
    norm_eps: float = 1e-6,
    proj_init_std: float = 0.02,
    residual_init: float = 0.0,
    guidance_scale_init: float = 0.0,
    max_blocks: Optional[int] = None,
    sam_enable_transition_align: bool = True,
    visual_guidance_mode: str = "base+guidance",
    print_summary: bool = True,
) -> Dict[str, Any]:
    states: Dict[str, Any] = {}

    if use_medsam2:
        if sam is None:
            raise ValueError("sam must be provided when use_medsam2=True.")
        states["medsam2"] = inject_grda_to_medsam2(
            sam,
            groups=medsam2_groups,
            target_groups=medsam2_target_groups,
            temperature=temperature,
            norm_eps=norm_eps,
            proj_init_std=proj_init_std,
            residual_init=residual_init,
            max_blocks=max_blocks,
            enable_transition_align=sam_enable_transition_align,
            print_summary=print_summary,
        )

    if use_biomedclip_visual or use_biomedclip_text:
        if clip is None:
            raise ValueError("clip must be provided when using BiomedCLIP GRDA.")
        target: List[str] = []
        if use_biomedclip_visual:
            target.append("visual")
        if use_biomedclip_text:
            target.append("text")

        states["biomedclip"] = inject_grda_to_biomedclip(
            clip,
            target=tuple(target),
            visual_layers=visual_layers,
            text_layers=text_layers,
            visual_group_size=visual_group_size,
            text_group_size=text_group_size,
            visual_groups=visual_groups,
            text_groups=text_groups,
            visual_target_groups=visual_target_groups,
            text_target_groups=text_target_groups,
            temperature=temperature,
            text_temperature=text_temperature,
            text_pool_mode=text_pool_mode,
            norm_eps=norm_eps,
            proj_init_std=proj_init_std,
            residual_init=residual_init,
            guidance_scale_init=guidance_scale_init,
            max_blocks=max_blocks,
            visual_guidance_mode=visual_guidance_mode,
            print_summary=print_summary,
        )

    return states


def set_grda_text_guidance(
    module: nn.Module, attr_name: str, txt_tok: torch.Tensor, txt_mask: torch.Tensor
) -> None:
    for st in _iter_state_objects(getattr(module, attr_name, None)):
        for branch_state in st.branch_states:
            branch_state.set_text_guidance(txt_tok, txt_mask)


GRDA_STATE_ATTRS = (
    "_grda_state_medsam2",
    "_grda_state_visual",
    "_grda_state_text",
    "_grda_state",
)


def clear_grda_guidance(module: nn.Module) -> None:
    for st in _iter_unique_grda_states(module):
        for branch_state in st.branch_states:
            branch_state.clear_guidance()


def clear_grda_states(module: nn.Module) -> None:
    for st in _iter_unique_grda_states(module):
        for branch_state in st.branch_states:
            branch_state.clear_history()


__all__ = [
    "GRDAInjectionState",
    "GRDASharedBank",
    "BranchBlockState",
    "GRDAResidual",
    "TensorFirstArgBlockGRDAWrapper",
    "BertLayerBlockGRDAWrapper",
    "set_grda_text_guidance",
    "clear_grda_guidance",
    "inject_grda_to_medsam2",
    "inject_grda_to_biomedclip_visual",
    "inject_grda_to_biomedclip_text",
    "inject_grda_to_biomedclip",
    "inject_grda_to_radar_backbones",
    "clear_grda_states",
]
