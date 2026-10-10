"""Full RADAR model: BiomedCLIP guidance with MedSAM2 segmentation."""

from __future__ import annotations

import math
from typing import Dict, List, Optional, Sequence, Tuple

import torch
import torch.nn.functional as F
from torch import nn

from model.grda import set_grda_text_guidance


class RADAR(nn.Module):
    """Full RADAR model for text-guided medical image segmentation.

    Inference requires only the MedSAM2 image, the BiomedCLIP image, and text
    tokens. SDRA tensors are accepted solely for training-time supervision.
    """

    MPG_QUANTILE = 0.90
    MPG_MIN_COMPONENT_AREA = 16
    MPG_MAX_COMPONENTS = 2
    MPG_EROSION_KERNEL = 5
    MPG_EROSION_ITERATIONS = 2
    MPG_INNER_DILATION = 17
    MPG_OUTER_DILATION = 27
    MPG_MAX_CANDIDATES = 128
    MPG_MIN_POINT_DISTANCE = 12.0
    MPG_POSITIVE_POINTS = 2
    MPG_NEGATIVE_POINTS = 2
    MPG_RESOLUTION = (512, 512)

    def __init__(
        self,
        *,
        medsam2: nn.Module,
        biomedclip: nn.Module,
        sdra_teacher: Optional[nn.Module] = None,
        torch_dtype: torch.dtype = torch.float32,
        output_size: Tuple[int, int] = (224, 224),
        clip_indices: Sequence[int] = (11,),
        use_grda_clip_visual: bool = True,
    ) -> None:
        super().__init__()
        self.sam = medsam2
        self.clip = biomedclip
        self.sdra_teacher = sdra_teacher
        self.torch_dtype = torch_dtype
        self.output_size = tuple(int(value) for value in output_size)
        self.clip_indices = [int(index) for index in clip_indices]
        self.use_grda_clip_visual = bool(use_grda_clip_visual)

        self.clip.text.output_tokens = True
        text_config = getattr(getattr(self.clip, "text", None), "config", None)
        self.pad_token_id = int(getattr(text_config, "pad_token_id", 0))

        self.clip_embed_dim = 512
        self.clip_token_dim = 768
        self.prompt_dim = 256

        self.sim_logit_scale = nn.Parameter(torch.tensor(math.log(1.0 / 0.07)))

        # Map the global text embedding into the MedSAM2 prompt space.
        self.txt_to_prompt = nn.Sequential(
            nn.Linear(self.clip_embed_dim, self.prompt_dim),
            nn.GELU(),
            nn.Linear(self.prompt_dim, self.prompt_dim),
        )

        if self.sdra_teacher is not None:
            self.sdra_teacher.eval()
            for parameter in self.sdra_teacher.parameters():
                parameter.requires_grad = False

    def train(self, mode: bool = True) -> "RADAR":
        super().train(mode)
        # The SDRA crop encoder remains a frozen BiomedCLIP teacher.
        if self.sdra_teacher is not None:
            self.sdra_teacher.eval()
        return self

    @property
    def device(self) -> torch.device:
        return next(self.parameters()).device

    @staticmethod
    def _zero_loss(reference: torch.Tensor) -> torch.Tensor:
        return reference.new_zeros(())

    @staticmethod
    def _as_list(value):
        return list(value) if isinstance(value, (list, tuple)) else [value]

    @staticmethod
    def _tokens_to_map(tokens: torch.Tensor) -> torch.Tensor:
        batch, count, channels = tokens.shape
        height = int(round(math.sqrt(count)))
        if height * height != count:
            raise ValueError(f"Token count {count} is not a square feature map.")
        return tokens.transpose(1, 2).contiguous().view(batch, channels, height, height)

    def _number_of_clip_blocks(self, clip_model: nn.Module) -> int:
        visual = getattr(clip_model, "visual", None)
        if visual is None:
            raise AttributeError("BiomedCLIP has no visual encoder.")
        candidates = (
            getattr(getattr(visual, "trunk", None), "blocks", None),
            getattr(getattr(visual, "transformer", None), "resblocks", None),
            getattr(getattr(visual, "transformer", None), "blocks", None),
            getattr(visual, "blocks", None),
            getattr(visual, "resblocks", None),
        )
        for blocks in candidates:
            if blocks is not None:
                return len(blocks)
        raise AttributeError("Cannot locate the BiomedCLIP visual blocks.")

    def _resolved_clip_indices(self, clip_model: nn.Module) -> List[int]:
        total = self._number_of_clip_blocks(clip_model)
        resolved = []
        for index in self.clip_indices:
            index = total + index if index < 0 else index
            if not 0 <= index < total:
                raise ValueError(
                    f"BiomedCLIP block index {index} is outside [0, {total})."
                )
            resolved.append(index)
        return sorted(set(resolved))

    @staticmethod
    def _grda_states(module: nn.Module, attribute: str):
        value = getattr(module, attribute, None)
        if value is None:
            return []
        if isinstance(value, dict):
            return [item for item in value.values() if hasattr(item, "branch_states")]
        return [value] if hasattr(value, "branch_states") else []

    def _clear_grda_branch(self, module: nn.Module, attribute: str) -> None:
        for state in self._grda_states(module, attribute):
            for branch_state in state.branch_states:
                branch_state.clear()

    def _encode_text(
        self, tokens: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        self._clear_grda_branch(self.clip, "_grda_state_text")
        global_text, text_tokens = self.clip.text(tokens)
        global_text = F.normalize(global_text.to(self.torch_dtype), dim=-1)
        text_tokens = text_tokens.to(self.torch_dtype)
        # Padding must never contribute to GRDA textual routing.
        valid_token_mask = tokens.ne(self.pad_token_id)
        return global_text, text_tokens, valid_token_mask

    def _encode_clip_image(
        self,
        image_clip: torch.Tensor,
        text_tokens: torch.Tensor,
        valid_token_mask: torch.Tensor,
    ) -> List[torch.Tensor]:
        self._clear_grda_branch(self.clip, "_grda_state_visual")
        if self.use_grda_clip_visual:
            set_grda_text_guidance(
                self.clip,
                "_grda_state_visual",
                txt_tok=text_tokens,
                txt_mask=valid_token_mask,
            )

        resolved = self._resolved_clip_indices(self.clip)
        result = self.clip.forward_intermediates(
            image=image_clip,
            image_indices=self.clip_indices,
            stop_early=True,
            normalize=True,
            normalize_intermediates=True,
            intermediates_only=False,
            image_output_fmt="NCHW",
            image_output_extra_tokens=True,
        )
        maps = self._as_list(result.get("image_intermediates"))
        if len(maps) != len(resolved):
            raise RuntimeError(
                f"Expected {len(resolved)} dense maps, received {len(maps)}."
            )
        return [feature.to(self.torch_dtype) for feature in maps]

    def _encode_medsam2(
        self, image_sam: torch.Tensor
    ) -> Tuple[torch.Tensor, List[torch.Tensor]]:
        self._clear_grda_branch(self.sam, "_grda_state_medsam2")
        backbone_output = self.sam.forward_image(image_sam)
        _, tokens, _, _ = self.sam._prepare_backbone_features(backbone_output)
        tokens = [feature.to(image_sam.dtype) for feature in tokens]
        if getattr(self.sam, "directly_add_no_mem_embed", False) and hasattr(
            self.sam, "no_mem_embed"
        ):
            tokens[-1] = tokens[-1] + self.sam.no_mem_embed
        maps = [self._tokens_to_map(feature.transpose(0, 1)) for feature in tokens]
        return maps[-1], maps[:-1]

    def _visual_projection(self) -> torch.Tensor:
        projection = self.clip.visual.head.proj
        if not isinstance(projection, nn.Linear):
            raise TypeError("BiomedCLIP visual projection must be a linear layer.")
        if (projection.in_features, projection.out_features) != (
            self.clip_token_dim,
            self.clip_embed_dim,
        ):
            raise ValueError(
                "Expected a 768 -> 512 BiomedCLIP visual projection, "
                f"received {projection.in_features} -> {projection.out_features}."
            )
        return projection.weight.t().contiguous()

    def _project_dense_to_shared_space(self, dense_map: torch.Tensor) -> torch.Tensor:
        """Project 768-d dense student features into the frozen 512-d shared space."""

        batch, channels, height, width = dense_map.shape
        if channels != self.clip_token_dim:
            raise ValueError(
                f"Expected {self.clip_token_dim} dense channels, received {channels}."
            )
        tokens = dense_map.permute(0, 2, 3, 1).reshape(-1, channels)
        projection = self._visual_projection().to(
            device=tokens.device, dtype=tokens.dtype
        )
        # SDRA uses the frozen BiomedCLIP 768 -> 512 shared-space projection.
        projected = tokens @ projection
        return (
            projected.view(batch, height, width, self.clip_embed_dim)
            .permute(0, 3, 1, 2)
            .contiguous()
        )

    def _similarity_logits(
        self, dense_map: torch.Tensor, global_text: torch.Tensor
    ) -> torch.Tensor:
        dense_shared = F.normalize(
            self._project_dense_to_shared_space(dense_map), dim=1
        )
        text = global_text.to(device=dense_shared.device, dtype=dense_shared.dtype)
        similarity = torch.einsum("bchw,bc->bhw", dense_shared, text).unsqueeze(1)
        logits = self.sim_logit_scale.exp().clamp(max=100.0) * similarity
        return F.interpolate(
            logits, size=self.output_size, mode="bilinear", align_corners=False
        )

    @staticmethod
    def _binary_dilate(mask: torch.Tensor, kernel_size: int) -> torch.Tensor:
        value = F.max_pool2d(
            mask.float()[None, None],
            kernel_size=kernel_size,
            stride=1,
            padding=kernel_size // 2,
        )
        return value[0, 0].gt(0.5)

    @staticmethod
    def _binary_erode(
        mask: torch.Tensor, kernel_size: int, iterations: int
    ) -> torch.Tensor:
        value = mask.float()[None, None]
        for _ in range(iterations):
            value = 1.0 - F.max_pool2d(
                1.0 - value, kernel_size=kernel_size, stride=1, padding=kernel_size // 2
            )
            value = value.gt(0.5).float()
        return value[0, 0].gt(0.5)

    @staticmethod
    def _candidate_points(
        response: torch.Tensor,
        region: torch.Tensor,
        *,
        largest: bool,
        maximum: int,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        yx = torch.nonzero(region, as_tuple=False)
        if yx.numel() == 0:
            return yx.new_empty((0, 2)), response.new_empty((0,))
        scores = response[yx[:, 0], yx[:, 1]]
        count = min(int(maximum), int(scores.numel()))
        selected = torch.topk(scores, k=count, largest=largest, sorted=True).indices
        yx = yx[selected]
        return torch.stack((yx[:, 1], yx[:, 0]), dim=1), scores[selected]

    @staticmethod
    def _spread_select(
        candidate_xy: torch.Tensor,
        candidate_scores: torch.Tensor,
        count: int,
        existing: List[torch.Tensor],
        *,
        largest: bool,
        minimum_distance: float,
    ) -> List[torch.Tensor]:
        if count <= 0 or candidate_xy.numel() == 0:
            return []
        order = torch.argsort(candidate_scores, descending=largest)
        chosen: List[torch.Tensor] = []
        threshold = float(minimum_distance) ** 2
        for index in order.tolist():
            point = candidate_xy[index]
            previous = existing + chosen
            if all(
                float(((point.float() - item.float()) ** 2).sum()) >= threshold
                for item in previous
            ):
                chosen.append(point)
                if len(chosen) == count:
                    break
        return chosen

    def _component_masks(self, high_mask: torch.Tensor) -> List[torch.Tensor]:
        height, width = high_mask.shape
        labels = (
            torch.arange(
                1, height * width + 1, device=high_mask.device, dtype=torch.float32
            ).view(1, 1, height, width)
            * high_mask.float()[None, None]
        )
        # Propagating the maximum label through each 8-connected component is
        # fully tensorized and keeps connected-component extraction on-device.
        for _ in range(max(height, width)):
            labels = (
                F.max_pool2d(labels, kernel_size=3, stride=1, padding=1)
                * high_mask.float()[None, None]
            )
        labels = labels[0, 0].long()
        component_ids = torch.unique(labels)
        component_ids = component_ids[component_ids.gt(0)]
        components: List[Tuple[int, torch.Tensor]] = []
        for component_id in component_ids.tolist():
            component = labels.eq(component_id)
            area = int(component.sum())
            if area >= self.MPG_MIN_COMPONENT_AREA:
                components.append((area, component))
        components.sort(key=lambda item: item[0], reverse=True)
        return [component for _, component in components[: self.MPG_MAX_COMPONENTS]]

    def _positive_core(self, component: torch.Tensor) -> torch.Tensor:
        # Two 5x5 erosions are performed in the 512x512 MedSAM2 coordinate system.
        for iterations in range(self.MPG_EROSION_ITERATIONS, -1, -1):
            core = (
                component
                if iterations == 0
                else self._binary_erode(component, self.MPG_EROSION_KERNEL, iterations)
            )
            if int(core.sum()) >= 4:
                return core
        return component

    def _hard_background(
        self, component: torch.Tensor, retained: torch.Tensor
    ) -> torch.Tensor:
        inner = self._binary_dilate(component, self.MPG_INNER_DILATION)
        outer = self._binary_dilate(component, self.MPG_OUTER_DILATION)
        return outer & ~inner & ~retained

    def _fill_points(
        self,
        response: torch.Tensor,
        region: torch.Tensor,
        chosen: List[torch.Tensor],
        count: int,
        *,
        largest: bool,
        existing: List[torch.Tensor],
    ) -> None:
        candidate_xy, candidate_scores = self._candidate_points(
            response,
            region,
            largest=largest,
            maximum=self.MPG_MAX_CANDIDATES,
        )
        chosen.extend(
            self._spread_select(
                candidate_xy,
                candidate_scores,
                count - len(chosen),
                existing + chosen,
                largest=largest,
                minimum_distance=self.MPG_MIN_POINT_DISTANCE,
            )
        )

    @torch.no_grad()
    def generate_mpg_points(
        self, similarity_logits: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Generate point prompts in MedSAM2's 512x512 coordinate space."""

        response_512 = F.interpolate(
            similarity_logits.float(),
            size=self.MPG_RESOLUTION,
            mode="bilinear",
            align_corners=False,
        )
        coordinates, labels = [], []
        for batch_index in range(response_512.shape[0]):
            response = response_512[batch_index, 0]
            threshold = torch.maximum(
                torch.quantile(response.flatten(), self.MPG_QUANTILE),
                response.new_tensor(0.0),
            )
            high_mask = response.ge(threshold)
            components = self._component_masks(high_mask)
            if not components:
                raise RuntimeError(
                    "MPG found no connected component meeting the 16-pixel minimum."
                )

            retained = torch.zeros_like(high_mask)
            for component in components:
                retained |= component
            positive_regions = [
                self._positive_core(component) for component in components
            ]
            negative_regions = [
                self._hard_background(component, retained) for component in components
            ]

            positive: List[torch.Tensor] = []
            negative: List[torch.Tensor] = []
            for region in positive_regions:
                self._fill_points(
                    response,
                    region,
                    positive,
                    min(self.MPG_POSITIVE_POINTS, len(positive) + 1),
                    largest=True,
                    existing=[],
                )
            positive = positive[: self.MPG_POSITIVE_POINTS]
            if len(positive) < self.MPG_POSITIVE_POINTS:
                self._fill_points(
                    response,
                    torch.stack(positive_regions).any(dim=0),
                    positive,
                    self.MPG_POSITIVE_POINTS,
                    largest=True,
                    existing=[],
                )
            for region in negative_regions:
                self._fill_points(
                    response,
                    region,
                    negative,
                    min(self.MPG_NEGATIVE_POINTS, len(negative) + 1),
                    largest=True,
                    existing=positive,
                )
            negative = negative[: self.MPG_NEGATIVE_POINTS]
            negative_union = torch.stack(negative_regions).any(dim=0)
            if len(negative) < self.MPG_NEGATIVE_POINTS:
                self._fill_points(
                    response,
                    negative_union,
                    negative,
                    self.MPG_NEGATIVE_POINTS,
                    largest=True,
                    existing=positive,
                )
            if (
                len(positive) != self.MPG_POSITIVE_POINTS
                or len(negative) != self.MPG_NEGATIVE_POINTS
            ):
                raise RuntimeError(
                    "MPG could not produce two positive and two negative points."
                )
            coordinates.append(torch.stack(positive + negative).float())
            labels.append(
                torch.tensor([1, 1, 0, 0], device=response.device, dtype=torch.long)
            )
        return torch.stack(coordinates), torch.stack(labels)

    def _similarity_mask_prompt(self, similarity_logits: torch.Tensor) -> torch.Tensor:
        mask_height, mask_width = self.sam.sam_prompt_encoder.mask_input_size
        value = similarity_logits.float()
        mean = value.mean(dim=(-2, -1), keepdim=True)
        std = value.std(dim=(-2, -1), keepdim=True, unbiased=False).clamp_min(1e-6)
        value = (value - mean) / std
        threshold = torch.quantile(
            value.flatten(2), self.MPG_QUANTILE, dim=2, keepdim=True
        ).unsqueeze(-1)
        value = value * value.ge(threshold)
        return F.interpolate(
            value, size=(mask_height, mask_width), mode="bilinear", align_corners=False
        ).to(similarity_logits.dtype)

    @staticmethod
    def vectorized_regional_pool(
        dense_shared: torch.Tensor,
        region_weights: torch.Tensor,
    ) -> torch.Tensor:
        """Pool all target/boundary/context regions without per-ROI Python loops."""

        if dense_shared.ndim != 4 or region_weights.ndim != 5:
            raise ValueError("Expected dense [B,C,H,W] and weights [B,G,3,H,W].")
        batch, channels, height, width = dense_shared.shape
        weight_batch, groups, regions, _, _ = region_weights.shape
        if batch != weight_batch or regions != 3:
            raise ValueError(
                "SDRA weights must contain target, boundary, and context for each image."
            )

        weights = F.interpolate(
            region_weights.reshape(
                batch * groups * regions, 1, *region_weights.shape[-2:]
            ).float(),
            size=(height, width),
            mode="bilinear",
            align_corners=False,
        ).to(dtype=dense_shared.dtype)
        batch_ids = (
            torch.arange(batch, device=dense_shared.device)
            .view(batch, 1, 1)
            .expand(batch, groups, regions)
            .reshape(-1)
        )
        dense = dense_shared[batch_ids]
        numerator = (dense * weights).flatten(2).sum(dim=-1)
        denominator = weights.flatten(2).sum(dim=-1).clamp_min(1e-6)
        return (numerator / denominator).view(batch, groups, regions, channels)

    def _sdra_loss(
        self,
        dense_map: torch.Tensor,
        teacher_crops: Optional[torch.Tensor],
        region_weights: Optional[torch.Tensor],
        valid_mask: Optional[torch.Tensor],
    ) -> torch.Tensor:
        if not self.training or self.sdra_teacher is None:
            return self._zero_loss(dense_map)
        if teacher_crops is None or region_weights is None or valid_mask is None:
            return self._zero_loss(dense_map)

        teacher_crops = teacher_crops.to(device=dense_map.device, dtype=dense_map.dtype)
        region_weights = region_weights.to(
            device=dense_map.device, dtype=dense_map.dtype
        )
        valid_mask = valid_mask.to(device=dense_map.device, dtype=torch.bool)
        if not torch.any(valid_mask):
            return self._zero_loss(dense_map)

        dense_shared = self._project_dense_to_shared_space(dense_map)
        # All three structured regions are pooled in one batched tensor reduction.
        student = self.vectorized_regional_pool(dense_shared, region_weights)[
            valid_mask
        ]
        crops = teacher_crops[valid_mask]
        with torch.no_grad():
            teacher = self.sdra_teacher.encode_image(
                crops.reshape(-1, *crops.shape[-3:]), normalize=False
            ).view(crops.shape[0], 3, -1)

        student = F.normalize(student, dim=-1)
        teacher = F.normalize(teacher.to(student.dtype), dim=-1)
        positive = 1.0 - F.cosine_similarity(student, teacher, dim=-1)
        target_loss = positive[:, 0].mean()
        boundary_loss = positive[:, 1].mean()
        context_loss = positive[:, 2].mean()
        boundary_rejection = F.relu(
            F.cosine_similarity(student[:, 1], teacher[:, 0], dim=-1)
            - F.cosine_similarity(student[:, 0], teacher[:, 0], dim=-1)
        ).mean()
        return (
            0.5 * target_loss
            + 0.2 * boundary_loss
            + 0.2 * context_loss
            + 0.2 * boundary_rejection
        )

    def forward(
        self,
        image_sam: torch.Tensor,
        image_clip: torch.Tensor,
        tokens: torch.Tensor,
        *,
        sdra_teacher_crops: Optional[torch.Tensor] = None,
        sdra_region_weights: Optional[torch.Tensor] = None,
        sdra_valid_mask: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:
        global_text, text_tokens, valid_token_mask = self._encode_text(tokens)
        dense_maps = self._encode_clip_image(image_clip, text_tokens, valid_token_mask)
        medsam_embedding, high_resolution_features = self._encode_medsam2(image_sam)

        text_prompt = self.txt_to_prompt(global_text).unsqueeze(1)
        similarity_logits = self._similarity_logits(dense_maps[-1], global_text)
        with torch.no_grad():
            point_prompt = self.generate_mpg_points(
                torch.sigmoid(similarity_logits.detach())
            )
            mask_prompt = self._similarity_mask_prompt(similarity_logits.detach())

        sparse_prompt, dense_prompt = self.sam.sam_prompt_encoder(
            points=point_prompt,
            boxes=None,
            masks=mask_prompt,
        )
        # Text is a learned sparse prompt token, produced only by txt_to_prompt.
        sparse_prompt = torch.cat(
            (sparse_prompt, text_prompt.to(sparse_prompt.dtype)), dim=1
        )
        low_resolution_logits, _, _, _ = self.sam.sam_mask_decoder(
            image_embeddings=medsam_embedding,
            image_pe=self.sam.sam_prompt_encoder.get_dense_pe(),
            sparse_prompt_embeddings=sparse_prompt,
            dense_prompt_embeddings=dense_prompt,
            multimask_output=False,
            repeat_image=False,
            high_res_features=high_resolution_features,
        )
        logits = F.interpolate(
            low_resolution_logits,
            size=self.output_size,
            mode="bilinear",
            align_corners=False,
        )
        probability = torch.sigmoid(logits)
        sdra_loss = self._sdra_loss(
            dense_maps[-1], sdra_teacher_crops, sdra_region_weights, sdra_valid_mask
        )
        return probability, {
            "similarity_probability": torch.sigmoid(similarity_logits),
            "segmentation_probability": probability,
            "sdra_loss": sdra_loss,
        }


__all__ = ["RADAR"]
