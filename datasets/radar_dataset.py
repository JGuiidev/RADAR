"""Datasets and data loaders for RADAR medical image segmentation."""

from __future__ import annotations

import random
from pathlib import Path
from typing import Callable, Dict, List, Mapping, Tuple

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from PIL import Image
from scipy import ndimage
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms

CLIP_MEAN = (0.48145466, 0.4578275, 0.40821073)
CLIP_STD = (0.26862954, 0.26130258, 0.27577711)
SAM2_PIXEL_MEAN = (123.675, 116.28, 103.53)
SAM2_PIXEL_STD = (58.395, 57.12, 57.375)
IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff", ".webp"}

# SDRA region construction settings.
SDRA_GROUP_MERGE_KERNEL = 13
SDRA_MIN_GROUP_PIXELS = 64
SDRA_TARGET_EXPANSION = 3
SDRA_BOUNDARY_EXPANSION = 8
SDRA_CONTEXT_EXPANSION = 20


def load_text_annotations(path: str | Path) -> Dict[str, str]:
    """Load Image/Description annotations from the split spreadsheet."""

    path = Path(path)
    if not path.is_file():
        raise FileNotFoundError(f"Text annotation file not found: {path}")
    frame = pd.read_excel(path, engine="openpyxl")
    required = {"Image", "Description"}
    missing = required.difference(frame.columns)
    if missing:
        raise ValueError(f"Missing annotation columns in {path}: {sorted(missing)}")
    annotations: Dict[str, str] = {}
    for row in frame.itertuples(index=False):
        image_name = getattr(row, "Image")
        if pd.isna(image_name):
            continue
        description = getattr(row, "Description")
        annotations[str(image_name)] = "" if pd.isna(description) else str(description)
    return annotations


def build_clip_preprocess(image_size: int = 224) -> transforms.Compose:
    return transforms.Compose(
        [
            transforms.Resize(
                (image_size, image_size),
                interpolation=transforms.InterpolationMode.BICUBIC,
                antialias=True,
            ),
            transforms.ToTensor(),
            transforms.Normalize(CLIP_MEAN, CLIP_STD),
        ]
    )


def build_resize_to_tensor(image_size: int) -> transforms.Compose:
    return transforms.Compose(
        [
            transforms.Resize(
                (image_size, image_size),
                interpolation=transforms.InterpolationMode.BICUBIC,
                antialias=True,
            ),
            transforms.ToTensor(),
        ]
    )


def sam2_normalize(image: torch.Tensor) -> torch.Tensor:
    mean = torch.tensor(SAM2_PIXEL_MEAN, dtype=torch.float32).view(3, 1, 1)
    std = torch.tensor(SAM2_PIXEL_STD, dtype=torch.float32).view(3, 1, 1)
    return (image * 255.0 - mean) / std


def binary_mask(mask: Image.Image, size: int) -> torch.Tensor:
    resized = mask.convert("L").resize((size, size), resample=Image.Resampling.NEAREST)
    return torch.from_numpy(
        (np.asarray(resized, dtype=np.uint8) >= 128).astype(np.float32)
    ).unsqueeze(0)


def pad_or_truncate(
    tokens: torch.Tensor, length: int, pad_token_id: int
) -> torch.Tensor:
    tokens = tokens.flatten().to(torch.int64)
    if tokens.numel() >= length:
        return tokens[:length]
    padding = torch.full((length - tokens.numel(),), pad_token_id, dtype=tokens.dtype)
    return torch.cat((tokens, padding))


def _random_geometric_augmentation(
    image: Image.Image, mask: Image.Image
) -> Tuple[Image.Image, Image.Image]:
    image_array = np.asarray(image.convert("RGB"), dtype=np.uint8)
    mask_array = np.asarray(mask.convert("L"), dtype=np.uint8)
    choice = random.random()
    if choice > 0.5:
        rotations = np.random.randint(0, 4)
        axis = np.random.randint(0, 2)
        image_array = np.flip(np.rot90(image_array, rotations), axis=axis).copy()
        mask_array = np.flip(np.rot90(mask_array, rotations), axis=axis).copy()
    elif choice > 0.25:
        angle = np.random.randint(-20, 20)
        image_array = ndimage.rotate(image_array, angle, order=3, reshape=False)
        mask_array = ndimage.rotate(mask_array, angle, order=0, reshape=False)
        image_array = np.clip(image_array, 0, 255).astype(np.uint8)
    mask_array = (mask_array > 0).astype(np.uint8) * 255
    return Image.fromarray(image_array).convert("RGB"), Image.fromarray(
        mask_array
    ).convert("L")


def _clamp_box(box: torch.Tensor, height: int, width: int) -> torch.Tensor:
    x1, y1, x2, y2 = box.tolist()
    x1 = max(0.0, min(float(width - 1), x1))
    y1 = max(0.0, min(float(height - 1), y1))
    x2 = max(x1 + 1.0, min(float(width), x2))
    y2 = max(y1 + 1.0, min(float(height), y2))
    return torch.tensor((x1, y1, x2, y2), dtype=torch.float32)


def _mask_box(mask: torch.Tensor) -> torch.Tensor:
    y, x = torch.where(mask)
    if y.numel() == 0:
        raise ValueError("Cannot create an SDRA box from an empty region.")
    return torch.tensor(
        (float(x.min()), float(y.min()), float(x.max() + 1), float(y.max() + 1)),
        dtype=torch.float32,
    )


def _expand_box(
    box: torch.Tensor, pixels: int, height: int, width: int
) -> torch.Tensor:
    x1, y1, x2, y2 = box.tolist()
    return _clamp_box(
        torch.tensor((x1 - pixels, y1 - pixels, x2 + pixels, y2 + pixels)),
        height,
        width,
    )


def _box_mask(box: torch.Tensor, height: int, width: int) -> torch.Tensor:
    x1, y1, x2, y2 = box.round().to(torch.int64).tolist()
    x1, y1 = max(0, x1), max(0, y1)
    x2, y2 = min(width, max(x1 + 1, x2)), min(height, max(y1 + 1, y2))
    result = torch.zeros((height, width), dtype=torch.bool)
    result[y1:y2, x1:x2] = True
    return result


def _scale_box(
    box: torch.Tensor, source_hw: Tuple[int, int], target_hw: Tuple[int, int]
) -> torch.Tensor:
    source_h, source_w = source_hw
    target_h, target_w = target_hw
    scaled = box.clone()
    scaled[[0, 2]] *= float(target_w) / float(source_w)
    scaled[[1, 3]] *= float(target_h) / float(source_h)
    return _clamp_box(scaled, target_h, target_w)


def _crop(image: Image.Image, box: torch.Tensor) -> Image.Image:
    x1, y1, x2, y2 = box.round().to(torch.int64).tolist()
    return image.crop((x1, y1, x2, y2)).convert("RGB")


class RADARDataset(Dataset):
    """A single dataset split with cached text lookup and optional validation."""

    def __init__(
        self,
        split_root: str | Path,
        text_map: Mapping[str, str],
        tokenizer: Callable,
        *,
        is_train: bool,
        max_text_length: int,
        validate_data: bool = False,
        sam_size: int = 512,
        clip_size: int = 224,
        mask_size: int = 224,
    ) -> None:
        self.root = Path(split_root)
        self.image_directory = self.root / "images"
        self.mask_directory = self.root / "masks"
        self.tokenizer = tokenizer
        self.is_train = bool(is_train)
        self.max_text_length = int(max_text_length)
        self.sam_size = int(sam_size)
        self.clip_size = int(clip_size)
        self.mask_size = int(mask_size)

        text_config = getattr(tokenizer, "pad_token_id", None)
        self.pad_token_id = int(0 if text_config is None else text_config)
        # Filename and case normalization are cached once, never rebuilt per sample.
        self.text_map_lower = {
            str(name).strip().lower(): str(description).splitlines()[0].strip()
            for name, description in text_map.items()
        }

        self.to_sam = build_resize_to_tensor(self.sam_size)
        self.to_clip = build_clip_preprocess(self.clip_size)
        self.samples = self._build_index()
        if validate_data:
            self._validate_files()

    def _build_index(self) -> List[Tuple[Path, Path, str]]:
        if not self.image_directory.is_dir():
            raise FileNotFoundError(
                f"Image directory not found: {self.image_directory}"
            )
        if not self.mask_directory.is_dir():
            raise FileNotFoundError(f"Mask directory not found: {self.mask_directory}")
        masks = sorted(
            path
            for path in self.mask_directory.iterdir()
            if path.suffix.lower() in IMAGE_EXTENSIONS
        )
        samples = []
        for mask_path in masks:
            image_name = (
                mask_path.name[5:]
                if mask_path.name.lower().startswith("mask_")
                else mask_path.name
            )
            image_path = self.image_directory / image_name
            if image_path.is_file():
                samples.append((image_path, mask_path, image_name))
        if not samples:
            raise RuntimeError(f"No image-mask pairs found in {self.root}")
        return samples

    def _validate_files(self) -> None:
        # Expensive full-file verification is explicitly opt-in.
        for image_path, mask_path, _ in self.samples:
            for path in (image_path, mask_path):
                try:
                    with Image.open(path) as image:
                        image.verify()
                except Exception as error:
                    raise ValueError(f"Corrupted image file: {path}") from error

    def _lookup_text(self, image_name: str, mask_name: str) -> str:
        for candidate in (
            image_name,
            mask_name,
            Path(image_name).stem,
            Path(mask_name).stem,
        ):
            value = self.text_map_lower.get(candidate.lower())
            if value:
                return value
        raise KeyError(f"Missing text annotation for {image_name}")

    def _sdra_groups(self, mask_clip: torch.Tensor) -> List[torch.Tensor]:
        foreground = mask_clip[0].ge(0.5)
        if not torch.any(foreground):
            return []
        merged = F.max_pool2d(
            foreground.float()[None, None],
            kernel_size=SDRA_GROUP_MERGE_KERNEL,
            stride=1,
            padding=SDRA_GROUP_MERGE_KERNEL // 2,
        )[0, 0].gt(0.5)
        labels_array, count = ndimage.label(merged.numpy().astype(np.uint8))
        labels = torch.from_numpy(labels_array)
        groups = [foreground & labels.eq(index) for index in range(1, count + 1)]
        groups = [
            group for group in groups if int(group.sum()) >= SDRA_MIN_GROUP_PIXELS
        ]
        return sorted(groups, key=lambda group: int(group.sum()), reverse=True)

    def _build_sdra_regions(
        self,
        image: Image.Image,
        mask: Image.Image,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        original_width, original_height = image.size
        mask_clip = binary_mask(mask, self.clip_size)
        groups = self._sdra_groups(mask_clip)
        if not groups:
            empty_crops = torch.empty(
                (0, 3, 3, self.clip_size, self.clip_size), dtype=torch.float32
            )
            empty_weights = torch.empty(
                (0, 3, self.clip_size, self.clip_size), dtype=torch.float32
            )
            return empty_crops, empty_weights, torch.empty((0,), dtype=torch.bool)

        crops, weights = [], []
        for target_region in groups:
            target_box = _expand_box(
                _mask_box(target_region),
                SDRA_TARGET_EXPANSION,
                self.clip_size,
                self.clip_size,
            )
            boundary_box = _expand_box(
                target_box,
                SDRA_BOUNDARY_EXPANSION,
                self.clip_size,
                self.clip_size,
            )
            context_box = _expand_box(
                target_box,
                SDRA_CONTEXT_EXPANSION,
                self.clip_size,
                self.clip_size,
            )

            target_box_mask = _box_mask(target_box, self.clip_size, self.clip_size)
            boundary_box_mask = _box_mask(boundary_box, self.clip_size, self.clip_size)
            context_box_mask = _box_mask(context_box, self.clip_size, self.clip_size)
            target_weight = target_region.float()
            boundary_weight = (boundary_box_mask & ~target_box_mask).float()
            context_weight = (context_box_mask & ~boundary_box_mask).float()
            if any(
                int(region.sum()) == 0
                for region in (target_weight, boundary_weight, context_weight)
            ):
                continue

            boxes_original = [
                _scale_box(
                    box,
                    (self.clip_size, self.clip_size),
                    (original_height, original_width),
                )
                for box in (target_box, boundary_box, context_box)
            ]
            crops.append(
                torch.stack([self.to_clip(_crop(image, box)) for box in boxes_original])
            )
            weights.append(
                torch.stack((target_weight, boundary_weight, context_weight))
            )

        if not crops:
            empty_crops = torch.empty(
                (0, 3, 3, self.clip_size, self.clip_size), dtype=torch.float32
            )
            empty_weights = torch.empty(
                (0, 3, self.clip_size, self.clip_size), dtype=torch.float32
            )
            return empty_crops, empty_weights, torch.empty((0,), dtype=torch.bool)
        crop_tensor = torch.stack(crops)
        weight_tensor = torch.stack(weights)
        return crop_tensor, weight_tensor, torch.ones((len(crops),), dtype=torch.bool)

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, index: int) -> Dict[str, object]:
        image_path, mask_path, image_name = self.samples[index]
        with Image.open(image_path) as image_file, Image.open(mask_path) as mask_file:
            image = image_file.convert("RGB")
            mask = mask_file.convert("L")
            if self.is_train:
                image, mask = _random_geometric_augmentation(image, mask)

            image_sam = sam2_normalize(self.to_sam(image))
            image_clip = self.to_clip(image)
            target = binary_mask(mask, self.mask_size)
            if self.is_train:
                sdra_crops, sdra_weights, sdra_valid = self._build_sdra_regions(
                    image, mask
                )
            else:
                sdra_crops = sdra_weights = sdra_valid = None

        text = self._lookup_text(image_name, mask_path.name)
        tokenized = self.tokenizer(text)
        tokens = (
            tokenized[0]
            if torch.is_tensor(tokenized) and tokenized.ndim > 1
            else tokenized
        )
        tokens = pad_or_truncate(tokens, self.max_text_length, self.pad_token_id)
        return {
            "image_sam": image_sam,
            "image_clip": image_clip,
            "tokens": tokens,
            "mask": target,
            "sdra_teacher_crops": sdra_crops,
            "sdra_region_weights": sdra_weights,
            "sdra_valid_mask": sdra_valid,
            "filename": image_name,
            "text": text,
        }


def radar_collate_fn(batch: List[Dict[str, object]]) -> Dict[str, object]:
    output: Dict[str, object] = {
        key: torch.stack([sample[key] for sample in batch])
        for key in ("image_sam", "image_clip", "tokens", "mask")
    }
    output["filename"] = [sample["filename"] for sample in batch]
    output["text"] = [sample["text"] for sample in batch]

    crops = [sample["sdra_teacher_crops"] for sample in batch]
    if all(value is None for value in crops):
        output["sdra_teacher_crops"] = None
        output["sdra_region_weights"] = None
        output["sdra_valid_mask"] = None
        return output

    maximum_groups = max(int(value.shape[0]) for value in crops if value is not None)
    batch_size = len(batch)
    clip_size = int(batch[0]["image_clip"].shape[-1])
    padded_crops = torch.zeros((batch_size, maximum_groups, 3, 3, clip_size, clip_size))
    padded_weights = torch.zeros((batch_size, maximum_groups, 3, clip_size, clip_size))
    padded_valid = torch.zeros((batch_size, maximum_groups), dtype=torch.bool)
    for batch_index, sample in enumerate(batch):
        sample_crops = sample["sdra_teacher_crops"]
        if sample_crops is None or sample_crops.shape[0] == 0:
            continue
        count = sample_crops.shape[0]
        padded_crops[batch_index, :count] = sample_crops
        padded_weights[batch_index, :count] = sample["sdra_region_weights"]
        padded_valid[batch_index, :count] = sample["sdra_valid_mask"]
    output["sdra_teacher_crops"] = padded_crops
    output["sdra_region_weights"] = padded_weights
    output["sdra_valid_mask"] = padded_valid
    return output


def build_dataloader(
    split_root: str | Path,
    text_map: Mapping[str, str],
    tokenizer: Callable,
    *,
    is_train: bool,
    max_text_length: int,
    batch_size: int,
    num_workers: int,
    validate_data: bool = False,
) -> DataLoader:
    dataset = RADARDataset(
        split_root,
        text_map,
        tokenizer,
        is_train=is_train,
        max_text_length=max_text_length,
        validate_data=validate_data,
    )
    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=is_train,
        num_workers=num_workers,
        pin_memory=True,
        persistent_workers=num_workers > 0,
        collate_fn=radar_collate_fn,
    )


__all__ = [
    "RADARDataset",
    "build_dataloader",
    "load_text_annotations",
    "radar_collate_fn",
]
