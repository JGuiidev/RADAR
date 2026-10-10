"""Standalone preflight validation for RADAR datasets."""

from __future__ import annotations

import argparse
import sys
from collections import Counter
from pathlib import Path
from typing import Dict, List, Tuple

import pandas as pd
from PIL import Image

IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff", ".webp"}
SPLITS = ("Train", "Val", "Test")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Validate a RADAR dataset before training or testing."
    )
    parser.add_argument(
        "dataset_root",
        type=Path,
        nargs="?",
        default=Path("data/QaTa-COV19"),
        help="Dataset directory containing Train_Folder, Val_Folder, and Test_Folder.",
    )
    return parser.parse_args()


def _files(directory: Path) -> List[Path]:
    return sorted(
        path
        for path in directory.iterdir()
        if path.is_file() and path.suffix.lower() in IMAGE_EXTENSIONS
    )


def _duplicates(paths: List[Path]) -> List[str]:
    counts = Counter(path.name.lower() for path in paths)
    return sorted(name for name, count in counts.items() if count > 1)


def _verify_image(path: Path) -> str | None:
    try:
        with Image.open(path) as image:
            image.verify()
        return None
    except Exception as error:
        return f"Corrupted file {path}: {error}"


def _read_annotations(path: Path) -> Tuple[Dict[str, str], List[str]]:
    errors: List[str] = []
    if not path.is_file():
        return {}, [f"Missing text annotation file: {path}"]
    try:
        frame = pd.read_excel(path, engine="openpyxl")
    except Exception as error:
        return {}, [f"Malformed annotation file {path}: {error}"]
    required = {"Image", "Description"}
    missing_columns = required.difference(frame.columns)
    if missing_columns:
        return {}, [
            f"Malformed annotation file {path}: missing columns {sorted(missing_columns)}"
        ]

    annotations: Dict[str, str] = {}
    names: List[str] = []
    for row_number, row in frame.iterrows():
        image_value = row["Image"]
        if pd.isna(image_value) or not str(image_value).strip():
            errors.append(
                f"Malformed annotation at row {row_number + 2} in {path}: empty Image value"
            )
            continue
        name = str(image_value).strip().lower()
        names.append(name)
        description = row["Description"]
        if pd.isna(description) or not str(description).strip():
            errors.append(f"Missing text description for {image_value} in {path}")
            continue
        annotations[name] = str(description).strip()
    duplicate_names = sorted(
        name for name, count in Counter(names).items() if count > 1
    )
    if duplicate_names:
        errors.append(f"Duplicate annotation filenames in {path}: {duplicate_names}")
    return annotations, errors


def check_split(dataset_root: Path, split: str) -> Tuple[int, List[str]]:
    split_root = dataset_root / f"{split}_Folder"
    image_directory = split_root / "images"
    mask_directory = split_root / "masks"
    annotation_path = split_root / f"{split}_text.xlsx"
    errors: List[str] = []
    for label, directory in (("image", image_directory), ("mask", mask_directory)):
        if not directory.is_dir():
            errors.append(f"Missing {label} directory: {directory}")
    if errors:
        return 0, errors

    images = _files(image_directory)
    masks = _files(mask_directory)
    if not images:
        errors.append(f"Empty split: no images in {image_directory}")
    if not masks:
        errors.append(f"Empty split: no masks in {mask_directory}")
    for label, paths in (("image", images), ("mask", masks)):
        duplicates = _duplicates(paths)
        if duplicates:
            errors.append(f"Duplicate {label} filenames in {split}: {duplicates}")

    image_by_name = {path.name.lower(): path for path in images}
    paired_image_names = set()
    for mask_path in masks:
        expected = (
            mask_path.name[5:]
            if mask_path.name.lower().startswith("mask_")
            else mask_path.name
        )
        image_path = image_by_name.get(expected.lower())
        if image_path is None:
            errors.append(f"Missing image for mask: {mask_path}")
        else:
            paired_image_names.add(image_path.name.lower())
    for image_path in images:
        if image_path.name.lower() not in paired_image_names:
            errors.append(f"Missing mask for image: {image_path}")

    annotations, annotation_errors = _read_annotations(annotation_path)
    errors.extend(annotation_errors)
    annotation_keys = set(annotations)
    annotation_stems = {Path(name).stem for name in annotation_keys}
    for image_path in images:
        name = image_path.name.lower()
        if (
            name not in annotation_keys
            and image_path.stem.lower() not in annotation_stems
        ):
            errors.append(f"Missing text annotation for image: {image_path}")

    for path in images + masks:
        error = _verify_image(path)
        if error:
            errors.append(error)
    return len(paired_image_names), errors


def main() -> int:
    args = parse_args()
    dataset_root = args.dataset_root
    if not dataset_root.is_dir():
        print(f"ERROR: Dataset root not found: {dataset_root}")
        return 1

    all_errors: List[str] = []
    print(f"Dataset preflight: {dataset_root}")
    for split in SPLITS:
        count, errors = check_split(dataset_root, split)
        print(f"{split}: {count} paired samples")
        all_errors.extend(errors)
    if all_errors:
        print(f"Preflight failed with {len(all_errors)} error(s):")
        for error in all_errors:
            print(f"  - {error}")
        return 1
    print(
        "Preflight passed: directory structure, files, pairs, and annotations are valid."
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
