from __future__ import annotations

from pathlib import Path
from typing import Iterable

import cv2
import numpy as np

from .landmarks import extract_landmarks_from_bgr


def list_image_files(directory: Path) -> Iterable[Path]:
    patterns = ("*.jpg", "*.jpeg", "*.png", "*.bmp")
    for pattern in patterns:
        yield from directory.glob(pattern)


def simple_augmentations(image: np.ndarray) -> list[np.ndarray]:
    return [
        image,
        cv2.flip(image, 1),
        cv2.convertScaleAbs(image, alpha=1.0, beta=15),
    ]


def load_dataset(dataset_path: str | Path) -> tuple[np.ndarray, np.ndarray, list[str]]:
    root = Path(dataset_path)
    labels = sorted([d.name for d in root.iterdir() if d.is_dir()])
    if not labels:
        raise ValueError(f"No class directories found under {root}")

    features, targets = [], []
    for class_index, label in enumerate(labels):
        class_dir = root / label
        for image_path in list_image_files(class_dir):
            image = cv2.imread(str(image_path))
            if image is None:
                continue
            for augmented in simple_augmentations(image):
                result = extract_landmarks_from_bgr(augmented)
                if result is None:
                    continue
                features.append(result.feature_vector)
                targets.append(class_index)

    if not features:
        raise ValueError("No valid hand landmarks extracted from dataset.")

    return np.asarray(features), np.asarray(targets), labels
