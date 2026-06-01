"""
preprocess_asl_alphabet.py
===========================

Convert the **ASL Alphabet** image dataset (29 STATIC classes: A-Z, ``del``,
``nothing``, ``space``) into a compact ``.npz`` file of RAW MediaPipe hand
landmarks suitable for training the static MLP classifier.

What it does
------------
1. Walks the sorted class folders under ``config.ASL_TRAIN_DIR``.
2. For each image it runs MediaPipe Hands (``static_image_mode=True``,
   ``max_num_hands=1``) and extracts the RAW 63-vector
   ``[x, y, z] * 21`` of the first detected hand.
3. Images where no hand is detected are skipped and counted per class.
4. Saves an ``.npz`` containing:
     * ``X``       -- float32 array of shape ``(N, 63)`` of RAW landmarks,
     * ``y``       -- int64 array of class indices,
     * ``classes`` -- object array of class-name strings (index -> name).

The data stored is RAW (un-normalized): normalization is applied later, in the
training ``Dataset`` and in the inference predictors, so there is exactly one
normalization implementation in the project.

This script NEVER touches a webcam -- it reads images from disk only.

Run from the project root, e.g.::

    python -m src.preprocessing.preprocess_asl_alphabet
    python -m src.preprocessing.preprocess_asl_alphabet --max-per-class 1500
"""

# ---------------------------------------------------------------------------
# sys.path bootstrap: insert the PROJECT ROOT (two levels up from this file:
# src/preprocessing/ -> src/ -> ROOT) onto sys.path so that "import config" and
# "from src.common... import ..." both resolve regardless of how the script is
# launched.  Computed from __file__, never hardcoded.
# ---------------------------------------------------------------------------
import os
import sys

_PROJECT_ROOT = os.path.abspath(
    os.path.join(os.path.dirname(__file__), os.pardir, os.pardir)
)
if _PROJECT_ROOT not in sys.path:
    sys.path.insert(0, _PROJECT_ROOT)

import argparse

import numpy as np
import cv2
from tqdm import tqdm

import config
from src.common.landmarks import make_hands_detector, extract_hand_landmarks_from_image
from src.common.utils import get_logger, ensure_dir

logger = get_logger("preprocess_asl_alphabet")

# Image file extensions we will attempt to read inside each class folder.
_IMAGE_EXTENSIONS = (".jpg", ".jpeg", ".png", ".bmp")


def list_class_folders(train_dir):
    """Return the sorted list of class-folder names under ``train_dir``.

    Parameters
    ----------
    train_dir : str
        Path to the ASL alphabet training directory.

    Returns
    -------
    list of str
        Sorted subdirectory names (each one a class label such as ``"A"`` or
        ``"space"``).
    """
    if not os.path.isdir(train_dir):
        raise FileNotFoundError(
            "ASL train directory not found: %s\n"
            "Expected the layout "
            "datasets/asl/asl_alphabet_train/asl_alphabet_train/<CLASS>/*.jpg"
            % train_dir
        )
    entries = [
        name for name in os.listdir(train_dir)
        if os.path.isdir(os.path.join(train_dir, name))
    ]
    return sorted(entries)


def list_images(class_dir):
    """Return the sorted list of image file paths inside ``class_dir``.

    Parameters
    ----------
    class_dir : str
        Path to a single class folder.

    Returns
    -------
    list of str
        Sorted absolute image paths with a recognized extension.
    """
    files = [
        os.path.join(class_dir, name)
        for name in sorted(os.listdir(class_dir))
        if name.lower().endswith(_IMAGE_EXTENSIONS)
    ]
    return files


def preprocess(max_per_class, output_path, min_detection_confidence):
    """Run the full ASL-alphabet preprocessing pipeline.

    Parameters
    ----------
    max_per_class : int
        Maximum number of images to *attempt* per class (kept may be fewer when
        some images contain no detectable hand).
    output_path : str
        Destination ``.npz`` path.
    min_detection_confidence : float
        MediaPipe minimum detection confidence.
    """
    classes = list_class_folders(config.ASL_TRAIN_DIR)
    logger.info("Found %d ASL classes: %s", len(classes), ", ".join(classes))

    features = []           # list of (63,) float32 arrays
    labels = []             # list of int class indices
    kept_per_class = {c: 0 for c in classes}
    skipped_per_class = {c: 0 for c in classes}

    # One detector reused across all images (static_image_mode=True means each
    # image is treated independently, so reuse is safe and fast).
    hands = make_hands_detector(
        static_image_mode=True,
        max_num_hands=1,
        min_detection_confidence=min_detection_confidence,
    )

    try:
        for class_idx, class_name in enumerate(classes):
            class_dir = os.path.join(config.ASL_TRAIN_DIR, class_name)
            image_paths = list_images(class_dir)
            if max_per_class is not None and max_per_class > 0:
                image_paths = image_paths[:max_per_class]

            for img_path in tqdm(image_paths,
                                 desc="[%2d/%2d] %s" % (class_idx + 1,
                                                        len(classes),
                                                        class_name),
                                 unit="img"):
                image = cv2.imread(img_path)
                if image is None:
                    # Unreadable / corrupt file -- count as a skip.
                    skipped_per_class[class_name] += 1
                    continue

                landmarks = extract_hand_landmarks_from_image(image, hands)
                if landmarks is None:
                    skipped_per_class[class_name] += 1
                    continue

                features.append(landmarks.astype(np.float32))
                labels.append(class_idx)
                kept_per_class[class_name] += 1
    finally:
        # Always release MediaPipe resources.
        hands.close()

    if not features:
        raise RuntimeError(
            "No hand landmarks were extracted from ANY image. "
            "Check the dataset path and that MediaPipe is working."
        )

    X = np.stack(features).astype(np.float32)          # (N, 63)
    y = np.asarray(labels, dtype=np.int64)             # (N,)
    classes_arr = np.array(classes, dtype=object)      # (num_classes,)

    ensure_dir(os.path.dirname(os.path.abspath(output_path)))
    np.savez(output_path, X=X, y=y, classes=classes_arr)

    # ----------------------- Summary report --------------------------------
    total_kept = int(sum(kept_per_class.values()))
    total_skipped = int(sum(skipped_per_class.values()))
    logger.info("=" * 60)
    logger.info("Saved RAW landmarks to: %s", os.path.abspath(output_path))
    logger.info("X shape: %s (dtype=%s)", X.shape, X.dtype)
    logger.info("y shape: %s (dtype=%s)", y.shape, y.dtype)
    logger.info("Total kept: %d   Total skipped (no hand / unreadable): %d",
                total_kept, total_skipped)
    logger.info("-" * 60)
    logger.info("Per-class kept / skipped:")
    for class_name in classes:
        logger.info("  %-8s kept=%-6d skipped=%-6d",
                    class_name, kept_per_class[class_name],
                    skipped_per_class[class_name])
    logger.info("=" * 60)


def build_arg_parser():
    """Construct the command-line argument parser for this script.

    Returns
    -------
    argparse.ArgumentParser
        The configured parser.
    """
    parser = argparse.ArgumentParser(
        description="Preprocess the ASL alphabet image dataset into a .npz of "
                    "RAW MediaPipe hand landmarks (63 floats per image)."
    )
    parser.add_argument(
        "--max-per-class", type=int, default=1500,
        help="Maximum images to process per class (default: 1500). "
             "Use a non-positive value to process all available images.",
    )
    parser.add_argument(
        "--output", type=str, default=config.ASL_LANDMARKS_NPZ,
        help="Output .npz path (default: config.ASL_LANDMARKS_NPZ).",
    )
    parser.add_argument(
        "--min-detection-confidence", type=float, default=0.5,
        help="MediaPipe minimum detection confidence (default: 0.5).",
    )
    return parser


def main():
    """Parse arguments and run the preprocessing pipeline."""
    args = build_arg_parser().parse_args()
    logger.info("Configuration:")
    logger.info("  max_per_class            = %s", args.max_per_class)
    logger.info("  output                   = %s", args.output)
    logger.info("  min_detection_confidence = %s", args.min_detection_confidence)
    preprocess(
        max_per_class=args.max_per_class,
        output_path=args.output,
        min_detection_confidence=args.min_detection_confidence,
    )


if __name__ == "__main__":
    main()
