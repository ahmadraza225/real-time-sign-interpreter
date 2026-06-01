"""
config.py
=========

Central configuration module for the **Real-Time Sign Language Interpreter**
project. This file is the *single source of truth* for every shared constant
used across preprocessing, training, inference, and the backend API.

Design rules (important!):
---------------------------
* **Never redefine** these constants anywhere else. Always `import config`
  and reference, e.g., ``config.STATIC_FEATURE_DIM``.
* All path constants are **absolute** and are built with ``os.path.join`` from
  ``BASE_DIR`` (the directory that contains *this* file). This keeps the
  project portable: clone it anywhere and the paths still resolve.
* ``torch`` is imported lazily inside :func:`get_device` so that lightweight
  scripts (for example a documentation generator or a pure-numpy utility) can
  ``import config`` without paying the cost of importing PyTorch.

Author: Sign Language Interpreter Team
"""

import os

# ---------------------------------------------------------------------------
# Base directory
# ---------------------------------------------------------------------------
# BASE_DIR is the absolute path of the folder that holds this config.py file.
# Because every other path is derived from it, the project can live anywhere
# on disk and still work without edits.
BASE_DIR = os.path.dirname(os.path.abspath(__file__))


# ===========================================================================
# 1. LANDMARK / FEATURE DIMENSIONS
# ===========================================================================
# MediaPipe Hands returns 21 landmarks per hand, each with (x, y, z).
NUM_HAND_LANDMARKS = 21          # number of landmark points per detected hand
LANDMARK_DIMS = 3                # each landmark is (x, y, z)

# ---- Static (alphabet) model: a single hand ----
# 21 landmarks * 3 dims = 63 features for one hand.
STATIC_FEATURE_DIM = NUM_HAND_LANDMARKS * LANDMARK_DIMS  # = 63

# ---- Dynamic (word/gloss) model: up to two hands ----
MAX_HANDS_DYNAMIC = 2            # word-level signs may use both hands
# 2 hands * 63 = 126 features per frame.
DYNAMIC_FEATURE_DIM = MAX_HANDS_DYNAMIC * STATIC_FEATURE_DIM  # = 126

# Every video clip is normalized (padded or truncated) to a fixed number of
# frames so the LSTM always receives sequences of identical length.
SEQUENCE_LENGTH = 30             # frames per dynamic sample (pad/truncate)

# How many word-level glosses to use by default. The WLASL set has ~2000
# glosses but only a handful of videos each, so we keep a demo-scale subset
# ranked by number of available videos.
WLASL_DEFAULT_NUM_GLOSSES = 20


# ===========================================================================
# 2. DATASET PATHS (input data already on disk)
# ===========================================================================
# Root folder that holds all raw datasets.
DATASETS_DIR = os.path.join(BASE_DIR, "datasets")

# ---- ASL Alphabet (STATIC images) ----
# Training images: ASL_TRAIN_DIR/<CLASS>/<image>.jpg
#   29 classes: A..Z, del, nothing, space  (~3000 images per class)
ASL_TRAIN_DIR = os.path.join(
    DATASETS_DIR, "asl", "asl_alphabet_train", "asl_alphabet_train"
)
# Test images: ASL_TEST_DIR/<LETTER>_test.jpg
ASL_TEST_DIR = os.path.join(
    DATASETS_DIR, "asl", "asl_alphabet_test", "asl_alphabet_test"
)

# ---- WLASL word-level videos (DYNAMIC) ----
# Layout: WLASL_ROOT/<gloss>/<video_id>.mp4
# The GLOSS is the folder name; there is NO json metadata file.
WLASL_ROOT = os.path.join(DATASETS_DIR, "archive", "dataset", "SL")


# ===========================================================================
# 3. OUTPUT PATHS (generated artifacts)
# ===========================================================================
# Top-level data folder for everything the project produces.
DATA_DIR = os.path.join(BASE_DIR, "data")
# Processed (preprocessed) feature arrays live here.
PROCESSED_DIR = os.path.join(DATA_DIR, "processed")
# Trained model checkpoints live here.
MODELS_DIR = os.path.join(BASE_DIR, "models")

# ---- Preprocessed feature archives (.npz) ----
# Static: extracted hand landmarks for the ASL alphabet.
ASL_LANDMARKS_NPZ = os.path.join(PROCESSED_DIR, "asl_landmarks.npz")
# Dynamic: per-frame two-hand landmark sequences for the WLASL subset.
WLASL_SEQ_NPZ = os.path.join(PROCESSED_DIR, "wlasl_sequences.npz")

# ---- Trained model checkpoints + label files ----
# Static MLP (alphabet recognition).
STATIC_MODEL_PATH = os.path.join(MODELS_DIR, "static_mlp.pt")
STATIC_LABELS_PATH = os.path.join(MODELS_DIR, "static_labels.json")
# Dynamic LSTM (word/gloss recognition).
DYNAMIC_MODEL_PATH = os.path.join(MODELS_DIR, "dynamic_lstm.pt")
DYNAMIC_LABELS_PATH = os.path.join(MODELS_DIR, "dynamic_labels.json")


# ===========================================================================
# 4. DEVICE SELECTION
# ===========================================================================
# DEVICE is a plain string ("cuda" or "cpu") that can be referenced without
# importing torch. We detect CUDA availability defensively: if torch is not
# installed or errors during import, we transparently fall back to "cpu".
try:
    import torch as _torch  # imported here only to probe CUDA availability

    DEVICE = "cuda" if _torch.cuda.is_available() else "cpu"
except Exception:  # torch missing, broken install, or no driver
    DEVICE = "cpu"


def get_device():
    """Return a ``torch.device`` for the best available compute backend.

    PyTorch is imported lazily *inside* this function so that modules which
    only need the configuration constants (paths, dimensions) do not have to
    import the heavy ``torch`` package.

    Returns
    -------
    torch.device
        ``torch.device("cuda")`` when a CUDA-capable GPU is available,
        otherwise ``torch.device("cpu")``.
    """
    import torch  # local import keeps torch optional for non-ML callers

    return torch.device("cuda" if torch.cuda.is_available() else "cpu")
