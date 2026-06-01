"""
static_predictor.py
===================

Inference wrapper for the *static* (single-frame) ASL alphabet model.

This module exposes a small, dependency-light :class:`StaticPredictor` class
that loads a trained :class:`StaticSignMLP` from disk and turns a *raw*
(un-normalized) 63-value hand-landmark vector into a human-readable prediction.

Design notes
------------
* The model ALWAYS sees *normalized* input.  The frontend, the webcam demo and
  the Flask backend all send RAW landmarks; normalization happens here (and in
  the training Datasets) using the single source of truth in
  ``src.common.landmarks.normalize_hand``.  This guarantees train/inference
  parity.
* A 63-vector is ``[x, y, z]`` for each of the 21 MediaPipe hand landmarks,
  flattened in landmark order (see :mod:`config`).

The class is intentionally self-contained so it can be reused by the Flask
backend (``backend/model_manager.py``), the standalone webcam demo
(``src/inference/webcam_demo.py``) and any test scripts.

Run directly (after the ``sys.path`` bootstrap below) to prove the model loads::

    python -m src.inference.static_predictor
    python src/inference/static_predictor.py
"""

# ---------------------------------------------------------------------------
# sys.path bootstrap
# ---------------------------------------------------------------------------
# Every runnable entry script inserts the PROJECT ROOT onto sys.path so that
# both ``import config`` and ``from src.common.landmarks import ...`` resolve,
# whether the file is launched as ``python -m src.inference.static_predictor``
# or ``python src/inference/static_predictor.py``.  The root is computed from
# ``__file__`` (this file lives at <ROOT>/src/inference/static_predictor.py, so
# the root is three directories up) and is NEVER hardcoded.
import os
import sys

_THIS_DIR = os.path.dirname(os.path.abspath(__file__))
_PROJECT_ROOT = os.path.abspath(os.path.join(_THIS_DIR, os.pardir, os.pardir))
if _PROJECT_ROOT not in sys.path:
    sys.path.insert(0, _PROJECT_ROOT)

# ---------------------------------------------------------------------------
# Standard / third-party imports
# ---------------------------------------------------------------------------
import numpy as np
import torch
import torch.nn.functional as F

# ---------------------------------------------------------------------------
# Project imports (resolve thanks to the bootstrap above)
# ---------------------------------------------------------------------------
import config
from src.common.landmarks import normalize_hand
from src.models.static_model import load_static_model


class StaticPredictor:
    """Load a trained static MLP and classify single-frame hand landmarks.

    Parameters
    ----------
    model_path : str, optional
        Path to the saved checkpoint produced by training.  Defaults to
        :data:`config.STATIC_MODEL_PATH`.  The checkpoint is the symmetric
        dictionary ``{"state_dict", "arch", "classes"}`` understood by
        :func:`src.models.static_model.load_static_model`.
    device : torch.device or str, optional
        Device to run inference on.  When ``None`` (the default) the device is
        chosen by :func:`config.get_device` (CUDA when available, else CPU).

    Raises
    ------
    FileNotFoundError
        If ``model_path`` does not exist.  The message points the user at the
        training script so the failure is self-explanatory.
    """

    def __init__(self, model_path: str = config.STATIC_MODEL_PATH, device=None):
        # Resolve the device first so a clear value is always available.
        if device is None:
            self.device = config.get_device()
        elif isinstance(device, str):
            self.device = torch.device(device)
        else:
            self.device = device

        self.model_path = model_path

        # Fail loudly and helpfully when the checkpoint is missing rather than
        # letting torch raise a cryptic error deep in the load call.
        if not os.path.isfile(self.model_path):
            raise FileNotFoundError(
                "Static model checkpoint not found at '{}'. "
                "Train it first with: python -m src.training.train_static".format(
                    self.model_path
                )
            )

        # ``load_static_model`` rebuilds the module from the saved ``arch`` dict,
        # loads the weights, moves it to ``device`` and returns the class list.
        self.model, self.classes = load_static_model(self.model_path, self.device)

        # Inference mode: disable dropout / use running BatchNorm statistics.
        self.model.eval()

    # ------------------------------------------------------------------ #
    # Public API
    # ------------------------------------------------------------------ #
    @property
    def is_available(self) -> bool:
        """``True`` when a model is loaded and ready to predict.

        Kept as a property so callers (e.g. the Flask backend) can probe
        readiness without inspecting internal attributes.
        """
        return self.model is not None and len(self.classes) > 0

    def predict(self, raw_landmarks_63) -> dict:
        """Classify a single RAW 63-value hand-landmark vector.

        Parameters
        ----------
        raw_landmarks_63 : array-like
            63 RAW (un-normalized, image-coordinate) floats laid out as
            ``[x, y, z]`` for each of the 21 landmarks.  Accepts a Python list,
            tuple or NumPy array.

        Returns
        -------
        dict
            ``{"prediction": str, "confidence": float,
               "top_k": [{"label": str, "confidence": float}, ... up to 3]}``

        Raises
        ------
        ValueError
            If the input cannot be coerced to exactly
            :data:`config.STATIC_FEATURE_DIM` (63) float values.
        """
        # --- Validate / coerce the input ---------------------------------
        arr = np.asarray(raw_landmarks_63, dtype=np.float32).reshape(-1)
        if arr.shape[0] != config.STATIC_FEATURE_DIM:
            raise ValueError(
                "Expected {} landmark values, got {}.".format(
                    config.STATIC_FEATURE_DIM, arr.shape[0]
                )
            )

        # --- Normalize (single source of truth) --------------------------
        # Translation + scale invariant so the model is robust to where the
        # hand sits in frame and how big it appears.
        normalized = normalize_hand(arr)  # -> np.float32 (63,)

        # --- Forward pass -------------------------------------------------
        # Shape (1, 63); models expect a batch dimension.
        x = torch.from_numpy(normalized).float().unsqueeze(0).to(self.device)
        with torch.no_grad():
            logits = self.model(x)                 # (1, num_classes)
            probs = F.softmax(logits, dim=1)[0]    # (num_classes,)

        return self._format_result(probs)

    # ------------------------------------------------------------------ #
    # Internal helpers
    # ------------------------------------------------------------------ #
    def _format_result(self, probs: torch.Tensor) -> dict:
        """Turn a probability tensor into the API-shaped result dictionary."""
        probs_np = probs.detach().cpu().numpy()

        # Top class.
        best_idx = int(np.argmax(probs_np))
        prediction = self.classes[best_idx]
        confidence = float(probs_np[best_idx])

        # Top-k (up to 3) sorted by descending probability.
        k = min(3, len(self.classes))
        top_indices = np.argsort(probs_np)[::-1][:k]
        top_k = [
            {"label": self.classes[int(i)], "confidence": float(probs_np[int(i)])}
            for i in top_indices
        ]

        return {
            "prediction": prediction,
            "confidence": confidence,
            "top_k": top_k,
        }


# ---------------------------------------------------------------------------
# Self-test / demonstration entry point
# ---------------------------------------------------------------------------
def _main() -> None:
    """Load the model and predict on a random vector to prove the pipeline.

    The random vector is meaningless sign-language-wise; this only confirms the
    checkpoint loads and a forward pass runs end-to-end.
    """
    print("StaticPredictor self-test")
    print("-" * 40)
    print("Model path : {}".format(config.STATIC_MODEL_PATH))

    try:
        predictor = StaticPredictor()
    except FileNotFoundError as exc:
        # Friendly message instead of a traceback when the model is untrained.
        print("ERROR: {}".format(exc))
        return

    print("Device     : {}".format(predictor.device))
    print("Num classes: {}".format(len(predictor.classes)))

    # Deterministic random input so repeated runs are reproducible.
    rng = np.random.default_rng(seed=0)
    sample = rng.random(config.STATIC_FEATURE_DIM).astype(np.float32)

    result = predictor.predict(sample)
    print("\nPrediction (on synthetic random landmarks; meaningless content):")
    print("  prediction : {}".format(result["prediction"]))
    print("  confidence : {:.4f}".format(result["confidence"]))
    print("  top_k      :")
    for entry in result["top_k"]:
        print("    {:<10s} {:.4f}".format(entry["label"], entry["confidence"]))


if __name__ == "__main__":
    _main()
