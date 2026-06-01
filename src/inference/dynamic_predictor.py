"""
dynamic_predictor.py
====================

Inference wrapper for the *dynamic* (sequence / word-level) sign model.

This module exposes :class:`DynamicPredictor`, which loads a trained
:class:`DynamicSignLSTM` and classifies a *variable-length* sequence of RAW
126-value two-hand landmark frames into a single WLASL gloss.

Pipeline (mirrors training so inference matches what the model learnt)
----------------------------------------------------------------------
1. Coerce the input to a ``(T, 126)`` float32 array.
2. Pad / truncate to exactly :data:`config.SEQUENCE_LENGTH` frames
   (:func:`pad_or_truncate_sequence`).
3. Normalize every frame with the single source of truth
   :func:`src.common.landmarks.normalize_sequence`.
4. Run the LSTM and softmax the logits.

A 126-vector is two concatenated 63-value hands (Left block first, Right block
second); see :mod:`config` and ``src.common.landmarks``.

Run directly to prove the model loads::

    python -m src.inference.dynamic_predictor
    python src/inference/dynamic_predictor.py
"""

# ---------------------------------------------------------------------------
# sys.path bootstrap (see static_predictor.py for the rationale)
# ---------------------------------------------------------------------------
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
# Project imports
# ---------------------------------------------------------------------------
import config
from src.common.landmarks import normalize_sequence
from src.common.utils import pad_or_truncate_sequence
from src.models.dynamic_model import load_dynamic_model


class DynamicPredictor:
    """Load a trained dynamic LSTM and classify two-hand landmark sequences.

    Parameters
    ----------
    model_path : str, optional
        Path to the saved checkpoint.  Defaults to
        :data:`config.DYNAMIC_MODEL_PATH`.
    device : torch.device or str, optional
        Inference device.  ``None`` -> :func:`config.get_device`.

    Raises
    ------
    FileNotFoundError
        If ``model_path`` does not exist.
    """

    def __init__(self, model_path: str = config.DYNAMIC_MODEL_PATH, device=None):
        if device is None:
            self.device = config.get_device()
        elif isinstance(device, str):
            self.device = torch.device(device)
        else:
            self.device = device

        self.model_path = model_path

        if not os.path.isfile(self.model_path):
            raise FileNotFoundError(
                "Dynamic model checkpoint not found at '{}'. "
                "Train it first with: python -m src.training.train_dynamic".format(
                    self.model_path
                )
            )

        # Rebuild the LSTM from the saved ``arch`` and load its weights.
        self.model, self.classes = load_dynamic_model(self.model_path, self.device)
        self.model.eval()

    # ------------------------------------------------------------------ #
    # Public API
    # ------------------------------------------------------------------ #
    @property
    def is_available(self) -> bool:
        """``True`` when a model is loaded and ready to predict."""
        return self.model is not None and len(self.classes) > 0

    def predict(self, raw_sequence) -> dict:
        """Classify a variable-length sequence of RAW 126-value frames.

        Parameters
        ----------
        raw_sequence : array-like
            Iterable of frames; each frame is 126 RAW (un-normalized) floats
            (two hands, Left block then Right block).  Any length is accepted;
            it is padded / truncated to :data:`config.SEQUENCE_LENGTH`.

        Returns
        -------
        dict
            ``{"prediction": str, "confidence": float,
               "top_k": [{"label": str, "confidence": float}, ... up to 3]}``
        """
        # --- Fixed length, then normalize each frame ---------------------
        # Shared single source of truth from src.common.utils:
        #   pad_or_truncate_sequence(seq, length, feature_dim)
        fixed = pad_or_truncate_sequence(
            raw_sequence,
            config.SEQUENCE_LENGTH,
            config.DYNAMIC_FEATURE_DIM,
        )  # (SEQUENCE_LENGTH, 126)
        normalized = normalize_sequence(fixed)  # (SEQUENCE_LENGTH, 126) float32

        # --- Forward pass: add batch dim -> (1, T, 126) ------------------
        x = torch.from_numpy(np.asarray(normalized, dtype=np.float32))
        x = x.float().unsqueeze(0).to(self.device)
        with torch.no_grad():
            logits = self.model(x)                 # (1, num_glosses)
            probs = F.softmax(logits, dim=1)[0]    # (num_glosses,)

        return self._format_result(probs)

    # ------------------------------------------------------------------ #
    # Internal helpers
    # ------------------------------------------------------------------ #
    def _format_result(self, probs: torch.Tensor) -> dict:
        """Turn a probability tensor into the API-shaped result dictionary."""
        probs_np = probs.detach().cpu().numpy()

        best_idx = int(np.argmax(probs_np))
        prediction = self.classes[best_idx]
        confidence = float(probs_np[best_idx])

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
    """Load the model and predict on a random sequence to prove the pipeline."""
    print("DynamicPredictor self-test")
    print("-" * 40)
    print("Model path : {}".format(config.DYNAMIC_MODEL_PATH))

    try:
        predictor = DynamicPredictor()
    except FileNotFoundError as exc:
        print("ERROR: {}".format(exc))
        return

    print("Device     : {}".format(predictor.device))
    print("Num glosses: {}".format(len(predictor.classes)))

    # Deterministic random sequence with a deliberately "odd" length (17) to
    # exercise the padding path.  Content is meaningless; this only proves the
    # forward pass runs end-to-end.
    rng = np.random.default_rng(seed=0)
    odd_length = 17
    sample = rng.random((odd_length, config.DYNAMIC_FEATURE_DIM)).astype(np.float32)

    result = predictor.predict(sample)
    print("\nPrediction (on synthetic random sequence; meaningless content):")
    print("  prediction : {}".format(result["prediction"]))
    print("  confidence : {:.4f}".format(result["confidence"]))
    print("  top_k      :")
    for entry in result["top_k"]:
        print("    {:<20s} {:.4f}".format(entry["label"], entry["confidence"]))


if __name__ == "__main__":
    _main()
