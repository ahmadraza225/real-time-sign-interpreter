from __future__ import annotations

import json
import logging
import os
from pathlib import Path
from typing import Any

import joblib
import numpy as np

try:
    from tensorflow import keras
except Exception:  # pragma: no cover
    keras = None

MODEL_DIR = Path(__file__).resolve().parents[2] / "models" / "v1"
LABELS_PATH = MODEL_DIR / "labels.json"
RF_PATH = MODEL_DIR / "random_forest.joblib"
NN_PATH = MODEL_DIR / "mlp_classifier.keras"

LOGGER = logging.getLogger(__name__)
DEFAULT_LABELS = [chr(code) for code in range(ord("A"), ord("Z") + 1)]


class ModelManager:
    def __init__(self) -> None:
        self.labels = self._load_labels()
        self.rf_model = self._safe_load_rf()
        self.nn_model = self._safe_load_nn()

    def _load_labels(self) -> list[str]:
        if LABELS_PATH.exists():
            return json.loads(LABELS_PATH.read_text(encoding="utf-8"))
        return DEFAULT_LABELS

    def _safe_load_rf(self) -> Any:
        if not RF_PATH.exists():
            return None
        try:
            return joblib.load(RF_PATH)
        except Exception as exc:  # pragma: no cover
            LOGGER.warning("Failed to load RandomForest model: %s", exc)
            return None

    def _safe_load_nn(self) -> Any:
        if not NN_PATH.exists() or keras is None:
            return None
        try:
            return keras.models.load_model(NN_PATH)
        except Exception as exc:  # pragma: no cover
            LOGGER.warning("Failed to load NN model: %s", exc)
            return None

    def predict(self, features: np.ndarray, preferred_model: str | None = None) -> dict[str, Any]:
        preferred = preferred_model or os.getenv("MODEL_PREFERENCE", "rf")

        if preferred == "nn" and self.nn_model is not None:
            return self._predict_with_nn(features)
        if self.rf_model is not None:
            return self._predict_with_rf(features)
        if self.nn_model is not None:
            return self._predict_with_nn(features)

        return {
            "predicted_sign": "UNAVAILABLE",
            "confidence": 0.0,
            "model_used": "stub",
            "message": "No trained model artifact found in backend/models/v1.",
        }

    def _predict_with_rf(self, features: np.ndarray) -> dict[str, Any]:
        probabilities = self.rf_model.predict_proba(features.reshape(1, -1))[0]
        class_idx = int(np.argmax(probabilities))
        return {
            "predicted_sign": self.labels[class_idx],
            "confidence": float(probabilities[class_idx]),
            "model_used": "random_forest",
        }

    def _predict_with_nn(self, features: np.ndarray) -> dict[str, Any]:
        raw = self.nn_model.predict(features.reshape(1, -1), verbose=0)[0]
        class_idx = int(np.argmax(raw))
        return {
            "predicted_sign": self.labels[class_idx],
            "confidence": float(raw[class_idx]),
            "model_used": "neural_network",
        }
