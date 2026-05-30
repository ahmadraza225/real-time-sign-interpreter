from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np

try:
    import mediapipe as mp
except Exception:  # pragma: no cover
    mp = None


@dataclass
class LandmarkExtractionResult:
    feature_vector: np.ndarray
    raw_landmarks: list[dict[str, float]]
    bounding_box: dict[str, float] | None


def _normalize_landmarks(landmarks: list[dict[str, float]]) -> np.ndarray:
    wrist = landmarks[0]
    normalized = []
    for point in landmarks:
        normalized.extend(
            [
                point["x"] - wrist["x"],
                point["y"] - wrist["y"],
                point["z"] - wrist["z"],
            ]
        )
    return np.asarray(normalized, dtype=np.float32)


def extract_landmarks_from_bgr(frame_bgr: np.ndarray) -> LandmarkExtractionResult | None:
    if mp is None:
        return None

    mp_hands = mp.solutions.hands
    with mp_hands.Hands(static_image_mode=True, max_num_hands=1, min_detection_confidence=0.5) as hands:
        frame_rgb = frame_bgr[:, :, ::-1]
        results = hands.process(frame_rgb)

    if not results.multi_hand_landmarks:
        return None

    hand = results.multi_hand_landmarks[0]
    raw_landmarks: list[dict[str, float]] = []
    xs, ys = [], []
    for lm in hand.landmark:
        raw_landmarks.append({"x": float(lm.x), "y": float(lm.y), "z": float(lm.z)})
        xs.append(float(lm.x))
        ys.append(float(lm.y))

    feature_vector = _normalize_landmarks(raw_landmarks)
    bbox = {
        "x_min": min(xs),
        "y_min": min(ys),
        "x_max": max(xs),
        "y_max": max(ys),
    }
    return LandmarkExtractionResult(feature_vector=feature_vector, raw_landmarks=raw_landmarks, bounding_box=bbox)


def landmarks_to_feature_array(landmarks: list[dict[str, Any]]) -> np.ndarray:
    parsed = [
        {
            "x": float(point.get("x", 0.0)),
            "y": float(point.get("y", 0.0)),
            "z": float(point.get("z", 0.0)),
        }
        for point in landmarks
    ]
    if len(parsed) != 21:
        raise ValueError("Expected 21 hand landmarks.")
    return _normalize_landmarks(parsed)
