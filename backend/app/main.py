from __future__ import annotations

import json
import logging
import os
from collections import deque
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import cv2
import numpy as np
from fastapi import FastAPI, File, HTTPException, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse

from app.ml.inference import ModelManager
from app.ml.landmarks import extract_landmarks_from_bgr

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s")
LOGGER = logging.getLogger("sign_interpreter")

app = FastAPI(title="Real-Time Sign Language Interpreter API", version="1.0.0")

origins = [origin.strip() for origin in os.getenv("ALLOWED_ORIGINS", "http://localhost:5173").split(",")]
app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

model_manager = ModelManager()
detections: deque[dict[str, Any]] = deque(maxlen=200)
METRICS_PATH = Path(__file__).resolve().parents[2] / "report" / "assets" / "metrics.json"


@app.get("/health")
def health() -> dict[str, str]:
    return {"status": "ok"}


@app.get("/history")
def history() -> dict[str, Any]:
    return {"count": len(detections), "items": list(detections)}


@app.get("/metrics")
def metrics() -> dict[str, Any]:
    default_payload = {
        "models": {
            "random_forest": {"accuracy": 0.0, "precision": 0.0, "recall": 0.0, "f1": 0.0},
            "neural_network": {"accuracy": 0.0, "precision": 0.0, "recall": 0.0, "f1": 0.0},
        },
        "confusion_matrix": {"labels": [], "matrix": []},
        "message": "Run backend/scripts/evaluate_models.py to populate real metrics.",
    }
    if METRICS_PATH.exists():
        return json.loads(METRICS_PATH.read_text(encoding="utf-8"))
    return default_payload


@app.post("/predict")
async def predict(frame: UploadFile = File(...)) -> dict[str, Any]:
    if not frame.content_type or not frame.content_type.startswith("image/"):
        raise HTTPException(status_code=400, detail="Uploaded file must be an image.")

    payload = await frame.read()
    if not payload:
        raise HTTPException(status_code=400, detail="Uploaded frame is empty.")

    img_arr = np.frombuffer(payload, dtype=np.uint8)
    image = cv2.imdecode(img_arr, cv2.IMREAD_COLOR)
    if image is None:
        raise HTTPException(status_code=400, detail="Unable to decode image frame.")

    extraction = extract_landmarks_from_bgr(image)
    if extraction is None:
        result = {
            "predicted_sign": "NO_HAND",
            "confidence": 0.0,
            "model_used": "vision",
            "message": "No hand detected in frame.",
            "landmarks": [],
            "bounding_box": None,
        }
    else:
        prediction = model_manager.predict(extraction.feature_vector)
        result = {
            **prediction,
            "landmarks": extraction.raw_landmarks,
            "bounding_box": extraction.bounding_box,
        }

    result["timestamp"] = datetime.now(timezone.utc).isoformat()
    detections.appendleft(result)
    LOGGER.info("Prediction generated: %s (%.3f)", result["predicted_sign"], result["confidence"])
    return result


@app.exception_handler(Exception)
async def unhandled_exception_handler(_, exc: Exception):
    LOGGER.exception("Unexpected server error: %s", exc)
    return JSONResponse(status_code=500, content={"detail": "Internal server error"})
