from __future__ import annotations

import argparse
import json
from datetime import datetime
from pathlib import Path

import joblib
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split

from app.ml.dataset import load_dataset


def train_rf(x_train: np.ndarray, y_train: np.ndarray) -> RandomForestClassifier:
    model = RandomForestClassifier(n_estimators=300, random_state=42, n_jobs=-1)
    model.fit(x_train, y_train)
    return model


def train_nn(x_train: np.ndarray, y_train: np.ndarray, num_classes: int):
    from tensorflow import keras

    model = keras.Sequential(
        [
            keras.layers.Input(shape=(x_train.shape[1],)),
            keras.layers.Dense(256, activation="relu"),
            keras.layers.Dropout(0.3),
            keras.layers.Dense(128, activation="relu"),
            keras.layers.Dense(num_classes, activation="softmax"),
        ]
    )
    model.compile(optimizer="adam", loss="sparse_categorical_crossentropy", metrics=["accuracy"])
    model.fit(x_train, y_train, validation_split=0.2, epochs=20, batch_size=32, verbose=1)
    return model


def main() -> None:
    parser = argparse.ArgumentParser(description="Train RF and NN models for ASL A-Z recognition")
    parser.add_argument("--dataset", required=True, help="Path to dataset organized as <root>/<class>/*.jpg")
    args = parser.parse_args()

    features, targets, labels = load_dataset(args.dataset)
    x_train, _, y_train, _ = train_test_split(features, targets, test_size=0.2, stratify=targets, random_state=42)

    model_dir = Path(__file__).resolve().parents[1] / "models" / "v1"
    model_dir.mkdir(parents=True, exist_ok=True)

    rf_model = train_rf(x_train, y_train)
    joblib.dump(rf_model, model_dir / "random_forest.joblib")

    try:
        nn_model = train_nn(x_train, y_train, num_classes=len(labels))
        nn_model.save(model_dir / "mlp_classifier.keras")
    except Exception as exc:
        print(f"[WARN] Skipping NN training because TensorFlow runtime is unavailable: {exc}")

    (model_dir / "labels.json").write_text(json.dumps(labels, indent=2), encoding="utf-8")
    (model_dir / "VERSION.txt").write_text(datetime.utcnow().isoformat(), encoding="utf-8")
    print(f"Saved model artifacts to {model_dir}")


if __name__ == "__main__":
    main()
