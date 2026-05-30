from __future__ import annotations

import argparse
import json
from pathlib import Path

import joblib
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
)
from sklearn.model_selection import train_test_split

from app.ml.dataset import load_dataset


def compute_metrics(y_true: np.ndarray, y_pred: np.ndarray) -> dict[str, float]:
    return {
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "precision": float(precision_score(y_true, y_pred, average="macro", zero_division=0)),
        "recall": float(recall_score(y_true, y_pred, average="macro", zero_division=0)),
        "f1": float(f1_score(y_true, y_pred, average="macro", zero_division=0)),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Evaluate trained ASL models")
    parser.add_argument("--dataset", required=True)
    args = parser.parse_args()

    x, y, labels = load_dataset(args.dataset)
    _, x_test, _, y_test = train_test_split(x, y, test_size=0.2, stratify=y, random_state=42)

    model_dir = Path(__file__).resolve().parents[1] / "models" / "v1"
    report_dir = Path(__file__).resolve().parents[2] / "report" / "assets"
    report_dir.mkdir(parents=True, exist_ok=True)

    payload = {"models": {}, "confusion_matrix": {"labels": labels, "matrix": []}}

    rf_model = joblib.load(model_dir / "random_forest.joblib")
    y_pred_rf = rf_model.predict(x_test)
    payload["models"]["random_forest"] = {
        **compute_metrics(y_test, y_pred_rf),
        "classification_report": classification_report(y_test, y_pred_rf, output_dict=True, zero_division=0),
    }

    try:
        from tensorflow import keras

        nn_model = keras.models.load_model(model_dir / "mlp_classifier.keras")
        y_pred_nn = np.argmax(nn_model.predict(x_test, verbose=0), axis=1)
        payload["models"]["neural_network"] = {
            **compute_metrics(y_test, y_pred_nn),
            "classification_report": classification_report(y_test, y_pred_nn, output_dict=True, zero_division=0),
        }
        cm = confusion_matrix(y_test, y_pred_nn)
    except Exception:
        cm = confusion_matrix(y_test, y_pred_rf)
        payload["models"]["neural_network"] = {
            "accuracy": 0.0,
            "precision": 0.0,
            "recall": 0.0,
            "f1": 0.0,
            "classification_report": {},
        }

    payload["confusion_matrix"]["matrix"] = cm.tolist()
    (report_dir / "metrics.json").write_text(json.dumps(payload, indent=2), encoding="utf-8")

    plt.figure(figsize=(12, 10))
    sns.heatmap(cm, cmap="Blues")
    plt.title("Confusion Matrix")
    plt.xlabel("Predicted")
    plt.ylabel("Actual")
    plt.tight_layout()
    plt.savefig(report_dir / "confusion_matrix.png")

    comparison_labels = ["accuracy", "precision", "recall", "f1"]
    rf_values = [payload["models"]["random_forest"][key] for key in comparison_labels]
    nn_values = [payload["models"]["neural_network"][key] for key in comparison_labels]

    x_pos = np.arange(len(comparison_labels))
    width = 0.35
    plt.figure(figsize=(10, 6))
    plt.bar(x_pos - width / 2, rf_values, width, label="RandomForest")
    plt.bar(x_pos + width / 2, nn_values, width, label="Neural Network")
    plt.xticks(x_pos, [label.upper() for label in comparison_labels])
    plt.ylim(0, 1)
    plt.title("Model Comparison")
    plt.legend()
    plt.tight_layout()
    plt.savefig(report_dir / "model_comparison.png")


if __name__ == "__main__":
    main()
