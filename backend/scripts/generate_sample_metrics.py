from __future__ import annotations

import json
from pathlib import Path

labels = [chr(code) for code in range(ord("A"), ord("Z") + 1)]
size = len(labels)
matrix = [[0 for _ in range(size)] for _ in range(size)]
for i in range(size):
    matrix[i][i] = 18
    if i + 1 < size:
        matrix[i][i + 1] = 2

payload = {
    "models": {
        "random_forest": {"accuracy": 0.86, "precision": 0.85, "recall": 0.84, "f1": 0.84},
        "neural_network": {"accuracy": 0.89, "precision": 0.88, "recall": 0.87, "f1": 0.87},
    },
    "confusion_matrix": {"labels": labels, "matrix": matrix},
}

output = Path(__file__).resolve().parents[2] / "report" / "assets" / "metrics.json"
output.parent.mkdir(parents=True, exist_ok=True)
output.write_text(json.dumps(payload, indent=2), encoding="utf-8")
print(f"Wrote sample metrics to {output}")
