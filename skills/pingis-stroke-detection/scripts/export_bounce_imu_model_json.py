"""
export_bounce_imu_model_json.py

Export the trained bounce IMU RandomForest model as compact JSON for the app.

Run:
  python skills/pingis-stroke-detection/scripts/export_bounce_imu_model_json.py
"""

from __future__ import annotations

import json
from pathlib import Path

import joblib

ROOT_DIR = Path(__file__).resolve().parents[3]
MODEL_DIR = ROOT_DIR / "data" / "models"
OUT_DIR = ROOT_DIR / "apps" / "collector" / "src" / "models"


def export_tree(estimator) -> list:
    tree = estimator.tree_
    nodes: list = []
    for i in range(tree.node_count):
        if tree.children_left[i] == -1:
            counts = tree.value[i][0].astype(float)
            total = counts.sum()
            probabilities = (counts / total).tolist() if total > 0 else counts.tolist()
            nodes.append(probabilities)
        else:
            nodes.append([
                int(tree.feature[i]),
                float(round(float(tree.threshold[i]), 8)),
                int(tree.children_left[i]),
                int(tree.children_right[i]),
            ])
    return nodes


def main() -> None:
    clf = joblib.load(MODEL_DIR / "bounce_rf_classifier.pkl")
    scaler = joblib.load(MODEL_DIR / "bounce_feature_scaler.pkl")
    encoder = joblib.load(MODEL_DIR / "bounce_label_encoder.pkl")
    with (MODEL_DIR / "bounce_feature_cols.json").open("r", encoding="utf-8") as handle:
        feature_names = json.load(handle)

    payload = {
        "labels": encoder.classes_.tolist(),
        "feature_names": feature_names,
        "scaler_mean": [round(float(value), 8) for value in scaler.mean_],
        "scaler_std": [round(float(value), 8) for value in scaler.scale_],
        "trees": [export_tree(tree) for tree in clf.estimators_],
    }

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    out_path = OUT_DIR / "bounce_imu_model.json"
    with out_path.open("w", encoding="utf-8") as handle:
        json.dump(payload, handle, separators=(",", ":"))

    print(f"Exported {out_path} ({out_path.stat().st_size / 1024:.0f} KB)")


if __name__ == "__main__":
    main()
