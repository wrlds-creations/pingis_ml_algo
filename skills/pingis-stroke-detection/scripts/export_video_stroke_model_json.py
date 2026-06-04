"""
Export the trained video-stroke RandomForest model for the Collector app.

Usage:
  python skills/pingis-stroke-detection/scripts/export_video_stroke_model_json.py
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import joblib


ROOT_DIR = Path(__file__).resolve().parents[3]
DEFAULT_MODEL_DIR = ROOT_DIR / "data" / "video" / "models"
DEFAULT_OUT_PATH = ROOT_DIR / "apps" / "collector" / "src" / "models" / "video_stroke_model.json"
FEATURE_SPEC = "video_stroke_features_v1"


def export_tree(estimator) -> list:
    tree = estimator.tree_
    nodes: list = []
    for node_index in range(tree.node_count):
        if tree.children_left[node_index] == -1:
            counts = tree.value[node_index][0].astype(float)
            total = counts.sum()
            probabilities = (counts / total).tolist() if total > 0 else counts.tolist()
            nodes.append([round(float(probability), 8) for probability in probabilities])
        else:
            nodes.append([
                int(tree.feature[node_index]),
                float(round(float(tree.threshold[node_index]), 8)),
                int(tree.children_left[node_index]),
                int(tree.children_right[node_index]),
            ])
    return nodes


def main() -> None:
    parser = argparse.ArgumentParser(description="Export video stroke RF to app JSON.")
    parser.add_argument("--model-dir", default=str(DEFAULT_MODEL_DIR))
    parser.add_argument("--out-path", default=str(DEFAULT_OUT_PATH))
    parser.add_argument("--model-version", default=None)
    args = parser.parse_args()

    model_dir = Path(args.model_dir)
    out_path = Path(args.out_path)
    classifier = joblib.load(model_dir / "video_stroke_rf_classifier.pkl")
    scaler = joblib.load(model_dir / "video_stroke_feature_scaler.pkl")
    encoder = joblib.load(model_dir / "video_stroke_label_encoder.pkl")
    feature_cols = joblib.load(model_dir / "video_stroke_feature_cols.pkl")

    model = {
        "trained": True,
        "model_version": args.model_version or model_dir.name,
        "feature_spec": FEATURE_SPEC,
        "metadata": {
            "model_type": "random_forest_video_stroke",
            "tree_count": len(classifier.estimators_),
            "classes": encoder.classes_.tolist(),
        },
        "labels": encoder.classes_.tolist(),
        "feature_names": feature_cols,
        "scaler_mean": [round(float(value), 8) for value in scaler.mean_],
        "scaler_std": [round(float(value), 8) for value in scaler.scale_],
        "trees": [export_tree(estimator) for estimator in classifier.estimators_],
    }

    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(model, separators=(",", ":")), encoding="utf-8")
    print(f"Exported {len(classifier.estimators_)} trees to {out_path}")
    print(f"Labels: {model['labels']}")


if __name__ == "__main__":
    main()
