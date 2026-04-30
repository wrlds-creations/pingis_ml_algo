"""
export_model_json.py

Export the trained IMU RandomForest models as compact JSON files for
direct use in the React Native app.

Run:
  python skills/pingis-stroke-detection/scripts/export_model_json.py
"""

import json
import joblib
from pathlib import Path

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


def build_model(clf, scaler, labels, feature_names):
    return {
        "labels": labels,
        "feature_names": feature_names,
        "scaler_mean": [round(float(value), 8) for value in scaler.mean_],
        "scaler_std": [round(float(value), 8) for value in scaler.scale_],
        "trees": [export_tree(tree) for tree in clf.estimators_],
    }


def write_model(filename: str, model: dict) -> None:
    path = OUT_DIR / filename
    with path.open("w", encoding="utf-8") as handle:
        json.dump(model, handle, separators=(",", ":"))
    print(f"Exported {filename} ({path.stat().st_size / 1024:.0f} KB)")


def main() -> None:
    hit_clf = joblib.load(MODEL_DIR / "rf_classifier.pkl")
    hit_scaler = joblib.load(MODEL_DIR / "feature_scaler.pkl")
    hit_encoder = joblib.load(MODEL_DIR / "label_encoder.pkl")

    stroke_clf = joblib.load(MODEL_DIR / "rf_stroke_type.pkl")
    stroke_scaler = joblib.load(MODEL_DIR / "stroke_type_scaler.pkl")
    stroke_encoder = joblib.load(MODEL_DIR / "stroke_type_encoder.pkl")

    with (MODEL_DIR / "feature_cols.json").open("r", encoding="utf-8") as handle:
      feature_names = json.load(handle)

    OUT_DIR.mkdir(parents=True, exist_ok=True)

    write_model(
        "stroke_hit_model.json",
        build_model(hit_clf, hit_scaler, hit_encoder.classes_.tolist(), feature_names),
    )
    write_model(
        "stroke_type_model.json",
        build_model(stroke_clf, stroke_scaler, stroke_encoder.classes_.tolist(), feature_names),
    )


if __name__ == "__main__":
    main()
