"""
export_model_json.py

Exporterar tränad RandomForest + scaler + labels som en kompakt JSON-fil
för användning direkt i React Native-appen (ingen server behövs).

Kör: python skills/pingis-audio-classification/scripts/export_model_json.py
"""

import json
import argparse
import numpy as np
import joblib
from pathlib import Path

ROOT_DIR  = Path(__file__).resolve().parents[3]
DEFAULT_MODEL_DIR = ROOT_DIR / "data" / "audio" / "models"
DEFAULT_OUT_PATH  = ROOT_DIR / "apps" / "collector" / "src" / "models" / "audio_model.json"


def export_tree(estimator) -> list:
    """Exporterar ett sklearn DecisionTree som en flat lista av noder.

    Varje nod är antingen:
      Intern:  [feature_idx, threshold, left_child, right_child]  (4 element)
      Löv:     [p0, p1, p2, p3, ...]  (len == n_classes, float)

    Lövnoderna lagrar normaliserade klassannolikheter.
    """
    t = tree = estimator.tree_
    nodes: list = []
    for i in range(t.node_count):
        if t.children_left[i] == -1:  # löv
            counts = t.value[i][0].astype(float)
            total  = counts.sum()
            proba  = (counts / total).tolist() if total > 0 else counts.tolist()
            nodes.append(proba)
        else:
            nodes.append([
                int(t.feature[i]),
                float(round(float(t.threshold[i]), 8)),
                int(t.children_left[i]),
                int(t.children_right[i]),
            ])
    return nodes


def main() -> None:
    parser = argparse.ArgumentParser(description="Export multiclass audio RF to app JSON.")
    parser.add_argument("--model-dir", default=str(DEFAULT_MODEL_DIR), help="Directory containing trained audio model artifacts.")
    parser.add_argument("--out-path", default=str(DEFAULT_OUT_PATH), help="Output JSON path for the app model.")
    parser.add_argument("--model-version", default=None, help="Human-readable model version id stored in JSON metadata.")
    parser.add_argument("--train-dataset", default=None, help="Dataset id/variant stored in JSON metadata.")
    parser.add_argument("--feature-version", default="audio_features_62_v1", help="Feature version stored in JSON metadata.")
    args = parser.parse_args()

    model_dir = Path(args.model_dir)
    out_path = Path(args.out_path)

    clf    = joblib.load(model_dir / "audio_rf_classifier.pkl")
    scaler = joblib.load(model_dir / "audio_feature_scaler.pkl")
    le     = joblib.load(model_dir / "audio_label_encoder.pkl")
    feature_cols = joblib.load(model_dir / "audio_feature_cols.pkl")

    total_nodes = sum(t.tree_.node_count for t in clf.estimators_)
    print(f"Träd: {len(clf.estimators_)}  ·  Totalt antal noder: {total_nodes}")

    model = {
        "metadata": {
            "model_version": args.model_version or model_dir.name,
            "train_dataset": args.train_dataset or model_dir.name,
            "feature_version": args.feature_version,
            "model_type": "random_forest_4class_audio",
            "classes": le.classes_.tolist(),
            "tree_count": len(clf.estimators_),
        },
        "labels":        le.classes_.tolist(),
        "feature_names": feature_cols,
        "scaler_mean":   [round(float(v), 8) for v in scaler.mean_],
        "scaler_std":    [round(float(v), 8) for v in scaler.scale_],
        "trees":         [export_tree(t) for t in clf.estimators_],
    }

    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w") as f:
        json.dump(model, f, separators=(",", ":"))

    size_kb = out_path.stat().st_size / 1024
    print(f"Model directory: {model_dir}")
    print(f"Exporterad: {out_path}")
    print(f"Storlek: {size_kb:.0f} KB")
    print(f"Klasser: {model['labels']}")


if __name__ == "__main__":
    main()
