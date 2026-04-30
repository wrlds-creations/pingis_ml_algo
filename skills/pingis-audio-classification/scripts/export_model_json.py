"""
export_model_json.py

Exporterar tränad RandomForest + scaler + labels som en kompakt JSON-fil
för användning direkt i React Native-appen (ingen server behövs).

Kör: python skills/pingis-audio-classification/scripts/export_model_json.py
"""

import json
import numpy as np
import joblib
from pathlib import Path

ROOT_DIR  = Path(__file__).resolve().parents[3]
MODEL_DIR = ROOT_DIR / "data" / "audio" / "models"
OUT_PATH  = ROOT_DIR / "apps" / "collector" / "src" / "models" / "audio_model.json"


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
    clf    = joblib.load(MODEL_DIR / "audio_rf_classifier.pkl")
    scaler = joblib.load(MODEL_DIR / "audio_feature_scaler.pkl")
    le     = joblib.load(MODEL_DIR / "audio_label_encoder.pkl")
    feature_cols = joblib.load(MODEL_DIR / "audio_feature_cols.pkl")

    total_nodes = sum(t.tree_.node_count for t in clf.estimators_)
    print(f"Träd: {len(clf.estimators_)}  ·  Totalt antal noder: {total_nodes}")

    model = {
        "labels":        le.classes_.tolist(),
        "feature_names": feature_cols,
        "scaler_mean":   [round(float(v), 8) for v in scaler.mean_],
        "scaler_std":    [round(float(v), 8) for v in scaler.scale_],
        "trees":         [export_tree(t) for t in clf.estimators_],
    }

    OUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    with open(OUT_PATH, "w") as f:
        json.dump(model, f, separators=(",", ":"))

    size_kb = OUT_PATH.stat().st_size / 1024
    print(f"Exporterad: {OUT_PATH}")
    print(f"Storlek: {size_kb:.0f} KB")
    print(f"Klasser: {model['labels']}")


if __name__ == "__main__":
    main()
