"""
export_contact_model_json.py

Export the trained binary contact RandomForest to a JSON artifact that can be
loaded directly in the React Native app.

Run:
  python skills/pingis-audio-classification/scripts/export_contact_model_json.py
  python skills/pingis-audio-classification/scripts/export_contact_model_json.py --model-dir <dir> --out-path <file>
"""

import argparse
import json
from pathlib import Path

import joblib

ROOT_DIR = Path(__file__).resolve().parents[3]
DEFAULT_MODEL_DIR = ROOT_DIR / "data" / "audio" / "models"
DEFAULT_OUT_PATH = ROOT_DIR / "apps" / "collector" / "src" / "models" / "audio_contact_model.json"


def export_tree(estimator) -> list:
    tree = estimator.tree_
    nodes: list = []
    for i in range(tree.node_count):
        if tree.children_left[i] == -1:
            counts = tree.value[i][0].astype(float)
            total = counts.sum()
            proba = (counts / total).tolist() if total > 0 else counts.tolist()
            nodes.append(proba)
        else:
            nodes.append([
                int(tree.feature[i]),
                float(round(float(tree.threshold[i]), 8)),
                int(tree.children_left[i]),
                int(tree.children_right[i]),
            ])
    return nodes


def main() -> None:
    parser = argparse.ArgumentParser(description="Export binary audio contact RF to app JSON.")
    parser.add_argument("--model-dir", default=str(DEFAULT_MODEL_DIR), help="Directory containing trained contact model artifacts.")
    parser.add_argument("--out-path", default=str(DEFAULT_OUT_PATH), help="Output JSON path for the app model.")
    parser.add_argument("--model-version", default=None, help="Human-readable model version id stored in JSON metadata.")
    parser.add_argument("--train-dataset", default=None, help="Dataset id/variant stored in JSON metadata.")
    parser.add_argument("--feature-version", default="audio_features_62_v1", help="Feature version stored in JSON metadata.")
    args = parser.parse_args()

    model_dir = Path(args.model_dir)
    out_path = Path(args.out_path)

    clf = joblib.load(model_dir / "audio_contact_rf_classifier.pkl")
    scaler = joblib.load(model_dir / "audio_contact_feature_scaler.pkl")
    le = joblib.load(model_dir / "audio_contact_label_encoder.pkl")
    feature_cols = joblib.load(model_dir / "audio_contact_feature_cols.pkl")

    model = {
        "metadata": {
            "model_version": args.model_version or model_dir.name,
            "train_dataset": args.train_dataset or model_dir.name,
            "feature_version": args.feature_version,
            "model_type": "random_forest_binary_audio_contact",
            "classes": le.classes_.tolist(),
            "tree_count": len(clf.estimators_),
        },
        "labels": le.classes_.tolist(),
        "feature_names": feature_cols,
        "scaler_mean": [round(float(v), 8) for v in scaler.mean_],
        "scaler_std": [round(float(v), 8) for v in scaler.scale_],
        "trees": [export_tree(tree) for tree in clf.estimators_],
    }

    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(model, f, separators=(",", ":"))

    print(f"Model directory: {model_dir}")
    print(f"Exporterad: {out_path}")
    print(f"Klasser: {model['labels']}")
    print(f"Storlek: {out_path.stat().st_size / 1024:.0f} KB")


if __name__ == "__main__":
    main()
