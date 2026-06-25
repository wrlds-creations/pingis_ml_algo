"""
export_fable_hgb_model_json.py

Exports the selected noise-robust HistGradientBoostingClassifier (v3, all83)
to the Collector app as `apps/collector/src/models/fable_audio_model.json`,
consumed by the new `Fable-algoritm` live test mode via hgbRuntime.ts.

JSON format (new, NOT the RF flat-tree format):
{
  "metadata": { model_version, feature_version, model_type, classes,
                n_iterations, trees_per_iteration, total_nodes,
                engine_defaults: {confidence, retrigger_ms, merge_ms,
                                  group_ms, echo_ms, echo_ratio,
                                  gate_mode, spectral_gate, onset_ratio,
                                  abs_min_rms} },
  "labels": ["floor_bounce", "noise", "racket_bounce", "table_bounce"],
  "feature_names": [83 names in training column order],
  "scaler_mean": [83], "scaler_std": [83],
  "baseline": [4 raw scores],
  "trees": [ ... 400*4 trees, iteration-major (it0/class0, it0/class1, ...) ]
}
Tree node encoding (flat array, root = index 0):
  internal node: [feature_idx, threshold, left_idx, right_idx]
  leaf node:     [value]                      (length 1 => leaf)
Decision: scaled[feature_idx] <= threshold -> left, else right.
Prediction: raw[k] = baseline[k] + sum over iterations of leaf value from
tree (iteration, k); probabilities = softmax(raw).

Run:
  python skills/pingis-audio-classification/scripts/noise_robust/export_fable_hgb_model_json.py
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import joblib
import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
import nr_config  # noqa: E402

ROOT_DIR = Path(__file__).resolve().parents[4]
if not (ROOT_DIR / "data" / "audio" / "raw").exists():
    raise RuntimeError(f"Repo root resolution failed: {ROOT_DIR}")

DEFAULT_MODEL_DIR = ROOT_DIR / "data" / "audio" / "models" / "noise_robust_v3"
DEFAULT_OUT = ROOT_DIR / "apps" / "collector" / "src" / "models" / "fable_audio_model.json"
DEFAULT_VAL_CSV = ROOT_DIR / "data" / "audio" / "processed" / "noise_robust" / "nr_val.csv"

# Settings selected on val (see RESULTS.md, sensitive profile).
ENGINE_DEFAULTS = {
    "confidence": 0.5,
    "retrigger_ms": 120,
    "merge_ms": 120,
    "group_ms": 80,
    "echo_ms": 300,
    "echo_ratio": 0.6,
    "gate_mode": "bandpass",
    "spectral_gate": False,
    "onset_ratio": 1.5,
    "abs_min_rms": 0.0015,
}


def export_tree(nodes: np.ndarray) -> list:
    """Flat node list from a HistGB TreePredictor.nodes structured array.

    Thresholds and leaf values are exported at FULL float64 precision:
    rounding to 8 decimals flips threshold comparisons for samples near the
    split point (measured max prob diff 1e-2 over 200 val rows)."""
    out = []
    for node in nodes:
        if node["is_leaf"]:
            out.append([float(node["value"])])
        else:
            out.append([
                int(node["feature_idx"]),
                float(node["num_threshold"]),
                int(node["left"]),
                int(node["right"]),
            ])
    return out


def predict_from_export(trees: list, baseline: list, x_scaled: np.ndarray, n_classes: int) -> np.ndarray:
    """Pure-Python reimplementation of the app runtime for the round-trip check."""
    raw = np.array(baseline, dtype=np.float64).copy()
    for tree_idx, tree in enumerate(trees):
        k = tree_idx % n_classes
        node = tree[0]
        idx = 0
        while len(node) != 1:
            feature_idx, threshold, left, right = node
            idx = left if x_scaled[int(feature_idx)] <= threshold else right
            node = tree[int(idx)]
        raw[k] += node[0]
    e = np.exp(raw - raw.max())
    return e / e.sum()


def main() -> None:
    parser = argparse.ArgumentParser(description="Export Fable HistGB model to app JSON.")
    parser.add_argument("--model-dir", default=str(DEFAULT_MODEL_DIR))
    parser.add_argument("--out", default=str(DEFAULT_OUT))
    parser.add_argument("--val-csv", default=str(DEFAULT_VAL_CSV))
    parser.add_argument("--model-version", default="fable_audio_hgb_v3_2026_06_10")
    parser.add_argument("--n-check-rows", type=int, default=200)
    parser.add_argument("--feature-set", default="all83",
                        help="Featureset-suffix i artefaktnamnen (t.ex. all83, stable).")
    args = parser.parse_args()

    model_dir = Path(args.model_dir)
    clf = joblib.load(model_dir / f"nr_histgb_{args.feature_set}.pkl")
    scaler = joblib.load(model_dir / f"nr_scaler_{args.feature_set}.pkl")
    feature_cols = list(joblib.load(model_dir / f"nr_feature_cols_{args.feature_set}.pkl"))
    encoder = joblib.load(model_dir / "nr_label_encoder.pkl")
    labels = [str(c) for c in encoder.classes_]
    if labels != nr_config.CLASSES:
        raise SystemExit(f"Label order mismatch: {labels} vs {nr_config.CLASSES}")

    n_classes = len(labels)
    baseline = [float(v) for v in np.ravel(clf._baseline_prediction)]
    trees = []
    for iteration in clf._predictors:
        if len(iteration) != n_classes:
            raise SystemExit("Expected one tree per class per iteration")
        for predictor in iteration:
            trees.append(export_tree(predictor.nodes))
    total_nodes = sum(len(t) for t in trees)

    payload = {
        "metadata": {
            "model_version": args.model_version,
            "feature_version": "nr_features_83_v1",
            "model_type": "hist_gradient_boosting_4class_audio",
            "classes": labels,
            "n_iterations": len(clf._predictors),
            "trees_per_iteration": n_classes,
            "total_nodes": total_nodes,
            "engine_defaults": ENGINE_DEFAULTS,
        },
        "labels": labels,
        "feature_names": feature_cols,
        "scaler_mean": [float(v) for v in scaler.mean_],
        "scaler_std": [float(v) for v in scaler.scale_],
        "baseline": baseline,
        "trees": trees,
    }

    # Round-trip check on real validation rows BEFORE writing.
    val_df = pd.read_csv(args.val_csv)
    rng = np.random.default_rng(7)
    rows = val_df.sample(n=min(args.n_check_rows, len(val_df)), random_state=7)
    X = rows[feature_cols].to_numpy(dtype=np.float64)
    X = np.nan_to_num(X, nan=0.0, posinf=0.0, neginf=0.0)
    X_scaled = scaler.transform(X)
    expected = clf.predict_proba(X_scaled)
    max_diff = 0.0
    for i in range(len(rows)):
        got = predict_from_export(payload["trees"], payload["baseline"], X_scaled[i], n_classes)
        max_diff = max(max_diff, float(np.max(np.abs(got - expected[i]))))
    print(f"Round-trip check on {len(rows)} val rows: max |export - sklearn| = {max_diff:.3e}")
    if max_diff > 1e-9:
        raise SystemExit("Round-trip check FAILED; not writing export.")

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(payload, separators=(",", ":")), encoding="utf-8")
    size_kb = out_path.stat().st_size / 1024
    print(f"Wrote {out_path} ({size_kb:.0f} KB, {len(trees)} trees, {total_nodes} nodes)")

    # Also dump a small parity fixture used by the Node TS-parity harness:
    # 20 random scaled val rows + expected probabilities.
    fixture_rows = rows.head(20)
    Xf = scaler.transform(np.nan_to_num(fixture_rows[feature_cols].to_numpy(dtype=np.float64), nan=0.0))
    # Full float64 precision: rounding x_scaled flips tree thresholds and
    # makes the harness report false model divergence.
    fixture = {
        "feature_names": feature_cols,
        "x_scaled": [[float(v) for v in row] for row in Xf],
        "expected_proba": [[float(v) for v in row] for row in clf.predict_proba(Xf)],
    }
    fixture_path = ROOT_DIR / "data" / "audio" / "processed" / "noise_robust" / "fable_model_parity_fixture.json"
    fixture_path.write_text(json.dumps(fixture), encoding="utf-8")
    print(f"Wrote parity fixture: {fixture_path}")


if __name__ == "__main__":
    main()
