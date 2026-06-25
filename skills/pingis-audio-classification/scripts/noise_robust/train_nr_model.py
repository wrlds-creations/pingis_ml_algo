"""
train_nr_model.py

Module 3 of NR_SPEC.md: train noise-robust 4-class audio bounce models on the
nr_train.csv / nr_val.csv datasets produced by build_nr_dataset.py.

Per feature set (base62 / robust21 / all83):
  - RandomForest: GridSearchCV over a small grid, CV = GroupKFold grouped by
    group_id (= session) on TRAIN ONLY, scoring f1_macro.
  - HistGradientBoosting: fixed hyperparameters, same grouped CV for sanity.

All shipped models are train-only fits (no refit on val/test). Every
(feature set x model) combo is scored on the clean reviewed-marker rows of
nr_val.csv. Winner = max(0.6 * racket_recall + 0.4 * racket_precision)
subject to racket_precision >= 0.90 on val (if no combo passes, the best
score is reported anyway).

Outputs in --out-dir:
  - nr_rf_<set>.pkl / nr_histgb_<set>.pkl / nr_scaler_<set>.pkl /
    nr_feature_cols_<set>.pkl / nr_label_encoder.pkl
  - training_log.json (configs, CV results, val metrics, durations, versions)
  - val_results.md (markdown results table, also printed to stdout)
  - nr_audio_model.json: app-format export of the best RandomForest, with the
    tree encoding copied from ../export_model_json.py (internal node =
    [feature_idx, threshold, left_idx, right_idx], leaf = normalized class
    probability list, 8-decimal rounding). A pure-Python round-trip check
    verifies the exported JSON against sklearn predict_proba after export.

Run:
  python skills/pingis-audio-classification/scripts/noise_robust/train_nr_model.py
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

import joblib
import librosa
import numpy as np
import pandas as pd
import scipy
import sklearn
from sklearn.ensemble import HistGradientBoostingClassifier, RandomForestClassifier
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.model_selection import GridSearchCV, GroupKFold, cross_validate
from sklearn.preprocessing import LabelEncoder, StandardScaler

sys.path.insert(0, str(Path(__file__).resolve().parent))
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import nr_config  # noqa: E402
from preprocess_audio import extract_features  # noqa: E402

ROOT_DIR = Path(__file__).resolve().parents[4]
if not (ROOT_DIR / "data" / "audio" / "raw").exists():
    raise RuntimeError(
        f"Repo root resolution failed: {ROOT_DIR / 'data' / 'audio' / 'raw'} does not exist."
    )

DEFAULT_DATA_DIR = ROOT_DIR / "data" / "audio" / "processed" / "noise_robust"
DEFAULT_OUT_DIR = ROOT_DIR / "data" / "audio" / "models" / "noise_robust_v1"
DEFAULT_MODEL_VERSION = "nr_bounce_v1_2026_06_10"
DEFAULT_FEATURE_VERSION = "nr_features_83_v1"

META_COLS = [
    "clip_id",
    "split",
    "session_id",
    "wav_filename",
    "scenario_id",
    "background_condition",
    "label",
    "source",
    "anchor_ms",
    "jitter_ms",
    "augment",
    "aug_bed",
    "group_id",
    "close_event_bucket",
]

FEATURE_SET_ORDER = ["base62", "robust21", "all83", "stable"]
RACKET_LABEL = "racket_bounce"
RACKET_PRECISION_FLOOR = 0.90
SCORE_RECALL_WEIGHT = 0.6
SCORE_PRECISION_WEIGHT = 0.4

RF_PARAM_GRID = {
    "n_estimators": [300],
    "max_depth": [None, 25],
    "min_samples_leaf": [1, 3],
}
RF_FIXED_PARAMS = {"class_weight": "balanced_subsample", "n_jobs": -1}
HISTGB_PARAMS = {
    "max_iter": 400,
    "learning_rate": 0.08,
    "max_depth": None,
    "early_stopping": False,
    "class_weight": "balanced",
}


def expected_base_feature_names() -> list[str]:
    """Base 62 feature names in preprocess_audio.extract_features order."""
    zeros = np.zeros(nr_config.FEATURE_BUFFER_SAMPLES, dtype=np.float64)
    return list(extract_features(zeros, nr_config.TARGET_SR).keys())


def resolve_feature_sets(feature_cols: list[str]) -> dict[str, list[str]]:
    base_cols = [col for col in feature_cols if not col.startswith("nr_")]
    robust_cols = [col for col in feature_cols if col.startswith("nr_")]

    expected_base = expected_base_feature_names()
    if base_cols != expected_base:
        missing = [col for col in expected_base if col not in base_cols]
        extra = [col for col in base_cols if col not in expected_base]
        raise ValueError(
            "Base feature columns do not match preprocess_audio.extract_features "
            f"order. missing={missing} extra={extra}"
        )
    if feature_cols != base_cols + robust_cols:
        raise ValueError("Feature columns must be base62 first, then nr_ columns.")
    if not robust_cols:
        raise ValueError("No nr_ feature columns found in the dataset.")

    sets = {
        "base62": base_cols,
        "robust21": robust_cols,
        "all83": base_cols + robust_cols,
    }
    # Approximations-stabilt urval (TS-extraktorn i appen == Python-referensen,
    # max |delta|/scaler-std < 0.05 på paritetsfixturen, se
    # report_feature_parity.js). Klipp nära beslutsgränsen flippade i appen
    # när modellen lutade sig på de divergerande featurerna (2026-06-12).
    stable_path = ROOT_DIR / "data" / "audio" / "processed" / "noise_robust" / "stable_feature_cols.json"
    if stable_path.exists():
        stable = [c for c in json.loads(stable_path.read_text(encoding="utf-8")) if c in feature_cols]
        if stable:
            sets["stable"] = stable
    return sets


def sanitize_features(df: pd.DataFrame, feature_cols: list[str], name: str) -> int:
    """Replace NaN/inf in feature columns with 0.0; return replacement count."""
    block = df[feature_cols].to_numpy(dtype=np.float64)
    bad_mask = ~np.isfinite(block)
    n_bad = int(bad_mask.sum())
    if n_bad > 0:
        block[bad_mask] = 0.0
        df[feature_cols] = block
        print(f"Warning: replaced {n_bad} non-finite feature values with 0.0 in {name}.")
    return n_bad


def load_split_csv(path: Path) -> pd.DataFrame:
    if not path.exists():
        raise FileNotFoundError(f"Dataset missing: {path} (run build_nr_dataset.py first)")
    return pd.read_csv(path)


def select_val_clean(val_df: pd.DataFrame) -> pd.DataFrame:
    source = val_df["source"].fillna("").astype(str)
    augment = val_df["augment"].fillna("").astype(str)
    clean = val_df[(source == "reviewed_marker") & (augment.isin(["", "none"]))].copy()
    if clean.empty:
        raise ValueError("No clean reviewed_marker rows found in nr_val.csv.")
    return clean.reset_index(drop=True)


def build_group_cv(train_df: pd.DataFrame) -> tuple[GroupKFold, int]:
    n_groups = int(train_df["group_id"].nunique())
    if n_groups < 2:
        raise ValueError("Need at least 2 distinct group_id values for grouped CV.")
    n_splits = min(5, n_groups)
    return GroupKFold(n_splits=n_splits), n_splits


def compute_val_metrics(y_true: np.ndarray, y_pred: np.ndarray, class_names: list[str]) -> dict:
    label_ids = list(range(len(class_names)))
    report = classification_report(
        y_true,
        y_pred,
        labels=label_ids,
        target_names=class_names,
        zero_division=0,
        output_dict=True,
    )
    cm = confusion_matrix(y_true, y_pred, labels=label_ids)
    racket = report[RACKET_LABEL]
    racket_precision = float(racket["precision"])
    racket_recall = float(racket["recall"])
    score = SCORE_RECALL_WEIGHT * racket_recall + SCORE_PRECISION_WEIGHT * racket_precision
    return {
        "n_rows": int(len(y_true)),
        "accuracy": float(np.mean(y_true == y_pred)),
        "macro_f1": float(report["macro avg"]["f1-score"]),
        "per_class": {
            name: {
                "precision": float(report[name]["precision"]),
                "recall": float(report[name]["recall"]),
                "f1": float(report[name]["f1-score"]),
                "support": int(report[name]["support"]),
            }
            for name in class_names
        },
        "confusion_matrix": {"labels": list(class_names), "rows_true_cols_pred": cm.tolist()},
        "racket_precision": racket_precision,
        "racket_recall": racket_recall,
        "racket_f1": float(racket["f1-score"]),
        "selection_score": float(score),
        "meets_precision_constraint": bool(racket_precision >= RACKET_PRECISION_FLOOR),
    }


def train_rf(
    X_train: np.ndarray,
    y_train: np.ndarray,
    groups: np.ndarray,
    cv: GroupKFold,
    seed: int,
) -> tuple[RandomForestClassifier, dict]:
    base_clf = RandomForestClassifier(random_state=seed, **RF_FIXED_PARAMS)
    grid = GridSearchCV(
        base_clf,
        RF_PARAM_GRID,
        cv=cv,
        scoring="f1_macro",
        refit=True,
        n_jobs=1,
        verbose=0,
    )
    t0 = time.perf_counter()
    grid.fit(X_train, y_train, groups=groups)
    fit_seconds = time.perf_counter() - t0

    grid_summary = []
    cv_results = grid.cv_results_
    for i, params in enumerate(cv_results["params"]):
        grid_summary.append(
            {
                "params": {k: (None if v is None else v) for k, v in params.items()},
                "mean_f1_macro": float(cv_results["mean_test_score"][i]),
                "std_f1_macro": float(cv_results["std_test_score"][i]),
                "rank": int(cv_results["rank_test_score"][i]),
            }
        )

    info = {
        "fixed_params": dict(RF_FIXED_PARAMS, random_state=seed),
        "param_grid": {k: list(v) for k, v in RF_PARAM_GRID.items()},
        "best_params": {k: (None if v is None else v) for k, v in grid.best_params_.items()},
        "cv_best_f1_macro": float(grid.best_score_),
        "cv_grid": grid_summary,
        "fit_seconds": float(fit_seconds),
    }
    return grid.best_estimator_, info


def train_histgb(
    X_train: np.ndarray,
    y_train: np.ndarray,
    groups: np.ndarray,
    cv: GroupKFold,
    seed: int,
) -> tuple[HistGradientBoostingClassifier, dict]:
    proto = HistGradientBoostingClassifier(random_state=seed, **HISTGB_PARAMS)
    t0 = time.perf_counter()
    cv_res = cross_validate(
        proto,
        X_train,
        y_train,
        groups=groups,
        cv=cv,
        scoring="f1_macro",
        n_jobs=1,
    )
    cv_seconds = time.perf_counter() - t0
    fold_scores = [float(v) for v in cv_res["test_score"]]

    clf = HistGradientBoostingClassifier(random_state=seed, **HISTGB_PARAMS)
    t0 = time.perf_counter()
    clf.fit(X_train, y_train)
    fit_seconds = time.perf_counter() - t0

    info = {
        "fixed_params": dict(HISTGB_PARAMS, random_state=seed),
        "cv_f1_macro_mean": float(np.mean(fold_scores)),
        "cv_f1_macro_std": float(np.std(fold_scores)),
        "cv_fold_scores": fold_scores,
        "cv_seconds": float(cv_seconds),
        "fit_seconds": float(fit_seconds),
    }
    return clf, info


def export_tree(estimator) -> list:
    """Exports a sklearn DecisionTree as a flat node list.

    Encoding copied exactly from export_model_json.py:
      Internal: [feature_idx, threshold, left_child, right_child] (4 elements)
      Leaf:     [p0, p1, p2, p3, ...] (len == n_classes, float)

    Leaf nodes store normalized class probabilities.
    """
    t = estimator.tree_
    nodes: list = []
    for i in range(t.node_count):
        if t.children_left[i] == -1:  # leaf
            counts = t.value[i][0].astype(float)
            total = counts.sum()
            proba = (counts / total).tolist() if total > 0 else counts.tolist()
            nodes.append(proba)
        else:
            nodes.append([
                int(t.feature[i]),
                float(round(float(t.threshold[i]), 8)),
                int(t.children_left[i]),
                int(t.children_right[i]),
            ])
    return nodes


def export_rf_json(
    clf: RandomForestClassifier,
    scaler: StandardScaler,
    le: LabelEncoder,
    feature_cols: list[str],
    feature_set: str,
    model_version: str,
    feature_version: str,
    out_path: Path,
) -> dict:
    model = {
        "metadata": {
            "model_version": model_version,
            "feature_version": feature_version,
            "model_type": "random_forest_4class_audio_nr",
            "feature_set": feature_set,
            "classes": le.classes_.tolist(),
            "tree_count": len(clf.estimators_),
        },
        "labels": le.classes_.tolist(),
        "feature_names": list(feature_cols),
        "scaler_mean": [round(float(v), 8) for v in scaler.mean_],
        "scaler_std": [round(float(v), 8) for v in scaler.scale_],
        "trees": [export_tree(t) for t in clf.estimators_],
    }
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w") as f:
        json.dump(model, f, separators=(",", ":"))
    size_kb = out_path.stat().st_size / 1024
    total_nodes = sum(t.tree_.node_count for t in clf.estimators_)
    print(
        f"App export: {out_path} ({size_kb:.0f} KB, {len(clf.estimators_)} trees, "
        f"{total_nodes} nodes, feature_set={feature_set})"
    )
    return model


def _is_leaf_node(node: list, n_classes: int) -> bool:
    """Leaf detection identical to apps/collector/src/rfRuntime.ts isLeaf()."""
    if len(node) != n_classes:
        return len(node) != 4
    total = 0.0
    for value in node:
        if value < 0 or value > 1:
            return False
        total += value
    return abs(total - 1.0) < 0.01


def _tree_proba(tree: list, scaled: np.ndarray, n_classes: int) -> list:
    idx = 0
    while not _is_leaf_node(tree[idx], n_classes):
        feature_idx, threshold, left_idx, right_idx = tree[idx]
        idx = int(left_idx) if scaled[int(feature_idx)] <= threshold else int(right_idx)
    return tree[idx]


def predict_proba_from_export(model_json: dict, raw_rows: np.ndarray) -> np.ndarray:
    """Pure-Python re-implementation of the app's RF runtime on raw features."""
    mean = np.asarray(model_json["scaler_mean"], dtype=np.float64)
    std = np.asarray(model_json["scaler_std"], dtype=np.float64)
    std = np.where(std == 0.0, 1.0, std)
    n_classes = len(model_json["labels"])
    probas = np.zeros((len(raw_rows), n_classes), dtype=np.float64)
    for row_idx, raw in enumerate(raw_rows):
        scaled = (np.asarray(raw, dtype=np.float64) - mean) / std
        total = np.zeros(n_classes, dtype=np.float64)
        for tree in model_json["trees"]:
            total += np.asarray(_tree_proba(tree, scaled, n_classes), dtype=np.float64)
        probas[row_idx] = total / len(model_json["trees"])
    return probas


def run_roundtrip_check(
    export_path: Path,
    clf: RandomForestClassifier,
    scaler: StandardScaler,
    raw_rows: np.ndarray,
    rng: np.random.Generator,
    n_samples: int = 5,
) -> dict:
    """Walk sample feature vectors through the exported JSON trees in pure
    Python and compare against sklearn predict_proba (tolerance 1e-6)."""
    with open(export_path, "r") as f:
        model_json = json.load(f)

    n_samples = min(n_samples, len(raw_rows))
    sample_idx = rng.choice(len(raw_rows), size=n_samples, replace=False)
    sample_raw = raw_rows[np.sort(sample_idx)]

    json_proba = predict_proba_from_export(model_json, sample_raw)
    sk_proba = clf.predict_proba(scaler.transform(sample_raw))
    max_abs_diff = float(np.max(np.abs(json_proba - sk_proba)))
    passed = bool(max_abs_diff < 1e-6)
    print(
        f"Round-trip check ({n_samples} vectors): max |json - sklearn| = "
        f"{max_abs_diff:.3e} -> {'PASS' if passed else 'FAIL'}"
    )
    if not passed:
        raise RuntimeError(
            f"Exported JSON does not round-trip against sklearn predict_proba "
            f"(max abs diff {max_abs_diff:.3e} >= 1e-6)."
        )
    return {"n_samples": int(n_samples), "max_abs_diff": max_abs_diff, "passed": passed}


def select_winner(results: list[dict]) -> tuple[dict, bool]:
    eligible = [r for r in results if r["val"]["meets_precision_constraint"]]
    pool = eligible if eligible else results
    winner = max(pool, key=lambda r: r["val"]["selection_score"])
    return winner, bool(eligible)


def build_results_markdown(results: list[dict], winner: dict, constraint_met: bool, best_rf: dict) -> str:
    lines = [
        "# Noise-robust model validation results",
        "",
        f"Selection rule: `{SCORE_RECALL_WEIGHT} * racket_recall + "
        f"{SCORE_PRECISION_WEIGHT} * racket_precision`, subject to "
        f"`racket_precision >= {RACKET_PRECISION_FLOOR}` on val (clean reviewed_marker rows).",
        "",
        f"Winner: **{winner['model']} / {winner['feature_set']}** "
        f"(score {winner['val']['selection_score']:.4f}, precision constraint "
        f"{'met' if constraint_met else 'NOT met by any combo - best score reported anyway'}).",
        f"App JSON export (best RF): **rf / {best_rf['feature_set']}** "
        f"(score {best_rf['val']['selection_score']:.4f}).",
        "",
        "| feature_set | model | n_features | cv_f1_macro | val_macro_f1 | "
        "racket_precision | racket_recall | racket_f1 | score | prec>=0.90 | selected |",
        "|---|---|---|---|---|---|---|---|---|---|---|",
    ]
    for r in results:
        val = r["val"]
        lines.append(
            f"| {r['feature_set']} | {r['model']} | {r['n_features']} "
            f"| {r['cv_f1_macro']:.4f} | {val['macro_f1']:.4f} "
            f"| {val['racket_precision']:.4f} | {val['racket_recall']:.4f} "
            f"| {val['racket_f1']:.4f} | {val['selection_score']:.4f} "
            f"| {'yes' if val['meets_precision_constraint'] else 'no'} "
            f"| {'<-- winner' if r is winner else ('app export' if r is best_rf else '')} |"
        )
    lines.append("")
    return "\n".join(lines)


def main() -> None:
    parser = argparse.ArgumentParser(description="Train noise-robust 4-class audio models (NR_SPEC Module 3).")
    parser.add_argument("--data-dir", default=str(DEFAULT_DATA_DIR), help="Directory with nr_train.csv / nr_val.csv.")
    parser.add_argument("--out-dir", default=str(DEFAULT_OUT_DIR), help="Output directory for model artifacts.")
    parser.add_argument("--seed", type=int, default=nr_config.RNG_SEED, help="Random seed.")
    parser.add_argument("--model-version", default=DEFAULT_MODEL_VERSION, help="model_version stored in JSON metadata.")
    parser.add_argument("--feature-version", default=DEFAULT_FEATURE_VERSION, help="feature_version stored in JSON metadata.")
    parser.add_argument(
        "--extra-train-csv",
        action="append",
        default=[],
        metavar="PATH",
        help=(
            "Repeatable: path to an extra training-rows CSV (e.g. nr_train_mined.csv) "
            "appended to the nr_train.csv dataframe after schema validation."
        ),
    )
    args = parser.parse_args()

    data_dir = Path(args.data_dir)
    out_dir = Path(args.out_dir)
    seed = int(args.seed)
    rng = np.random.default_rng(seed)
    started_utc = datetime.now(timezone.utc).isoformat()

    # Refuse to train on a smoke-test build (--limit-sessions leaves a trace
    # in the dataset summary and is not the spec dataset).
    summary_path = data_dir / "nr_dataset_summary.json"
    if summary_path.exists():
        dataset_summary = json.loads(summary_path.read_text(encoding="utf-8"))
        limited = dataset_summary.get("limit_sessions") or dataset_summary.get("args", {}).get("limit_sessions")
        if limited:
            raise SystemExit(
                f"Dataset in {data_dir} was built with --limit-sessions {limited}; "
                "rebuild the full dataset before training."
            )

    train_df = load_split_csv(data_dir / "nr_train.csv")
    val_df = load_split_csv(data_dir / "nr_val.csv")

    extra_train_csvs: list[dict] = []
    for extra_arg in args.extra_train_csv:
        extra_path = Path(extra_arg)
        if not extra_path.exists():
            raise FileNotFoundError(f"--extra-train-csv not found: {extra_path}")
        extra_df = pd.read_csv(extra_path)
        if list(extra_df.columns) != list(train_df.columns):
            raise ValueError(
                f"--extra-train-csv {extra_path} columns do not match nr_train.csv columns."
            )
        train_df = pd.concat([train_df, extra_df], ignore_index=True)
        extra_train_csvs.append({"path": str(extra_path), "rows": int(len(extra_df))})
        print(f"Extra train rows appended from {extra_path}: {len(extra_df)} (train total now {len(train_df)})")

    missing_meta = [col for col in META_COLS if col not in train_df.columns]
    if missing_meta:
        raise ValueError(f"nr_train.csv is missing meta columns: {missing_meta}")

    feature_cols = [col for col in train_df.columns if col not in META_COLS]
    val_feature_cols = [col for col in val_df.columns if col not in META_COLS]
    if feature_cols != val_feature_cols:
        raise ValueError("nr_train.csv and nr_val.csv feature columns differ.")

    feature_sets = resolve_feature_sets(feature_cols)
    nan_replaced = {
        "nr_train": sanitize_features(train_df, feature_cols, "nr_train.csv"),
        "nr_val": sanitize_features(val_df, feature_cols, "nr_val.csv"),
    }

    train_labels = sorted(train_df["label"].astype(str).unique())
    if train_labels != sorted(nr_config.CLASSES):
        raise ValueError(
            f"Train labels {train_labels} do not match expected classes {sorted(nr_config.CLASSES)}."
        )
    unexpected_val = sorted(set(val_df["label"].astype(str).unique()) - set(nr_config.CLASSES))
    if unexpected_val:
        raise ValueError(f"nr_val.csv contains unexpected labels: {unexpected_val}")

    le = LabelEncoder()
    y_train = le.fit_transform(train_df["label"].astype(str).to_numpy())
    class_names = le.classes_.tolist()
    if class_names != nr_config.CLASSES:
        raise ValueError(f"LabelEncoder classes {class_names} != nr_config.CLASSES {nr_config.CLASSES}.")

    val_clean = select_val_clean(val_df)
    y_val = le.transform(val_clean["label"].astype(str).to_numpy())

    groups = train_df["group_id"].astype(str).to_numpy()
    cv, n_splits = build_group_cv(train_df)

    print(f"Train rows: {len(train_df)}  |  labels: {train_df['label'].value_counts().to_dict()}")
    print(f"Val rows: {len(val_df)}  |  clean reviewed_marker eval rows: {len(val_clean)}")
    print(f"Val eval labels: {val_clean['label'].value_counts().to_dict()}")
    print(f"Feature sets: {[f'{k}({len(v)})' for k, v in feature_sets.items()]}")
    print(f"Grouped CV: GroupKFold({n_splits}) over {train_df['group_id'].nunique()} group_id values")

    out_dir.mkdir(parents=True, exist_ok=True)

    results: list[dict] = []
    fitted: dict[tuple[str, str], tuple] = {}
    for set_name in FEATURE_SET_ORDER:
        if set_name not in feature_sets:
            continue
        cols = feature_sets[set_name]
        X_train_raw = train_df[cols].to_numpy(dtype=np.float64)
        X_val_raw = val_clean[cols].to_numpy(dtype=np.float64)

        scaler = StandardScaler()
        X_train = scaler.fit_transform(X_train_raw)
        X_val = scaler.transform(X_val_raw)

        joblib.dump(scaler, out_dir / f"nr_scaler_{set_name}.pkl")
        joblib.dump(list(cols), out_dir / f"nr_feature_cols_{set_name}.pkl")

        for model_name in ("rf", "histgb"):
            print(f"\n=== Training {model_name} on {set_name} ({len(cols)} features) ===")
            if model_name == "rf":
                clf, train_info = train_rf(X_train, y_train, groups, cv, seed)
                cv_f1 = train_info["cv_best_f1_macro"]
            else:
                clf, train_info = train_histgb(X_train, y_train, groups, cv, seed)
                cv_f1 = train_info["cv_f1_macro_mean"]

            t0 = time.perf_counter()
            y_pred = clf.predict(X_val)
            eval_seconds = time.perf_counter() - t0
            val_metrics = compute_val_metrics(y_val, y_pred, class_names)

            joblib.dump(clf, out_dir / f"nr_{model_name}_{set_name}.pkl")
            fitted[(set_name, model_name)] = (clf, scaler, cols, X_val_raw)

            print(
                f"cv_f1_macro={cv_f1:.4f}  val_macro_f1={val_metrics['macro_f1']:.4f}  "
                f"racket P={val_metrics['racket_precision']:.4f} "
                f"R={val_metrics['racket_recall']:.4f}  "
                f"score={val_metrics['selection_score']:.4f}"
            )
            results.append(
                {
                    "feature_set": set_name,
                    "model": model_name,
                    "n_features": len(cols),
                    "cv_f1_macro": float(cv_f1),
                    "train": train_info,
                    "val": val_metrics,
                    "eval_seconds": float(eval_seconds),
                }
            )

    joblib.dump(le, out_dir / "nr_label_encoder.pkl")

    winner, constraint_met = select_winner(results)
    rf_results = [r for r in results if r["model"] == "rf"]
    best_rf, rf_constraint_met = select_winner(rf_results)
    if not constraint_met:
        print(
            f"\nWARNING: no combo reached racket_precision >= {RACKET_PRECISION_FLOOR} "
            "on val; reporting the best score anyway."
        )

    markdown = build_results_markdown(results, winner, constraint_met, best_rf)
    print("\n" + markdown)
    (out_dir / "val_results.md").write_text(markdown, encoding="utf-8")
    print(f"Val results written: {out_dir / 'val_results.md'}")

    feature_version = args.feature_version
    if best_rf["feature_set"] != "all83" and feature_version == DEFAULT_FEATURE_VERSION:
        feature_version = f"nr_features_{best_rf['n_features']}_v1"
        print(
            f"Note: best RF uses feature set {best_rf['feature_set']}; "
            f"feature_version adjusted to {feature_version}."
        )

    export_clf, export_scaler, export_cols, export_val_raw = fitted[(best_rf["feature_set"], "rf")]
    export_path = out_dir / "nr_audio_model.json"
    export_rf_json(
        export_clf,
        export_scaler,
        le,
        export_cols,
        best_rf["feature_set"],
        args.model_version,
        feature_version,
        export_path,
    )
    roundtrip = run_roundtrip_check(export_path, export_clf, export_scaler, export_val_raw, rng)

    training_log = {
        "generated_utc": started_utc,
        "script": str(Path(__file__).resolve()),
        "seed": seed,
        "data_dir": str(data_dir),
        "out_dir": str(out_dir),
        "train_rows": int(len(train_df)),
        "extra_train_csvs": extra_train_csvs,
        "val_rows": int(len(val_df)),
        "val_eval_rows": int(len(val_clean)),
        "train_label_counts": {str(k): int(v) for k, v in train_df["label"].value_counts().to_dict().items()},
        "val_eval_label_counts": {str(k): int(v) for k, v in val_clean["label"].value_counts().to_dict().items()},
        "classes": class_names,
        "cv": {"strategy": "GroupKFold", "n_splits": int(n_splits), "group_column": "group_id"},
        "nan_replaced": nan_replaced,
        "feature_sets": {name: {"n_features": len(cols), "columns": list(cols)} for name, cols in feature_sets.items()},
        "selection": {
            "rule": f"{SCORE_RECALL_WEIGHT}*racket_recall + {SCORE_PRECISION_WEIGHT}*racket_precision",
            "constraint": f"racket_precision >= {RACKET_PRECISION_FLOOR}",
            "constraint_met": constraint_met,
            "winner": {"feature_set": winner["feature_set"], "model": winner["model"], "score": winner["val"]["selection_score"]},
            "best_rf": {"feature_set": best_rf["feature_set"], "model": "rf", "score": best_rf["val"]["selection_score"], "constraint_met": rf_constraint_met},
        },
        "app_export": {
            "path": str(export_path),
            "model_version": args.model_version,
            "feature_version": feature_version,
            "feature_set": best_rf["feature_set"],
            "tree_count": len(export_clf.estimators_),
            "roundtrip_check": roundtrip,
        },
        "results": results,
        "library_versions": {
            "python": sys.version.split()[0],
            "numpy": np.__version__,
            "scipy": scipy.__version__,
            "pandas": pd.__version__,
            "sklearn": sklearn.__version__,
            "librosa": librosa.__version__,
            "joblib": joblib.__version__,
        },
    }
    log_path = out_dir / "training_log.json"
    log_path.write_text(json.dumps(training_log, indent=2), encoding="utf-8")
    print(f"Training log written: {log_path}")
    print(f"Artifacts saved under: {out_dir}")


if __name__ == "__main__":
    main()
