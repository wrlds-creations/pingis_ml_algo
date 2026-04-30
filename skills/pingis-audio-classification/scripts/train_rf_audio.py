"""
train_rf_audio.py

Train a RandomForest classifier on audio_dataset.csv and save the model,
scaler, label encoder, and feature order under data/audio/models/.

This version evaluates with grouped splits by original recording so augmented
rows from the same source cannot leak across train and test.

Run: python skills/pingis-audio-classification/scripts/train_rf_audio.py
"""

import warnings
from pathlib import Path

import joblib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import ConfusionMatrixDisplay, classification_report, confusion_matrix
from sklearn.model_selection import GridSearchCV, StratifiedKFold, train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler

warnings.filterwarnings("ignore")

ROOT_DIR = Path(__file__).resolve().parents[3]
DATASET = ROOT_DIR / "data" / "audio" / "processed" / "audio_dataset.csv"
MODEL_DIR = ROOT_DIR / "data" / "audio" / "models"

TARGET_LABELS = ["racket_bounce", "table_bounce", "floor_bounce", "noise"]
META_COLS = {
    "label",
    "recorder_name",
    "session_id",
    "source_file",
    "group_id",
    "scenario_id",
    "background_condition",
    "take_index",
    "target_duration_s",
    "clip_id",
    "augmentation",
    "onset_index",
    "review_completed",
    "marker_source",
    "anchor_rule",
}
SCENARIO_BREAKDOWN_IDS = [
    "racket_quiet",
    "racket_counting",
    "racket_music_low",
    "racket_music_mid",
    "speech_only",
    "desk_keyboard_only",
]


def make_group_split(df: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    work = df.copy()
    if "group_id" not in work.columns:
        if "source_file" in work.columns:
            work["group_id"] = work["source_file"].astype(str)
        else:
            work["group_id"] = work.index.astype(str)

    groups = work.groupby("group_id", as_index=False).agg(label=("label", "first"))
    min_groups_per_label = int(groups["label"].value_counts().min())
    if min_groups_per_label < 2:
        raise ValueError(
            "Need at least 2 unique recording groups per class for grouped train/test split."
        )
    train_groups, test_groups = train_test_split(
        groups["group_id"],
        test_size=0.2,
        random_state=42,
        stratify=groups["label"],
    )

    train_df = work[work["group_id"].isin(train_groups)].copy().reset_index(drop=True)
    test_df = work[work["group_id"].isin(test_groups)].copy().reset_index(drop=True)
    return train_df, test_df


def build_group_cv_splits(train_df: pd.DataFrame) -> tuple[list[tuple[np.ndarray, np.ndarray]], int]:
    groups = train_df.groupby("group_id", as_index=False).agg(label=("label", "first"))
    min_groups_per_label = int(groups["label"].value_counts().min())
    if min_groups_per_label < 2:
        raise ValueError(
            "Need at least 2 unique recording groups per class in the training split for grouped CV."
        )
    n_splits = min(5, min_groups_per_label)

    splitter = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=42)
    group_ids = groups["group_id"].to_numpy()
    group_labels = groups["label"].to_numpy()
    row_groups = train_df["group_id"].to_numpy()
    row_index = np.arange(len(train_df))

    splits: list[tuple[np.ndarray, np.ndarray]] = []
    for train_group_idx, val_group_idx in splitter.split(group_ids, group_labels):
        train_group_ids = group_ids[train_group_idx]
        val_group_ids = group_ids[val_group_idx]
        train_rows = row_index[np.isin(row_groups, train_group_ids)]
        val_rows = row_index[np.isin(row_groups, val_group_ids)]
        splits.append((train_rows, val_rows))

    return splits, n_splits


def print_scenario_breakdown(test_df: pd.DataFrame, y_true: np.ndarray, y_pred: np.ndarray, le: LabelEncoder) -> None:
    if "scenario_id" not in test_df.columns:
        print("\nScenario breakdown: scenario_id missing in dataset.")
        return

    print("\nScenario breakdown:")
    y_true_labels = le.inverse_transform(y_true)
    y_pred_labels = le.inverse_transform(y_pred)

    for scenario_id in SCENARIO_BREAKDOWN_IDS:
        mask = test_df["scenario_id"].astype(str) == scenario_id
        rows = int(mask.sum())
        if rows == 0:
            print(f"  {scenario_id}: no grouped test rows")
            continue

        expected = pd.Series(y_true_labels[mask.to_numpy()]).value_counts().to_dict()
        predicted = pd.Series(y_pred_labels[mask.to_numpy()]).value_counts().to_dict()
        accuracy = float(np.mean(y_true_labels[mask.to_numpy()] == y_pred_labels[mask.to_numpy()]))
        print(
            f"  {scenario_id}: rows={rows} | exact_match={accuracy:.3f}"
            f" | expected={expected} | predicted={predicted}"
        )


def main() -> None:
    if not DATASET.exists():
        print(f"Dataset missing: {DATASET}")
        print("Run preprocess_audio.py first.")
        return

    df = pd.read_csv(DATASET)
    df = df[df["label"].isin(TARGET_LABELS)].copy()
    if "group_id" not in df.columns:
        df["group_id"] = df["source_file"].astype(str) if "source_file" in df.columns else df.index.astype(str)
    if "scenario_id" not in df.columns:
        df["scenario_id"] = "legacy_unspecified"

    min_samples = 5
    counts = df["label"].value_counts()
    drop = counts[counts < min_samples].index.tolist()
    if drop:
        print(f"Skipping classes with < {min_samples} samples: {drop}")
        df = df[~df["label"].isin(drop)].copy()

    print(f"Loaded {len(df)} rows  |  labels: {df['label'].value_counts().to_dict()}")

    feature_cols = [column for column in df.columns if column not in META_COLS]
    train_df, test_df = make_group_split(df)

    print(
        "Grouped split"
        f"  |  train rows {len(train_df)} / test rows {len(test_df)}"
        f"  |  train groups {train_df['group_id'].nunique()} / test groups {test_df['group_id'].nunique()}"
    )
    print(f"  Train labels: {train_df['label'].value_counts().to_dict()}")
    print(f"  Test labels: {test_df['label'].value_counts().to_dict()}")

    X_train_raw = train_df[feature_cols].values.astype(np.float32)
    X_test_raw = test_df[feature_cols].values.astype(np.float32)

    le = LabelEncoder()
    y_train = le.fit_transform(train_df["label"].values)
    y_test = le.transform(test_df["label"].values)

    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train_raw)
    X_test = scaler.transform(X_test_raw)

    param_grid = {
        "n_estimators": [200, 300],
        "max_depth": [None, 25],
        "min_samples_leaf": [1, 3],
    }
    base_clf = RandomForestClassifier(
        class_weight="balanced_subsample",
        random_state=42,
        n_jobs=-1,
    )
    cv_splits, n_splits = build_group_cv_splits(train_df)
    grid = GridSearchCV(
        base_clf,
        param_grid,
        cv=cv_splits,
        scoring="f1_macro",
        refit=True,
        verbose=0,
        n_jobs=1,
    )
    grid.fit(X_train, y_train)
    clf = grid.best_estimator_
    print(f"\nBest hyperparameters: {grid.best_params_}")
    print(f"Grouped CV F1 (macro, {n_splits} folds): {grid.best_score_:.3f}")

    y_pred = clf.predict(X_test)
    print("\nGrouped test report:\n")
    print(classification_report(y_test, y_pred, target_names=le.classes_, zero_division=0))
    print_scenario_breakdown(test_df, y_test, y_pred, le)

    X_full_raw = df[feature_cols].values.astype(np.float32)
    y_full = le.transform(df["label"].values)
    final_scaler = StandardScaler()
    X_full = final_scaler.fit_transform(X_full_raw)
    final_clf = RandomForestClassifier(
        **grid.best_params_,
        class_weight="balanced_subsample",
        random_state=42,
        n_jobs=-1,
    )
    final_clf.fit(X_full, y_full)

    importances = final_clf.feature_importances_
    indices = np.argsort(importances)[::-1]
    print("\nTop-15 features:")
    for rank in range(min(15, len(feature_cols))):
        idx = indices[rank]
        print(f"  {rank + 1:2d}. {feature_cols[idx]:30s}  {importances[idx]:.4f}")

    low_importance = [feature_cols[i] for i in range(len(feature_cols)) if importances[i] < 0.005]
    if low_importance:
        print(f"\nFeatures with importance < 0.5%: {low_importance}")

    MODEL_DIR.mkdir(parents=True, exist_ok=True)
    joblib.dump(final_clf, MODEL_DIR / "audio_rf_classifier.pkl")
    joblib.dump(final_scaler, MODEL_DIR / "audio_feature_scaler.pkl")
    joblib.dump(le, MODEL_DIR / "audio_label_encoder.pkl")
    joblib.dump(feature_cols, MODEL_DIR / "audio_feature_cols.pkl")
    print(f"\nFinal full-data model saved to {MODEL_DIR}")

    cm = confusion_matrix(y_test, y_pred)
    disp = ConfusionMatrixDisplay(cm, display_labels=le.classes_)
    disp.plot(colorbar=False)
    plt.title("Audio bounce confusion matrix (grouped test)")
    plt.tight_layout()
    out_fig = MODEL_DIR / "confusion_matrix.png"
    plt.savefig(out_fig, dpi=120)
    print(f"Confusion matrix saved: {out_fig}")

    fig, ax = plt.subplots(figsize=(10, 8))
    top_n = min(25, len(feature_cols))
    top_idx = indices[:top_n]
    ax.barh(range(top_n), importances[top_idx], align="center")
    ax.set_yticks(range(top_n))
    ax.set_yticklabels([feature_cols[i] for i in top_idx], fontsize=8)
    ax.invert_yaxis()
    ax.set_xlabel("Importance")
    ax.set_title("Top-25 feature importance")
    plt.tight_layout()
    fig.savefig(MODEL_DIR / "feature_importance.png", dpi=120)
    print(f"Feature importance saved: {MODEL_DIR / 'feature_importance.png'}")


if __name__ == "__main__":
    main()
