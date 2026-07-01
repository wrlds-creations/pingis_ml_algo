"""
T0049 targeted speech-veto candidate for the bundled Fable live detector.

This is an offline, evaluation-only experiment. It uses the Love-approved
T0047/T0048 Fable debug feature table to test whether a tiny post-count veto
can reject speech false positives without losing real racket counts.

It does not export a model, modify app JSON, build an APK, or change runtime
behavior.
"""

from __future__ import annotations

import argparse
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Callable

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.tree import DecisionTreeClassifier, export_text

ROOT_DIR = Path(__file__).resolve().parents[4]
DEFAULT_INPUT = (
    ROOT_DIR
    / "data"
    / "audio"
    / "models"
    / "evaluations"
    / "t0048_speech_fp_messy_recall"
    / "t0048_event_features.csv"
)
DEFAULT_OUT_DIR = (
    ROOT_DIR
    / "data"
    / "audio"
    / "models"
    / "evaluations"
    / "t0049_speech_veto_candidate"
)

META_COLS = {
    "round",
    "scenario",
    "expected",
    "true_context",
    "source_file",
    "event_index",
    "group",
    "counted",
    "saved_label",
    "offline_label",
    "reject_reason",
    "bg_mode",
    "decoded_wav",
}

APPROVED_RACKET_GROUPS = {
    "real_counted_high_conf",
    "real_counted_mid_conf",
    "real_counted_low_conf",
    "real_rejected_low_conf_racket_label",
}
APPROVED_NOISE_GROUPS = {
    "speech_false_positive",
    "speech_rejected",
}


@dataclass(frozen=True)
class CandidateSpec:
    name: str
    model: object


def parse_bool(value: object) -> bool:
    if isinstance(value, bool):
        return value
    return str(value).strip().lower() in {"1", "true", "yes", "y"}


def ensure_out_dir(path: Path) -> None:
    path.mkdir(parents=True, exist_ok=True)


def load_features(path: Path) -> pd.DataFrame:
    if not path.exists():
        raise FileNotFoundError(f"Missing T0048 feature CSV: {path}")
    df = pd.read_csv(path)
    if "counted" not in df.columns or "group" not in df.columns:
        raise ValueError("Input CSV must contain at least counted and group columns.")
    df = df.copy()
    df["counted_bool"] = df["counted"].map(parse_bool)
    return df


def numeric_feature_cols(df: pd.DataFrame) -> list[str]:
    cols: list[str] = []
    blocked = META_COLS | {
        "counted_bool",
        # Timing/order fields can encode the hand-run block rather than sound.
        "native_onset_time_ms",
        "event_index",
        "expected",
    }
    for col in df.columns:
        if col in blocked:
            continue
        if pd.api.types.is_numeric_dtype(df[col]):
            cols.append(col)
    return cols


def acoustic_feature_cols(df: pd.DataFrame) -> list[str]:
    blocked = {
        "saved_confidence",
        "saved_racket_prob",
        "saved_noise_prob",
        "offline_confidence",
    }
    return [col for col in numeric_feature_cols(df) if col not in blocked]


def counted_candidate_rows(df: pd.DataFrame) -> pd.DataFrame:
    counted = df[df["counted_bool"]].copy()
    wanted = counted["group"].isin(
        ["real_counted_high_conf", "real_counted_mid_conf", "real_counted_low_conf", "speech_false_positive"]
    )
    candidate = counted[wanted].copy()
    candidate["target_keep"] = (candidate["true_context"] == "racket_practice").astype(int)
    candidate["fold_real_scenario"] = candidate["round"].astype(str) + ":" + candidate["scenario"].astype(str)
    return candidate.reset_index(drop=True)


def context_training_rows(df: pd.DataFrame) -> pd.DataFrame:
    wanted = df["group"].isin(
        [
            "real_counted_high_conf",
            "real_counted_mid_conf",
            "real_counted_low_conf",
            "real_rejected_low_conf_racket_label",
            "speech_rejected",
        ]
    )
    out = df[wanted].copy()
    out["target_keep"] = out["group"].isin(APPROVED_RACKET_GROUPS).astype(int)
    out["fold_real_scenario"] = out["round"].astype(str) + ":" + out["scenario"].astype(str)
    return out.reset_index(drop=True)


def approved_extra_rows(df: pd.DataFrame, feature_cols: list[str]) -> pd.DataFrame:
    rows = df[df["group"].isin(APPROVED_RACKET_GROUPS | APPROVED_NOISE_GROUPS)].copy()
    labels = np.where(rows["group"].isin(APPROVED_RACKET_GROUPS), "racket_bounce", "noise")
    out = pd.DataFrame(
        {
            "clip_id": [
                f"t0049_{str(row.round)}_{str(row.scenario)}_ev{int(row.event_index):03d}"
                for row in rows.itertuples(index=False)
            ],
            "split": "train",
            "session_id": "t0047_fable_debug_love_approved",
            "wav_filename": rows["decoded_wav"].fillna("").astype(str),
            "scenario_id": rows["scenario"].astype(str),
            "background_condition": rows["bg_mode"].fillna("").astype(str),
            "label": labels,
            "source": "t0049_love_approved_fable_debug",
            "anchor_ms": rows["native_onset_time_ms"].fillna(0.0),
            "jitter_ms": 0,
            "augment": "none",
            "aug_bed": "",
            "group_id": rows["source_file"].astype(str),
            "close_event_bucket": rows["group"].astype(str),
        }
    )
    for col in feature_cols:
        out[col] = rows[col].to_numpy()
    return out


def evaluate_rule(df: pd.DataFrame, name: str, veto: Callable[[pd.DataFrame], pd.Series]) -> dict[str, object]:
    mask = veto(df).astype(bool)
    speech = df["target_keep"] == 0
    real = df["target_keep"] == 1
    return {
        "candidate": name,
        "type": "fixed_rule",
        "speech_total": int(speech.sum()),
        "speech_vetoed": int((mask & speech).sum()),
        "speech_veto_rate": float((mask & speech).sum() / max(1, speech.sum())),
        "real_total": int(real.sum()),
        "real_rejected": int((mask & real).sum()),
        "real_reject_rate": float((mask & real).sum() / max(1, real.sum())),
        "kept_real": int((~mask & real).sum()),
        "kept_speech": int((~mask & speech).sum()),
    }


def build_candidate_specs() -> list[CandidateSpec]:
    return [
        CandidateSpec(
            "logreg_balanced",
            Pipeline(
                [
                    ("imputer", SimpleImputer(strategy="median")),
                    ("scaler", StandardScaler()),
                    (
                        "clf",
                        LogisticRegression(
                            class_weight="balanced",
                            max_iter=2000,
                            random_state=20260628,
                        ),
                    ),
                ]
            ),
        ),
        CandidateSpec(
            "tree_depth1_balanced",
            Pipeline(
                [
                    ("imputer", SimpleImputer(strategy="median")),
                    (
                        "clf",
                        DecisionTreeClassifier(
                            max_depth=1,
                            class_weight="balanced",
                            random_state=20260628,
                        ),
                    ),
                ]
            ),
        ),
        CandidateSpec(
            "tree_depth2_balanced",
            Pipeline(
                [
                    ("imputer", SimpleImputer(strategy="median")),
                    (
                        "clf",
                        DecisionTreeClassifier(
                            max_depth=2,
                            min_samples_leaf=2,
                            class_weight="balanced",
                            random_state=20260628,
                        ),
                    ),
                ]
            ),
        ),
        CandidateSpec(
            "rf_tiny_balanced",
            Pipeline(
                [
                    ("imputer", SimpleImputer(strategy="median")),
                    (
                        "clf",
                        RandomForestClassifier(
                            n_estimators=200,
                            max_depth=3,
                            min_samples_leaf=1,
                            class_weight="balanced_subsample",
                            random_state=20260628,
                            n_jobs=-1,
                        ),
                    ),
                ]
            ),
        ),
    ]


def predict_keep_scores(model: object, x: pd.DataFrame) -> np.ndarray:
    if hasattr(model, "predict_proba"):
        probs = model.predict_proba(x)
        classes = list(getattr(model, "classes_", []))
        if 1 in classes:
            return probs[:, classes.index(1)]
        return probs[:, -1]
    return np.asarray(model.predict(x), dtype=float)


def evaluate_model_folds(df: pd.DataFrame, feature_cols: list[str], spec: CandidateSpec) -> tuple[list[dict], dict]:
    rows: list[dict] = []
    speech_indices = list(df.index[df["target_keep"] == 0])
    real_scenarios = sorted(df.loc[df["target_keep"] == 1, "fold_real_scenario"].unique())

    for speech_idx in speech_indices:
        for scenario in real_scenarios:
            test_mask = (df.index == speech_idx) | ((df["target_keep"] == 1) & (df["fold_real_scenario"] == scenario))
            train = df[~test_mask].copy()
            test = df[test_mask].copy()
            if train["target_keep"].nunique() < 2:
                continue
            model = spec.model
            model.fit(train[feature_cols], train["target_keep"])
            keep_score = predict_keep_scores(model, test[feature_cols])
            veto = keep_score < 0.5
            speech = test["target_keep"].to_numpy() == 0
            real = test["target_keep"].to_numpy() == 1
            rows.append(
                {
                    "candidate": spec.name,
                    "held_speech_row": int(speech_idx),
                    "held_speech_event": str(df.loc[speech_idx, "source_file"]) + f"#ev{int(df.loc[speech_idx, 'event_index']):03d}",
                    "held_real_scenario": scenario,
                    "speech_total": int(speech.sum()),
                    "speech_vetoed": int((veto & speech).sum()),
                    "real_total": int(real.sum()),
                    "real_rejected": int((veto & real).sum()),
                    "real_reject_rate": float((veto & real).sum() / max(1, real.sum())),
                }
            )

    fold_df = pd.DataFrame(rows)
    if fold_df.empty:
        summary = {
            "candidate": spec.name,
            "type": "trainable_fold",
            "folds": 0,
            "speech_veto_rate": 0.0,
            "mean_real_reject_rate": 1.0,
            "max_real_reject_rate": 1.0,
            "zero_real_loss_and_speech_caught_folds": 0,
        }
        return rows, summary

    summary = {
        "candidate": spec.name,
        "type": "trainable_fold",
        "folds": int(len(fold_df)),
        "speech_veto_rate": float(fold_df["speech_vetoed"].sum() / max(1, fold_df["speech_total"].sum())),
        "mean_real_reject_rate": float(fold_df["real_reject_rate"].mean()),
        "max_real_reject_rate": float(fold_df["real_reject_rate"].max()),
        "total_real_rejected": int(fold_df["real_rejected"].sum()),
        "total_real_tested": int(fold_df["real_total"].sum()),
        "zero_real_loss_and_speech_caught_folds": int(
            ((fold_df["speech_vetoed"] == fold_df["speech_total"]) & (fold_df["real_rejected"] == 0)).sum()
        ),
    }
    return rows, summary


def evaluate_context_model_folds(
    train_df: pd.DataFrame,
    test_counted_df: pd.DataFrame,
    feature_cols: list[str],
    spec: CandidateSpec,
) -> tuple[list[dict], list[dict], dict]:
    rows: list[dict] = []
    score_rows: list[dict] = []
    real_scenarios = sorted(test_counted_df.loc[test_counted_df["target_keep"] == 1, "fold_real_scenario"].unique())
    speech_test = test_counted_df[test_counted_df["target_keep"] == 0].copy()
    if speech_test.empty:
        raise ValueError("No counted speech false positives available for context-model test.")

    for scenario in real_scenarios:
        train = train_df[train_df["fold_real_scenario"] != scenario].copy()
        test = pd.concat(
            [
                speech_test,
                test_counted_df[(test_counted_df["target_keep"] == 1) & (test_counted_df["fold_real_scenario"] == scenario)],
            ],
            ignore_index=True,
        )
        if train["target_keep"].nunique() < 2:
            continue
        model = spec.model
        model.fit(train[feature_cols], train["target_keep"])
        keep_score = predict_keep_scores(model, test[feature_cols])
        veto = keep_score < 0.5
        speech = test["target_keep"].to_numpy() == 0
        real = test["target_keep"].to_numpy() == 1
        rows.append(
            {
                "candidate": spec.name,
                "held_real_scenario": scenario,
                "speech_total": int(speech.sum()),
                "speech_vetoed": int((veto & speech).sum()),
                "real_total": int(real.sum()),
                "real_rejected": int((veto & real).sum()),
                "real_reject_rate": float((veto & real).sum() / max(1, real.sum())),
            }
        )
        for row_idx, test_row in enumerate(test.itertuples(index=False)):
            score_rows.append(
                {
                    "candidate": spec.name,
                    "held_real_scenario": scenario,
                    "source_file": str(test_row.source_file),
                    "event_index": int(test_row.event_index),
                    "group": str(test_row.group),
                    "target_keep": int(test_row.target_keep),
                    "keep_score": float(keep_score[row_idx]),
                }
            )

    fold_df = pd.DataFrame(rows)
    if fold_df.empty:
        summary = {
            "candidate": spec.name,
            "type": "speech_rejected_context_fold",
            "folds": 0,
            "speech_veto_rate": 0.0,
            "mean_real_reject_rate": 1.0,
            "max_real_reject_rate": 1.0,
        }
        return rows, score_rows, summary
    summary = {
        "candidate": spec.name,
        "type": "speech_rejected_context_fold",
        "folds": int(len(fold_df)),
        "speech_veto_rate": float(fold_df["speech_vetoed"].sum() / max(1, fold_df["speech_total"].sum())),
        "mean_real_reject_rate": float(fold_df["real_reject_rate"].mean()),
        "max_real_reject_rate": float(fold_df["real_reject_rate"].max()),
        "total_real_rejected": int(fold_df["real_rejected"].sum()),
        "total_real_tested": int(fold_df["real_total"].sum()),
        "zero_real_loss_and_all_speech_caught_folds": int(
            ((fold_df["speech_vetoed"] == fold_df["speech_total"]) & (fold_df["real_rejected"] == 0)).sum()
        ),
    }
    return rows, score_rows, summary


def sweep_context_thresholds(score_rows: list[dict]) -> list[dict]:
    if not score_rows:
        return []
    scores = pd.DataFrame(score_rows)
    thresholds = sorted(set([0.01, 0.02, 0.03, 0.04, 0.05, 0.075, 0.1, 0.125, 0.15, 0.2, 0.25, 0.3, 0.4, 0.5]))
    rows: list[dict] = []
    for candidate, part in scores.groupby("candidate"):
        for threshold in thresholds:
            veto = part["keep_score"] < threshold
            speech = part["target_keep"] == 0
            real = part["target_keep"] == 1
            by_fold = part.assign(veto=veto, speech=speech, real=real).groupby("held_real_scenario")
            max_real_reject_rate = 0.0
            zero_real_all_speech_folds = 0
            for _, fold in by_fold:
                real_total = int(fold["real"].sum())
                real_rejected = int((fold["veto"] & fold["real"]).sum())
                speech_total = int(fold["speech"].sum())
                speech_vetoed = int((fold["veto"] & fold["speech"]).sum())
                max_real_reject_rate = max(max_real_reject_rate, real_rejected / max(1, real_total))
                if real_rejected == 0 and speech_vetoed == speech_total:
                    zero_real_all_speech_folds += 1
            rows.append(
                {
                    "candidate": candidate,
                    "keep_threshold_veto_below": threshold,
                    "speech_total": int(speech.sum()),
                    "speech_vetoed": int((veto & speech).sum()),
                    "speech_veto_rate": float((veto & speech).sum() / max(1, speech.sum())),
                    "real_total": int(real.sum()),
                    "real_rejected": int((veto & real).sum()),
                    "real_reject_rate": float((veto & real).sum() / max(1, real.sum())),
                    "max_real_reject_rate": float(max_real_reject_rate),
                    "zero_real_all_speech_folds": int(zero_real_all_speech_folds),
                }
            )
    return rows


def fit_apparent_model(df: pd.DataFrame, feature_cols: list[str], spec: CandidateSpec) -> tuple[dict, object]:
    model = spec.model
    model.fit(df[feature_cols], df["target_keep"])
    keep_score = predict_keep_scores(model, df[feature_cols])
    veto = keep_score < 0.5
    speech = df["target_keep"].to_numpy() == 0
    real = df["target_keep"].to_numpy() == 1
    result = {
        "candidate": spec.name,
        "type": "trainable_apparent",
        "speech_total": int(speech.sum()),
        "speech_vetoed": int((veto & speech).sum()),
        "speech_veto_rate": float((veto & speech).sum() / max(1, speech.sum())),
        "real_total": int(real.sum()),
        "real_rejected": int((veto & real).sum()),
        "real_reject_rate": float((veto & real).sum() / max(1, real.sum())),
        "kept_real": int((~veto & real).sum()),
        "kept_speech": int((~veto & speech).sum()),
    }
    return result, model


def model_note(model: object, feature_cols: list[str]) -> str:
    try:
        clf = model.named_steps.get("clf")  # type: ignore[attr-defined]
        if isinstance(clf, DecisionTreeClassifier):
            return export_text(clf, feature_names=feature_cols, max_depth=3)
        if isinstance(clf, LogisticRegression):
            coefs = clf.coef_[0]
            ranked = sorted(zip(feature_cols, coefs), key=lambda pair: abs(pair[1]), reverse=True)[:12]
            return "\n".join(f"{name}: {coef:.4f}" for name, coef in ranked)
        if isinstance(clf, RandomForestClassifier):
            ranked = sorted(zip(feature_cols, clf.feature_importances_), key=lambda pair: pair[1], reverse=True)[:12]
            return "\n".join(f"{name}: {score:.4f}" for name, score in ranked)
    except Exception as exc:  # pragma: no cover - explanatory only
        return f"Could not describe model: {exc}"
    return ""


def write_markdown(
    path: Path,
    counted_df: pd.DataFrame,
    fixed_results: list[dict],
    fold_summaries: list[dict],
    apparent_results: list[dict],
    context_summaries: list[dict],
    context_threshold_rows: list[dict],
    notes: dict[str, str],
    extra_rows: pd.DataFrame,
) -> None:
    lines: list[str] = []
    lines.append("# T0049 Speech Veto Candidate")
    lines.append("")
    lines.append("## Scope")
    lines.append("")
    lines.append("- Offline candidate only.")
    lines.append("- Input is the T0048 event feature table from Love-approved June 28 Fable debug clips.")
    lines.append("- No model JSON, app runtime, APK, raw labels, Roboflow/cloud/API, or AWS changes.")
    lines.append("")
    lines.append("## Counted Candidate Set")
    lines.append("")
    lines.append(f"- Counted real racket events: `{int((counted_df['target_keep'] == 1).sum())}`.")
    lines.append(f"- Counted speech false positives: `{int((counted_df['target_keep'] == 0).sum())}`.")
    lines.append("")
    lines.append("Real counted events by scenario:")
    lines.append("")
    real_counts = (
        counted_df[counted_df["target_keep"] == 1]
        .groupby("fold_real_scenario")
        .size()
        .reset_index(name="count")
        .sort_values("fold_real_scenario")
    )
    lines.append("| Scenario | Count |")
    lines.append("|---|---:|")
    for row in real_counts.itertuples(index=False):
        lines.append(f"| `{row.fold_real_scenario}` | {int(row.count)} |")
    lines.append("")
    lines.append("## Fixed Gates")
    lines.append("")
    lines.append("| Candidate | Speech Vetoed | Real Rejected | Real Reject Rate |")
    lines.append("|---|---:|---:|---:|")
    for row in fixed_results:
        lines.append(
            f"| `{row['candidate']}` | {row['speech_vetoed']}/{row['speech_total']} "
            f"| {row['real_rejected']}/{row['real_total']} | {row['real_reject_rate']:.3f} |"
        )
    lines.append("")
    lines.append("## Trainable Veto Folds")
    lines.append("")
    lines.append(
        "Each fold holds out one speech false-positive event and one real-racket scenario. "
        "This is intentionally harsh for a dataset with only four speech false positives."
    )
    lines.append("")
    lines.append("| Candidate | Folds | Held Speech Veto Rate | Mean Real Reject Rate | Max Real Reject Rate | Zero-Loss Speech-Caught Folds |")
    lines.append("|---|---:|---:|---:|---:|---:|")
    for row in fold_summaries:
        lines.append(
            f"| `{row['candidate']}` | {row['folds']} | {row['speech_veto_rate']:.3f} "
            f"| {row['mean_real_reject_rate']:.3f} | {row['max_real_reject_rate']:.3f} "
            f"| {row['zero_real_loss_and_speech_caught_folds']} |"
        )
    lines.append("")
    lines.append("## Apparent Fit")
    lines.append("")
    lines.append("Apparent fit trains and tests on the same counted rows. It is over-optimistic and is shown only to inspect whether the features can memorize the tiny set.")
    lines.append("")
    lines.append("| Candidate | Speech Vetoed | Real Rejected | Real Reject Rate |")
    lines.append("|---|---:|---:|---:|")
    for row in apparent_results:
        lines.append(
            f"| `{row['candidate']}` | {row['speech_vetoed']}/{row['speech_total']} "
            f"| {row['real_rejected']}/{row['real_total']} | {row['real_reject_rate']:.3f} |"
        )
    lines.append("")
    lines.append("## Speech-Rejected Context Model")
    lines.append("")
    lines.append(
        "This variant trains on real racket rows plus speech rows that the existing app already rejected, "
        "then tests against the four speech false positives and a held-out real-racket scenario. "
        "It uses acoustic features only, excluding the current model's saved probabilities/confidence."
    )
    lines.append("")
    lines.append("| Candidate | Folds | Speech Veto Rate | Mean Real Reject Rate | Max Real Reject Rate | Zero-Loss All-Speech Folds |")
    lines.append("|---|---:|---:|---:|---:|---:|")
    for row in context_summaries:
        lines.append(
            f"| `{row['candidate']}` | {row['folds']} | {row['speech_veto_rate']:.3f} "
            f"| {row['mean_real_reject_rate']:.3f} | {row['max_real_reject_rate']:.3f} "
            f"| {row.get('zero_real_loss_and_all_speech_caught_folds', 0)} |"
        )
    lines.append("")
    lines.append("Conservative context-threshold sweep, sorted by zero real loss and speech caught:")
    lines.append("")
    lines.append("| Candidate | Threshold | Speech Vetoed | Real Rejected | Max Fold Real Reject Rate |")
    lines.append("|---|---:|---:|---:|---:|")
    context_threshold_df = pd.DataFrame(context_threshold_rows)
    if not context_threshold_df.empty:
        ranked = context_threshold_df.sort_values(
            ["real_rejected", "speech_vetoed", "max_real_reject_rate"],
            ascending=[True, False, True],
        ).head(12)
        for row in ranked.itertuples(index=False):
            lines.append(
                f"| `{row.candidate}` | {float(row.keep_threshold_veto_below):.3f} "
                f"| {int(row.speech_vetoed)}/{int(row.speech_total)} "
                f"| {int(row.real_rejected)}/{int(row.real_total)} "
                f"| {float(row.max_real_reject_rate):.3f} |"
            )
    lines.append("")
    lines.append("## Candidate Notes")
    lines.append("")
    for name, note in notes.items():
        lines.append(f"### `{name}`")
        lines.append("")
        lines.append("```text")
        lines.append(note.strip() or "(no model note)")
        lines.append("```")
        lines.append("")
    lines.append("## Extra Rows")
    lines.append("")
    lines.append(
        f"Wrote `{len(extra_rows)}` approved extra rows for future full retraining when the full noise-robust training split is available."
    )
    lines.append("")
    lines.append("## Interpretation")
    lines.append("")
    lines.append(
        "- This tiny approved dataset can help build future training rows, but it is still too small to promote a trainable speech veto by itself."
    )
    lines.append(
        "- A candidate should only move toward app runtime after it passes a larger local holdout with speech/no-bounce negatives and normal/messy racket positives."
    )
    lines.append("")
    path.write_text("\n".join(lines), encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser(description="Evaluate T0049 speech-veto candidates from T0048 feature rows.")
    parser.add_argument("--input", type=Path, default=DEFAULT_INPUT)
    parser.add_argument("--out-dir", type=Path, default=DEFAULT_OUT_DIR)
    args = parser.parse_args()

    ensure_out_dir(args.out_dir)
    df = load_features(args.input)
    feature_cols = numeric_feature_cols(df)
    acoustic_cols = acoustic_feature_cols(df)
    counted = counted_candidate_rows(df)
    context_rows = context_training_rows(df)
    if counted.empty:
        raise ValueError("No counted candidate rows found.")
    if int((counted["target_keep"] == 0).sum()) < 2:
        raise ValueError("Need at least two speech false-positive rows for this audit.")

    fixed_rules = [
        ("t0048_all_speech_peak_veto_clip_abs_peak_lt_0.22299", lambda d: d["clip_abs_peak"] < 0.222991943359375),
        ("t0048_partial_veto_nr_bp_peak_ratio_gt_0.76144", lambda d: d["nr_bp_peak_ratio"] > 0.76144),
        ("confidence_only_lt_0.95", lambda d: d["saved_confidence"] < 0.95),
        ("confidence_only_lt_0.90", lambda d: d["saved_confidence"] < 0.90),
    ]
    fixed_results = [evaluate_rule(counted, name, rule) for name, rule in fixed_rules]

    fold_rows: list[dict] = []
    fold_summaries: list[dict] = []
    context_fold_rows: list[dict] = []
    context_score_rows: list[dict] = []
    context_summaries: list[dict] = []
    apparent_results: list[dict] = []
    notes: dict[str, str] = {}
    for spec in build_candidate_specs():
        rows, summary = evaluate_model_folds(counted, feature_cols, spec)
        fold_rows.extend(rows)
        fold_summaries.append(summary)
        context_rows_fold, context_scores_fold, context_summary = evaluate_context_model_folds(
            context_rows, counted, acoustic_cols, spec
        )
        context_fold_rows.extend(context_rows_fold)
        context_score_rows.extend(context_scores_fold)
        context_summaries.append(context_summary)
        apparent, fitted = fit_apparent_model(counted, feature_cols, spec)
        apparent_results.append(apparent)
        notes[spec.name] = model_note(fitted, feature_cols)

    extra = approved_extra_rows(df, feature_cols)
    context_threshold_rows = sweep_context_thresholds(context_score_rows)

    pd.DataFrame(fixed_results).to_csv(args.out_dir / "t0049_fixed_gate_results.csv", index=False)
    pd.DataFrame(fold_rows).to_csv(args.out_dir / "t0049_trainable_veto_folds.csv", index=False)
    pd.DataFrame(fold_summaries).to_csv(args.out_dir / "t0049_trainable_veto_summary.csv", index=False)
    pd.DataFrame(context_fold_rows).to_csv(args.out_dir / "t0049_context_veto_folds.csv", index=False)
    pd.DataFrame(context_score_rows).to_csv(args.out_dir / "t0049_context_veto_scores.csv", index=False)
    pd.DataFrame(context_summaries).to_csv(args.out_dir / "t0049_context_veto_summary.csv", index=False)
    pd.DataFrame(context_threshold_rows).to_csv(args.out_dir / "t0049_context_threshold_sweep.csv", index=False)
    pd.DataFrame(apparent_results).to_csv(args.out_dir / "t0049_apparent_fit_results.csv", index=False)
    extra.to_csv(args.out_dir / "t0049_love_approved_extra_rows.csv", index=False)
    (args.out_dir / "t0049_model_notes.json").write_text(json.dumps(notes, indent=2), encoding="utf-8")
    write_markdown(
        args.out_dir / "t0049_speech_veto_candidate_summary.md",
        counted,
        fixed_results,
        fold_summaries,
        apparent_results,
        context_summaries,
        context_threshold_rows,
        notes,
        extra,
    )

    print(f"wrote {args.out_dir}")
    print(f"counted_real={int((counted['target_keep'] == 1).sum())}")
    print(f"speech_fp={int((counted['target_keep'] == 0).sum())}")
    for row in fold_summaries:
        print(
            row["candidate"],
            f"speech_veto_rate={row['speech_veto_rate']:.3f}",
            f"mean_real_reject_rate={row['mean_real_reject_rate']:.3f}",
            f"max_real_reject_rate={row['max_real_reject_rate']:.3f}",
        )
    print("context model:")
    for row in context_summaries:
        print(
            row["candidate"],
            f"speech_veto_rate={row['speech_veto_rate']:.3f}",
            f"mean_real_reject_rate={row['mean_real_reject_rate']:.3f}",
            f"max_real_reject_rate={row['max_real_reject_rate']:.3f}",
        )


if __name__ == "__main__":
    main()
