"""
T0057 autonomous Fable audio improvement loop.

This is an evaluation-only loop. It trains local diagnostic candidates from the
currently approved row material, replays them against exact C2 truth and weaker
block-level phone-debug targets, and writes a decision report with the next
required user input when local data is no longer enough.

It does not export app model JSON, change runtime thresholds, build/install an
APK, mutate raw labels, or use cloud services.
"""

from __future__ import annotations

import argparse
import base64
import json
import math
import sys
from collections import Counter
from pathlib import Path
from typing import Any

import joblib
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

SCRIPT_DIR = Path(__file__).resolve().parent
SCRIPTS_DIR = SCRIPT_DIR.parent
for _path in (str(SCRIPT_DIR), str(SCRIPTS_DIR)):
    if _path not in sys.path:
        sys.path.insert(0, _path)

import nr_config  # noqa: E402
import nr_features  # noqa: E402
from evaluate_t0050_fable_targeted_round import BLOCKS, boolish  # noqa: E402
from evaluate_t0056_fable_candidate_retrain_replay import (  # noqa: E402
    build_t0055_rows,
    evaluate_rows,
    finite_float,
    fit_candidate,
    load_t0049_rows,
    predict_candidate,
    read_csv,
    read_wav,
    replay_c2,
    row_training_sets,
    safe_feature,
    write_csv,
)
from evaluate_fable_audio_reliability_t0044 import (  # noqa: E402
    TT_SOUNDS_LICENSE,
    TT_SOUNDS_PAGE,
    discover_tt_records,
    discover_local_clip_records,
)
from preprocess_audio import load_audio  # noqa: E402

ROOT_DIR = Path(__file__).resolve().parents[4]

SESSION_ID = "fable_live_session_2026-06-28T16-26-01-662Z"
DEFAULT_T0049_ROWS = (
    ROOT_DIR
    / "data"
    / "audio"
    / "models"
    / "evaluations"
    / "t0049_speech_veto_candidate"
    / "t0049_love_approved_extra_rows.csv"
)
DEFAULT_T0055_TIMELINE = (
    ROOT_DIR
    / "data"
    / "audio"
    / "models"
    / "evaluations"
    / "t0055_fable_label_ingest_plan"
    / "t0055_reviewed_timeline.csv"
)
DEFAULT_T0055_CANDIDATES = (
    ROOT_DIR
    / "data"
    / "audio"
    / "models"
    / "evaluations"
    / "t0055_fable_label_ingest_plan"
    / "t0055_candidate_rows.csv"
)
DEFAULT_C2_WAV = (
    ROOT_DIR
    / "data"
    / "audio"
    / "raw"
    / "t0052_fable_continuous_debug_round"
    / "fable_live_debug"
    / f"{SESSION_ID}.wav"
)
DEFAULT_T0050_DEBUG_DIR = (
    ROOT_DIR
    / "data"
    / "audio"
    / "raw"
    / "t0050_fable_targeted_round"
    / "fable_live_debug"
)
DEFAULT_TT_ROOT = ROOT_DIR / "data" / "audio" / "external" / "tt_sounds"
DEFAULT_LOCAL_REVIEWED_ROOT = ROOT_DIR / "data" / "audio" / "raw"
DEFAULT_OUT_DIR = (
    ROOT_DIR
    / "data"
    / "audio"
    / "models"
    / "evaluations"
    / "t0057_fable_auto_improvement_loop"
)
LOOP_THRESHOLDS = [0.30, 0.40, 0.50, 0.60, 0.70, 0.80]


def loop_candidate_specs(seed: int) -> list[tuple[str, Any]]:
    return [
        (
            "logreg_balanced",
            make_pipeline(
                StandardScaler(),
                LogisticRegression(class_weight="balanced", max_iter=1500, random_state=seed),
            ),
        ),
        (
            "rf_fast_balanced",
            RandomForestClassifier(
                n_estimators=120,
                max_depth=16,
                min_samples_leaf=2,
                class_weight="balanced_subsample",
                random_state=seed,
                n_jobs=-1,
            ),
        ),
    ]


def decode_audio_b64(audio_b64: str) -> np.ndarray:
    pcm_i16 = np.frombuffer(base64.b64decode(audio_b64), dtype="<i2")
    return pcm_i16.astype(np.float32) / 32768.0


def should_score_t0050_event(event: dict[str, Any]) -> bool:
    label = str(event.get("model_label") or "")
    probs = event.get("model_probabilities") or {}
    prob_racket = finite_float(probs.get("racket_bounce"), 0.0)
    return (
        boolish(event.get("counted"))
        or label in {"racket_bounce", "table_bounce"}
        or prob_racket >= 0.10
    )


def truth_times_ms(timeline_path: Path) -> list[float]:
    rows = read_csv(timeline_path)
    out: list[float] = []
    for row in rows:
        if row.get("review_label") != "racket":
            continue
        time_s = finite_float(row.get("reviewed_time_s"))
        if math.isfinite(time_s):
            out.append(time_s * 1000.0)
    return out


def truth_times_ms_from_labels_csv(labels_path: Path) -> list[float]:
    """Read a simple review CSV for an optional held-out run.

    The review UIs have used a few field names across tickets, so this accepts
    the common label/time variants instead of locking the hook to one file shape.
    """
    if not labels_path.exists():
        return []
    label_fields = ("review_label", "label", "true_label", "contact_label")
    time_fields = ("reviewed_time_s", "adjusted_time_s", "time_s", "timestamp_s", "anchor_time_s")
    out: list[float] = []
    for row in read_csv(labels_path):
        label = ""
        for field in label_fields:
            label = str(row.get(field) or "").strip().lower()
            if label:
                break
        if label not in {"racket", "racket_bounce", "racket_contact"}:
            continue
        time_s = float("nan")
        for field in time_fields:
            time_s = finite_float(row.get(field))
            if math.isfinite(time_s):
                break
        if math.isfinite(time_s):
            out.append(time_s * 1000.0)
    return sorted(out)


class BinaryLiveCounter:
    """Small mirror of the T0056 app-like duplicate/echo suppression."""

    def __init__(self, threshold: float) -> None:
        self.threshold = threshold
        self.last_counted: tuple[float, float] | None = None
        self.group_start_ms: float | None = None

    def process(self, prob_racket: float, onset_ms: float, frame_rms: float) -> tuple[bool, str]:
        if self.last_counted is not None:
            since_counted = onset_ms - self.last_counted[0]
            rms_ratio = frame_rms / max(self.last_counted[1], 1e-9)
            if since_counted <= 250.0:
                if rms_ratio >= 1.1:
                    self.last_counted = (onset_ms, frame_rms)
                    self.group_start_ms = onset_ms
                    return False, "same_bounce"
                if rms_ratio <= 0.6:
                    return False, "echo_window"
                if since_counted < 150.0:
                    return False, "same_bounce"
            elif self.group_start_ms is not None and onset_ms - self.group_start_ms <= 80.0:
                return False, "group_window"
        if prob_racket < self.threshold:
            return False, "low_probability"
        self.last_counted = (onset_ms, frame_rms)
        self.group_start_ms = onset_ms
        return True, ""


def replay_t0050_block(
    *,
    model: Any,
    feature_names: list[str],
    threshold: float,
    debug_dir: Path,
    spec: Any,
) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    path = debug_dir / spec.filename
    if not path.exists():
        return {
            "block": spec.block,
            "description": spec.description,
            "kind": spec.kind,
            "filename": spec.filename,
            "missing": True,
        }, []

    payload = json.loads(path.read_text(encoding="utf-8"))
    events = sorted(
        list(payload.get("events") or []),
        key=lambda event: finite_float(event.get("native_onset_time_ms"), 0.0),
    )
    counter = BinaryLiveCounter(threshold)
    counted = 0
    saved_counted = 0
    audio_events = 0
    no_audio_events = 0
    reject_reasons: Counter[str] = Counter()
    event_rows: list[dict[str, Any]] = []

    for index, event in enumerate(events, start=1):
        saved_is_counted = boolish(event.get("counted"))
        saved_counted += 1 if saved_is_counted else 0
        onset_ms = finite_float(event.get("native_onset_time_ms"), 0.0)
        frame_rms = finite_float(event.get("native_rms"), 0.0)
        if not event.get("audio_b64"):
            no_audio_events += 1
            reject_reasons["missing_audio_b64"] += 1
            event_rows.append(
                {
                    "block": spec.block,
                    "event_index": event.get("index", index),
                    "onset_ms": onset_ms,
                    "candidate_counted": False,
                    "candidate_reject_reason": "missing_audio_b64",
                    "candidate_prob_racket_bounce": "",
                    "saved_counted": saved_is_counted,
                    "saved_label": event.get("model_label", ""),
                    "saved_reject_reason": event.get("reject_reason", ""),
                }
            )
            continue

        audio_events += 1
        clip = decode_audio_b64(str(event["audio_b64"]))
        features = nr_features.extract_all_features(clip, nr_config.TARGET_SR)
        prediction = predict_candidate(model, features, feature_names)
        prob = float(prediction["probabilities"]["racket_bounce"])
        is_counted, reason = counter.process(prob, onset_ms, frame_rms)
        counted += 1 if is_counted else 0
        reject_reasons[reason or "counted"] += 1
        event_rows.append(
            {
                "block": spec.block,
                "event_index": event.get("index", index),
                "onset_ms": onset_ms,
                "candidate_counted": is_counted,
                "candidate_reject_reason": reason,
                "candidate_prob_racket_bounce": prob,
                "saved_counted": saved_is_counted,
                "saved_label": event.get("model_label", ""),
                "saved_reject_reason": event.get("reject_reason", ""),
                "native_rms": frame_rms,
            }
        )

    expected = spec.expected
    delta = "" if expected is None else counted - int(expected)
    abs_error = "" if expected is None else abs(counted - int(expected))
    summary = {
        "block": spec.block,
        "description": spec.description,
        "kind": spec.kind,
        "filename": spec.filename,
        "threshold": threshold,
        "expected": "" if expected is None else expected,
        "saved_counted": saved_counted,
        "candidate_counted": counted,
        "candidate_minus_expected": delta,
        "candidate_abs_error": abs_error,
        "audio_events": audio_events,
        "missing_audio_events": no_audio_events,
        "candidate_reject_reason_counts": json.dumps(dict(sorted(reject_reasons.items())), sort_keys=True),
    }
    return summary, event_rows


def replay_t0050_blocks(
    *,
    model: Any,
    feature_names: list[str],
    threshold: float,
    debug_dir: Path,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]], dict[str, Any]]:
    summaries: list[dict[str, Any]] = []
    events: list[dict[str, Any]] = []
    for spec in BLOCKS:
        if spec.kind == "extra":
            continue
        summary, block_events = replay_t0050_block(
            model=model,
            feature_names=feature_names,
            threshold=threshold,
            debug_dir=debug_dir,
            spec=spec,
        )
        summaries.append(summary)
        events.extend(block_events)

    real_rows = [row for row in summaries if row.get("kind") == "real" and row.get("expected") != ""]
    negative_rows = [row for row in summaries if row.get("kind") == "negative"]
    real_expected = sum(int(row["expected"]) for row in real_rows)
    real_counted = sum(int(row["candidate_counted"]) for row in real_rows)
    real_abs_error = sum(int(row["candidate_abs_error"]) for row in real_rows)
    negative_false_counts = sum(int(row["candidate_counted"]) for row in negative_rows)
    saved_negative_false_counts = sum(int(row["saved_counted"]) for row in negative_rows)

    aggregate = {
        "t0050_real_expected": real_expected,
        "t0050_real_candidate_counted": real_counted,
        "t0050_real_abs_error": real_abs_error,
        "t0050_negative_false_counts": negative_false_counts,
        "t0050_saved_negative_false_counts": saved_negative_false_counts,
        "t0050_negative_improvement": saved_negative_false_counts - negative_false_counts,
    }
    return summaries, events, aggregate


def build_t0050_feature_cache(debug_dir: Path) -> list[dict[str, Any]]:
    """Decode/extract T0050 event features once so the candidate sweep is fast."""
    rows: list[dict[str, Any]] = []
    for spec in BLOCKS:
        if spec.kind == "extra":
            continue
        path = debug_dir / spec.filename
        if not path.exists():
            rows.append(
                {
                    "block": spec.block,
                    "description": spec.description,
                    "kind": spec.kind,
                    "filename": spec.filename,
                    "expected": spec.expected,
                    "missing_file": True,
                    "has_audio": False,
                    "features": None,
                }
            )
            continue
        payload = json.loads(path.read_text(encoding="utf-8"))
        events = sorted(
            list(payload.get("events") or []),
            key=lambda event: finite_float(event.get("native_onset_time_ms"), 0.0),
        )
        for index, event in enumerate(events, start=1):
            feature_payload = None
            has_audio = bool(event.get("audio_b64"))
            score_event = has_audio and should_score_t0050_event(event)
            if score_event:
                clip = decode_audio_b64(str(event["audio_b64"]))
                feature_payload = nr_features.extract_all_features(clip, nr_config.TARGET_SR)
            rows.append(
                {
                    "block": spec.block,
                    "description": spec.description,
                    "kind": spec.kind,
                    "filename": spec.filename,
                    "expected": spec.expected,
                    "missing_file": False,
                    "event_index": event.get("index", index),
                    "onset_ms": finite_float(event.get("native_onset_time_ms"), 0.0),
                    "frame_rms": finite_float(event.get("native_rms"), 0.0),
                    "has_audio": has_audio,
                    "score_event": score_event,
                    "features": feature_payload,
                    "saved_counted": boolish(event.get("counted")),
                    "saved_label": event.get("model_label", ""),
                    "saved_reject_reason": event.get("reject_reason", ""),
                }
            )
    return rows


def replay_t0050_feature_cache(
    *,
    model: Any,
    feature_names: list[str],
    threshold: float,
    cache_rows: list[dict[str, Any]],
) -> tuple[list[dict[str, Any]], list[dict[str, Any]], dict[str, Any]]:
    summaries: list[dict[str, Any]] = []
    events_out: list[dict[str, Any]] = []
    blocks = sorted({str(row["block"]) for row in cache_rows})
    for block in blocks:
        block_rows = [row for row in cache_rows if row["block"] == block]
        first = block_rows[0]
        if first.get("missing_file"):
            summaries.append(
                {
                    "block": block,
                    "description": first.get("description", ""),
                    "kind": first.get("kind", ""),
                    "filename": first.get("filename", ""),
                    "missing": True,
                }
            )
            continue

        counter = BinaryLiveCounter(threshold)
        counted = 0
        saved_counted = 0
        audio_events = 0
        missing_audio_events = 0
        unscored_low_info_events = 0
        reject_reasons: Counter[str] = Counter()
        for row in block_rows:
            saved_counted += 1 if row.get("saved_counted") else 0
            if row.get("has_audio") and not row.get("score_event"):
                unscored_low_info_events += 1
                reject_reasons["not_scored_low_info"] += 1
                events_out.append(
                    {
                        "block": block,
                        "event_index": row.get("event_index", ""),
                        "onset_ms": row.get("onset_ms", ""),
                        "candidate_counted": False,
                        "candidate_reject_reason": "not_scored_low_info",
                        "candidate_prob_racket_bounce": "",
                        "saved_counted": row.get("saved_counted", False),
                        "saved_label": row.get("saved_label", ""),
                        "saved_reject_reason": row.get("saved_reject_reason", ""),
                    }
                )
                continue
            if not row.get("has_audio") or row.get("features") is None:
                missing_audio_events += 1
                reject_reasons["missing_audio_b64"] += 1
                events_out.append(
                    {
                        "block": block,
                        "event_index": row.get("event_index", ""),
                        "onset_ms": row.get("onset_ms", ""),
                        "candidate_counted": False,
                        "candidate_reject_reason": "missing_audio_b64",
                        "candidate_prob_racket_bounce": "",
                        "saved_counted": row.get("saved_counted", False),
                        "saved_label": row.get("saved_label", ""),
                        "saved_reject_reason": row.get("saved_reject_reason", ""),
                    }
                )
                continue

            audio_events += 1
            prediction = predict_candidate(model, row["features"], feature_names)
            prob = float(prediction["probabilities"]["racket_bounce"])
            is_counted, reason = counter.process(prob, float(row["onset_ms"]), float(row["frame_rms"]))
            counted += 1 if is_counted else 0
            reject_reasons[reason or "counted"] += 1
            events_out.append(
                {
                    "block": block,
                    "event_index": row.get("event_index", ""),
                    "onset_ms": row.get("onset_ms", ""),
                    "candidate_counted": is_counted,
                    "candidate_reject_reason": reason,
                    "candidate_prob_racket_bounce": prob,
                    "saved_counted": row.get("saved_counted", False),
                    "saved_label": row.get("saved_label", ""),
                    "saved_reject_reason": row.get("saved_reject_reason", ""),
                    "native_rms": row.get("frame_rms", ""),
                }
            )

        expected = first.get("expected")
        delta = "" if expected is None else counted - int(expected)
        abs_error = "" if expected is None else abs(counted - int(expected))
        summaries.append(
            {
                "block": block,
                "description": first.get("description", ""),
                "kind": first.get("kind", ""),
                "filename": first.get("filename", ""),
                "threshold": threshold,
                "expected": "" if expected is None else expected,
                "saved_counted": saved_counted,
                "candidate_counted": counted,
                "candidate_minus_expected": delta,
                "candidate_abs_error": abs_error,
                "audio_events": audio_events,
                "missing_audio_events": missing_audio_events,
                "unscored_low_info_events": unscored_low_info_events,
                "candidate_reject_reason_counts": json.dumps(dict(sorted(reject_reasons.items())), sort_keys=True),
            }
        )

    real_rows = [row for row in summaries if row.get("kind") == "real" and row.get("expected") != ""]
    negative_rows = [row for row in summaries if row.get("kind") == "negative"]
    real_expected = sum(int(row["expected"]) for row in real_rows)
    real_counted = sum(int(row["candidate_counted"]) for row in real_rows)
    real_abs_error = sum(int(row["candidate_abs_error"]) for row in real_rows)
    negative_false_counts = sum(int(row["candidate_counted"]) for row in negative_rows)
    saved_negative_false_counts = sum(int(row["saved_counted"]) for row in negative_rows)
    aggregate = {
        "t0050_real_expected": real_expected,
        "t0050_real_candidate_counted": real_counted,
        "t0050_real_abs_error": real_abs_error,
        "t0050_negative_false_counts": negative_false_counts,
        "t0050_saved_negative_false_counts": saved_negative_false_counts,
        "t0050_negative_improvement": saved_negative_false_counts - negative_false_counts,
    }
    return summaries, events_out, aggregate


def t0055_fold_metrics(
    *,
    base_rows: list[dict[str, Any]],
    c2_rows: list[dict[str, Any]],
    feature_names: list[str],
    seed: int,
) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    if not c2_rows:
        return rows
    label_to_indices: dict[str, list[int]] = {}
    for idx, row in enumerate(c2_rows):
        label_to_indices.setdefault(str(row["label"]), []).append(idx)
    fold_count = max(2, min(5, *(len(v) for v in label_to_indices.values())))
    folds: list[list[int]] = [[] for _ in range(fold_count)]
    for indices in label_to_indices.values():
        for offset, idx in enumerate(indices):
            folds[offset % fold_count].append(idx)

    for candidate_name, _proto in loop_candidate_specs(seed):
        correct = 0
        probs: list[float] = []
        heldout_total = 0
        for holdout_indices in folds:
            holdout_set = set(holdout_indices)
            train_rows = base_rows + [row for idx, row in enumerate(c2_rows) if idx not in holdout_set]
            estimator = dict(loop_candidate_specs(seed))[candidate_name]
            model = fit_candidate(estimator, train_rows, feature_names, use_weights=True)
            for holdout_idx in holdout_indices:
                holdout = c2_rows[holdout_idx]
                prediction = predict_candidate(model, holdout, feature_names)
                prob = float(prediction["probabilities"]["racket_bounce"])
                pred_label = "racket_bounce" if prob >= 0.5 else "noise"
                expected = str(holdout["label"])
                correct += 1 if pred_label == expected else 0
                heldout_total += 1
                probs.append(prob if expected == "racket_bounce" else 1.0 - prob)
        rows.append(
            {
                "candidate": candidate_name,
                "folds": fold_count,
                "heldout_rows": heldout_total,
                "accuracy": correct / max(1, heldout_total),
                "correct": correct,
                "prob_true_class_mean": float(np.mean(probs)) if probs else 0.0,
                "prob_true_class_min": float(np.min(probs)) if probs else 0.0,
            }
        )
    return rows


def replay_count_only(
    *,
    model: Any,
    feature_names: list[str],
    y: np.ndarray,
    sample_rate: int,
    threshold: float,
    expected_count: int | None,
) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    triggers = nr_features.simulate_gate(
        y,
        sample_rate,
        onset_ratio=1.5,
        retrigger_ms=120,
        abs_min_rms=0.0015,
        mode="bandpass",
        spectral_gate=False,
    )
    counter = BinaryLiveCounter(threshold)
    counted = 0
    rows: list[dict[str, Any]] = []
    reject_reasons: Counter[str] = Counter()
    for index, trigger in enumerate(triggers, start=1):
        onset_sample = int(trigger["onset_sample"])
        onset_ms = float(trigger["onset_ms"])
        clip = nr_features.extract_live_clip(y, onset_sample)
        features = nr_features.extract_all_features(clip, sample_rate)
        prediction = predict_candidate(model, features, feature_names)
        prob = float(prediction["probabilities"]["racket_bounce"])
        is_counted, reason = counter.process(prob, onset_ms, float(trigger["frame_rms"]))
        counted += 1 if is_counted else 0
        reject_reasons[reason or "counted"] += 1
        rows.append(
            {
                "trigger_index": index,
                "onset_ms": onset_ms,
                "counted": is_counted,
                "reject_reason": reason,
                "prob_racket_bounce": prob,
                "prob_noise": 1.0 - prob,
                "frame_rms": trigger.get("frame_rms", ""),
                "background_rms": trigger.get("bg_rms", ""),
            }
        )

    expected_delta = "" if expected_count is None else counted - expected_count
    expected_abs_error = "" if expected_count is None else abs(counted - expected_count)
    summary = {
        "threshold": threshold,
        "triggers": len(triggers),
        "candidate_counted": counted,
        "expected_count": "" if expected_count is None else expected_count,
        "candidate_minus_expected": expected_delta,
        "candidate_abs_error": expected_abs_error,
        "reject_reason_counts": json.dumps(dict(sorted(reject_reasons.items())), sort_keys=True),
    }
    return summary, rows


def run_heldout_replay(
    *,
    args: argparse.Namespace,
    model: Any,
    feature_names: list[str],
    threshold: float,
    out_dir: Path,
) -> dict[str, Any]:
    heldout_path = Path(str(args.heldout_wav)) if args.heldout_wav else None
    if heldout_path is None or not heldout_path.exists():
        return {"status": "skipped", "reason": "no --heldout-wav provided"}

    y, sample_rate = read_wav(heldout_path)
    if sample_rate != nr_config.TARGET_SR:
        raise ValueError(f"Expected {nr_config.TARGET_SR} Hz held-out WAV, got {sample_rate}")

    expected_count = args.heldout_expected_count
    count_summary, count_events = replay_count_only(
        model=model,
        feature_names=feature_names,
        y=y,
        sample_rate=sample_rate,
        threshold=threshold,
        expected_count=expected_count,
    )
    count_summary = {
        "status": "evaluated",
        "mode": "count_only",
        "heldout_wav": str(heldout_path),
        **count_summary,
    }
    write_csv(out_dir / "t0057_heldout_count_replay_summary.csv", [count_summary])
    write_csv(out_dir / "t0057_heldout_count_replay_events.csv", count_events)

    labels_path = Path(str(args.heldout_labels_csv)) if args.heldout_labels_csv else None
    exact_summary: dict[str, Any] | None = None
    if labels_path is not None and labels_path.exists():
        truth_ms = truth_times_ms_from_labels_csv(labels_path)
        if truth_ms:
            exact, exact_events = replay_c2(model, feature_names, y, sample_rate, truth_ms, threshold)
            exact_summary = {
                "status": "evaluated",
                "mode": "exact_labels",
                "heldout_wav": str(heldout_path),
                "heldout_labels_csv": str(labels_path),
                **exact,
            }
            write_csv(out_dir / "t0057_heldout_exact_replay_summary.csv", [exact_summary])
            write_csv(out_dir / "t0057_heldout_exact_replay_events.csv", exact_events)

    return {
        **count_summary,
        "exact_label_replay": exact_summary or {
            "status": "skipped",
            "reason": "no exact held-out labels CSV with racket timestamps provided",
        },
    }


def tt_clip_features(record: Any) -> tuple[dict[str, Any], dict[str, Any]]:
    y, sample_rate = load_audio(str(record.path))
    if sample_rate != nr_config.TARGET_SR:
        raise ValueError(f"Expected {nr_config.TARGET_SR} Hz TT clip, got {sample_rate}")
    if record.anchor_ms is not None:
        anchor_sample = int(float(record.anchor_ms) * sample_rate / 1000.0)
    else:
        anchor_sample = int(np.argmax(np.abs(y))) if len(y) else nr_config.CLIP_PRE_SAMPLES
    clip = nr_features.extract_live_clip(y, anchor_sample)
    features = nr_features.extract_all_features(clip, sample_rate)
    meta = {
        "duration_ms": round(len(y) / sample_rate * 1000.0, 3),
        "anchor_ms": round(anchor_sample / sample_rate * 1000.0, 3),
    }
    return features, meta


def run_tt_sounds_safety_sample(
    *,
    args: argparse.Namespace,
    model: Any,
    feature_names: list[str],
    threshold: float,
    out_dir: Path,
) -> dict[str, Any]:
    max_per_label = int(args.tt_sounds_sample_per_label or 0)
    if max_per_label <= 0:
        return {"status": "skipped", "reason": "--tt-sounds-sample-per-label is 0"}

    tt_root = Path(args.tt_root)
    records, manifest = discover_tt_records(tt_root, max_per_label=max_per_label)
    if not records:
        return {
            "status": "skipped",
            "reason": "no TT Sounds records found",
            "manifest": manifest,
        }

    rows: list[dict[str, Any]] = []
    for record in records:
        try:
            features, meta = tt_clip_features(record)
            prediction = predict_candidate(model, features, feature_names)
            prob = float(prediction["probabilities"]["racket_bounce"])
            predicted_positive = prob >= threshold
            rows.append(
                {
                    "source": record.source,
                    "sample_id": record.sample_id,
                    "path": str(record.path),
                    "true_label": record.true_label,
                    "predicted_positive": predicted_positive,
                    "prob_racket_bounce": prob,
                    "threshold": threshold,
                    "correct_binary": predicted_positive == (record.true_label == "racket_bounce"),
                    **meta,
                }
            )
        except Exception as exc:
            rows.append(
                {
                    "source": getattr(record, "source", "tt_sounds"),
                    "sample_id": getattr(record, "sample_id", ""),
                    "path": str(getattr(record, "path", "")),
                    "true_label": getattr(record, "true_label", ""),
                    "predicted_positive": "",
                    "prob_racket_bounce": "",
                    "threshold": threshold,
                    "correct_binary": False,
                    "error": str(exc),
                }
            )

    by_label: list[dict[str, Any]] = []
    label_order = sorted({str(row.get("true_label") or "") for row in rows if row.get("true_label")})
    for label in label_order:
        label_rows = [row for row in rows if row.get("true_label") == label and row.get("prob_racket_bounce") != ""]
        support = len(label_rows)
        predicted_positive = sum(1 for row in label_rows if boolish(row.get("predicted_positive")))
        probs = [float(row["prob_racket_bounce"]) for row in label_rows]
        by_label.append(
            {
                "true_label": label,
                "support": support,
                "predicted_positive": predicted_positive,
                "positive_rate": predicted_positive / max(1, support),
                "prob_racket_mean": float(np.mean(probs)) if probs else "",
                "prob_racket_max": float(np.max(probs)) if probs else "",
            }
        )

    racket_rows = [row for row in rows if row.get("true_label") == "racket_bounce" and row.get("prob_racket_bounce") != ""]
    non_racket_rows = [
        row
        for row in rows
        if row.get("true_label") not in {"", "racket_bounce"} and row.get("prob_racket_bounce") != ""
    ]
    racket_hits = sum(1 for row in racket_rows if boolish(row.get("predicted_positive")))
    non_racket_fp = sum(1 for row in non_racket_rows if boolish(row.get("predicted_positive")))
    summary = {
        "status": "evaluated",
        "source": "TT Sounds",
        "source_url": TT_SOUNDS_PAGE,
        "license": TT_SOUNDS_LICENSE,
        "root": str(tt_root),
        "sample_per_label_cap": max_per_label,
        "records": len(rows),
        "racket_records": len(racket_rows),
        "racket_predicted_positive": racket_hits,
        "racket_positive_rate": racket_hits / max(1, len(racket_rows)),
        "non_racket_records": len(non_racket_rows),
        "non_racket_predicted_positive": non_racket_fp,
        "non_racket_positive_rate": non_racket_fp / max(1, len(non_racket_rows)),
        "manifest": manifest,
    }
    write_csv(out_dir / "t0057_tt_sounds_safety_sample.csv", rows)
    write_csv(out_dir / "t0057_tt_sounds_safety_by_label.csv", by_label)
    (out_dir / "t0057_tt_sounds_safety_summary.json").write_text(
        json.dumps(summary, indent=2, sort_keys=True),
        encoding="utf-8",
    )
    return {**summary, "by_label": by_label}


def run_local_reviewed_safety_sample(
    *,
    args: argparse.Namespace,
    model: Any,
    feature_names: list[str],
    threshold: float,
    out_dir: Path,
) -> dict[str, Any]:
    max_per_label = int(args.local_reviewed_sample_per_label or 0)
    if max_per_label <= 0:
        return {"status": "skipped", "reason": "--local-reviewed-sample-per-label is 0"}

    local_root = Path(args.local_reviewed_root)
    records, manifest = discover_local_clip_records(local_root, max_per_label=max_per_label)
    if not records:
        return {
            "status": "skipped",
            "reason": "no local reviewed records found",
            "manifest": manifest,
        }

    rows: list[dict[str, Any]] = []
    for record in records:
        try:
            features, meta = tt_clip_features(record)
            prediction = predict_candidate(model, features, feature_names)
            prob = float(prediction["probabilities"]["racket_bounce"])
            predicted_positive = prob >= threshold
            metadata = record.metadata or {}
            rows.append(
                {
                    "source": record.source,
                    "sample_id": record.sample_id,
                    "path": str(record.path),
                    "true_label": record.true_label,
                    "predicted_positive": predicted_positive,
                    "prob_racket_bounce": prob,
                    "threshold": threshold,
                    "correct_binary": predicted_positive == (record.true_label == "racket_bounce"),
                    "session_id": metadata.get("session_id", ""),
                    "background_condition": metadata.get("background_condition", ""),
                    "wav_filename": metadata.get("wav_filename", ""),
                    **meta,
                }
            )
        except Exception as exc:
            rows.append(
                {
                    "source": getattr(record, "source", "local_reviewed_marker"),
                    "sample_id": getattr(record, "sample_id", ""),
                    "path": str(getattr(record, "path", "")),
                    "true_label": getattr(record, "true_label", ""),
                    "predicted_positive": "",
                    "prob_racket_bounce": "",
                    "threshold": threshold,
                    "correct_binary": False,
                    "error": str(exc),
                }
            )

    by_label: list[dict[str, Any]] = []
    label_order = sorted({str(row.get("true_label") or "") for row in rows if row.get("true_label")})
    for label in label_order:
        label_rows = [row for row in rows if row.get("true_label") == label and row.get("prob_racket_bounce") != ""]
        support = len(label_rows)
        predicted_positive = sum(1 for row in label_rows if boolish(row.get("predicted_positive")))
        probs = [float(row["prob_racket_bounce"]) for row in label_rows]
        by_label.append(
            {
                "true_label": label,
                "support": support,
                "predicted_positive": predicted_positive,
                "positive_rate": predicted_positive / max(1, support),
                "prob_racket_mean": float(np.mean(probs)) if probs else "",
                "prob_racket_max": float(np.max(probs)) if probs else "",
            }
        )

    label_counts = Counter(str(row.get("true_label") or "") for row in rows if row.get("true_label"))
    safety_negative_labels = {"table_bounce", "floor_bounce", "noise"}
    safety_negative_count = sum(label_counts.get(label, 0) for label in safety_negative_labels)
    summary = {
        "status": "evaluated",
        "source": "local_reviewed_sessions",
        "root": str(local_root),
        "sample_per_label_cap": max_per_label,
        "records": len(rows),
        "label_counts": dict(sorted(label_counts.items())),
        "safety_negative_records": safety_negative_count,
        "has_table_floor_noise_safety": safety_negative_count > 0,
        "manifest": manifest,
    }
    write_csv(out_dir / "t0059_local_reviewed_safety_sample.csv", rows)
    write_csv(out_dir / "t0059_local_reviewed_safety_by_label.csv", by_label)
    (out_dir / "t0059_local_reviewed_safety_summary.json").write_text(
        json.dumps(summary, indent=2, sort_keys=True),
        encoding="utf-8",
    )
    return {**summary, "by_label": by_label}


def candidate_score(row: dict[str, Any]) -> float:
    return (
        float(row["c2_exact_recall"]) * 120.0
        + float(row["c2_exact_precision"]) * 60.0
        - float(row["c2_exact_fp"]) * 12.0
        - float(row["t0050_negative_false_counts"]) * 8.0
        - float(row["t0050_real_abs_error"]) * 1.25
    )


def infer_ticket(out_dir: Path) -> str:
    name = out_dir.name.lower()
    if name.startswith("t0060"):
        return "T0060-fresh-heldout-c2-pull-and-loop-rerun"
    if name.startswith("t0059"):
        return "T0059-fable-local-reviewed-safety-hook"
    if name.startswith("t0058"):
        return "T0058-fable-loop-heldout-safety-hooks"
    return "T0057-fable-auto-improvement-loop"


def choose_decision_and_next_input(
    best_candidate: dict[str, Any],
    heldout_replay: dict[str, Any],
    tt_sounds_safety: dict[str, Any],
    local_reviewed_safety: dict[str, Any],
) -> tuple[str, dict[str, Any], bool]:
    uses_c2_training = "plus_c2" in str(best_candidate["training_set"])
    export_ready = False
    heldout_status = heldout_replay.get("status")
    heldout_expected = int(heldout_replay.get("expected_count") or 0)
    heldout_counted = int(heldout_replay.get("candidate_counted") or 0)
    heldout_abs_error = int(heldout_replay.get("candidate_abs_error") or 0)
    heldout_failed = heldout_status == "evaluated" and heldout_expected > 0 and heldout_abs_error > max(3, heldout_expected // 4)
    tt_unsafe = (
        tt_sounds_safety.get("status") == "evaluated"
        and float(tt_sounds_safety.get("non_racket_positive_rate") or 0.0) > 0.10
    )
    lacks_local_safety = (
        local_reviewed_safety.get("status") == "evaluated"
        and not bool(local_reviewed_safety.get("has_table_floor_noise_safety"))
    )

    if heldout_failed:
        decision = (
            "The fresh held-out replay failed: the best local candidate counted "
            f"{heldout_counted}/{heldout_expected} with abs error {heldout_abs_error}. "
            "This rules out exporting the current diagnostic candidate. The next useful step is exact timestamp "
            "review for this held-out WAV, then a broader 4-class retrain/replay with hard speech/background "
            "positives and table/floor/noise safety negatives."
        )
        next_user_input = {
            "needed": True,
            "reason": (
                "The count-only held-out result is too poor to promote, and exact timestamps are needed to tell "
                "which of the 60 trigger candidates are real racket contacts versus speech/background/duplicate events."
            ),
            "request": "timestamp_labels_for_current_heldout_wav",
        }
        return decision, next_user_input, export_ready

    if heldout_status != "evaluated":
        decision = (
            "The best local candidate still lacks independent held-out C2-like proof. Do not export. The loop needs "
            "one fresh held-out continuous-WAV run or the missing full historical 4-class train/val CSVs before a "
            "real promotion decision."
        )
        next_user_input = {
            "needed": True,
            "reason": (
                "Local data cannot prove generalization: the best candidate is binary, lacks table/floor safety, "
                "and either uses the C2 slice in training or lacks an independent C2-like holdout."
            ),
            "request": "fresh_heldout_c2_like_continuous_wav",
        }
        return decision, next_user_input, export_ready

    if uses_c2_training:
        decision = (
            "The held-out count did not fail badly, but the best local candidate still depends on earlier C2 rows "
            "in training, so exact C2 replay remains diagnostic rather than export proof. Do not export until a "
            "broader 4-class replay passes."
        )
    else:
        decision = (
            "The held-out count did not fail badly, but this local loop is still binary and not a ship gate by itself. "
            "Do not export until a broader 4-class replay passes."
        )
    if tt_unsafe:
        decision += " The TT Sounds safety sample also still shows too many non-racket positives."
    if lacks_local_safety:
        decision += " The local reviewed sample still has no table/floor/noise safety negatives."
    next_user_input = {
        "needed": True,
        "reason": "Held-out count-only replay is not enough without timestamp labels and 4-class safety coverage.",
        "request": "broader_4class_safety_replay_or_exact_heldout_labels",
    }
    return decision, next_user_input, export_ready


def render_report(summary: dict[str, Any], out_dir: Path) -> None:
    best = summary["best_candidate"]
    user_input = summary["next_user_input"]
    heldout = summary.get("heldout_replay") or {"status": "skipped", "reason": "not run"}
    tt_sounds = summary.get("tt_sounds_safety_sample") or {"status": "skipped", "reason": "not run"}
    local_reviewed = summary.get("local_reviewed_safety_sample") or {"status": "skipped", "reason": "not run"}
    lines = [
        f"# {summary['ticket']} Fable Auto Improvement Loop",
        "",
        "## Scope",
        "",
        "- Evaluation-only local loop, extending the T0057 candidate sweep.",
        "- No app model JSON export, APK build/install, runtime threshold change, raw label mutation, cloud/API, or camera change.",
        "- Candidate models are still binary `racket_bounce` vs `noise`; this loop is not a 4-class ship gate.",
        "",
        "## Inputs",
        "",
        f"- T0049 approved rows: `{summary['input_counts']['t0049_rows']}`",
        f"- T0055 C2 rows: `{summary['input_counts']['t0055_rows']}`",
        f"- Training label counts: `{json.dumps(summary['input_counts']['combined_label_counts'], sort_keys=True)}`",
        "- T0050 targeted debug blocks are used as weak block-level replay checks, not row-level training truth.",
        "",
        "## Best Local Candidate",
        "",
        f"- Candidate: `{best['candidate']}`",
        f"- Training set: `{best['training_set']}`",
        f"- Threshold: `{best['threshold']}`",
        f"- Exact C2 replay: `TP {best['c2_exact_tp']} / FP {best['c2_exact_fp']} / missed {best['c2_exact_missed']}`",
        f"- Exact C2 precision/recall: `{best['c2_exact_precision']:.3f}` / `{best['c2_exact_recall']:.3f}`",
        f"- T0050 real-block abs count error: `{best['t0050_real_abs_error']}`",
        f"- T0050 negative false counts: `{best['t0050_negative_false_counts']}` (saved app baseline `{best['t0050_saved_negative_false_counts']}`)",
        "",
        "## Optional Held-Out Replay Hook",
        "",
        f"- Status: `{heldout.get('status', 'skipped')}`",
    ]
    if heldout.get("status") == "evaluated":
        lines.extend(
            [
                f"- WAV: `{heldout.get('heldout_wav', '')}`",
                f"- Count-only candidate count: `{heldout.get('candidate_counted', '')}`",
                f"- Expected count: `{heldout.get('expected_count', '')}`",
                f"- Count abs error: `{heldout.get('candidate_abs_error', '')}`",
            ]
        )
        exact = heldout.get("exact_label_replay") or {}
        if exact.get("status") == "evaluated":
            lines.append(
                f"- Exact-label replay: `TP {exact.get('tp')} / FP {exact.get('fp')} / missed {exact.get('missed')}`"
            )
    else:
        lines.append(f"- Reason: {heldout.get('reason', '')}")
    lines.extend(
        [
            "",
            "## Optional TT Sounds Safety Sample",
            "",
            f"- Status: `{tt_sounds.get('status', 'skipped')}`",
        ]
    )
    if tt_sounds.get("status") == "evaluated":
        lines.extend(
            [
                f"- Records: `{tt_sounds.get('records', 0)}` capped at `{tt_sounds.get('sample_per_label_cap', '')}` per label",
                f"- Racket positive rate: `{float(tt_sounds.get('racket_positive_rate', 0.0)):.3f}`",
                f"- Non-racket positive rate: `{float(tt_sounds.get('non_racket_positive_rate', 0.0)):.3f}`",
                f"- License note: `{tt_sounds.get('license', '')}` diagnostic use only",
            ]
        )
    else:
        lines.append(f"- Reason: {tt_sounds.get('reason', '')}")
    lines.extend(
        [
            "",
            "## Optional Local Reviewed Safety/Coverage Sample",
            "",
            f"- Status: `{local_reviewed.get('status', 'skipped')}`",
        ]
    )
    if local_reviewed.get("status") == "evaluated":
        lines.extend(
            [
                f"- Records: `{local_reviewed.get('records', 0)}` capped at `{local_reviewed.get('sample_per_label_cap', '')}` per label",
                f"- Label counts: `{json.dumps(local_reviewed.get('label_counts', {}), sort_keys=True)}`",
                f"- Table/floor/noise safety records: `{local_reviewed.get('safety_negative_records', 0)}`",
            ]
        )
        if not local_reviewed.get("has_table_floor_noise_safety"):
            lines.append("- Local reviewed data does not currently prove table/floor/noise safety for this candidate.")
    else:
        lines.append(f"- Reason: {local_reviewed.get('reason', '')}")
    lines.extend(
        [
            "",
            "## Decision",
            "",
            summary["decision"],
            "",
            "## User Input Needed",
            "",
            f"- Needed now: `{user_input['needed']}`",
            f"- Reason: {user_input['reason']}",
            "",
        ]
    )
    if user_input["needed"]:
        if user_input.get("request") == "timestamp_labels_for_current_heldout_wav":
            next_steps = [
                "### Exact Next Review",
                "",
                "1. Open the local trigger-review UI for the pulled held-out WAV.",
                "2. Label each real racket contact as `racket` and clear non-contact triggers as `noise`/`duplicate`/`unclear`.",
                "3. Add any missed real contacts at the playhead.",
                "4. Save labels.",
                "",
                "Codex will then ingest those exact timestamps and rerun the loop as a reviewed held-out slice.",
                "",
            ]
        elif user_input.get("request") == "fresh_heldout_c2_like_continuous_wav":
            next_steps = [
                "### Exact Next Test",
                "",
                "1. Open `Fable-algoritm` on the Motorola.",
                "2. Run one fresh C2-like held-out test: bounce while speaking/counting out loud with realistic kid/background sound.",
                "3. Aim for `30` actual racket contacts; stop after about `20-40 s`.",
                "4. Press `STOPPA` so the matching `.json` and continuous `.wav` are saved.",
                "5. Tell Codex only: expected racket contacts and app count.",
                "",
                "Codex will then pull the new JSON/WAV, add it as held-out validation, and rerun this loop.",
                "",
            ]
        else:
            next_steps = [
                "### Exact Next Step",
                "",
                f"- Request: `{user_input.get('request', 'unknown')}`",
                "",
            ]
        lines.extend(
            next_steps
        )
    lines.extend(
        [
            "## Outputs",
            "",
            "- `t0057_candidate_summary.csv`",
            "- `t0057_c2_exact_replay_summary.csv`",
            "- `t0057_t0050_block_replay_summary.csv`",
            "- `t0057_t0050_best_block_events.csv`",
            "- `t0057_t0055_fold_check.csv`",
            "- `t0057_summary.json`",
            "- optional `t0057_heldout_*` outputs when `--heldout-wav` is provided",
            "- optional `t0057_tt_sounds_safety_*` outputs when `--tt-sounds-sample-per-label` is greater than `0`",
            "- optional `t0059_local_reviewed_safety_*` outputs when `--local-reviewed-sample-per-label` is greater than `0`",
            "- local diagnostic `.joblib` models",
        ]
    )
    (out_dir / "t0057_report.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    if user_input["needed"]:
        next_start = next(
            (
                lines.index(heading)
                for heading in ("### Exact Next Review", "### Exact Next Test", "### Exact Next Step")
                if heading in lines
            ),
            0,
        )
        (out_dir / "T0057_NEXT_USER_INPUT.md").write_text(
            "\n".join(lines[next_start : lines.index("## Outputs")]) + "\n",
            encoding="utf-8",
        )


def run(args: argparse.Namespace) -> dict[str, Any]:
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    models_dir = out_dir / "candidate_models"
    models_dir.mkdir(parents=True, exist_ok=True)
    for old_model in models_dir.glob("*.joblib"):
        old_model.unlink()

    feature_names = nr_features.all_feature_names()
    c2_audio, sample_rate = read_wav(Path(args.c2_wav))
    if sample_rate != nr_config.TARGET_SR:
        raise ValueError(f"Expected {nr_config.TARGET_SR} Hz WAV, got {sample_rate}")

    t0049_rows = load_t0049_rows(Path(args.t0049_rows), feature_names)
    t0055_rows = build_t0055_rows(Path(args.t0055_candidates), c2_audio, sample_rate, feature_names)
    combined_rows = t0049_rows + t0055_rows
    truth_ms = truth_times_ms(Path(args.t0055_timeline))
    training_sets = row_training_sets(t0049_rows, t0055_rows)
    t0050_cache_rows = build_t0050_feature_cache(Path(args.t0050_debug_dir))

    fold_rows = t0055_fold_metrics(
        base_rows=t0049_rows,
        c2_rows=t0055_rows,
        feature_names=feature_names,
        seed=args.seed,
    )
    write_csv(out_dir / "t0057_t0055_fold_check.csv", fold_rows)

    row_metrics: list[dict[str, Any]] = []
    c2_replay_rows: list[dict[str, Any]] = []
    block_replay_rows: list[dict[str, Any]] = []
    candidate_rows: list[dict[str, Any]] = []
    best_block_events: list[dict[str, Any]] = []
    best_candidate: dict[str, Any] | None = None
    best_score = -1e18

    for training_set_name, train_rows in training_sets.items():
        use_weights = training_set_name.endswith("_weighted")
        for candidate_name, estimator in loop_candidate_specs(args.seed):
            model = fit_candidate(estimator, train_rows, feature_names, use_weights=use_weights)
            candidate_id = f"{candidate_name}__{training_set_name}"
            joblib.dump(model, models_dir / f"{candidate_id}.joblib")

            row_metrics.append(
                {
                    "candidate": candidate_name,
                    "training_set": training_set_name,
                    **evaluate_rows(model, t0049_rows, feature_names, "t0049_rows"),
                }
            )
            row_metrics.append(
                {
                    "candidate": candidate_name,
                    "training_set": training_set_name,
                    **evaluate_rows(model, t0055_rows, feature_names, "t0055_c2_rows"),
                }
            )

            for threshold in LOOP_THRESHOLDS:
                c2_summary, _c2_events = replay_c2(
                    model,
                    feature_names,
                    c2_audio,
                    sample_rate,
                    truth_ms,
                    threshold,
                )
                block_rows, block_events, block_aggregate = replay_t0050_feature_cache(
                    model=model,
                    feature_names=feature_names,
                    threshold=threshold,
                    cache_rows=t0050_cache_rows,
                )
                c2_row = {
                    "candidate": candidate_name,
                    "training_set": training_set_name,
                    **c2_summary,
                }
                c2_replay_rows.append(c2_row)
                for row in block_rows:
                    block_replay_rows.append(
                        {
                            "candidate": candidate_name,
                            "training_set": training_set_name,
                            **row,
                        }
                    )
                summary_row = {
                    "candidate": candidate_name,
                    "training_set": training_set_name,
                    "model_id": candidate_id,
                    "model_path": str(models_dir / f"{candidate_id}.joblib"),
                    "threshold": threshold,
                    "score": 0.0,
                    "c2_exact_tp": c2_summary["tp"],
                    "c2_exact_fp": c2_summary["fp"],
                    "c2_exact_missed": c2_summary["missed"],
                    "c2_exact_precision": c2_summary["precision"],
                    "c2_exact_recall": c2_summary["recall"],
                    **block_aggregate,
                }
                summary_row["score"] = candidate_score(summary_row)
                candidate_rows.append(summary_row)
                if float(summary_row["score"]) > best_score:
                    best_score = float(summary_row["score"])
                    best_candidate = dict(summary_row)
                    best_block_events = [
                        {
                            "candidate": candidate_name,
                            "training_set": training_set_name,
                            "threshold": threshold,
                            **event,
                        }
                        for event in block_events
                    ]

    if best_candidate is None:
        raise RuntimeError("No candidate results were generated")

    write_csv(out_dir / "t0057_candidate_row_metrics.csv", row_metrics)
    write_csv(out_dir / "t0057_c2_exact_replay_summary.csv", c2_replay_rows)
    write_csv(out_dir / "t0057_t0050_block_replay_summary.csv", block_replay_rows)
    write_csv(out_dir / "t0057_candidate_summary.csv", candidate_rows)
    write_csv(out_dir / "t0057_t0050_best_block_events.csv", best_block_events)

    best_model = joblib.load(str(best_candidate["model_path"]))
    best_threshold = float(best_candidate["threshold"])
    heldout_replay = run_heldout_replay(
        args=args,
        model=best_model,
        feature_names=feature_names,
        threshold=best_threshold,
        out_dir=out_dir,
    )
    tt_sounds_safety = run_tt_sounds_safety_sample(
        args=args,
        model=best_model,
        feature_names=feature_names,
        threshold=best_threshold,
        out_dir=out_dir,
    )
    local_reviewed_safety = run_local_reviewed_safety_sample(
        args=args,
        model=best_model,
        feature_names=feature_names,
        threshold=best_threshold,
        out_dir=out_dir,
    )

    decision, next_user_input, export_ready = choose_decision_and_next_input(
        best_candidate=best_candidate,
        heldout_replay=heldout_replay,
        tt_sounds_safety=tt_sounds_safety,
        local_reviewed_safety=local_reviewed_safety,
    )

    summary = {
        "ticket": infer_ticket(out_dir),
        "export_ready": export_ready,
        "input_counts": {
            "t0049_rows": len(t0049_rows),
            "t0055_rows": len(t0055_rows),
            "combined_rows": len(combined_rows),
            "combined_label_counts": dict(sorted(Counter(row["label"] for row in combined_rows).items())),
        },
        "best_candidate": best_candidate,
        "t0055_fold_check": fold_rows,
        "heldout_replay": heldout_replay,
        "tt_sounds_safety_sample": tt_sounds_safety,
        "local_reviewed_safety_sample": local_reviewed_safety,
        "decision": decision,
        "next_user_input": next_user_input,
        "outputs": {
            "output_dir": str(out_dir),
            "report": str(out_dir / "t0057_report.md"),
            "next_user_input": str(out_dir / "T0057_NEXT_USER_INPUT.md"),
        },
    }
    (out_dir / "t0057_summary.json").write_text(json.dumps(summary, indent=2, sort_keys=True), encoding="utf-8")
    render_report(summary, out_dir)
    return summary


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--t0049-rows", default=str(DEFAULT_T0049_ROWS))
    parser.add_argument("--t0055-timeline", default=str(DEFAULT_T0055_TIMELINE))
    parser.add_argument("--t0055-candidates", default=str(DEFAULT_T0055_CANDIDATES))
    parser.add_argument("--c2-wav", default=str(DEFAULT_C2_WAV))
    parser.add_argument("--t0050-debug-dir", default=str(DEFAULT_T0050_DEBUG_DIR))
    parser.add_argument("--heldout-wav", default="")
    parser.add_argument("--heldout-expected-count", type=int, default=None)
    parser.add_argument("--heldout-labels-csv", default="")
    parser.add_argument("--tt-root", default=str(DEFAULT_TT_ROOT))
    parser.add_argument("--local-reviewed-root", default=str(DEFAULT_LOCAL_REVIEWED_ROOT))
    parser.add_argument(
        "--local-reviewed-sample-per-label",
        type=int,
        default=0,
        help="0 skips local reviewed records. Use a cap such as 100 for recursive local coverage diagnostics.",
    )
    parser.add_argument(
        "--tt-sounds-sample-per-label",
        type=int,
        default=0,
        help="0 skips TT Sounds. Use a small cap, e.g. 5-25, for quick safety diagnostics.",
    )
    parser.add_argument("--out-dir", default=str(DEFAULT_OUT_DIR))
    parser.add_argument("--seed", type=int, default=20260629)
    return parser.parse_args()


if __name__ == "__main__":
    result = run(parse_args())
    print(
        json.dumps(
            {
                "ticket": result["ticket"],
                "export_ready": result["export_ready"],
                "input_counts": result["input_counts"],
                "best_candidate": result["best_candidate"],
                "decision": result["decision"],
                "next_user_input": result["next_user_input"],
                "heldout_replay": result["heldout_replay"],
                "tt_sounds_safety_sample": result["tt_sounds_safety_sample"],
                "local_reviewed_safety_sample": result["local_reviewed_safety_sample"],
                "report": result["outputs"]["report"],
            },
            indent=2,
            sort_keys=True,
        )
    )
