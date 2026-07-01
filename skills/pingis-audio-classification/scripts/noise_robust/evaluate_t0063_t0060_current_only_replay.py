#!/usr/bin/env python3
"""T0063 current-only exact replay for the T0060 held-out Fable run.

Inputs are deliberately limited to:
- the T0060 continuous WAV,
- Love's T0060 exact label CSV,
- the current bundled Fable model JSON,
- optionally the T0060 debug JSON for saved-app metadata.

No training, TT Sounds, local reviewed sessions, historical C2 data, app export,
APK build, or runtime change happens here.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import sys
import wave
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np

SCRIPT_DIR = Path(__file__).resolve().parent
SCRIPTS_DIR = SCRIPT_DIR.parent
for _path in (str(SCRIPT_DIR), str(SCRIPTS_DIR)):
    if _path not in sys.path:
        sys.path.insert(0, _path)

import nr_config  # noqa: E402
import nr_features  # noqa: E402
from evaluate_fable_audio_reliability_t0044 import (  # noqa: E402
    FableAppModel,
    FableOfflineCounter,
    FableRuntimeConfig,
)

ROOT_DIR = Path(__file__).resolve().parents[4]
SESSION_ID = "fable_live_session_2026-06-29T13-29-50-713Z"
DEFAULT_WAV = (
    ROOT_DIR
    / "data/audio/raw/t0060_fresh_heldout_c2/fable_live_debug"
    / f"{SESSION_ID}.wav"
)
DEFAULT_JSON = (
    ROOT_DIR
    / "data/audio/raw/t0060_fresh_heldout_c2/fable_live_debug"
    / f"{SESSION_ID}.json"
)
DEFAULT_LABELS_CSV = (
    ROOT_DIR
    / "data/audio/models/evaluations/t0063_t0060_heldout_label_ingest"
    / "t0063_exact_heldout_labels.csv"
)
DEFAULT_MODEL_JSON = ROOT_DIR / "apps/collector/src/models/fable_audio_model.json"
DEFAULT_OUT_DIR = ROOT_DIR / "data/audio/models/evaluations/t0063_t0060_current_only_replay"
MATCH_TOLERANCE_MS = 140.0

# Mirrors apps/collector/src/fableEngine.ts for the installed current Fable path.
CURRENT_APP_CONFIG = FableRuntimeConfig(
    quiet_confidence=0.65,
    loud_confidence=0.9,
    loud_bg_db=-42.0,
    merge_ms=120,
    same_bounce_ms=250,
    group_ms=80,
    echo_ms=300,
    echo_ratio=0.6,
)


def finite_float(value: Any, default: float = float("nan")) -> float:
    try:
        out = float(value)
    except Exception:
        return default
    return out if math.isfinite(out) else default


def boolish(value: Any) -> bool:
    return value is True or str(value).lower() == "true"


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open("r", newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fields: list[str] = []
    seen: set[str] = set()
    for row in rows:
        for key in row:
            if key not in seen:
                fields.append(key)
                seen.add(key)
    if not fields:
        fields = ["empty"]
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        for row in rows:
            writer.writerow({key: row.get(key, "") for key in fields})


def read_wav(path: Path) -> tuple[np.ndarray, int]:
    with wave.open(str(path), "rb") as wav:
        channels = wav.getnchannels()
        sample_width = wav.getsampwidth()
        sample_rate = wav.getframerate()
        frames = wav.readframes(wav.getnframes())
    if channels != 1 or sample_width != 2:
        raise ValueError(f"Expected mono 16-bit PCM WAV, got channels={channels}, width={sample_width}")
    y = np.frombuffer(frames, dtype="<i2").astype(np.float32) / 32768.0
    return y, sample_rate


def truth_times_ms(labels_csv: Path) -> list[float]:
    times = []
    for row in read_csv(labels_csv):
        label = str(row.get("label") or row.get("review_label") or "").strip().lower()
        if label not in {"racket", "racket_bounce", "racket_contact"}:
            continue
        time_s = finite_float(row.get("reviewed_time_s"), finite_float(row.get("time_s")))
        if math.isfinite(time_s):
            times.append(time_s * 1000.0)
    return sorted(times)


def match_counts(predicted_ms: list[float], truth_ms: list[float], tolerance_ms: float) -> dict[str, Any]:
    truth_sorted = sorted(truth_ms)
    used_truth: set[int] = set()
    matched: list[tuple[float, float, float]] = []
    false_counts: list[float] = []
    for pred in sorted(predicted_ms):
        best_idx = None
        best_delta = float("inf")
        for idx, truth in enumerate(truth_sorted):
            if idx in used_truth:
                continue
            delta = pred - truth
            if abs(delta) <= tolerance_ms and abs(delta) < abs(best_delta):
                best_idx = idx
                best_delta = delta
        if best_idx is None:
            false_counts.append(pred)
        else:
            used_truth.add(best_idx)
            matched.append((pred, truth_sorted[best_idx], best_delta))
    missed = [truth for idx, truth in enumerate(truth_sorted) if idx not in used_truth]
    return {
        "tp": len(matched),
        "fp": len(false_counts),
        "missed": len(missed),
        "predicted": len(predicted_ms),
        "truth": len(truth_ms),
        "precision": len(matched) / len(predicted_ms) if predicted_ms else 0.0,
        "recall": len(matched) / len(truth_ms) if truth_ms else 0.0,
        "matched": matched,
        "false_counts": false_counts,
        "missed_truth": missed,
    }


def threshold_for_features(features: dict[str, Any]) -> tuple[float, str]:
    bg_rms_db = float(features.get("nr_bg_rms_db", -100.0) or -100.0)
    loud = bg_rms_db >= CURRENT_APP_CONFIG.loud_bg_db
    return (
        CURRENT_APP_CONFIG.loud_confidence if loud else CURRENT_APP_CONFIG.quiet_confidence,
        "loud" if loud else "quiet",
    )


def predict_at_sample(model: FableAppModel, y: np.ndarray, sample_rate: int, sample: int) -> tuple[dict[str, Any], dict[str, Any]]:
    clip = nr_features.extract_live_clip(y, sample)
    features = nr_features.extract_all_features(clip, sample_rate)
    return model.predict_features(features), features


def nearest_delta_ms(value: float, candidates: list[float]) -> float:
    return min((candidate - value for candidate in candidates), key=abs) if candidates else float("nan")


def current_app_trigger_replay(
    *,
    y: np.ndarray,
    sample_rate: int,
    model: FableAppModel,
    truth_ms: list[float],
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
    counter = FableOfflineCounter(model, CURRENT_APP_CONFIG)
    counted_ms: list[float] = []
    rows: list[dict[str, Any]] = []
    reject_reasons: Counter[str] = Counter()
    model_labels: Counter[str] = Counter()

    for index, trigger in enumerate(triggers, start=1):
        onset_sample = int(trigger["onset_sample"])
        onset_ms = float(trigger["onset_ms"])
        clip = nr_features.extract_live_clip(y, onset_sample)
        result = counter.process_clip(clip, onset_ms, float(trigger["frame_rms"]))
        prediction = result.get("prediction") or {}
        probabilities = prediction.get("probabilities") or {}
        counted = bool(result.get("counted"))
        if counted:
            counted_ms.append(onset_ms)
        reject_reason = str(result.get("reject_reason") or "counted")
        model_label = str(prediction.get("label") or "")
        reject_reasons[reject_reason] += 1
        model_labels[model_label or "-"] += 1
        nearest_truth_delta = nearest_delta_ms(onset_ms, truth_ms)
        rows.append(
            {
                "trigger_index": index,
                "onset_ms": round(onset_ms, 3),
                "counted": counted,
                "reject_reason": result.get("reject_reason") or "",
                "model_label": model_label,
                "model_confidence": prediction.get("confidence", ""),
                "prob_floor_bounce": probabilities.get("floor_bounce", ""),
                "prob_noise": probabilities.get("noise", ""),
                "prob_racket_bounce": probabilities.get("racket_bounce", ""),
                "prob_table_bounce": probabilities.get("table_bounce", ""),
                "confidence_threshold": result.get("confidence_threshold", ""),
                "bg_mode": result.get("bg_mode", ""),
                "bg_rms_db": result.get("bg_rms_db", ""),
                "frame_rms": trigger.get("frame_rms", ""),
                "background_rms": trigger.get("bg_rms", ""),
                "nearest_truth_delta_ms": round(nearest_truth_delta, 3) if math.isfinite(nearest_truth_delta) else "",
                "nearest_truth_abs_delta_ms": round(abs(nearest_truth_delta), 3) if math.isfinite(nearest_truth_delta) else "",
            }
        )

    match = match_counts(counted_ms, truth_ms, MATCH_TOLERANCE_MS)
    summary = {
        "triggers": len(triggers),
        "counted": match["predicted"],
        "truth": match["truth"],
        "tp": match["tp"],
        "fp": match["fp"],
        "missed": match["missed"],
        "precision": match["precision"],
        "recall": match["recall"],
        "match_tolerance_ms": MATCH_TOLERANCE_MS,
        "reject_reason_counts": dict(sorted(reject_reasons.items())),
        "model_label_counts": dict(sorted(model_labels.items())),
    }
    return summary, rows


def label_centered_predictions(
    *,
    y: np.ndarray,
    sample_rate: int,
    model: FableAppModel,
    truth_ms: list[float],
    trigger_rows: list[dict[str, Any]],
) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    trigger_times = [float(row["onset_ms"]) for row in trigger_rows]
    rows: list[dict[str, Any]] = []
    labels = Counter()
    clears_threshold = 0
    racket_labels = 0
    for index, truth in enumerate(truth_ms, start=1):
        sample = int(round(truth / 1000.0 * sample_rate))
        prediction, features = predict_at_sample(model, y, sample_rate, sample)
        threshold, bg_mode = threshold_for_features(features)
        probabilities = prediction.get("probabilities") or {}
        is_racket_label = prediction["label"] == "racket_bounce"
        clears = is_racket_label and float(prediction["confidence"]) >= threshold
        labels[str(prediction["label"])] += 1
        racket_labels += 1 if is_racket_label else 0
        clears_threshold += 1 if clears else 0
        nearest_trigger_delta = nearest_delta_ms(truth, trigger_times)
        nearest_trigger = min(trigger_rows, key=lambda row: abs(float(row["onset_ms"]) - truth)) if trigger_rows else {}
        rows.append(
            {
                "label_index": index,
                "truth_ms": round(truth, 3),
                "truth_s": round(truth / 1000.0, 6),
                "centered_model_label": prediction["label"],
                "centered_model_confidence": prediction["confidence"],
                "centered_prob_floor_bounce": probabilities.get("floor_bounce", ""),
                "centered_prob_noise": probabilities.get("noise", ""),
                "centered_prob_racket_bounce": probabilities.get("racket_bounce", ""),
                "centered_prob_table_bounce": probabilities.get("table_bounce", ""),
                "centered_bg_mode": bg_mode,
                "centered_bg_rms_db": features.get("nr_bg_rms_db", ""),
                "centered_confidence_threshold": threshold,
                "centered_clears_current_threshold": clears,
                "nearest_trigger_index": nearest_trigger.get("trigger_index", ""),
                "nearest_trigger_onset_ms": nearest_trigger.get("onset_ms", ""),
                "nearest_trigger_delta_ms": round(nearest_trigger_delta, 3) if math.isfinite(nearest_trigger_delta) else "",
                "nearest_trigger_abs_delta_ms": round(abs(nearest_trigger_delta), 3) if math.isfinite(nearest_trigger_delta) else "",
                "nearest_trigger_counted": nearest_trigger.get("counted", ""),
                "nearest_trigger_reject_reason": nearest_trigger.get("reject_reason", ""),
                "nearest_trigger_model_label": nearest_trigger.get("model_label", ""),
                "nearest_trigger_prob_racket_bounce": nearest_trigger.get("prob_racket_bounce", ""),
            }
        )
    summary = {
        "truth": len(truth_ms),
        "centered_model_label_counts": dict(sorted(labels.items())),
        "centered_racket_labels": racket_labels,
        "centered_clears_current_threshold": clears_threshold,
        "centered_racket_recall_if_perfect_gate": racket_labels / len(truth_ms) if truth_ms else 0.0,
        "centered_threshold_recall_if_perfect_gate": clears_threshold / len(truth_ms) if truth_ms else 0.0,
        "labels_with_nearest_trigger_within_60ms": sum(
            float(row["nearest_trigger_abs_delta_ms"]) <= 60.0 for row in rows if row["nearest_trigger_abs_delta_ms"] != ""
        ),
        "labels_with_nearest_trigger_within_120ms": sum(
            float(row["nearest_trigger_abs_delta_ms"]) <= 120.0 for row in rows if row["nearest_trigger_abs_delta_ms"] != ""
        ),
        "labels_with_nearest_trigger_within_140ms": sum(
            float(row["nearest_trigger_abs_delta_ms"]) <= MATCH_TOLERANCE_MS for row in rows if row["nearest_trigger_abs_delta_ms"] != ""
        ),
        "labels_with_nearest_trigger_within_250ms": sum(
            float(row["nearest_trigger_abs_delta_ms"]) <= 250.0 for row in rows if row["nearest_trigger_abs_delta_ms"] != ""
        ),
        "median_nearest_trigger_abs_delta_ms": float(np.median([
            float(row["nearest_trigger_abs_delta_ms"]) for row in rows if row["nearest_trigger_abs_delta_ms"] != ""
        ])) if rows else None,
        "max_nearest_trigger_abs_delta_ms": max(
            [float(row["nearest_trigger_abs_delta_ms"]) for row in rows if row["nearest_trigger_abs_delta_ms"] != ""],
            default=None,
        ),
    }
    return summary, rows


def saved_json_summary(path: Path) -> dict[str, Any]:
    if not path.exists():
        return {"status": "missing", "path": str(path)}
    payload = json.loads(path.read_text(encoding="utf-8"))
    events = list(payload.get("events") or [])
    labels = Counter(str(event.get("model_label") or "-") for event in events)
    reasons = Counter(str(event.get("reject_reason") or "counted") for event in events)
    probs = []
    for event in events:
        p = ((event.get("model_probabilities") or {}).get("racket_bounce"))
        value = finite_float(p)
        if math.isfinite(value):
            probs.append(value)
    return {
        "status": "loaded",
        "path": str(path),
        "native_candidates": int((payload.get("counts") or {}).get("native_candidates") or len(events)),
        "counted": int((payload.get("counts") or {}).get("counted") or sum(boolish(event.get("counted")) for event in events)),
        "event_rows": len(events),
        "model_label_counts": dict(sorted(labels.items())),
        "reject_reason_counts": dict(sorted(reasons.items())),
        "max_prob_racket_bounce": max(probs) if probs else None,
    }


def render_report(summary: dict[str, Any]) -> str:
    trigger = summary["current_app_trigger_replay"]
    centered = summary["label_centered_predictions"]
    saved = summary["saved_json"]
    return "\n".join(
        [
            "# T0063 T0060 Current-Only Fable Replay",
            "",
            "## Scope",
            "",
            "- Evaluation only; no training, export, APK build, app runtime change, TT Sounds, or historical local reviewed data.",
            f"- WAV: `{summary['wav']}`",
            f"- Labels CSV: `{summary['labels_csv']}`",
            f"- Model JSON: `{summary['model_json']}`",
            "",
            "## Saved App JSON",
            "",
            f"- Status: `{saved.get('status')}`",
            f"- Native candidates: `{saved.get('native_candidates')}`",
            f"- Counted: `{saved.get('counted')}`",
            f"- Model label counts: `{json.dumps(saved.get('model_label_counts', {}), sort_keys=True)}`",
            f"- Reject reason counts: `{json.dumps(saved.get('reject_reason_counts', {}), sort_keys=True)}`",
            f"- Max saved `racket_bounce` probability: `{saved.get('max_prob_racket_bounce')}`",
            "",
            "## Current-App Full-WAV Trigger Replay",
            "",
            f"- Truth labels: `{trigger['truth']}`",
            f"- Native-style gate triggers: `{trigger['triggers']}`",
            f"- Counted by current bundled Fable config: `{trigger['counted']}`",
            f"- Exact replay TP / FP / missed: `{trigger['tp']} / {trigger['fp']} / {trigger['missed']}`",
            f"- Precision / recall: `{trigger['precision']:.3f}` / `{trigger['recall']:.3f}`",
            f"- Model label counts: `{json.dumps(trigger['model_label_counts'], sort_keys=True)}`",
            f"- Reject reason counts: `{json.dumps(trigger['reject_reason_counts'], sort_keys=True)}`",
            "",
            "## Centered On Love's Labels",
            "",
            f"- Centered model label counts: `{json.dumps(centered['centered_model_label_counts'], sort_keys=True)}`",
            f"- Centered labels predicted as `racket_bounce`: `{centered['centered_racket_labels']}/{centered['truth']}`",
            f"- Centered labels clearing current confidence threshold: `{centered['centered_clears_current_threshold']}/{centered['truth']}`",
            f"- Labels with nearest trigger within 140 ms: `{centered['labels_with_nearest_trigger_within_140ms']}/{centered['truth']}`",
            f"- Labels with nearest trigger within 250 ms: `{centered['labels_with_nearest_trigger_within_250ms']}/{centered['truth']}`",
            f"- Median/max nearest-trigger delta: `{centered['median_nearest_trigger_abs_delta_ms']}` / `{centered['max_nearest_trigger_abs_delta_ms']}` ms",
            "",
            "## Interpretation",
            "",
            "- If centered predictions are weak, the current Fable model does not recognize this sound domain even at corrected timestamps.",
            "- If nearest-trigger coverage is weak, native/gate timing also contributes misses.",
            "- This report should be used as held-out evidence before any broader retrain/export decision.",
            "",
            "## Outputs",
            "",
            f"- Trigger replay CSV: `{summary['outputs']['trigger_replay_csv']}`",
            f"- Label-centered CSV: `{summary['outputs']['label_centered_csv']}`",
            f"- Summary JSON: `{summary['outputs']['summary_json']}`",
            "",
        ]
    )


def run(args: argparse.Namespace) -> dict[str, Any]:
    wav_path = Path(args.wav)
    labels_csv = Path(args.labels_csv)
    model_json = Path(args.model_json)
    debug_json = Path(args.debug_json)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    truth_ms = truth_times_ms(labels_csv)
    if not truth_ms:
        raise ValueError(f"No racket labels found in {labels_csv}")
    y, sample_rate = read_wav(wav_path)
    if sample_rate != nr_config.TARGET_SR:
        raise ValueError(f"Expected {nr_config.TARGET_SR} Hz WAV, got {sample_rate}")
    model = FableAppModel.load(model_json)

    trigger_summary, trigger_rows = current_app_trigger_replay(
        y=y,
        sample_rate=sample_rate,
        model=model,
        truth_ms=truth_ms,
    )
    centered_summary, centered_rows = label_centered_predictions(
        y=y,
        sample_rate=sample_rate,
        model=model,
        truth_ms=truth_ms,
        trigger_rows=trigger_rows,
    )

    trigger_csv = out_dir / "t0063_current_fable_trigger_replay.csv"
    centered_csv = out_dir / "t0063_current_fable_label_centered_predictions.csv"
    summary_json = out_dir / "t0063_current_only_summary.json"
    report_md = out_dir / "t0063_current_only_report.md"

    summary = {
        "ticket": "T0063-t0060-current-only-fable-replay",
        "changed_app_behavior": False,
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "wav": str(wav_path),
        "labels_csv": str(labels_csv),
        "model_json": str(model_json),
        "sample_rate_hz": sample_rate,
        "duration_s": len(y) / sample_rate,
        "saved_json": saved_json_summary(debug_json),
        "current_app_trigger_replay": trigger_summary,
        "label_centered_predictions": centered_summary,
        "outputs": {
            "trigger_replay_csv": str(trigger_csv),
            "label_centered_csv": str(centered_csv),
            "summary_json": str(summary_json),
            "report_md": str(report_md),
        },
    }

    write_csv(trigger_csv, trigger_rows)
    write_csv(centered_csv, centered_rows)
    summary_json.write_text(json.dumps(summary, indent=2, sort_keys=True), encoding="utf-8")
    report_md.write_text(render_report(summary), encoding="utf-8")
    return summary


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--wav", default=str(DEFAULT_WAV))
    parser.add_argument("--debug-json", default=str(DEFAULT_JSON))
    parser.add_argument("--labels-csv", default=str(DEFAULT_LABELS_CSV))
    parser.add_argument("--model-json", default=str(DEFAULT_MODEL_JSON))
    parser.add_argument("--out-dir", default=str(DEFAULT_OUT_DIR))
    return parser.parse_args()


def main() -> None:
    summary = run(parse_args())
    trigger = summary["current_app_trigger_replay"]
    centered = summary["label_centered_predictions"]
    print(f"truth={trigger['truth']} triggers={trigger['triggers']} counted={trigger['counted']}")
    print(f"exact TP/FP/missed={trigger['tp']}/{trigger['fp']}/{trigger['missed']}")
    print(
        "centered_racket="
        f"{centered['centered_racket_labels']}/{centered['truth']} "
        "centered_threshold="
        f"{centered['centered_clears_current_threshold']}/{centered['truth']}"
    )
    print(f"wrote {summary['outputs']['report_md']}")


if __name__ == "__main__":
    main()
