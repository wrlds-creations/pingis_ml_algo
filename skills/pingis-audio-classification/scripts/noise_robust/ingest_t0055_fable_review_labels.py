"""
T0055 Fable review-label ingest and retrain/replay planning.

Evaluation-only script. It reads Love's saved T0053/T0054 review labels for the
T0052 continuous Fable WAV, merges them with saved app/native trigger metadata,
and writes local ignored artifacts for the next retrain/replay decision.

No model training, export, APK build, or app runtime change happens here.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import statistics
import sys
import wave
from collections import Counter
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
from evaluate_fable_audio_reliability_t0044 import FableAppModel  # noqa: E402

ROOT_DIR = Path(__file__).resolve().parents[4]
SESSION_ID = "fable_live_session_2026-06-28T16-26-01-662Z"
T0052_EVAL_DIR = ROOT_DIR / "data" / "audio" / "models" / "evaluations" / "t0052_fable_continuous_debug_round"
T0053_EVAL_DIR = ROOT_DIR / "data" / "audio" / "models" / "evaluations" / "t0053_fable_trigger_review_ui"
T0052_RAW_DIR = ROOT_DIR / "data" / "audio" / "raw" / "t0052_fable_continuous_debug_round" / "fable_live_debug"
DEFAULT_LABELS_PATH = T0053_EVAL_DIR / f"{SESSION_ID}_review_labels.json"
DEFAULT_SAVED_EVENTS_CSV = T0052_EVAL_DIR / "t0052_saved_json_events.csv"
DEFAULT_OFFLINE_TRIGGERS_CSV = T0052_EVAL_DIR / "t0052_offline_full_wav_triggers.csv"
DEFAULT_WAV_PATH = T0052_RAW_DIR / f"{SESSION_ID}.wav"
DEFAULT_MODEL_JSON = ROOT_DIR / "apps" / "collector" / "src" / "models" / "fable_audio_model.json"
DEFAULT_OUT_DIR = ROOT_DIR / "data" / "audio" / "models" / "evaluations" / "t0055_fable_label_ingest_plan"

NEGATIVE_NEAR_RACKET_MS = 250.0
NATIVE_COVERAGE_MS = 180.0


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
        raise ValueError(f"Expected mono 16-bit PCM WAV, got channels={channels} width={sample_width}")
    y = np.frombuffer(frames, dtype="<i2").astype(np.float32) / 32768.0
    return y, sample_rate


def nearest_delta_ms(value_ms: float, candidates_ms: list[float]) -> float | None:
    if not candidates_ms or not math.isfinite(value_ms):
        return None
    return min((candidate - value_ms for candidate in candidates_ms), key=abs)


def stats(values: list[float]) -> dict[str, Any]:
    clean = [value for value in values if math.isfinite(value)]
    if not clean:
        return {"count": 0}
    ordered = sorted(clean)
    return {
        "count": len(clean),
        "min": ordered[0],
        "median": statistics.median(ordered),
        "mean": statistics.mean(ordered),
        "max": ordered[-1],
    }


def prediction_fields(prefix: str, prediction: dict[str, Any]) -> dict[str, Any]:
    probs = prediction.get("probabilities") or {}
    return {
        f"{prefix}_label": prediction.get("label") or "",
        f"{prefix}_confidence": prediction.get("confidence") or "",
        f"{prefix}_prob_racket_bounce": probs.get("racket_bounce", ""),
        f"{prefix}_prob_noise": probs.get("noise", ""),
        f"{prefix}_prob_table_bounce": probs.get("table_bounce", ""),
        f"{prefix}_prob_floor_bounce": probs.get("floor_bounce", ""),
    }


def predict_at_time(model: FableAppModel, y: np.ndarray, sample_rate: int, time_s: float) -> dict[str, Any]:
    sample = int(round(time_s * sample_rate))
    clip = nr_features.extract_live_clip(y, sample)
    features = nr_features.extract_all_features(clip, nr_config.TARGET_SR)
    return model.predict_features(features)


def target_for_label(label: str) -> str:
    if label == "racket":
        return "racket_bounce"
    if label == "noise":
        return "noise"
    return label or "unlabeled"


def train_role_for(label: str, source: str, nearest_racket_delta_ms: float | None) -> tuple[str, bool, str]:
    if label == "racket":
        role = "hard_positive" if source == "phone_trigger" else "manual_positive"
        return role, True, "reviewed racket contact"
    if label == "noise":
        if nearest_racket_delta_ms is not None and abs(nearest_racket_delta_ms) < NEGATIVE_NEAR_RACKET_MS:
            return "unsafe_negative_near_racket", False, f"noise marker is within {NEGATIVE_NEAR_RACKET_MS:.0f} ms of reviewed racket"
        return "hard_negative", True, "reviewed non-racket trigger"
    if label == "duplicate":
        return "duplicate_exclude", False, "duplicate label should not become independent training truth"
    return "unclear_exclude", False, "unclear/unlabeled rows need review before training"


def build_rows(
    review: dict[str, Any],
    saved_events: list[dict[str, str]],
    offline_triggers: list[dict[str, str]],
    model: FableAppModel,
    y: np.ndarray,
    sample_rate: int,
    wav_path: Path,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    events_by_index = {str(row.get("event_index")): row for row in saved_events}
    offline_times_ms = [
        finite_float(row.get("onset_ms_wav"))
        for row in offline_triggers
        if math.isfinite(finite_float(row.get("onset_ms_wav")))
    ]
    saved_times_ms = [
        finite_float(row.get("estimated_wav_ms"))
        for row in saved_events
        if math.isfinite(finite_float(row.get("estimated_wav_ms")))
    ]

    trigger_rows: list[dict[str, Any]] = []
    for trigger_id, label_info in sorted(
        (review.get("trigger_labels") or {}).items(),
        key=lambda item: finite_float((item[1] or {}).get("time_s"), 0.0),
    ):
        event_index = str(label_info.get("event_index") or trigger_id.replace("event_", ""))
        event = events_by_index.get(event_index, {})
        raw_time_s = finite_float(event.get("estimated_wav_ms")) / 1000.0
        original_time_s = finite_float(label_info.get("original_time_s"), raw_time_s)
        adjusted = finite_float(label_info.get("adjusted_time_s"))
        reviewed_time_s = adjusted if math.isfinite(adjusted) else finite_float(label_info.get("time_s"), raw_time_s)
        label = str(label_info.get("label") or "")
        prediction_reviewed = predict_at_time(model, y, sample_rate, reviewed_time_s)
        nearest_offline_delta = nearest_delta_ms(reviewed_time_s * 1000.0, offline_times_ms)
        nearest_saved_delta = nearest_delta_ms(reviewed_time_s * 1000.0, saved_times_ms)
        trigger_rows.append({
            "session_id": review.get("session_id") or SESSION_ID,
            "row_id": trigger_id,
            "source": "phone_trigger",
            "event_index": event_index,
            "review_label": label,
            "target_label": target_for_label(label),
            "reviewed_time_s": round(reviewed_time_s, 6),
            "raw_saved_time_s": round(raw_time_s, 6) if math.isfinite(raw_time_s) else "",
            "original_time_s": round(original_time_s, 6) if math.isfinite(original_time_s) else "",
            "adjusted_time_s": round(adjusted, 6) if math.isfinite(adjusted) else "",
            "time_adjustment_ms": round((reviewed_time_s - raw_time_s) * 1000.0, 3) if math.isfinite(raw_time_s) else "",
            "nearest_offline_trigger_delta_ms": "" if nearest_offline_delta is None else round(nearest_offline_delta, 3),
            "nearest_saved_event_delta_ms": "" if nearest_saved_delta is None else round(nearest_saved_delta, 3),
            "saved_counted": event.get("counted", ""),
            "saved_reject_reason": event.get("reject_reason", ""),
            "saved_model_label": event.get("model_label", ""),
            "saved_model_confidence": event.get("model_confidence", ""),
            "saved_prob_racket_bounce": event.get("prob_racket_bounce", ""),
            "saved_prob_noise": event.get("prob_noise", ""),
            "native_rms": event.get("native_rms", ""),
            "native_background_rms": event.get("native_background_rms", ""),
            "bg_mode": event.get("bg_mode", ""),
            "note": label_info.get("note", ""),
            **prediction_fields("reviewed_anchor_current_model", prediction_reviewed),
        })

    manual_rows: list[dict[str, Any]] = []
    for marker in review.get("manual_markers") or []:
        reviewed_time_s = finite_float(marker.get("time_s"))
        label = str(marker.get("label") or "")
        prediction_reviewed = predict_at_time(model, y, sample_rate, reviewed_time_s)
        nearest_offline_delta = nearest_delta_ms(reviewed_time_s * 1000.0, offline_times_ms)
        nearest_saved_delta = nearest_delta_ms(reviewed_time_s * 1000.0, saved_times_ms)
        manual_rows.append({
            "session_id": review.get("session_id") or SESSION_ID,
            "row_id": marker.get("id") or "",
            "source": "manual_marker",
            "event_index": "manual",
            "review_label": label,
            "target_label": target_for_label(label),
            "reviewed_time_s": round(reviewed_time_s, 6),
            "raw_saved_time_s": "",
            "original_time_s": "",
            "adjusted_time_s": "",
            "time_adjustment_ms": "",
            "nearest_offline_trigger_delta_ms": "" if nearest_offline_delta is None else round(nearest_offline_delta, 3),
            "nearest_saved_event_delta_ms": "" if nearest_saved_delta is None else round(nearest_saved_delta, 3),
            "saved_counted": "",
            "saved_reject_reason": "manual_missing_saved_trigger",
            "saved_model_label": "",
            "saved_model_confidence": "",
            "saved_prob_racket_bounce": "",
            "saved_prob_noise": "",
            "native_rms": "",
            "native_background_rms": "",
            "bg_mode": "",
            "note": marker.get("note", ""),
            **prediction_fields("reviewed_anchor_current_model", prediction_reviewed),
        })

    timeline = sorted(trigger_rows + manual_rows, key=lambda row: finite_float(row.get("reviewed_time_s"), 0.0))
    racket_times_ms = [finite_float(row["reviewed_time_s"]) * 1000.0 for row in timeline if row.get("review_label") == "racket"]
    candidate_rows: list[dict[str, Any]] = []
    for row in timeline:
        nearest_racket_delta = nearest_delta_ms(finite_float(row["reviewed_time_s"]) * 1000.0, racket_times_ms)
        if row.get("review_label") == "racket":
            nearest_racket_delta = 0.0
        role, include, reason = train_role_for(str(row.get("review_label") or ""), str(row.get("source") or ""), nearest_racket_delta)
        candidate_rows.append({
            "session_id": row["session_id"],
            "source_wav": str(wav_path),
            "row_id": row["row_id"],
            "source": row["source"],
            "event_index": row["event_index"],
            "reviewed_time_s": row["reviewed_time_s"],
            "anchor_sample": int(round(finite_float(row["reviewed_time_s"]) * sample_rate)),
            "sample_rate_hz": sample_rate,
            "clip_pre_ms": 100,
            "clip_post_ms": 200,
            "review_label": row["review_label"],
            "target_label": row["target_label"],
            "train_role_suggestion": role,
            "include_in_candidate_rows": include,
            "exclusion_reason": "" if include else reason,
            "nearest_reviewed_racket_delta_ms": "" if nearest_racket_delta is None else round(nearest_racket_delta, 3),
            "requires_love_training_approval": True,
            "bucket_suggestion": "ordinary_self_practice_messy_speech_background",
            "notes": reason if include else "",
        })
    return timeline, candidate_rows


def summarize(
    review: dict[str, Any],
    timeline: list[dict[str, Any]],
    candidate_rows: list[dict[str, Any]],
    saved_events: list[dict[str, str]],
    offline_triggers: list[dict[str, str]],
    wav_duration_s: float,
) -> dict[str, Any]:
    trigger_rows = [row for row in timeline if row.get("source") == "phone_trigger"]
    manual_rows = [row for row in timeline if row.get("source") == "manual_marker"]
    racket_rows = [row for row in timeline if row.get("review_label") == "racket"]
    trigger_racket_rows = [row for row in trigger_rows if row.get("review_label") == "racket"]
    manual_racket_rows = [row for row in manual_rows if row.get("review_label") == "racket"]
    adjusted_rows = [row for row in trigger_rows if row.get("adjusted_time_s") != ""]
    time_adjustments = [finite_float(row.get("time_adjustment_ms")) for row in adjusted_rows]
    abs_time_adjustments = [abs(value) for value in time_adjustments if math.isfinite(value)]
    saved_model_confusion = Counter(
        f"{row.get('review_label') or '-'} -> {row.get('saved_model_label') or '-'}"
        for row in trigger_rows
    )
    reviewed_anchor_confusion = Counter(
        f"{row.get('review_label') or '-'} -> {row.get('reviewed_anchor_current_model_label') or '-'}"
        for row in timeline
    )
    saved_racket_probs = [
        finite_float(row.get("saved_prob_racket_bounce"))
        for row in trigger_racket_rows
        if row.get("saved_prob_racket_bounce") != ""
    ]
    reviewed_anchor_racket_probs = [
        finite_float(row.get("reviewed_anchor_current_model_prob_racket_bounce"))
        for row in racket_rows
        if row.get("reviewed_anchor_current_model_prob_racket_bounce") != ""
    ]
    included = [row for row in candidate_rows if boolish(row.get("include_in_candidate_rows"))]
    included_by_target = Counter(str(row.get("target_label") or "-") for row in included)
    train_roles = Counter(str(row.get("train_role_suggestion") or "-") for row in candidate_rows)
    unsafe_negatives = [row for row in candidate_rows if row.get("train_role_suggestion") == "unsafe_negative_near_racket"]
    total_racket = len(racket_rows)
    native_coverage = len(trigger_racket_rows) / total_racket if total_racket else 0.0
    current_counted_reviewed_rackets = sum(
        1 for row in trigger_racket_rows if boolish(row.get("saved_counted"))
    )
    centered_racket_label_hits = sum(
        1 for row in racket_rows if row.get("reviewed_anchor_current_model_label") == "racket_bounce"
    )
    return {
        "ticket": "T0055-fable-label-ingest-and-retrain-plan",
        "changed_app_behavior": False,
        "session_id": review.get("session_id") or SESSION_ID,
        "expected_count": review.get("expected_count"),
        "reported_app_count": review.get("reported_app_count"),
        "wav_duration_s": round(wav_duration_s, 6),
        "saved_trigger_rows": len(saved_events),
        "offline_trigger_rows": len(offline_triggers),
        "reviewed_rows": len(timeline),
        "reviewed_trigger_labels": len(trigger_rows),
        "manual_markers": len(manual_rows),
        "review_label_counts": dict(sorted(Counter(str(row.get("review_label") or "-") for row in timeline).items())),
        "trigger_review_label_counts": dict(sorted(Counter(str(row.get("review_label") or "-") for row in trigger_rows).items())),
        "manual_review_label_counts": dict(sorted(Counter(str(row.get("review_label") or "-") for row in manual_rows).items())),
        "reviewed_racket_total": total_racket,
        "reviewed_racket_from_native_trigger": len(trigger_racket_rows),
        "reviewed_racket_manual_missing_trigger": len(manual_racket_rows),
        "native_trigger_coverage_of_reviewed_rackets": round(native_coverage, 6),
        "adjusted_trigger_rows": len(adjusted_rows),
        "time_adjustment_ms_stats": stats(time_adjustments),
        "abs_time_adjustment_ms_stats": stats(abs_time_adjustments),
        "saved_model_confusion_review_to_model": dict(sorted(saved_model_confusion.items())),
        "reviewed_anchor_current_model_confusion": dict(sorted(reviewed_anchor_confusion.items())),
        "saved_racket_probability_on_reviewed_racket_triggers": stats(saved_racket_probs),
        "reviewed_anchor_racket_probability_on_reviewed_rackets": stats(reviewed_anchor_racket_probs),
        "current_app_counted_reviewed_racket_triggers": current_counted_reviewed_rackets,
        "current_app_recall_on_reviewed_rackets": round(current_counted_reviewed_rackets / total_racket, 6) if total_racket else 0.0,
        "current_model_centered_racket_label_hits": centered_racket_label_hits,
        "current_model_centered_racket_label_recall": round(centered_racket_label_hits / total_racket, 6) if total_racket else 0.0,
        "candidate_rows": len(candidate_rows),
        "candidate_rows_included_suggestion": len(included),
        "candidate_rows_included_by_target": dict(sorted(included_by_target.items())),
        "candidate_train_role_counts": dict(sorted(train_roles.items())),
        "unsafe_negative_near_racket_rows": len(unsafe_negatives),
        "decision": {
            "is_enough_for_safe_standalone_retrain": False,
            "recommended_next_ticket": "T0056-feature-window-audit-plus-candidate-retrain-replay",
            "recommendation": (
                "Use these labels as a high-value C2 hard-positive/negative slice, but do not train/export from this "
                "single 20.5 s clip alone. T0056 should first run a feature/window audit using reviewed anchors, then "
                "combine approved local hard positives and hard negatives for an app-style candidate retrain/replay."
            ),
            "why": [
                "The app counted 0 of 30 reviewed racket contacts.",
                "Native/saved trigger coverage is high: most reviewed racket contacts already have a phone trigger candidate.",
                "The saved model labels reviewed racket trigger candidates as noise, so a simple threshold change is not enough.",
                "The dataset is only one short session and should be combined with other approved local positives and hard negatives before promotion.",
            ],
        },
    }


def write_report(path: Path, summary: dict[str, Any], timeline: list[dict[str, Any]]) -> None:
    racket_rows = [row for row in timeline if row.get("review_label") == "racket"]
    noise_rows = [row for row in timeline if row.get("review_label") == "noise"]
    lines = [
        "# T0055 Fable Label Ingest Plan",
        "",
        "## Inputs",
        "",
        f"- Session: `{summary['session_id']}`",
        f"- Expected count: `{summary['expected_count']}`",
        f"- App count: `{summary['reported_app_count']}`",
        f"- WAV duration: `{summary['wav_duration_s']}` s",
        "",
        "## Label Summary",
        "",
        f"- Reviewed rows: `{summary['reviewed_rows']}`",
        f"- Reviewed racket contacts: `{summary['reviewed_racket_total']}`",
        f"- Native-trigger racket contacts: `{summary['reviewed_racket_from_native_trigger']}`",
        f"- Manual missing-trigger racket contacts: `{summary['reviewed_racket_manual_missing_trigger']}`",
        f"- Noise labels: `{len(noise_rows)}`",
        f"- Adjusted trigger rows: `{summary['adjusted_trigger_rows']}`",
        f"- Native trigger coverage of reviewed rackets: `{summary['native_trigger_coverage_of_reviewed_rackets']}`",
        "",
        "## Current Model Result",
        "",
        f"- Current app counted reviewed rackets: `{summary['current_app_counted_reviewed_racket_triggers']}/{summary['reviewed_racket_total']}`",
        f"- Current model on reviewed-centered anchors predicted racket: `{summary['current_model_centered_racket_label_hits']}/{summary['reviewed_racket_total']}`",
        f"- Saved model confusion: `{json.dumps(summary['saved_model_confusion_review_to_model'], sort_keys=True)}`",
        f"- Reviewed-anchor model confusion: `{json.dumps(summary['reviewed_anchor_current_model_confusion'], sort_keys=True)}`",
        "",
        "## Candidate Rows",
        "",
        f"- Candidate rows: `{summary['candidate_rows']}`",
        f"- Suggested included rows: `{summary['candidate_rows_included_suggestion']}`",
        f"- Included by target: `{json.dumps(summary['candidate_rows_included_by_target'], sort_keys=True)}`",
        f"- Role counts: `{json.dumps(summary['candidate_train_role_counts'], sort_keys=True)}`",
        f"- Unsafe noise rows near reviewed racket: `{summary['unsafe_negative_near_racket_rows']}`",
        "",
        "## Decision",
        "",
        f"- Safe standalone retrain from this clip only: `{summary['decision']['is_enough_for_safe_standalone_retrain']}`",
        f"- Recommended next ticket: `{summary['decision']['recommended_next_ticket']}`",
        "",
        summary["decision"]["recommendation"],
        "",
        "Reasons:",
    ]
    lines.extend(f"- {reason}" for reason in summary["decision"]["why"])
    lines.extend([
        "",
        "## Reviewed Racket Timeline",
        "",
        "| Row | Source | Time s | Saved model | Reviewed-anchor model | Saved racket prob | Reviewed-anchor racket prob |",
        "|---|---|---:|---|---|---:|---:|",
    ])
    for row in racket_rows:
        lines.append(
            "| {row_id} | {source} | {time} | {saved_model} | {anchor_model} | {saved_prob} | {anchor_prob} |".format(
                row_id=row.get("row_id", ""),
                source=row.get("source", ""),
                time=row.get("reviewed_time_s", ""),
                saved_model=row.get("saved_model_label", "-") or "-",
                anchor_model=row.get("reviewed_anchor_current_model_label", "-") or "-",
                saved_prob=row.get("saved_prob_racket_bounce", "") or "-",
                anchor_prob=row.get("reviewed_anchor_current_model_prob_racket_bounce", "") or "-",
            )
        )
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def run(args: argparse.Namespace) -> dict[str, Any]:
    labels_path = Path(args.labels_path)
    saved_events_csv = Path(args.saved_events_csv)
    offline_triggers_csv = Path(args.offline_triggers_csv)
    wav_path = Path(args.wav_path)
    model_json = Path(args.model_json)
    out_dir = Path(args.out_dir)
    review = json.loads(labels_path.read_text(encoding="utf-8"))
    saved_events = read_csv(saved_events_csv)
    offline_triggers = read_csv(offline_triggers_csv)
    y, sample_rate = read_wav(wav_path)
    model = FableAppModel.load(model_json)
    timeline, candidate_rows = build_rows(review, saved_events, offline_triggers, model, y, sample_rate, wav_path)
    summary = summarize(review, timeline, candidate_rows, saved_events, offline_triggers, len(y) / sample_rate)
    out_dir.mkdir(parents=True, exist_ok=True)
    write_csv(out_dir / "t0055_reviewed_timeline.csv", timeline)
    write_csv(out_dir / "t0055_candidate_rows.csv", candidate_rows)
    (out_dir / "t0055_summary.json").write_text(json.dumps(summary, indent=2, sort_keys=True), encoding="utf-8")
    write_report(out_dir / "t0055_report.md", summary, timeline)
    return summary


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--labels-path", default=str(DEFAULT_LABELS_PATH))
    parser.add_argument("--saved-events-csv", default=str(DEFAULT_SAVED_EVENTS_CSV))
    parser.add_argument("--offline-triggers-csv", default=str(DEFAULT_OFFLINE_TRIGGERS_CSV))
    parser.add_argument("--wav-path", default=str(DEFAULT_WAV_PATH))
    parser.add_argument("--model-json", default=str(DEFAULT_MODEL_JSON))
    parser.add_argument("--out-dir", default=str(DEFAULT_OUT_DIR))
    return parser.parse_args()


def main() -> None:
    summary = run(parse_args())
    print(json.dumps({
        "session_id": summary["session_id"],
        "reviewed_racket_total": summary["reviewed_racket_total"],
        "native_trigger_coverage_of_reviewed_rackets": summary["native_trigger_coverage_of_reviewed_rackets"],
        "current_app_recall_on_reviewed_rackets": summary["current_app_recall_on_reviewed_rackets"],
        "current_model_centered_racket_label_recall": summary["current_model_centered_racket_label_recall"],
        "candidate_rows_included_by_target": summary["candidate_rows_included_by_target"],
        "recommended_next_ticket": summary["decision"]["recommended_next_ticket"],
    }, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
