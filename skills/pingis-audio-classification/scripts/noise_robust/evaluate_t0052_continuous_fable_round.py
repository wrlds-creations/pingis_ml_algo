"""
T0052 continuous Fable debug round audit.

Evaluation-only script for the first hard-mode Fable-algoritm run after T0051
added continuous WAV capture. It compares the saved JSON event timeline with
the full continuous WAV and Love's expected/app counts.

No model training, export, APK build, or app runtime change happens here.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import sys
import wave
from collections import Counter
from datetime import datetime
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
DEFAULT_DEBUG_DIR = ROOT_DIR / "data" / "audio" / "raw" / "t0052_fable_continuous_debug_round" / "fable_live_debug"
DEFAULT_SESSION_ID = "fable_live_session_2026-06-28T16-26-01-662Z"
DEFAULT_OUT_DIR = ROOT_DIR / "data" / "audio" / "models" / "evaluations" / "t0052_fable_continuous_debug_round"
MODEL_JSON = ROOT_DIR / "apps" / "collector" / "src" / "models" / "fable_audio_model.json"

EXPECTED_COUNT = 30
REPORTED_APP_COUNT = 0

# Mirror apps/collector/src/fableEngine.ts as of T0051/T0052.
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


def boolish(value: Any) -> bool:
    return value is True or str(value).lower() == "true"


def finite_float(value: Any, default: float = 0.0) -> float:
    try:
        out = float(value)
    except Exception:
        return default
    return out if math.isfinite(out) else default


def parse_iso_ms(value: str) -> float:
    return datetime.fromisoformat(value.replace("Z", "+00:00")).timestamp() * 1000.0


def read_wav(path: Path) -> tuple[np.ndarray, int, int]:
    with wave.open(str(path), "rb") as wav:
        channels = wav.getnchannels()
        sample_width = wav.getsampwidth()
        sample_rate = wav.getframerate()
        frames = wav.readframes(wav.getnframes())
    if channels != 1 or sample_width != 2:
        raise ValueError(f"Expected mono 16-bit PCM WAV, got channels={channels} width={sample_width}")
    y = np.frombuffer(frames, dtype="<i2").astype(np.float32) / 32768.0
    return y, sample_rate, len(y)


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fields: list[str] = []
    seen: set[str] = set()
    for row in rows:
        for key in row:
            if key not in seen:
                seen.add(key)
                fields.append(key)
    if not fields:
        fields = ["empty"]
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        for row in rows:
            writer.writerow({key: row.get(key, "") for key in fields})


def counter_json(counter: Counter[str]) -> dict[str, int]:
    return dict(sorted(counter.items(), key=lambda item: (-item[1], item[0])))


def nearest_delta_ms(value: float, candidates: list[float]) -> float | None:
    if not candidates:
        return None
    return min((candidate - value for candidate in candidates), key=abs)


def event_rows(payload: dict[str, Any], start_ms: float) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for event in payload.get("events") or []:
        probs = event.get("model_probabilities") or {}
        onset_epoch_ms = finite_float(event.get("native_onset_time_ms"), default=float("nan"))
        rows.append({
            "event_index": event.get("index", ""),
            "event_rel_ms_json_start": round(onset_epoch_ms - start_ms, 3) if math.isfinite(onset_epoch_ms) else "",
            "received_minus_onset_ms": (
                round(finite_float(event.get("received_at_ms")) - onset_epoch_ms, 3)
                if math.isfinite(onset_epoch_ms) and event.get("received_at_ms") is not None
                else ""
            ),
            "counted": boolish(event.get("counted")),
            "reject_reason": event.get("reject_reason") or "",
            "model_label": event.get("model_label") or "",
            "model_confidence": event.get("model_confidence") or "",
            "prob_racket_bounce": probs.get("racket_bounce", ""),
            "prob_noise": probs.get("noise", ""),
            "prob_table_bounce": probs.get("table_bounce", ""),
            "prob_floor_bounce": probs.get("floor_bounce", ""),
            "native_rms": event.get("native_rms", ""),
            "native_background_rms": event.get("native_background_rms", ""),
            "bg_mode": event.get("bg_mode", ""),
            "bg_rms_db": event.get("bg_rms_db", ""),
            "feature_ms": event.get("feature_ms", ""),
            "predict_ms": event.get("predict_ms", ""),
            "has_audio_b64": bool(event.get("audio_b64")),
        })
    return rows


def offline_trigger_rows(y: np.ndarray, sample_rate: int, model: FableAppModel) -> list[dict[str, Any]]:
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
    rows: list[dict[str, Any]] = []
    for index, trigger in enumerate(triggers, start=1):
        if not trigger.get("passed_spectral"):
            continue
        onset_sample = int(trigger["onset_sample"])
        clip = nr_features.extract_live_clip(y, onset_sample)
        result = counter.process_clip(clip, float(trigger["onset_ms"]), float(trigger["frame_rms"]))
        prediction = result.get("prediction") or {}
        probs = prediction.get("probabilities") or {}
        rows.append({
            "trigger_index": index,
            "onset_ms_wav": round(float(trigger["onset_ms"]), 3),
            "onset_sample": onset_sample,
            "frame_rms": trigger.get("frame_rms", ""),
            "background_rms": trigger.get("bg_rms", ""),
            "counted_current_app_config": bool(result.get("counted")),
            "reject_reason_current_app_config": result.get("reject_reason") or "",
            "model_label_current_app_config": prediction.get("label") or "",
            "model_confidence_current_app_config": prediction.get("confidence") or "",
            "prob_racket_bounce_current_app_config": probs.get("racket_bounce", ""),
            "prob_noise_current_app_config": probs.get("noise", ""),
            "prob_table_bounce_current_app_config": probs.get("table_bounce", ""),
            "prob_floor_bounce_current_app_config": probs.get("floor_bounce", ""),
            "bg_mode_current_app_config": result.get("bg_mode") or "",
            "bg_rms_db_current_app_config": result.get("bg_rms_db") or "",
        })
    return rows


def add_alignment(event_rows_in: list[dict[str, Any]], trigger_rows_in: list[dict[str, Any]]) -> float:
    event_times = [
        finite_float(row.get("event_rel_ms_json_start"), default=float("nan"))
        for row in event_rows_in
        if row.get("event_rel_ms_json_start") != ""
    ]
    trigger_times = [finite_float(row.get("onset_ms_wav")) for row in trigger_rows_in]
    paired = [
        event_time - trigger_time
        for event_time, trigger_time in zip(event_times, trigger_times)
        if math.isfinite(event_time) and math.isfinite(trigger_time)
    ]
    offset_ms = float(np.median(paired)) if paired else 0.0
    for row in event_rows_in:
        event_rel = finite_float(row.get("event_rel_ms_json_start"), default=float("nan"))
        if not math.isfinite(event_rel):
            row["estimated_wav_ms"] = ""
            row["nearest_offline_trigger_delta_ms"] = ""
            continue
        wav_ms = event_rel - offset_ms
        row["estimated_wav_ms"] = round(wav_ms, 3)
        delta = nearest_delta_ms(wav_ms, trigger_times)
        row["nearest_offline_trigger_delta_ms"] = "" if delta is None else round(delta, 3)
    event_wav_times = [
        finite_float(row.get("estimated_wav_ms"), default=float("nan"))
        for row in event_rows_in
        if row.get("estimated_wav_ms") != ""
    ]
    for row in trigger_rows_in:
        trigger_ms = finite_float(row.get("onset_ms_wav"), default=float("nan"))
        delta = nearest_delta_ms(trigger_ms, event_wav_times)
        row["nearest_json_event_delta_ms"] = "" if delta is None else round(delta, 3)
    return offset_ms


def summarize(payload: dict[str, Any], y: np.ndarray, sample_rate: int, events: list[dict[str, Any]], triggers: list[dict[str, Any]], offset_ms: float) -> dict[str, Any]:
    counted_events = [row for row in events if boolish(row.get("counted"))]
    labels = Counter(str(row.get("model_label") or "-") for row in events)
    reasons = Counter(str(row.get("reject_reason") or "counted") for row in events)
    trigger_labels = Counter(str(row.get("model_label_current_app_config") or "-") for row in triggers)
    trigger_reasons = Counter(str(row.get("reject_reason_current_app_config") or "counted") for row in triggers)
    app_racket_probs = [
        finite_float(row.get("prob_racket_bounce"), default=float("nan"))
        for row in events
        if row.get("prob_racket_bounce") != ""
    ]
    app_racket_probs = [value for value in app_racket_probs if math.isfinite(value)]
    offline_racket_probs = [
        finite_float(row.get("prob_racket_bounce_current_app_config"), default=float("nan"))
        for row in triggers
        if row.get("prob_racket_bounce_current_app_config") != ""
    ]
    offline_racket_probs = [value for value in offline_racket_probs if math.isfinite(value)]
    stale_rows = [row for row in events if row.get("reject_reason") == "stale_backlog"]
    received_lags = [
        finite_float(row.get("received_minus_onset_ms"), default=float("nan"))
        for row in events
        if row.get("received_minus_onset_ms") != ""
    ]
    received_lags = [value for value in received_lags if math.isfinite(value)]
    return {
        "ticket": "T0052-fable-continuous-debug-round-audit",
        "changed_app_behavior": False,
        "expected_count": EXPECTED_COUNT,
        "reported_app_count": REPORTED_APP_COUNT,
        "json_file": payload.get("_json_file", ""),
        "wav_file": payload.get("_wav_file", ""),
        "started_at": payload.get("started_at", ""),
        "stopped_at": payload.get("stopped_at", ""),
        "wav_duration_s": round(len(y) / sample_rate, 6),
        "wav_sample_rate_hz": sample_rate,
        "json_native_candidates": int((payload.get("counts") or {}).get("native_candidates") or len(events)),
        "json_counted": int((payload.get("counts") or {}).get("counted") or len(counted_events)),
        "json_event_rows": len(events),
        "json_model_label_counts": counter_json(labels),
        "json_reject_reason_counts": counter_json(reasons),
        "json_max_racket_probability": max(app_racket_probs) if app_racket_probs else None,
        "json_events_with_racket_label": labels.get("racket_bounce", 0),
        "json_stale_rows": len(stale_rows),
        "received_minus_onset_ms_max": max(received_lags) if received_lags else None,
        "received_minus_onset_ms_median": float(np.median(received_lags)) if received_lags else None,
        "offline_gate_trigger_rows": len(triggers),
        "offline_counted_current_app_config": sum(1 for row in triggers if boolish(row.get("counted_current_app_config"))),
        "offline_model_label_counts_current_app_config": counter_json(trigger_labels),
        "offline_reject_reason_counts_current_app_config": counter_json(trigger_reasons),
        "offline_events_with_racket_label_current_app_config": trigger_labels.get("racket_bounce", 0),
        "offline_max_racket_probability_current_app_config": max(offline_racket_probs) if offline_racket_probs else None,
        "json_to_wav_alignment_offset_ms_median": offset_ms,
        "interpretation": {
            "primary_failure": "model_rejection_after_native_onset",
            "notes": [
                "The saved app timeline has native candidates but zero counted racket bounces.",
                "Every classified saved app candidate was rejected as not_racket with model_label=noise.",
                "The four stale_backlog rows happened early and cannot explain the whole 30-contact miss.",
                "Offline gate replay over the full WAV produces the same trigger count scale, so the continuous WAV is usable and the native gate is not the only zero-count bottleneck.",
            ],
        },
    }


def top_rows_by_racket_probability(events: list[dict[str, Any]], triggers: list[dict[str, Any]]) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for row in events:
        if row.get("prob_racket_bounce") == "":
            continue
        rows.append({
            "source": "saved_json_event",
            "time_ms": row.get("estimated_wav_ms", ""),
            "index": row.get("event_index", ""),
            "counted": row.get("counted", ""),
            "reject_reason": row.get("reject_reason", ""),
            "model_label": row.get("model_label", ""),
            "model_confidence": row.get("model_confidence", ""),
            "prob_racket_bounce": row.get("prob_racket_bounce", ""),
            "prob_noise": row.get("prob_noise", ""),
            "native_rms": row.get("native_rms", ""),
        })
    for row in triggers:
        rows.append({
            "source": "offline_full_wav_trigger",
            "time_ms": row.get("onset_ms_wav", ""),
            "index": row.get("trigger_index", ""),
            "counted": row.get("counted_current_app_config", ""),
            "reject_reason": row.get("reject_reason_current_app_config", ""),
            "model_label": row.get("model_label_current_app_config", ""),
            "model_confidence": row.get("model_confidence_current_app_config", ""),
            "prob_racket_bounce": row.get("prob_racket_bounce_current_app_config", ""),
            "prob_noise": row.get("prob_noise_current_app_config", ""),
            "native_rms": row.get("frame_rms", ""),
        })
    rows.sort(key=lambda row: finite_float(row.get("prob_racket_bounce")), reverse=True)
    return rows[:30]


def write_report(path: Path, summary: dict[str, Any]) -> None:
    lines = [
        "# T0052 Fable Continuous Debug Round Audit",
        "",
        "## Scope",
        "",
        "- Evaluation only; no app/model/APK/runtime change.",
        "- Source is Love's first continuous-WAV hard-mode `Fable-algoritm` run after T0051.",
        f"- Expected human count: `{summary['expected_count']}`.",
        f"- Reported app count: `{summary['reported_app_count']}`.",
        "",
        "## Files",
        "",
        f"- JSON: `{summary['json_file']}`",
        f"- WAV: `{summary['wav_file']}`",
        f"- WAV duration/sample rate: `{summary['wav_duration_s']}` s / `{summary['wav_sample_rate_hz']}` Hz.",
        f"- Median JSON-start to WAV-start alignment offset: `{summary['json_to_wav_alignment_offset_ms_median']:.3f}` ms.",
        "",
        "## Saved App Timeline",
        "",
        f"- Native candidates: `{summary['json_native_candidates']}`.",
        f"- Counted by saved JSON: `{summary['json_counted']}`.",
        f"- Events with `racket_bounce` model label: `{summary['json_events_with_racket_label']}`.",
        f"- Max saved-app `racket_bounce` probability: `{summary['json_max_racket_probability']}`.",
        f"- Model label counts: `{json.dumps(summary['json_model_label_counts'], sort_keys=True)}`.",
        f"- Reject reason counts: `{json.dumps(summary['json_reject_reason_counts'], sort_keys=True)}`.",
        f"- Stale backlog rows: `{summary['json_stale_rows']}`.",
        "",
        "## Full-WAV Offline Gate Replay",
        "",
        f"- Offline native-style gate triggers: `{summary['offline_gate_trigger_rows']}`.",
        f"- Offline counted with current app config: `{summary['offline_counted_current_app_config']}`.",
        f"- Offline `racket_bounce` model labels: `{summary['offline_events_with_racket_label_current_app_config']}`.",
        f"- Offline max `racket_bounce` probability: `{summary['offline_max_racket_probability_current_app_config']}`.",
        f"- Offline label counts: `{json.dumps(summary['offline_model_label_counts_current_app_config'], sort_keys=True)}`.",
        f"- Offline reject reason counts: `{json.dumps(summary['offline_reject_reason_counts_current_app_config'], sort_keys=True)}`.",
        "",
        "## Interpretation",
        "",
        "- This is primarily a model/domain rejection failure after native onset detection, not a total mic/native-trigger failure.",
        "- The app saw `48` native candidates in a `20.5 s` WAV, but all classified saved events were labeled `noise` and rejected as `not_racket`.",
        "- The `4` stale backlog rows are real latency evidence, but they occurred early and cannot explain `30 -> 0` by themselves.",
        "- The full-WAV offline gate also finds the same trigger-count scale. It produces a few borderline `racket_bounce` predictions, but still nowhere near the expected count.",
        "- A confidence increase or speech veto would make this worse. The next useful fix is a fuller local retrain/replay or feature/window audit using hard positives like this run plus talking/handling hard negatives.",
        "",
        "## Recommended Next Step",
        "",
        "- Treat this run as diagnostic/holdout until exact bounce timestamps are reviewed.",
        "- For T0053, build a candidate retrain/replay plan that includes: C2-style positives, speaking/counting positives, messy/failed-practice positives, and talking/racket-handling negatives.",
        "- If we want to train from this WAV, first create exact timestamp labels from the continuous audio instead of using only the block-level count `30`.",
        "",
    ]
    path.write_text("\n".join(lines), encoding="utf-8")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Audit T0052 Fable continuous debug JSON/WAV pair.")
    parser.add_argument("--debug-dir", type=Path, default=DEFAULT_DEBUG_DIR)
    parser.add_argument("--session-id", default=DEFAULT_SESSION_ID)
    parser.add_argument("--model-json", type=Path, default=MODEL_JSON)
    parser.add_argument("--out-dir", type=Path, default=DEFAULT_OUT_DIR)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    json_path = args.debug_dir / f"{args.session_id}.json"
    wav_path = args.debug_dir / f"{args.session_id}.wav"
    if not json_path.exists():
        raise SystemExit(f"Missing JSON: {json_path}")
    if not wav_path.exists():
        raise SystemExit(f"Missing WAV: {wav_path}")

    payload = json.loads(json_path.read_text(encoding="utf-8"))
    payload["_json_file"] = str(json_path)
    payload["_wav_file"] = str(wav_path)
    y, sample_rate, _n_samples = read_wav(wav_path)
    if sample_rate != nr_config.TARGET_SR:
        raise SystemExit(f"Expected {nr_config.TARGET_SR} Hz WAV, got {sample_rate} Hz")

    model = FableAppModel.load(args.model_json)
    start_ms = parse_iso_ms(str(payload["started_at"]))
    events = event_rows(payload, start_ms)
    triggers = offline_trigger_rows(y, sample_rate, model)
    offset_ms = add_alignment(events, triggers)
    summary = summarize(payload, y, sample_rate, events, triggers, offset_ms)

    args.out_dir.mkdir(parents=True, exist_ok=True)
    write_csv(args.out_dir / "t0052_saved_json_events.csv", events)
    write_csv(args.out_dir / "t0052_offline_full_wav_triggers.csv", triggers)
    write_csv(args.out_dir / "t0052_top_racket_probability_rows.csv", top_rows_by_racket_probability(events, triggers))
    (args.out_dir / "t0052_summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    write_report(args.out_dir / "t0052_report.md", summary)

    print(f"expected={EXPECTED_COUNT} app={REPORTED_APP_COUNT}")
    print(f"json candidates={summary['json_native_candidates']} counted={summary['json_counted']}")
    print(f"json labels={summary['json_model_label_counts']}")
    print(f"json rejects={summary['json_reject_reason_counts']}")
    print(f"offline triggers={summary['offline_gate_trigger_rows']} counted={summary['offline_counted_current_app_config']}")
    print(f"wrote {args.out_dir / 't0052_report.md'}")


if __name__ == "__main__":
    main()
