"""Summarize Fable live-debug count behavior from app JSON dumps.

Use this when a live run over-counts, under-counts, or feels slow. The script
does not retrain or replay the model; it inspects exactly what the app saved:
counted events, reject reasons, native RMS, background mode, confidence, and
JS feature/predict timings.

Examples:
  python skills/pingis-audio-classification/scripts/noise_robust/analyze_fable_live_debug_counts.py \
      --dump-dir data/audio/raw/fable_live_debug2
  python skills/pingis-audio-classification/scripts/noise_robust/analyze_fable_live_debug_counts.py \
      --dump-dir data/audio/raw/fable_live_debug2 --glob "fable_live_session_2026-06-11*.json"
"""

from __future__ import annotations

import argparse
import json
from collections import Counter
from pathlib import Path
from typing import Any

import numpy as np

DEFAULT_DUMP_DIR = Path("data/audio/raw/fable_live_debug2")


def event_time_ms(event: dict[str, Any]) -> float | None:
    for key in ("native_onset_time_ms", "onset_time_ms", "timestamp_ms", "time_ms"):
        value = event.get(key)
        if value is not None:
            return float(value)
    return None


def event_rms(event: dict[str, Any]) -> float:
    for key in ("native_rms", "rms", "frame_rms"):
        value = event.get(key)
        if value is not None:
            return float(value)
    return 0.0


def reject_reason(event: dict[str, Any]) -> str:
    for key in ("reject_reason", "native_reject_reason", "drop_reason"):
        value = event.get(key)
        if value:
            return str(value)
    if event.get("model_label") and event.get("model_label") != "racket_bounce":
        return f"model:{event.get('model_label')}"
    return "unknown"


def latency_ms(event: dict[str, Any]) -> float | None:
    feature = event.get("feature_ms")
    predict = event.get("predict_ms")
    if feature is None and predict is None:
        return None
    return float(feature or 0.0) + float(predict or 0.0)


def load_session(path: Path) -> list[dict[str, Any]]:
    data = json.loads(path.read_text(encoding="utf-8"))
    events = data.get("events")
    if not isinstance(events, list):
        raise ValueError(f"{path} has no list field named 'events'")
    return [e for e in events if isinstance(e, dict)]


def resolve_dump_dir(path: Path) -> Path:
    nested = path / "fable_live_debug"
    if nested.exists() and any(nested.glob("fable_live_session_*.json")):
        return nested
    return path


def typical_gap_ms(times_ms: list[float], min_gap_ms: float, max_gap_ms: float) -> float:
    if len(times_ms) < 2:
        return 0.0
    gaps = np.diff(np.array(sorted(times_ms), dtype=float))
    plausible = gaps[(gaps >= min_gap_ms) & (gaps <= max_gap_ms)]
    if len(plausible) == 0:
        return 0.0
    return float(np.median(plausible))


def percentile_text(values: list[float]) -> str:
    if not values:
        return "-"
    arr = np.array(values, dtype=float)
    return (
        f"p50={np.percentile(arr, 50):.0f} ms, "
        f"p95={np.percentile(arr, 95):.0f} ms, "
        f"max={np.max(arr):.0f} ms"
    )


def analyze_file(path: Path, args: argparse.Namespace) -> dict[str, Any]:
    events = load_session(path)
    counted = [e for e in events if e.get("counted") and event_time_ms(e) is not None]
    counted.sort(key=lambda e: float(event_time_ms(e) or 0.0))
    counted_times = [float(event_time_ms(e) or 0.0) for e in counted]
    gaps = np.diff(np.array(counted_times, dtype=float)) if len(counted_times) >= 2 else np.array([])
    rhythm_gap = typical_gap_ms(counted_times, args.min_gap_ms, args.max_gap_ms)

    suspected_duplicates: list[tuple[float, dict[str, Any], dict[str, Any]]] = []
    if rhythm_gap:
        for idx, gap in enumerate(gaps):
            if gap < args.duplicate_ratio * rhythm_gap:
                suspected_duplicates.append((float(gap), counted[idx], counted[idx + 1]))

    rejects = Counter(reject_reason(e) for e in events if not e.get("counted"))
    labels = Counter(str(e.get("model_label")) for e in events if e.get("model_label"))
    latencies = [value for e in events if (value := latency_ms(e)) is not None]

    summary = {
        "path": path,
        "events": len(events),
        "counted": len(counted),
        "typical_gap_ms": rhythm_gap,
        "suspected_duplicates": len(suspected_duplicates),
        "rejects": rejects,
        "labels": labels,
        "latencies": latencies,
    }

    print("=" * 78)
    print(
        f"{path.name} | events={len(events)} | counted={len(counted)} | "
        f"typical_gap={rhythm_gap:.0f} ms | suspected_duplicates={len(suspected_duplicates)}"
    )
    if labels:
        print(f"  labels: {dict(labels)}")
    if rejects:
        print(f"  rejects: {dict(rejects)}")
    if latencies:
        print(f"  JS feature+predict: {percentile_text(latencies)}")

    for gap, previous, current in suspected_duplicates:
        prev_rms = event_rms(previous)
        curr_rms = event_rms(current)
        ratio = curr_rms / max(prev_rms, 1e-9)
        print(
            "  DUP? "
            f"gap={gap:.0f} ms (rhythm={rhythm_gap:.0f}) "
            f"rms={prev_rms:.4f}->{curr_rms:.4f} ratio={ratio:.2f} "
            f"conf={float(previous.get('model_confidence') or 0.0):.2f}->"
            f"{float(current.get('model_confidence') or 0.0):.2f} "
            f"bg={current.get('bg_mode') or current.get('background_mode') or '-'}"
        )
    return summary


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dump-dir", type=Path, default=DEFAULT_DUMP_DIR)
    parser.add_argument("--glob", default="fable_live_session_*.json")
    parser.add_argument("--duplicate-ratio", type=float, default=0.60)
    parser.add_argument("--min-gap-ms", type=float, default=200.0)
    parser.add_argument("--max-gap-ms", type=float, default=1500.0)
    args = parser.parse_args()

    dump_dir = resolve_dump_dir(args.dump_dir)
    paths = sorted(dump_dir.glob(args.glob))
    if not paths:
        raise SystemExit(f"No files matched {dump_dir / args.glob}")

    summaries = [analyze_file(path, args) for path in paths]
    total_events = sum(int(s["events"]) for s in summaries)
    total_counted = sum(int(s["counted"]) for s in summaries)
    total_duplicates = sum(int(s["suspected_duplicates"]) for s in summaries)
    all_latencies = [v for s in summaries for v in s["latencies"]]

    print("=" * 78)
    print(
        f"TOTAL | sessions={len(summaries)} | events={total_events} | "
        f"counted={total_counted} | suspected_duplicates={total_duplicates}"
    )
    if all_latencies:
        print(f"TOTAL JS feature+predict: {percentile_text(all_latencies)}")


if __name__ == "__main__":
    main()
