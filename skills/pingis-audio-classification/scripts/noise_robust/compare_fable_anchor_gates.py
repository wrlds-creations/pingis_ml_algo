"""Compare app-style audio anchors with the Fable/NR adaptive onset gate.

This is a diagnostic script for videos or wav files where the app finds too
few or too many audio anchors before Fable classification. It reproduces the
legacy `findAudioPeaks` RMS anchor picker and compares it with
`nr_features.simulate_gate`, the offline replica of the native bandpass gate.

Examples:
  python skills/pingis-audio-classification/scripts/noise_robust/compare_fable_anchor_gates.py \
      data/video/raw/diag_0611/media/video_stroke_session_2026-06-11_001.mp4
  python skills/pingis-audio-classification/scripts/noise_robust/compare_fable_anchor_gates.py \
      session.wav --mode bandpass --retrigger-ms 350 --match-tolerance-ms 140
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import Any

import numpy as np

SCRIPT_DIR = Path(__file__).resolve().parent
SCRIPTS_DIR = SCRIPT_DIR.parent
for _path in (str(SCRIPT_DIR), str(SCRIPTS_DIR)):
    if _path not in sys.path:
        sys.path.insert(0, _path)

import nr_features  # noqa: E402
from preprocess_audio import load_audio  # noqa: E402


def percentile(sorted_values: list[float], ratio: float) -> float:
    if not sorted_values:
        return 0.0
    idx = int((len(sorted_values) - 1) * ratio)
    idx = max(0, min(len(sorted_values) - 1, idx))
    return float(sorted_values[idx])


def app_find_audio_peaks(
    y: np.ndarray,
    sr: int,
    *,
    frame_ms: float,
    min_rms_floor: float,
    median_mult: float,
    p75_mult: float,
    max_mult: float,
    local_radius: int,
    anchor_gap_ms: float,
    max_anchors: int,
) -> dict[str, Any]:
    frame_samples = max(32, round(sr * frame_ms / 1000.0))
    frames: list[tuple[int, float]] = []
    for start in range(0, len(y) - frame_samples + 1, frame_samples):
        segment = y[start : start + frame_samples]
        center_ms = round((start + frame_samples / 2) / sr * 1000.0)
        frames.append((center_ms, float(np.sqrt(np.mean(segment**2)))))

    rms_sorted = sorted(rms for _, rms in frames)
    if not rms_sorted:
        return {
            "frames": frames,
            "threshold": 0.0,
            "threshold_terms": {},
            "candidates": [],
            "anchors": [],
            "local_maxima": [],
        }

    median = percentile(rms_sorted, 0.50)
    p75 = percentile(rms_sorted, 0.75)
    max_rms = float(rms_sorted[-1])
    terms = {
        "min_floor": min_rms_floor,
        "median": median * median_mult,
        "p75": p75 * p75_mult,
        "max": max_rms * max_mult,
    }
    threshold = max(terms.values())

    candidates: list[tuple[int, float]] = []
    local_maxima: list[tuple[int, float]] = []
    for idx in range(local_radius, len(frames) - local_radius):
        ts_ms, rms = frames[idx]
        neighbor_offsets = [o for o in range(-local_radius, local_radius + 1) if o != 0]
        is_local_max = all(frames[idx + offset][1] <= rms for offset in neighbor_offsets)
        if not is_local_max:
            continue
        if rms > min_rms_floor:
            local_maxima.append((ts_ms, rms))
        if rms >= threshold:
            candidates.append((ts_ms, rms))

    anchors: list[tuple[int, float]] = []
    for ts_ms, rms in sorted(candidates, key=lambda item: -item[1]):
        if len(anchors) >= max_anchors:
            break
        if all(abs(ts_ms - existing_ts) >= anchor_gap_ms for existing_ts, _ in anchors):
            anchors.append((ts_ms, rms))
    anchors.sort(key=lambda item: item[0])

    local_maxima.sort(key=lambda item: item[1], reverse=True)
    return {
        "frames": frames,
        "threshold": threshold,
        "threshold_terms": terms,
        "candidates": candidates,
        "anchors": anchors,
        "local_maxima": local_maxima,
    }


def nearest_delta(value: float, candidates: list[float]) -> float | None:
    if not candidates:
        return None
    return min(abs(value - candidate) for candidate in candidates)


def unmatched(values: list[float], candidates: list[float], tolerance_ms: float) -> list[float]:
    return [
        value
        for value in values
        if (nearest_delta(value, candidates) is None or nearest_delta(value, candidates) > tolerance_ms)
    ]


def format_times(values: list[float], limit: int) -> str:
    if not values:
        return "-"
    head = ", ".join(f"{value:.0f}" for value in values[:limit])
    suffix = "" if len(values) <= limit else f", ... (+{len(values) - limit})"
    return head + suffix


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("input", type=Path, help="Video or audio file.")
    parser.add_argument("--mode", choices=["bandpass", "broadband"], default="bandpass")
    parser.add_argument("--onset-ratio", type=float, default=1.5)
    parser.add_argument("--retrigger-ms", type=float, default=350.0)
    parser.add_argument("--abs-min-rms", type=float, default=None)
    parser.add_argument("--spectral-gate", action="store_true")
    parser.add_argument("--frame-ms", type=float, default=10.0)
    parser.add_argument("--anchor-gap-ms", type=float, default=350.0)
    parser.add_argument("--match-tolerance-ms", type=float, default=140.0)
    parser.add_argument("--max-anchors", type=int, default=220)
    parser.add_argument("--show-limit", type=int, default=12)
    args = parser.parse_args()

    if not args.input.exists():
        raise SystemExit(f"Missing input: {args.input}")

    y, sr = load_audio(str(args.input))
    duration_s = len(y) / sr if sr else 0.0
    print(f"input={args.input}")
    print(f"audio={duration_s:.1f} s @ {sr} Hz")

    app = app_find_audio_peaks(
        y,
        sr,
        frame_ms=args.frame_ms,
        min_rms_floor=0.004,
        median_mult=4.0,
        p75_mult=1.8,
        max_mult=0.16,
        local_radius=2,
        anchor_gap_ms=args.anchor_gap_ms,
        max_anchors=args.max_anchors,
    )
    print(
        "app threshold="
        f"{app['threshold']:.5f} "
        f"(floor={app['threshold_terms']['min_floor']:.5f}, "
        f"median*4={app['threshold_terms']['median']:.5f}, "
        f"p75*1.8={app['threshold_terms']['p75']:.5f}, "
        f"max*0.16={app['threshold_terms']['max']:.5f})"
    )
    print(
        f"app candidates={len(app['candidates'])} | "
        f"app anchors={len(app['anchors'])} (gap={args.anchor_gap_ms:.0f} ms)"
    )

    abs_min_rms = args.abs_min_rms
    if abs_min_rms is None:
        abs_min_rms = 0.0015 if args.mode == "bandpass" else 0.003
    triggers = nr_features.simulate_gate(
        y,
        sr,
        onset_ratio=args.onset_ratio,
        retrigger_ms=args.retrigger_ms,
        abs_min_rms=abs_min_rms,
        mode=args.mode,
        spectral_gate=args.spectral_gate,
    )
    trigger_times = [float(t["onset_ms"]) for t in triggers if t.get("passed_spectral", True)]
    app_times = [float(ts) for ts, _ in app["anchors"]]
    print(
        f"nr gate triggers={len(trigger_times)} "
        f"(mode={args.mode}, ratio={args.onset_ratio}, retrigger={args.retrigger_ms:.0f} ms, "
        f"abs_min={abs_min_rms:.4f}, spectral_gate={args.spectral_gate})"
    )

    app_only = unmatched(app_times, trigger_times, args.match_tolerance_ms)
    nr_only = unmatched(trigger_times, app_times, args.match_tolerance_ms)
    print(f"match tolerance={args.match_tolerance_ms:.0f} ms")
    print(f"app anchors without nr trigger: {len(app_only)} | {format_times(app_only, args.show_limit)}")
    print(f"nr triggers without app anchor: {len(nr_only)} | {format_times(nr_only, args.show_limit)}")

    local_maxima = app["local_maxima"]
    half_threshold = app["threshold"] * 0.5
    print(f"local maxima > 0.004: {len(local_maxima)}")
    print(f"local maxima >= 0.5*app_threshold: {sum(1 for _, rms in local_maxima if rms >= half_threshold)}")
    if local_maxima:
        idx = min(79, len(local_maxima) - 1)
        print(f"80th strongest local max RMS: {local_maxima[idx][1]:.5f}")


if __name__ == "__main__":
    main()
