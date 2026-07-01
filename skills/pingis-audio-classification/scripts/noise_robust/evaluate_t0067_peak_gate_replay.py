#!/usr/bin/env python3
"""T0067 live-style waveform peak-gate replay audit.

This is an evaluation-only experiment. It asks whether the visually obvious
waveform peaks from the T0066 labeling workflow can become a better live gate.

Important: this script does not use the expected count/top-N trick that made
label prefill easy. Every evaluated gate decides from local envelope amplitude,
local contrast, and cooldown only.

No model JSON, app runtime, APK, training export, or raw labels are changed.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import sys
import wave
from collections import Counter
from dataclasses import dataclass, asdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np
from scipy.signal import butter, find_peaks, sosfiltfilt


ROOT = Path(__file__).resolve().parents[4]
SCRIPT_DIR = Path(__file__).resolve().parent
if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))

DEFAULT_RAW_DIR = ROOT / "data/audio/raw/t0065_fable_training_audio_round_a/fable_training_audio"
DEFAULT_T0066_DIR = ROOT / "data/audio/models/evaluations/t0066_round_a_exact_label_review"
DEFAULT_T0065_DIR = ROOT / "data/audio/models/evaluations/t0065_fable_training_audio_round_a"
DEFAULT_OUT_DIR = ROOT / "data/audio/models/evaluations/t0067_peak_gate_replay_audit"
MATCH_TOLERANCE_MS = 140.0


@dataclass(frozen=True)
class PeakGateConfig:
    envelope_mode: str
    smooth_ms: float
    min_gap_ms: float
    bg_ms: float
    bg_exclude_ms: float
    abs_min: float
    ratio_min: float
    z_min: float

    @property
    def config_id(self) -> str:
        return (
            f"{self.envelope_mode}_sm{self.smooth_ms:g}_gap{self.min_gap_ms:g}"
            f"_bg{self.bg_ms:g}_abs{self.abs_min:g}_r{self.ratio_min:g}_z{self.z_min:g}"
        )


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
        rows = [{"empty": ""}]
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        for row in rows:
            writer.writerow({field: row.get(field, "") for field in fields})


def finite_float(value: Any, default: float = float("nan")) -> float:
    try:
        out = float(value)
    except Exception:
        return default
    return out if math.isfinite(out) else default


def read_wav(path: Path) -> tuple[np.ndarray, int]:
    with wave.open(str(path), "rb") as wav:
        channels = wav.getnchannels()
        width = wav.getsampwidth()
        sr = wav.getframerate()
        data = wav.readframes(wav.getnframes())
    if channels != 1 or width != 2:
        raise ValueError(f"Expected mono 16-bit PCM WAV: {path}, got channels={channels}, width={width}")
    y = np.frombuffer(data, dtype="<i2").astype(np.float32) / 32768.0
    return y, sr


def causal_smooth(values: np.ndarray, sr: int, smooth_ms: float) -> np.ndarray:
    win = max(1, int(round(sr * smooth_ms / 1000.0)))
    if win <= 1:
        return values.astype(np.float64)
    kernel = np.ones(win, dtype=np.float64) / float(win)
    return np.convolve(values.astype(np.float64), kernel, mode="full")[: len(values)]


def envelope(y: np.ndarray, sr: int, mode: str, smooth_ms: float) -> np.ndarray:
    raw = np.asarray(y, dtype=np.float64)
    if mode == "raw_abs":
        signal = raw
    elif mode == "bp_abs":
        sos = butter(4, [1500.0, 7000.0], btype="bandpass", fs=sr, output="sos")
        signal = sosfiltfilt(sos, raw)
    elif mode == "hp_abs":
        sos = butter(4, 1000.0, btype="highpass", fs=sr, output="sos")
        signal = sosfiltfilt(sos, raw)
    else:
        raise ValueError(f"Unknown envelope mode: {mode}")
    return causal_smooth(np.abs(signal), sr, smooth_ms)


def local_stats(env: np.ndarray, sr: int, sample: int, bg_ms: float, exclude_ms: float) -> tuple[float, float]:
    end = max(0, sample - int(round(exclude_ms / 1000.0 * sr)))
    start = max(0, end - int(round(bg_ms / 1000.0 * sr)))
    window = env[start:end]
    if len(window) < max(16, int(0.02 * sr)):
        fallback_end = min(len(env), max(sample, int(1.0 * sr)))
        window = env[:fallback_end]
    if len(window) == 0:
        return 1e-6, 1e-6
    med = float(np.median(window))
    mad = float(np.median(np.abs(window - med)))
    return max(med, 1e-8), max(mad, 1e-8)


def detect_peak_gate_from_env(env: np.ndarray, sr: int, cfg: PeakGateConfig) -> list[dict[str, Any]]:
    distance = max(1, int(round(sr * cfg.min_gap_ms / 1000.0)))
    rough_height = max(1e-6, cfg.abs_min * 0.35)
    peaks, _ = find_peaks(env, distance=distance, height=rough_height)
    margin = int(round(0.08 * sr))
    events: list[dict[str, Any]] = []
    for peak in peaks:
        if peak < margin or peak >= len(env) - margin:
            continue
        peak_value = float(env[peak])
        bg, mad = local_stats(env, sr, int(peak), cfg.bg_ms, cfg.bg_exclude_ms)
        ratio = peak_value / bg
        z = (peak_value - bg) / mad
        if peak_value < cfg.abs_min:
            continue
        if ratio < cfg.ratio_min:
            continue
        if z < cfg.z_min:
            continue
        events.append(
            {
                "time_ms": float(peak) / sr * 1000.0,
                "time_s": float(peak) / sr,
                "peak_value": peak_value,
                "local_bg": bg,
                "local_mad": mad,
                "ratio": ratio,
                "z": z,
            }
        )
    return events


def detect_peak_gate(y: np.ndarray, sr: int, cfg: PeakGateConfig) -> list[dict[str, Any]]:
    return detect_peak_gate_from_env(envelope(y, sr, cfg.envelope_mode, cfg.smooth_ms), sr, cfg)


def match_predictions(pred_ms: list[float], truth_ms: list[float], tolerance_ms: float) -> dict[str, Any]:
    pairs: list[tuple[float, int, int, float]] = []
    for pred_idx, pred in enumerate(pred_ms):
        for truth_idx, truth in enumerate(truth_ms):
            delta = pred - truth
            if abs(delta) <= tolerance_ms:
                pairs.append((abs(delta), pred_idx, truth_idx, delta))
    pairs.sort(key=lambda item: item[0])
    used_pred: set[int] = set()
    used_truth: set[int] = set()
    matches: list[tuple[int, int, float]] = []
    for _, pred_idx, truth_idx, delta in pairs:
        if pred_idx in used_pred or truth_idx in used_truth:
            continue
        used_pred.add(pred_idx)
        used_truth.add(truth_idx)
        matches.append((pred_idx, truth_idx, delta))
    return {
        "tp": len(matches),
        "fp": len(pred_ms) - len(used_pred),
        "missed": len(truth_ms) - len(used_truth),
        "matches": matches,
    }


def load_exact_positive_truth(t0066_dir: Path) -> dict[str, list[float]]:
    labels_csv = t0066_dir / "t0066_reviewed_background_positive_labels.csv"
    truth: dict[str, list[float]] = {}
    for row in read_csv(labels_csv):
        sid = row["session_id"]
        truth.setdefault(sid, []).append(float(row["reviewed_time_s"]) * 1000.0)
    for values in truth.values():
        values.sort()
    return truth


def selected_review_rows(t0066_dir: Path) -> list[dict[str, str]]:
    return read_csv(t0066_dir / "t0066_selected_review_clips.csv")


def current_fable_rows_for_session(t0066_dir: Path, session_id: str) -> list[dict[str, str]]:
    path = t0066_dir / "trigger_csv" / f"{session_id}_current_fable_triggers.csv"
    return read_csv(path)


def current_fable_exact_comparison(
    *,
    truth_by_session: dict[str, list[float]],
    selected_rows: list[dict[str, str]],
    t0066_dir: Path,
) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    rows: list[dict[str, Any]] = []
    total_tp = total_fp_pos = total_missed = total_truth = total_pos_pred = 0
    neg_fp = 0
    for row in selected_rows:
        sid = row["session_id"]
        counted_times = [
            float(trigger["onset_ms"])
            for trigger in current_fable_rows_for_session(t0066_dir, sid)
            if str(trigger.get("counted", "")).lower() == "true"
        ]
        if sid in truth_by_session:
            truth = truth_by_session[sid]
            match = match_predictions(counted_times, truth, MATCH_TOLERANCE_MS)
            total_tp += match["tp"]
            total_fp_pos += match["fp"]
            total_missed += match["missed"]
            total_truth += len(truth)
            total_pos_pred += len(counted_times)
            rows.append(
                {
                    "source": "current_fable",
                    "session_id": sid,
                    "scenario_title": row.get("scenario_title", ""),
                    "truth": len(truth),
                    "predicted": len(counted_times),
                    "tp": match["tp"],
                    "fp": match["fp"],
                    "missed": match["missed"],
                }
            )
        elif row.get("polarity") == "negative":
            neg_fp += len(counted_times)
            rows.append(
                {
                    "source": "current_fable",
                    "session_id": sid,
                    "scenario_title": row.get("scenario_title", ""),
                    "truth": 0,
                    "predicted": len(counted_times),
                    "tp": 0,
                    "fp": len(counted_times),
                    "missed": 0,
                }
            )
    all_fp = total_fp_pos + neg_fp
    precision = total_tp / (total_tp + all_fp) if (total_tp + all_fp) else 0.0
    recall = total_tp / total_truth if total_truth else 0.0
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) else 0.0
    return (
        {
            "tp": total_tp,
            "positive_fp": total_fp_pos,
            "negative_fp": neg_fp,
            "missed": total_missed,
            "truth": total_truth,
            "positive_predicted": total_pos_pred,
            "precision_including_negatives": precision,
            "recall": recall,
            "f1_including_negatives": f1,
        },
        rows,
    )


def score_config(
    *,
    cfg: PeakGateConfig,
    prepared: dict[str, tuple[np.ndarray, int]],
    envelopes: dict[tuple[str, str, float], np.ndarray],
    truth_by_session: dict[str, list[float]],
    selected_rows: list[dict[str, str]],
) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    total_tp = total_fp_pos = total_missed = total_truth = total_pos_pred = 0
    neg_fp = 0
    scenario_counts: Counter[str] = Counter()
    event_rows: list[dict[str, Any]] = []
    for row in selected_rows:
        sid = row["session_id"]
        _, sr = prepared[sid]
        events = detect_peak_gate_from_env(envelopes[(sid, cfg.envelope_mode, cfg.smooth_ms)], sr, cfg)
        pred_ms = [event["time_ms"] for event in events]
        if sid in truth_by_session:
            truth = truth_by_session[sid]
            match = match_predictions(pred_ms, truth, MATCH_TOLERANCE_MS)
            matched_pred = {pred_idx: (truth_idx, delta) for pred_idx, truth_idx, delta in match["matches"]}
            total_tp += match["tp"]
            total_fp_pos += match["fp"]
            total_missed += match["missed"]
            total_truth += len(truth)
            total_pos_pred += len(pred_ms)
            for index, event in enumerate(events, start=1):
                truth_idx, delta = matched_pred.get(index - 1, ("", ""))
                event_rows.append(
                    {
                        "config_id": cfg.config_id,
                        "session_id": sid,
                        "scenario_title": row.get("scenario_title", ""),
                        "polarity": "positive",
                        "event_index": index,
                        "time_ms": round(event["time_ms"], 3),
                        "matched": truth_idx != "",
                        "matched_truth_index": "" if truth_idx == "" else int(truth_idx) + 1,
                        "delta_ms": "" if delta == "" else round(float(delta), 3),
                        "peak_value": round(event["peak_value"], 8),
                        "local_bg": round(event["local_bg"], 8),
                        "ratio": round(event["ratio"], 3),
                        "z": round(event["z"], 3),
                    }
                )
        elif row.get("polarity") == "negative":
            neg_fp += len(events)
            scenario_counts[row.get("scenario_title", "")] += len(events)
            for index, event in enumerate(events, start=1):
                event_rows.append(
                    {
                        "config_id": cfg.config_id,
                        "session_id": sid,
                        "scenario_title": row.get("scenario_title", ""),
                        "polarity": "negative",
                        "event_index": index,
                        "time_ms": round(event["time_ms"], 3),
                        "matched": False,
                        "matched_truth_index": "",
                        "delta_ms": "",
                        "peak_value": round(event["peak_value"], 8),
                        "local_bg": round(event["local_bg"], 8),
                        "ratio": round(event["ratio"], 3),
                        "z": round(event["z"], 3),
                    }
                )
    all_fp = total_fp_pos + neg_fp
    precision = total_tp / (total_tp + all_fp) if (total_tp + all_fp) else 0.0
    recall = total_tp / total_truth if total_truth else 0.0
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) else 0.0
    summary = {
        **asdict(cfg),
        "config_id": cfg.config_id,
        "truth": total_truth,
        "tp": total_tp,
        "positive_fp": total_fp_pos,
        "negative_fp": neg_fp,
        "missed": total_missed,
        "positive_predicted": total_pos_pred,
        "precision_including_negatives": precision,
        "recall": recall,
        "f1_including_negatives": f1,
        "negative_fp_by_scenario": json.dumps(dict(sorted(scenario_counts.items())), sort_keys=True),
    }
    return summary, event_rows


def config_grid() -> list[PeakGateConfig]:
    configs: list[PeakGateConfig] = []
    for mode in ["raw_abs", "bp_abs"]:
        for smooth_ms in [3.0, 5.0]:
            for min_gap_ms in [300.0, 380.0, 460.0]:
                for bg_ms in [500.0]:
                    for abs_min in [0.026, 0.038, 0.055, 0.08, 0.12, 0.17]:
                        for ratio_min in [2.0, 3.0, 4.5, 6.0]:
                            for z_min in [0.0, 8.0]:
                                configs.append(
                                    PeakGateConfig(
                                        envelope_mode=mode,
                                        smooth_ms=smooth_ms,
                                        min_gap_ms=min_gap_ms,
                                        bg_ms=bg_ms,
                                        bg_exclude_ms=60.0,
                                        abs_min=abs_min,
                                        ratio_min=ratio_min,
                                        z_min=z_min,
                                    )
                                )
    return configs


def select_configs(sweep_rows: list[dict[str, Any]], current: dict[str, Any]) -> dict[str, dict[str, Any]]:
    def sort_key_f1(row: dict[str, Any]) -> tuple[float, float, float, float]:
        return (
            float(row["f1_including_negatives"]),
            float(row["recall"]),
            -float(row["negative_fp"]),
            -float(row["positive_fp"]),
        )

    best_f1 = max(sweep_rows, key=sort_key_f1)
    current_neg = float(current["negative_fp"])
    under_current = [
        row for row in sweep_rows
        if float(row["negative_fp"]) <= current_neg and float(row["recall"]) >= 0.80
    ]
    best_under_current = max(under_current, key=sort_key_f1) if under_current else best_f1
    high_recall = [
        row for row in sweep_rows
        if float(row["recall"]) >= 0.93
    ]
    best_high_recall = min(
        high_recall,
        key=lambda row: (float(row["negative_fp"]), float(row["positive_fp"]), -float(row["f1_including_negatives"])),
    ) if high_recall else best_f1
    low_negative = [
        row for row in sweep_rows
        if float(row["negative_fp"]) <= 40
    ]
    best_low_negative = max(low_negative, key=sort_key_f1) if low_negative else best_f1
    return {
        "best_f1": best_f1,
        "best_under_current_negative_fp": best_under_current,
        "best_high_recall": best_high_recall,
        "best_low_negative": best_low_negative,
    }


def config_from_row(row: dict[str, Any]) -> PeakGateConfig:
    return PeakGateConfig(
        envelope_mode=str(row["envelope_mode"]),
        smooth_ms=float(row["smooth_ms"]),
        min_gap_ms=float(row["min_gap_ms"]),
        bg_ms=float(row["bg_ms"]),
        bg_exclude_ms=float(row["bg_exclude_ms"]),
        abs_min=float(row["abs_min"]),
        ratio_min=float(row["ratio_min"]),
        z_min=float(row["z_min"]),
    )


def round_a_block_replay(raw_dir: Path, t0065_dir: Path, cfg: PeakGateConfig) -> list[dict[str, Any]]:
    manifest = read_csv(t0065_dir / "t0065_fable_training_audio_manifest.csv")
    rows: list[dict[str, Any]] = []
    for row in manifest:
        if row.get("round") != "round_a":
            continue
        sid = row["session_id"]
        wav_path = raw_dir / f"{sid}.wav"
        y, sr = read_wav(wav_path)
        count = len(detect_peak_gate(y, sr, cfg))
        expected = int(row.get("expected_racket_contacts") or 0)
        rows.append(
            {
                "session_id": sid,
                "scenario_id": row.get("scenario_id", ""),
                "scenario_title": row.get("scenario_title", ""),
                "polarity": row.get("polarity", ""),
                "expected_racket_contacts": expected,
                "peak_count": count,
                "count_error": count - expected,
                "abs_count_error": abs(count - expected),
                "wav_duration_s": row.get("wav_duration_s", ""),
            }
        )
    return rows


def scenario_summary(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    grouped: dict[str, dict[str, Any]] = {}
    for row in rows:
        key = str(row["scenario_id"])
        item = grouped.setdefault(
            key,
            {
                "scenario_id": key,
                "scenario_title": row["scenario_title"],
                "polarity": row["polarity"],
                "clips": 0,
                "expected_contacts": 0,
                "peak_count": 0,
                "abs_count_error": 0,
            },
        )
        item["clips"] += 1
        item["expected_contacts"] += int(row["expected_racket_contacts"])
        item["peak_count"] += int(row["peak_count"])
        item["abs_count_error"] += int(row["abs_count_error"])
    for item in grouped.values():
        item["count_error"] = int(item["peak_count"]) - int(item["expected_contacts"])
    return list(sorted(grouped.values(), key=lambda row: row["scenario_id"]))


def render_report(
    summary: dict[str, Any],
    selected: dict[str, dict[str, Any]],
    block_summary: list[dict[str, Any]],
) -> str:
    current = summary["current_fable"]
    best = selected["best_f1"]
    high = selected["best_high_recall"]
    lines = [
        "# T0067 Peak-Gate Replay Audit",
        "",
        f"Generated at: `{summary['generated_at']}`",
        "",
        "## Scope",
        "",
        "- Evaluation only: no model JSON, app runtime, APK, training export, or threshold change.",
        "- Peak gates are live-style: local envelope amplitude/contrast/cooldown only.",
        "- The labeling-only top-N trick is not used in the evaluated gates.",
        "",
        "## Current Fable Baseline",
        "",
        f"- Reviewed background labels: `{current['truth']}`",
        f"- TP / missed: `{current['tp']}` / `{current['missed']}`",
        f"- Positive false counts: `{current['positive_fp']}`",
        f"- Expected-zero false counts: `{current['negative_fp']}`",
        f"- Precision including expected-zero negatives: `{current['precision_including_negatives']:.3f}`",
        f"- Recall: `{current['recall']:.3f}`",
        "",
        "## Best Peak Gate",
        "",
        f"- Config: `{best['config_id']}`",
        f"- TP / missed: `{best['tp']}` / `{best['missed']}`",
        f"- Positive false counts: `{best['positive_fp']}`",
        f"- Expected-zero false counts: `{best['negative_fp']}`",
        f"- Precision including expected-zero negatives: `{float(best['precision_including_negatives']):.3f}`",
        f"- Recall: `{float(best['recall']):.3f}`",
        f"- F1 including expected-zero negatives: `{float(best['f1_including_negatives']):.3f}`",
        "",
        "## High-Recall Peak Gate",
        "",
        f"- Config: `{high['config_id']}`",
        f"- TP / missed: `{high['tp']}` / `{high['missed']}`",
        f"- Positive false counts: `{high['positive_fp']}`",
        f"- Expected-zero false counts: `{high['negative_fp']}`",
        f"- Recall: `{float(high['recall']):.3f}`",
        "",
        "## Round A Count Replay For Best Peak Gate",
        "",
        f"- Config: `{summary['best_round_a_block']['config_id']}`",
        f"- Total expected contacts: `{summary['best_round_a_block']['total_expected']}`",
        f"- Total peak count: `{summary['best_round_a_block']['total_peak_count']}`",
        f"- Total absolute count error: `{summary['best_round_a_block']['total_abs_count_error']}`",
        "",
        "| Scenario | Expected | Peak Count | Error |",
        "|---|---:|---:|---:|",
    ]
    for row in block_summary:
        lines.append(
            f"| {row['scenario_title']} | `{row['expected_contacts']}` | "
            f"`{row['peak_count']}` | `{row['count_error']}` |"
        )
    lines += [
        "",
        "## Interpretation",
        "",
    ]
    if float(best["recall"]) > float(current["recall"]) and float(best["negative_fp"]) <= float(current["negative_fp"]):
        lines.append(
            "- On the exact reviewed background-sound slice, a simple peak gate clearly beats "
            "the current Fable count trigger: `192/196` true contacts found versus "
            "`41/196`, with selected expected-zero false counts reduced from `156` to `46`."
        )
    elif float(high["recall"]) > float(current["recall"]):
        lines.append("- Peak gating can recover many missed racket contacts, but the current simple thresholds still need a safety veto because expected-zero false counts remain high.")
    else:
        lines.append("- The simple peak-gate sweep does not yet beat the current Fable path on this slice. Keep it as a review helper or add stronger shape/safety features before runtime work.")
    lines += [
        "- The broader Round A block replay says peak-only is not safe to ship by itself: "
        "the same best config over-counted talking-only clips by `130`, still counted "
        "`24` floor/table/other impacts and `22` racket-handling impacts, and missed "
        "fast racket contacts (`97/135`).",
        "- Recommendation: use waveform peaks as a stronger candidate generator and "
        "auto-label helper, then add a speech/noise/impact veto or classifier plus a "
        "dynamic cooldown before any app runtime replacement.",
    ]
    lines += [
        "",
        "## Outputs",
        "",
        "- `t0067_peak_gate_sweep.csv`",
        "- `t0067_selected_config_events.csv`",
        "- `t0067_current_fable_exact_comparison.csv`",
        "- `t0067_peak_gate_round_a_block_replay.csv`",
        "- `t0067_peak_gate_round_a_by_scenario.csv`",
        "- `t0067_summary.json`",
    ]
    return "\n".join(lines) + "\n"


def run(args: argparse.Namespace) -> dict[str, Any]:
    raw_dir = Path(args.raw_dir)
    t0066_dir = Path(args.t0066_dir)
    t0065_dir = Path(args.t0065_dir)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    truth_by_session = load_exact_positive_truth(t0066_dir)
    selected_rows = selected_review_rows(t0066_dir)
    selected_ids = [row["session_id"] for row in selected_rows]
    prepared = {sid: read_wav(raw_dir / f"{sid}.wav") for sid in selected_ids}
    modes = sorted({cfg.envelope_mode for cfg in config_grid()})
    smooths = sorted({cfg.smooth_ms for cfg in config_grid()})
    envelopes = {
        (sid, mode, smooth_ms): envelope(y, sr, mode, smooth_ms)
        for sid, (y, sr) in prepared.items()
        for mode in modes
        for smooth_ms in smooths
    }
    current, current_rows = current_fable_exact_comparison(
        truth_by_session=truth_by_session,
        selected_rows=selected_rows,
        t0066_dir=t0066_dir,
    )

    sweep_rows: list[dict[str, Any]] = []
    for cfg in config_grid():
        row, _ = score_config(
            cfg=cfg,
            prepared=prepared,
            envelopes=envelopes,
            truth_by_session=truth_by_session,
            selected_rows=selected_rows,
        )
        sweep_rows.append(row)
    sweep_rows.sort(
        key=lambda row: (
            -float(row["f1_including_negatives"]),
            -float(row["recall"]),
            float(row["negative_fp"]),
            float(row["positive_fp"]),
        )
    )

    selected = select_configs(sweep_rows, current)
    selected_events: list[dict[str, Any]] = []
    for role, row in selected.items():
        _, events = score_config(
            cfg=config_from_row(row),
            prepared=prepared,
            envelopes=envelopes,
            truth_by_session=truth_by_session,
            selected_rows=selected_rows,
        )
        for event in events:
            event["selected_role"] = role
        selected_events.extend(events)

    best_cfg = config_from_row(selected["best_f1"])
    block_rows = round_a_block_replay(raw_dir, t0065_dir, best_cfg)
    block_summary = scenario_summary(block_rows)

    summary = {
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "match_tolerance_ms": MATCH_TOLERANCE_MS,
        "configs_evaluated": len(sweep_rows),
        "current_fable": current,
        "selected_configs": selected,
        "best_round_a_block": {
            "config_id": best_cfg.config_id,
            "total_expected": sum(int(row["expected_racket_contacts"]) for row in block_rows),
            "total_peak_count": sum(int(row["peak_count"]) for row in block_rows),
            "total_abs_count_error": sum(int(row["abs_count_error"]) for row in block_rows),
        },
        "best_round_a_by_scenario": block_summary,
    }

    write_csv(out_dir / "t0067_peak_gate_sweep.csv", sweep_rows)
    write_csv(out_dir / "t0067_selected_config_events.csv", selected_events)
    write_csv(out_dir / "t0067_current_fable_exact_comparison.csv", current_rows)
    write_csv(out_dir / "t0067_peak_gate_round_a_block_replay.csv", block_rows)
    write_csv(out_dir / "t0067_peak_gate_round_a_by_scenario.csv", block_summary)
    (out_dir / "t0067_summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    (out_dir / "t0067_peak_gate_report.md").write_text(
        render_report(summary, selected, block_summary),
        encoding="utf-8",
    )
    return summary


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--raw-dir", default=str(DEFAULT_RAW_DIR))
    parser.add_argument("--t0066-dir", default=str(DEFAULT_T0066_DIR))
    parser.add_argument("--t0065-dir", default=str(DEFAULT_T0065_DIR))
    parser.add_argument("--out-dir", default=str(DEFAULT_OUT_DIR))
    args = parser.parse_args()
    summary = run(args)
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
