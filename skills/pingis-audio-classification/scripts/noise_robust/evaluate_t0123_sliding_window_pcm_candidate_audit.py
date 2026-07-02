#!/usr/bin/env python3
"""T0123 sliding-window PCM candidate audit.

Evaluation only. This asks whether a buffered/sliding-window PCM candidate
generator can recover reviewed racket contacts missed by the current peak gate.

It does not train, export, install, or change app/runtime behavior.
"""

from __future__ import annotations

import csv
import json
import math
import sys
from collections import defaultdict
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np

ROOT = Path(__file__).resolve().parents[4]
SCRIPT_DIR = Path(__file__).resolve().parent
if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))

from evaluate_t0067_peak_gate_replay import (  # noqa: E402
    PeakGateConfig,
    causal_smooth,
    detect_peak_gate,
    envelope,
    read_wav,
)

OUT_DIR = ROOT / "data/audio/models/evaluations/t0123_sliding_window_pcm_candidate_audit"
T0119_RAW_DIR = ROOT / "data/audio/raw/t0119_calibrated_transient_phone_check/bounce_audio_test_debug"
T0119_REVIEW_LABELS = (
    ROOT
    / "data/audio/models/evaluations/t0119_calibrated_transient_phone_check/review_pages"
    / "bounce_audio_test_session_2026-07-02T17-48-43-807Z_review_labels.json"
)
T0104D_LABELS_CSV = (
    ROOT / "data/audio/models/evaluations/t0104d_t0104b_positive_label_ingest/t0104d_reviewed_positive_labels.csv"
)
T0104_SUMMARY_CSV = (
    ROOT / "data/audio/models/evaluations/t0104_bounce_audio_test_live_validation/t0104_session_summary.csv"
)
T0104_RAW_DIR = ROOT / "data/audio/raw/t0104_bounce_audio_test_live_validation/bounce_audio_test_debug"

MATCH_140_MS = 140.0
MATCH_250_MS = 250.0


@dataclass(frozen=True)
class EvalSession:
    session_id: str
    wav_path: Path
    scenario_id: str
    scenario_title: str
    polarity: str
    expected_count: int
    label_times_ms: tuple[float, ...]
    source: str


@dataclass(frozen=True)
class SlidingConfig:
    config_id: str
    envelope_mode: str
    smooth_ms: float
    stride_ms: float
    search_ms: float
    bg_ms: float
    bg_exclude_ms: float
    min_gap_ms: float
    min_abs: float
    ratio_min: float
    z_min: float


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open("r", newline="", encoding="utf-8-sig") as handle:
        return list(csv.DictReader(handle))


def write_csv(path: Path, rows: list[dict[str, Any]], fields: list[str] | None = None) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if fields is None:
        fields = []
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
        writer = csv.DictWriter(handle, fieldnames=fields, extrasaction="ignore")
        writer.writeheader()
        for row in rows:
            writer.writerow(row)


def write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")


def ff(value: Any, default: float = 0.0) -> float:
    try:
        out = float(value)
    except Exception:
        return default
    return out if math.isfinite(out) else default


def intish(value: Any, default: int = 0) -> int:
    try:
        return int(float(value))
    except Exception:
        return default


def load_sessions() -> list[EvalSession]:
    sessions: list[EvalSession] = []

    if T0119_REVIEW_LABELS.exists():
        payload = json.loads(T0119_REVIEW_LABELS.read_text(encoding="utf-8"))
        label_times = sorted(
            ff(marker.get("time_s")) * 1000.0
            for marker in payload.get("manual_markers", [])
            if str(marker.get("label", "")).lower() in {"racket", "racket_bounce"}
        )
        session_id = str(payload.get("session_id", "bounce_audio_test_session_2026-07-02T17-48-43-807Z"))
        sessions.append(
            EvalSession(
                session_id=session_id,
                wav_path=T0119_RAW_DIR / f"{session_id}.wav",
                scenario_id="t0119_speaking_counting_miss",
                scenario_title="T0119 speaking/counting miss",
                polarity="positive",
                expected_count=intish(payload.get("expected_count"), len(label_times)),
                label_times_ms=tuple(label_times),
                source="t0121_saved_labels",
            )
        )

    if T0104D_LABELS_CSV.exists():
        grouped: dict[str, list[dict[str, str]]] = defaultdict(list)
        for row in read_csv(T0104D_LABELS_CSV):
            grouped[row["session_id"]].append(row)
        for session_id, rows in sorted(grouped.items()):
            label_times = sorted(ff(row.get("reviewed_time_ms")) for row in rows)
            first = rows[0]
            wav_path = T0104_RAW_DIR / f"{session_id}.wav"
            if not wav_path.exists():
                continue
            sessions.append(
                EvalSession(
                    session_id=session_id,
                    wav_path=wav_path,
                    scenario_id=first.get("scenario_id", ""),
                    scenario_title=first.get("scenario_title", ""),
                    polarity="positive",
                    expected_count=intish(first.get("expected_count"), len(label_times)),
                    label_times_ms=tuple(label_times),
                    source="t0104d_reviewed_positive_labels",
                )
            )

    if T0104_SUMMARY_CSV.exists():
        for row in read_csv(T0104_SUMMARY_CSV):
            if row.get("polarity") != "negative":
                continue
            if row.get("include_in_metrics") != "True":
                continue
            session_id = row["session_id"]
            wav_path = T0104_RAW_DIR / f"{session_id}.wav"
            if not wav_path.exists():
                continue
            sessions.append(
                EvalSession(
                    session_id=session_id,
                    wav_path=wav_path,
                    scenario_id=row.get("scenario_id", ""),
                    scenario_title=row.get("scenario_title", ""),
                    polarity="negative",
                    expected_count=0,
                    label_times_ms=(),
                    source="t0104_expected_zero_summary",
                )
            )

    return sessions


def two_sided_local_stats(
    env: np.ndarray,
    sr: int,
    sample: int,
    bg_ms: float,
    exclude_ms: float,
) -> tuple[float, float]:
    half_bg = int(round(sr * bg_ms / 1000.0 / 2.0))
    exclude = int(round(sr * exclude_ms / 1000.0))
    start = max(0, sample - half_bg)
    end = min(len(env), sample + half_bg)
    left = env[start : max(start, sample - exclude)]
    right = env[min(end, sample + exclude) : end]
    window = np.concatenate([left, right]) if len(left) or len(right) else np.array([], dtype=np.float64)
    if len(window) < max(16, int(0.02 * sr)):
        fallback_start = max(0, sample - int(0.5 * sr))
        fallback_end = min(len(env), sample + int(0.5 * sr))
        window = env[fallback_start:fallback_end]
    if len(window) == 0:
        return 1e-8, 1e-8
    med = float(np.median(window))
    mad = float(np.median(np.abs(window - med)))
    return max(med, 1e-8), max(mad, 1e-8)


def nms_by_score(candidates: list[dict[str, Any]], sr: int, min_gap_ms: float) -> list[dict[str, Any]]:
    min_gap_samples = max(1, int(round(sr * min_gap_ms / 1000.0)))
    selected: list[dict[str, Any]] = []
    for cand in sorted(candidates, key=lambda row: (-ff(row.get("score")), ff(row.get("time_ms")))):
        sample = intish(cand.get("sample"))
        if all(abs(sample - intish(prev.get("sample"))) >= min_gap_samples for prev in selected):
            selected.append(cand)
    return sorted(selected, key=lambda row: ff(row.get("time_ms")))


def detect_sliding_window_pcm(y: np.ndarray, sr: int, cfg: SlidingConfig) -> list[dict[str, Any]]:
    env = envelope(y, sr, cfg.envelope_mode, cfg.smooth_ms)
    stride = max(1, int(round(sr * cfg.stride_ms / 1000.0)))
    search = max(1, int(round(sr * cfg.search_ms / 1000.0 / 2.0)))
    margin = max(search + 1, int(round(0.08 * sr)))
    raw_candidates: list[dict[str, Any]] = []

    for center in range(margin, len(env) - margin, stride):
        start = max(0, center - search)
        end = min(len(env), center + search + 1)
        if end <= start:
            continue
        local = env[start:end]
        rel = int(np.argmax(local))
        sample = start + rel
        peak_value = float(env[sample])
        bg, mad = two_sided_local_stats(env, sr, sample, cfg.bg_ms, cfg.bg_exclude_ms)
        ratio = peak_value / bg
        z = (peak_value - bg) / mad
        if peak_value < cfg.min_abs:
            continue
        if ratio < cfg.ratio_min:
            continue
        if z < cfg.z_min:
            continue
        score = z + math.log(max(ratio, 1.0))
        raw_candidates.append(
            {
                "time_ms": float(sample) * 1000.0 / sr,
                "time_s": float(sample) / sr,
                "sample": sample,
                "peak_value": peak_value,
                "local_bg": bg,
                "local_mad": mad,
                "ratio": ratio,
                "z": z,
                "score": score,
                "window_center_ms": float(center) * 1000.0 / sr,
            }
        )
    return nms_by_score(raw_candidates, sr, cfg.min_gap_ms)


def match_candidates(
    label_times_ms: tuple[float, ...],
    candidates: list[dict[str, Any]],
    tolerance_ms: float,
) -> dict[str, Any]:
    candidate_times = [ff(row.get("time_ms")) for row in candidates]
    used: set[int] = set()
    matches: list[dict[str, Any]] = []
    misses: list[float] = []
    for label_idx, label_ms in enumerate(label_times_ms, start=1):
        best_idx = None
        best_abs = float("inf")
        for idx, cand_ms in enumerate(candidate_times):
            if idx in used:
                continue
            delta = cand_ms - label_ms
            abs_delta = abs(delta)
            if abs_delta <= tolerance_ms and abs_delta < best_abs:
                best_idx = idx
                best_abs = abs_delta
        if best_idx is None:
            misses.append(label_ms)
        else:
            used.add(best_idx)
            matches.append(
                {
                    "label_index": label_idx,
                    "label_ms": label_ms,
                    "candidate_index": best_idx + 1,
                    "candidate_ms": candidate_times[best_idx],
                    "delta_ms": candidate_times[best_idx] - label_ms,
                    "abs_delta_ms": best_abs,
                }
            )
    unmatched_candidate_indexes = [idx for idx in range(len(candidates)) if idx not in used]
    return {
        "matched": len(matches),
        "missed": len(misses),
        "matches": matches,
        "missed_label_times_ms": misses,
        "unmatched_candidate_indexes": unmatched_candidate_indexes,
    }


def get_configs() -> list[tuple[str, str, Any]]:
    peak_configs: list[tuple[str, str, PeakGateConfig]] = [
        ("current_peak_abs008", "peak_gate", PeakGateConfig("raw_abs", 3.0, 220.0, 500.0, 60.0, 0.08, 2.0, 0.0)),
        ("soft_peak_abs003", "peak_gate", PeakGateConfig("raw_abs", 3.0, 220.0, 500.0, 60.0, 0.03, 2.0, 0.0)),
        ("soft_peak_hp_abs0015", "peak_gate", PeakGateConfig("hp_abs", 3.0, 220.0, 500.0, 60.0, 0.015, 2.5, 4.0)),
    ]
    sliding_configs: list[tuple[str, str, SlidingConfig]] = [
        (
            "sw_raw_abs030_r2_z4",
            "sliding_window_pcm",
            SlidingConfig("sw_raw_abs030_r2_z4", "raw_abs", 3.0, 20.0, 120.0, 700.0, 80.0, 180.0, 0.030, 2.0, 4.0),
        ),
        (
            "sw_raw_abs020_r3_z6",
            "sliding_window_pcm",
            SlidingConfig("sw_raw_abs020_r3_z6", "raw_abs", 3.0, 20.0, 120.0, 700.0, 80.0, 180.0, 0.020, 3.0, 6.0),
        ),
        (
            "sw_raw_abs015_r4_z8",
            "sliding_window_pcm",
            SlidingConfig("sw_raw_abs015_r4_z8", "raw_abs", 3.0, 20.0, 120.0, 700.0, 80.0, 180.0, 0.015, 4.0, 8.0),
        ),
        (
            "sw_raw_abs020_r5_z12",
            "sliding_window_pcm",
            SlidingConfig("sw_raw_abs020_r5_z12", "raw_abs", 3.0, 20.0, 120.0, 900.0, 100.0, 180.0, 0.020, 5.0, 12.0),
        ),
        (
            "sw_raw_abs025_r6_z16",
            "sliding_window_pcm",
            SlidingConfig("sw_raw_abs025_r6_z16", "raw_abs", 3.0, 20.0, 120.0, 900.0, 100.0, 180.0, 0.025, 6.0, 16.0),
        ),
        (
            "sw_hp_abs010_r3_z8",
            "sliding_window_pcm",
            SlidingConfig("sw_hp_abs010_r3_z8", "hp_abs", 3.0, 20.0, 120.0, 700.0, 80.0, 180.0, 0.010, 3.0, 8.0),
        ),
        (
            "sw_hp_abs006_r4_z10",
            "sliding_window_pcm",
            SlidingConfig("sw_hp_abs006_r4_z10", "hp_abs", 3.0, 20.0, 120.0, 700.0, 80.0, 180.0, 0.006, 4.0, 10.0),
        ),
        (
            "sw_hp_abs004_r5_z12",
            "sliding_window_pcm",
            SlidingConfig("sw_hp_abs004_r5_z12", "hp_abs", 3.0, 20.0, 120.0, 900.0, 100.0, 180.0, 0.004, 5.0, 12.0),
        ),
        (
            "sw_hp_abs006_r6_z18",
            "sliding_window_pcm",
            SlidingConfig("sw_hp_abs006_r6_z18", "hp_abs", 3.0, 20.0, 120.0, 900.0, 100.0, 180.0, 0.006, 6.0, 18.0),
        ),
        (
            "sw_hp_abs008_r8_z24",
            "sliding_window_pcm",
            SlidingConfig("sw_hp_abs008_r8_z24", "hp_abs", 3.0, 20.0, 120.0, 900.0, 100.0, 180.0, 0.008, 8.0, 24.0),
        ),
    ]
    return [*peak_configs, *sliding_configs]


def detect_for_config(y: np.ndarray, sr: int, family: str, cfg: Any) -> list[dict[str, Any]]:
    if family == "peak_gate":
        rows = detect_peak_gate(y, sr, cfg)
        for row in rows:
            row.setdefault("time_ms", ff(row.get("time_s")) * 1000.0)
            row.setdefault("sample", int(round(ff(row.get("time_s")) * sr)))
            row.setdefault("score", ff(row.get("z")) + math.log(max(ff(row.get("ratio")), 1.0)))
        return rows
    if family == "sliding_window_pcm":
        return detect_sliding_window_pcm(y, sr, cfg)
    raise ValueError(f"Unknown config family: {family}")


def pct(part: int | float, total: int | float) -> str:
    return "0.0%" if not total else f"{float(part) / float(total) * 100.0:.1f}%"


def md_table(rows: list[dict[str, Any]], fields: list[str]) -> list[str]:
    lines = ["| " + " | ".join(fields) + " |", "| " + " | ".join("---" for _ in fields) + " |"]
    for row in rows:
        lines.append("| " + " | ".join(str(row.get(field, "")) for field in fields) + " |")
    return lines


def main() -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    sessions = load_sessions()
    configs = get_configs()
    if not sessions:
        raise SystemExit("No evaluation sessions found")

    session_rows: list[dict[str, Any]] = []
    config_rows: list[dict[str, Any]] = []
    critical_detail_rows: list[dict[str, Any]] = []
    config_payload: dict[str, Any] = {}

    wav_cache: dict[Path, tuple[np.ndarray, int]] = {}
    for config_id, family, cfg in configs:
        config_payload[config_id] = {"family": family, "config": asdict(cfg)}
        totals = {
            "positive_sessions": 0,
            "negative_sessions": 0,
            "truth_total": 0,
            "matched_140": 0,
            "matched_250": 0,
            "positive_candidates": 0,
            "positive_unmatched_140": 0,
            "negative_candidates": 0,
        }
        for session in sessions:
            if not session.wav_path.exists():
                continue
            if session.wav_path not in wav_cache:
                wav_cache[session.wav_path] = read_wav(session.wav_path)
            y, sr = wav_cache[session.wav_path]
            candidates = detect_for_config(y, sr, family, cfg)
            match140 = match_candidates(session.label_times_ms, candidates, MATCH_140_MS)
            match250 = match_candidates(session.label_times_ms, candidates, MATCH_250_MS)

            row = {
                "config_id": config_id,
                "family": family,
                "session_id": session.session_id,
                "scenario_id": session.scenario_id,
                "scenario_title": session.scenario_title,
                "polarity": session.polarity,
                "source": session.source,
                "expected_count": session.expected_count,
                "truth_count": len(session.label_times_ms),
                "candidate_count": len(candidates),
                "matched_140": match140["matched"],
                "recall_140": pct(match140["matched"], len(session.label_times_ms)),
                "missed_140": match140["missed"],
                "unmatched_candidates_140": len(match140["unmatched_candidate_indexes"]),
                "matched_250": match250["matched"],
                "recall_250": pct(match250["matched"], len(session.label_times_ms)),
                "missed_250": match250["missed"],
                "unmatched_candidates_250": len(match250["unmatched_candidate_indexes"]),
            }
            session_rows.append(row)

            if session.polarity == "positive":
                totals["positive_sessions"] += 1
                totals["truth_total"] += len(session.label_times_ms)
                totals["matched_140"] += int(match140["matched"])
                totals["matched_250"] += int(match250["matched"])
                totals["positive_candidates"] += len(candidates)
                totals["positive_unmatched_140"] += len(match140["unmatched_candidate_indexes"])
            else:
                totals["negative_sessions"] += 1
                totals["negative_candidates"] += len(candidates)

            if session.session_id == "bounce_audio_test_session_2026-07-02T17-48-43-807Z":
                matched_by_label = {int(row["label_index"]): row for row in match140["matches"]}
                for idx, label_ms in enumerate(session.label_times_ms, start=1):
                    match = matched_by_label.get(idx)
                    critical_detail_rows.append(
                        {
                            "config_id": config_id,
                            "label_index": idx,
                            "label_time_s": f"{label_ms / 1000.0:.6f}",
                            "matched_140": bool(match),
                            "candidate_time_s": "" if match is None else f"{ff(match['candidate_ms']) / 1000.0:.6f}",
                            "delta_ms": "" if match is None else f"{ff(match['delta_ms']):.1f}",
                        }
                    )

        config_rows.append(
            {
                "config_id": config_id,
                "family": family,
                "positive_sessions": totals["positive_sessions"],
                "truth_total": totals["truth_total"],
                "matched_140": totals["matched_140"],
                "recall_140": pct(totals["matched_140"], totals["truth_total"]),
                "matched_250": totals["matched_250"],
                "recall_250": pct(totals["matched_250"], totals["truth_total"]),
                "positive_candidates": totals["positive_candidates"],
                "positive_unmatched_140": totals["positive_unmatched_140"],
                "negative_sessions": totals["negative_sessions"],
                "negative_candidates": totals["negative_candidates"],
                "negative_candidates_per_session": (
                    f"{totals['negative_candidates'] / totals['negative_sessions']:.1f}"
                    if totals["negative_sessions"]
                    else "0.0"
                ),
            }
        )

    config_rows.sort(
        key=lambda row: (
            -ff(row["matched_140"]),
            ff(row["negative_candidates"]),
            ff(row["positive_unmatched_140"]),
        )
    )
    session_rows.sort(key=lambda row: (row["config_id"], row["polarity"], row["scenario_id"], row["session_id"]))

    write_csv(OUT_DIR / "t0123_config_summary.csv", config_rows)
    write_csv(OUT_DIR / "t0123_session_summary.csv", session_rows)
    write_csv(OUT_DIR / "t0123_t0119_truth_detail.csv", critical_detail_rows)
    summary = {
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "sessions": len(sessions),
        "configs": config_payload,
        "top_configs": config_rows[:5],
        "t0119_reverted_app_reference": {
            "source": "T0121 saved-label analysis",
            "actual_app_candidates": 28,
            "truth_with_nearby_app_candidate_140ms": 20,
            "counted_truth_140ms": 18,
            "truth_without_app_candidate_140ms": 10,
            "note": "This was the reverted calibrated/transient app run, not the current fixed peak gate.",
        },
        "notes": [
            "Candidate-generation audit only.",
            "No classifier/veto/model export/app change.",
            "Negative candidate counts are candidate load, not final false counts.",
        ],
    }
    write_json(OUT_DIR / "t0123_summary.json", summary)

    best_recall = config_rows[0] if config_rows else {}
    best_sliding = next((row for row in config_rows if row.get("family") == "sliding_window_pcm"), {})
    lowest_negative = min(config_rows, key=lambda row: ff(row.get("negative_candidates"))) if config_rows else {}

    report_lines = [
        "# T0123 Sliding-Window PCM Candidate Audit",
        "",
        "Evaluation-only audit. No model training, model export, app runtime change, APK install, push, or production promotion.",
        "",
        "## Short Conclusion",
        "",
        f"- Best recall row: `{best_recall.get('config_id', '')}` at "
        f"{best_recall.get('matched_140', '')}/{best_recall.get('truth_total', '')} "
        f"({best_recall.get('recall_140', '')}) with "
        f"{best_recall.get('negative_candidates', '')} expected-zero candidates.",
        f"- Best sliding-window row: `{best_sliding.get('config_id', '')}` at "
        f"{best_sliding.get('matched_140', '')}/{best_sliding.get('truth_total', '')} "
        f"({best_sliding.get('recall_140', '')}) with "
        f"{best_sliding.get('negative_candidates', '')} expected-zero candidates.",
        f"- Lowest-negative row: `{lowest_negative.get('config_id', '')}` with "
        f"{lowest_negative.get('negative_candidates', '')} expected-zero candidates, but only "
        f"{lowest_negative.get('matched_140', '')}/{lowest_negative.get('truth_total', '')} "
        f"({lowest_negative.get('recall_140', '')}) positive recall.",
        "- Interpretation: sliding-window PCM is promising as a candidate-recovery layer, especially for T0119, but not as a standalone counter. A second-layer classifier/veto is still required.",
        "- T0119 reference: the reverted calibrated/transient app run had `28` actual app candidates, only `20/30` truth labels near candidates at `140 ms`, and only `18/30` truth labels counted. The `current_peak_abs008` row is the current repo's fixed peak-gate reference, not that reverted app mode.",
        "",
        "## What Was Tested",
        "",
        "- Current fixed peak gate: `raw_abs`, `abs_min=0.08`, `ratio>=2`, `min_gap=220 ms`.",
        "- Soft peak gates with lower absolute floors.",
        "- Sliding-window PCM candidates: fixed-stride windows score local raw/high-pass PCM envelope peaks against two-sided local background, then non-max suppress by `180 ms`.",
        "",
        "## Aggregate Results",
        "",
        *md_table(
            config_rows,
            [
                "config_id",
                "family",
                "truth_total",
                "matched_140",
                "recall_140",
                "matched_250",
                "recall_250",
                "positive_candidates",
                "positive_unmatched_140",
                "negative_candidates",
                "negative_candidates_per_session",
            ],
        ),
        "",
        "## Critical T0119 Session",
        "",
    ]
    critical_session_rows = [
        row
        for row in session_rows
        if row["session_id"] == "bounce_audio_test_session_2026-07-02T17-48-43-807Z"
    ]
    report_lines.extend(
        md_table(
            critical_session_rows,
            [
                "config_id",
                "candidate_count",
                "matched_140",
                "recall_140",
                "matched_250",
                "recall_250",
                "unmatched_candidates_140",
            ],
        )
    )
    report_lines.extend(
        [
            "",
            "## Interpretation",
            "",
            "- A useful next candidate generator must recover T0119/T0121 misses without producing an impractical candidate load on expected-zero talking/handling sessions.",
            "- Sliding-window rows here are candidate generators only; they still need a classifier/veto/replay layer before any phone install.",
            "- The `negative_candidates` column is intentionally strict pressure: each expected-zero candidate would need to be rejected downstream.",
            "",
            "## Outputs",
            "",
            "- `t0123_config_summary.csv`",
            "- `t0123_session_summary.csv`",
            "- `t0123_t0119_truth_detail.csv`",
            "- `t0123_summary.json`",
            "",
        ]
    )
    (OUT_DIR / "t0123_report.md").write_text("\n".join(report_lines), encoding="utf-8")
    print(f"Wrote T0123 outputs to {OUT_DIR}")
    print("Top configs:")
    for row in config_rows[:5]:
        print(
            f"  {row['config_id']}: {row['matched_140']}/{row['truth_total']} "
            f"({row['recall_140']}), negative candidates {row['negative_candidates']}"
        )


if __name__ == "__main__":
    main()
