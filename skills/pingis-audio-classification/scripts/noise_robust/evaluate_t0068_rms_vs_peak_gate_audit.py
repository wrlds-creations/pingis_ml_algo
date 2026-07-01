#!/usr/bin/env python3
"""T0068 RMS/native gate versus peak-gate candidate audit.

Evaluation only. This script compares candidate generation before any final
Fable count decision:

- current RMS/native-style gate and a few RMS variants;
- T0067 strict peak gate and faster/lower-cooldown peak variants.

It does not train, export, build, install, or change app runtime behavior.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import sys
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable

import numpy as np

ROOT = Path(__file__).resolve().parents[4]
SCRIPT_DIR = Path(__file__).resolve().parent
if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))

import nr_features  # noqa: E402
from evaluate_t0067_peak_gate_replay import (  # noqa: E402
    PeakGateConfig,
    detect_peak_gate,
    load_exact_positive_truth,
    match_predictions,
    read_csv,
    read_wav,
    selected_review_rows,
    write_csv,
)


DEFAULT_RAW_DIR = ROOT / "data/audio/raw/t0065_fable_training_audio_round_a/fable_training_audio"
DEFAULT_T0066_DIR = ROOT / "data/audio/models/evaluations/t0066_round_a_exact_label_review"
DEFAULT_T0065_DIR = ROOT / "data/audio/models/evaluations/t0065_fable_training_audio_round_a"
DEFAULT_OUT_DIR = ROOT / "data/audio/models/evaluations/t0068_rms_vs_peak_gate_audit"
TOLERANCES_MS = (140.0, 250.0)


@dataclass(frozen=True)
class RmsGateConfig:
    mode: str
    onset_ratio: float
    retrigger_ms: int
    abs_min_rms: float
    spectral_gate: bool = False

    @property
    def config_id(self) -> str:
        spec = "spec" if self.spectral_gate else "nospec"
        return (
            f"rms_{self.mode}_r{self.onset_ratio:g}_gap{self.retrigger_ms:g}"
            f"_abs{self.abs_min_rms:g}_{spec}"
        )


@dataclass(frozen=True)
class GateSpec:
    family: str
    label: str
    config: RmsGateConfig | PeakGateConfig

    @property
    def config_id(self) -> str:
        return self.config.config_id


def finite_float(value: Any, default: float = float("nan")) -> float:
    try:
        out = float(value)
    except Exception:
        return default
    return out if math.isfinite(out) else default


def detect_rms_gate(y: np.ndarray, sr: int, cfg: RmsGateConfig) -> list[dict[str, Any]]:
    triggers = nr_features.simulate_gate(
        y,
        sr,
        onset_ratio=cfg.onset_ratio,
        retrigger_ms=cfg.retrigger_ms,
        abs_min_rms=cfg.abs_min_rms,
        mode=cfg.mode,
        spectral_gate=cfg.spectral_gate,
    )
    rows: list[dict[str, Any]] = []
    for trigger in triggers:
        if not trigger.get("passed_spectral", True):
            continue
        rows.append(
            {
                "time_ms": float(trigger["onset_ms"]),
                "onset_sample": int(trigger["onset_sample"]),
                "frame_rms": finite_float(trigger.get("frame_rms")),
                "bg_rms": finite_float(trigger.get("bg_rms")),
            }
        )
    return rows


def detect_for_spec(y: np.ndarray, sr: int, spec: GateSpec) -> list[dict[str, Any]]:
    if spec.family == "rms":
        return detect_rms_gate(y, sr, spec.config)  # type: ignore[arg-type]
    if spec.family == "peak":
        return detect_peak_gate(y, sr, spec.config)  # type: ignore[arg-type]
    raise ValueError(f"Unknown gate family: {spec.family}")


def gate_specs() -> list[GateSpec]:
    rms = [
        GateSpec("rms", "RMS current/native", RmsGateConfig("bandpass", 1.5, 120, 0.0015)),
        GateSpec("rms", "RMS liberal fast", RmsGateConfig("bandpass", 1.3, 80, 0.0010)),
        GateSpec("rms", "RMS fast gap only", RmsGateConfig("bandpass", 1.5, 80, 0.0015)),
        GateSpec("rms", "RMS slower gap", RmsGateConfig("bandpass", 1.5, 220, 0.0015)),
        GateSpec("rms", "RMS conservative", RmsGateConfig("bandpass", 2.0, 220, 0.0025)),
        GateSpec("rms", "RMS broadband", RmsGateConfig("broadband", 1.5, 120, 0.0030)),
    ]
    peak = [
        GateSpec("peak", "Peak T0067 strict", PeakGateConfig("raw_abs", 5.0, 380.0, 500.0, 60.0, 0.17, 2.0, 0.0)),
        GateSpec("peak", "Peak strict shorter gap", PeakGateConfig("raw_abs", 5.0, 300.0, 500.0, 60.0, 0.17, 2.0, 0.0)),
        GateSpec("peak", "Peak fast balanced", PeakGateConfig("raw_abs", 3.0, 220.0, 500.0, 60.0, 0.08, 2.0, 0.0)),
        GateSpec("peak", "Peak fast liberal", PeakGateConfig("raw_abs", 3.0, 160.0, 500.0, 60.0, 0.055, 2.0, 0.0)),
        GateSpec("peak", "Peak very liberal", PeakGateConfig("raw_abs", 3.0, 140.0, 500.0, 60.0, 0.038, 2.0, 0.0)),
        GateSpec("peak", "Peak bandpass fast", PeakGateConfig("bp_abs", 3.0, 160.0, 500.0, 60.0, 0.055, 2.0, 0.0)),
        GateSpec("peak", "Peak bandpass balanced", PeakGateConfig("bp_abs", 3.0, 220.0, 500.0, 60.0, 0.08, 2.0, 0.0)),
    ]
    return rms + peak


def prepared_selected_audio(raw_dir: Path, t0066_dir: Path) -> dict[str, tuple[np.ndarray, int]]:
    ids = [row["session_id"] for row in selected_review_rows(t0066_dir)]
    return {sid: read_wav(raw_dir / f"{sid}.wav") for sid in ids}


def summarize_matches(pred_ms: list[float], truth_ms: list[float], tolerance_ms: float) -> dict[str, Any]:
    result = match_predictions(pred_ms, truth_ms, tolerance_ms)
    deltas = [float(delta) for _, _, delta in result["matches"]]
    abs_deltas = [abs(delta) for delta in deltas]
    return {
        "tp": result["tp"],
        "fp": result["fp"],
        "missed": result["missed"],
        "deltas": deltas,
        "median_abs_delta_ms": float(np.median(abs_deltas)) if abs_deltas else None,
        "p95_abs_delta_ms": float(np.percentile(abs_deltas, 95)) if abs_deltas else None,
    }


def score_exact_selected(
    spec: GateSpec,
    raw_dir: Path,
    t0066_dir: Path,
    prepared: dict[str, tuple[np.ndarray, int]],
) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    truth_by_session = load_exact_positive_truth(t0066_dir)
    selected_rows = selected_review_rows(t0066_dir)
    exact_rows: list[dict[str, Any]] = []
    truth_total = 0
    positive_candidate_total = 0
    negative_candidate_total = 0
    positive_duration_s = 0.0
    negative_duration_s = 0.0
    matches_by_tol: dict[float, dict[str, Any]] = {
        tol: {"tp": 0, "fp": 0, "missed": 0, "deltas": []} for tol in TOLERANCES_MS
    }
    negative_by_scenario: dict[str, int] = {}

    for row in selected_rows:
        sid = row["session_id"]
        y, sr = prepared[sid]
        events = detect_for_spec(y, sr, spec)
        pred_ms = [float(event["time_ms"]) for event in events]
        duration_s = finite_float(row.get("wav_duration_s"), 0.0)
        if sid in truth_by_session:
            truth = truth_by_session[sid]
            truth_total += len(truth)
            positive_candidate_total += len(pred_ms)
            positive_duration_s += duration_s
            for tol in TOLERANCES_MS:
                matched = summarize_matches(pred_ms, truth, tol)
                matches_by_tol[tol]["tp"] += matched["tp"]
                matches_by_tol[tol]["fp"] += matched["fp"]
                matches_by_tol[tol]["missed"] += matched["missed"]
                matches_by_tol[tol]["deltas"].extend(matched["deltas"])
            exact_rows.append(
                {
                    "config_id": spec.config_id,
                    "label": spec.label,
                    "family": spec.family,
                    "session_id": sid,
                    "scenario_title": row.get("scenario_title", ""),
                    "polarity": "positive_exact",
                    "truth": len(truth),
                    "candidates": len(pred_ms),
                }
            )
        elif row.get("polarity") == "negative":
            negative_candidate_total += len(pred_ms)
            negative_duration_s += duration_s
            scenario = row.get("scenario_title", "")
            negative_by_scenario[scenario] = negative_by_scenario.get(scenario, 0) + len(pred_ms)
            exact_rows.append(
                {
                    "config_id": spec.config_id,
                    "label": spec.label,
                    "family": spec.family,
                    "session_id": sid,
                    "scenario_title": scenario,
                    "polarity": "expected_zero",
                    "truth": 0,
                    "candidates": len(pred_ms),
                }
            )

    def aggregate_tol(tol: float) -> dict[str, Any]:
        item = matches_by_tol[tol]
        abs_deltas = [abs(float(delta)) for delta in item["deltas"]]
        return {
            f"tp_{int(tol)}ms": int(item["tp"]),
            f"missed_{int(tol)}ms": int(item["missed"]),
            f"positive_extra_{int(tol)}ms": int(item["fp"]),
            f"recall_{int(tol)}ms": item["tp"] / truth_total if truth_total else 0.0,
            f"median_abs_delta_{int(tol)}ms": float(np.median(abs_deltas)) if abs_deltas else "",
            f"p95_abs_delta_{int(tol)}ms": float(np.percentile(abs_deltas, 95)) if abs_deltas else "",
        }

    summary = {
        "config_id": spec.config_id,
        "label": spec.label,
        "family": spec.family,
        **asdict(spec.config),
        "exact_truth": truth_total,
        "positive_candidates": positive_candidate_total,
        "positive_candidates_per_truth": positive_candidate_total / truth_total if truth_total else 0.0,
        "positive_candidates_per_min": positive_candidate_total / (positive_duration_s / 60.0) if positive_duration_s else 0.0,
        "selected_expected_zero_candidates": negative_candidate_total,
        "selected_expected_zero_candidates_per_min": (
            negative_candidate_total / (negative_duration_s / 60.0) if negative_duration_s else 0.0
        ),
        "selected_expected_zero_by_scenario": json.dumps(negative_by_scenario, sort_keys=True),
    }
    for tol in TOLERANCES_MS:
        summary.update(aggregate_tol(tol))
    return summary, exact_rows


def round_a_block_replay(
    spec: GateSpec,
    raw_dir: Path,
    t0065_dir: Path,
    audio_cache: dict[str, tuple[np.ndarray, int]],
) -> list[dict[str, Any]]:
    manifest = read_csv(t0065_dir / "t0065_fable_training_audio_manifest.csv")
    rows: list[dict[str, Any]] = []
    for row in manifest:
        if row.get("round") != "round_a":
            continue
        sid = row["session_id"]
        if sid not in audio_cache:
            audio_cache[sid] = read_wav(raw_dir / f"{sid}.wav")
        y, sr = audio_cache[sid]
        candidates = len(detect_for_spec(y, sr, spec))
        expected = int(row.get("expected_racket_contacts") or 0)
        duration_s = finite_float(row.get("wav_duration_s"), 0.0)
        rows.append(
            {
                "config_id": spec.config_id,
                "label": spec.label,
                "family": spec.family,
                "session_id": sid,
                "scenario_id": row.get("scenario_id", ""),
                "scenario_title": row.get("scenario_title", ""),
                "polarity": row.get("polarity", ""),
                "expected_contacts": expected,
                "candidate_count": candidates,
                "count_error": candidates - expected,
                "abs_count_error": abs(candidates - expected),
                "duration_s": duration_s,
                "candidates_per_min": candidates / (duration_s / 60.0) if duration_s else 0.0,
            }
        )
    return rows


def summarize_round_a(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    grouped: dict[tuple[str, str], dict[str, Any]] = {}
    for row in rows:
        key = (str(row["config_id"]), str(row["scenario_id"]))
        item = grouped.setdefault(
            key,
            {
                "config_id": row["config_id"],
                "label": row["label"],
                "family": row["family"],
                "scenario_id": row["scenario_id"],
                "scenario_title": row["scenario_title"],
                "polarity": row["polarity"],
                "clips": 0,
                "expected_contacts": 0,
                "candidate_count": 0,
                "abs_count_error": 0,
                "duration_s": 0.0,
            },
        )
        item["clips"] += 1
        item["expected_contacts"] += int(row["expected_contacts"])
        item["candidate_count"] += int(row["candidate_count"])
        item["abs_count_error"] += int(row["abs_count_error"])
        item["duration_s"] += float(row["duration_s"])
    for item in grouped.values():
        item["count_error"] = int(item["candidate_count"]) - int(item["expected_contacts"])
        item["candidates_per_min"] = (
            float(item["candidate_count"]) / (float(item["duration_s"]) / 60.0)
            if float(item["duration_s"]) > 0
            else 0.0
        )
    return sorted(grouped.values(), key=lambda row: (row["config_id"], row["scenario_id"]))


def summarize_configs(exact_rows: list[dict[str, Any]], round_summary: list[dict[str, Any]]) -> list[dict[str, Any]]:
    config_rows: dict[str, dict[str, Any]] = {}
    for row in exact_rows:
        config_rows[row["config_id"]] = dict(row)
    for row in round_summary:
        config_id = str(row["config_id"])
        item = config_rows.setdefault(
            config_id,
            {
                "config_id": config_id,
                "label": row["label"],
                "family": row["family"],
            },
        )
        scenario_id = str(row["scenario_id"])
        if scenario_id == "fast_racket_bounce":
            item["round_a_fast_expected"] = row["expected_contacts"]
            item["round_a_fast_candidates"] = row["candidate_count"]
            item["round_a_fast_error"] = row["count_error"]
        if str(row["polarity"]) == "negative":
            item["round_a_negative_candidates"] = item.get("round_a_negative_candidates", 0) + int(row["candidate_count"])
        item["round_a_total_expected"] = item.get("round_a_total_expected", 0) + int(row["expected_contacts"])
        item["round_a_total_candidates"] = item.get("round_a_total_candidates", 0) + int(row["candidate_count"])
        item["round_a_total_abs_error"] = item.get("round_a_total_abs_error", 0) + int(row["abs_count_error"])
    for item in config_rows.values():
        expected = int(item.get("round_a_total_expected", 0))
        candidates = int(item.get("round_a_total_candidates", 0))
        item["round_a_total_error"] = candidates - expected
    return sorted(
        config_rows.values(),
        key=lambda row: (
            -float(row.get("recall_140ms", 0.0)),
            int(row.get("selected_expected_zero_candidates", 10**9)),
            abs(int(row.get("round_a_fast_error", 10**9))),
        ),
    )


def md_table(rows: list[dict[str, Any]], fields: list[str], labels: list[str] | None = None, limit: int | None = None) -> list[str]:
    if labels is None:
        labels = fields
    shown = rows[:limit] if limit is not None else rows
    lines = [
        "| " + " | ".join(labels) + " |",
        "| " + " | ".join("---" for _ in labels) + " |",
    ]
    for row in shown:
        values = []
        for field in fields:
            value = row.get(field, "")
            if isinstance(value, float):
                if "recall" in field:
                    values.append(f"{value:.3f}")
                elif "per_" in field or "delta" in field:
                    values.append(f"{value:.1f}")
                else:
                    values.append(f"{value:.3f}")
            else:
                values.append(str(value))
        lines.append("| " + " | ".join(values) + " |")
    return lines


def render_report(summary: dict[str, Any], exact_rows: list[dict[str, Any]], config_rows: list[dict[str, Any]], scenario_rows: list[dict[str, Any]]) -> str:
    rms_current = next(row for row in config_rows if row["config_id"].startswith("rms_bandpass_r1.5_gap120_abs0.0015"))
    peak_best_recall = max(
        [row for row in config_rows if row["family"] == "peak"],
        key=lambda row: (float(row.get("recall_140ms", 0.0)), -int(row.get("selected_expected_zero_candidates", 0))),
    )
    peak_best_fast = min(
        [row for row in config_rows if row["family"] == "peak"],
        key=lambda row: abs(int(row.get("round_a_fast_error", 999999))),
    )
    lines = [
        "# T0068 RMS vs Peak Gate Audit",
        "",
        f"Generated at: `{summary['generated_at']}`",
        "",
        "## Scope",
        "",
        "- Evaluation only: no model JSON, app runtime, APK, training export, or threshold change.",
        "- RMS/native rows are candidate triggers before final Fable accept/reject.",
        "- Peak rows are candidate triggers before any classifier/veto.",
        "- Round A positive clips have expected counts, not exact timestamps, except the reviewed background-sound subset.",
        "- This proves candidate-gate behavior only; it does not prove final peak-plus-Fable counted behavior.",
        "",
        "## Exact Reviewed Background Labels",
        "",
        *md_table(
            config_rows,
            [
                "label",
                "family",
                "recall_140ms",
                "recall_250ms",
                "positive_candidates",
                "positive_extra_140ms",
                "selected_expected_zero_candidates",
                "median_abs_delta_140ms",
                "p95_abs_delta_140ms",
            ],
            [
                "Gate",
                "Type",
                "Recall 140",
                "Recall 250",
                "Positive Cand.",
                "Extra 140",
                "Selected Neg. Cand.",
                "Median Delta",
                "P95 Delta",
            ],
        ),
        "",
        "## Round A Summary",
        "",
        *md_table(
            config_rows,
            [
                "label",
                "family",
                "round_a_fast_candidates",
                "round_a_fast_error",
                "round_a_negative_candidates",
                "round_a_total_candidates",
                "round_a_total_error",
                "round_a_total_abs_error",
            ],
            [
                "Gate",
                "Type",
                "Fast Cand.",
                "Fast Error",
                "Neg. Cand.",
                "Total Cand.",
                "Total Error",
                "Abs Error",
            ],
        ),
        "",
        "## Scenario Details",
        "",
        *md_table(
            scenario_rows,
            [
                "label",
                "scenario_title",
                "expected_contacts",
                "candidate_count",
                "count_error",
                "candidates_per_min",
            ],
            ["Gate", "Scenario", "Expected", "Cand.", "Error", "Cand/min"],
        ),
        "",
        "## Recommendation",
        "",
    ]
    peak_dominates_rms = (
        float(peak_best_recall.get("recall_140ms", 0.0)) >= float(rms_current.get("recall_140ms", 0.0))
        and int(peak_best_fast.get("round_a_fast_candidates", 0)) >= int(peak_best_fast.get("round_a_fast_expected", 1))
        and int(peak_best_recall.get("selected_expected_zero_candidates", 999999))
        <= int(rms_current.get("selected_expected_zero_candidates", 0))
    )
    if peak_dominates_rms:
        lines.append(
            "- A tested peak config dominates current RMS/native as a candidate gate in this offline audit. "
            "It is reasonable to prototype peak as the primary candidate gate next, but keep RMS as a fallback until hybrid app-style replay passes."
        )
    else:
        lines.append(
            "- Do not discard RMS yet. Peak configs improve exact background-bounce recall, but no tested peak config dominates current RMS across fast coverage and hard-negative candidate load."
        )
    lines += [
        f"- Current RMS/native exact recall: `{float(rms_current['recall_140ms']):.3f}` at 140 ms, fast Round A candidates `{rms_current.get('round_a_fast_candidates')}` for expected `{rms_current.get('round_a_fast_expected')}`.",
        f"- Best peak exact-recall config: `{peak_best_recall['label']}` with recall `{float(peak_best_recall['recall_140ms']):.3f}` and selected negative candidates `{peak_best_recall.get('selected_expected_zero_candidates')}`.",
        f"- Best peak fast-coverage config: `{peak_best_fast['label']}` with fast candidates `{peak_best_fast.get('round_a_fast_candidates')}` for expected `{peak_best_fast.get('round_a_fast_expected')}`.",
        "- Practical next step: run peak-candidate + classifier/veto + smart dedupe replay. Keep RMS available as baseline/fallback until that full chain beats the current app behavior.",
        "",
        "## Outputs",
        "",
        "- `t0068_exact_candidate_comparison.csv`",
        "- `t0068_selected_clip_rows.csv`",
        "- `t0068_round_a_block_replay.csv`",
        "- `t0068_round_a_by_scenario.csv`",
        "- `t0068_config_summary.csv`",
        "- `t0068_summary.json`",
    ]
    return "\n".join(lines) + "\n"


def run(args: argparse.Namespace) -> dict[str, Any]:
    raw_dir = Path(args.raw_dir)
    t0066_dir = Path(args.t0066_dir)
    t0065_dir = Path(args.t0065_dir)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    selected_audio = prepared_selected_audio(raw_dir, t0066_dir)
    specs = gate_specs()

    exact_summary_rows: list[dict[str, Any]] = []
    selected_clip_rows: list[dict[str, Any]] = []
    round_rows: list[dict[str, Any]] = []
    audio_cache: dict[str, tuple[np.ndarray, int]] = dict(selected_audio)

    for spec in specs:
        exact_summary, exact_detail = score_exact_selected(spec, raw_dir, t0066_dir, selected_audio)
        exact_summary_rows.append(exact_summary)
        selected_clip_rows.extend(exact_detail)
        round_rows.extend(round_a_block_replay(spec, raw_dir, t0065_dir, audio_cache))

    scenario_rows = summarize_round_a(round_rows)
    config_rows = summarize_configs(exact_summary_rows, scenario_rows)
    rms_current = next(row for row in config_rows if row["config_id"].startswith("rms_bandpass_r1.5_gap120_abs0.0015"))
    peak_dominates = any(
        row["family"] == "peak"
        and float(row.get("recall_140ms", 0.0)) >= float(rms_current.get("recall_140ms", 0.0))
        and int(row.get("selected_expected_zero_candidates", 999999))
        <= int(rms_current.get("selected_expected_zero_candidates", 0))
        and int(row.get("round_a_fast_candidates", 0)) >= int(row.get("round_a_fast_expected", 1))
        for row in config_rows
    )
    summary = {
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "tolerances_ms": list(TOLERANCES_MS),
        "configs_evaluated": len(specs),
        "exact_truth_labels": int(config_rows[0].get("exact_truth", 0)) if config_rows else 0,
        "recommendation": (
            "peak_candidate_gate_dominates_current_rms_keep_rms_fallback_until_hybrid_replay"
            if peak_dominates
            else "do_not_discard_rms_yet"
        ),
    }

    write_csv(out_dir / "t0068_exact_candidate_comparison.csv", exact_summary_rows)
    write_csv(out_dir / "t0068_selected_clip_rows.csv", selected_clip_rows)
    write_csv(out_dir / "t0068_round_a_block_replay.csv", round_rows)
    write_csv(out_dir / "t0068_round_a_by_scenario.csv", scenario_rows)
    write_csv(out_dir / "t0068_config_summary.csv", config_rows)
    (out_dir / "t0068_summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    (out_dir / "t0068_rms_vs_peak_gate_report.md").write_text(
        render_report(summary, exact_summary_rows, config_rows, scenario_rows),
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
    print(json.dumps(run(args), indent=2))


if __name__ == "__main__":
    main()
