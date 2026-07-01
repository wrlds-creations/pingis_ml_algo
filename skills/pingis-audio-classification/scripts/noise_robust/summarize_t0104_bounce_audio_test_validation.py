#!/usr/bin/env python3
"""Summarize fresh Bounce audio test debug sessions for T0104.

This is an audit/report helper only. It reads ignored pulled device JSON/WAV
debug files and writes ignored summary artifacts. It does not train, export, or
change app runtime behavior.
"""

from __future__ import annotations

import argparse
import csv
import json
import statistics
from collections import Counter, defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[4]
DEFAULT_RAW_DIR = ROOT / "data/audio/raw/t0104_bounce_audio_test_live_validation/bounce_audio_test_debug"
DEFAULT_OUT_DIR = ROOT / "data/audio/models/evaluations/t0104_bounce_audio_test_live_validation"
THRESHOLDS = [0.2, 0.3, 0.4, 0.5, 0.575]
LOVE_REPORTED_EXPECTED_BY_SCENARIO = {
    "normal_racket_bounce": 30,
    "slow_high_racket_bounce": 30,
    "fast_racket_bounce": 30,
    "racket_bounce_speaking_counting": 30,
    "racket_bounce_background_sound": 30,
    "far_soft_racket_bounce_background": 30,
    "talking_only_no_bounce": 0,
    "racket_handling_no_bounce": 0,
}
MANUAL_SESSION_REVIEW = {
    "bounce_audio_test_session_2026-07-01T13-37-11-083Z": {
        "expected": 20,
        "include_in_metrics": True,
        "note": "Love confirmed this slow/high run should be expected 20.",
    },
    "bounce_audio_test_session_2026-07-01T13-38-19-066Z": {
        "expected": None,
        "include_in_metrics": False,
        "note": "Love was unsure about the actual bounce count; exclude this clip from truth/validation.",
    },
}


@dataclass(frozen=True)
class SessionSummary:
    session_id: str
    started_at: str
    scenario_id: str
    scenario_title: str
    polarity: str
    saved_expected: int | None
    expected: int | None
    include_in_metrics: bool
    review_note: str
    app_count: int
    candidate_count: int
    counted: int
    low_probability: int
    deduped: int
    fable_noise_vetoed: int
    median_probability: float | None
    max_probability: float | None
    p25_probability: float | None
    p75_probability: float | None
    fable_labels: dict[str, int]
    reject_reasons: dict[str, int]
    threshold_counts: dict[str, int]


def _read_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8-sig"))


def _probability(candidate: dict[str, Any]) -> float:
    value = candidate.get("classifier_probability")
    return float(value) if isinstance(value, (int, float)) else 0.0


def _candidate_time_ms(candidate: dict[str, Any]) -> float:
    native_debug = candidate.get("native_debug") if isinstance(candidate.get("native_debug"), dict) else {}
    onset_ms = candidate.get("native_onset_time_ms", native_debug.get("onset_time_ms"))
    if isinstance(onset_ms, (int, float)):
        return float(onset_ms)
    onset_pos = candidate.get("native_onset_pos", native_debug.get("onset_pos"))
    if isinstance(onset_pos, (int, float)):
        return float(onset_pos) / 22050.0 * 1000.0
    return 0.0


def _smart_count(candidates: list[dict[str, Any]], threshold: float, dedupe_ms: float) -> int:
    accepted: list[dict[str, Any]] = []
    last_time = -1e18
    for candidate in sorted(candidates, key=_candidate_time_ms):
        if _probability(candidate) < threshold:
            continue
        time_ms = _candidate_time_ms(candidate)
        if accepted and time_ms - last_time < dedupe_ms:
            previous = accepted[-1]
            if _probability(candidate) > _probability(previous):
                accepted[-1] = candidate
                last_time = time_ms
            continue
        accepted.append(candidate)
        last_time = time_ms
    return len(accepted)


def summarize_session(path: Path) -> SessionSummary | None:
    data = _read_json(path)
    if data.get("type") != "bounce_audio_test_debug_session":
        return None
    candidates = data.get("candidates") if isinstance(data.get("candidates"), list) else []
    review = data.get("review") if isinstance(data.get("review"), dict) else {}
    scenario = review.get("scenario") if isinstance(review.get("scenario"), dict) else {}
    counts = data.get("counts") if isinstance(data.get("counts"), dict) else {}
    decision_config = data.get("decision_config") if isinstance(data.get("decision_config"), dict) else {}
    probs = [_probability(candidate) for candidate in candidates]
    saved_expected = review.get("expected_racket_contacts")
    if not isinstance(saved_expected, int):
        saved_expected = None
    scenario_id = str(scenario.get("id") or "unknown")
    expected = LOVE_REPORTED_EXPECTED_BY_SCENARIO.get(scenario_id, saved_expected)
    manual_review = MANUAL_SESSION_REVIEW.get(path.stem, {})
    if "expected" in manual_review:
        expected = manual_review["expected"]
    include_in_metrics = bool(manual_review.get("include_in_metrics", True))
    review_note = str(manual_review.get("note") or "")
    fable_labels = Counter(str(candidate.get("fable_label") or "unknown") for candidate in candidates)
    reject_reasons = Counter()
    for candidate in candidates:
        if candidate.get("counted"):
            reject_reasons["counted"] += 1
        else:
            reject_reasons[str(candidate.get("reject_reason") or candidate.get("decision") or "unknown")] += 1
    dedupe_ms = float(decision_config.get("smartDedupeMs", 180))
    threshold_counts = {f"p>={threshold:g}": _smart_count(candidates, threshold, dedupe_ms) for threshold in THRESHOLDS}
    return SessionSummary(
        session_id=path.stem,
        started_at=str(data.get("started_at") or ""),
        scenario_id=scenario_id,
        scenario_title=str(scenario.get("title") or "Unknown"),
        polarity=str(scenario.get("polarity") or "unknown"),
        saved_expected=saved_expected,
        expected=expected,
        include_in_metrics=include_in_metrics,
        review_note=review_note,
        app_count=int(review.get("app_count_at_stop") if isinstance(review.get("app_count_at_stop"), int) else counts.get("counted", 0)),
        candidate_count=len(candidates),
        counted=int(counts.get("counted", sum(1 for candidate in candidates if candidate.get("counted")))),
        low_probability=int(counts.get("low_probability", 0)),
        deduped=int(counts.get("deduped", 0)),
        fable_noise_vetoed=int(counts.get("fable_noise_vetoed", 0)),
        median_probability=statistics.median(probs) if probs else None,
        max_probability=max(probs) if probs else None,
        p25_probability=statistics.quantiles(probs, n=4)[0] if len(probs) >= 4 else None,
        p75_probability=statistics.quantiles(probs, n=4)[2] if len(probs) >= 4 else None,
        fable_labels=dict(sorted(fable_labels.items())),
        reject_reasons=dict(sorted(reject_reasons.items())),
        threshold_counts=threshold_counts,
    )


def write_csv(path: Path, rows: list[SessionSummary]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fields = [
        "session_id",
        "started_at",
        "scenario_id",
        "scenario_title",
        "polarity",
        "saved_expected",
        "expected",
        "include_in_metrics",
        "review_note",
        "app_count",
        "candidate_count",
        "counted",
        "low_probability",
        "deduped",
        "fable_noise_vetoed",
        "median_probability",
        "max_probability",
        "p25_probability",
        "p75_probability",
        "fable_labels",
        "reject_reasons",
        *[f"p>={threshold:g}" for threshold in THRESHOLDS],
    ]
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        for row in rows:
            record = row.__dict__.copy()
            record["fable_labels"] = json.dumps(row.fable_labels, sort_keys=True)
            record["reject_reasons"] = json.dumps(row.reject_reasons, sort_keys=True)
            for key, value in row.threshold_counts.items():
                record[key] = value
            record.pop("threshold_counts", None)
            writer.writerow(record)


def aggregate_by_scenario(rows: list[SessionSummary]) -> list[dict[str, Any]]:
    grouped: dict[str, list[SessionSummary]] = defaultdict(list)
    for row in rows:
        if not row.include_in_metrics:
            continue
        grouped[row.scenario_id].append(row)
    output: list[dict[str, Any]] = []
    for scenario_id, items in grouped.items():
        expected = sum(item.expected or 0 for item in items)
        app_count = sum(item.app_count for item in items)
        candidates = sum(item.candidate_count for item in items)
        low_probability = sum(item.low_probability for item in items)
        max_probability = max((item.max_probability or 0.0) for item in items) if items else 0.0
        threshold_counts = {
            f"p>={threshold:g}": sum(item.threshold_counts[f"p>={threshold:g}"] for item in items)
            for threshold in THRESHOLDS
        }
        output.append(
            {
                "scenario_id": scenario_id,
                "scenario_title": items[0].scenario_title,
                "polarity": items[0].polarity,
                "runs": len(items),
                "expected": expected,
                "app_count": app_count,
                "candidate_count": candidates,
                "low_probability": low_probability,
                "max_probability": max_probability,
                **threshold_counts,
            }
        )
    return sorted(output, key=lambda row: (row["polarity"] != "positive", row["scenario_id"]))


def write_report(path: Path, rows: list[SessionSummary]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    aggregate = aggregate_by_scenario(rows)
    metric_rows = [row for row in rows if row.include_in_metrics]
    positive = [row for row in metric_rows if row.polarity == "positive"]
    negative = [row for row in metric_rows if row.polarity == "negative"]
    excluded = [row for row in rows if not row.include_in_metrics]
    total_expected_positive = sum(row.expected or 0 for row in positive)
    total_counted_positive = sum(row.app_count for row in positive)
    total_candidates_positive = sum(row.candidate_count for row in positive)
    total_low_positive = sum(row.low_probability for row in positive)
    total_expected_negative = sum(row.expected or 0 for row in negative)
    total_counted_negative = sum(row.app_count for row in negative)
    lines = [
        "# T0104 Bounce Audio Test Live Validation",
        "",
        "## Summary",
        "",
        f"- Fresh JSON sessions analyzed: `{len(rows)}`.",
        f"- Positive expected/count: `{total_counted_positive}/{total_expected_positive}` with `{total_candidates_positive}` peak candidates.",
        f"- Positive low-probability rejections: `{total_low_positive}`.",
        f"- Negative expected/count: `{total_counted_negative}/{total_expected_negative}`.",
        "- `Expected` uses Love's reported counts plus manual corrections. After T0104A review, the first slow/high run is confirmed as `20`, and the second slow/high run is excluded because the true count is unclear.",
        "- Dedupe and Fable-noise veto were not material in this pull; the dominant positive miss reason is `below_threshold`.",
        f"- Excluded unclear sessions: `{len(excluded)}`.",
        "",
        "## Scenario Totals",
        "",
        "| Scenario | Runs | Expected | App Count | Candidates | Low Prob | Max p | p>=0.3 replay | p>=0.5 replay | p>=0.575 replay |",
        "|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for row in aggregate:
        lines.append(
            "| {scenario_title} | {runs} | {expected} | {app_count} | {candidate_count} | {low_probability} | {max_probability:.3f} | {p03} | {p05} | {p0575} |".format(
                **row,
                p03=row["p>=0.3"],
                p05=row["p>=0.5"],
                p0575=row["p>=0.575"],
            )
        )
    lines.extend(
        [
            "",
            "## Per Session",
            "",
            "| Started | Scenario | Expected | App | Candidates | Low Prob | Median p | Max p | Fable labels | Reject reasons |",
            "|---|---|---:|---:|---:|---:|---:|---:|---|---|",
        ]
    )
    for row in rows:
        lines.append(
            "| {started} | {scenario} | {expected} | {app} | {candidates} | {low} | {median:.3f} | {maxp:.3f} | `{fable}` | `{rejects}` |".format(
                started=row.started_at,
                scenario=f"{row.scenario_title}{' (excluded)' if not row.include_in_metrics else ''}",
                expected=(
                    "excluded"
                    if not row.include_in_metrics
                    else "" if row.expected is None
                    else f"{row.expected}*" if row.saved_expected != row.expected
                    else row.expected
                ),
                app=row.app_count,
                candidates=row.candidate_count,
                low=row.low_probability,
                median=row.median_probability or 0.0,
                maxp=row.max_probability or 0.0,
                fable=json.dumps(row.fable_labels, sort_keys=True),
                rejects=json.dumps(row.reject_reasons, sort_keys=True),
            )
        )
    lines.extend(
        [
            "",
            "## Interpretation",
            "",
            "- The peak gate is not the main blocker for most fresh positives: every included positive run has roughly the expected candidate volume. The first slow/high run is now confirmed as expected `20`; the second slow/high run is not used because the true count is unclear.",
            "- The final classifier threshold is the main blocker in the weak runs: far/soft + background had `88` candidates across two runs but only `4` counted at `p>=0.575`.",
            "- The current T0103 threshold stayed very safe on the two fresh hard-negative types: talking-only and racket-handling-only both counted `0`, despite `287` negative peak candidates.",
            "- A lower threshold such as `p>=0.3` would recover many positives in this exact pull without counting these two negative types, but older Round A/T0073 safety says threshold-only lowering is risky. Treat it as a diagnostic, not a promotion.",
        ]
    )
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--raw-dir", type=Path, default=DEFAULT_RAW_DIR)
    parser.add_argument("--out-dir", type=Path, default=DEFAULT_OUT_DIR)
    args = parser.parse_args()
    rows = [row for path in sorted(args.raw_dir.glob("bounce_audio_test_session_2026-07-01T*.json")) if (row := summarize_session(path))]
    args.out_dir.mkdir(parents=True, exist_ok=True)
    write_csv(args.out_dir / "t0104_session_summary.csv", rows)
    aggregate = aggregate_by_scenario(rows)
    (args.out_dir / "t0104_summary.json").write_text(
        json.dumps({"sessions": [row.__dict__ for row in rows], "scenarios": aggregate}, indent=2, sort_keys=True),
        encoding="utf-8",
    )
    write_report(args.out_dir / "t0104_report.md", rows)
    print(f"Wrote {len(rows)} session summaries to {args.out_dir}")


if __name__ == "__main__":
    main()
