"""
Build a candidate-centered report for playing retro audio.

This script does not train or export a model. It compares candidate peaks from:
- saved Collector review JSON (`model_candidates`)
- offline replay of the native-like live chain

against reviewed racket/table markers. The output is meant to show what the app
actually found, what it missed, and where close table/racket events need special
handling before a future `spel_retro_audio` model is trained.

Run:
  python skills/pingis-audio-classification/scripts/build_playing_retro_candidate_report.py
"""

from __future__ import annotations

import argparse
import csv
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import pandas as pd

from preprocess_audio import TARGET_SR, extract_features, load_audio
from replay_live_bounce import (
    DEFAULT_CONTACT_MODEL_DIR,
    OUT_DIR,
    RAW_DIR,
    REPLAY_CONFIGS,
    ReplayConfig,
    load_model_bundle,
    qualified_prediction,
    resolve_wav_path,
    simulate_native_candidates,
)

ROOT_DIR = Path(__file__).resolve().parents[3]
DEFAULT_REPORT_FOUR_CLASS_MODEL_DIR = (
    ROOT_DIR
    / "data"
    / "audio"
    / "models"
    / "audio_4class_2026-05-28_tomas_stiga_C_hybrid_100_200_60_140_gap300"
)

DEFAULT_SESSION_IDS = [
    "audio_session_2026-05-28_002",
    "audio_session_2026-05-29_001",
    "audio_session_2026-05-29_002",
]

SKIPPED_STATUSES = {"pending", "deleted", "filtered"}
MATCH_TOLERANCE_MS = 140
CLOSE_EVENT_MS = 300
DEFAULT_ROWS_CSV = OUT_DIR / "playing_retro_candidate_peak_rows.csv"
DEFAULT_SUMMARY_CSV = OUT_DIR / "playing_retro_candidate_peak_summary.csv"
DEFAULT_REPORT_MD = OUT_DIR / "playing_retro_candidate_peak_report.md"


@dataclass(frozen=True)
class TruthMarker:
    marker_id: str
    timestamp_ms: int
    truth_kind: str
    review_status: str
    marker_source: str
    nearest_prev_ms: int | None
    nearest_next_ms: int | None
    nearest_prev_kind: str
    nearest_next_kind: str

    @property
    def nearest_neighbor_ms(self) -> int | None:
        values = [value for value in (self.nearest_prev_ms, self.nearest_next_ms) if value is not None]
        return min(values) if values else None

    @property
    def close_event_bucket(self) -> str:
        value = self.nearest_neighbor_ms
        if value is None:
            return "isolated"
        if value < 80:
            return "under_80ms"
        if value < 120:
            return "80_119ms"
        if value < 180:
            return "120_179ms"
        if value <= CLOSE_EVENT_MS:
            return "180_300ms"
        return "isolated"

    @property
    def neighbor_sequence(self) -> str:
        parts = []
        if self.nearest_prev_kind:
            parts.append(self.nearest_prev_kind)
        parts.append(self.truth_kind)
        if self.nearest_next_kind:
            parts.append(self.nearest_next_kind)
        return ">".join(parts)


@dataclass
class CandidatePeak:
    candidate_source: str
    source_config: str
    candidate_id: str
    timestamp_ms: int
    predicted_kind: str
    predicted_label: str
    confidence: float | str
    surface_label: str
    surface_confidence: float | str
    candidate_status: str
    reject_reason: str
    review_relevant: bool | str
    detection_mode: str
    detection_config_id: str
    rms: float | str = ""
    ball_ratio: float | str = ""
    flatness: float | str = ""

    @property
    def detectable(self) -> bool:
        if self.candidate_source == "app_saved":
            return self.candidate_status == "review_relevant"
        return self.candidate_status == "accepted_counted"

    @property
    def matchable(self) -> bool:
        if self.candidate_source == "app_saved":
            return True
        return self.candidate_status != "native_rejected"


def normalize_predicted_kind(label: str, surface_label: str = "", class_label: str = "") -> str:
    label = str(label or "")
    surface_label = str(surface_label or "")
    class_label = str(class_label or "")
    if label == "racket_contact" or surface_label == "racket_bounce" or class_label == "racket_bounce":
        return "racket_contact"
    if surface_label == "table_bounce" or class_label == "table_bounce":
        return "table_bounce"
    if surface_label == "floor_bounce" or class_label == "floor_bounce":
        return "floor_bounce"
    if surface_label == "noise" or class_label == "noise":
        return "noise"
    if label == "not_racket_contact":
        return "not_racket_contact"
    return label or surface_label or class_label or "unknown"


def truth_kind_for_marker(marker: dict[str, Any]) -> str:
    final_label = str(marker.get("final_label") or "")
    class_label = str(
        marker.get("class_label")
        or marker.get("not_racket_kind")
        or marker.get("surface_label")
        or ""
    )
    if final_label == "racket_contact":
        return "racket_contact"
    if final_label == "not_racket_contact" and class_label == "table_bounce":
        return "table_bounce"
    if final_label == "not_racket_contact" and class_label:
        return class_label
    return final_label


def is_trainable_audio_truth(marker: dict[str, Any]) -> bool:
    status = str(marker.get("review_status") or "confirmed")
    if status in SKIPPED_STATUSES:
        return False
    if str(marker.get("final_label") or "") == "ignore":
        return False
    return truth_kind_for_marker(marker) in {"racket_contact", "table_bounce"}


def build_truth_markers(markers: list[dict[str, Any]]) -> list[TruthMarker]:
    raw = []
    for index, marker in enumerate(markers):
        if not is_trainable_audio_truth(marker):
            continue
        raw.append({
            "marker_id": str(marker.get("id") or f"marker_{index}"),
            "timestamp_ms": int(round(float(marker.get("timestamp_ms") or 0))),
            "truth_kind": truth_kind_for_marker(marker),
            "review_status": str(marker.get("review_status") or "confirmed"),
            "marker_source": str(marker.get("source") or ""),
        })
    raw.sort(key=lambda item: item["timestamp_ms"])

    result: list[TruthMarker] = []
    for index, marker in enumerate(raw):
        prev_marker = raw[index - 1] if index > 0 else None
        next_marker = raw[index + 1] if index + 1 < len(raw) else None
        result.append(TruthMarker(
            marker_id=marker["marker_id"],
            timestamp_ms=marker["timestamp_ms"],
            truth_kind=marker["truth_kind"],
            review_status=marker["review_status"],
            marker_source=marker["marker_source"],
            nearest_prev_ms=(
                marker["timestamp_ms"] - prev_marker["timestamp_ms"] if prev_marker else None
            ),
            nearest_next_ms=(
                next_marker["timestamp_ms"] - marker["timestamp_ms"] if next_marker else None
            ),
            nearest_prev_kind=prev_marker["truth_kind"] if prev_marker else "",
            nearest_next_kind=next_marker["truth_kind"] if next_marker else "",
        ))
    return result


def is_dense_play_event(event: dict[str, Any]) -> bool:
    fields = [
        event.get("scenario_id", ""),
        event.get("background_condition", ""),
        event.get("bounce_context", ""),
        event.get("evaluation_bucket", ""),
        event.get("scenario", ""),
    ]
    joined = " ".join(str(item).lower() for item in fields)
    return "playing_dense" in joined or "stiga" in joined or "racket_table" in joined


def event_bucket(event: dict[str, Any]) -> str:
    return str(
        event.get("evaluation_bucket")
        or event.get("background_condition")
        or event.get("scenario_id")
        or event.get("scenario")
        or "unknown"
    )


def candidate_peaks_from_app(event: dict[str, Any]) -> list[CandidatePeak]:
    peaks: list[CandidatePeak] = []
    for index, candidate in enumerate(event.get("model_candidates") or []):
        timestamp_ms = int(round(float(candidate.get("timestamp_ms") or 0)))
        surface_label = str(candidate.get("surface_label") or "")
        class_label = str(candidate.get("class_label") or "")
        predicted_label = str(candidate.get("suggested_label") or "")
        review_relevant = bool(candidate.get("review_relevant"))
        peaks.append(CandidatePeak(
            candidate_source="app_saved",
            source_config=str(candidate.get("detection_config_id") or "saved_model_candidates"),
            candidate_id=str(candidate.get("id") or f"app_candidate_{index}_{timestamp_ms}"),
            timestamp_ms=timestamp_ms,
            predicted_kind=normalize_predicted_kind(predicted_label, surface_label, class_label),
            predicted_label=predicted_label,
            confidence=candidate.get("contact_confidence", ""),
            surface_label=surface_label,
            surface_confidence=candidate.get("surface_confidence", ""),
            candidate_status="review_relevant" if review_relevant else "analysis_only",
            reject_reason=str(candidate.get("ignored_reason") or ""),
            review_relevant=review_relevant,
            detection_mode=str(candidate.get("detection_mode") or ""),
            detection_config_id=str(candidate.get("detection_config_id") or ""),
        ))
    return peaks


def candidate_peaks_from_replay(
    y: Any,
    configs: list[ReplayConfig],
    four_class: Any,
    contact: Any,
) -> dict[str, list[CandidatePeak]]:
    by_config: dict[str, list[CandidatePeak]] = {}
    feature_cache: dict[int, dict[str, float]] = {}
    for config in configs:
        peaks: list[CandidatePeak] = []
        last_counted_ms: int | None = None
        active_group_start_ms: int | None = None
        for index, native_candidate in enumerate(simulate_native_candidates(y, config)):
            event_ms = int(native_candidate["event_ms"])
            reject_reason = str(native_candidate.get("native_reject_reason") or "")
            predicted_label = ""
            confidence: float | str = ""
            surface_label = ""
            surface_confidence: float | str = ""
            candidate_status = "native_rejected" if reject_reason else "model_pending"
            if not reject_reason:
                onset_sample = int(native_candidate["onset_sample"])
                features = feature_cache.get(onset_sample)
                if features is None:
                    features = extract_features(native_candidate["clip"], TARGET_SR)
                    feature_cache[onset_sample] = features
                qualified, predicted_label, confidence, reject_reason, surface_label, surface_confidence = (
                    qualified_prediction(features, config, four_class, contact)
                )
                if not qualified:
                    candidate_status = "model_rejected"
                elif last_counted_ms is not None and event_ms - last_counted_ms <= config.merge_ms:
                    candidate_status = "timing_rejected"
                    reject_reason = "merge_window"
                elif active_group_start_ms is not None and event_ms - active_group_start_ms <= config.group_ms:
                    candidate_status = "timing_rejected"
                    reject_reason = "group_window"
                else:
                    candidate_status = "accepted_counted"
                    reject_reason = ""
                    active_group_start_ms = event_ms
                    last_counted_ms = event_ms

            peaks.append(CandidatePeak(
                candidate_source="replay",
                source_config=config.name,
                candidate_id=f"replay_{config.name}_{index}_{event_ms}",
                timestamp_ms=event_ms,
                predicted_kind=normalize_predicted_kind(str(predicted_label), str(surface_label)),
                predicted_label=str(predicted_label),
                confidence=confidence,
                surface_label=str(surface_label),
                surface_confidence=surface_confidence,
                candidate_status=candidate_status,
                reject_reason=reject_reason,
                review_relevant=candidate_status == "accepted_counted",
                detection_mode=config.mode,
                detection_config_id=config.name,
                rms=round(float(native_candidate.get("rms") or 0), 6),
                ball_ratio=round(float(native_candidate.get("ball_ratio") or 0), 6),
                flatness=round(float(native_candidate.get("flatness") or 0), 6),
            ))
        by_config[config.name] = peaks
    return by_config


def nearest_truth(candidate: CandidatePeak, truths: list[TruthMarker]) -> tuple[TruthMarker | None, int | None]:
    best_truth: TruthMarker | None = None
    best_delta: int | None = None
    for truth in truths:
        delta = int(candidate.timestamp_ms - truth.timestamp_ms)
        if best_delta is None or abs(delta) < abs(best_delta):
            best_truth = truth
            best_delta = delta
    return best_truth, best_delta


def nearest_candidate(truth: TruthMarker, candidates: list[CandidatePeak]) -> tuple[CandidatePeak | None, int | None]:
    best_candidate: CandidatePeak | None = None
    best_delta: int | None = None
    for candidate in candidates:
        delta = int(candidate.timestamp_ms - truth.timestamp_ms)
        if best_delta is None or abs(delta) < abs(best_delta):
            best_candidate = candidate
            best_delta = delta
    return best_candidate, best_delta


def match_detectable_candidates(
    candidates: list[CandidatePeak],
    truths: list[TruthMarker],
    tolerance_ms: int,
) -> dict[str, str]:
    pairs: list[tuple[int, str, str]] = []
    for candidate in candidates:
        if not candidate.matchable:
            continue
        for truth in truths:
            delta = abs(candidate.timestamp_ms - truth.timestamp_ms)
            if delta <= tolerance_ms:
                pairs.append((delta, candidate.candidate_id, truth.marker_id))
    pairs.sort(key=lambda item: item[0])

    matched_candidates: set[str] = set()
    matched_truths: set[str] = set()
    result: dict[str, str] = {}
    for _delta, candidate_id, marker_id in pairs:
        if candidate_id in matched_candidates or marker_id in matched_truths:
            continue
        matched_candidates.add(candidate_id)
        matched_truths.add(marker_id)
        result[candidate_id] = marker_id
    return result


def outcome_for(candidate: CandidatePeak, matched_truth: TruthMarker | None, nearest_delta: int | None) -> str:
    if matched_truth:
        if candidate.predicted_kind == matched_truth.truth_kind:
            return "matched_racket" if matched_truth.truth_kind == "racket_contact" else "matched_table"
        if candidate.predicted_kind == "racket_contact" and matched_truth.truth_kind == "table_bounce":
            return "wrong_class_table_as_racket"
        if candidate.predicted_kind == "table_bounce" and matched_truth.truth_kind == "racket_contact":
            return "wrong_class_racket_as_table"
        return "wrong_class_other"
    if candidate.matchable:
        return "false_positive"
    if nearest_delta is not None and abs(nearest_delta) <= MATCH_TOLERANCE_MS:
        return "near_truth_rejected_or_hidden"
    return "background_rejected_or_hidden"


def base_context(session_path: Path, event_index: int, event: dict[str, Any]) -> dict[str, Any]:
    return {
        "session_id": session_path.stem,
        "event_index": event_index,
        "wav_filename": event.get("wav_filename", ""),
        "scenario": event.get("scenario", ""),
        "scenario_id": event.get("scenario_id", ""),
        "evaluation_bucket": event_bucket(event),
        "background_condition": event.get("background_condition", ""),
        "bounce_context": event.get("bounce_context", ""),
        "impact_style": event.get("impact_style") or (event.get("audio_training_intake") or {}).get("impact_style", ""),
        "ordinary_bounce_policy": event.get("ordinary_bounce_policy") or (event.get("audio_training_intake") or {}).get("ordinary_bounce_policy", ""),
        "take_index": event.get("take_index", ""),
    }


def row_for_candidate(
    context: dict[str, Any],
    candidate: CandidatePeak,
    truth: TruthMarker | None,
    nearest: TruthMarker | None,
    nearest_delta: int | None,
    outcome: str,
) -> dict[str, Any]:
    truth_for_spacing = truth or nearest
    return {
        **context,
        "row_type": "candidate",
        "candidate_source": candidate.candidate_source,
        "source_config": candidate.source_config,
        "candidate_id": candidate.candidate_id,
        "candidate_timestamp_ms": candidate.timestamp_ms,
        "candidate_status": candidate.candidate_status,
        "candidate_detectable": candidate.detectable,
        "candidate_matchable": candidate.matchable,
        "candidate_predicted_kind": candidate.predicted_kind,
        "candidate_predicted_label": candidate.predicted_label,
        "candidate_confidence": candidate.confidence,
        "candidate_surface_label": candidate.surface_label,
        "candidate_surface_confidence": candidate.surface_confidence,
        "reject_reason": candidate.reject_reason,
        "review_relevant": candidate.review_relevant,
        "detection_mode": candidate.detection_mode,
        "detection_config_id": candidate.detection_config_id,
        "rms": candidate.rms,
        "ball_ratio": candidate.ball_ratio,
        "flatness": candidate.flatness,
        "match_outcome": outcome,
        "matched_marker_id": truth.marker_id if truth else "",
        "matched_truth_kind": truth.truth_kind if truth else "",
        "matched_truth_timestamp_ms": truth.timestamp_ms if truth else "",
        "candidate_to_truth_offset_ms": (
            candidate.timestamp_ms - truth.timestamp_ms if truth else ""
        ),
        "nearest_truth_marker_id": nearest.marker_id if nearest else "",
        "nearest_truth_kind": nearest.truth_kind if nearest else "",
        "nearest_truth_timestamp_ms": nearest.timestamp_ms if nearest else "",
        "nearest_truth_offset_ms": nearest_delta if nearest_delta is not None else "",
        "truth_nearest_neighbor_ms": truth_for_spacing.nearest_neighbor_ms if truth_for_spacing else "",
        "close_event_bucket": truth_for_spacing.close_event_bucket if truth_for_spacing else "",
        "neighbor_sequence": truth_for_spacing.neighbor_sequence if truth_for_spacing else "",
    }


def row_for_missed_marker(
    context: dict[str, Any],
    source: str,
    config: str,
    truth: TruthMarker,
    nearest_peak: CandidatePeak | None,
    nearest_delta: int | None,
) -> dict[str, Any]:
    return {
        **context,
        "row_type": "missed_marker",
        "candidate_source": source,
        "source_config": config,
        "candidate_id": "",
        "candidate_timestamp_ms": "",
        "candidate_status": "",
        "candidate_detectable": False,
        "candidate_matchable": False,
        "candidate_predicted_kind": "",
        "candidate_predicted_label": "",
        "candidate_confidence": "",
        "candidate_surface_label": "",
        "candidate_surface_confidence": "",
        "reject_reason": "",
        "review_relevant": "",
        "detection_mode": "",
        "detection_config_id": "",
        "rms": "",
        "ball_ratio": "",
        "flatness": "",
        "match_outcome": "missed_racket" if truth.truth_kind == "racket_contact" else "missed_table",
        "matched_marker_id": truth.marker_id,
        "matched_truth_kind": truth.truth_kind,
        "matched_truth_timestamp_ms": truth.timestamp_ms,
        "candidate_to_truth_offset_ms": "",
        "nearest_truth_marker_id": truth.marker_id,
        "nearest_truth_kind": truth.truth_kind,
        "nearest_truth_timestamp_ms": truth.timestamp_ms,
        "nearest_truth_offset_ms": 0,
        "truth_nearest_neighbor_ms": truth.nearest_neighbor_ms if truth.nearest_neighbor_ms is not None else "",
        "close_event_bucket": truth.close_event_bucket,
        "neighbor_sequence": truth.neighbor_sequence,
        "nearest_candidate_id": nearest_peak.candidate_id if nearest_peak else "",
        "nearest_candidate_timestamp_ms": nearest_peak.timestamp_ms if nearest_peak else "",
        "nearest_candidate_status": nearest_peak.candidate_status if nearest_peak else "",
        "nearest_candidate_predicted_kind": nearest_peak.predicted_kind if nearest_peak else "",
        "nearest_candidate_offset_ms": nearest_delta if nearest_delta is not None else "",
    }


def build_rows_for_group(
    context: dict[str, Any],
    source: str,
    config: str,
    candidates: list[CandidatePeak],
    truths: list[TruthMarker],
    tolerance_ms: int,
) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    candidate_to_truth = match_detectable_candidates(candidates, truths, tolerance_ms)
    truths_by_id = {truth.marker_id: truth for truth in truths}
    matched_truth_ids = set(candidate_to_truth.values())

    for candidate in candidates:
        matched_truth = truths_by_id.get(candidate_to_truth.get(candidate.candidate_id, ""))
        near_truth, near_delta = nearest_truth(candidate, truths)
        rows.append(row_for_candidate(
            context=context,
            candidate=candidate,
            truth=matched_truth,
            nearest=near_truth,
            nearest_delta=near_delta,
            outcome=outcome_for(candidate, matched_truth, near_delta),
        ))

    for truth in truths:
        if truth.marker_id in matched_truth_ids:
            continue
        nearest_peak, nearest_delta = nearest_candidate(truth, candidates)
        rows.append(row_for_missed_marker(
            context=context,
            source=source,
            config=config,
            truth=truth,
            nearest_peak=nearest_peak,
            nearest_delta=nearest_delta,
        ))
    return rows


def selected_replay_configs(names: list[str]) -> list[ReplayConfig]:
    if not names:
        return list(REPLAY_CONFIGS)
    by_name = {config.name: config for config in REPLAY_CONFIGS}
    missing = [name for name in names if name not in by_name]
    if missing:
        raise ValueError(f"Unknown replay config(s): {', '.join(missing)}")
    return [by_name[name] for name in names]


def default_session_paths() -> list[Path]:
    paths = []
    for session_id in DEFAULT_SESSION_IDS:
        path = RAW_DIR / f"{session_id}.json"
        if path.exists():
            paths.append(path)
    return paths


def all_dense_session_paths() -> list[Path]:
    paths = []
    for path in sorted(RAW_DIR.glob("audio_session_*.json")):
        data = json.loads(path.read_text(encoding="utf-8"))
        if any(is_dense_play_event(event) for event in data.get("events") or []):
            paths.append(path)
    return paths


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames: list[str] = []
    seen = set()
    for row in rows:
        for key in row:
            if key not in seen:
                seen.add(key)
                fieldnames.append(key)
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def build_summary(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    if not rows:
        return []
    df = pd.DataFrame(rows)
    group_cols = ["candidate_source", "source_config", "session_id", "evaluation_bucket"]
    summary_rows: list[dict[str, Any]] = []
    for keys, group in df.groupby(group_cols, dropna=False):
        item = dict(zip(group_cols, keys))
        outcomes = group["match_outcome"].value_counts().to_dict()
        item.update({
            "candidate_rows": int((group["row_type"] == "candidate").sum()),
            "detectable_candidates": int(group.get("candidate_detectable", pd.Series(dtype=bool)).fillna(False).astype(bool).sum()),
            "matchable_candidates": int(group.get("candidate_matchable", pd.Series(dtype=bool)).fillna(False).astype(bool).sum()),
            "matched_racket": int(outcomes.get("matched_racket", 0)),
            "matched_table": int(outcomes.get("matched_table", 0)),
            "wrong_class_table_as_racket": int(outcomes.get("wrong_class_table_as_racket", 0)),
            "wrong_class_racket_as_table": int(outcomes.get("wrong_class_racket_as_table", 0)),
            "wrong_class_other": int(outcomes.get("wrong_class_other", 0)),
            "false_positive": int(outcomes.get("false_positive", 0)),
            "missed_racket": int(outcomes.get("missed_racket", 0)),
            "missed_table": int(outcomes.get("missed_table", 0)),
            "near_truth_rejected_or_hidden": int(outcomes.get("near_truth_rejected_or_hidden", 0)),
            "background_rejected_or_hidden": int(outcomes.get("background_rejected_or_hidden", 0)),
            "close_event_rows": int((group["close_event_bucket"] != "isolated").sum()),
        })
        summary_rows.append(item)
    return summary_rows


def write_markdown_report(path: Path, rows: list[dict[str, Any]], summary: list[dict[str, Any]]) -> None:
    summary_df = pd.DataFrame(summary)
    lines = [
        "# Playing Retro Candidate Peak Report",
        "",
        "Generated by `build_playing_retro_candidate_report.py`.",
        "",
        "This report is diagnostic only: no model was trained, exported, or installed.",
        "",
        "## Outputs",
        "",
        f"- Row CSV: `{DEFAULT_ROWS_CSV.as_posix()}`",
        f"- Summary CSV: `{DEFAULT_SUMMARY_CSV.as_posix()}`",
        "",
    ]
    if not summary_df.empty:
        lines.extend([
            "## Summary",
            "",
            "| Source | Config | Session | Bucket | Racket TP | Table TP | FP | Missed Racket | Missed Table | Wrong Class | Close Rows |",
            "|---|---|---|---|---:|---:|---:|---:|---:|---:|---:|",
        ])
        for row in summary:
            wrong = int(row["wrong_class_table_as_racket"]) + int(row["wrong_class_racket_as_table"]) + int(row["wrong_class_other"])
            lines.append(
                f"| {row['candidate_source']} | {row['source_config']} | {row['session_id']} | "
                f"{row['evaluation_bucket']} | {row['matched_racket']} | {row['matched_table']} | "
                f"{row['false_positive']} | {row['missed_racket']} | {row['missed_table']} | "
                f"{wrong} | {row['close_event_rows']} |"
            )

    rows_df = pd.DataFrame(rows)
    if not rows_df.empty and "audio_session_2026-05-29_002" in set(rows_df["session_id"]):
        target = rows_df[rows_df["session_id"] == "audio_session_2026-05-29_002"]
        close = target[target["close_event_bucket"] != "isolated"]
        lines.extend([
            "",
            "## Manual Check: audio_session_2026-05-29_002",
            "",
            f"- Rows: {len(target)}",
            f"- Close-event rows: {len(close)}",
            f"- Outcomes: `{target['match_outcome'].value_counts().to_dict()}`",
        ])
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build a candidate-centered playing retro audio report.")
    parser.add_argument("--session-json", action="append", default=[], help="Audio session JSON path. Repeatable.")
    parser.add_argument("--all-dense", action="store_true", help="Use all dense playing sessions found in data/audio/raw.")
    parser.add_argument("--candidate-source", choices=["app", "replay", "both"], default="both")
    parser.add_argument("--replay-config", action="append", default=[], help="Replay config name. Repeatable.")
    parser.add_argument("--four-class-model-dir", default=str(DEFAULT_REPORT_FOUR_CLASS_MODEL_DIR))
    parser.add_argument("--contact-model-dir", default=str(DEFAULT_CONTACT_MODEL_DIR))
    parser.add_argument("--match-tolerance-ms", type=int, default=MATCH_TOLERANCE_MS)
    parser.add_argument("--rows-csv", default=str(DEFAULT_ROWS_CSV))
    parser.add_argument("--summary-csv", default=str(DEFAULT_SUMMARY_CSV))
    parser.add_argument("--report-md", default=str(DEFAULT_REPORT_MD))
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.session_json:
        session_paths = [Path(path) for path in args.session_json]
    elif args.all_dense:
        session_paths = all_dense_session_paths()
    else:
        session_paths = default_session_paths()

    if not session_paths:
        raise SystemExit("No session JSON files selected.")

    replay_configs = selected_replay_configs(args.replay_config)
    four_class = None
    contact = None
    if args.candidate_source in {"replay", "both"}:
        four_class = load_model_bundle(Path(args.four_class_model_dir), "audio")
        contact_dir = Path(args.contact_model_dir)
        contact = load_model_bundle(contact_dir, "audio_contact") if contact_dir.exists() else None

    rows: list[dict[str, Any]] = []
    for session_path in session_paths:
        data = json.loads(session_path.read_text(encoding="utf-8"))
        for event_index, event in enumerate(data.get("events") or []):
            if args.all_dense and not is_dense_play_event(event):
                continue
            markers = (event.get("review") or {}).get("markers") or []
            truths = build_truth_markers(markers)
            if not truths:
                continue
            context = base_context(session_path, event_index, event)

            if args.candidate_source in {"app", "both"}:
                app_candidates = candidate_peaks_from_app(event)
                app_config_names = sorted({candidate.source_config for candidate in app_candidates}) or ["saved_model_candidates"]
                for config_name in app_config_names:
                    config_candidates = [
                        candidate for candidate in app_candidates if candidate.source_config == config_name
                    ]
                    rows.extend(build_rows_for_group(
                        context=context,
                        source="app_saved",
                        config=config_name,
                        candidates=config_candidates,
                        truths=truths,
                        tolerance_ms=args.match_tolerance_ms,
                    ))

            if args.candidate_source in {"replay", "both"}:
                wav_path = resolve_wav_path(session_path, event)
                if not wav_path:
                    continue
                y, _sr = load_audio(str(wav_path))
                replay_by_config = candidate_peaks_from_replay(y, replay_configs, four_class, contact)
                for config_name, replay_candidates in replay_by_config.items():
                    rows.extend(build_rows_for_group(
                        context=context,
                        source="replay",
                        config=config_name,
                        candidates=replay_candidates,
                        truths=truths,
                        tolerance_ms=args.match_tolerance_ms,
                    ))

    if not rows:
        raise SystemExit("No report rows produced.")

    summary = build_summary(rows)
    rows_csv = Path(args.rows_csv)
    summary_csv = Path(args.summary_csv)
    report_md = Path(args.report_md)
    write_csv(rows_csv, rows)
    write_csv(summary_csv, summary)
    write_markdown_report(report_md, rows, summary)

    print(f"Wrote {rows_csv}")
    print(f"Wrote {summary_csv}")
    print(f"Wrote {report_md}")
    print("source,config,session,bucket,racket_tp,table_tp,fp,missed_racket,missed_table,wrong_class,close_rows")
    for row in summary:
        wrong = int(row["wrong_class_table_as_racket"]) + int(row["wrong_class_racket_as_table"]) + int(row["wrong_class_other"])
        print(
            f"{row['candidate_source']},{row['source_config']},{row['session_id']},"
            f"{row['evaluation_bucket']},{row['matched_racket']},{row['matched_table']},"
            f"{row['false_positive']},{row['missed_racket']},{row['missed_table']},"
            f"{wrong},{row['close_event_rows']}"
        )


if __name__ == "__main__":
    main()
