"""Compare the deployed Fable gate with the selected HF candidate gate."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import precision_recall_fscore_support

from .audio_io import load_mono_audio
from .candidate_labels import assign_candidates, canonical_four_class
from .data_sources import (
    AudioSession,
    deduplicate_sessions,
    discover_legacy_sessions,
    discover_stiga_sessions,
)
from .fable_baseline import (
    FableHgbRuntime,
    detect_deployed_fable_candidates,
    extract_and_predict_fable,
    extract_fable_clip,
    replay_deployed_fable_timing,
)
from .hf_gate import TARGET_SAMPLE_RATE


RACKET_LABEL = "racket_bounce"


def _parse_legacy_root(value: str) -> tuple[str, Path]:
    if "=" not in value:
        raise argparse.ArgumentTypeError("legacy root must use DEVICE_HINT=PATH")
    device, path = value.split("=", 1)
    return device.strip(), Path(path.strip())


def _base_session_id(session_id: str) -> str:
    return str(session_id).split(":", 1)[0]


def _split_map(path: Path) -> dict[str, str]:
    frame = pd.read_csv(path, usecols=["base_session_id", "split"])
    output: dict[str, str] = {}
    priority = {"external_device": 3, "validation": 2, "train": 1, "not_trainable": 0}
    for base_session_id, rows in frame.groupby("base_session_id", sort=False):
        choices = [str(value) for value in rows["split"].dropna().unique()]
        output[str(base_session_id)] = max(choices, key=lambda value: priority.get(value, -1))
    return output


def _discover_sessions(args: argparse.Namespace) -> list[AudioSession]:
    sessions: list[AudioSession] = []
    for root in args.stiga_root:
        sessions.extend(discover_stiga_sessions(root))
    for device_hint, root in args.legacy_root:
        sessions.extend(
            discover_legacy_sessions(
                root,
                device_hint=device_hint,
                source_name=f"legacy:{device_hint}",
            )
        )
    return deduplicate_sessions(sessions)


def _candidate_assignments(session: AudioSession, candidates: list[dict]) -> list:
    return assign_candidates(
        [float(candidate["onset_ms"]) for candidate in candidates],
        list(session.reviewed_events),
        review_complete=session.review_complete,
        explicit_negative_session=session.explicit_negative,
    )


def _hf_candidates(frame: pd.DataFrame) -> list[dict[str, float | int]]:
    if frame.empty:
        return []
    output: list[dict[str, float | int]] = []
    for row in frame.sort_values("candidate_index").itertuples(index=False):
        output.append(
            {
                "onset_sample": int(row.onset_sample),
                "onset_ms": float(row.onset_ms),
                "frame_rms": float(row.hf_rms),
                "background_rms": float(row.background_rms),
                "threshold": float(row.threshold),
            }
        )
    return output


def _score_candidates(frame: pd.DataFrame) -> dict:
    selected = frame[
        frame["disposition"].isin(("positive", "hard_negative"))
        & frame["label"].notna()
    ]
    if selected.empty:
        return {"rows": 0}
    truth = selected["label"].eq(RACKET_LABEL).to_numpy()
    predicted = selected["classifier_accepts_racket"].to_numpy(dtype=bool)
    precision, recall, f1, _ = precision_recall_fscore_support(
        truth,
        predicted,
        average="binary",
        zero_division=0,
    )
    return {
        "rows": int(len(selected)),
        "positives": int(truth.sum()),
        "predicted_positives": int(predicted.sum()),
        "precision": float(precision),
        "recall": float(recall),
        "f1": float(f1),
    }


def _count_summary(frame: pd.DataFrame, truth_column: str) -> dict:
    selected = frame[frame[truth_column].notna()].copy()
    if selected.empty:
        return {"sessions": 0}
    truth = selected[truth_column].to_numpy(dtype=np.float64)
    counted = selected["counted"].to_numpy(dtype=np.float64)
    error = counted - truth
    return {
        "sessions": int(len(selected)),
        "expected_total": int(truth.sum()),
        "counted_total": int(counted.sum()),
        "mae_per_session": float(np.abs(error).mean()),
        "mean_signed_error": float(error.mean()),
        "exact_session_count": int((error == 0).sum()),
    }


def _method_report(candidate_rows: pd.DataFrame, session_rows: pd.DataFrame) -> dict:
    reviewed = session_rows[session_rows["reviewed_truth"].notna()]
    count_only = session_rows[session_rows["expected_count"].notna()]
    explicit_negative = session_rows[session_rows["explicit_negative"]]
    reviewed_events = int(session_rows["reviewed_events"].sum())
    matched_events = int(session_rows["matched_events"].sum())
    duration_minutes = float(session_rows["duration_s"].sum()) / 60.0
    return {
        "gate": {
            "sessions": int(len(session_rows)),
            "reviewed_events": reviewed_events,
            "matched_events": matched_events,
            "reviewed_event_recall": matched_events / reviewed_events if reviewed_events else None,
            "candidates": int(len(candidate_rows)),
            "candidates_per_minute": len(candidate_rows) / duration_minutes if duration_minutes else 0.0,
        },
        "classifier": {
            split: _score_candidates(candidate_rows[candidate_rows["split"] == split])
            for split in ("validation", "external_device")
        },
        "final_counts": {
            "count_only_sessions": _count_summary(count_only, "expected_count"),
            "explicit_negative_sessions": _count_summary(explicit_negative, "negative_truth"),
            "review_complete_reviewed_sessions": _count_summary(reviewed, "reviewed_truth"),
        },
    }


def evaluate_method(
    method: str,
    sessions: list[AudioSession],
    split_by_base: dict[str, str],
    runtime: FableHgbRuntime,
    hf_manifest: pd.DataFrame,
) -> tuple[pd.DataFrame, pd.DataFrame, dict]:
    candidate_output: list[dict] = []
    session_output: list[dict] = []
    hf_by_session = {
        str(session_id): frame
        for session_id, frame in hf_manifest.groupby("session_id", sort=False)
    }

    for session_number, session in enumerate(sessions, start=1):
        pcm, _ = load_mono_audio(session.audio_path, TARGET_SAMPLE_RATE)
        if method == "default_fable":
            candidates = detect_deployed_fable_candidates(pcm, TARGET_SAMPLE_RATE)
        else:
            candidates = _hf_candidates(hf_by_session.get(session.session_id, pd.DataFrame()))
        assignments = _candidate_assignments(session, candidates)
        evaluated: list[dict] = []
        print(f"[{method} {session_number}/{len(sessions)}] {session.session_id}: {len(candidates)} candidates")
        for candidate, assignment in zip(candidates, assignments, strict=True):
            clip = extract_fable_clip(pcm, int(candidate["onset_sample"]))
            features, prediction = extract_and_predict_fable(runtime, clip)
            background_db = float(features.get("nr_bg_rms_db", -100.0))
            loud = background_db >= -36.0
            confidence_threshold = 0.85 if loud else 0.65
            row = {
                "method": method,
                "session_id": session.session_id,
                "base_session_id": _base_session_id(session.session_id),
                "source": session.source,
                "device_id": session.device_id,
                "scenario_id": session.scenario_id,
                "split": split_by_base.get(_base_session_id(session.session_id), "external_device"),
                "candidate_index": int(assignment.candidate_index),
                "onset_sample": int(candidate["onset_sample"]),
                "onset_ms": float(candidate["onset_ms"]),
                "frame_rms": float(candidate["frame_rms"]),
                "background_rms": float(candidate["background_rms"]),
                "disposition": assignment.disposition,
                "label": assignment.label,
                "distance_ms": assignment.distance_ms,
                "predicted_label": str(prediction["label"]),
                "confidence": float(prediction["confidence"]),
                "prob_racket_bounce": float(prediction["probabilities"].get(RACKET_LABEL, 0.0)),
                "background_db": background_db,
                "classifier_accepts_racket": (
                    prediction["label"] == RACKET_LABEL
                    and float(prediction["confidence"]) >= confidence_threshold
                ),
            }
            evaluated.append(row)

        decisions = replay_deployed_fable_timing(evaluated)
        for row, decision in zip(evaluated, decisions, strict=True):
            row["counted"] = decision.counted
            row["timing_reason"] = decision.reason
            candidate_output.append(row)

        trainable_events = [
            event
            for event in session.reviewed_events
            if event.trainable and canonical_four_class(event.label) is not None
        ]
        matched_events = {
            assignment.matched_event_index
            for assignment in assignments
            if assignment.disposition == "positive"
            and assignment.matched_event_index is not None
        }
        reviewed_racket_count = sum(
            canonical_four_class(event.label) == RACKET_LABEL for event in trainable_events
        )
        reviewed_truth = (
            reviewed_racket_count
            if session.review_complete and trainable_events
            else None
        )
        duration_s = len(pcm) / TARGET_SAMPLE_RATE
        session_output.append(
            {
                "method": method,
                "session_id": session.session_id,
                "base_session_id": _base_session_id(session.session_id),
                "source": session.source,
                "device_id": session.device_id,
                "scenario_id": session.scenario_id,
                "split": split_by_base.get(_base_session_id(session.session_id), "external_device"),
                "duration_s": duration_s,
                "candidate_count": len(candidates),
                "reviewed_events": len(trainable_events),
                "matched_events": len(matched_events),
                "review_complete": session.review_complete,
                "explicit_negative": session.explicit_negative,
                "expected_count": session.expected_bounce_count,
                "negative_truth": 0 if session.explicit_negative else None,
                "reviewed_truth": reviewed_truth,
                "counted": sum(decision.counted for decision in decisions),
            }
        )

    candidates_frame = pd.DataFrame(candidate_output)
    sessions_frame = pd.DataFrame(session_output)
    return candidates_frame, sessions_frame, _method_report(candidates_frame, sessions_frame)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--stiga-root", action="append", default=[], type=Path)
    parser.add_argument("--legacy-root", action="append", default=[], type=_parse_legacy_root)
    parser.add_argument("--split-csv", type=Path, required=True)
    parser.add_argument("--hf-manifest", type=Path, required=True)
    parser.add_argument(
        "--model-json",
        type=Path,
        default=Path("apps/collector/src/models/fable_audio_model.json"),
    )
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()

    split_by_base = _split_map(args.split_csv)
    eval_bases = {
        base_session_id
        for base_session_id, split in split_by_base.items()
        if split in {"validation", "external_device"}
    }
    sessions = [
        session
        for session in _discover_sessions(args)
        if session.source == "stiga_native_recorder"
        or _base_session_id(session.session_id) in eval_bases
    ]
    if not sessions:
        raise SystemExit("No evaluation sessions discovered")

    hf_manifest = pd.read_csv(args.hf_manifest)
    selected_session_ids = {session.session_id for session in sessions}
    hf_manifest = hf_manifest[
        hf_manifest["session_id"].isin(selected_session_ids)
    ].copy()
    runtime = FableHgbRuntime.from_path(args.model_json)
    args.output_dir.mkdir(parents=True, exist_ok=True)

    report: dict[str, dict] = {}
    for method in ("default_fable", "hf_gated_fable"):
        candidate_rows, session_rows, method_report = evaluate_method(
            method,
            sessions,
            split_by_base,
            runtime,
            hf_manifest,
        )
        candidate_rows.to_csv(args.output_dir / f"{method}_candidates.csv", index=False)
        session_rows.to_csv(args.output_dir / f"{method}_sessions.csv", index=False)
        report[method] = method_report

    report["contract"] = {
        "default_fable": "causal 1.5-7 kHz native gate -> 100 ms pre + 200 ms post full-band clip -> 83 features -> four-class HGB -> deployed timing",
        "hf_gated_fable": "selected 7 kHz HF gate -> 100 ms pre + 200 ms post full-band clip -> same 83 features/HGB/timing",
        "cnn_comparison": "HF CNN uses a separate 100 ms pre + 300 ms post 400 ms clip contract",
        "limitations": [
            "STIGA positive sessions contain expected counts but no event timestamps.",
            "Only review-complete timestamped sessions are used as reviewed final-count truth.",
            "Offline timing excludes JS queue staleness because no runtime queue exists.",
        ],
    }
    report_path = args.output_dir / "fable_baseline_comparison.json"
    report_path.write_text(json.dumps(report, indent=2, sort_keys=True), encoding="utf-8")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
