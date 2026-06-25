#!/usr/bin/env python3
"""Retrain the local T0030 playing-retro audio candidate.

This wraps the proven T0026 training pipeline with a new focus session and
artifact names. It trains local joblib artifacts only; it does not export
Collector JSON, build an APK, or change studs_live.
"""

from __future__ import annotations

import json
from pathlib import Path

import train_playing_retro_audio_t0026 as base


FOCUS_SESSION = "audio_session_2026-06-04_006"
MODEL_ID = "playing_retro_audio_rf_v2026_06_04_t0030_multi_window_context"

base.FOCUS_SESSION = FOCUS_SESSION
base.MODEL_ID = MODEL_ID
base.CANDIDATE_ROWS_CSV = base.OUT_DIR / "playing_retro_audio_candidate_rows_t0030_2026_06_04_006.csv"
base.DATASET_CSV = base.OUT_DIR / "playing_retro_audio_multi_window_dataset_t0030_2026_06_04_006.csv"
base.EVAL_CSV = base.EVAL_DIR / "playing_retro_audio_t0030_retrain_eval.csv"
base.PREDICTIONS_CSV = base.EVAL_DIR / "playing_retro_audio_t0030_retrain_predictions.csv"
base.REPORT_MD = base.EVAL_DIR / "playing_retro_audio_t0030_retrain_report.md"
base.MODEL_DIR = base.MODEL_ROOT / MODEL_ID
base.DEFAULT_HOLDOUT_SESSIONS = [
    "audio_session_2026-05-28_002",
    "audio_session_2026-05-29_001",
    "audio_session_2026-05-29_002",
    "audio_session_2026-06-03_005",
    "audio_session_2026-06-04_001",
    FOCUS_SESSION,
]

# T0028 shipped the T0026 candidate. Use its leave-one-session-out numbers as
# the old-session safety reference when selecting a T0030 variant.
base.REFERENCE_T0022 = {
    "audio_session_2026-05-28_002": {
        "accuracy": 0.9371980676328503,
        "racket_contact_recall": 0.9253731343283582,
        "table_bounce_recall": 0.9594594594594594,
        "non_target_recall": 0.9242424242424242,
    },
    "audio_session_2026-05-29_001": {
        "accuracy": 0.9207920792079208,
        "racket_contact_recall": 0.9393939393939394,
        "table_bounce_recall": 0.9583333333333334,
        "non_target_recall": 0.859375,
    },
    "audio_session_2026-05-29_002": {
        "accuracy": 0.8995983935742972,
        "racket_contact_recall": 0.8867924528301887,
        "table_bounce_recall": 0.9159663865546218,
        "non_target_recall": 0.875,
    },
    "audio_session_2026-06-03_005": {
        "accuracy": 0.847972972972973,
        "racket_contact_recall": 0.8952380952380953,
        "table_bounce_recall": 0.9158878504672897,
        "non_target_recall": 0.7023809523809523,
    },
    "audio_session_2026-06-04_001": {
        "accuracy": 0.8540145985401459,
        "racket_contact_recall": 0.8125,
        "table_bounce_recall": 0.9058823529411765,
        "non_target_recall": 0.8440366972477065,
    },
}


def patch_report_json(path: Path) -> None:
    report = json.loads(path.read_text(encoding="utf-8"))
    report["ticket"] = "T0030"
    report["recommendation"] = "proceed_to_t0030_replay_tune_before_export"
    report["training_choice"] = "fresh_multi_window_context_dataset_from_historical_playing_plus_2026_06_04_006"
    report["reference_model"] = "playing_retro_audio_rf_v2026_06_04_t0026_multi_window_context"
    path.write_text(json.dumps(report, indent=2), encoding="utf-8")


def patch_report_md(path: Path) -> None:
    text = path.read_text(encoding="utf-8")
    replacements = {
        "T0026": "T0030",
        "T0027": "T0030 replay",
        "T0024": "T0028",
        "06-04 was added": "06-04_006 was added",
        "2026-06-04 Held-Out Check": "2026-06-04_006 Held-Out Check",
        "2026-06-04 Focus Check": "2026-06-04_006 Focus Check",
        "Final 06-04 rows": "Final 06-04_006 rows",
        "reviewed 2026-06-04 clip": "reviewed 2026-06-04_006 clip",
    }
    for before, after in replacements.items():
        text = text.replace(before, after)
    path.write_text(text, encoding="utf-8")


def main() -> None:
    base.main()
    patch_report_json(base.MODEL_DIR / "report.json")
    patch_report_md(base.REPORT_MD)
    print(f"patched T0030 report metadata for {MODEL_ID}")


if __name__ == "__main__":
    main()
