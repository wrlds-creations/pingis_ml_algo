"""
Replay the Collector live racket-bounce chain on saved WAV + review markers.

This is a diagnostic tool, not a trainer. It runs native-like onset gating,
the exported RF models and the same retrigger/group/merge rules so live
settings can be compared without standing with the phone.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import joblib
import numpy as np
import pandas as pd

from preprocess_audio import TARGET_SR, extract_features, load_audio


ROOT_DIR = Path(__file__).resolve().parents[3]
RAW_DIR = ROOT_DIR / "data" / "audio" / "raw"
OUT_DIR = ROOT_DIR / "data" / "audio" / "processed"
DEFAULT_OUT_CSV = OUT_DIR / "live_bounce_replay_report.csv"
DEFAULT_FOUR_CLASS_MODEL_DIR = (
    ROOT_DIR
    / "data"
    / "audio"
    / "models"
    / "audio_4class_2026-05-25_with_007_C_hybrid_100_200_60_140_gap300"
)
DEFAULT_CONTACT_MODEL_DIR = (
    ROOT_DIR
    / "data"
    / "audio"
    / "models"
    / "audio_contact_2026-05-25_with_007_C_hybrid_100_200_60_140_gap300"
)

FRAME_SIZE = 220
PRE_SAMPLES = 2_205
POST_SAMPLES = 4_410
CLIP_SAMPLES = PRE_SAMPLES + POST_SAMPLES
BG_FRAMES = 30
ABS_MIN_RMS = 0.003
SPECTRAL_FFT = 256
BALL_LO_HZ = 200.0
BALL_HI_HZ = 6000.0
MIN_BALL_RATIO = 0.55
MAX_FLATNESS = 0.6

SKIPPED_STATUSES = {"pending", "deleted", "filtered"}


@dataclass(frozen=True)
class ReplayConfig:
    name: str
    mode: str
    confidence: float
    onset_ratio: float
    retrigger_ms: int
    group_ms: int
    merge_ms: int


REPLAY_CONFIGS = [
    ReplayConfig("baseline_normal_4class_220_80_220", "four_class_only", 0.65, 1.5, 220, 80, 220),
    ReplayConfig("fast_4class_120_80_180", "four_class_only", 0.65, 1.5, 120, 80, 180),
    ReplayConfig("no_group_4class_220_0_180", "four_class_only", 0.65, 1.5, 220, 0, 180),
    ReplayConfig("sensitive_4class_220_80_180", "four_class_only", 0.50, 1.5, 220, 80, 180),
    ReplayConfig("strict_onset_4class_220_80_220", "four_class_only", 0.65, 2.5, 220, 80, 220),
    ReplayConfig("hybrid_normal_220_80_220", "hybrid", 0.65, 1.5, 220, 80, 220),
]


@dataclass
class ModelBundle:
    classifier: Any
    scaler: Any
    label_encoder: Any
    feature_cols: list[str]

    def predict(self, features: dict[str, float]) -> tuple[str, float, dict[str, float]]:
        row = pd.DataFrame([features])
        for column in self.feature_cols:
            if column not in row:
                row[column] = 0.0
        x = row[self.feature_cols].fillna(0.0).values.astype(np.float32)
        probs = self.classifier.predict_proba(self.scaler.transform(x))[0]
        labels = list(self.label_encoder.classes_)
        prob_map = {str(label): float(prob) for label, prob in zip(labels, probs)}
        best_idx = int(np.argmax(probs))
        label = str(labels[best_idx])
        return label, float(probs[best_idx]), prob_map


def load_model_bundle(model_dir: Path, prefix: str) -> ModelBundle:
    return ModelBundle(
        classifier=joblib.load(model_dir / f"{prefix}_rf_classifier.pkl"),
        scaler=joblib.load(model_dir / f"{prefix}_feature_scaler.pkl"),
        label_encoder=joblib.load(model_dir / f"{prefix}_label_encoder.pkl"),
        feature_cols=list(joblib.load(model_dir / f"{prefix}_feature_cols.pkl")),
    )


def compute_rms(frame: np.ndarray) -> float:
    if len(frame) == 0:
        return 0.0
    return float(np.sqrt(np.mean(np.square(frame, dtype=np.float64))))


def spectral_gate(frame: np.ndarray) -> tuple[bool, float, float]:
    work = np.zeros(SPECTRAL_FFT, dtype=np.float64)
    length = min(len(frame), SPECTRAL_FFT)
    if length > 1:
        work[:length] = frame[:length] * np.hanning(length)
    spectrum = np.fft.rfft(work)
    power = np.abs(spectrum) ** 2
    freqs = np.fft.rfftfreq(SPECTRAL_FFT, d=1.0 / TARGET_SR)
    valid = np.arange(len(power)) > 0
    total = float(np.sum(power[valid]))
    band = (freqs >= BALL_LO_HZ) & (freqs <= BALL_HI_HZ)
    ball = float(np.sum(power[band]))
    ball_ratio = ball / total if total > 0 else 0.0
    if total > 0 and ball_ratio < MIN_BALL_RATIO:
        return False, ball_ratio, 0.0
    values = power[valid] + 1e-10
    flatness = float(math.exp(float(np.mean(np.log(values)))) / float(np.mean(values))) if len(values) else 0.0
    if flatness > MAX_FLATNESS:
        return False, ball_ratio, flatness
    return True, ball_ratio, flatness


def extract_live_clip(y: np.ndarray, onset_sample: int) -> np.ndarray:
    clip = np.zeros(CLIP_SAMPLES, dtype=np.float32)
    start = onset_sample - PRE_SAMPLES
    end = onset_sample + POST_SAMPLES
    src_start = max(0, start)
    src_end = min(len(y), end)
    dst_start = src_start - start
    if src_end > src_start:
        clip[dst_start:dst_start + (src_end - src_start)] = y[src_start:src_end]
    return clip


def simulate_native_candidates(y: np.ndarray, config: ReplayConfig) -> list[dict[str, Any]]:
    candidates: list[dict[str, Any]] = []
    bg: list[float] = []
    last_onset_ms = -10_000_000
    frame_start = 0
    while frame_start + FRAME_SIZE <= len(y):
        frame = y[frame_start:frame_start + FRAME_SIZE]
        now_ms = int(round(frame_start / TARGET_SR * 1000))
        if now_ms - last_onset_ms < config.retrigger_ms:
            frame_start += FRAME_SIZE
            continue
        rms = compute_rms(frame)
        bg_avg = float(np.mean(bg)) if bg else rms
        adaptive = max(bg_avg * config.onset_ratio, ABS_MIN_RMS)
        if rms >= adaptive:
            passed, ball_ratio, flatness = spectral_gate(frame)
            candidate = {
                "event_ms": now_ms,
                "onset_sample": frame_start,
                "rms": rms,
                "background_rms": bg_avg,
                "adaptive_threshold": adaptive,
                "onset_ratio": config.onset_ratio,
                "spectral_passed": passed,
                "ball_ratio": ball_ratio,
                "flatness": flatness,
                "native_reject_reason": "" if passed else "spectral_gate",
            }
            if passed:
                last_onset_ms = now_ms
                candidate["clip"] = extract_live_clip(y, frame_start)
                bg = [bg_avg * 2.0] * BG_FRAMES
            else:
                bg.append(rms)
                bg = bg[-BG_FRAMES:]
            candidates.append(candidate)
        else:
            bg.append(rms)
            bg = bg[-BG_FRAMES:]
        frame_start += FRAME_SIZE
    return candidates


def qualified_prediction(
    features: dict[str, float],
    config: ReplayConfig,
    four_class: ModelBundle,
    contact: ModelBundle | None,
) -> tuple[bool, str, float, str, str, float]:
    surface_label, surface_conf, _surface_probs = four_class.predict(features)
    contact_label = ""
    contact_conf = 0.0
    if contact is not None:
        contact_label, contact_conf, _contact_probs = contact.predict(features)

    if config.mode == "four_class_only":
        if surface_label != "racket_bounce":
            return False, surface_label, surface_conf, "surface_not_racket", surface_label, surface_conf
        if surface_conf < config.confidence:
            return False, surface_label, surface_conf, "low_confidence", surface_label, surface_conf
        return True, surface_label, surface_conf, "counted", surface_label, surface_conf

    if config.mode == "binary_only":
        if contact_label != "racket_contact":
            return False, contact_label, contact_conf, "binary_not_contact", surface_label, surface_conf
        if contact_conf < config.confidence:
            return False, contact_label, contact_conf, "low_confidence", surface_label, surface_conf
        return True, contact_label, contact_conf, "counted", surface_label, surface_conf

    if contact_label != "racket_contact":
        return False, contact_label, contact_conf, "binary_not_contact", surface_label, surface_conf
    if contact_conf < config.confidence:
        return False, contact_label, contact_conf, "low_confidence", surface_label, surface_conf
    if surface_label in {"table_bounce", "floor_bounce", "noise"} and surface_conf >= 0.75:
        return False, contact_label, contact_conf, f"surface_veto_{surface_label}", surface_label, surface_conf
    return True, contact_label, contact_conf, "counted", surface_label, surface_conf


def count_live_events(
    y: np.ndarray,
    config: ReplayConfig,
    four_class: ModelBundle,
    contact: ModelBundle | None,
    feature_cache: dict[int, dict[str, float]],
) -> tuple[list[dict[str, Any]], dict[str, int]]:
    candidates = simulate_native_candidates(y, config)
    counted: list[dict[str, Any]] = []
    blockers: dict[str, int] = {}
    last_counted_ms: int | None = None
    active_group_start_ms: int | None = None
    for candidate in candidates:
        reason = candidate["native_reject_reason"]
        if reason:
            blockers[reason] = blockers.get(reason, 0) + 1
            continue
        onset_sample = int(candidate["onset_sample"])
        features = feature_cache.get(onset_sample)
        if features is None:
            features = extract_features(candidate["clip"], TARGET_SR)
            feature_cache[onset_sample] = features
        qualified, label, conf, reason, surface_label, surface_conf = qualified_prediction(
            features, config, four_class, contact
        )
        candidate["label"] = label
        candidate["confidence"] = conf
        candidate["surface_label"] = surface_label
        candidate["surface_confidence"] = surface_conf
        if not qualified:
            blockers[reason] = blockers.get(reason, 0) + 1
            continue
        event_ms = int(candidate["event_ms"])
        if last_counted_ms is not None and event_ms - last_counted_ms <= config.merge_ms:
            blockers["merge_window"] = blockers.get("merge_window", 0) + 1
            continue
        if active_group_start_ms is not None and event_ms - active_group_start_ms <= config.group_ms:
            blockers["group_window"] = blockers.get("group_window", 0) + 1
            continue
        active_group_start_ms = event_ms
        last_counted_ms = event_ms
        counted.append(candidate)
    return counted, blockers


def trainable_racket_truth(markers: list[dict[str, Any]]) -> list[int]:
    truth = []
    for marker in markers:
        if marker.get("final_label") != "racket_contact":
            continue
        if str(marker.get("review_status") or "confirmed") in SKIPPED_STATUSES:
            continue
        if str(marker.get("class_label") or "") == "no_bounce_motion":
            continue
        truth.append(int(round(float(marker.get("timestamp_ms") or 0))))
    return sorted(truth)


def match_counts(counted_ms: list[int], truth_ms: list[int], tolerance_ms: int) -> dict[str, int]:
    matched_truth: set[int] = set()
    true_positive = 0
    false_positive = 0
    duplicates = 0
    for event_ms in sorted(counted_ms):
        nearest_idx = None
        nearest_delta = tolerance_ms + 1
        for idx, ts in enumerate(truth_ms):
            delta = abs(event_ms - ts)
            if delta <= tolerance_ms and delta < nearest_delta:
                nearest_idx = idx
                nearest_delta = delta
        if nearest_idx is None:
            false_positive += 1
        elif nearest_idx in matched_truth:
            duplicates += 1
        else:
            matched_truth.add(nearest_idx)
            true_positive += 1
    return {
        "truth_hits": len(truth_ms),
        "counted": len(counted_ms),
        "true_positive": true_positive,
        "missed": len(truth_ms) - true_positive,
        "false_positive": false_positive,
        "duplicates": duplicates,
    }


def resolve_wav_path(session_path: Path, event: dict[str, Any]) -> Path | None:
    filename = event.get("wav_filename") or event.get("wav_file") or event.get("audio_file")
    if not filename:
        return None
    candidates = [
        session_path.with_suffix("") / str(filename),
        session_path.parent / session_path.stem / str(filename),
        RAW_DIR / session_path.stem / str(filename),
    ]
    for path in candidates:
        if path.exists():
            return path
    return None


def replay_session(
    session_path: Path,
    configs: list[ReplayConfig],
    four_class: ModelBundle,
    contact: ModelBundle | None,
    tolerance_ms: int,
    event_limit: int | None,
) -> list[dict[str, Any]]:
    session = json.loads(session_path.read_text(encoding="utf-8"))
    rows: list[dict[str, Any]] = []
    events = session.get("events") or []
    if event_limit:
        events = events[:event_limit]
    for event_index, event in enumerate(events):
        markers = (event.get("review") or {}).get("markers") or []
        truth = trainable_racket_truth(markers)
        if not truth:
            continue
        wav_path = resolve_wav_path(session_path, event)
        if not wav_path:
            rows.append({
                "session": session_path.stem,
                "event_index": event_index,
                "scenario_id": event.get("scenario_id", ""),
                "config": "missing_wav",
                "truth_hits": len(truth),
            })
            continue
        y, _sr = load_audio(str(wav_path))
        feature_cache: dict[int, dict[str, float]] = {}
        for config in configs:
            counted, blockers = count_live_events(y, config, four_class, contact, feature_cache)
            metrics = match_counts([int(item["event_ms"]) for item in counted], truth, tolerance_ms)
            rows.append({
                "session": session_path.stem,
                "event_index": event_index,
                "wav_filename": wav_path.name,
                "scenario_id": event.get("scenario_id", ""),
                "background_condition": event.get("background_condition", ""),
                "config": config.name,
                "mode": config.mode,
                "confidence": config.confidence,
                "onset_ratio": config.onset_ratio,
                "retrigger_ms": config.retrigger_ms,
                "group_ms": config.group_ms,
                "merge_ms": config.merge_ms,
                **metrics,
                "native_spectral_rejects": blockers.get("spectral_gate", 0),
                "model_rejects": sum(value for key, value in blockers.items() if key not in {"spectral_gate", "merge_window", "group_window"}),
                "merge_rejects": blockers.get("merge_window", 0),
                "group_rejects": blockers.get("group_window", 0),
                "blockers_json": json.dumps(blockers, sort_keys=True),
            })
    return rows


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Replay live racket-bounce detection on reviewed WAV files.")
    parser.add_argument("--session-json", action="append", default=[], help="Path to an audio_session JSON. Repeatable.")
    parser.add_argument("--out-csv", default=str(DEFAULT_OUT_CSV))
    parser.add_argument("--four-class-model-dir", default=str(DEFAULT_FOUR_CLASS_MODEL_DIR))
    parser.add_argument("--contact-model-dir", default=str(DEFAULT_CONTACT_MODEL_DIR))
    parser.add_argument("--event-limit", type=int, default=None)
    parser.add_argument("--match-tolerance-ms", type=int, default=140)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    session_paths = [Path(path) for path in args.session_json]
    if not session_paths:
        session_paths = sorted(RAW_DIR.glob("audio_session_2026-05-25_*.json"))[-5:]
    four_class = load_model_bundle(Path(args.four_class_model_dir), "audio")
    contact_dir = Path(args.contact_model_dir)
    contact = load_model_bundle(contact_dir, "audio_contact") if contact_dir.exists() else None
    rows: list[dict[str, Any]] = []
    for session_path in session_paths:
        rows.extend(replay_session(
            session_path=session_path,
            configs=REPLAY_CONFIGS,
            four_class=four_class,
            contact=contact,
            tolerance_ms=args.match_tolerance_ms,
            event_limit=args.event_limit,
        ))
    if not rows:
        print("No replay rows produced.")
        return

    out_csv = Path(args.out_csv)
    out_csv.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = list(rows[0].keys())
    with out_csv.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)

    print(f"Wrote {out_csv}")
    print("config,truth,counted,tp,missed,fp,dup")
    summary: dict[str, dict[str, int]] = {}
    for row in rows:
        item = summary.setdefault(row["config"], {
            "truth_hits": 0, "counted": 0, "true_positive": 0,
            "missed": 0, "false_positive": 0, "duplicates": 0,
        })
        for key in item:
            item[key] += int(row.get(key) or 0)
    for name, item in summary.items():
        print(
            f"{name},{item['truth_hits']},{item['counted']},{item['true_positive']},"
            f"{item['missed']},{item['false_positive']},{item['duplicates']}"
        )


if __name__ == "__main__":
    main()
