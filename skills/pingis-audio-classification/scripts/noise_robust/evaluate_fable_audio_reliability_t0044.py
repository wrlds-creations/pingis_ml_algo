"""
T0044 audio bounce reliability audit for the bundled Fable app model.

This script evaluates the exact Collector app JSON artifact
`apps/collector/src/models/fable_audio_model.json` without training,
exporting, building an APK, or changing live app behavior.

It can evaluate three sources when available:
- TT Sounds extracted samples (`full.csv` + `sounds/*.wav`, preferred).
- Local reviewed Collector sessions under `data/audio/raw`.
- Fable live debug dumps with `audio_b64` clips.

Outputs are written under `data/audio/models/evaluations/...`, which is
gitignored by the repo-wide `/data/` rule.
"""

from __future__ import annotations

import argparse
import base64
import csv
import json
import math
import statistics
import sys
import urllib.request
import zipfile
from collections import Counter, defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np

SCRIPT_DIR = Path(__file__).resolve().parent
SCRIPTS_DIR = SCRIPT_DIR.parent
for _path in (str(SCRIPT_DIR), str(SCRIPTS_DIR)):
    if _path not in sys.path:
        sys.path.insert(0, _path)

import nr_config  # noqa: E402
import nr_features  # noqa: E402
from preprocess_audio import (  # noqa: E402
    contact_kind_for,
    is_trainable_racket_marker,
    is_trainable_review_marker,
    load_audio,
    multiclass_label_for_marker,
    negative_marker_overlaps_racket,
    not_racket_kind_for,
)

ROOT_DIR = Path(__file__).resolve().parents[4]
MODEL_JSON = ROOT_DIR / "apps" / "collector" / "src" / "models" / "fable_audio_model.json"
DEFAULT_TT_ROOT = ROOT_DIR / "data" / "audio" / "external" / "tt_sounds"
DEFAULT_RAW_DIR = ROOT_DIR / "data" / "audio" / "raw"
DEFAULT_OUT_DIR = ROOT_DIR / "data" / "audio" / "models" / "evaluations" / "t0044_audio_bounce_reliability"
TT_SOUNDS_PAGE = "https://github.com/cogsys-tuebingen/tt_sounds"
TT_SOUNDS_DOWNLOAD = "https://cloud.cs.uni-tuebingen.de/index.php/s/p3tw3EqE9csXoRn/download"
TT_SOUNDS_LICENSE = "CC BY-NC 4.0"

AUDIO_EXTENSIONS = {".wav", ".flac", ".ogg", ".mp3", ".m4a"}
APP_LABELS = ["floor_bounce", "noise", "racket_bounce", "table_bounce"]


@dataclass(frozen=True)
class FableRuntimeConfig:
    quiet_confidence: float = 0.65
    loud_confidence: float = 0.85
    loud_bg_db: float = -36.0
    merge_ms: int = 120
    same_bounce_ms: int = 250
    group_ms: int = 80
    echo_ms: int = 300
    echo_ratio: float = 0.6


@dataclass(frozen=True)
class AudioRecord:
    source: str
    sample_id: str
    path: Path
    true_label: str
    anchor_ms: float | None = None
    metadata: dict[str, Any] | None = None


class FableAppModel:
    def __init__(self, payload: dict[str, Any]) -> None:
        self.metadata = dict(payload.get("metadata") or {})
        self.labels = [str(v) for v in payload["labels"]]
        self.feature_names = [str(v) for v in payload["feature_names"]]
        self.scaler_mean = np.asarray(payload["scaler_mean"], dtype=np.float64)
        self.scaler_std = np.asarray(payload["scaler_std"], dtype=np.float64)
        self.baseline = np.asarray(payload["baseline"], dtype=np.float64)
        self.trees = payload["trees"]
        if len(self.labels) != len(self.baseline):
            raise ValueError("Fable model labels/baseline length mismatch")
        if len(self.feature_names) != len(self.scaler_mean):
            raise ValueError("Fable model feature/scaler length mismatch")

    @classmethod
    def load(cls, path: Path) -> "FableAppModel":
        return cls(json.loads(path.read_text(encoding="utf-8")))

    def predict_features(self, features: dict[str, float]) -> dict[str, Any]:
        scaled = np.zeros(len(self.feature_names), dtype=np.float64)
        for i, name in enumerate(self.feature_names):
            raw = float(features.get(name, 0.0) or 0.0)
            if not math.isfinite(raw):
                raw = 0.0
            std = float(self.scaler_std[i]) or 1.0
            scaled[i] = (raw - float(self.scaler_mean[i])) / std
        return self.predict_scaled(scaled)

    def predict_scaled(self, scaled: np.ndarray) -> dict[str, Any]:
        raw = self.baseline.copy()
        n_classes = len(self.labels)
        for tree_index, tree in enumerate(self.trees):
            node = tree[0]
            while len(node) != 1:
                feature_idx, threshold, left_idx, right_idx = node
                node = tree[int(left_idx) if scaled[int(feature_idx)] <= float(threshold) else int(right_idx)]
            raw[tree_index % n_classes] += float(node[0])
        exps = np.exp(raw - np.max(raw))
        probs = exps / float(np.sum(exps))
        best_idx = int(np.argmax(probs))
        return {
            "label": self.labels[best_idx],
            "confidence": float(probs[best_idx]),
            "probabilities": {label: float(probs[i]) for i, label in enumerate(self.labels)},
        }


class FableOfflineCounter:
    """Python mirror of apps/collector/src/fableEngine.ts for T0044 audit."""

    def __init__(self, model: FableAppModel, config: FableRuntimeConfig) -> None:
        self.model = model
        self.config = config
        self.last_counted: tuple[float, float] | None = None
        self.group_start_ms: float | None = None

    def process_clip(self, clip: np.ndarray, onset_ms: float, frame_rms: float) -> dict[str, Any]:
        fast_rebound = False
        if self.last_counted is not None:
            since_counted = onset_ms - self.last_counted[0]
            rms_ratio = frame_rms / max(self.last_counted[1], 1e-9)
            if since_counted <= self.config.same_bounce_ms:
                if rms_ratio >= 1.1:
                    self.last_counted = (onset_ms, frame_rms)
                    self.group_start_ms = onset_ms
                    return {"counted": False, "reject_reason": "same_bounce"}
                if rms_ratio <= self.config.echo_ratio:
                    return {"counted": False, "reject_reason": "echo_window"}
                if since_counted < 150:
                    return {"counted": False, "reject_reason": "same_bounce"}
                fast_rebound = True
            else:
                if self.group_start_ms is not None and onset_ms - self.group_start_ms <= self.config.group_ms:
                    return {"counted": False, "reject_reason": "group_window"}
                if since_counted <= self.config.echo_ms and rms_ratio <= self.config.echo_ratio:
                    return {"counted": False, "reject_reason": "echo_window"}

        features = nr_features.extract_all_features(clip, nr_config.TARGET_SR)
        prediction = self.model.predict_features(features)
        bg_rms_db = float(features.get("nr_bg_rms_db", -100.0))
        loud = bg_rms_db >= self.config.loud_bg_db
        confidence_threshold = self.config.loud_confidence if loud else self.config.quiet_confidence
        if fast_rebound:
            confidence_threshold = max(confidence_threshold, 0.9)

        result = {
            "counted": False,
            "prediction": prediction,
            "bg_mode": "loud" if loud else "quiet",
            "bg_rms_db": bg_rms_db,
            "confidence_threshold": confidence_threshold,
        }
        if prediction["label"] != "racket_bounce":
            result["reject_reason"] = "not_racket"
            return result
        if float(prediction["confidence"]) < confidence_threshold:
            result["reject_reason"] = "low_confidence_loud_bg" if loud else "low_confidence"
            return result

        self.last_counted = (onset_ms, frame_rms)
        self.group_start_ms = onset_ms
        result["counted"] = True
        result["reject_reason"] = ""
        return result


def read_csv_dicts(path: Path) -> list[dict[str, str]]:
    text = path.read_text(encoding="utf-8-sig")
    first_line = text.splitlines()[0] if text.splitlines() else ""
    delimiter = ";" if first_line.count(";") > first_line.count(",") else ","
    return list(csv.DictReader(text.splitlines(), delimiter=delimiter))


def map_tt_surface(row: dict[str, str]) -> str:
    surface = str(row.get("surface") or "").strip().lower()
    if surface == "racket":
        racket_type = str(row.get("racket-type") or "").strip().lower()
        if racket_type in {"", "none", "nan"}:
            return ""
        return "racket_bounce"
    if surface == "table":
        return "table_bounce"
    if surface == "floor":
        return "floor_bounce"
    if surface == "other":
        return "noise"
    return ""


def cap_records(records: list[AudioRecord], max_per_label: int) -> list[AudioRecord]:
    if max_per_label <= 0:
        return records
    counts: Counter[str] = Counter()
    capped: list[AudioRecord] = []
    for record in sorted(records, key=lambda r: (r.true_label, r.sample_id)):
        if counts[record.true_label] >= max_per_label:
            continue
        counts[record.true_label] += 1
        capped.append(record)
    return capped


def discover_tt_records(tt_root: Path, max_per_label: int = 0) -> tuple[list[AudioRecord], dict[str, Any]]:
    manifest = {
        "source": "TT Sounds",
        "source_url": TT_SOUNDS_PAGE,
        "download_url": TT_SOUNDS_DOWNLOAD,
        "license": TT_SOUNDS_LICENSE,
        "root": str(tt_root),
        "mode": "missing",
        "label_mapping": {
            "surface=racket": "racket_bounce",
            "surface=table": "table_bounce",
            "surface=floor": "floor_bounce",
            "surface=other": "noise",
        },
    }
    if not tt_root.exists():
        return [], manifest

    csv_candidates = sorted(tt_root.rglob("full.csv"), key=lambda p: (len(p.parts), str(p)))
    if csv_candidates:
        metadata_path = csv_candidates[0]
        manifest["mode"] = "metadata_full_csv"
        manifest["metadata_csv"] = str(metadata_path)
        sound_dirs = sorted(tt_root.rglob("sounds"), key=lambda p: (len(p.parts), str(p)))
        preferred_sound_dir = metadata_path.parent / "sounds"
        if preferred_sound_dir.exists():
            sound_dirs.insert(0, preferred_sound_dir)

        records: list[AudioRecord] = []
        missing_files = 0
        skipped_labels = 0
        for row in read_csv_dicts(metadata_path):
            label = map_tt_surface(row)
            bounce_id = str(row.get("bounce-id") or "").strip()
            if not label or not bounce_id:
                skipped_labels += 1
                continue
            candidates = [sound_dir / f"{bounce_id}.wav" for sound_dir in sound_dirs]
            audio_path = next((p for p in candidates if p.exists()), None)
            if audio_path is None:
                matches = list(tt_root.rglob(f"{bounce_id}.wav"))
                audio_path = matches[0] if matches else None
            if audio_path is None:
                missing_files += 1
                continue
            records.append(AudioRecord(
                source="tt_sounds",
                sample_id=bounce_id,
                path=audio_path,
                true_label=label,
                metadata={k: v for k, v in row.items() if k},
            ))
        manifest["rows_in_csv"] = len(read_csv_dicts(metadata_path))
        manifest["missing_audio_files"] = missing_files
        manifest["skipped_unknown_labels"] = skipped_labels
        return cap_records(records, max_per_label), manifest

    records = []
    manifest["mode"] = "path_inference"
    for path in sorted(p for p in tt_root.rglob("*") if p.suffix.lower() in AUDIO_EXTENSIONS):
        joined = " ".join(part.lower() for part in path.parts)
        label = ""
        if "racket" in joined:
            label = "racket_bounce"
        elif "table" in joined:
            label = "table_bounce"
        elif "floor" in joined:
            label = "floor_bounce"
        elif "other" in joined or "noise" in joined:
            label = "noise"
        if label:
            records.append(AudioRecord("tt_sounds", path.stem, path, label))
    return cap_records(records, max_per_label), manifest


def marker_label(marker: dict[str, Any], event_label: str, scenario_id: str) -> str:
    return multiclass_label_for_marker(
        str(marker.get("final_label") or ""),
        str(marker.get("contact_kind") or contact_kind_for(event_label, scenario_id) or ""),
        str(marker.get("not_racket_kind") or not_racket_kind_for(event_label, scenario_id) or ""),
    )


def resolve_wav(raw_dir: Path, session_path: Path, session_id: str, wav_filename: str) -> Path:
    candidates = [
        session_path.parent / session_id / wav_filename,
        raw_dir / session_id / wav_filename,
        session_path.parent / wav_filename,
    ]
    for candidate in candidates:
        if candidate.exists():
            return candidate
    return candidates[0]


def discover_local_clip_records(raw_dir: Path, max_per_label: int = 0) -> tuple[list[AudioRecord], dict[str, Any]]:
    manifest = {"source": "local_reviewed_sessions", "root": str(raw_dir), "mode": "missing"}
    if not raw_dir.exists():
        return [], manifest

    records: list[AudioRecord] = []
    skipped = Counter()
    for session_path in sorted(raw_dir.rglob("audio_session_*.json")):
        session_id = session_path.stem
        session = json.loads(session_path.read_text(encoding="utf-8"))
        for event_index, event in enumerate(session.get("events") or []):
            wav_filename = str(event.get("wav_filename") or "")
            markers = (event.get("review") or {}).get("markers") or []
            if not wav_filename or not markers:
                skipped["missing_wav_or_markers"] += 1
                continue
            wav_path = resolve_wav(raw_dir, session_path, session_id, wav_filename)
            if not wav_path.exists():
                skipped["missing_wav_file"] += 1
                continue
            event_label = str(event.get("label") or "")
            scenario_id = str(event.get("scenario_id") or "")
            racket_ts = [
                float(m.get("timestamp_ms") or 0.0)
                for m in markers
                if is_trainable_racket_marker(m)
            ]
            for marker_index, marker in enumerate(markers):
                if not is_trainable_review_marker(marker):
                    continue
                label = marker_label(marker, event_label, scenario_id)
                if label not in APP_LABELS:
                    skipped[f"unsupported_label:{label}"] += 1
                    continue
                if label != "racket_bounce" and negative_marker_overlaps_racket(marker, racket_ts):
                    skipped["negative_overlaps_racket"] += 1
                    continue
                records.append(AudioRecord(
                    source="local_reviewed_marker",
                    sample_id=f"{session_id}:event{event_index}:marker{marker_index}",
                    path=wav_path,
                    true_label=label,
                    anchor_ms=float(marker.get("timestamp_ms") or 0.0),
                    metadata={
                        "session_id": session_id,
                        "event_index": event_index,
                        "wav_filename": wav_filename,
                        "background_condition": event.get("background_condition") or "",
                    },
                ))
    manifest["mode"] = "reviewed_markers"
    manifest["skipped"] = dict(skipped)
    return cap_records(records, max_per_label), manifest


def clip_from_peak(y: np.ndarray) -> np.ndarray:
    if len(y) == 0:
        return np.zeros(nr_config.CLIP_SAMPLES, dtype=np.float32)
    peak_idx = int(np.argmax(np.abs(y)))
    return nr_features.extract_live_clip(y, peak_idx)


def clip_from_record(y: np.ndarray, record: AudioRecord) -> np.ndarray:
    if record.anchor_ms is None:
        return clip_from_peak(y)
    anchor_sample = int(round(record.anchor_ms / 1000.0 * nr_config.TARGET_SR))
    return nr_features.extract_live_clip(y, anchor_sample)


def synthetic_stream_from_sample(y: np.ndarray, onset_ms: float = 300.0, duration_s: float = 1.0) -> np.ndarray:
    stream = np.zeros(int(round(duration_s * nr_config.TARGET_SR)), dtype=np.float32)
    if len(y) == 0:
        return stream
    peak_idx = int(np.argmax(np.abs(y)))
    onset_sample = int(round(onset_ms / 1000.0 * nr_config.TARGET_SR))
    start = onset_sample - peak_idx
    src_start = max(0, -start)
    dst_start = max(0, start)
    count = min(len(y) - src_start, len(stream) - dst_start)
    if count > 0:
        stream[dst_start:dst_start + count] = y[src_start:src_start + count]
    return stream


def safe_probabilities(prediction: dict[str, Any]) -> dict[str, float]:
    probs = prediction.get("probabilities") or {}
    return {f"prob_{label}": float(probs.get(label, 0.0)) for label in APP_LABELS}


def evaluate_clip_records(model: FableAppModel, records: list[AudioRecord]) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    cache: dict[Path, np.ndarray] = {}
    for record in records:
        try:
            if record.path not in cache:
                cache[record.path] = load_audio(str(record.path))[0]
            y = cache[record.path]
            clip = clip_from_record(y, record)
            features = nr_features.extract_all_features(clip, nr_config.TARGET_SR)
            prediction = model.predict_features(features)
            row = {
                "source": record.source,
                "sample_id": record.sample_id,
                "path": str(record.path),
                "true_label": record.true_label,
                "predicted_label": prediction["label"],
                "confidence": round(float(prediction["confidence"]), 8),
                "correct": prediction["label"] == record.true_label,
                "duration_ms": round(len(y) / nr_config.TARGET_SR * 1000.0, 3),
                "anchor_ms": "" if record.anchor_ms is None else round(record.anchor_ms, 3),
                "nr_bg_rms_db": round(float(features.get("nr_bg_rms_db", -100.0)), 6),
                **safe_probabilities(prediction),
            }
            if record.metadata:
                for key, value in record.metadata.items():
                    row[f"meta_{key}"] = value
            rows.append(row)
        except Exception as exc:  # keep audit moving; error rows are actionable.
            rows.append({
                "source": record.source,
                "sample_id": record.sample_id,
                "path": str(record.path),
                "true_label": record.true_label,
                "predicted_label": "error",
                "confidence": 0.0,
                "correct": False,
                "error": str(exc),
            })
    return rows


def evaluate_tt_synthetic_live(
    model: FableAppModel,
    records: list[AudioRecord],
    config: FableRuntimeConfig,
) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for record in records:
        try:
            y, _sr = load_audio(str(record.path))
            stream = synthetic_stream_from_sample(y)
            triggers = nr_features.simulate_gate(
                stream,
                nr_config.TARGET_SR,
                onset_ratio=1.5,
                retrigger_ms=120,
                abs_min_rms=0.0015,
                mode="bandpass",
                spectral_gate=False,
            )
            counter = FableOfflineCounter(model, config)
            detections = []
            for trigger in triggers:
                if not trigger["passed_spectral"]:
                    continue
                clip = nr_features.extract_live_clip(stream, int(trigger["onset_sample"]))
                result = counter.process_clip(clip, float(trigger["onset_ms"]), float(trigger["frame_rms"]))
                detections.append((trigger, result))
            counted = [item for item in detections if item[1].get("counted")]
            first_result = detections[0][1] if detections else {}
            prediction = first_result.get("prediction") or {}
            is_racket_truth = record.true_label == "racket_bounce"
            n_counted = len(counted)
            rows.append({
                "source": "tt_sounds_synthetic_live",
                "sample_id": record.sample_id,
                "path": str(record.path),
                "true_label": record.true_label,
                "truth_is_racket": is_racket_truth,
                "n_raw_triggers": len(triggers),
                "n_classified": len(detections),
                "n_counted": n_counted,
                "counted": n_counted > 0,
                "duplicate_count": max(0, n_counted - 1) if is_racket_truth else n_counted,
                "live_outcome": live_outcome(is_racket_truth, n_counted > 0),
                "predicted_label": prediction.get("label", ""),
                "confidence": round(float(prediction.get("confidence", 0.0)), 8),
                "reject_reason": first_result.get("reject_reason", "gate_miss" if not detections else ""),
                "bg_mode": first_result.get("bg_mode", ""),
                "bg_rms_db": round(float(first_result.get("bg_rms_db", -100.0)), 6)
                if first_result else "",
                **(safe_probabilities(prediction) if prediction else {}),
            })
        except Exception as exc:
            rows.append({
                "source": "tt_sounds_synthetic_live",
                "sample_id": record.sample_id,
                "path": str(record.path),
                "true_label": record.true_label,
                "truth_is_racket": record.true_label == "racket_bounce",
                "n_raw_triggers": 0,
                "n_classified": 0,
                "n_counted": 0,
                "counted": False,
                "duplicate_count": 0,
                "live_outcome": "error",
                "predicted_label": "error",
                "confidence": 0.0,
                "reject_reason": "error",
                "error": str(exc),
            })
    return rows


def live_outcome(truth_is_racket: bool, counted: bool) -> str:
    if truth_is_racket and counted:
        return "tp"
    if truth_is_racket and not counted:
        return "missed"
    if not truth_is_racket and counted:
        return "fp"
    return "tn"


def score_matches(detections_ms: list[float], truth_ms: list[float], tolerance_ms: int = 140) -> dict[str, int]:
    matched: set[int] = set()
    tp = fp = duplicate = 0
    for detection in sorted(detections_ms):
        nearest_idx: int | None = None
        nearest_delta = tolerance_ms + 1
        for idx, truth in enumerate(truth_ms):
            delta = abs(detection - truth)
            if delta <= tolerance_ms and delta < nearest_delta:
                nearest_idx = idx
                nearest_delta = delta
        if nearest_idx is None:
            fp += 1
        elif nearest_idx in matched:
            duplicate += 1
        else:
            matched.add(nearest_idx)
            tp += 1
    return {"true_positive": tp, "false_positive": fp, "duplicates": duplicate, "missed": len(truth_ms) - tp}


def evaluate_local_live(raw_dir: Path, model: FableAppModel, config: FableRuntimeConfig) -> list[dict[str, Any]]:
    if not raw_dir.exists():
        return []
    rows: list[dict[str, Any]] = []
    for session_path in sorted(raw_dir.glob("audio_session_*.json")):
        session_id = session_path.stem
        session = json.loads(session_path.read_text(encoding="utf-8"))
        for event_index, event in enumerate(session.get("events") or []):
            wav_filename = str(event.get("wav_filename") or "")
            markers = (event.get("review") or {}).get("markers") or []
            if not wav_filename or not markers:
                continue
            wav_path = resolve_wav(raw_dir, session_path, session_id, wav_filename)
            if not wav_path.exists():
                continue
            truth = sorted(float(m.get("timestamp_ms") or 0.0) for m in markers if is_trainable_racket_marker(m))
            y, _sr = load_audio(str(wav_path))
            triggers = nr_features.simulate_gate(
                y,
                nr_config.TARGET_SR,
                onset_ratio=1.5,
                retrigger_ms=120,
                abs_min_rms=0.0015,
                mode="bandpass",
                spectral_gate=False,
            )
            counter = FableOfflineCounter(model, config)
            counted_ms: list[float] = []
            classified = 0
            reject_counts: Counter[str] = Counter()
            for trigger in triggers:
                if not trigger["passed_spectral"]:
                    continue
                classified += 1
                clip = nr_features.extract_live_clip(y, int(trigger["onset_sample"]))
                result = counter.process_clip(clip, float(trigger["onset_ms"]), float(trigger["frame_rms"]))
                if result.get("counted"):
                    counted_ms.append(float(trigger["onset_ms"]))
                else:
                    reject_counts[str(result.get("reject_reason") or "unknown")] += 1
            metrics = score_matches(counted_ms, truth)
            rows.append({
                "source": "local_reviewed_live_replay",
                "session_id": session_id,
                "event_index": event_index,
                "wav_filename": wav_filename,
                "background_condition": str(event.get("background_condition") or ""),
                "duration_s": round(len(y) / nr_config.TARGET_SR, 3),
                "n_truth_racket": len(truth),
                "n_raw_triggers": len(triggers),
                "n_classified": classified,
                "n_counted": len(counted_ms),
                **metrics,
                "reject_counts": json.dumps(dict(reject_counts), sort_keys=True),
            })
    return rows


def decode_debug_clip(audio_b64: str) -> np.ndarray:
    pcm = np.frombuffer(base64.b64decode(audio_b64), dtype="<i2")
    return (pcm.astype(np.float32) / 32768.0)


def evaluate_fable_debug(debug_dir: Path, model: FableAppModel, config: FableRuntimeConfig) -> list[dict[str, Any]]:
    if not debug_dir.exists():
        return []
    rows: list[dict[str, Any]] = []
    for path in sorted(debug_dir.rglob("fable_live_session_*.json")):
        data = json.loads(path.read_text(encoding="utf-8"))
        counter = FableOfflineCounter(model, config)
        for event in sorted(data.get("events") or [], key=lambda e: float(e.get("native_onset_time_ms") or 0.0)):
            audio_b64 = event.get("audio_b64")
            if not audio_b64:
                continue
            clip = decode_debug_clip(str(audio_b64))
            onset_ms = float(event.get("native_onset_time_ms") or 0.0)
            frame_rms = float(event.get("native_rms") or 0.0)
            result = counter.process_clip(clip, onset_ms, frame_rms)
            prediction = result.get("prediction") or {}
            rows.append({
                "source": "fable_live_debug",
                "dump_file": str(path),
                "event_index": event.get("index", ""),
                "onset_ms": onset_ms,
                "saved_counted": bool(event.get("counted")),
                "offline_counted": bool(result.get("counted")),
                "saved_label": event.get("model_label", ""),
                "offline_label": prediction.get("label", ""),
                "saved_confidence": event.get("model_confidence", ""),
                "offline_confidence": round(float(prediction.get("confidence", 0.0)), 8),
                "saved_reject_reason": event.get("reject_reason", ""),
                "offline_reject_reason": result.get("reject_reason", ""),
                "parity_match": (
                    bool(event.get("counted")) == bool(result.get("counted"))
                    and str(event.get("model_label") or "") == str(prediction.get("label") or "")
                ),
            })
    return rows


def confusion_matrix(rows: list[dict[str, Any]]) -> dict[str, dict[str, int]]:
    matrix: dict[str, dict[str, int]] = {label: {pred: 0 for pred in APP_LABELS + ["error"]} for label in APP_LABELS}
    for row in rows:
        true_label = str(row.get("true_label") or "")
        predicted = str(row.get("predicted_label") or "")
        if true_label not in matrix:
            matrix[true_label] = {pred: 0 for pred in APP_LABELS + ["error"]}
        if predicted not in matrix[true_label]:
            matrix[true_label][predicted] = 0
        matrix[true_label][predicted] += 1
    return matrix


def classification_metrics(rows: list[dict[str, Any]]) -> dict[str, Any]:
    if not rows:
        return {"rows": 0}
    labels = sorted({str(r.get("true_label") or "") for r in rows} | {str(r.get("predicted_label") or "") for r in rows})
    labels = [label for label in labels if label]
    per_label: dict[str, dict[str, float | int]] = {}
    total_correct = 0
    for label in labels:
        tp = sum(1 for r in rows if r.get("true_label") == label and r.get("predicted_label") == label)
        fp = sum(1 for r in rows if r.get("true_label") != label and r.get("predicted_label") == label)
        fn = sum(1 for r in rows if r.get("true_label") == label and r.get("predicted_label") != label)
        support = sum(1 for r in rows if r.get("true_label") == label)
        total_correct += tp
        precision = tp / (tp + fp) if tp + fp else 0.0
        recall = tp / (tp + fn) if tp + fn else 0.0
        f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0.0
        per_label[label] = {
            "support": support,
            "precision": round(precision, 6),
            "recall": round(recall, 6),
            "f1": round(f1, 6),
            "false_positive": fp,
            "false_negative": fn,
        }
    return {
        "rows": len(rows),
        "accuracy": round(total_correct / len(rows), 6),
        "per_label": per_label,
        "confusion_matrix": confusion_matrix(rows),
        "confidence": confidence_summary(rows),
    }


def confidence_summary(rows: list[dict[str, Any]]) -> dict[str, Any]:
    groups: dict[str, list[float]] = defaultdict(list)
    for row in rows:
        try:
            conf = float(row.get("confidence") or 0.0)
        except (TypeError, ValueError):
            continue
        correct = bool(row.get("correct"))
        true_label = str(row.get("true_label") or "")
        predicted = str(row.get("predicted_label") or "")
        groups["all"].append(conf)
        groups["correct" if correct else "wrong"].append(conf)
        if predicted == "racket_bounce" and true_label != "racket_bounce":
            groups["non_racket_predicted_racket"].append(conf)
        if true_label == "racket_bounce" and predicted != "racket_bounce":
            groups["racket_missed_or_misclassified"].append(conf)
    return {name: describe_values(values) for name, values in sorted(groups.items())}


def describe_values(values: list[float]) -> dict[str, float | int | None]:
    if not values:
        return {"n": 0, "mean": None, "p50": None, "p90": None, "min": None, "max": None}
    ordered = sorted(values)
    p90_idx = min(len(ordered) - 1, int(math.ceil(0.9 * len(ordered))) - 1)
    return {
        "n": len(values),
        "mean": round(float(statistics.fmean(values)), 6),
        "p50": round(float(statistics.median(values)), 6),
        "p90": round(float(ordered[p90_idx]), 6),
        "min": round(float(ordered[0]), 6),
        "max": round(float(ordered[-1]), 6),
    }


def live_metrics(rows: list[dict[str, Any]]) -> dict[str, Any]:
    if not rows:
        return {"rows": 0}
    outcomes = Counter(str(row.get("live_outcome") or "") for row in rows)
    tp = outcomes["tp"]
    fp = outcomes["fp"]
    missed = outcomes["missed"]
    tn = outcomes["tn"]
    precision = tp / (tp + fp) if tp + fp else 0.0
    recall = tp / (tp + missed) if tp + missed else 0.0
    duplicate_count = sum(int(row.get("duplicate_count") or 0) for row in rows)
    per_label = Counter(str(row.get("true_label") or "") for row in rows)
    return {
        "rows": len(rows),
        "true_positive": tp,
        "false_positive": fp,
        "true_negative": tn,
        "missed": missed,
        "duplicates": duplicate_count,
        "precision": round(precision, 6),
        "recall": round(recall, 6),
        "outcomes": dict(outcomes),
        "per_true_label": dict(per_label),
        "gate_misses": sum(1 for row in rows if str(row.get("reject_reason") or "") == "gate_miss"),
        "reject_reasons": dict(Counter(str(row.get("reject_reason") or "") for row in rows)),
    }


def local_live_metrics(rows: list[dict[str, Any]]) -> dict[str, Any]:
    if not rows:
        return {"rows": 0}
    totals = Counter()
    minutes = 0.0
    for row in rows:
        for key in ("true_positive", "false_positive", "duplicates", "missed", "n_truth_racket", "n_counted"):
            totals[key] += int(row.get(key) or 0)
        minutes += float(row.get("duration_s") or 0.0) / 60.0
    precision = totals["true_positive"] / (totals["true_positive"] + totals["false_positive"]) if totals["true_positive"] + totals["false_positive"] else 0.0
    recall = totals["true_positive"] / (totals["true_positive"] + totals["missed"]) if totals["true_positive"] + totals["missed"] else 0.0
    return {
        "rows": len(rows),
        **dict(totals),
        "precision": round(precision, 6),
        "recall": round(recall, 6),
        "fp_per_min": round(totals["false_positive"] / minutes, 6) if minutes else None,
    }


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames: list[str] = []
    seen: set[str] = set()
    for row in rows:
        for key in row:
            if key not in seen:
                seen.add(key)
                fieldnames.append(key)
    if not fieldnames:
        fieldnames = ["empty"]
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow({key: row.get(key, "") for key in fieldnames})


def bad_case_rows(clip_rows: list[dict[str, Any]], live_rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for row in clip_rows:
        if row.get("true_label") != row.get("predicted_label"):
            rows.append({"case_type": "clip_wrong_class", **row})
    for row in live_rows:
        if row.get("live_outcome") in {"fp", "missed", "error"}:
            rows.append({"case_type": f"live_{row.get('live_outcome')}", **row})
    rows.sort(key=lambda r: float(r.get("confidence") or 0.0), reverse=True)
    return rows[:500]


def write_markdown(path: Path, summary: dict[str, Any]) -> None:
    model = summary["model"]
    clip = summary["clip_level"]
    live = summary["synthetic_live"]
    local_live = summary["local_live_replay"]
    lines = [
        "# T0044 Fable Audio Reliability Audit",
        "",
        f"- Model: `{model.get('model_version', 'unknown')}`",
        f"- Feature version: `{model.get('feature_version', 'unknown')}`",
        f"- TT Sounds license: `{TT_SOUNDS_LICENSE}`; evaluation/internal diagnostics only.",
        f"- App artifacts changed: `false`",
        "",
        "## Clip-Level Classification",
        "",
        f"- Rows: `{clip.get('rows', 0)}`",
        f"- Accuracy: `{clip.get('accuracy', 0)}`",
    ]
    racket_clip = (clip.get("per_label") or {}).get("racket_bounce") or {}
    if racket_clip:
        lines.append(
            "- Racket precision / recall: "
            f"`{racket_clip.get('precision')}` / `{racket_clip.get('recall')}`"
        )
    lines.extend([
        "",
        "## Synthetic Live Replay",
        "",
        f"- Rows: `{live.get('rows', 0)}`",
        f"- TP / FP / missed / duplicates: "
        f"`{live.get('true_positive', 0)}` / `{live.get('false_positive', 0)}` / "
        f"`{live.get('missed', 0)}` / `{live.get('duplicates', 0)}`",
        f"- Precision / recall: `{live.get('precision', 0)}` / `{live.get('recall', 0)}`",
        "",
        "## Local Reviewed Live Replay",
        "",
        f"- Event rows: `{local_live.get('rows', 0)}`",
        f"- TP / FP / missed / duplicates: "
        f"`{local_live.get('true_positive', 0)}` / `{local_live.get('false_positive', 0)}` / "
        f"`{local_live.get('missed', 0)}` / `{local_live.get('duplicates', 0)}`",
        f"- Precision / recall / FP-min: "
        f"`{local_live.get('precision', 0)}` / `{local_live.get('recall', 0)}` / "
        f"`{local_live.get('fp_per_min')}`",
        "",
        "## Notes",
        "",
        "- This ticket does not train, export, build, install, or replace `fable_audio_model.json`.",
        "- TT Sounds extracted samples are 15 ms snippets, so synthetic live replay is a stress proxy, not a substitute for Motorola live sessions.",
        "- Promotion decisions should use local reviewed Motorola data as the gate.",
    ])
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def download_tt_sounds(tt_root: Path) -> Path:
    tt_root.mkdir(parents=True, exist_ok=True)
    zip_path = tt_root / "tt_sounds_nextcloud.zip"
    print(f"Downloading TT Sounds to {zip_path}")
    urllib.request.urlretrieve(TT_SOUNDS_DOWNLOAD, zip_path)
    if not zipfile.is_zipfile(zip_path):
        raise SystemExit(
            f"Downloaded file is not a zip archive: {zip_path}. "
            "Open the Nextcloud link in a browser and extract it under --tt-root."
        )
    extract_dir = tt_root / "extracted"
    extract_dir.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(zip_path) as archive:
        archive.extractall(extract_dir)
    return extract_dir


def run_self_test() -> None:
    toy = FableAppModel({
        "metadata": {"model_version": "toy"},
        "labels": ["floor_bounce", "racket_bounce"],
        "feature_names": ["x"],
        "scaler_mean": [0.0],
        "scaler_std": [1.0],
        "baseline": [0.0, 0.0],
        "trees": [
            [[0, 0.5, 1, 2], [1.0], [-1.0]],
            [[0, 0.5, 1, 2], [-1.0], [1.0]],
        ],
    })
    assert toy.predict_features({"x": 0.0})["label"] == "floor_bounce"
    assert toy.predict_features({"x": 1.0})["label"] == "racket_bounce"
    scores = score_matches([1000.0, 1100.0, 3000.0], [1000.0, 3000.0])
    assert scores == {"true_positive": 2, "false_positive": 0, "duplicates": 1, "missed": 0}
    print("self-test OK: HGB JSON traversal and event scorer behave as expected")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Evaluate bundled Fable audio reliability for T0044.")
    parser.add_argument("--model-json", type=Path, default=MODEL_JSON)
    parser.add_argument("--tt-root", type=Path, default=DEFAULT_TT_ROOT)
    parser.add_argument("--raw-dir", type=Path, default=DEFAULT_RAW_DIR)
    parser.add_argument("--fable-debug-dir", type=Path, default=None)
    parser.add_argument("--out-dir", type=Path, default=DEFAULT_OUT_DIR)
    parser.add_argument("--max-per-label", type=int, default=0, help="0 = no cap; useful for smoke tests.")
    parser.add_argument("--download-tt", action="store_true", help="Download/extract TT Sounds into --tt-root.")
    parser.add_argument("--skip-tt", action="store_true")
    parser.add_argument("--skip-local", action="store_true")
    parser.add_argument("--self-test", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.self_test:
        run_self_test()
        return

    if args.download_tt:
        args.tt_root = download_tt_sounds(args.tt_root)

    model = FableAppModel.load(args.model_json)
    config = FableRuntimeConfig()
    args.out_dir.mkdir(parents=True, exist_ok=True)

    source_manifests: list[dict[str, Any]] = []
    tt_records: list[AudioRecord] = []
    local_records: list[AudioRecord] = []
    if not args.skip_tt:
        tt_records, tt_manifest = discover_tt_records(args.tt_root, args.max_per_label)
        source_manifests.append(tt_manifest)
    if not args.skip_local:
        local_records, local_manifest = discover_local_clip_records(args.raw_dir, args.max_per_label)
        source_manifests.append(local_manifest)

    clip_rows = evaluate_clip_records(model, tt_records + local_records)
    live_rows = evaluate_tt_synthetic_live(model, tt_records, config)
    local_live_rows = [] if args.skip_local else evaluate_local_live(args.raw_dir, model, config)
    fable_debug_rows = evaluate_fable_debug(args.fable_debug_dir, model, config) if args.fable_debug_dir else []

    write_csv(args.out_dir / "t0044_clip_predictions.csv", clip_rows)
    write_csv(args.out_dir / "t0044_tt_synthetic_live_predictions.csv", live_rows)
    write_csv(args.out_dir / "t0044_local_live_replay.csv", local_live_rows)
    write_csv(args.out_dir / "t0044_fable_debug_parity.csv", fable_debug_rows)
    write_csv(args.out_dir / "t0044_bad_cases.csv", bad_case_rows(clip_rows, live_rows))

    summary = {
        "ticket": "T0044-audio-bounce-reliability-audit",
        "changed_app_artifacts": False,
        "model": {
            "path": str(args.model_json),
            "model_version": model.metadata.get("model_version"),
            "feature_version": model.metadata.get("feature_version"),
            "model_type": model.metadata.get("model_type"),
            "labels": model.labels,
            "features": len(model.feature_names),
            "trees": len(model.trees),
        },
        "runtime_config": config.__dict__,
        "sources": source_manifests,
        "clip_level": classification_metrics(clip_rows),
        "synthetic_live": live_metrics(live_rows),
        "local_live_replay": local_live_metrics(local_live_rows),
        "fable_debug_parity": {
            "rows": len(fable_debug_rows),
            "parity_matches": sum(1 for row in fable_debug_rows if row.get("parity_match")),
        },
        "outputs": {
            "clip_predictions_csv": str(args.out_dir / "t0044_clip_predictions.csv"),
            "tt_synthetic_live_csv": str(args.out_dir / "t0044_tt_synthetic_live_predictions.csv"),
            "local_live_replay_csv": str(args.out_dir / "t0044_local_live_replay.csv"),
            "fable_debug_parity_csv": str(args.out_dir / "t0044_fable_debug_parity.csv"),
            "bad_cases_csv": str(args.out_dir / "t0044_bad_cases.csv"),
        },
    }
    (args.out_dir / "t0044_audio_reliability_summary.json").write_text(
        json.dumps(summary, indent=2),
        encoding="utf-8",
    )
    write_markdown(args.out_dir / "t0044_audio_reliability_summary.md", summary)
    (args.out_dir / "t0044_source_manifest.json").write_text(
        json.dumps(source_manifests, indent=2),
        encoding="utf-8",
    )

    print(f"Model: {summary['model']['model_version']}")
    print(f"Clip rows: {summary['clip_level']['rows']}")
    print(f"Synthetic live rows: {summary['synthetic_live']['rows']}")
    print(f"Local live rows: {summary['local_live_replay']['rows']}")
    print(f"Wrote {args.out_dir / 't0044_audio_reliability_summary.md'}")


if __name__ == "__main__":
    main()
