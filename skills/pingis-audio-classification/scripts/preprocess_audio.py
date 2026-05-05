"""
preprocess_audio.py

Builds two datasets from raw audio sessions:

1. data/audio/processed/audio_dataset.csv
   Secondary multiclass debug dataset:
   racket_bounce / table_bounce / floor_bounce / noise

2. data/audio/processed/audio_contact_dataset.csv
   Primary binary contact dataset:
   racket_contact / not_racket_contact

Reviewed takes use only saved review markers.
Auto-only takes continue to use onset/chunk segmentation.

Run:
  python skills/pingis-audio-classification/scripts/preprocess_audio.py
  python skills/pingis-audio-classification/scripts/preprocess_audio.py --bootstrap-unreviewed
"""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
import tempfile
from pathlib import Path

import librosa
import numpy as np
import pandas as pd

_FFMPEG_CANDIDATES = [
    "ffmpeg",
    r"C:\Users\lovea\AppData\Local\Microsoft\WinGet\Packages\Gyan.FFmpeg_Microsoft.Winget.Source_8wekyb3d8bbwe\ffmpeg-8.1-full_build\bin\ffmpeg.exe",
]


def _find_ffmpeg() -> str:
    for candidate in _FFMPEG_CANDIDATES:
        try:
            subprocess.run([candidate, "-version"], capture_output=True, check=True)
            return candidate
        except (FileNotFoundError, subprocess.CalledProcessError):
            continue
    raise RuntimeError("ffmpeg hittades inte. Installera med: winget install ffmpeg")


FFMPEG = _find_ffmpeg()


def load_audio(path: str) -> tuple[np.ndarray, int]:
    suffix = Path(path).suffix.lower()
    if suffix in (".wav", ".flac", ".ogg"):
        return librosa.load(path, sr=TARGET_SR, mono=True)

    with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as tmp:
        tmp_path = tmp.name
    try:
        subprocess.run(
            [FFMPEG, "-y", "-i", path, "-ar", str(TARGET_SR), "-ac", "1", tmp_path],
            capture_output=True,
            check=True,
        )
        y, sr = librosa.load(tmp_path, sr=TARGET_SR, mono=True)
    finally:
        os.unlink(tmp_path)
    return y, sr


ROOT_DIR = Path(__file__).resolve().parents[3]
RAW_DIR = ROOT_DIR / "data" / "audio" / "raw"
OUT_DIR = ROOT_DIR / "data" / "audio" / "processed"
OUT_FILE = OUT_DIR / "audio_dataset.csv"
OUT_CONTACT_FILE = OUT_DIR / "audio_contact_dataset.csv"

TARGET_SR = 22050
CLIP_FRAMES = TARGET_SR

MIN_ONSET_GAP_S = 0.30
WINDOW_BEFORE_S = 0.30
WINDOW_AFTER_S = 0.70
MIN_CLIP_RMS = 0.005

AUGMENT_SNR_DB = [20.0, 8.0]
VALID_AUDIO_LABELS = {"racket_bounce", "table_bounce", "floor_bounce", "noise"}


def contact_kind_for(label: str, scenario_id: str) -> str:
    if label == "racket_bounce" or scenario_id.startswith("racket_bounce") or scenario_id == "free_recording":
        return "racket_bounce"
    return ""


def not_racket_kind_for(label: str, scenario_id: str) -> str:
    if label == "table_bounce" or scenario_id == "table_bounce":
        return "table_bounce"
    if label == "floor_bounce" or scenario_id == "floor_bounce":
        return "floor_bounce"
    if scenario_id == "catch_after_sound":
        return "catch_after_sound"
    if scenario_id == "speech_music_noise":
        return "voice_music_noise"
    if label == "noise":
        return "other_impact"
    return ""


def multiclass_label_for_marker(final_label: str, contact_kind: str, not_racket_kind: str) -> str:
    if final_label == "racket_contact":
        return contact_kind or "racket_bounce"
    if not_racket_kind in {"table_bounce", "floor_bounce"}:
        return not_racket_kind
    return "noise"


def binary_label_for_audio_label(label: str) -> str:
    if label == "racket_bounce":
        return "racket_contact"
    if label in {"table_bounce", "floor_bounce", "noise"}:
        return "not_racket_contact"
    return ""


def event_type_for_class_label(class_label: str) -> str:
    if class_label in {"racket_bounce", "forehand", "backhand", "forehand_hit", "backhand_hit"}:
        return "racket_hit"
    if class_label in {"table_bounce", "floor_bounce"}:
        return "bounce"
    if class_label == "ignore":
        return "ignore"
    return "noise"


def extract_transient_features(y: np.ndarray, sr: int = TARGET_SR) -> dict:
    features: dict = {}
    abs_y = np.abs(y)
    peak_idx = int(np.argmax(abs_y))
    peak_val = abs_y[peak_idx]

    if peak_val < 1e-6:
        return {
            "attack_time_ms": 0.0,
            "decay_time_ms": 0.0,
            "attack_slope": 0.0,
            "crest_factor": 0.0,
            "temporal_centroid": 0.5,
            "energy_decay_rate": 0.0,
        }

    thresh_10 = peak_val * 0.1
    thresh_90 = peak_val * 0.9
    search_start = max(0, peak_idx - int(0.05 * sr))
    attack_region = abs_y[search_start:peak_idx + 1]
    idx_10 = np.searchsorted(attack_region, thresh_10)
    idx_90 = np.searchsorted(attack_region, thresh_90)
    attack_samples = max(1, idx_90 - idx_10)
    features["attack_time_ms"] = float(attack_samples / sr * 1000)

    thresh_50 = peak_val * 0.5
    search_end = min(len(y), peak_idx + int(0.1 * sr))
    decay_region = abs_y[peak_idx:search_end]
    below_50 = np.where(decay_region < thresh_50)[0]
    decay_samples = int(below_50[0]) if len(below_50) > 0 else len(decay_region)
    features["decay_time_ms"] = float(decay_samples / sr * 1000)

    features["attack_slope"] = float(peak_val / max(features["attack_time_ms"], 0.01))

    region_start = max(0, peak_idx - int(0.05 * sr))
    region_end = min(len(y), peak_idx + int(0.05 * sr))
    region = y[region_start:region_end]
    region_rms = np.sqrt(np.mean(region ** 2)) + 1e-9
    features["crest_factor"] = float(peak_val / region_rms)

    tc_start = max(0, peak_idx - int(0.1 * sr))
    tc_end = min(len(y), peak_idx + int(0.1 * sr))
    tc_region = y[tc_start:tc_end] ** 2
    tc_sum = np.sum(tc_region)
    if tc_sum > 0:
        indices = np.arange(len(tc_region))
        features["temporal_centroid"] = float(np.sum(indices * tc_region) / tc_sum / max(1, len(tc_region)))
    else:
        features["temporal_centroid"] = 0.5

    frame_size = max(1, int(0.005 * sr))
    n_frames = min(20, (len(y) - peak_idx) // frame_size)
    if n_frames >= 3:
        log_rms_vals = []
        for i in range(n_frames):
            start = peak_idx + i * frame_size
            frame = y[start:start + frame_size]
            frms = np.sqrt(np.mean(frame ** 2)) + 1e-9
            log_rms_vals.append(np.log10(frms))
        x = np.arange(n_frames, dtype=float)
        slope = float(np.polyfit(x, log_rms_vals, 1)[0])
        features["energy_decay_rate"] = slope
    else:
        features["energy_decay_rate"] = 0.0

    return features


def extract_subband_features(y: np.ndarray, sr: int = TARGET_SR) -> dict:
    features: dict = {}
    frame_size = int(0.05 * sr)
    peak_idx = int(np.argmax(np.abs(y)))
    onset_start = max(0, peak_idx - frame_size // 2)
    onset_end = min(len(y), onset_start + frame_size)
    onset_frame = y[onset_start:onset_end]

    n_fft = 2048
    windowed = onset_frame * np.hanning(len(onset_frame))
    padded = np.zeros(n_fft)
    padded[:len(windowed)] = windowed
    spectrum = np.abs(np.fft.rfft(padded)) ** 2
    freqs = np.fft.rfftfreq(n_fft, 1.0 / sr)

    bands = [
        ("low", 200, 800),
        ("mid", 800, 2500),
        ("high", 2500, 6000),
        ("vhigh", 6000, sr // 2),
    ]
    energies = {}
    for name, lo, hi in bands:
        mask = (freqs >= lo) & (freqs < hi)
        energies[name] = float(np.sum(spectrum[mask])) + 1e-12
        features[f"band_energy_{name}"] = float(np.log10(energies[name]))

    features["ratio_mid_low"] = float(np.log10(energies["mid"] / energies["low"]))
    features["ratio_high_mid"] = float(np.log10(energies["high"] / energies["mid"]))
    features["ratio_low_high"] = float(np.log10(energies["low"] / energies["high"]))
    band_names = list(energies.keys())
    features["band_peak_idx"] = float(np.argmax([energies[b] for b in band_names]))
    return features


def extract_features(y: np.ndarray, sr: int = TARGET_SR) -> dict:
    y = librosa.util.fix_length(y, size=CLIP_FRAMES)
    features: dict = {}

    mfccs = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=13)
    for i in range(13):
        features[f"mfcc_{i}_mean"] = float(np.mean(mfccs[i]))
        features[f"mfcc_{i}_std"] = float(np.std(mfccs[i]))

    centroid = librosa.feature.spectral_centroid(y=y, sr=sr)[0]
    features["spectral_centroid_mean"] = float(np.mean(centroid))
    features["spectral_centroid_std"] = float(np.std(centroid))

    rolloff = librosa.feature.spectral_rolloff(y=y, sr=sr, roll_percent=0.85)[0]
    features["spectral_rolloff_mean"] = float(np.mean(rolloff))
    features["spectral_rolloff_std"] = float(np.std(rolloff))

    zcr = librosa.feature.zero_crossing_rate(y)[0]
    features["zcr_mean"] = float(np.mean(zcr))
    features["zcr_std"] = float(np.std(zcr))

    rms = librosa.feature.rms(y=y)[0]
    features["rms_mean"] = float(np.mean(rms))
    features["rms_std"] = float(np.std(rms))

    onset_env = librosa.onset.onset_strength(y=y, sr=sr)
    features["onset_strength_max"] = float(np.max(onset_env))

    features.update(extract_transient_features(y, sr))
    features.update(extract_subband_features(y, sr))

    contrast = librosa.feature.spectral_contrast(y=y, sr=sr, n_bands=6)
    for i in range(contrast.shape[0]):
        features[f"spectral_contrast_band_{i}"] = float(np.mean(contrast[i]))

    flatness = librosa.feature.spectral_flatness(y=y)
    features["spectral_flatness_mean"] = float(np.mean(flatness))
    features["spectral_flatness_std"] = float(np.std(flatness))

    rms_frames = librosa.feature.rms(y=y)[0]
    peak_frame = int(np.argmax(rms_frames))
    for i in range(4):
        features[f"onset_mfcc_{i}"] = float(mfccs[i, min(peak_frame, mfccs.shape[1] - 1)])

    return features


def extract_clips_onset(y: np.ndarray, sr: int) -> tuple[list[np.ndarray], int]:
    onset_frames = librosa.onset.onset_detect(
        y=y,
        sr=sr,
        units="frames",
        hop_length=512,
        backtrack=True,
    )
    onset_times = librosa.frames_to_time(onset_frames, sr=sr, hop_length=512)
    if len(onset_times) == 0:
        return [], 0

    filtered: list[float] = [onset_times[0]]
    for t in onset_times[1:]:
        if t - filtered[-1] >= MIN_ONSET_GAP_S:
            filtered.append(t)

    clips = []
    dropped = 0
    for onset_time in filtered:
        start = max(0, int((onset_time - WINDOW_BEFORE_S) * sr))
        end = min(len(y), start + CLIP_FRAMES)
        clip = y[start:end]
        rms = np.sqrt(np.mean(clip ** 2))
        if rms < MIN_CLIP_RMS:
            dropped += 1
            continue
        clips.append(clip)
    return clips, dropped


def extract_clips_chunks(y: np.ndarray, sr: int) -> list[np.ndarray]:
    n_chunks = len(y) // sr
    return [y[i * sr : (i + 1) * sr] for i in range(n_chunks)]


def extract_clip_around_ms(y: np.ndarray, sr: int, timestamp_ms: int) -> np.ndarray:
    center_sample = int(round((timestamp_ms / 1000.0) * sr))
    start = max(0, center_sample - int(WINDOW_BEFORE_S * sr))
    end = min(len(y), start + CLIP_FRAMES)
    clip = y[start:end]
    return librosa.util.fix_length(clip, size=CLIP_FRAMES)


def mix_with_noise(signal: np.ndarray, noise_pool: list[np.ndarray], snr_db: float, rng: np.random.Generator) -> np.ndarray:
    if not noise_pool:
        return signal

    noise = noise_pool[rng.integers(len(noise_pool))].copy()
    noise = librosa.util.fix_length(noise, size=len(signal))

    sig_rms = np.sqrt(np.mean(signal ** 2)) + 1e-9
    noise_rms = np.sqrt(np.mean(noise ** 2)) + 1e-9
    target_noise_rms = sig_rms / (10 ** (snr_db / 20.0))
    noise_scaled = noise * (target_noise_rms / noise_rms)

    mixed = signal + noise_scaled
    peak = np.max(np.abs(mixed))
    if peak > 1.0:
        mixed /= peak
    return mixed.astype(np.float32)


def main() -> None:
    parser = argparse.ArgumentParser(description="Preprocess audio sessions.")
    parser.add_argument(
        "--bootstrap-unreviewed",
        action="store_true",
        help="Include unreviewed racket/noise/table/floor takes in the contact dataset as a temporary bootstrap.",
    )
    args = parser.parse_args()

    session_files = sorted(RAW_DIR.glob("audio_session_*.json"))
    archive_dir = RAW_DIR / "archive_m4a"
    if archive_dir.exists():
        session_files += sorted(archive_dir.glob("audio_session_*.json"))
    if not session_files:
        print(f"Inga sessioner hittades i {RAW_DIR}")
        sys.exit(1)

    rng = np.random.default_rng(42)
    multiclass_rows: list[dict] = []
    contact_rows: list[dict] = []
    errors = 0
    raw_multiclass = 0
    raw_contact = 0

    multiclass_positive_examples: list[dict] = []
    multiclass_noise_clips: list[np.ndarray] = []
    contact_positive_examples: list[dict] = []
    contact_negative_clips: list[np.ndarray] = []

    def append_row(
        rows: list[dict],
        label: str,
        clip: np.ndarray,
        sr: int,
        recorder: str,
        session_id: str,
        source_file: str,
        group_id: str,
        scenario_id: str,
        background_condition: str,
        take_index: int,
        target_duration_s: int,
        clip_id: str,
        augmentation: str,
        review_completed: bool | None = None,
        marker_source: str = "auto",
        anchor_rule: str | None = None,
        source_trust: str = "legacy_auto",
        review_status: str = "",
        contact_kind: str = "",
        not_racket_kind: str = "",
        bounce_side: str = "",
        binary_label: str = "",
        class_label: str = "",
        event_type: str = "",
        scenario: str = "",
        bounce_context: str = "",
        calibration_status: str = "",
        contact_confidence: str | float = "",
        surface_label: str = "",
        surface_confidence: str | float = "",
    ) -> bool:
        try:
            feats = extract_features(clip, sr)
        except Exception as e:
            print(f"  Feature-fel i {source_file} klipp {clip_id}: {e}")
            return False

        feats["label"] = label
        feats["binary_label"] = binary_label or binary_label_for_audio_label(label)
        feats["class_label"] = class_label or label
        feats["event_type"] = event_type or event_type_for_class_label(class_label or label)
        feats["recorder_name"] = recorder
        feats["session_id"] = session_id
        feats["source_file"] = source_file
        feats["group_id"] = group_id
        feats["scenario_id"] = scenario_id
        feats["background_condition"] = background_condition
        feats["take_index"] = take_index
        feats["target_duration_s"] = target_duration_s
        feats["clip_id"] = clip_id
        feats["augmentation"] = augmentation
        feats["source_trust"] = source_trust
        feats["review_status"] = review_status
        feats["contact_kind"] = contact_kind
        feats["not_racket_kind"] = not_racket_kind
        feats["bounce_side"] = bounce_side
        feats["scenario"] = scenario
        feats["bounce_context"] = bounce_context
        feats["calibration_status"] = calibration_status
        feats["contact_confidence"] = contact_confidence
        feats["surface_label"] = surface_label
        feats["surface_confidence"] = surface_confidence
        if review_completed is not None:
            feats["review_completed"] = review_completed
            feats["marker_source"] = marker_source
        if anchor_rule is not None:
            feats["anchor_rule"] = anchor_rule
        rows.append(feats)
        return True

    def auto_segment(label: str, y: np.ndarray, sr: int, session_mode: bool) -> tuple[list[np.ndarray], int]:
        if session_mode:
            if label == "noise":
                return extract_clips_chunks(y, sr), 0
            return extract_clips_onset(y, sr)
        return [y], 0

    for session_path in session_files:
        with open(session_path, encoding="utf-8") as f:
            session = json.load(f)

        session_dir = session_path.parent / session_path.stem
        recorder = session["session_meta"].get("recorder_name", "unknown")
        session_mode = session["session_meta"].get("clip_duration_ms", 1000) == 0

        for event in session["events"]:
            audio_path = session_dir / event["wav_filename"]
            if not audio_path.exists():
                print(f"  Saknas: {audio_path}")
                errors += 1
                continue

            label = str(event["label"])
            session_id = session_path.stem
            source_file = str(event["wav_filename"])
            group_id = str(event.get("group_id") or f"{session_id}:{source_file}")
            scenario_id = str(event.get("scenario_id", "legacy_unspecified"))
            recording_scenario = str(event.get("scenario") or "")
            bounce_context = str(event.get("bounce_context") or "")
            calibration_status = str(
                event.get("calibration_status") or session["session_meta"].get("calibration_status") or ""
            )
            background_condition = str(event.get("background_condition", "quiet"))
            take_index = int(event.get("take_index", 0))
            target_duration_s = int(event.get("target_duration_s", 0))
            review = event.get("review") or {}
            markers = review.get("markers") or []
            review_completed = bool(review.get("completed_at")) and len(markers) > 0
            anchor_rule = str(review.get("anchor_rule") or "attack_start")

            try:
                y, sr = load_audio(str(audio_path))
            except Exception as e:
                print(f"  Fel vid laddning av {audio_path.name}: {e}")
                errors += 1
                continue

            if review_completed:
                accepted_markers = 0
                for marker_idx, marker in enumerate(markers):
                    final_label = marker.get("final_label")
                    review_status = str(marker.get("review_status") or "confirmed")
                    if review_status in {"pending", "deleted", "filtered"} or final_label == "ignore":
                        continue
                    if final_label not in {"racket_contact", "not_racket_contact"}:
                        continue

                    timestamp_ms = int(marker.get("timestamp_ms", 0))
                    marker_source = str(marker.get("source", "auto"))
                    contact_kind = str(
                        marker.get("contact_kind")
                        or ("racket_bounce" if final_label == "racket_contact" else "")
                    )
                    not_racket_kind = str(
                        marker.get("not_racket_kind")
                        or (not_racket_kind_for(label, scenario_id) if final_label == "not_racket_contact" else "")
                    )
                    bounce_side = str(marker.get("bounce_side") or "unknown")
                    marker_class_label = str(marker.get("class_label") or "")
                    marker_event_type = str(marker.get("event_type") or "")
                    marker_contact_confidence = marker.get("contact_confidence", "")
                    marker_surface_label = str(marker.get("surface_label") or "")
                    marker_surface_confidence = marker.get("surface_confidence", "")
                    clip = extract_clip_around_ms(y, sr, timestamp_ms)
                    clip_id = f"{group_id}:review:{marker_idx:03d}"

                    multi_label = multiclass_label_for_marker(str(final_label), contact_kind, not_racket_kind)
                    export_class_label = marker_class_label or multi_label
                    export_event_type = marker_event_type or event_type_for_class_label(export_class_label)
                    if append_row(
                        multiclass_rows,
                        multi_label,
                        clip,
                        sr,
                        recorder,
                        session_id,
                        source_file,
                        group_id,
                        scenario_id,
                        background_condition,
                        take_index,
                        target_duration_s,
                        clip_id,
                        "none",
                        anchor_rule=anchor_rule,
                        source_trust="human_reviewed",
                        review_status=review_status,
                        contact_kind=contact_kind,
                        not_racket_kind=not_racket_kind,
                        bounce_side=bounce_side,
                        binary_label=str(final_label),
                        class_label=export_class_label,
                        event_type=export_event_type,
                        scenario=recording_scenario,
                        bounce_context=bounce_context,
                        calibration_status=calibration_status,
                        contact_confidence=marker_contact_confidence,
                        surface_label=marker_surface_label,
                        surface_confidence=marker_surface_confidence,
                    ):
                        raw_multiclass += 1
                        fixed_clip = librosa.util.fix_length(clip.copy(), size=TARGET_SR)
                        if multi_label == "noise":
                            multiclass_noise_clips.append(fixed_clip)
                        else:
                            multiclass_positive_examples.append({
                                "label": multi_label,
                                "clip": fixed_clip,
                                "recorder_name": recorder,
                                "session_id": session_id,
                                "source_file": source_file,
                                "group_id": group_id,
                                "scenario_id": scenario_id,
                                "background_condition": background_condition,
                                "take_index": take_index,
                                "target_duration_s": target_duration_s,
                                "source_trust": "human_reviewed_augmented",
                                "review_status": review_status,
                                "contact_kind": contact_kind,
                                "not_racket_kind": not_racket_kind,
                                "bounce_side": bounce_side,
                                "binary_label": str(final_label),
                                "class_label": export_class_label,
                                "event_type": export_event_type,
                                "scenario": recording_scenario,
                                "bounce_context": bounce_context,
                                "calibration_status": calibration_status,
                                "contact_confidence": marker_contact_confidence,
                                "surface_label": marker_surface_label,
                                "surface_confidence": marker_surface_confidence,
                            })

                    if append_row(
                        contact_rows,
                        str(final_label),
                        clip,
                        sr,
                        recorder,
                        session_id,
                        source_file,
                        group_id,
                        scenario_id,
                        background_condition,
                        take_index,
                        target_duration_s,
                        clip_id,
                        "none",
                        review_completed=True,
                        marker_source=marker_source,
                        anchor_rule=anchor_rule,
                        source_trust="human_reviewed",
                        review_status=review_status,
                        contact_kind=contact_kind,
                        not_racket_kind=not_racket_kind,
                        bounce_side=bounce_side,
                        binary_label=str(final_label),
                        class_label=export_class_label,
                        event_type=export_event_type,
                        scenario=recording_scenario,
                        bounce_context=bounce_context,
                        calibration_status=calibration_status,
                        contact_confidence=marker_contact_confidence,
                        surface_label=marker_surface_label,
                        surface_confidence=marker_surface_confidence,
                    ):
                        raw_contact += 1
                        accepted_markers += 1
                        fixed_clip = librosa.util.fix_length(clip.copy(), size=TARGET_SR)
                        if final_label == "racket_contact":
                            contact_positive_examples.append({
                                "label": "racket_contact",
                                "clip": fixed_clip,
                                "recorder_name": recorder,
                                "session_id": session_id,
                                "source_file": source_file,
                                "group_id": group_id,
                                "scenario_id": scenario_id,
                                "background_condition": background_condition,
                                "take_index": take_index,
                                "target_duration_s": target_duration_s,
                                "source_trust": "human_reviewed_augmented",
                                "review_status": review_status,
                                "contact_kind": contact_kind,
                                "not_racket_kind": not_racket_kind,
                                "bounce_side": bounce_side,
                                "binary_label": str(final_label),
                                "class_label": export_class_label,
                                "event_type": export_event_type,
                                    "scenario": recording_scenario,
                                    "bounce_context": bounce_context,
                                    "calibration_status": calibration_status,
                                    "contact_confidence": marker_contact_confidence,
                                    "surface_label": marker_surface_label,
                                    "surface_confidence": marker_surface_confidence,
                                })
                        else:
                            contact_negative_clips.append(fixed_clip)

                print(f"  {audio_path.name}: {accepted_markers} review-markorer")
                continue

            if label not in VALID_AUDIO_LABELS:
                print(f"  {audio_path.name}: hoppar över ogranskad label '{label}'")
                continue

            auto_clips, dropped = auto_segment(label, y, sr, session_mode)
            if not auto_clips:
                print(f"  Inga auto-klipp hittades i {audio_path.name} — hoppar över")
                continue

            for clip_idx, clip in enumerate(auto_clips):
                clip_id = f"{group_id}:{clip_idx:03d}"
                added_multi = append_row(
                    multiclass_rows,
                    label,
                    clip,
                    sr,
                    recorder,
                    session_id,
                    source_file,
                    group_id,
                    scenario_id,
                    background_condition,
                    take_index,
                    target_duration_s,
                    clip_id,
                    "none",
                    source_trust="legacy_auto",
                    contact_kind=contact_kind_for(label, scenario_id),
                    not_racket_kind=not_racket_kind_for(label, scenario_id),
                    bounce_side="unknown",
                    binary_label=binary_label_for_audio_label(label),
                    class_label=label,
                    event_type=event_type_for_class_label(label),
                    scenario=recording_scenario,
                    bounce_context=bounce_context,
                    calibration_status=calibration_status,
                )
                if added_multi:
                    raw_multiclass += 1
                    fixed_clip = librosa.util.fix_length(clip.copy(), size=TARGET_SR)
                    if label == "noise":
                        multiclass_noise_clips.append(fixed_clip)
                    else:
                        multiclass_positive_examples.append({
                            "label": label,
                            "clip": fixed_clip,
                            "recorder_name": recorder,
                            "session_id": session_id,
                            "source_file": source_file,
                            "group_id": group_id,
                            "scenario_id": scenario_id,
                            "background_condition": background_condition,
                            "take_index": take_index,
                            "target_duration_s": target_duration_s,
                            "source_trust": "legacy_auto_augmented",
                            "review_status": "",
                            "contact_kind": contact_kind_for(label, scenario_id),
                            "not_racket_kind": not_racket_kind_for(label, scenario_id),
                            "bounce_side": "unknown",
                            "binary_label": binary_label_for_audio_label(label),
                            "class_label": label,
                            "event_type": event_type_for_class_label(label),
                            "scenario": recording_scenario,
                            "bounce_context": bounce_context,
                            "calibration_status": calibration_status,
                        })

                should_bootstrap_contact = args.bootstrap_unreviewed and label in {"racket_bounce", "noise"}
                should_auto_negative = args.bootstrap_unreviewed and label in {"table_bounce", "floor_bounce"}
                if not should_bootstrap_contact and not should_auto_negative:
                    continue

                contact_label = "racket_contact" if label == "racket_bounce" else "not_racket_contact"
                contact_kind = contact_kind_for(label, scenario_id) if contact_label == "racket_contact" else ""
                not_racket_kind = not_racket_kind_for(label, scenario_id) if contact_label == "not_racket_contact" else ""
                added_contact = append_row(
                    contact_rows,
                    contact_label,
                    clip,
                    sr,
                    recorder,
                    session_id,
                    source_file,
                    group_id,
                    scenario_id,
                    background_condition,
                    take_index,
                    target_duration_s,
                    clip_id,
                    "none",
                    review_completed=False,
                    marker_source="auto",
                    source_trust="bootstrap",
                    review_status="",
                    contact_kind=contact_kind,
                    not_racket_kind=not_racket_kind,
                    bounce_side="unknown",
                    binary_label=contact_label,
                    class_label=label,
                    event_type=event_type_for_class_label(label),
                    scenario=recording_scenario,
                    bounce_context=bounce_context,
                    calibration_status=calibration_status,
                )
                if added_contact:
                    raw_contact += 1
                    fixed_clip = librosa.util.fix_length(clip.copy(), size=TARGET_SR)
                    if contact_label == "racket_contact":
                        contact_positive_examples.append({
                            "label": contact_label,
                            "clip": fixed_clip,
                            "recorder_name": recorder,
                            "session_id": session_id,
                            "source_file": source_file,
                            "group_id": group_id,
                            "scenario_id": scenario_id,
                            "background_condition": background_condition,
                            "take_index": take_index,
                            "target_duration_s": target_duration_s,
                            "source_trust": "bootstrap_augmented",
                            "review_status": "",
                            "contact_kind": contact_kind,
                            "not_racket_kind": not_racket_kind,
                            "bounce_side": "unknown",
                            "binary_label": contact_label,
                            "class_label": label,
                            "event_type": event_type_for_class_label(label),
                            "scenario": recording_scenario,
                            "bounce_context": bounce_context,
                            "calibration_status": calibration_status,
                        })
                    else:
                        contact_negative_clips.append(fixed_clip)

            if session_mode:
                drop_info = f" ({dropped} kastade, låg energi)" if dropped else ""
                print(f"  {audio_path.name}: {len(auto_clips)} auto-klipp ({label}){drop_info}")

    print("\n  Gain-augmentation: disabled in this iteration")

    multiclass_aug_count = 0
    if multiclass_noise_clips:
        for example in multiclass_positive_examples:
            for snr in AUGMENT_SNR_DB:
                mixed = mix_with_noise(example["clip"], multiclass_noise_clips, snr, rng)
                if append_row(
                    multiclass_rows,
                    example["label"],
                    mixed,
                    TARGET_SR,
                    example["recorder_name"],
                    example["session_id"],
                    example["source_file"],
                    example["group_id"],
                    example["scenario_id"],
                    example["background_condition"],
                    example["take_index"],
                    example["target_duration_s"],
                    f"{example['group_id']}:snr:{int(snr)}db",
                    f"snr_{int(snr)}db",
                    source_trust=example.get("source_trust", "legacy_auto_augmented"),
                    review_status=example.get("review_status", ""),
                    contact_kind=example.get("contact_kind", ""),
                    not_racket_kind=example.get("not_racket_kind", ""),
                    bounce_side=example.get("bounce_side", "unknown"),
                    binary_label=example.get("binary_label", ""),
                    class_label=example.get("class_label", example["label"]),
                    event_type=example.get("event_type", ""),
                    scenario=example.get("scenario", ""),
                    bounce_context=example.get("bounce_context", ""),
                    calibration_status=example.get("calibration_status", ""),
                ):
                    multiclass_aug_count += 1
        print(f"  Multiclass SNR-augmenterat: {multiclass_aug_count} extra klipp")
    else:
        print("  Multiclass: ingen brus-data — hoppar över SNR-augmentation")

    contact_aug_count = 0
    if contact_negative_clips:
        for example in contact_positive_examples:
            for snr in AUGMENT_SNR_DB:
                mixed = mix_with_noise(example["clip"], contact_negative_clips, snr, rng)
                if append_row(
                    contact_rows,
                    example["label"],
                    mixed,
                    TARGET_SR,
                    example["recorder_name"],
                    example["session_id"],
                    example["source_file"],
                    example["group_id"],
                    example["scenario_id"],
                    example["background_condition"],
                    example["take_index"],
                    example["target_duration_s"],
                    f"{example['group_id']}:contact_snr:{int(snr)}db",
                    f"snr_{int(snr)}db",
                    review_completed=True,
                    marker_source="augmented",
                    source_trust=example.get("source_trust", "human_reviewed_augmented"),
                    review_status=example.get("review_status", ""),
                    contact_kind=example.get("contact_kind", ""),
                    not_racket_kind=example.get("not_racket_kind", ""),
                    bounce_side=example.get("bounce_side", "unknown"),
                    binary_label=example.get("binary_label", example["label"]),
                    class_label=example.get("class_label", ""),
                    event_type=example.get("event_type", ""),
                    scenario=example.get("scenario", ""),
                    bounce_context=example.get("bounce_context", ""),
                    calibration_status=example.get("calibration_status", ""),
                ):
                    contact_aug_count += 1
        print(f"  Contact SNR-augmenterat: {contact_aug_count} extra klipp")
    else:
        print("  Contact: ingen negativ pool — hoppar över SNR-augmentation")

    if not multiclass_rows:
        print("Inga multiclass-klipp bearbetades — avbryter.")
        sys.exit(1)

    OUT_DIR.mkdir(parents=True, exist_ok=True)

    multi_df = pd.DataFrame(multiclass_rows)
    multi_df.to_csv(OUT_FILE, index=False)
    multi_source_counts = multi_df["source_trust"].value_counts().to_dict() if "source_trust" in multi_df.columns else {}
    print(f"\nMulticlass dataset sparat: {OUT_FILE}")
    print(f"  {len(multi_df)} rader totalt ({raw_multiclass} råa + {multiclass_aug_count} augmenterade)")
    print(f"  Etikettfördelning: {multi_df['label'].value_counts().to_dict()}")

    print(f"  Source/trust: {multi_source_counts}")

    if contact_rows:
        contact_df = pd.DataFrame(contact_rows)
        contact_df.to_csv(OUT_CONTACT_FILE, index=False)
        contact_source_counts = contact_df["source_trust"].value_counts().to_dict() if "source_trust" in contact_df.columns else {}
        hard_negative_counts = contact_df["not_racket_kind"].value_counts().to_dict() if "not_racket_kind" in contact_df.columns else {}
        print(f"\nContact dataset sparat: {OUT_CONTACT_FILE}")
        print(f"  {len(contact_df)} rader totalt ({raw_contact} råa + {contact_aug_count} augmenterade)")
        print(f"  Etikettfördelning: {contact_df['label'].value_counts().to_dict()}")
        print(f"  Source/trust: {contact_source_counts}")
        print(f"  Hard negatives: {hard_negative_counts}")
        if "scenario_id" in contact_df.columns:
            print(f"  Scenariofördelning: {contact_df['scenario_id'].value_counts().to_dict()}")
    else:
        print("\nContact dataset inte skapat: inga reviewed markers ännu.")
        print("  Kör collector-review först eller använd --bootstrap-unreviewed för en temporär startmodell.")

    if errors:
        print(f"\n  {errors} filer/klipp kunde inte bearbetas")


if __name__ == "__main__":
    main()
