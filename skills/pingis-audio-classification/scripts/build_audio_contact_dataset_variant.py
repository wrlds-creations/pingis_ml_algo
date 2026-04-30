"""
build_audio_contact_dataset_variant.py

Build binary racket-contact dataset variants from raw audio sessions.

Modes:
  reviewed_only
    Use only reviewed markers.

  trusted_legacy
    Use reviewed markers plus explicit one-second legacy clips.

  all_legacy
    Use reviewed markers plus all legacy clips, including auto/onset-derived ones.

Run:
  python skills/pingis-audio-classification/scripts/build_audio_contact_dataset_variant.py --mode reviewed_only
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import librosa
import numpy as np
import pandas as pd

from preprocess_audio import (
    AUGMENT_SNR_DB,
    OUT_DIR,
    RAW_DIR,
    TARGET_SR,
    extract_clip_around_ms,
    extract_clips_chunks,
    extract_clips_onset,
    extract_features,
    load_audio,
    mix_with_noise,
)

TARGET_LABELS = {"racket_bounce", "table_bounce", "floor_bounce", "noise"}
VARIANT_DIR = OUT_DIR / "contact_variants"


def append_contact_row(
    rows: list[dict],
    *,
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
    contact_origin: str = "unknown",
) -> bool:
    try:
        feats = extract_features(clip, sr)
    except Exception as exc:
        print(f"  Feature-fel i {source_file} klipp {clip_id}: {exc}")
        return False

    feats["label"] = label
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
    feats["contact_origin"] = contact_origin
    if review_completed is not None:
        feats["review_completed"] = review_completed
        feats["marker_source"] = marker_source
    if anchor_rule is not None:
        feats["anchor_rule"] = anchor_rule
    rows.append(feats)
    return True


def iter_session_files() -> list[Path]:
    session_files = sorted(RAW_DIR.glob("audio_session_*.json"))
    archive_dir = RAW_DIR / "archive_m4a"
    if archive_dir.exists():
        session_files += sorted(archive_dir.glob("audio_session_*.json"))
    return session_files


def contact_label_from_audio_label(label: str) -> str | None:
    if label not in TARGET_LABELS:
        return None
    return "racket_contact" if label == "racket_bounce" else "not_racket_contact"


def should_include_unreviewed(mode: str, session_mode: bool) -> bool:
    if mode == "reviewed_only":
        return False
    if mode == "trusted_legacy":
        return not session_mode
    if mode == "all_legacy":
        return True
    raise ValueError(f"Unknown mode: {mode}")


def build_variant(mode: str) -> tuple[pd.DataFrame, int, int]:
    rows: list[dict] = []
    positive_examples: list[dict] = []
    negative_clips: list[np.ndarray] = []
    raw_count = 0
    errors = 0

    for session_path in iter_session_files():
        with session_path.open(encoding="utf-8") as fh:
            session = json.load(fh)

        session_dir = session_path.parent / session_path.stem
        recorder = session["session_meta"].get("recorder_name", "unknown")
        session_mode = session["session_meta"].get("clip_duration_ms", 1000) == 0

        for event in session.get("events", []):
            audio_path = session_dir / event["wav_filename"]
            if not audio_path.exists():
                print(f"  Saknas: {audio_path}")
                errors += 1
                continue

            label = str(event.get("label", ""))
            contact_label = contact_label_from_audio_label(label)
            if contact_label is None:
                continue

            session_id = session_path.stem
            source_file = str(event["wav_filename"])
            group_id = str(event.get("group_id") or f"{session_id}:{source_file}")
            scenario_id = str(event.get("scenario_id", "legacy_unspecified"))
            background_condition = str(event.get("background_condition", "quiet"))
            take_index = int(event.get("take_index", 0))
            target_duration_s = int(event.get("target_duration_s", 0))
            review = event.get("review") or {}
            markers = review.get("markers") or []
            review_completed = bool(review.get("completed_at")) and len(markers) > 0
            anchor_rule = str(review.get("anchor_rule") or "attack_start")

            try:
                y, sr = load_audio(str(audio_path))
            except Exception as exc:
                print(f"  Fel vid laddning av {audio_path.name}: {exc}")
                errors += 1
                continue

            if review_completed:
                accepted_markers = 0
                for marker_idx, marker in enumerate(markers):
                    final_label = str(marker.get("final_label", "ignore"))
                    if final_label == "ignore":
                        continue

                    timestamp_ms = int(marker.get("timestamp_ms", 0))
                    marker_source = str(marker.get("source", "auto"))
                    clip = extract_clip_around_ms(y, sr, timestamp_ms)
                    clip_id = f"{group_id}:review:{marker_idx:03d}"
                    if append_contact_row(
                        rows,
                        label=final_label,
                        clip=clip,
                        sr=sr,
                        recorder=recorder,
                        session_id=session_id,
                        source_file=source_file,
                        group_id=group_id,
                        scenario_id=scenario_id,
                        background_condition=background_condition,
                        take_index=take_index,
                        target_duration_s=target_duration_s,
                        clip_id=clip_id,
                        augmentation="none",
                        review_completed=True,
                        marker_source=marker_source,
                        anchor_rule=anchor_rule,
                        contact_origin="reviewed_marker",
                    ):
                        raw_count += 1
                        accepted_markers += 1
                        fixed_clip = librosa.util.fix_length(clip.copy(), size=TARGET_SR)
                        if final_label == "racket_contact":
                            positive_examples.append(
                                {
                                    "label": final_label,
                                    "clip": fixed_clip,
                                    "recorder_name": recorder,
                                    "session_id": session_id,
                                    "source_file": source_file,
                                    "group_id": group_id,
                                    "scenario_id": scenario_id,
                                    "background_condition": background_condition,
                                    "take_index": take_index,
                                    "target_duration_s": target_duration_s,
                                    "contact_origin": "reviewed_marker",
                                }
                            )
                        else:
                            negative_clips.append(fixed_clip)
                print(f"  {audio_path.name}: {accepted_markers} reviewed contact clips")
                continue

            if not should_include_unreviewed(mode, session_mode):
                continue

            if session_mode:
                if label == "noise":
                    clips = extract_clips_chunks(y, sr)
                else:
                    clips, _ = extract_clips_onset(y, sr)
                origin = "legacy_auto"
            else:
                clips = [y]
                origin = "legacy_explicit"

            for clip_idx, clip in enumerate(clips):
                clip_id = f"{group_id}:{clip_idx:03d}"
                if append_contact_row(
                    rows,
                    label=contact_label,
                    clip=clip,
                    sr=sr,
                    recorder=recorder,
                    session_id=session_id,
                    source_file=source_file,
                    group_id=group_id,
                    scenario_id=scenario_id,
                    background_condition=background_condition,
                    take_index=take_index,
                    target_duration_s=target_duration_s,
                    clip_id=clip_id,
                    augmentation="none",
                    review_completed=False,
                    marker_source="auto",
                    contact_origin=origin,
                ):
                    raw_count += 1
                    fixed_clip = librosa.util.fix_length(clip.copy(), size=TARGET_SR)
                    if contact_label == "racket_contact":
                        positive_examples.append(
                            {
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
                                "contact_origin": origin,
                            }
                        )
                    else:
                        negative_clips.append(fixed_clip)

    rng = np.random.default_rng(42)
    aug_count = 0
    if positive_examples and negative_clips:
        for example in positive_examples:
            for snr in AUGMENT_SNR_DB:
                mixed = mix_with_noise(example["clip"], negative_clips, snr, rng)
                clip_id = f"{example['group_id']}:contact_snr:{int(snr)}db"
                if append_contact_row(
                    rows,
                    label=example["label"],
                    clip=mixed,
                    sr=TARGET_SR,
                    recorder=example["recorder_name"],
                    session_id=example["session_id"],
                    source_file=example["source_file"],
                    group_id=example["group_id"],
                    scenario_id=example["scenario_id"],
                    background_condition=example["background_condition"],
                    take_index=example["take_index"],
                    target_duration_s=example["target_duration_s"],
                    clip_id=clip_id,
                    augmentation=f"snr_{int(snr)}db",
                    review_completed=True,
                    marker_source="augmented",
                    contact_origin="augmented_from_positive",
                ):
                    aug_count += 1

    return pd.DataFrame(rows), raw_count, aug_count


def main() -> None:
    parser = argparse.ArgumentParser(description="Build a binary audio contact dataset variant.")
    parser.add_argument(
        "--mode",
        choices=["reviewed_only", "trusted_legacy", "all_legacy"],
        required=True,
        help="Which legacy inclusion strategy to use.",
    )
    parser.add_argument(
        "--output",
        default="",
        help="Optional output CSV path. Defaults to data/audio/processed/contact_variants/<mode>.csv",
    )
    args = parser.parse_args()

    output = Path(args.output) if args.output else VARIANT_DIR / f"{args.mode}.csv"
    output.parent.mkdir(parents=True, exist_ok=True)

    df, raw_count, aug_count = build_variant(args.mode)
    if df.empty:
        raise SystemExit(f"Inga rader skapades för mode={args.mode}")

    df.to_csv(output, index=False)
    print(f"\nContact variant saved: {output}")
    print(f"  mode={args.mode}")
    print(f"  rows total={len(df)} | raw={raw_count} | augmented={aug_count}")
    print(f"  labels={df['label'].value_counts().to_dict()}")
    if "contact_origin" in df.columns:
        print(f"  origins={df['contact_origin'].value_counts().to_dict()}")
    if "scenario_id" in df.columns:
        scenario_counts = df["scenario_id"].value_counts().to_dict()
        print(f"  scenarios={scenario_counts}")


if __name__ == "__main__":
    main()
