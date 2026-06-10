"""
build_nr_dataset.py

Module 2 of the noise-robust racket bounce detector (see NR_SPEC.md).

Builds per-split clip datasets from reviewed markers plus hard negatives /
augmentation noise from the train bed takes:

  data/audio/processed/noise_robust/nr_train.csv
  data/audio/processed/noise_robust/nr_val.csv
  data/audio/processed/noise_robust/nr_test.csv
  data/audio/processed/noise_robust/nr_dataset_summary.json

Run:
  python skills/pingis-audio-classification/scripts/noise_robust/build_nr_dataset.py
  python skills/pingis-audio-classification/scripts/noise_robust/build_nr_dataset.py --limit-sessions id1,id2  # smoke testing only
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import librosa
import numpy as np
import pandas as pd

SCRIPT_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import preprocess_audio  # noqa: E402
import nr_config  # noqa: E402
import nr_features  # noqa: E402

ROOT_DIR = Path(__file__).resolve().parents[4]
if not (ROOT_DIR / "data" / "audio" / "raw").exists():
    raise RuntimeError(f"Repo root resolution failed: {ROOT_DIR}/data/audio/raw does not exist")

INVENTORY_PATH = ROOT_DIR / "data" / "audio" / "processed" / "audio_inventory_2026_06_10.json"
DEFAULT_OUT_DIR = ROOT_DIR / "data" / "audio" / "processed" / "noise_robust"

DIAGNOSTIC_EXCEPTION_SESSION = "audio_session_2026-05-29_001"  # audio markers documented valid

ROW_SPLITS = {"train", "val", "test"}
NEGATIVE_OVERLAP_BEFORE_S = 0.1
NEGATIVE_OVERLAP_AFTER_S = 0.2
BED_GATE_ONSET_RATIO = 1.3
BED_GATE_RETRIGGER_MS = 150
BED_GATE_CAP = 150
BED_CHUNK_PERIOD_S = 4.0

META_COLUMNS = [
    "clip_id",
    "split",
    "session_id",
    "wav_filename",
    "scenario_id",
    "background_condition",
    "label",
    "source",
    "anchor_ms",
    "jitter_ms",
    "augment",
    "aug_bed",
    "group_id",
    "close_event_bucket",
]


def load_inventory() -> dict:
    with open(INVENTORY_PATH, encoding="utf-8") as f:
        return json.load(f)


def load_session_json(json_path: str) -> dict:
    with open(ROOT_DIR / json_path, encoding="utf-8") as f:
        return json.load(f)


def snr_augment_name(snr_db: float) -> str:
    return f"snr{int(snr_db)}"


def apply_random_gain(clip: np.ndarray, rng: np.random.Generator) -> np.ndarray:
    """Random gain in +/- AUGMENT_GAIN_DB, peak-normalized if above 1."""
    gain_db = float(rng.uniform(-nr_config.AUGMENT_GAIN_DB, nr_config.AUGMENT_GAIN_DB))
    out = clip * (10.0 ** (gain_db / 20.0))
    peak = float(np.max(np.abs(out))) if len(out) else 0.0
    if peak > 1.0:
        out = out / peak
    return out.astype(np.float32)


def impact_reference_rms(clip: np.ndarray, sr: int) -> float:
    """RMS of the 60 ms around the clip's absolute peak (SNR reference)."""
    half = int(0.030 * sr)
    peak_idx = int(np.argmax(np.abs(clip)))
    start = max(0, peak_idx - half)
    end = min(len(clip), peak_idx + half)
    window = clip[start:end]
    return float(np.sqrt(np.mean(window.astype(np.float64) ** 2)) + 1e-9)


def mix_clip_with_bed_segment(
    clip: np.ndarray,
    segment: np.ndarray,
    snr_db: float,
    sr: int,
    rng: np.random.Generator,
) -> np.ndarray:
    """Mix a bed segment into the clip at a target SNR (reference = 60 ms
    around the clip peak), then random gain and peak normalization."""
    sig_rms = impact_reference_rms(clip, sr)
    noise_rms = float(np.sqrt(np.mean(segment.astype(np.float64) ** 2)) + 1e-9)
    target_noise_rms = sig_rms / (10.0 ** (snr_db / 20.0))
    mixed = clip.astype(np.float64) + segment.astype(np.float64) * (target_noise_rms / noise_rms)
    return apply_random_gain(mixed.astype(np.float32), rng)


def random_bed_segment(
    bed_pool: list[tuple[str, np.ndarray]],
    rng: np.random.Generator,
) -> tuple[str, np.ndarray]:
    """Pick a random bed and a random 6615-sample segment from it."""
    bed_name, bed_audio = bed_pool[int(rng.integers(len(bed_pool)))]
    max_start = len(bed_audio) - nr_config.CLIP_SAMPLES
    start = int(rng.integers(0, max_start + 1)) if max_start > 0 else 0
    segment = bed_audio[start:start + nr_config.CLIP_SAMPLES]
    if len(segment) < nr_config.CLIP_SAMPLES:
        segment = librosa.util.fix_length(segment, size=nr_config.CLIP_SAMPLES)
    return bed_name, segment


def make_row(
    *,
    clip: np.ndarray,
    clip_id: str,
    split: str,
    session_id: str,
    wav_filename: str,
    scenario_id: str,
    background_condition: str,
    label: str,
    source: str,
    anchor_ms: float,
    jitter_ms: int,
    augment: str,
    aug_bed: str,
    close_event_bucket: str,
) -> dict:
    row = {
        "clip_id": clip_id,
        "split": split,
        "session_id": session_id,
        "wav_filename": wav_filename,
        "scenario_id": scenario_id,
        "background_condition": background_condition,
        "label": label,
        "source": source,
        "anchor_ms": anchor_ms,
        "jitter_ms": jitter_ms,
        "augment": augment,
        "aug_bed": aug_bed,
        "group_id": session_id,
        "close_event_bucket": close_event_bucket,
    }
    row.update(nr_features.extract_all_features(clip))
    return row


def snrs_for_label(label: str, rng: np.random.Generator) -> list[float]:
    """gain copy is handled separately; this returns the SNR list per row:
    2 SNRs sampled without replacement, or all 4 for floor_bounce rows."""
    if label == "floor_bounce":
        return [float(s) for s in nr_config.AUGMENT_SNR_DB]
    picked = rng.choice(nr_config.AUGMENT_SNR_DB, size=2, replace=False)
    return [float(s) for s in picked]


def main() -> None:
    parser = argparse.ArgumentParser(description="Build noise-robust clip datasets per split.")
    parser.add_argument(
        "--out-dir",
        default=str(DEFAULT_OUT_DIR),
        help="Output directory for nr_train/nr_val/nr_test CSVs.",
    )
    parser.add_argument("--seed", type=int, default=nr_config.RNG_SEED, help="RNG seed.")
    parser.add_argument(
        "--limit-sessions",
        default="",
        help=(
            "TEMPORARY testing flag (not in NR_SPEC.md): comma-separated session ids; "
            "restricts both reviewed-marker sessions and train bed takes to those "
            "sessions. For smoke tests only - never use for a real dataset build."
        ),
    )
    args = parser.parse_args()

    out_dir = Path(args.out_dir)
    rng = np.random.default_rng(args.seed)
    limit_sessions = {s.strip() for s in args.limit_sessions.split(",") if s.strip()}

    inventory = load_inventory()
    sessions = sorted(inventory["sessions"], key=lambda s: s["session_id"])
    sessions_by_id = {s["session_id"]: s for s in sessions}

    skipped = {
        "sessions_archive_m4a": 0,
        "sessions_excluded": 0,
        "sessions_diagnostic": 0,
        "sessions_fp_bed_replay_only": 0,
        "sessions_unassigned_no_markers": 0,
        "sessions_limited_out": 0,
        "events_missing_wav": 0,
        "events_without_trainable_markers": 0,
        "markers_not_trainable": 0,
        "markers_negative_overlap": 0,
        "bed_gate_triggers_capped": 0,
        "nan_or_inf_replaced": 0,
    }

    # ---- Split classification + fail-loudly check ----------------------
    unassigned_with_markers: list[str] = []
    row_sessions: list[tuple[str, dict, str]] = []  # (session_id, inventory session, split)
    for session in sessions:
        session_id = session["session_id"]
        split = nr_config.split_for_session(session_id)
        if session.get("group") == "archive_m4a":
            skipped["sessions_archive_m4a"] += 1
            continue
        if split == "unassigned":
            if int(session.get("totals", {}).get("n_markers", 0)) > 0:
                unassigned_with_markers.append(session_id)
            else:
                skipped["sessions_unassigned_no_markers"] += 1
            continue
        if split == "excluded":
            skipped["sessions_excluded"] += 1
            continue
        if session.get("is_diagnostic") and session_id != DIAGNOSTIC_EXCEPTION_SESSION:
            skipped["sessions_diagnostic"] += 1
            continue
        if split in ("val_fp_bed", "test_fp_bed"):
            skipped["sessions_fp_bed_replay_only"] += 1
            continue
        if split in ROW_SPLITS:
            if limit_sessions and session_id not in limit_sessions:
                skipped["sessions_limited_out"] += 1
                continue
            row_sessions.append((session_id, session, split))

    if unassigned_with_markers:
        print("ERROR: sessions with reviewed markers have no split assignment in nr_config.py:")
        for session_id in unassigned_with_markers:
            print(f"  - {session_id}")
        sys.exit(1)

    # ---- Bed pool (loaded once, reused for hard negatives + mixing) ----
    bed_takes = [
        (sid, wav) for sid, wav in nr_config.TRAIN_BED_TAKES
        if not limit_sessions or sid in limit_sessions
    ]
    bed_pool: list[tuple[str, np.ndarray]] = []  # ("session:wav", audio)
    bed_take_meta: list[dict] = []
    for bed_session_id, bed_wav in bed_takes:
        session = sessions_by_id.get(bed_session_id)
        if session is None:
            raise RuntimeError(f"Bed session not in inventory: {bed_session_id}")
        wav_path = ROOT_DIR / session["media_dir"] / bed_wav
        if not wav_path.exists():
            raise RuntimeError(f"Bed wav missing: {wav_path}")
        session_json = load_session_json(session["json_path"])
        event = next((e for e in session_json["events"] if e.get("wav_filename") == bed_wav), None)
        if event is None:
            raise RuntimeError(f"Bed wav has no event entry: {bed_session_id}/{bed_wav}")
        y, sr = preprocess_audio.load_audio(str(wav_path))
        bed_name = f"{bed_session_id}:{bed_wav}"
        bed_pool.append((bed_name, y.astype(np.float32)))
        bed_take_meta.append({
            "session_id": bed_session_id,
            "wav_filename": bed_wav,
            "scenario_id": str(event.get("scenario_id") or ""),
            "background_condition": str(event.get("background_condition") or "(none)"),
            "audio": y.astype(np.float32),
            "sr": sr,
        })
        print(f"Bed loaded: {bed_name} ({len(y) / sr:.1f}s)")

    rows: list[dict] = []
    clean_train_rows = 0
    augmented_rows = 0

    def add_augmented_copies(
        clip: np.ndarray,
        base_clip_id: str,
        *,
        with_snr: bool,
        label: str,
        sr: int,
        meta: dict,
    ) -> None:
        nonlocal augmented_rows
        gain_clip = apply_random_gain(clip, rng)
        rows.append(make_row(
            clip=gain_clip,
            clip_id=f"{base_clip_id}:gain",
            augment="gain",
            aug_bed="",
            label=label,
            **meta,
        ))
        augmented_rows += 1
        if not with_snr or not bed_pool:
            return
        for snr_db in snrs_for_label(label, rng):
            bed_name, segment = random_bed_segment(bed_pool, rng)
            mixed = mix_clip_with_bed_segment(clip, segment, snr_db, sr, rng)
            rows.append(make_row(
                clip=mixed,
                clip_id=f"{base_clip_id}:{snr_augment_name(snr_db)}",
                augment=snr_augment_name(snr_db),
                aug_bed=bed_name,
                label=label,
                **meta,
            ))
            augmented_rows += 1

    # ---- Rows from reviewed markers (train/val/test) --------------------
    for session_id, session, split in row_sessions:
        session_json = load_session_json(session["json_path"])
        media_dir = ROOT_DIR / session["media_dir"]
        events = sorted(session_json.get("events", []), key=lambda e: str(e.get("wav_filename") or ""))
        for event in events:
            wav_filename = str(event.get("wav_filename") or "")
            review = event.get("review") or {}
            markers = review.get("markers") or []
            trainable = [
                (idx, marker) for idx, marker in enumerate(markers)
                if preprocess_audio.is_trainable_review_marker(marker)
            ]
            skipped["markers_not_trainable"] += len(markers) - len(trainable)
            if not trainable:
                skipped["events_without_trainable_markers"] += 1
                continue
            wav_path = media_dir / wav_filename
            if not wav_path.exists():
                skipped["events_missing_wav"] += 1
                print(f"  Missing wav: {wav_path}")
                continue

            y, sr = preprocess_audio.load_audio(str(wav_path))
            event_label = str(event.get("label") or "")
            scenario_id = str(event.get("scenario_id") or "")
            background_condition = str(event.get("background_condition") or "quiet")

            racket_ts = preprocess_audio.reviewed_racket_timestamps(markers)
            review_ts = preprocess_audio.reviewed_marker_timestamps(markers)

            trainable.sort(key=lambda pair: (int(pair[1].get("timestamp_ms", 0)), pair[0]))
            for marker_idx, marker in trainable:
                final_label = str(marker.get("final_label") or "")
                contact_kind = str(marker.get("contact_kind") or "") or preprocess_audio.contact_kind_for(event_label, scenario_id)
                not_racket_kind = str(marker.get("not_racket_kind") or "") or preprocess_audio.not_racket_kind_for(event_label, scenario_id)
                label = preprocess_audio.multiclass_label_for_marker(final_label, contact_kind, not_racket_kind)

                if preprocess_audio.negative_marker_overlaps_racket(
                    marker,
                    racket_ts,
                    before_s=NEGATIVE_OVERLAP_BEFORE_S,
                    after_s=NEGATIVE_OVERLAP_AFTER_S,
                ):
                    skipped["markers_negative_overlap"] += 1
                    continue

                anchor_ms = int(marker.get("timestamp_ms", 0))
                jitter_ms = int(rng.integers(-nr_config.ANCHOR_JITTER_MS, nr_config.ANCHOR_JITTER_MS + 1)) if split == "train" else 0
                anchor_sample = int(round((anchor_ms + jitter_ms) / 1000.0 * sr))
                clip = nr_features.extract_live_clip(y, anchor_sample)

                spacing = preprocess_audio.spacing_metadata_for_timestamp(review_ts, anchor_ms)
                close_event_bucket = str(spacing["close_event_bucket"])

                base_clip_id = f"{session_id}:{wav_filename}:rm{marker_idx:03d}"
                meta = {
                    "split": split,
                    "session_id": session_id,
                    "wav_filename": wav_filename,
                    "scenario_id": scenario_id,
                    "background_condition": background_condition,
                    "source": "reviewed_marker",
                    "anchor_ms": anchor_ms,
                    "jitter_ms": jitter_ms,
                    "close_event_bucket": close_event_bucket,
                }
                rows.append(make_row(
                    clip=clip,
                    clip_id=base_clip_id,
                    augment="none",
                    aug_bed="",
                    label=label,
                    **meta,
                ))
                if split == "train":
                    clean_train_rows += 1
                    add_augmented_copies(clip, base_clip_id, with_snr=True, label=label, sr=sr, meta=meta)
        print(f"{session_id} [{split}]: processed")

    # ---- Hard negatives from train beds ---------------------------------
    for bed in bed_take_meta:
        session_id = bed["session_id"]
        wav_filename = bed["wav_filename"]
        y = bed["audio"]
        sr = bed["sr"]
        triggers = nr_features.simulate_gate(
            y,
            sr,
            onset_ratio=BED_GATE_ONSET_RATIO,
            retrigger_ms=BED_GATE_RETRIGGER_MS,
            mode="broadband",
            spectral_gate=False,
        )
        # Drop triggers inside the first 100 ms: their clips would start with
        # left zero-padding, a pattern that would otherwise exist almost only
        # in the noise class and could be learned as a provenance shortcut.
        n_before = len(triggers)
        triggers = [t for t in triggers if t["onset_sample"] >= nr_config.CLIP_PRE_SAMPLES]
        skipped["bed_gate_triggers_early_dropped"] = (
            skipped.get("bed_gate_triggers_early_dropped", 0) + (n_before - len(triggers))
        )
        if len(triggers) > BED_GATE_CAP:
            skipped["bed_gate_triggers_capped"] += len(triggers) - BED_GATE_CAP
            keep = np.linspace(0, len(triggers) - 1, BED_GATE_CAP).astype(int)
            selected = [(int(i), triggers[int(i)]) for i in keep]
        else:
            selected = list(enumerate(triggers))

        base_meta = {
            "split": "train",
            "session_id": session_id,
            "wav_filename": wav_filename,
            "scenario_id": bed["scenario_id"],
            "background_condition": bed["background_condition"],
            "jitter_ms": 0,
            "close_event_bucket": "",
        }
        for trig_idx, trigger in selected:
            clip = nr_features.extract_live_clip(y, trigger["onset_sample"])
            base_clip_id = f"{session_id}:{wav_filename}:gate{trig_idx:04d}"
            meta = dict(base_meta, source="bed_gate", anchor_ms=round(float(trigger["onset_ms"]), 3))
            rows.append(make_row(
                clip=clip,
                clip_id=base_clip_id,
                augment="none",
                aug_bed="",
                label="noise",
                **meta,
            ))
            add_augmented_copies(clip, base_clip_id, with_snr=False, label="noise", sr=sr, meta=meta)

        # Non-overlapping random 300 ms chunks, 1 per 4 s of bed audio.
        block_samples = int(BED_CHUNK_PERIOD_S * sr)
        n_blocks = len(y) // block_samples
        n_chunks = 0
        for block_idx in range(n_blocks):
            start_min = block_idx * block_samples
            start_max = (block_idx + 1) * block_samples - nr_config.CLIP_SAMPLES
            if start_max < start_min:
                continue
            start = int(rng.integers(start_min, start_max + 1))
            onset_sample = start + nr_config.CLIP_PRE_SAMPLES
            clip = nr_features.extract_live_clip(y, onset_sample)
            base_clip_id = f"{session_id}:{wav_filename}:chunk{block_idx:03d}"
            meta = dict(base_meta, source="bed_chunk", anchor_ms=round(onset_sample / sr * 1000.0, 3))
            rows.append(make_row(
                clip=clip,
                clip_id=base_clip_id,
                augment="none",
                aug_bed="",
                label="noise",
                **meta,
            ))
            add_augmented_copies(clip, base_clip_id, with_snr=False, label="noise", sr=sr, meta=meta)
            n_chunks += 1
        print(f"Bed {session_id}:{wav_filename}: {len(selected)} gate triggers, {n_chunks} chunks")

    if not rows:
        print("No rows produced - aborting.")
        sys.exit(1)

    # ---- Validate + write ------------------------------------------------
    feature_columns = nr_features.all_feature_names()
    columns = META_COLUMNS + feature_columns
    df = pd.DataFrame(rows, columns=columns)

    clip_ids = df["clip_id"]
    if clip_ids.duplicated().any():
        dupes = sorted(clip_ids[clip_ids.duplicated()].unique().tolist())
        raise AssertionError(f"clip_id collisions: {dupes[:20]} (total {len(dupes)})")

    feat = df[feature_columns]
    bad_mask = ~np.isfinite(feat.to_numpy(dtype=np.float64))
    n_bad = int(bad_mask.sum())
    if n_bad:
        skipped["nan_or_inf_replaced"] = n_bad
        cleaned = feat.to_numpy(dtype=np.float64)
        cleaned[bad_mask] = 0.0
        df[feature_columns] = cleaned
    assert np.isfinite(df[feature_columns].to_numpy(dtype=np.float64)).all(), "non-finite feature values remain"

    out_dir.mkdir(parents=True, exist_ok=True)
    split_files = {}
    for split in ("train", "val", "test"):
        split_df = df[df["split"] == split].reset_index(drop=True)
        out_path = out_dir / f"nr_{split}.csv"
        split_df.to_csv(out_path, index=False)
        split_files[split] = str(out_path)
        print(f"Wrote {out_path} ({len(split_df)} rows)")

    # ---- Summary ----------------------------------------------------------
    count_groups = (
        df.groupby(["split", "label", "background_condition", "augment"], sort=True)
        .size()
        .reset_index(name="rows")
    )
    split_label_matrix = df.pivot_table(
        index="label", columns="split", values="clip_id", aggfunc="count", fill_value=0,
    )

    summary = {
        "seed": args.seed,
        "limit_sessions": sorted(limit_sessions),
        "out_dir": str(out_dir),
        "n_rows_total": int(len(df)),
        "n_rows_per_split": {split: int((df["split"] == split).sum()) for split in ("train", "val", "test")},
        "clean_train_reviewed_rows": clean_train_rows,
        "augmented_rows": augmented_rows,
        "row_counts": count_groups.to_dict(orient="records"),
        "split_label_matrix": {
            str(label): {str(col): int(v) for col, v in row.items()}
            for label, row in split_label_matrix.iterrows()
        },
        "skipped": skipped,
        "bed_takes_used": [f"{sid}:{wav}" for sid, wav in bed_takes],
        "files": split_files,
    }
    summary_path = out_dir / "nr_dataset_summary.json"
    with open(summary_path, "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)
    print(f"Wrote {summary_path}")

    print("\nSplit x label matrix:")
    print(split_label_matrix.to_string())
    print("\nRow counts (split x label x background x augment):")
    print(count_groups.to_string(index=False))
    print("\nSkipped counters:")
    for key, value in skipped.items():
        print(f"  {key}: {value}")


if __name__ == "__main__":
    main()
