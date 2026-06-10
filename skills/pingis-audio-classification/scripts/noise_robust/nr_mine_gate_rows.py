"""
nr_mine_gate_rows.py

Gate-aligned training-row mining for the noise-robust detector (companion to
build_nr_dataset.py; shares its helpers so the SNR-mix / gain augmentation
code path is byte-for-byte the same logic).

Motivation: build_nr_dataset.py anchors training clips on reviewed marker
timestamps, but the live runtime anchors clips on GATE trigger frames. This
script runs the live gate configuration over every TRAIN session event WAV
that has reviewed markers, plus every TRAIN bed take, and turns every gate
trigger into a training row whose clip is aligned exactly like the runtime
(onset at sample 2205).

Gate config (both sessions and beds):
  simulate_gate(onset_ratio=1.5, retrigger_ms=120, abs_min_rms=0.0015,
                mode="bandpass", spectral_gate=False)

Trigger labeling (TRAIN sessions, events with trainable reviewed markers):
  - nearest trainable reviewed marker within 140 ms -> the marker's
    multiclass label (racket_contact -> racket_bounce via
    multiclass_label_for_marker with the contact_kind fallback, exactly like
    build_nr_dataset.py; not_racket_contact -> table_bounce / floor_bounce /
    noise). Negative markers overlapping a racket window (0.1 s before /
    0.2 s after) are excluded from the candidate set, mirroring the builder.
  - no trainable marker within 140 ms but a trainable racket marker within
    140-300 ms -> DROPPED entirely (ambiguous attack/decay region; counted).
  - no trainable marker within 140 ms but a NON-trainable marker within
    140 ms -> DROPPED (reviewer flagged something untrainable there; counted).
  - otherwise -> noise.

TRAIN bed takes (nr_config.TRAIN_BED_TAKES): all triggers with
onset_sample >= 2205 -> noise, capped at 200 per bed by even subsampling.

Augmentation (mirrors build_nr_dataset.py exactly, same helper functions and
single rng seeded from nr_config.RNG_SEED):
  - mined racket/table/floor rows: 1 gain copy + 2 SNR copies sampled from
    AUGMENT_SNR_DB without replacement (floor_bounce rows: all 4 SNRs),
    mixed against random TRAIN bed segments with the 60 ms peak-region RMS
    as SNR reference.
  - mined noise rows (sessions and beds): 1 gain copy.

Output: data/audio/processed/noise_robust/nr_train_mined.csv with EXACTLY
the nr_train.csv column schema (asserted against the nr_train.csv header).
source = "gate_mined" (session rows) or "gate_mined_bed" (bed rows),
split = "train", anchor_ms = trigger onset ms, jitter_ms = 0,
group_id = session_id, close_event_bucket = "".

Run:
  python skills/pingis-audio-classification/scripts/noise_robust/nr_mine_gate_rows.py
  # smoke testing only (TEMPORARY flag, never for a real mine):
  python ... --limit-sessions audio_session_2026-05-06_010,audio_session_2026-04-09_003
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import preprocess_audio  # noqa: E402
import nr_config  # noqa: E402
import nr_features  # noqa: E402
import build_nr_dataset  # noqa: E402  (shared row/augment/bed helpers)

ROOT_DIR = Path(__file__).resolve().parents[4]
if not (ROOT_DIR / "data" / "audio" / "raw").exists():
    raise RuntimeError(f"Repo root resolution failed: {ROOT_DIR}/data/audio/raw does not exist")

NR_DIR = ROOT_DIR / "data" / "audio" / "processed" / "noise_robust"
DEFAULT_OUT_CSV = NR_DIR / "nr_train_mined.csv"
NR_TRAIN_CSV = NR_DIR / "nr_train.csv"

# Live gate configuration mined against (same for sessions and beds).
MINE_GATE_PARAMS = {
    "onset_ratio": 1.5,
    "retrigger_ms": 120,
    "abs_min_rms": 0.0015,
    "mode": "bandpass",
    "spectral_gate": False,
}

MATCH_MS = float(nr_config.MATCH_TOLERANCE_MS)  # 140 ms marker match window
RACKET_AMBIGUOUS_MS = 300.0  # 140-300 ms from a racket marker: drop entirely
BED_TRIGGER_CAP = 200


def label_for_trigger(
    onset_ms: float,
    candidates: list[tuple[int, str]],
    racket_ts: list[int],
    non_trainable_ts: list[int],
) -> tuple[str | None, str | None]:
    """Return (label, drop_reason); exactly one of the two is not None.

    candidates: (timestamp_ms, multiclass_label) for trainable reviewed
    markers, sorted by timestamp (earliest wins distance ties).
    """
    best_dist: float | None = None
    best_label: str | None = None
    for ts, marker_label in candidates:
        dist = abs(float(ts) - onset_ms)
        if best_dist is None or dist < best_dist:
            best_dist = dist
            best_label = marker_label
    if best_dist is not None and best_dist <= MATCH_MS:
        return best_label, None
    if racket_ts:
        d_racket = min(abs(float(ts) - onset_ms) for ts in racket_ts)
        if d_racket <= RACKET_AMBIGUOUS_MS:
            return None, "racket_decay"
    if non_trainable_ts:
        d_nt = min(abs(float(ts) - onset_ms) for ts in non_trainable_ts)
        if d_nt <= MATCH_MS:
            return None, "near_non_trainable"
    return "noise", None


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Mine gate-aligned training rows from TRAIN sessions and TRAIN beds."
    )
    parser.add_argument(
        "--out-csv",
        default=str(DEFAULT_OUT_CSV),
        help="Output CSV path (default: data/audio/processed/noise_robust/nr_train_mined.csv).",
    )
    parser.add_argument("--seed", type=int, default=nr_config.RNG_SEED, help="RNG seed.")
    parser.add_argument(
        "--bed-cap",
        type=int,
        default=BED_TRIGGER_CAP,
        help="Max gate-mined noise rows per bed take (even subsampling above this).",
    )
    parser.add_argument(
        "--limit-sessions",
        default="",
        help=(
            "TEMPORARY testing flag (not part of the real pipeline): comma-separated "
            "session ids; restricts both TRAIN sessions and TRAIN bed takes to those "
            "sessions. For smoke tests only - never use for a real mine."
        ),
    )
    args = parser.parse_args()

    out_csv = Path(args.out_csv)
    rng = np.random.default_rng(args.seed)
    limit_sessions = {s.strip() for s in args.limit_sessions.split(",") if s.strip()}

    # ---- Schema contract: assert column order against nr_train.csv ------
    if not NR_TRAIN_CSV.exists():
        raise FileNotFoundError(
            f"{NR_TRAIN_CSV} not found - run build_nr_dataset.py first (its header "
            "defines the output schema contract)."
        )
    expected_columns = list(pd.read_csv(NR_TRAIN_CSV, nrows=0).columns)
    columns = list(build_nr_dataset.META_COLUMNS) + nr_features.all_feature_names()
    if columns != expected_columns:
        raise AssertionError(
            "Output schema does not match nr_train.csv header. "
            f"expected={expected_columns} got={columns}"
        )

    inventory = build_nr_dataset.load_inventory()
    sessions = sorted(inventory["sessions"], key=lambda s: s["session_id"])
    sessions_by_id = {s["session_id"]: s for s in sessions}

    skipped = {
        "sessions_archive_m4a": 0,
        "sessions_diagnostic": 0,
        "sessions_limited_out": 0,
        "bed_takes_limited_out": 0,
        "events_without_trainable_markers": 0,
        "events_missing_wav": 0,
        "markers_not_trainable": 0,
        "markers_negative_overlap_excluded": 0,
        "triggers_racket_decay_dropped": 0,       # 140-300 ms from a racket marker
        "triggers_near_non_trainable_dropped": 0,  # within 140 ms of a non-trainable marker only
        "bed_triggers_early_dropped": 0,           # onset_sample < 2205
        "bed_triggers_capped": 0,
        "nan_or_inf_replaced": 0,
    }

    # ---- TRAIN session selection ----------------------------------------
    train_sessions: list[tuple[str, dict]] = []
    for session in sessions:
        session_id = session["session_id"]
        if session.get("group") == "archive_m4a":
            skipped["sessions_archive_m4a"] += 1
            continue
        if nr_config.split_for_session(session_id) != "train":
            continue
        if session.get("is_diagnostic") and session_id != build_nr_dataset.DIAGNOSTIC_EXCEPTION_SESSION:
            skipped["sessions_diagnostic"] += 1
            continue
        if limit_sessions and session_id not in limit_sessions:
            skipped["sessions_limited_out"] += 1
            continue
        train_sessions.append((session_id, session))

    # ---- Bed pool (mined for noise rows AND used for SNR mixing) --------
    bed_takes = sorted(
        (sid, wav) for sid, wav in nr_config.TRAIN_BED_TAKES
        if not limit_sessions or sid in limit_sessions
    )
    skipped["bed_takes_limited_out"] = len(nr_config.TRAIN_BED_TAKES) - len(bed_takes)
    bed_pool: list[tuple[str, np.ndarray]] = []  # ("session:wav", audio)
    bed_take_meta: list[dict] = []
    for bed_session_id, bed_wav in bed_takes:
        session = sessions_by_id.get(bed_session_id)
        if session is None:
            raise RuntimeError(f"Bed session not in inventory: {bed_session_id}")
        wav_path = ROOT_DIR / session["media_dir"] / bed_wav
        if not wav_path.exists():
            raise RuntimeError(f"Bed wav missing: {wav_path}")
        session_json = build_nr_dataset.load_session_json(session["json_path"])
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
    mined_base_rows = 0
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
        """Mirror of build_nr_dataset.main's add_augmented_copies, built on
        the exact same helper functions (gain, bed segment, SNR mix)."""
        nonlocal augmented_rows
        gain_clip = build_nr_dataset.apply_random_gain(clip, rng)
        rows.append(build_nr_dataset.make_row(
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
        for snr_db in build_nr_dataset.snrs_for_label(label, rng):
            bed_name, segment = build_nr_dataset.random_bed_segment(bed_pool, rng)
            mixed = build_nr_dataset.mix_clip_with_bed_segment(clip, segment, snr_db, sr, rng)
            rows.append(build_nr_dataset.make_row(
                clip=mixed,
                clip_id=f"{base_clip_id}:{build_nr_dataset.snr_augment_name(snr_db)}",
                augment=build_nr_dataset.snr_augment_name(snr_db),
                aug_bed=bed_name,
                label=label,
                **meta,
            ))
            augmented_rows += 1

    # ---- Mine TRAIN session events ---------------------------------------
    for session_id, session in train_sessions:
        session_json = build_nr_dataset.load_session_json(session["json_path"])
        media_dir = ROOT_DIR / session["media_dir"]
        events = sorted(session_json.get("events", []), key=lambda e: str(e.get("wav_filename") or ""))
        n_triggers_session = 0
        n_rows_session = 0
        for event in events:
            wav_filename = str(event.get("wav_filename") or "")
            review = event.get("review") or {}
            markers = review.get("markers") or []
            trainable = [m for m in markers if preprocess_audio.is_trainable_review_marker(m)]
            skipped["markers_not_trainable"] += len(markers) - len(trainable)
            if not trainable:
                skipped["events_without_trainable_markers"] += 1
                continue
            wav_path = media_dir / wav_filename
            if not wav_filename or not wav_path.is_file():
                skipped["events_missing_wav"] += 1
                print(f"  Missing wav: {wav_path}")
                continue

            y, sr = preprocess_audio.load_audio(str(wav_path))
            event_label = str(event.get("label") or "")
            scenario_id = str(event.get("scenario_id") or "")
            background_condition = str(event.get("background_condition") or "quiet")

            racket_ts = preprocess_audio.reviewed_racket_timestamps(markers)
            non_trainable_ts = sorted(
                int(m.get("timestamp_ms", 0)) for m in markers
                if not preprocess_audio.is_trainable_review_marker(m)
            )

            # Candidate (timestamp, label) pairs from trainable markers, with
            # labels resolved exactly like build_nr_dataset.py. Negative
            # markers overlapping a racket window are excluded (mirrors the
            # builder's markers_negative_overlap skip).
            candidates: list[tuple[int, str]] = []
            for marker in trainable:
                final_label = str(marker.get("final_label") or "")
                contact_kind = str(marker.get("contact_kind") or "") or preprocess_audio.contact_kind_for(event_label, scenario_id)
                not_racket_kind = str(marker.get("not_racket_kind") or "") or preprocess_audio.not_racket_kind_for(event_label, scenario_id)
                marker_label = preprocess_audio.multiclass_label_for_marker(final_label, contact_kind, not_racket_kind)
                if preprocess_audio.negative_marker_overlaps_racket(
                    marker,
                    racket_ts,
                    before_s=build_nr_dataset.NEGATIVE_OVERLAP_BEFORE_S,
                    after_s=build_nr_dataset.NEGATIVE_OVERLAP_AFTER_S,
                ):
                    skipped["markers_negative_overlap_excluded"] += 1
                    continue
                candidates.append((int(marker.get("timestamp_ms", 0)), marker_label))
            candidates.sort(key=lambda pair: pair[0])

            triggers = nr_features.simulate_gate(y, sr, **MINE_GATE_PARAMS)
            n_triggers_session += len(triggers)
            for trig_idx, trigger in enumerate(triggers):
                onset_ms = float(trigger["onset_ms"])
                label, drop_reason = label_for_trigger(onset_ms, candidates, racket_ts, non_trainable_ts)
                if drop_reason == "racket_decay":
                    skipped["triggers_racket_decay_dropped"] += 1
                    continue
                if drop_reason == "near_non_trainable":
                    skipped["triggers_near_non_trainable_dropped"] += 1
                    continue

                clip = nr_features.extract_live_clip(y, trigger["onset_sample"])
                base_clip_id = f"{session_id}:{wav_filename}:mine{trig_idx:04d}"
                meta = {
                    "split": "train",
                    "session_id": session_id,
                    "wav_filename": wav_filename,
                    "scenario_id": scenario_id,
                    "background_condition": background_condition,
                    "source": "gate_mined",
                    "anchor_ms": round(onset_ms, 3),
                    "jitter_ms": 0,
                    "close_event_bucket": "",
                }
                rows.append(build_nr_dataset.make_row(
                    clip=clip,
                    clip_id=base_clip_id,
                    augment="none",
                    aug_bed="",
                    label=label,
                    **meta,
                ))
                mined_base_rows += 1
                n_rows_session += 1
                add_augmented_copies(
                    clip,
                    base_clip_id,
                    with_snr=(label != "noise"),
                    label=label,
                    sr=sr,
                    meta=meta,
                )
        print(f"{session_id} [train]: {n_triggers_session} triggers -> {n_rows_session} mined rows")

    # ---- Mine TRAIN bed takes (all noise) --------------------------------
    for bed in bed_take_meta:
        session_id = bed["session_id"]
        wav_filename = bed["wav_filename"]
        y = bed["audio"]
        sr = bed["sr"]
        triggers = nr_features.simulate_gate(y, sr, **MINE_GATE_PARAMS)
        n_before = len(triggers)
        triggers = [t for t in triggers if t["onset_sample"] >= nr_config.CLIP_PRE_SAMPLES]
        skipped["bed_triggers_early_dropped"] += n_before - len(triggers)
        if len(triggers) > args.bed_cap:
            skipped["bed_triggers_capped"] += len(triggers) - args.bed_cap
            keep = np.linspace(0, len(triggers) - 1, args.bed_cap).astype(int)
            selected = [(int(i), triggers[int(i)]) for i in keep]
        else:
            selected = list(enumerate(triggers))

        for trig_idx, trigger in selected:
            clip = nr_features.extract_live_clip(y, trigger["onset_sample"])
            base_clip_id = f"{session_id}:{wav_filename}:minebed{trig_idx:04d}"
            meta = {
                "split": "train",
                "session_id": session_id,
                "wav_filename": wav_filename,
                "scenario_id": bed["scenario_id"],
                "background_condition": bed["background_condition"],
                "source": "gate_mined_bed",
                "anchor_ms": round(float(trigger["onset_ms"]), 3),
                "jitter_ms": 0,
                "close_event_bucket": "",
            }
            rows.append(build_nr_dataset.make_row(
                clip=clip,
                clip_id=base_clip_id,
                augment="none",
                aug_bed="",
                label="noise",
                **meta,
            ))
            mined_base_rows += 1
            add_augmented_copies(
                clip,
                base_clip_id,
                with_snr=False,
                label="noise",
                sr=sr,
                meta=meta,
            )
        print(f"Bed {session_id}:{wav_filename}: {n_before} triggers -> {len(selected)} mined rows")

    if not rows:
        print("No rows mined - aborting.")
        sys.exit(1)

    # ---- Validate + write -------------------------------------------------
    df = pd.DataFrame(rows, columns=columns)
    assert list(df.columns) == expected_columns, "output columns drifted from nr_train.csv schema"

    unexpected_labels = sorted(set(df["label"].astype(str).unique()) - set(nr_config.CLASSES))
    if unexpected_labels:
        raise AssertionError(f"Unexpected labels mined: {unexpected_labels}")

    clip_ids = df["clip_id"]
    if clip_ids.duplicated().any():
        dupes = sorted(clip_ids[clip_ids.duplicated()].unique().tolist())
        raise AssertionError(f"clip_id collisions: {dupes[:20]} (total {len(dupes)})")

    feature_columns = nr_features.all_feature_names()
    feat = df[feature_columns]
    bad_mask = ~np.isfinite(feat.to_numpy(dtype=np.float64))
    n_bad = int(bad_mask.sum())
    if n_bad:
        skipped["nan_or_inf_replaced"] = n_bad
        cleaned = feat.to_numpy(dtype=np.float64)
        cleaned[bad_mask] = 0.0
        df[feature_columns] = cleaned
    assert np.isfinite(df[feature_columns].to_numpy(dtype=np.float64)).all(), "non-finite feature values remain"

    out_csv.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(out_csv, index=False)
    print(f"\nWrote {out_csv} ({len(df)} rows: {mined_base_rows} mined + {augmented_rows} augmented)")

    label_aug = df.pivot_table(
        index="label", columns="augment", values="clip_id", aggfunc="count", fill_value=0,
    )
    label_source = df.pivot_table(
        index="label", columns="source", values="clip_id", aggfunc="count", fill_value=0,
    )
    print("\nLabel x augment counts:")
    print(label_aug.to_string())
    print("\nLabel x source counts:")
    print(label_source.to_string())
    print("\nSkipped counters:")
    for key, value in skipped.items():
        print(f"  {key}: {value}")


if __name__ == "__main__":
    main()
