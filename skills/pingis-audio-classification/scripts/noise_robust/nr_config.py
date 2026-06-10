"""Noise-robust racket bounce detector: split definition and shared constants.

This file is the single source of truth for the session-level train/val/test
split. The split is by SESSION (not take/group) so that no recording room,
background bed, or reviewed take can appear on both sides of a boundary.

Rules enforced here:
- TEST sessions are never used for training rows, augmentation beds, or
  threshold tuning. They are only touched by replay evaluation.
- VAL sessions are used for model selection and threshold tuning only.
- Augmentation noise beds are partitioned: TRAIN_BEDS may only be mixed into
  train rows; TEST_FP_BEDS are only used to measure false-positive rates.
- Diagnostic sessions are excluded everywhere.
"""

from __future__ import annotations

TARGET_SR = 22050

# Live clip geometry (must match AudioStreamModule.kt: PRE=2205, POST=4410).
CLIP_PRE_MS = 100
CLIP_POST_MS = 200
CLIP_PRE_SAMPLES = 2205
CLIP_POST_SAMPLES = 4410
CLIP_SAMPLES = CLIP_PRE_SAMPLES + CLIP_POST_SAMPLES  # 6615 = 300 ms
# App featurizer pads the live clip into a fixed 1 s buffer (zeros at end).
FEATURE_BUFFER_SAMPLES = TARGET_SR

# Anchor jitter applied to training clips to mimic the 10 ms onset frame
# quantization and marker placement variation (attack_start anchor rule).
ANCHOR_JITTER_MS = 15

# Augmentation settings (train only).
AUGMENT_SNR_DB = [15.0, 10.0, 5.0, 0.0]
AUGMENT_GAIN_DB = 6.0  # uniform random gain in [-6, +6] dB on top of mixes
RNG_SEED = 20260610

# 4-class label contract expected by the Collector app runtime.
CLASSES = ["floor_bounce", "noise", "racket_bounce", "table_bounce"]

# Match tolerance for event-level replay scoring (same as replay_live_bounce).
MATCH_TOLERANCE_MS = 140

# ---------------------------------------------------------------------------
# Session-level split.
#
# Buckets shown per session: reviewed racket / table / floor / noise-ish
# marker counts and background conditions, from
# data/audio/processed/audio_inventory_2026_06_10.json.
# ---------------------------------------------------------------------------

# Held-out TEST sessions. Cover quiet, music_low, music_high, speech and
# dense mixed play. Never used in training or augmentation.
TEST_SESSIONS = {
    "audio_session_2026-05-06_003",   # quiet bouncing, 239 racket
    "audio_session_2026-05-11_002",   # music_low, 26 racket + 15 floor + 15 vmn
    "audio_session_2026-05-12_005",   # music_high, 76 racket + 38 floor + 72 vmn
    "audio_session_2026-05-13_005",   # speech + quiet, 99 racket
    "audio_session_2026-04-23_013",   # speech (counting), 16 racket
    "audio_session_2026-06-04_006",   # dense mixed play, 65 racket + 74 table
}

# Bed-only TEST sessions used exclusively for false-positive-rate replay
# (no racket truth expected; any racket count is a false positive).
TEST_FP_BED_SESSIONS = {
    "audio_session_2026-04-09_005",   # ~10 min crowd/sorl beds
    "audio_session_2026-05-12_001",   # other-bounce impact takes (66 vmn markers)
}

# VAL sessions: model selection + threshold tuning only.
VAL_SESSIONS = {
    "audio_session_2026-05-12_004",   # music_high, 40 racket + 11 vmn
    "audio_session_2026-05-11_005",   # speech, 40 racket
    "audio_session_2026-05-06_007",   # quiet bouncing, 254 racket
    "audio_session_2026-05-13_008",   # dense mixed play, 14 racket + 17 table
}

# VAL false-positive material (floor under noisy background; floor truth
# markers double as "must not count" events).
VAL_FP_BED_SESSIONS = {
    "audio_session_2026-05-12_002",   # floor_noisy takes, 59 floor markers
}

# Diagnostic / broken sessions: excluded everywhere.
EXCLUDED_SESSIONS = {
    "audio_session_2026-05-26_002",   # diagnostic-only (uncorrected candidate flood)
    "audio_session_2026-05-26_003",   # diagnostic-only (same-media comparison)
    "audio_session_2026-05-26_004",   # diagnostic-only (same-media comparison)
    "audio_session_2026-05-11_001",   # documented diagnostic-only (not in raw)
    "audio_session_2026-04-21_001",   # broken: no session JSON, orphan wav
}

# Noise beds for TRAIN augmentation, as (session_id, wav_filename) pairs.
# 04-21_002 has the only music beds (low/mid) plus speech/desk beds;
# 04-23_014 adds two more speech beds; 04-09_003/_004 are long crowd/sorl
# beds (possibly clipped - acceptable for augmentation, not for testing).
TRAIN_BED_TAKES = [
    ("audio_session_2026-04-21_002", "music_low_only_001.wav"),
    ("audio_session_2026-04-21_002", "music_low_only_002.wav"),
    ("audio_session_2026-04-21_002", "music_mid_only_001.wav"),
    ("audio_session_2026-04-21_002", "music_mid_only_002.wav"),
    ("audio_session_2026-04-21_002", "speech_only_001.wav"),
    ("audio_session_2026-04-21_002", "speech_only_002.wav"),
    ("audio_session_2026-04-21_002", "speech_only_003.wav"),
    ("audio_session_2026-04-21_002", "speech_only_004.wav"),
    ("audio_session_2026-04-21_002", "desk_keyboard_only_001.wav"),
    ("audio_session_2026-04-21_002", "desk_keyboard_only_002.wav"),
    ("audio_session_2026-04-21_002", "desk_keyboard_only_003.wav"),
    ("audio_session_2026-04-21_002", "desk_keyboard_only_004.wav"),
    ("audio_session_2026-04-23_014", "speech_only_001.wav"),
    ("audio_session_2026-04-23_014", "speech_only_002.wav"),
    ("audio_session_2026-04-09_003", "noise_000.wav"),
    ("audio_session_2026-04-09_004", "noise_000.wav"),
]

# TRAIN sessions: every reviewed, non-diagnostic session not listed above.
# Listed explicitly so the builder fails loudly if a new session appears
# without a split decision.
TRAIN_SESSIONS = {
    "audio_session_2026-04-22_008",   # speech + music_mid counting
    "audio_session_2026-04-22_009",   # quiet + speech counting
    "audio_session_2026-04-23_014",   # quiet + speech counting (positives; its speech beds are train beds)
    "audio_session_2026-05-05_008",   # quiet fh/bh + floor (device_pull)
    "audio_session_2026-05-05_019",   # long free recording, mixed
    "audio_session_2026-05-06_008",   # speech/music_low racket + impact negatives
    "audio_session_2026-05-06_009",   # speech racket + floor + impact
    "audio_session_2026-05-06_010",   # music_mid backhand
    "audio_session_2026-05-11_004",   # music_mid racket + floor
    "audio_session_2026-05-12_006",   # imported, mixed
    "audio_session_2026-05-12_009",   # table only, mixed
    "audio_session_2026-05-12_010",   # racket + table, mixed
    "audio_session_2026-05-12_011",   # racket + table, mixed
    "audio_session_2026-05-13_001",   # racket + table, mixed
    "audio_session_2026-05-13_009",   # dense play, mixed
    "audio_session_2026-05-22_001",   # dense play CJ forehand
    "audio_session_2026-05-22_003",   # dense play CJ backhand
    "audio_session_2026-05-25_003",   # dense play
    "audio_session_2026-05-25_004",   # dense play
    "audio_session_2026-05-25_007",   # dense play
    "audio_session_2026-05-26_001",   # dense play (valid, reviewed)
    "audio_session_2026-05-27_001",   # dense play
    "audio_session_2026-05-27_002",   # dense play, stiga office
    "audio_session_2026-05-28_002",   # dense play, Tomas
    "audio_session_2026-05-29_001",   # dense play, Tomas hard (audio markers valid)
    "audio_session_2026-05-29_002",   # dense play, Tomas backhand
    "audio_session_2026-06-03_005",   # dense play
    "audio_session_2026-06-04_001",   # dense play
}


def split_for_session(session_id: str) -> str:
    """Return one of train / val / test / test_fp_bed / val_fp_bed /
    excluded / unassigned for a session id."""
    if session_id in EXCLUDED_SESSIONS:
        return "excluded"
    if session_id in TEST_SESSIONS:
        return "test"
    if session_id in TEST_FP_BED_SESSIONS:
        return "test_fp_bed"
    if session_id in VAL_SESSIONS:
        return "val"
    if session_id in VAL_FP_BED_SESSIONS:
        return "val_fp_bed"
    if session_id in TRAIN_SESSIONS:
        return "train"
    return "unassigned"
