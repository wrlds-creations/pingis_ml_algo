# Noise-robust racket bounce detector — implementation spec (v1)

Goal: a live racket-bounce detector (ball bounced up/down on racket) with high
racket recall and low false-positive rate that stays robust under background
music, speech and crowd noise. It must mirror the Collector app's real-time
cascade so offline metrics transfer to the device.

Pipeline stages (mirrors `AudioStreamModule.kt` + `audioContactEngine.ts`):

```
10 ms frames -> adaptive onset gate -> spectral gate -> 300 ms clip
            -> features -> 4-class model -> confidence/merge logic -> count
```

All code lives in `skills/pingis-audio-classification/scripts/noise_robust/`.
All scripts: Python 3.14, deps only numpy/scipy/librosa/sklearn/pandas/joblib.
Resolve repo root as `Path(__file__).resolve().parents[4]`... NO — use
`parents[3]` like preprocess_audio.py does from `scripts/`; from
`scripts/noise_robust/` it is `parents[4]`. Verify by checking that
`<root>/data/audio/raw` exists, raise otherwise.

Reuse from `../preprocess_audio.py` (import via
`sys.path.insert(0, str(Path(__file__).resolve().parents[1]))`):
`load_audio`, `extract_features` (base 62 features), `is_trainable_review_marker`,
`is_trainable_racket_marker`, `is_reviewed_table_marker`,
`negative_marker_overlaps_racket`, `contact_kind_for`, `not_racket_kind_for`,
`multiclass_label_for_marker`, `spacing_metadata_for_timestamp`.

Split config: import from `nr_config.py` (same dir). Never hardcode sessions.

Session discovery: read `data/audio/processed/audio_inventory_2026_06_10.json`
— per session it has `session_id`, `json_path`, `media_dir`, `group`,
`is_diagnostic` and per-event `wav_filename`, `scenario_id`,
`background_condition`, `duration_ms`. Resolve WAV as `<media_dir>/<wav_filename>`.
Load the session JSON for events + `review.markers`.

## Module 1: `nr_features.py`

### `simulate_gate(y, sr, *, onset_ratio=1.5, retrigger_ms=220, abs_min_rms=0.003, mode="broadband", bp_low=1500.0, bp_high=7000.0, spectral_gate=True) -> list[dict]`

Replicates the native onset detector frame-exactly:
- Frames of 220 samples, non-overlapping, in order.
- Background: rolling mean of the RMS of the last 30 frames that did NOT
  trigger (sub-threshold frames only). Initial background: 0.0 until at least
  1 frame observed (use mean of collected so far; if empty treat bg as 0).
- Trigger condition: `frame_rms >= max(bg_mean * onset_ratio, abs_min_rms)`.
- While within `retrigger_ms` of the last trigger: frames are skipped entirely
  (not added to background) — same as native.
- After a trigger: background buffer is reset to `[bg_mean * 2.0] * 30`.
- `mode="bandpass"`: frame RMS is computed on a band-passed copy of `y`
  (scipy.signal.butter order 4 bandpass `bp_low..bp_high`, sosfiltfilt) while
  the spectral gate still sees the raw signal. With bandpass mode use
  `abs_min_rms=0.0015` as the caller default (music energy is mostly low-freq;
  the ball transient keeps most of its energy in band).
- Spectral gate (if `spectral_gate`): on the RAW 220-sample trigger frame,
  256-pt FFT with Hann (pad frame to 256): ball-band (200–6000 Hz, DC excl.)
  power ratio must be >= 0.55 and spectral flatness (geo/arith mean of power
  spectrum) <= 0.6. Rejected triggers are still returned with
  `passed_spectral=False`, but — matching AudioStreamModule.kt exactly — a
  REJECTED trigger does NOT start the retrigger cooldown and does NOT reset
  the background buffer; its frame RMS is appended to the background buffer
  and scanning continues. Only ACCEPTED triggers set the cooldown and reset
  the background to `[bg_mean * 2.0] * 30`. Empty background buffer falls
  back to the current frame RMS (Kotlin `?: rms`), so the first frame of a
  file cannot trigger. (Amended 2026-06-10 after a parity audit against the
  Kotlin source; the original spec text mis-stated native behavior.)
- Return: list of dicts `{onset_sample, onset_ms, frame_rms, bg_rms,
  passed_spectral, ball_ratio, flatness}`.

### `extract_live_clip(y, onset_sample) -> np.ndarray`

100 ms before + 200 ms after the onset sample (6615 samples), zero-padded at
the edges if near the start/end of `y` so the onset always lands at sample
2205. Exactly like the native module.

### `extract_robust_features(clip, sr=22050) -> dict`

`clip` is the RAW 6615-sample live clip (NOT zero-padded to 1 s). All
features prefixed `nr_`. eps = 1e-10. Implement exactly:

- STFT: `librosa.stft(clip, n_fft=512, hop_length=128, center=False)`,
  power spectrum `S = |STFT|**2`. Frame k covers samples `[k*128, k*128+512)`.
- Background frames: frames fully inside the first 80 ms (start+512 <= 1764).
- Bands (Hz): low 200–800, mid 800–2500, high 2500–6000, vhigh 6000–11025.
  Band energy of a frame = sum of `S` bins whose center freq is in the band.
- `bg_band[b]` = median over background frames of band energy.
  `bg_total` = median over background frames of total energy (all bins >=200 Hz).
- Peak frame `p` = argmax over frames starting at sample >= 1764 of total energy.
- Features:
  - `nr_band_delta_low/mid/high/vhigh` = `log10(band_p + eps) - log10(bg_band + eps)`
  - `nr_band_delta_max` = max of the four; `nr_band_delta_argmax` = 0..3.
  - `nr_snr_db_est` = `10*log10((total_p + eps) / (bg_total + eps))`
  - `nr_bg_rms_db` = `20*log10(rms(clip[:1764]) + eps)`
  - `nr_flux_onset` = `log10(eps + max over frames f in 90..150 ms of
    sum(max(0, sqrt(S[:,f]) - sqrt(S[:,f-1]))))`
  - `nr_bg_flatness` = flatness (geo/arith mean) of the mean background power
    spectrum; `nr_impact_flatness` = flatness of `S[:, p]`.
  - Band-passed envelope (butter order 4, 1500–7000 Hz, sosfiltfilt on clip):
    env = moving average (110 samples ~5 ms) of `abs(bp)`.
    - `nr_bp_attack_ms`: time from 10% to 90% of env peak (search backward
      from peak, max lookback 50 ms; 0 if degenerate).
    - `nr_bp_decay50_ms`: time from env peak to first sample below 50% of
      peak (cap 150 ms).
    - `nr_bp_crest` = bp peak / (bp rms + eps).
    - `nr_bp_peak_ratio` = bp peak / (max(abs(clip)) + eps).
    - `nr_post_decay_db_50ms` / `nr_post_decay_db_100ms` =
      `20*log10((env[peak+50ms] + eps) / (env_peak + eps))` (and 100 ms; clip
      index capped at len-1).
  - PCEN: `M = librosa.feature.melspectrogram(y=clip, sr=sr, n_fft=512,
    hop_length=128, n_mels=40, center=False)`;
    `P = librosa.pcen(M * (2**31), sr=sr, hop_length=128)`;
    `t = P.mean(axis=0)` (per-frame mean over mel bins):
    `nr_pcen_max = t.max()`, `nr_pcen_mean = t.mean()`, `nr_pcen_std = t.std()`.

  - `nr_bp_peak_db` = `20*log10(env_peak + eps)` (band-passed envelope peak
    level in dB; added as the 21st feature 2026-06-10 — the original list
    enumerated only 20 names while stating a 21-feature total).

  Total: 21 features. Return plain dict, stable key order as listed.

### `extract_all_features(clip, sr=22050) -> dict`

Base 62: zero-pad clip to 22050 samples (zeros at END), call
`preprocess_audio.extract_features`. Then merge `extract_robust_features` on
the raw 300 ms clip. Returns 83 features.

## Module 2: `build_nr_dataset.py`

Builds clip datasets per split. CLI: `--out-dir data/audio/processed/noise_robust`
(default), `--seed` (default from nr_config.RNG_SEED).

For every session in the inventory: classify via `nr_config.split_for_session`.
Fail loudly (exit 1, listing them) if any session with reviewed markers is
`unassigned`. Sessions in archive_m4a group: skip (legacy, no markers).
Skip sessions with `is_diagnostic` even if assigned (belt and braces), EXCEPT
`audio_session_2026-05-29_001` whose AUDIO markers are documented valid.

### Rows from reviewed markers (train/val/test sessions)

For each event WAV, load audio once (`load_audio`). For each marker passing
`is_trainable_review_marker`:
- label = `multiclass_label_for_marker(final_label, contact_kind or
  contact_kind_for(event_label, scenario_id), not_racket_kind or
  not_racket_kind_for(event_label, scenario_id))`.
- Skip negatives overlapping racket windows: use
  `negative_marker_overlaps_racket(marker, racket_ts, before_s=0.1, after_s=0.2)`
  (reviewed table markers are exempt inside that helper — keep that).
- Anchor: `timestamp_ms`. TRAIN rows: add uniform integer jitter in
  [-15, +15] ms (rng). VAL/TEST rows: no jitter.
- Clip: `extract_live_clip(y, anchor_sample)`.
- Features: `extract_all_features`.

### Hard negatives from train beds (label `noise`, source `bed_gate`)

For each `(session, wav)` in `nr_config.TRAIN_BED_TAKES`: run
`simulate_gate(y, onset_ratio=1.3, retrigger_ms=150, mode="broadband",
spectral_gate=False)` and take ALL triggers (these are exactly what a
sensitive live gate would fire on: music beats, plosives, key clicks). Cap at
150 per bed by even subsampling. Clip + features per trigger. Additionally add
non-overlapping random 300 ms chunks, 1 per 4 s of bed audio, as `bed_chunk`
noise rows.

### Augmentation (TRAIN rows only)

Bed pool: load all TRAIN_BED_TAKES wavs once; cut a random 6615-sample segment
per use (uniform position, rng). Per reviewed-marker train row:
- copies: `gain` (no mixing, random gain ±6 dB) plus 2 SNRs sampled from
  AUGMENT_SNR_DB without replacement (floor_bounce rows: all 4 SNRs).
- Mix at clip level: scale bed segment to reach the target SNR vs the clip
  (same formula as preprocess_audio.mix_with_noise but with a chosen segment,
  not fix_length-looping), then random gain ±6 dB on the mix, peak-normalize
  if >1. SNR reference: RMS of the 60 ms around the clip's absolute peak (not
  whole-clip RMS, which underestimates impact loudness in padded clips).
- `bed_gate`/`bed_chunk` rows: 1 `gain` copy each, no SNR mixing.
- Augmented row keeps `group_id` of source row, `augment` column = `gain` /
  `snr15` / `snr10` / `snr5` / `snr0`, `aug_bed` = `session:wav` of the bed.

### Output CSVs

`nr_train.csv`, `nr_val.csv`, `nr_test.csv` with columns:
`clip_id, split, session_id, wav_filename, scenario_id, background_condition,
label, source (reviewed_marker|bed_gate|bed_chunk), anchor_ms, jitter_ms,
augment, aug_bed, group_id (= session_id), close_event_bucket` + 83 feature
columns (base 62 first, in preprocess_audio order, then nr_ in spec order).
Also write `nr_dataset_summary.json` (+ printed table): row counts per
split×label×background_condition×augment, skipped counters. Deterministic:
single `np.random.default_rng(seed)`; iterate sessions/events/markers in
sorted order.

## Module 3: `train_nr_model.py`

CLI: `--data-dir`, `--out-dir data/audio/models/noise_robust_v1`, `--seed`.

- Load nr_train/nr_val.
- Feature sets: `base62`, `robust21` (nr_ only), `all83`.
- Models (per feature set):
  - RF: GridSearch over `n_estimators=[300]`, `max_depth=[None, 25]`,
    `min_samples_leaf=[1, 3]`, `class_weight='balanced_subsample'`,
    `random_state=seed`, `n_jobs=-1`, CV = GroupKFold(5) grouped by
    `group_id` (= session) on TRAIN ONLY, scoring `f1_macro`.
  - HistGB: `max_iter=400, learning_rate=0.08, max_depth=None,
    early_stopping=False, class_weight='balanced'`, same CV for sanity (no grid).
- StandardScaler fit on train only (RF does not need it but the app contract
  expects scaler arrays — fit it and export; trees are scale-invariant so
  this is purely for the runtime contract).
- Evaluate every (feature set × model) on VAL clips (clean rows only,
  source=reviewed_marker): per-class precision/recall/F1, confusion matrix,
  racket recall and racket precision specifically; pick winner by
  `0.6*racket_recall + 0.4*racket_precision` subject to
  `racket_precision >= 0.90` on val (if none passes, report best anyway).
- NO refit on val/test. The shipped model is the train-only fit.
- Save: joblib artifacts (`nr_rf_<set>.pkl`, `nr_histgb_<set>.pkl`,
  `nr_scaler_<set>.pkl`, `nr_feature_cols_<set>.pkl`, `nr_label_encoder.pkl`
  — replay_nr_live.py resolves these names by default), `training_log.json`
  (all configs, CV results, val metrics, durations, library versions,
  hyperparams), and for the best RF an app-format JSON export
  `nr_audio_model.json`: `{metadata:{model_version:
  'nr_bounce_v1_2026_06_10', feature_version:'nr_features_83_v1', classes,
  tree_count}, labels (LabelEncoder alphabetical), feature_names (training
  column order), scaler_mean, scaler_std (8 dp), trees: flat node arrays —
  internal `[feature_idx, threshold, left_idx, right_idx]`, leaf = normalized
  class-probability list}`. Copy the encoding from
  `../export_model_json.py` exactly (read it first).
- Print a markdown results table to stdout AND write it to
  `<out-dir>/val_results.md`.

## Module 4: `replay_nr_live.py`

Event-level live simulation + scoring. CLI:
`--model-dir`, `--model {rf,histgb}`, `--feature-set {base62,robust21,all83}`,
`--split {val,test,val_fp_bed,test_fp_bed}` (repeatable) or `--sessions id,id`,
`--gate {broadband,bandpass}`, `--onset-ratio` (default 1.5),
`--abs-min-rms` (default per gate mode), `--spectral-gate/--no-spectral-gate`,
`--confidence` (default 0.5), `--decision {argmax,prob}` (default argmax:
count when argmax==racket_bounce AND conf>=threshold; prob: count when
P(racket_bounce)>=threshold), `--retrigger-ms 220`, `--merge-ms 220`,
`--group-ms 80`, `--out-prefix`.

For each event WAV of the selected sessions:
1. `simulate_gate` -> triggers (spectral-gate failures are recorded but not
   classified, like native).
2. For each passing trigger: clip, features (only the model's feature set),
   scale (RF: app-style (x-mean)/std using exported scaler for parity),
   predict probabilities.
3. Count logic: decision rule -> then merge window (drop if within merge_ms
   of last counted) -> group window (drop if within group_ms of group start).
4. Score vs truth (trainable racket markers): greedy nearest match within
   140 ms, one truth matchable once: TP / FP / duplicate / missed. Also
   GATE-LEVEL recall: fraction of truth markers with >=1 raw trigger
   (pre-spectral, and post-spectral, separately) within 140 ms.
5. Latency: wall-clock per-clip feature+predict time (p50/p95) +
   constant 200 ms post-window + 10 ms frame quantization -> report
   `latency_est_ms_p50/p95`.
6. FP-bed sessions: all counted racket events are FPs; report FP/min.

Output: per-event CSV (`<out-prefix>_events.csv`), per-bucket summary
(bucket = background_condition mapped to {quiet, music_low, music_mid,
music_high, speech, mixed, impact, crowd}) as JSON + markdown. The crowd
bucket = the 04-09 bed sessions (background `(none)`).

## Validation requirements (all modules)

- `python -m py_compile <file>` passes.
- Builder: after writing CSVs, assert no `clip_id` collisions, no NaN/inf in
  feature columns (replace with 0.0 and count occurrences in summary), and
  print split×label matrix.
- Determinism: same seed -> identical CSV bytes (iterate in sorted order,
  single rng). Spot-verified by the implementer for at least one tiny session.
