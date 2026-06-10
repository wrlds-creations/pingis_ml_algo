# Noise-robust racket bounce detection — results (2026-06-10)

Goal: live racket-bounce counting with high recall, table/floor rejection and
robustness to background music, speech and crowd noise. All numbers below are
EVENT-LEVEL (full live-cascade replay on session WAVs, greedy 140 ms matching
against reviewed markers), not clip-level classification accuracy.

## What was built

Pipeline in this directory (all runnable end to end):

```
build_nr_dataset.py   -> data/audio/processed/noise_robust/nr_{train,val,test}.csv
nr_mine_gate_rows.py  -> nr_train_mined_v3.csv      (gate-aligned self-labeled rows)
train_nr_model.py     -> data/audio/models/noise_robust_v3/   (RF + HistGB, app JSON export)
replay_nr_live.py     -> event-level replay + per-bucket metrics + FP/min on noise beds
```

Key design changes vs the shipped detector:

1. **Session-level leakage-free split** (`nr_config.py`): no session, room
   recording, or augmentation bed crosses train/val/test. Today's pipeline
   splits per take and trains the shipped model on 100 % of the data, so its
   historical metrics were optimistic; there was previously NO music/speech
   holdout at all.
2. **Band-passed onset gate** (1.5–7 kHz frame RMS instead of broadband):
   music energy is mostly low-frequency, ball impacts are broadband. Gate
   recall in music_high rose 0.78 -> 0.95+ on val. Requires a small change in
   `AudioStreamModule.kt` (filter before RMS; spectral-gate FFT already
   computes the bins needed).
3. **Spectral gate removed** from the candidate path (`--no-spectral-gate`):
   the crude 256-pt gate rejected 10–25 % of true bounces under music; the
   model does that job better. (Native: emit candidates regardless, let the
   classifier veto.)
4. **21 noise-robust features** (`nr_features.py`, `nr_` prefix) on the raw
   300 ms clip: background-subtracted per-band deltas vs the pre-onset 80 ms,
   local SNR estimate, background level/flatness, band-passed envelope
   transients, PCEN statistics. Combined with the existing 62 features
   (= `all83`, exported as `feature_version nr_features_83_v1`).
5. **Real-noise augmentation with partitioned beds**: train rows SNR-mixed at
   15/10/5/0 dB with music/speech/desk/crowd beds that never touch val/test.
6. **Gate-aligned self-training** (`nr_mine_gate_rows.py`): the live gate is
   replayed over all train sessions/beds; every trigger becomes a training row
   labeled by the nearest reviewed marker (ambiguous 140–300 ms racket-decay
   triggers dropped). The model trains on exactly the candidate distribution
   it will see on-device. This was the single biggest FP reduction.
7. **Echo gate** in counting logic: within (merge, 300] ms of a counted
   bounce, drop detections whose onset-frame RMS <= 0.6x the counted one
   (racket rattle / decay re-triggers). Removes the dominant quiet-mode FP
   class without the recall cost of a wide merge window.

## Final holdout results (TEST split, never used for training or tuning)

Test material: 521 reviewed racket bounces over ~12 min in quiet /
music_low / music_high (real recordings, not synthetic) / speech / dense play,
plus ~12.4 min bounce-free FP beds (crowd + impact noises).

Sensitive profile = HistGB all83, bandpass gate ratio 1.5, no spectral gate,
conf 0.5, retrigger 120, merge 120, echo 300/0.6 (model v3).
Baseline = the currently installed app model + "normal" preset, replayed in
the same harness. NOTE: the baseline was trained ON these test sessions, so
its numbers are optimistic; the new model has never seen them.

| Bucket      | Baseline (installed)     | New sensitive (v3)        |
|-------------|--------------------------|---------------------------|
|             | recall / prec / FP-min   | recall / prec / FP-min    |
| quiet       | 0.997 / 0.935 / 5.7      | 0.993 / 0.944 / 4.9       |
| music_low   | 0.577 / 0.833 / 3.0      | **0.962** / 0.893 / 3.0   |
| music_high  | 0.592 / 0.938 / 3.0      | **0.908** / 0.873 / 10.0  |
| speech      | 0.833 / 0.917 / 6.7      | **0.939** / 0.805 / 20.0  |
| dense play  | 0.185 / 0.800 / 2.1      | **0.646** / 0.875 / 4.1   |
| crowd bed   | — / — / 6.4              | — / — / 14.2              |
| impact bed  | — / — / 0.0              | — / — / 7.2               |
| ALL         | 0.795 / 0.810 / 4.8      | **0.929** / 0.698 / 10.4  |

Balanced profile (RF all83 v2, conf 0.5, merge 220, no echo): overall
0.870 / 0.791 / 6.0 FP-min; crowd 7.1/min, impact 2.4/min, music_high 0.816,
speech 0.727.

Gate-level recall ceiling (bandpass, no spectral gate): 0.979 on test
(baseline cascade: 0.870 after its spectral gate).

Latency (Python replay, single process): feature+predict p50 ≈ 47 ms
=> end-to-end estimate p50 ≈ 257 ms (10 ms frame + 200 ms post-window +
compute). The 200 ms post-window dominates and is unchanged from the current
app; the new features add no material latency.

## Confidence-threshold analysis (test detections, sensitive profile)

Raising the decision threshold to 0.9 in LOUD environments cuts crowd FP
14.2 -> 4.1/min and impact 7.2 -> 1.2/min while music_high recall drops
0.908 -> 0.829 and speech 0.939 -> 0.803. TP confidences are concentrated at
p_racket ~ 1.0; crowd FPs have median 0.77. Recommendation for the app:
**adaptive confidence** — keep 0.5 when the pre-onset background is quiet,
require 0.85–0.9 when `nr_bg_rms_db` is high (the feature is already
computed per clip). This gives both the quiet/music recall and a usable
loud-venue FP rate without two user-facing modes.

## Known limitations (the honest ceiling)

- **Crowd FP rate is the weak spot** (14/min sensitive, 7/min balanced, vs
  6/min baseline-with-spectral-gate). The only crowd material is the 04-09
  noise takes: train beds and the test bed are siblings from one recording
  series, partly clipped, and content is unverified by listening — some
  "FPs" may be real ball impacts in the recording. Fresh, verified crowd
  recordings (different venue) are the highest-value new data.
- Dense-play recall 0.65 is limited by the 120 ms retrigger and
  racket<->table confusion in fast rallies; live up/down bouncing is the
  primary use case, dense play belongs to the retro path.
- Test/val sessions share rooms/devices/days with train sessions (one phone,
  one home + one office). Numbers measure background robustness, not
  new-device generalization.
- Speech FP/min (20) is inflated by the tiny bucket size (0.75 min of audio;
  15 FPs, several near vmn markers); more reviewed speech material would
  stabilize the estimate.
- The v3 RF app-JSON export is 44 MB / 1.5 M nodes (current app model:
  20.7 MB / 714 k). Before app integration, retrain the export with
  max_depth ~20 / fewer trees, or ship the HistGB as a small flat-tree JSON
  (HistGB all83 has far fewer nodes; needs a one-page exporter + TS runtime
  addition).

## App integration plan (not yet done)

1. `AudioStreamModule.kt`: band-passed frame RMS (biquad bandpass 1.5–7 kHz),
   `retriggerMs` 120, emit all candidates (drop the hard spectral reject).
2. `audioFeatures.ts`: port the 21 `nr_` features (STFT 512/128 on the raw
   300 ms clip + PCEN; pure TS, same math as `nr_features.py`).
3. `audioContactEngine.ts`: echo gate (300 ms / 0.6 RMS ratio) + adaptive
   confidence by background level; classes/JSON contract unchanged.
4. Export a size-constrained model, validate with the existing app-feature
   parity replay before any APK build.

## Reproduce

```bash
python skills/pingis-audio-classification/scripts/noise_robust/build_nr_dataset.py
python skills/pingis-audio-classification/scripts/noise_robust/nr_mine_gate_rows.py --bed-cap 800 \
    --out-csv data/audio/processed/noise_robust/nr_train_mined_v3.csv
python skills/pingis-audio-classification/scripts/noise_robust/train_nr_model.py \
    --extra-train-csv data/audio/processed/noise_robust/nr_train_mined_v3.csv \
    --out-dir data/audio/models/noise_robust_v3 --model-version nr_bounce_v3_2026_06_10
python skills/pingis-audio-classification/scripts/noise_robust/replay_nr_live.py \
    --model-dir data/audio/models/noise_robust_v3 --model histgb --feature-set all83 \
    --split test --split test_fp_bed --gate bandpass --no-spectral-gate \
    --confidence 0.5 --retrigger-ms 120 --merge-ms 120 --echo-ms 300 --echo-ratio 0.6 \
    --out-prefix data/audio/processed/noise_robust/TEST_v3_hgb_sensitive
```

Iteration history: v1 (marker-anchored + SNR aug) -> v2 (+ gate-aligned mined
rows: val precision 0.78 -> 0.85 at equal recall) -> echo gate (quiet FP
7.3 -> 3.7/min) -> v3 (denser bed negatives: crowd FP 18 -> 14/min). Full
per-run outputs under data/audio/processed/noise_robust/ (local, gitignored).
