# Iteration Log: Pingis ML App

Date: 2026-04-24
Owner: Codex + Love
Scope: Collector app, audio review, racket/table/floor sound models, future AirHive IMU swing fusion

## Purpose

This file is the working log for:
- APK builds
- model versions and metrics
- device feedback
- product decisions
- next steps

This is the project memory and handoff file for AI assistants.
For bounce/audio/ML work, read this file after the relevant Skill before making changes.
Do not move day-to-day model state into [AGENTS.md](</C:/Users/lovea/Desktop/dev/STIGA SPORTS/pingis_ml_algo/AGENTS.md>); `AGENTS.md` should only point future agents here.

## Product Goal

The final goal is a full pingis training app that can:
- Collect data on-device with low friction.
- Record audio, video for review, and later synchronized AirHive IMU where needed.
- Let the user review and correct labels in the app so training data is trustworthy.
- Pull collected sessions back to the computer.
- Run deterministic preprocessing and ML training locally.
- Track model quality over time.
- Export updated model artifacts into the React Native app.
- Install test APKs on the Motorola for live validation.

The immediate product goal is a stable audio-first bounce detector:
- It should reliably detect `racket_contact`.
- It should reject `floor_bounce`, `table_bounce`, speech, music, desk sounds, claps, clicks, and other hard transients.
- It should avoid double-counting one physical racket contact.
- It should work for straight-up racket bounces and for forehand/backhand-style racket contacts.

The longer-term product goal is fusion:
- Audio says whether a ball likely contacted the racket.
- AirHive IMU says whether a real forehand/backhand swing happened.
- When swing evidence and racket-contact audio agree, the app can say with higher confidence that a real in-play stroke occurred.
- IMU should add confidence and side/type context; it should not replace the audio model unless evidence shows that is better.

## System Workflow

1. Collect on Motorola in the app.
2. Review audio markers in the app; video is review support only, not a training input.
3. Pull `audio_session_*.json` and matching media folders from `/sdcard/Download/pingis_sessions` to [data/audio/raw](</C:/Users/lovea/Desktop/dev/STIGA SPORTS/pingis_ml_algo/data/audio/raw>).
4. Preprocess with [preprocess_audio.py](</C:/Users/lovea/Desktop/dev/STIGA SPORTS/pingis_ml_algo/skills/pingis-audio-classification/scripts/preprocess_audio.py>).
5. Build contact variants with [build_audio_contact_dataset_variant.py](</C:/Users/lovea/Desktop/dev/STIGA SPORTS/pingis_ml_algo/skills/pingis-audio-classification/scripts/build_audio_contact_dataset_variant.py>), reporting `human_reviewed` separately from weak legacy/bootstrap variants.
6. Train the binary model with [train_rf_audio_contact.py](</C:/Users/lovea/Desktop/dev/STIGA SPORTS/pingis_ml_algo/skills/pingis-audio-classification/scripts/train_rf_audio_contact.py>).
7. Train the 4-class model with [train_rf_audio.py](</C:/Users/lovea/Desktop/dev/STIGA SPORTS/pingis_ml_algo/skills/pingis-audio-classification/scripts/train_rf_audio.py>).
8. Export app artifacts with [export_contact_model_json.py](</C:/Users/lovea/Desktop/dev/STIGA SPORTS/pingis_ml_algo/skills/pingis-audio-classification/scripts/export_contact_model_json.py>) and [export_model_json.py](</C:/Users/lovea/Desktop/dev/STIGA SPORTS/pingis_ml_algo/skills/pingis-audio-classification/scripts/export_model_json.py>).
9. Build and install the Android release APK locally before trusting live behavior.

## Model Map

| Model | Artifact | Task | App role |
|---|---|---|---|
| `audio_contact_model` | [audio_contact_model.json](</C:/Users/lovea/Desktop/dev/STIGA SPORTS/pingis_ml_algo/apps/collector/src/models/audio_contact_model.json>) | Binary `racket_contact` vs `not_racket_contact` | Primary live count engine |
| `audio_model` | [audio_model.json](</C:/Users/lovea/Desktop/dev/STIGA SPORTS/pingis_ml_algo/apps/collector/src/models/audio_model.json>) | 4-class `racket_bounce / table_bounce / floor_bounce / noise` | Secondary veto/debug model; keep training it |
| `stroke_hit_model` | [stroke_hit_model.json](</C:/Users/lovea/Desktop/dev/STIGA SPORTS/pingis_ml_algo/apps/collector/src/models/stroke_hit_model.json>) | IMU hit vs miss | Future swing fusion, currently paused |
| `stroke_type_model` | [stroke_type_model.json](</C:/Users/lovea/Desktop/dev/STIGA SPORTS/pingis_ml_algo/apps/collector/src/models/stroke_type_model.json>) | IMU forehand vs backhand | Future swing typing, currently paused |
| `bounce_imu_model` | not exported live yet | Bounce-motion contact vs not | Experimental, removable, currently paused |

## Current Focus

Audio quality and reviewed ground truth are the priority.
IMU is allowed as synchronized support data, but not as live count truth until audio labels and live grouping are stable.

The audio work should continue training both:
- Binary `audio_contact_model`, because it is the live count engine.
- 4-class `audio_model`, because it explains and can veto `table_bounce` and `floor_bounce`, and table bounce may matter later.

Current binary training strategy:
- Human-reviewed markers are the single source of truth.
- Legacy data is allowed only as weak/bootstrap coverage and must be reported separately from `human_reviewed`.
- Reports must compare source/trust, scenario, FH/BH prompt metadata, hard-negative kind, and live-chain replay before exporting a new APK model.
- Synced AirHive raw collection now targets 150 Hz when stable, stores AirHive `sensor_ts`, `received_at_ms`, `take_ts_ms`, measured interval stats, and a per-take quality flag.

## Current Known Problems

| Area | Symptom | Current interpretation | Next diagnostic |
|---|---|---|---|
| Double count | One racket bounce plus catching/after-sound can become two counted contacts | Likely live onset/merge-window/timing behavior, not only model quality | Inspect debug rows: counted vs `merge_window`; consider stronger merge after a counted contact |
| Missed angled contacts | Forehand/backhand-style contacts are missed more often than straight vertical bounces | Training data is biased toward vertical racket bounces; live onset may also miss lower or angled transients | Collect reviewed `racket_forehand_bounce` and `racket_backhand_bounce` style takes or fold this into prompts |
| Floor false positives | Floor bounces have historically been counted as racket contacts | Improved but not solved; latest 4-class `floor_bounce` recall is only 0.61 | More floor data in multiple positions/surfaces plus verify whether veto sees `floor_bounce` live |
| Video/audio offset | Review video and audio are not perfectly synced | Annoying for review but not the current ML blocker | Do not touch unless it blocks labeling again |

## Active Models

| Model | Task | Live role | Training source | Current status |
|---|---|---|---|---|
| `audio_contact_model` | `racket_contact` vs `not_racket_contact` | Primary count engine for `Studsdetektor`, `Studs fritt`, and `Studs vaxla sida` | Human-reviewed markers are truth; weak legacy/all-legacy variants are comparison/bootstrap only | Active and highest priority |
| `audio_model` | `racket_bounce / table_bounce / floor_bounce / noise` | Secondary veto/debug model for surface/noise decisions | Multiclass processed audio dataset | Active; keep training alongside binary model |
| `stroke_hit_model` | stroke hit vs miss | Future IMU swing evidence | IMU stroke dataset | Paused until audio is stable |
| `stroke_type_model` | forehand vs backhand stroke type | Future IMU forehand/backhand swing typing | IMU stroke dataset | Paused until audio is stable |
| `bounce_imu_model` | bounce contact motion vs not bounce contact | Not in live app yet | Reviewed synchronized audio + IMU takes | Experimental; scripts exist but live work is paused |

## Metric History

| Date | Model | Dataset | Grouped CV F1 macro | Grouped test acc | Key recall / note |
|---|---|---|---:|---:|---|
| 2026-06-02 | `spel_retro_audio` T0008 cross-session validation | T0005/T0007 candidate rows, leave-one-session-out on Tomas/Stiga dense holdouts | N/A | 0.913 / 0.921 / 0.908 holdouts | Local validation only, not exported: selected T0007 `multi_window_context_racket_weighted` passes cross-session gate across `audio_session_2026-05-28_002`, `audio_session_2026-05-29_001`, and `audio_session_2026-05-29_002`. Racket recall is 0.910 / 0.939 / 0.896, table recall is 0.932 / 0.958 / 0.933, and non-target recall is 0.894 / 0.859 / 0.833. Compared with T0006, racket is equal on 05-28, +0.030 on 05-29_001, and +0.274 on 05-29_002, with no table loss and non-target equal or better. Next step is a separate Review retro integration path, not `studs_live`. |
| 2026-06-02 | `spel_retro_audio` T0007 multi-window/context candidate | T0005 candidate-centered rows with tight/normal/wide real-WAV windows plus non-leaky candidate context | N/A | 0.908 holdout | Local only, not exported: selected `playing_retro_audio_rf_v2026_06_02_multi_window_context` / `multi_window_context_racket_weighted`. On `audio_session_2026-05-29_002`, holdout accuracy improved 0.771 -> 0.908 vs T0006, racket recall 0.623 -> 0.896, table recall stayed 0.933, and non-target recall improved 0.625 -> 0.833. Wrong-class racket-as-table recall was 0.860 and matched-table recall was 0.980. Truth-derived spacing fields stayed metadata-only; ordinary fallback is advisory because older ordinary rows lack exact multi-window timestamps. |
| 2026-06-02 | `spel_retro_audio` T0006 safe racket-weighted candidate | T0005 candidate-centered rows, focused one-window variant comparison | N/A | 0.771 holdout | Local only, not exported: selected `playing_retro_audio_rf_v2026_06_02_safe_racket_weighted`. On `audio_session_2026-05-29_002`, holdout accuracy improved 0.759 -> 0.771, racket recall 0.604 -> 0.623, table recall 0.924 -> 0.933, and non-target recall stayed 0.625. Aggressive weighting reached racket 0.679 but was rejected because non-target fell to 0.500. Next step is true multi-window/context features. |
| 2026-06-02 | `spel_retro_audio` T0005 local candidate | all matchable saved app candidates plus manually missed markers across 16 playing sessions | N/A | 0.759 holdout | Local only, not exported: `playing_retro_audio_rf_v2026_06_02_app_candidates_100_200`, 4,028 rows with 1,706 racket, 1,774 table, and 548 non-target. Holdout `audio_session_2026-05-29_002` beats old app prediction accuracy 0.759 vs 0.682, but racket recall is only 0.604 while table recall is 0.924 and non-target recall is 0.625. Ordinary up/down regression is separate at 10,353 rows / 0.714 and does not affect `studs_live`. |
| 2026-04-22 | `audio_contact_model` | `audio_contact_dataset.csv` | 0.914 | 0.79 | `racket_contact` recall 0.94, `not_racket_contact` recall 0.70 |
| 2026-04-22 | `audio_model` | `audio_dataset.csv` | 0.694 | 0.74 | Stronger on debug than as primary contact engine |
| 2026-04-22 | `bounce_imu_model` | `bounce_imu_dataset.csv` | 0.847 | 1.00 | First tiny baseline only; test fold happened to contain no negatives |
| 2026-04-22 | `audio_contact_model` | `contact_variants/current` | 0.807 | 0.98 | Current binary dataset is too strict: `racket_contact` recall 0.48, `not_racket_contact` recall 1.00 |
| 2026-04-22 | `audio_contact_model` | `contact_variants/trusted_legacy` | 0.497 | 0.96 | Too few trusted legacy negatives locally; collapsed to always-positive behaviour |
| 2026-04-22 | `audio_contact_model` | `contact_variants/all_legacy` | 0.899 | 0.87 | Best local binary variant so far: `racket_contact` recall 0.96, `not_racket_contact` recall 0.66 |
| 2026-04-23 | `audio_contact_model` | `contact_variants/all_legacy_2026-04-23` | 0.848 | 0.92 | Negative recall improved: `racket_contact` recall 0.96, `not_racket_contact` recall 0.78 |
| 2026-04-23 | `audio_model` | `audio_dataset.csv` | 0.770 | 0.86 | 4-class veto/debug model improved strongly; `table_bounce` recall 0.94, `floor_bounce` recall 0.61 |
| 2026-05-06 | `audio_contact_model` | `contact_variants/human_reviewed` short reviewed windows | 0.489 | 0.90 | Too positive-heavy for export: `racket_contact` recall 1.00, `not_racket_contact` recall 0.10 |
| 2026-05-06 | `audio_contact_model` | `contact_variants/legacy_hybrid` short reviewed windows | 0.539 | 0.92 | Better than reviewed-only but still weak negatives: `racket_contact` recall 1.00, `not_racket_contact` recall 0.31 |
| 2026-05-06 | `audio_contact_model` | `contact_variants/all_legacy` + short reviewed windows | 0.907 | 0.92 | Best candidate in this run: `racket_contact` recall 0.96, `not_racket_contact` recall 0.78; floor hard negatives remain weak |
| 2026-05-06 | `audio_model` | `audio_dataset.csv` + short reviewed windows | 0.754 | 0.78 | 4-class candidate: racket recall 0.95, table recall 0.77, floor recall 0.62, noise recall 0.89 |
| 2026-05-06 | `bounce_imu_side_check` | `audio_session_2026-05-06_003` racket-bouncing IMU | 1.000 | 1.00 | Diagnostic only: FH-side vs BH-side separates cleanly across 6 takes; binary bounce IMU cannot train yet because there are no reviewed not-bounce IMU rows |
| 2026-05-06 | `audio_contact_model` | `contact_variants/human_reviewed` + session 007/008 | 0.591 | 0.95 | Improved but still too positive-heavy: `not_racket_contact` recall 0.23; not exported |
| 2026-05-06 | `audio_contact_model` | `contact_variants/all_legacy` + session 007/008 | 0.888 | 0.92 | Worse than previous 0.907 candidate, so the app contact model was not replaced |
| 2026-05-06 | `audio_model` | `audio_dataset.csv` + session 007/008 | 0.823 | 0.80 | 4-class candidate improved clearly with speech/music/impact data and was exported to app `audio_model.json` |
| 2026-05-06 | `bounce_imu_model` | `bounce_imu_dataset.csv` + rescued no-bounce take | 0.656 | 1.00 | First binary run with both classes: 493 positive windows and 55 no-bounce windows; test fold had no negatives, so this is diagnostic only |
| 2026-05-06 | `audio_contact_model` | `contact_variants/all_legacy` + sessions 009/010 | 0.878 | 0.91 | Still below previous 0.907 candidate; `racket_contact` recall 0.96, `not_racket_contact` recall 0.70 |
| 2026-05-06 | `audio_model` | `audio_dataset.csv` + sessions 009/010 | 0.821 | 0.87 | Similar to prior 0.823 CV; `racket_bounce` recall 0.95, `table_bounce` recall 0.94, `floor_bounce` recall 0.64, `noise` recall 0.64 |
| 2026-05-06 | `bounce_imu_model` | `bounce_imu_dataset.csv` + 4 no-bounce takes | 0.870 | 0.99 | First promising binary bounce-IMU run: 525 positive windows and 221 no-bounce windows; needs more sessions before app export |
| 2026-05-11 | Review config diagnostic | `audio_session_2026-05-11_001` diagnostic-only | N/A | N/A | Same WAV compared across 9 review configs; not used for training by explicit instruction |
| 2026-05-12 | `audio_model` | `audio_dataset.csv` + sessions 2026-05-12 001/002 | 0.806 | 0.88 | Exported: `racket_bounce` recall 0.96 stayed stable, `floor_bounce` recall improved to 0.67, and `noise` recall improved to 0.64 |
| 2026-05-12 | `audio_contact_model` | `contact_variants/all_legacy_2026-05-12` | 0.883 | 0.94 | Exported: `racket_contact` recall 0.98 and `not_racket_contact` recall 0.83; new noisy floor/noise negatives helped hard-negative behavior |
| 2026-05-12 | `audio_model` | `audio_dataset.csv` + `audio_session_2026-05-12_004` | 0.797 | 0.88 | Candidate only, not exported: `racket_bounce` recall rose to 0.97, but `noise` recall fell to 0.60 versus the exported 0.64 |
| 2026-05-12 | `audio_contact_model` | `contact_variants/all_legacy_2026-05-12_with_004` | 0.876 | 0.94 | Candidate only, not exported: high-music racket rows are now represented, but macro-F1 is lower than the exported 0.883 model |
| 2026-05-12 | `audio_model` | `audio_dataset.csv` + sessions 005/006 | 0.805 | 0.90 | Candidate only, not exported: `racket_bounce` F1 improved to 0.94, `floor_bounce` F1 to 0.82, and `noise` F1 to 0.66, but macro-F1/test balance fell versus the active 05-12 pair |
| 2026-05-12 | `audio_contact_model` | `contact_variants/all_legacy_2026-05-12_with_005_006` | 0.889 | 0.95 | Candidate only, not exported: higher accuracy and `racket_contact` F1 0.97, but `racket_contact` recall dropped to 0.96 and `not_racket_contact` F1 dropped versus the active candidate |
| 2026-05-12 | `audio_model` | sessions 009/010 normal vs tight vs hybrid windows | 0.830 | 0.91 | Tight window won this experiment: 4-class macro-F1 79.7% vs normal 78.8% and hybrid 79.3%; table F1 improved from 74.8% normal to 76.9% tight |
| 2026-05-12 | `audio_contact_model` | sessions 009/010 normal vs tight vs hybrid windows | 0.818 | 0.95 | Binary contact did not benefit materially from tight windows; normal and tight were effectively equal, while hybrid was slightly weaker on grouped test |
| 2026-05-13 | `audio_model` | sessions 009/010/011 tight windows | 0.798 | 0.90 | Candidate only, not exported: adding 011 did not improve the tight candidate; `table_bounce` F1 was 76.5% and many dense table markers were skipped by the current negative-near-racket safety rule |
| 2026-05-13 | `audio_contact_model` | sessions 009/010/011 tight windows | 0.829 | 0.95 | Candidate only, not exported: binary stayed strong overall but did not beat the active model balance; racket recall was 95.6% and not-racket recall 90.7% |
| 2026-05-13 | `audio_model` | sessions 009/010/011 tight windows + true window-overlap skip | 0.821 | 0.90 | Exported after holdout validation: overlap skip restored dense table examples and improved vs old 011 run; table/racket behavior must be checked on Motorola |
| 2026-05-13 | `audio_contact_model` | sessions 009/010/011 tight windows + true window-overlap skip | 0.829 | 0.95 | Exported after holdout validation: binary metrics were unchanged by the overlap fix; not-racket recall stayed 90.7% while racket recall stayed 95.6% |
| 2026-05-13 | `audio_model` holdout | `audio_session_2026-05-13_001` with tight-overlap candidate | N/A | 0.97 | Holdout/prov: 39 trainable dense markers, 38 correct; racket recall 100%, table recall 94.7%, one table predicted as racket |
| 2026-05-13 | `audio_contact_model` holdout | `audio_session_2026-05-13_001` with tight-overlap candidate | N/A | 0.97 | Holdout/prov: 39 trainable dense markers, 38 correct; racket recall 100%, not-racket recall 94.7% |
| 2026-05-13 | Replay simulation | `audio_session_2026-05-13_001..005` with current 1 s candidate replay | N/A | N/A | Current Collector pair on reviewed session 005: sensitive/hybrid got 92/99 racket hits with 1 FP; normal/hybrid got 77/99. Dense imported session 001 exposed the remaining close table/racket candidate problem: 20 racket hits truth, 0 counted by the 1 s replay because windows were table-dominant |
| 2026-05-13 | Model/window sweep | Reviewed sessions from 2026-05-11..13 across historical exports | N/A | N/A | Broad replay compared historical model pairs, `-60/+140` through `-300/+700`, and Collector/Stiga hybrid semantics. Best overall was 2026-05-12-with-005/006 4-class-only at `-100/+200 ms` with F1 0.795; current Collector pair also peaked at `-100/+200 ms`, sensitive hybrid, F1 0.777 |
| 2026-05-13 | `audio_model` | `-100/+200 ms` reviewed-only source of truth | 0.847 | 0.91 | 5,826 rows from reviewed markers only. Strong racket recall, but table/floor/noise coverage is still thin, especially table in grouped test |
| 2026-05-13 | `audio_model` | `-100/+200 ms` master with weak legacy/bootstrap | 0.801 | 0.91 | 24,085 rows with explicit source/trust. Better broad class coverage than reviewed-only, but legacy is not treated as truth |
| 2026-05-13 | `audio_contact_model` | `-100/+200 ms` reviewed-only source of truth | 0.775 | 0.94 | 5,313 rows from reviewed markers only. Racket recall 0.99, not-racket recall 0.70; needs more reviewed negatives |
| 2026-05-13 | `audio_contact_model` | `-100/+200 ms` all-legacy comparison/bootstrap | 0.869 | 0.90 | 19,660 rows. More balanced hard-negative behavior, but this is a weak-data comparison path, not the single source of truth |
| 2026-05-13 | Final Collector candidate replay | 4-class all-sources JSON, native onset ratio 1.5, `-100/+200 ms`, normal confidence 0.65 | N/A | N/A | Representative replay: precision 0.895, recall 0.520, F1 0.658 with 145 TP / 17 FP / 134 FN before device validation. It improves over the rolled-back Collector replay but still needs Motorola live testing, especially high music and dense table/racket |
| 2026-05-13 | `audio_model` A/B/C window sweep | all current audio data + reviewed sessions 008/009 | 0.817 | 0.90 | Candidate only, not exported: A `-100/+200 ms` beat B `-60/+140 ms` and C adaptive on grouped test macro-F1/table F1; A table F1 54.1% vs B 46.8% vs C 48.0%, while racket recall stayed ~96% for all |
| 2026-05-13 | `audio_model` dense reviewed holdout | reviewed-only training with sessions 008/009 held out | N/A | N/A | Candidate only, not exported: current Collector and reviewed-only both scored 57.2% on raw 008/009 holdout; 009 table bounces were mostly predicted as racket. Training with 009-like dense data fixed 008 (90.6%) and in-sample 008/009 (98.8%), but 008 alone did not generalize to 009. Need more 009-like dense racket/table sessions before export |
| 2026-05-13 | `dense play collection scenarios` | Collector app + local raw metadata | N/A | N/A | Added `playing_dense_audio` and `playing_dense_imu` scenarios for realistic racket/table play; local sessions 008/009 were reclassified as dense play analysis data so future reports can separate them from imported/generic audio |

## Scenario Snapshot

| Date | Scenario | Exact match / note |
|---|---|---|
| 2026-04-22 | `racket_quiet` | 0.978 on grouped test |
| 2026-04-22 | `racket_counting` | 0.905 on grouped test |
| 2026-04-22 | `racket_music_low` | 0.950 on grouped test |
| 2026-04-22 | `speech_only` | 1.000 on grouped test |
| 2026-04-22 | `desk_keyboard_only` | 1.000 on grouped test |
| 2026-04-22 | `racket_music_mid` (`all_legacy` binary) | 0.838 exact match on grouped test |
| 2026-04-22 | `racket_counting` (`all_legacy` binary) | 0.953 exact match on grouped test |
| 2026-04-23 | `racket_quiet` (`all_legacy` binary) | 0.991 exact match on grouped test |
| 2026-04-23 | `racket_counting` (`all_legacy` binary) | 0.942 exact match on grouped test |
| 2026-04-23 | `racket_music_low` (`all_legacy` binary) | 0.980 exact match on grouped test |
| 2026-04-23 | `speech_only` (`all_legacy` binary) | 1.000 exact match on grouped test |
| 2026-04-23 | `desk_keyboard_only` (`all_legacy` binary) | 0.950 exact match on grouped test |

## Build History

| Date | Build / revision | Main change | Installed on Motorola | Result |
|---|---|---|---|---|
| 2026-04-22 | `review r3` | Slow playback, delete marker, native 15s stop | Yes | Usable but still too hard to review |
| 2026-04-22 | `review r5` | Landscape review UI, overview/detail model, clearer marker semantics | Yes | Better, but needed full-page scroll |
| 2026-04-22 | `review r5 + scroll` | Whole review page vertically scrollable in landscape | Yes | Latest installed build |
| 2026-04-22 | `audio+imu collector v1` | New `Studs audio + IMU` mode with table calibration, synced IMU per take, separate bounce IMU preprocess/train/export scripts | Yes | Ready for first synchronized data round |
| 2026-04-22 | `all_legacy audio_contact export` | Exported `all_legacy` binary audio contact model into app `audio_contact_model.json` and rebuilt release APK | Yes | Installed for on-device validation |
| 2026-04-23 | `audio models retrain + export` | Exported refreshed `all_legacy` binary contact model and updated 4-class audio model into new release APK | Yes | Installed as new test build after larger 30 s reviewed round |
| 2026-05-04 | `immersive recorder + 150hz imu metadata` | Immersive fullscreen shell, focused countdown recorder, current-session review queue, safe-area review padding, and IMU timestamp/quality metadata | Yes | Installed on Motorola; manual flow validation still pending |
| 2026-05-04 | `recorder crashfix + dashboard collection ui` | Added Android vibration permission and reshaped collection screen toward the provided dashboard mockup | Yes | Installed on Motorola; verify countdown no longer crashes and UI spacing matches device |
| 2026-05-04 | `review dashboard ui` | Reshaped review screen toward the provided video-first mockup with marker controls, waveform card, IMU preview, and large save/discard buttons | Yes | Installed on Motorola; verify physical marker review ergonomics |
| 2026-05-04 | `collection scroll/back + synced review timeline` | Fixed collection bottom scroll/back behavior, filtered standard review markers through contact/surface models, added review zoom, draggable synced timeline, and line-based IMU preview | Yes | Installed on Motorola; needs manual device pass for scroll, drag, zoom, and marker relevance |
| 2026-05-04 | `collection/review polish + zoom follow` | Polished collection typography/text, removed ML mark and V-style controls, added Swedish review copy, video aspect handling, throttled scrub, wider playhead hit target, and zoom-window follow/autoscroll | Yes | Installed on Motorola; verify portrait/landscape video and 2x/4x/8x timeline behavior |
| 2026-05-04 | `review audio-video sync` | Review video now seeks from `audio_origin_in_video_ms + audioMs`, with per-take manual `video_sync_offset_ms` controls | Yes | Installed on Motorola; verify a clear racket contact lines up in video and waveform |
| 2026-05-04 | `sync-marker video calibration` | Collection shows a clap/tap sync cue; Review detects the early sync spike, shows it on the audio+IMU timeline, and computes `video_sync_offset_ms` from `Synka här` | Yes | Installed on Motorola; validate with a fresh sync take |
| 2026-05-05 | `review approve-all + queue/progress polish` | Added `Godkänn alla`, fixed the session progress ring, made the current-session review queue open as a list, and hid the legacy IMU-only `Datainsamling` card | Yes | Installed on Motorola; validate approve-all edge cases, queue rows, and progress ring states |
| 2026-05-05 | `audio plus imu scenarios` | Moved `Playing` under `Audio plus IMU`, restored mixed racketstuds as a racket-bounce context, made pose calibration optional, and exported scenario metadata | Yes | Installed on Motorola; validate startsida Data cards, Racketstuds/Playing scenario UI, optional pose calibration, and JSON metadata |
| 2026-05-05 | `playing review confidence filter` | Limited Playing review to FH hit, BH hit, and table bounce, added auto-marker confidence filtering, collapsed video sync, added 12x/16x zoom, and improved long-video playback start by seeking the original WAV at 1x | Yes | Installed on Motorola; validate Playing label speed, confidence thresholds, sync collapse, long-video play start, and high-zoom playhead behavior |
| 2026-05-06 | `bounce imu negative collection` | Added guided `Racketrörelse utan studs`, hid guided mixed racketstuds, enabled racket-bouncing long-press labels, allowed markerless no-bounce save, and excluded no-bounce data from audio training | Yes | Installed on Motorola; collect and save a no-bounce take with zero markers, then rerun bounce-IMU preprocessing to confirm both classes |
| 2026-05-06 | `session008 audio model export` | Pulled session 008 noisy audio and session 007 left-handed IMU, retrained audio/IMU candidates, exported only the improved 4-class audio model | Yes | Installed on Motorola; binary contact model intentionally left unchanged |
| 2026-05-06 | `no-bounce markerless review crashfix` | Fixed markerless no-bounce review crash caused by selecting `orderedMarkers[0].id` when there are zero markers | No | Built release APK only; not installed on Motorola per Love's request |
| 2026-05-11 | `review config switcher` | Added Review algorithm controls for sensitivity/model mode, config regeneration, candidate snapshots, and negative-overlap training safety | Yes | Built from `codex/training-layers` and installed on Motorola `ZY22L6NDHV`; `audio_session_2026-05-11_001` remains diagnostic-only |
| 2026-05-13 | `tight-overlap audio model APK` | Exported tight-overlap 4-class and binary contact models, then aligned Review and Live model windows to `-60/+140 ms` | Yes | TypeScript/root validation passed, release build passed, installed on Motorola `ZY22L6NDHV` at 08:48 |
| 2026-05-13 | `stable live rollback APK` | Restored 2026-05-12 exported audio model pair and the previous 1 s `-300/+700 ms` Review/Live model window after tight live regression | Yes | Release build passed and installed on Motorola `ZY22L6NDHV` at 10:13 |
| 2026-05-13 | `reviewed-session replay simulation` | Pulled today's sessions 001-005 and replayed review-style candidates across Collector/Stiga model pairs and strict/normal/sensitive x hybrid/binary/4-class configs | No | No APK/model export; output written to `.tmp_device_sessions/simulation_2026-05-13_today.csv` |
| 2026-05-13 | `historical model/window sweep` | Compared app current, installed Stiga APK models, and historical 2026-05-11/12/13 export pairs across main inference windows and detection modes | No | No APK/model export; corrected output written to `.tmp_device_sessions/model_window_sweep_2026-05-13_corrected.csv` |
| 2026-05-13 | `4-class 100/200 Collector candidate` | Exported `bench_2026-05-13_100_200/audio_4class_all_sources`, set Collector default to `Normal / 4-klass`, changed live/review inference to `-100/+200 ms`, and used a lower native onset gate with 4-class filtering | Yes | TypeScript/root validation passed, Gradle release build passed, installed on Motorola `ZY22L6NDHV` |
| 2026-05-12 | `audio model retrain 001/002` | Pulled today's reviewed noise and noisy-floor sessions, skipped unfinished review-required takes, retrained and exported both audio JSON models | No | Model JSON artifacts updated only; APK not built or installed |
| 2026-05-12 | `audio candidate with high-music racket` | Added `audio_session_2026-05-12_004`, retrained both audio models, and compared against the already-exported 05-12 pair | No | Candidate trained locally but not exported because overall 4-class and binary macro-F1 were lower than the active app artifacts |
| 2026-05-12 | `audio candidate with sessions 005/006` | Added high-music racket, noisy floor/noise, clap/noise, and imported iPhone racket audio, then retrained 4-class and binary contact candidates | No | Candidate trained locally but not exported because it improves some class metrics while weakening overall validation balance |

## Data Collection History

| Date | Round | Data type | Count | Notes |
|---|---|---|---:|---|
| 2026-04-21 | `audio_round_001` | Guided audio takes | 36 takes | Scenario round used to retrain binary audio contact model |
| 2026-04-22 | `review_round_ongoing` | Reviewed markers | In progress | Review UX still being stabilized before more data |
| 2026-04-22 | `audio_imu_round_pending` | Synced audio + IMU takes | 0 takes | Collector mode implemented, waiting for first calibrated synchronized round |
| 2026-04-22 | `audio_imu_round_001` | Synced audio + IMU takes | 4 takes | 2 `racket_counting` + 2 `racket_music_mid`, all reviewed and ingested |
| 2026-04-22 | `audio_imu_round_002` | Synced audio + IMU takes | 7 takes | 6 `racket_quiet` + 1 `racket_counting`, all reviewed and ingested |
| 2026-04-23 | `audio_round_2026-04-23_014` | Guided audio takes | 14 takes | 3 `racket_quiet`, 3 `racket_counting`, 2 `speech_only`, 3 `table_quiet`, 3 `floor_quiet`, all ingested |
| 2026-05-05 | `audio_session_2026-05-05_019` | Playing audio/video/IMU review pull | 1 take | 8:09 long Playing take pulled from Motorola; JSON contains 324 markers, 313 filtered auto-candidates, and 11 edited ground-truth markers |
| 2026-05-06 | `audio_session_2026-05-06_003` | Racket-bouncing audio/video/IMU review pull | 6 takes | 3 forehand-side and 3 backhand-side takes, 239 reviewed racket contacts, roughly 160 Hz measured IMU |
| 2026-05-06 | `audio_session_2026-05-06_007` | Left-handed racket-bouncing audio/video/IMU pull | 7 takes | 3 FH-side + 3 BH-side reviewed takes with 254 racket contacts and one markerless no-bounce take rescued locally after crash |
| 2026-05-06 | `audio_session_2026-05-06_008` | Noisy audio-only pull | 5 takes | 2 racket+speech, 1 racket+music, and 2 other-bounce/noise takes; 158 reviewed markers added to audio training |
| 2026-05-06 | `audio_session_2026-05-06_009` | Noisy/floor audio-only pull | 4 takes | 1 `racket_speech`, 2 `floor_bounce`, and 1 `other_bounce_noise`; 156 reviewed markers, all 30 s WAVs present |
| 2026-05-06 | `audio_session_2026-05-06_010` | Music racket-bounce plus no-bounce IMU pull | 4 takes | 1 reviewed racket-bounce take with 32 manual/edited contacts and 3 markerless reviewed no-bounce IMU takes; racket take background was locally marked `music_mid` from Love's feedback |
| 2026-05-11 | `audio_session_2026-05-11_001` | Device diagnostic only | 1 take | `racket_music_low_001.wav` was inspected from device/temp storage to compare review configs; Love marked it as not trainable, so it must stay out of training raw data |
| 2026-05-11 | `audio_session_2026-05-11_002` | Reviewed music racket audio pull | 2 takes | Pulled from Motorola into `data/audio/raw`; Love marked both takes trainable. Sanity report: take 1 has 11 racket + 14 music/noise labels; take 2 has 15 racket + 15 floor + 1 music/noise labels and should be checked before final model export |
| 2026-05-12 | `audio_session_2026-05-12_001` | Reviewed noise audio pull | 3 takes | Pulled from Motorola into `data/audio/raw`; 66 reviewed `voice_music_noise` markers from loud music/whistling/noise |
| 2026-05-12 | `audio_session_2026-05-12_002` | Noisy floor-bounce audio pull | 2 takes | Pulled from Motorola into `data/audio/raw`; both takes are `floor_noisy` with 59 reviewed `floor_bounce` markers and no IMU |
| 2026-05-12 | `audio_session_2026-05-12_004` | Reviewed high-music racket audio pull | 1 take | Pulled from Motorola into `data/audio/raw`; `racket_music` with `music_high`, 40 reviewed racket contacts and 11 reviewed voice/music/noise markers |
| 2026-05-12 | `audio_session_2026-05-12_005` | Reviewed mixed audio pull | 5 takes | Pulled from Motorola into `data/audio/raw`; 2 high-music racket takes, 1 noisy floor take, and 2 noise/clap takes, all reviewed |
| 2026-05-12 | `audio_session_2026-05-12_006` | Imported iPhone audio review pull | 1 take | Pulled from Motorola into `data/audio/raw`; imported `Racket lugn 001.m4a`, 40 auto-confirmed racket markers, 42 model candidates |
| 2026-05-12 | `audio_session_2026-05-12_009` | Imported iPhone table-bounce review pull | 2 takes | Pulled from Motorola into `data/audio/raw`; `007` and `008` were empty session shells, while `009` contains 71 reviewed `table_bounce` markers across two imported audio takes |
| 2026-05-12 | `audio_session_2026-05-12_010` | Imported iPhone tight play-sequence review pull | 1 take | Pulled from Motorola into `data/audio/raw`; 53 reviewed markers alternating table/racket, with min gap around 180 ms |
| 2026-05-12 | `audio_session_2026-05-12_011` | Imported iPhone very tight play-sequence review pull | 1 take | Pulled from Motorola into `data/audio/raw`; 89 reviewed markers, 45 `table_bounce` and 44 `racket_bounce`, with min gap around 125 ms |
| 2026-05-13 | `audio_session_2026-05-13_001` | Imported iPhone tight play-sequence review pull | 1 take | Pulled from Motorola into `data/audio/raw`; 45 reviewed markers, 25 `table_bounce` and 20 `racket_bounce`, min gap 147 ms; keep as holdout/test until Love decides to add it to training |
| 2026-05-13 | `audio_session_2026-05-13_002..005` | Today's Motorola audio pulls | 4 sessions | Pulled from Motorola into `data/audio/raw`; `002`/`003` have 30 s racket WAVs but no review markers, `004` is empty, and `005` has two reviewed racket takes: 49 quiet racket hits and 50 racket+speech hits |

## Current Decisions

| Date | Decision | Why |
|---|---|---|
| 2026-05-11 | Add detection config, candidate, and review-truth layers | Love needs to see what the algorithm proposed and what was corrected without training on unreviewed auto detections; full model retraining remains offline |
| 2026-05-13 | Match app inference windows to tight model exports | Tight-trained audio models must classify the same `-60/+140 ms` event window in Review and Live; longer preview clips remain only for human listening |
| 2026-05-13 | Do not ship tight live inference until full-chain replay passes | Motorola testing showed tight live missed too many quiet racket bounces and could count speech at sensitive thresholds, despite strong marker-window/holdout metrics |
| 2026-05-13 | Human-reviewed markers are the single source of truth | Legacy rows can be useful for coverage, but they must stay explicitly marked as weak/bootstrap and never silently define final truth |
| 2026-05-13 | Use full live-chain replay before exporting the next Collector audio model | Stiga's better live behavior is mostly candidate/onset/gate behavior, not just a better model JSON |
| 2026-05-11 | Do not train on `audio_session_2026-05-11_001` | Love explicitly marked this session as diagnostic-only |
| 2026-05-11 | Allow `audio_session_2026-05-11_002` in training | Love explicitly marked both music takes as trainable |
| 2026-05-12 | Skip unfinished review-required takes in preprocessing | New takes can exist in a session before Love has reviewed them; they must not fall back to legacy auto-training |
| 2026-05-12 | Export the 2026-05-12 audio model pair | Today's reviewed noisy hard negatives improved the 4-class model and gave a stronger binary contact candidate than the active app artifact |
| 2026-05-12 | Do not export the first high-music-only retrain candidate | `audio_session_2026-05-12_004` adds useful coverage, but one high-music racket take is not enough to improve overall validation metrics |
| 2026-05-12 | Do not export the sessions 005/006 retrain candidate automatically | New iPhone/imported and high-music data is useful, but the candidate improves selected classes while weakening the overall validation balance; keep it as a local comparison model |
| 2026-05-13 | Do not export the first 009/010/011 tight candidate | The model trained successfully, but 011 exposed that the fixed 300 ms negative-overlap skip removes many real table bounces in dense table/racket sequences; preprocessing should skip only true window overlap before this data is used fully |
| 2026-05-13 | Replace fixed 300 ms negative skip with true training-window overlap | Dense play has legitimate table/racket gaps below 300 ms; with tight `-60/+140 ms` windows, those examples should train unless their actual extracted windows overlap |
| 2026-04-22 | Keep audio as the primary contact truth | It already works better than other paths and has real reviewed data |
| 2026-04-22 | Do not replace audio with IMU | IMU bounce work should be additive and removable |
| 2026-04-22 | If IMU is added, make it a separate bounce-specific model | Bounce motion is not the same as a normal pingis swing |
| 2026-04-22 | Keep audio review as the label source even in synced collection mode | One reviewed timeline should supervise both audio and IMU training |
| 2026-04-22 | Use `0.5-1.0 s` spacing between bounces in the base collection round | Keeps reviewed clips clean and easier to label |
| 2026-04-22 | Postpone fast double-bounce collection | Base review and base contact model are not stable enough yet |
| 2026-04-22 | Compare binary contact variants with legacy inclusion before shipping a new APK | We needed evidence, not guesses, about whether old 4-class data should feed the binary model |
| 2026-04-22 | Prefer `all_legacy` over current binary contact dataset for the next audio APK candidate | It is the first local binary variant that raises noisy positive recall without collapsing completely on negatives |
| 2026-04-23 | Keep IMU work paused and prioritize audio data plus audio model updates | New audio round materially improved negative recall and gave enough evidence to justify another audio-only test APK |
| 2026-04-24 | Treat review video as labeling support, not model input | The current ML pipeline trains from WAV features and review markers; adding video to ML would create a new problem before audio is stable |
| 2026-04-24 | Keep training the 4-class model even though binary is primary | Table/floor/noise separation is useful for live veto/debug and may become product behavior later |
| 2026-05-04 | Default contact training to human-reviewed markers | Prevents algorithm-proposed events from becoming final truth without review |
| 2026-05-04 | Keep legacy data as opt-in variants | Existing data is valuable for bootstrap/comparison but weaker than explicit review labels |
| 2026-05-04 | Split racket-bounce prompts from FH/BH labels | FH/BH is coverage and IMU-side metadata for bounce, not a separate audio class |
| 2026-05-04 | Keep bounce IMU and stroke IMU separate | Repeated racket-bounce motion and in-play stroke motion should not be trained as one task |
| 2026-05-06 | Use natural racket-arm motion as bounce-IMU negative data | Fake no-ball bounce rhythm is too similar to real racket bouncing and would create ambiguous IMU ground truth |
| 2026-05-04 | Prefer 150 Hz raw IMU collection when stable | Higher-resolution raw data can be downsampled later; measured rate and quality flags decide whether a take is trustworthy |
| 2026-05-04 | Hide old pending review samples from collection | Love should only see current-session pending work while collecting new takes; old samples are preserved on disk |
| 2026-05-04 | Use a clap/tap sync event for review video calibration | The saved `audio_origin_in_video_ms` is a good default, but Love needs a visible and audible anchor that can compute `video_sync_offset_ms` without guessing direction |
| 2026-05-05 | Hide legacy `Datainsamling` from startsida | It points to older IMU-only collection; keep the route/code as internal fallback while showing only active user-facing collection flows |
| 2026-05-05 | Move `Playing` under `Audio plus IMU` | Longer free play capture is a scenario for synced audio/video/IMU review, not a separate top-level home card |
| 2026-05-05 | Make FH/BH pose calibration optional after table baseline | Pose calibration is helper metadata; raw IMU should still be collected with `calibration_status: partial` if poses are skipped |

## Open Questions

| Date | Question | Current answer |
|---|---|---|
| 2026-04-22 | Is review UI simple enough to trust labeling? | Not fully confirmed yet |
| 2026-04-22 | Should the next implementation be synchronized audio + IMU collection? | Done |
| 2026-04-22 | Is the current binary audio model strong enough to preserve? | Yes |
| 2026-04-22 | Is synced review usable enough to start collecting bounce IMU data? | Not verified yet |
| 2026-04-22 | Should trusted legacy only be enough for binary contact? | No, not on the current local corpus; it is far too small and too positive-heavy |
| 2026-04-24 | Is the audio model stable enough for app-level bounce counting? | Not yet; duplicate counts and missed angled FH/BH contacts remain |
| 2026-04-24 | Is the next data gap more floor/table negatives or angled racket positives? | Both matter; current feedback says angled FH/BH contacts are under-covered and floor remains weak |
| 2026-05-04 | Should FH/BH racket-bounce be separate audio classes? | No; keep one audio contact label and store FH/BH as prompt/IMU-side metadata |
| 2026-05-04 | Should legacy data remain in the real model path? | Yes as opt-in variants, but primary metrics should report human-reviewed performance separately |

## Current Collection Checklist

Date: 2026-05-12

Goal: strengthen the audio models with noisy hard negatives and noisy racket positives. Use `Ljudinsamling` for all rows in this checklist; review each take before it is used for training.

| Priority | Scenario to record | Target | Status | Notes |
|---|---|---:|---|---|
| 1 | `Brus/negativt` with loud music or whistling | 3 x 30 s | Done | Pulled as `audio_session_2026-05-12_001`; 66 reviewed `not_racket_contact` / `voice_music_noise` markers are ready for the next training run |
| 2 | `Golvstuds stökigt` | 3 x 30 s | Partial | `audio_session_2026-05-12_002` has 2 reviewed takes with 59 `floor_bounce` markers; record 1 more take |
| 3 | `Bordsstuds stökigt` | 2-3 x 30 s | Pending | Bounce on a real ping-pong table or table-like playing surface, not an arbitrary desk |
| 4 | `Racket + hög musik` | 3 x 30 s | Partial | `audio_session_2026-05-12_004` has 1 reviewed `music_high` take with 40 racket contacts; record 2 more takes |
| 5 | `Racket + prat` | 1-2 x 30 s | Optional | `audio_session_2026-05-11_005` already gives useful corrected speech data; add more only if voice/distance/noise differs |
| 6 | `Brus/negativt` speech-only or music-only without ball impacts | 2 x 30 s | Optional | Useful if the app creates false peaks; label as `Brus`, not `Annat` |

Review rules for this round:
- Confirm or edit real racket, table, floor, and noise events; human-reviewed markers are the only training truth.
- Use `Brus` for music, speech, whistle, and phone noise.
- Use `Ignorera` for ambiguous overlaps near a real racket hit instead of forcing a negative label.
- Do not collect IMU for table, floor, or noise in this audio-only checklist.

## Next Planned Step

| Date | Step | Status |
|---|---|---|
| 2026-04-22 | Write concrete spec for `Bounce audio + IMU collection` | Done |
| 2026-04-22 | Implement synchronized collector, save IMU in session JSON, and add first bounce IMU scripts | Done |
| 2026-04-22 | Run first calibrated `Studs audio + IMU` collection round and review takes | Pending |
| 2026-04-22 | Rebuild binary contact model using `all_legacy` mapping if local/device validation still looks best after one more noisy round | Done |
| 2026-04-22 | Validate the installed `all_legacy` APK on Motorola in quiet, counting, and music scenarios before replacing the default training path permanently | Superseded by 2026-04-23 refreshed model |
| 2026-04-23 | Validate the refreshed `all_legacy` test APK on Motorola, especially `floor only`, `table only`, `racket_counting`, and duplicate-count behavior | Pending |
| 2026-04-24 | Decide whether to add explicit angled racket-contact collection prompts for forehand/backhand-style bounce hits | Pending |
| 2026-04-24 | Investigate duplicate count after one bounce plus catch/after-sound using live debug rows | Pending |
| 2026-05-04 | Collect the new reviewed racket-bounce protocol: FH, BH, mixed, table, floor, catch-after-sound, speech/music | Pending |
| 2026-05-04 | Compare `human_reviewed`, `legacy_hybrid`, and `bootstrap` contact variants before exporting a new APK model | Pending |
| 2026-05-04 | Validate `contact_group` debug on Motorola with 20 FH, 20 BH, table, floor, and catch-after-sound tests | Pending |
| 2026-05-04 | Validate immersive recording flow, current-session review queue, safe-area review, and 150 Hz IMU metadata on Motorola | Pending |

## AI Handoff Checklist

When another AI assistant starts work on this project:
1. Read [AGENTS.md](</C:/Users/lovea/Desktop/dev/STIGA SPORTS/pingis_ml_algo/AGENTS.md>).
2. Load relevant Skills, especially [pingis-audio-classification](</C:/Users/lovea/Desktop/dev/STIGA SPORTS/pingis_ml_algo/skills/pingis-audio-classification/SKILL.md>) for audio work and [pingis-stroke-detection](</C:/Users/lovea/Desktop/dev/STIGA SPORTS/pingis_ml_algo/skills/pingis-stroke-detection/SKILL.md>) for IMU work.
3. Read this `ITERATION_LOG.md` before changing collector, review, model scripts, or live bounce logic.
4. Be critical: compare model variants with grouped splits, inspect scenario breakdowns, and do not assume more data automatically means a better live model.
5. Keep audio as the active priority until the known audio problems are resolved on device.
6. Update this file after every meaningful data pull, retrain, APK build, or device feedback round.

## Entry Log

| Date | Entry | Built / changed | Feedback | Decision | Next |
|---|---|---|---|---|---|
| 2026-04-22 | A | Review-first binary contact labeling direction confirmed | Review too hard to use | Prioritize review UX before more data | Simplify review UI |
| 2026-04-22 | B | Landscape review UI, slower playback, orientation lock | Still too confusing and hard to control | Keep binary audio path, continue simplifying review | Make review physically usable on-device |
| 2026-04-22 | C | Scrollable review screen installed, collector copy clarified | User wants separate bounce IMU idea evaluated and audio path preserved | Create project memory outside `AGENTS.md` | Write synchronized collector spec |
| 2026-04-22 | D | Added `Bounce audio + IMU collection` build spec and table-based log | User wants shared visibility into model development over time | Keep log in repo root and update after every build / retrain / major feedback round | Implement synchronized collector or continue review fixes |
| 2026-04-22 | E | Added `Studs audio + IMU` collector path with table calibration reuse, synced IMU save per take, and bounce IMU preprocess/train/export scripts | No synced round has been collected yet, so the new IMU model is not trained | Keep audio model untouched and start with a separate removable bounce IMU path | Run first synced collection round and review it |
| 2026-04-22 | F | Pulled synced session `audio_session_2026-04-22_008`, ingested 4 reviewed takes, built first bounce IMU dataset and baseline RF | Data is usable, but still too thin and imbalanced to trust yet | Keep collecting synced noisy rounds before using IMU live | Add more `not_bounce_contact` and more varied synced takes |
| 2026-04-22 | G | Compared three binary audio-contact dataset variants: current, trusted legacy, and all legacy | User challenged why old 4-class data was excluded from the binary model | `all_legacy` is currently the strongest local binary candidate; `trusted_legacy` alone is too small and collapses on negatives | Collect one more noisy reviewed round, then decide whether to export an `all_legacy`-trained audio contact model to app |
| 2026-04-23 | H | Pulled `audio_session_2026-04-23_014`, ingested a larger 30 s round, retrained both binary and 4-class audio models, and exported a new test APK | Binary model improved where it matters most: `not_racket_contact` recall rose from 0.66 to 0.78 while `racket_contact` stayed at 0.96 | Ship another audio-only test APK and validate floor/table false positives and duplicate counts on device | Run Motorola validation before collecting the next targeted round |
| 2026-05-04 | I | Implemented the data-quality plan: strict review statuses, new bounce presets, synced IMU timestamps, contact grouping, source/trust dataset variants, report breakdowns, and stroke x-axis features | Not device-tested yet | Keep audio-first, make human review the primary truth, and keep legacy data as weaker opt-in variants | Run validation, then collect the new Motorola protocol before retraining/exporting |
| 2026-05-04 | J | Added immersive Android shell, focused countdown recording UI, current-session review queue, safe-area review padding, and 150 Hz raw IMU quality metadata | Built and installed on Motorola; manual flow validation still pending | Keep collection simple and preserve higher-resolution raw IMU when stable | Run Motorola fullscreen, review, and IMU metadata checks |
| 2026-05-04 | K | Fixed start-recording crash caused by missing `VIBRATE` permission and redesigned collection UI toward Love's dashboard mockup | Built and installed on Motorola | Vibration is acceptable feedback, but must be manifest-permitted; collection should be dashboard-first rather than full camera-first | Verify start countdown, camera preview, and review transition on device |
| 2026-05-04 | L | Rebuilt review UI around a large video panel, marker controls, full-take waveform, IMU preview, and bottom save/discard actions | Built and installed on Motorola | Review should prioritize fast human labeling over dense debug controls | Validate review flow on a new take and note any overflow/spacing issues |
| 2026-05-04 | M | Tightened collection scroll/back and review timeline without changing core data logging | Built and installed on Motorola | Standard review should show model-filtered, preset-relevant markers instead of raw envelope peaks | Run Motorola pass: small-screen scroll, back during countdown/recording, marker drag, zoom, and synced IMU line preview |
| 2026-05-04 | N | Polished Datainsamling and Review Take without a full redesign: smaller collection text, no ML badge, Swedish copy, text-based back, video `contain` aspect handling, throttled scrub, playhead hitbox, zoom follow and edge autoscroll | Built and installed on Motorola | Keep the current review structure, but make navigation and timeline interaction physically easier on device | Test portrait/landscape video, zoomed audio+IMU follow, playhead edge-scroll, and collection typography on Motorola |
| 2026-05-04 | O | Fixed Review Take audio/video sync by applying saved `audio_origin_in_video_ms`, added manual ±100/±250 ms video offset controls, and persisted `video_sync_offset_ms` per take | Built and installed on Motorola | Treat video as review support that must align with audio markers; no beep calibration added | Test existing offset take, then save/reopen after manual sync adjustment |
| 2026-05-04 | P | Added a guided clap/tap sync workflow: collection prompts one visible sync event, Review detects the first strong sync spike, separates it from training markers, and applies `Synka här` as `video_sync_offset_ms` | Built and installed on Motorola | Prefer a visible+audible sync event over ambiguous +/- nudges; keep video audio disabled to avoid mic conflicts | Verify sync with a fresh take |
| 2026-05-05 | Q | Added Review `Godkänn alla`, replaced the progress-ring border hack, opened current-session review queue as a list, and hid legacy `Datainsamling` from startsida | Built and installed on Motorola | Keep the active data path focused on reviewed audio and synced audio+IMU; preserve legacy IMU-only code as fallback | Test approve-all, queue navigation, progress states, and startsida Data cards |

| 2026-05-05 | R | Separated audio-only scenarios from audio+IMU FH/BH prompts, added `Fri inspelning`, and stored reviewed markers with binary plus detailed class labels | Built and installed on Motorola | Audio-only must not claim FH/BH truth; long free recordings should be labeled in Review after capture | Validate startsida, long free recording, queue opening, table/floor labels, and export columns |
| 2026-05-05 | S | Limited guided `Studs audio + IMU` to racket contacts, kept table/floor/noise in `Ljudinsamling`, and clarified that one session JSON contains multiple scenario events | Built and installed on Motorola | IMU should be collected only where it adds racket-motion value; hard negatives stay audio-only | Validate FH/BH/mixed IMU flow and audio-only floor/table/noise flow |
| 2026-05-05 | T | Pulled and inspected `audio_session_2026-05-05_008`: 3 FH takes, 3 BH takes, and 1 floor take | Local analysis only; no model retrain | Reviewed racket takes contain 40 confirmed markers each, first marker after sync cue, clean spacing around 0.54-0.58 s, and no clipped marker windows | Use the racket takes for reviewed audio+IMU data; ignore floor IMU for v1 and collect future floor via `Ljud-insamling` |
| 2026-05-05 | U | Reworked startsida Data and Audio plus IMU structure: `Ljudinsamling` plus `Audio plus IMU`, with `Racketstuds` and `Playing` scenarios and exported `scenario`/`bounce_context`/`calibration_status` metadata | Built and installed on Motorola | Racket bouncing and playing are separate IMU contexts; `Playing` replaces the separate `Fri inspelning` card | Validate scenario UI, optional pose calibration, JSON metadata, review labels, and bounce-IMU preprocessing filter |
| 2026-05-05 | V | Tightened Playing Review: only FH/BH/table labels, visible auto confidence, strict/medium/all confidence filters, collapsible video sync, higher timeline zoom, and original-WAV seek for 1x long-video playback | Built and installed on Motorola | Long Playing reviews need fewer labels and stricter auto-candidate control; filtered candidates must not become training truth | Device-test a four-minute Playing take for play startup, label speed, threshold counts, sync collapse, and 16x timeline dragging |
| 2026-05-05 | W | Pulled and inspected `audio_session_2026-05-05_019` from Motorola: one 8:09 Playing take with WAV, MP4, IMU, saved video sync, and filtered auto-candidates | Local analysis only; no model retrain | Session 019 is very likely Love's long test, but saved JSON has 11 edited markers rather than the expected 13 | Check whether two intended markers were not saved; otherwise treat this as a usable selective ground-truth review sample |
| 2026-05-06 | X | Added focused audio-only noisy/fast collection presets and changed reviewed audio preprocessing to short event-centered windows with nearest-event spacing metadata | App/script validation pending | Fast racket bounces and close table/racket sounds are normal data, not dirty edge cases; legacy data remains usable but weaker | Validate app UI, preprocess output columns, and scenario/background/spacing breakdowns before retraining |
| 2026-05-06 | Y | Pulled and inspected `audio_session_2026-05-06_003`: 3 forehand-side and 3 backhand-side racket-bounce IMU takes | Local analysis plus `preprocess_bounce_imu.py`; no model export | 239 reviewed racket-contact markers produced usable `-180/+220 ms` IMU windows at ~160 Hz; FH/BH side is highly separable in a quick grouped check, but only positive racket-contact data exists | Add reviewed not-bounce IMU or keep this dataset for side/orientation work rather than binary contact training |
| 2026-05-06 | Z | Rebuilt audio datasets with short reviewed windows, trained binary contact variants, trained a 4-class audio candidate, and reran bounce-IMU preprocessing | Local training only; no app model export | `all_legacy` remains the strongest binary candidate, reviewed-only is too positive-heavy, 4-class floor/table separation still needs more reviewed hard negatives, and binary bounce-IMU cannot train without reviewed not-bounce IMU windows | Collect reviewed noisy/fast audio hard negatives and add explicit no-contact/not-bounce IMU windows before exporting new production artifacts |
| 2026-05-06 | AA | Added guided Audio plus IMU negative collection for natural racket-arm motion without ball contact, hid mixed guided racketstuds, enabled racket-bouncing quick labels, and then allowed no-bounce takes to save with zero markers | TypeScript/Python/root validation passed; built and installed on Motorola | Bounce-IMU negatives are whole reviewed no-bounce takes; preprocessing samples their IMU windows automatically, and `no_bounce_motion` stays out of audio training | Record a no-bounce take, save without markers, then rerun `preprocess_bounce_imu.py` to confirm both classes |
| 2026-05-06 | AB | Fixed the markerless no-bounce review crash after Love's session 007 test | TypeScript/Python/root validation passed; release APK built but not installed | Empty marker lists are valid only for `racket_motion_no_bounce`; Review must not select `orderedMarkers[0]` when the list is empty | Install when Love is ready, then retest saving a no-bounce take with zero markers |
| 2026-05-06 | AC | Pulled session 008 noisy audio and session 007 left-handed IMU, rebuilt datasets, trained audio/IMU candidates, and exported the improved 4-class audio model only | TypeScript/root validation passed; release APK built and installed on Motorola | More reviewed noisy audio improved the 4-class surface/debug model, but the strongest binary contact model regressed and should stay unchanged | Test noisy audio debug/veto and markerless no-bounce save on Motorola |
| 2026-05-06 | AD | Pulled and inspected `audio_session_2026-05-06_009` while Love continued collecting IMU data | Local data pull only; no retrain | Session 009 adds useful reviewed racket+speech, floor, and voice/music/noise negatives for the audio model | Include 009 in the next audio retrain after the current IMU round is pulled |
| 2026-05-06 | AE | Pulled session 010, rebuilt audio/contact/IMU datasets, and trained contact, 4-class audio, and bounce-IMU candidates | Local training only; no app export | Audio candidates did not beat the best current app model, but bounce-IMU became meaningfully trainable with 221 no-bounce windows | Collect more no-bounce IMU groups and fix/confirm FH/BH-side labels before exporting IMU |
| 2026-05-11 | AF | Added review-time algorithm config switching and ran a diagnostic comparison on `audio_session_2026-05-11_001` | TypeScript/Python validation passed; no model retrain or APK install | Review can compare strict/normal/sensitive and hybrid/binary/4-class without replacing human markers; session 001 is diagnostic-only and not training data | Test config switching on Motorola, then collect a separate trainable noisy-music take if needed |
| 2026-05-11 | AG | Built and installed the review config switcher APK on Motorola `ZY22L6NDHV` | Release build successful and installed via ADB | Device now has the `codex/training-layers` build with Review config controls; no model retrain was done | Run a short review pass and confirm config switching preserves manual markers |
| 2026-05-11 | AH | Added colored review/candidate pins, a `Racket + musik` level selector, and pulled trainable session 002 | TypeScript/Python/preprocess/root validation passed; release APK built and installed on Motorola | Review now shows class colors in the audio timeline, new music takes can store low/medium/high background level, and session 002 is local training input while session 001 stays blocked | Review the unexpected floor labels in `racket_music_low_002.wav` before exporting a final model |
| 2026-05-11 | AI | Aligned Review timeline candidate pins to one row and filtered weak non-review candidate dots | TypeScript/root validation passed; release APK built and installed on Motorola `ZY22L6NDHV` | Colored pins should describe model class, not visual height; low-value candidates should not make the waveform feel zoomed out | Validate same-row class-colored pins in Review on a noisy music take |
| 2026-05-11 | AJ | Increased Review waveform bin density and visual amplitude so audio peaks are easier to inspect | TypeScript/root validation passed; release APK built and installed on Motorola `ZY22L6NDHV` | The timeline should show more detail per zoom level while markers stay on a single class-colored row | Reopen session 004 and compare waveform readability at 1x/4x/8x/16x |
| 2026-05-11 | AK | Pulled sessions 002-005, corrected session 005 from `racket_fast` to `racket_speech`, rebuilt audio datasets, retrained audio models, and exported only the improved 4-class model | Preprocess, 4-class/contact training, JSON export, TypeScript/root validation, release build, and Motorola install | New 4-class candidate improves today's reviewed rows and grouped macro-F1; binary contact candidate is kept as candidate only because grouped hard-negative behavior is still weak | Validate `Hybrid/4-klass` Review on today's music/speech/floor takes before replacing the binary contact model |
| 2026-05-12 | AL | Pulled sessions 005/006, rebuilt audio datasets, and retrained 4-class plus binary contact candidates | Preprocess and both training scripts completed; no export/build/install | The new data is useful, especially iPhone imported racket and high-music racket, but the retrain is not a clean overall win versus the active app artifacts | Keep collecting targeted floor/noise/table negatives and use this candidate for analysis, not release |
| 2026-05-12 | AM | Pulled and inspected imported iPhone table-bounce session 009 | Local data pull only; no retrain | Session 009 adds exactly the missing iPhone/table coverage: two reviewed imported-audio takes with 71 `table_bounce` markers | Include 009 in the next audio retrain after any remaining iPhone table/racket imports are reviewed |
| 2026-05-12 | AN | Added optional tight/hybrid reviewed audio windows and compared normal/tight/hybrid on sessions 009/010 | Preprocess and training completed for 4-class plus binary candidates; no export/build/install | Tight reviewed windows are the best current direction for dense table/racket sequences in 4-class audio; hybrid preprocessing did not beat tight in this run | Rerun the tight candidate including newly pulled session 011 before deciding whether to export |
| 2026-05-13 | AO | Trained the tight candidate including session 011 | Preprocess, contact variant build, 4-class training, and binary training completed; no export/build/install | 011 is valuable test/review data, but the current preprocessing skipped most dense table examples because they sit inside 300 ms of racket markers | Change negative-overlap safety to use the actual training window, then rerun tight before exporting |
| 2026-05-13 | AP | Replaced the fixed 300 ms negative skip with actual training-window overlap and retrained tight on sessions 009/010/011 | Preprocess, contact variant build, 4-class training, and binary training completed; no export/build/install | The fix restored more dense table/noise examples, but the candidate remains mixed: better than old 011 run, not a clean export win | Keep the overlap rule, collect/use fresh holdout validation before exporting a tight model |
| 2026-05-13 | AQ | Pulled and inspected new imported iPhone session 001 | Local data pull only; no retrain | Session 001 is useful dense table/racket validation data and should not be silently folded into training before we use it to test the current candidate | Evaluate the tight-overlap candidate against this holdout before deciding whether to train on it |
| 2026-05-13 | AR | Evaluated the tight-overlap candidate against holdout session 001 | Local evaluation only; no retrain/export/build/install | The candidate performed strongly on fresh dense iPhone table/racket data: 38/39 correct after excluding true overlapping windows | Consider exporting the tight-overlap model pair for Motorola validation, but keep session 001 as holdout unless Love asks to fold it into training |
| 2026-05-13 | AS | Exported the tight-overlap model pair, patched app model windows to `-60/+140 ms`, rebuilt release APK, and installed it on Motorola | TypeScript/root validation and Gradle release build passed; APK installed on `ZY22L6NDHV` | Avoid train/runtime mismatch: model inference is tight, review preview remains human-friendly | Test dense imported table/racket sessions and live racket counting on device before collecting the next round |
| 2026-05-13 | AT | Rolled Studsdetektor back to the 2026-05-12 stable model pair and previous 1 s live/review window after Love's live regression report | Gradle release build passed; APK installed on `ZY22L6NDHV` | Offline marker-window metrics are not enough; live must be evaluated with full-chain replay before export | Build a WAV+JSON replay simulator for onset -> model -> merge -> count and compare windows/configs before the next live export |
| 2026-05-13 | AU | Pulled today's sessions 001-005 and ran a review-style replay simulation across Collector/Stiga model pairs and all 9 config modes | Local replay only; no model retrain/export/build/install | Fresh reviewed racket data is useful, but the dense imported table/racket take still shows that candidate timing/windowing, not only model score, is the main blocker | Use session 005 for trainable racket/noisy-racket data, review or ignore sessions 002/003, and build a closer live-chain replay before exporting again |
| 2026-05-13 | AV | Ran a historical model/window sweep on reviewed sessions from 2026-05-11..13 | Local replay only; no model retrain/export/build/install | The best replay result is a compromise window around `-100/+200 ms`, not the old 1 s window and not the tightest `-60/+140`; 4-class-only often beats hybrid in this review-style replay, while Stiga's installed JSON pair only makes sense together with its WebAudio onset gate | Next test should replay the full live candidate generator, especially Stiga-style onset/spectral gate versus Collector native onset, before another APK export |
| 2026-05-13 | AW | Trained `-100/+200 ms` source variants and ran a reduced full-chain live replay on representative reviewed files | Local training/replay only; no model export/build/install | Human-reviewed-only is now the single truth path, but weak legacy still improves broad negative coverage; the strongest replay was Stiga-style sensitive candidate gating plus `-100/+200 ms` 4-class/all-sources, while current Collector live is held back by onset/gate behavior and dense table/racket timing | Align Collector's live candidate gate with the replay winner, then rerun replay and Motorola live validation before exporting |
| 2026-05-13 | AX | Built and installed the 4-class `-100/+200 ms` Collector candidate | TypeScript/root validation and Gradle release build passed; APK installed on Motorola `ZY22L6NDHV` | App default is now `Normal / 4-klass`; binary/hybrid remain selectable debug modes, while model export decisions still require human-reviewed truth plus live-chain replay | Love should test quiet racket, racket+speech, high music, speech-only, and dense table/racket before this becomes the next stable baseline |
| 2026-05-13 | AY | Fixed live bounce undercount by reducing the JS contact-group window from `650 ms` to `180 ms` and normal merge from `420 ms` to `260 ms` | TypeScript/root validation and Gradle release build passed; APK installed on Motorola `ZY22L6NDHV`, SHA256 `4E65BB500F09482A94D868C17C7DB3B1737FD1059EED8202F775C25A6EA57228` | The observed 50 real bounces -> ~25 counts matches an overlong live grouping window, not just a model problem | Retest quiet fast racket bounces first, then speech-only false positives |
| 2026-05-13 | AZ | Added adjustable native `Retrigger window` to live audio and lowered its default from `380 ms` to `220 ms` | TypeScript/root validation and Gradle release build passed; APK installed on Motorola `ZY22L6NDHV`, SHA256 `41235C7EC350BCDF30F977B8D4804F7B50D765951AE0E7A52B399E029922F3BF` | If merge changes do not affect counts, misses are likely happening before the model in native onset/retrigger | Test `Retrigger 220/180/120`, `Merge 220/180`, and speech-only false positives |
| 2026-05-13 | BA | Added visible adjustable `Group window` control in live detector and Studsdetektor | TypeScript/root validation and Gradle release build passed; APK installed on Motorola `ZY22L6NDHV`, SHA256 `EAFBD5FD3F34BF2360D70D015A46369C841530E5A9F8F3CB1A70524A4810CE90` | The app must expose all three live timing gates: retrigger, group, and merge | Compare `Group 0/80/180` at the same retrigger/merge values |
| 2026-05-13 | BB | Promoted the named working baseline `collector_bounce_live_v2026_05_13_normal_4class_220_80_220` | TypeScript/root validation and Gradle release build passed; APK installed on Motorola `ZY22L6NDHV`, SHA256 `0417E538A5D8B72639D6BA1BED441D8CB16C76488829F609684F20A70F80BAD8` | Love's Motorola test passed `40/40` for `Normal / 4-klass`, `Retrigger 220`, `Group 80`, `Merge 220`, while `Group 0` once double-counted `41/40`; model JSONs now include metadata | Promote this branch through PR/merge as the Collector single source of truth baseline |

## How To Update This File

After each meaningful round, update at least:
1. `Metric History`
2. `Build History`
3. `Data Collection History`
4. one row in `Entry Log`

If a decision changes, update `Current Decisions` instead of burying it in free text.
