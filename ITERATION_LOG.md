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
5. Build the binary contact variant with [build_audio_contact_dataset_variant.py](</C:/Users/lovea/Desktop/dev/STIGA SPORTS/pingis_ml_algo/skills/pingis-audio-classification/scripts/build_audio_contact_dataset_variant.py>), currently using `all_legacy`.
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

Audio quality is the priority.
Pause IMU work until the audio detector is stable enough in live tests.

The audio work should continue training both:
- Binary `audio_contact_model`, because it is the live count engine.
- 4-class `audio_model`, because it explains and can veto `table_bounce` and `floor_bounce`, and table bounce may matter later.

Current best binary training strategy is `all_legacy`:
- Map `racket_bounce -> racket_contact`.
- Map `table_bounce`, `floor_bounce`, and `noise -> not_racket_contact`.
- Include reviewed markers and legacy 4-class data.
- This has performed better locally than `current` and `trusted_legacy`.

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
| `audio_contact_model` | `racket_contact` vs `not_racket_contact` | Primary count engine for `Studsdetektor`, `Studs fritt`, and `Studs vaxla sida` | `all_legacy` binary mapping from reviewed markers + legacy 4-class data | Active and highest priority |
| `audio_model` | `racket_bounce / table_bounce / floor_bounce / noise` | Secondary veto/debug model for surface/noise decisions | Multiclass processed audio dataset | Active; keep training alongside binary model |
| `stroke_hit_model` | stroke hit vs miss | Future IMU swing evidence | IMU stroke dataset | Paused until audio is stable |
| `stroke_type_model` | forehand vs backhand stroke type | Future IMU forehand/backhand swing typing | IMU stroke dataset | Paused until audio is stable |
| `bounce_imu_model` | bounce contact motion vs not bounce contact | Not in live app yet | Reviewed synchronized audio + IMU takes | Experimental; scripts exist but live work is paused |

## Metric History

| Date | Model | Dataset | Grouped CV F1 macro | Grouped test acc | Key recall / note |
|---|---|---|---:|---:|---|
| 2026-04-22 | `audio_contact_model` | `audio_contact_dataset.csv` | 0.914 | 0.79 | `racket_contact` recall 0.94, `not_racket_contact` recall 0.70 |
| 2026-04-22 | `audio_model` | `audio_dataset.csv` | 0.694 | 0.74 | Stronger on debug than as primary contact engine |
| 2026-04-22 | `bounce_imu_model` | `bounce_imu_dataset.csv` | 0.847 | 1.00 | First tiny baseline only; test fold happened to contain no negatives |
| 2026-04-22 | `audio_contact_model` | `contact_variants/current` | 0.807 | 0.98 | Current binary dataset is too strict: `racket_contact` recall 0.48, `not_racket_contact` recall 1.00 |
| 2026-04-22 | `audio_contact_model` | `contact_variants/trusted_legacy` | 0.497 | 0.96 | Too few trusted legacy negatives locally; collapsed to always-positive behaviour |
| 2026-04-22 | `audio_contact_model` | `contact_variants/all_legacy` | 0.899 | 0.87 | Best local binary variant so far: `racket_contact` recall 0.96, `not_racket_contact` recall 0.66 |
| 2026-04-23 | `audio_contact_model` | `contact_variants/all_legacy_2026-04-23` | 0.848 | 0.92 | Negative recall improved: `racket_contact` recall 0.96, `not_racket_contact` recall 0.78 |
| 2026-04-23 | `audio_model` | `audio_dataset.csv` | 0.770 | 0.86 | 4-class veto/debug model improved strongly; `table_bounce` recall 0.94, `floor_bounce` recall 0.61 |

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

## Data Collection History

| Date | Round | Data type | Count | Notes |
|---|---|---|---:|---|
| 2026-04-21 | `audio_round_001` | Guided audio takes | 36 takes | Scenario round used to retrain binary audio contact model |
| 2026-04-22 | `review_round_ongoing` | Reviewed markers | In progress | Review UX still being stabilized before more data |
| 2026-04-22 | `audio_imu_round_pending` | Synced audio + IMU takes | 0 takes | Collector mode implemented, waiting for first calibrated synchronized round |
| 2026-04-22 | `audio_imu_round_001` | Synced audio + IMU takes | 4 takes | 2 `racket_counting` + 2 `racket_music_mid`, all reviewed and ingested |
| 2026-04-22 | `audio_imu_round_002` | Synced audio + IMU takes | 7 takes | 6 `racket_quiet` + 1 `racket_counting`, all reviewed and ingested |
| 2026-04-23 | `audio_round_2026-04-23_014` | Guided audio takes | 14 takes | 3 `racket_quiet`, 3 `racket_counting`, 2 `speech_only`, 3 `table_quiet`, 3 `floor_quiet`, all ingested |

## Current Decisions

| Date | Decision | Why |
|---|---|---|
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

## How To Update This File

After each meaningful round, update at least:
1. `Metric History`
2. `Build History`
3. `Data Collection History`
4. one row in `Entry Log`

If a decision changes, update `Current Decisions` instead of burying it in free text.
