# CODEX_TASK.md

Fill this in before asking Codex to implement project work. Use one active ticket per implementation pass, and keep the scope small enough to verify manually.

Quick read-only questions, repo exploration, and lightweight planning do not require a filled ticket. Code changes, infrastructure changes, dependency changes, and documentation updates that affect source-of-truth files should use a ticket.

## Ticket ID

`T0027`

## Branch

`codex/t0027-playing-retro-replay-tune-2026-06-04`

## Status

`Completed`

## Goal

Replay and tune the local T0026 `spel_retro_audio` candidate against the installed T0024 marker-level baseline before any app export, APK build, or device install.

This ticket should decide whether the T0026 candidate is safe enough for T0028 export/build/install by comparing marker outcomes on the dense playing sessions from 2026-05-28, 2026-05-29, 2026-06-03, and 2026-06-04.

## Dependencies

- T0024 installed baseline is `playing_retro_audio_rf_v2026_06_03_t0022_multi_window_context`, racket threshold `0.0`, table threshold `0.5`, and 80 ms same-label dedupe.
- T0025 audited `audio_session_2026-06-04_001` and found remaining misses are mostly classification/threshold misses rather than missing raw peaks.
- T0026 trained local candidate `playing_retro_audio_rf_v2026_06_04_t0026_multi_window_context`.
- T0026 local candidate artifacts exist under `data/audio/models/playing_retro_candidates/playing_retro_audio_rf_v2026_06_04_t0026_multi_window_context/`.
- T0026 multi-window dataset exists under `data/audio/processed/playing_retro_audio_multi_window_dataset_t0026_2026_06_04.csv`.

## Allowed Areas

- `skills/pingis-audio-classification/scripts/`
- `data/audio/models/evaluations/`
- `PROJECT_CONTEXT.md`
- `DECISIONS.md`
- `FOLLOWUPS.md`
- `ITERATION_LOG.md`
- `REPO_CURRENT_STATE.md`

## Do Not Touch

- `apps/collector/src/models/audio_model.json`
- `apps/collector/src/models/audio_contact_model.json`
- `apps/collector/src/models/playing_retro_audio_model.json`
- `apps/collector/src/playingRetroAudio.ts`
- `apps/collector/` UI/runtime code
- `studs_live` behavior or live detector thresholds
- Video-stroke model files
- Raw reviewed labels
- APK build artifacts
- T0026 training artifacts unless regeneration is explicitly needed

## Requirements

- Add a deterministic T0027 replay/tuning script, preferably by adapting the T0023 replay shape.
- Load the T0026 joblib candidate and T0026 multi-window dataset.
- Replay at least these sessions:
  - `audio_session_2026-05-28_002`
  - `audio_session_2026-05-29_001`
  - `audio_session_2026-05-29_002`
  - `audio_session_2026-06-03_005`
  - `audio_session_2026-06-04_001`
- Compare saved visible T0024 baseline candidates against T0026 predictions at marker level.
- Sweep racket/table confidence thresholds and keep 80 ms same-label dedupe.
- Report candidate-level metrics and marker-level TP, wrong class, FP, missed, racket/table TP, racket/table missed, and duplicate removals.
- Report per-session marker totals so 06-04 can be judged separately from older Tomas/Stiga sessions.
- Include focus summaries for 06-03 and 06-04 showing how many baseline target errors T0026 fixes.
- Save replay predictions, evaluation CSV, threshold sweep CSV, JSON report, and Markdown report under `data/audio/models/evaluations/`.
- Recommend whether T0028 should export/build/install the T0026 candidate and selected thresholds.

## Non-Goals

- No retraining.
- No app JSON export.
- No app code or UI changes.
- No APK build/install.
- No `studs_live`, ordinary bounce, `audio_model.json`, or `audio_contact_model.json` changes.
- No video-stroke work.

## Acceptance Criteria

- T0027 replay script exists and is reproducible.
- T0027 report compares T0026 against the installed T0024 marker baseline across the required sessions.
- T0027 report includes the selected racket/table thresholds and 80 ms same-label dedupe setting.
- T0027 recommendation is explicit: proceed to T0028 export/build/install, or do not export yet.
- Docs record the replay outcome and next ticket.
- Root validation passes.

## Completion Notes

- Added deterministic replay/tuning script `skills/pingis-audio-classification/scripts/replay_playing_retro_audio_t0027.py`.
- Replayed T0026 candidate `playing_retro_audio_rf_v2026_06_04_t0026_multi_window_context` against the installed T0024 visible-marker baseline.
- Sessions replayed:
  - `audio_session_2026-05-28_002`
  - `audio_session_2026-05-29_001`
  - `audio_session_2026-05-29_002`
  - `audio_session_2026-06-03_005`
  - `audio_session_2026-06-04_001`
- Candidate rows replayed: 1,202.
- Truth markers replayed: 881.
- Selected settings: racket threshold `0.0`, table threshold `0.45`, same-label dedupe `80 ms`.
- Marker replay improved baseline TP/wrong/FP/missed from `706/22/26/153` to `844/1/9/36`.
- Racket/table TP improved from `314/392` to `409/435`.
- Racket/table missed dropped from `91/62` to `15/21`.
- Candidate-level replay improved accuracy from `0.801` to `0.983`, racket recall from `0.793` to `0.993`, table recall from `0.921` to `0.984`, and non-target recall from `0.657` to `0.971`.
- 06-04 marker replay improved from `130/0/2/35` TP/wrong/FP/missed to `154/0/0/11`.
- 06-04 focus fixed 26/30 baseline target candidate errors and selected 25 recovery/analysis rows under the chosen threshold.
- Outputs were saved under:
  - `data/audio/models/evaluations/playing_retro_audio_t0027_replay_report.md`
  - `data/audio/models/evaluations/playing_retro_audio_t0027_replay_report.json`
  - `data/audio/models/evaluations/playing_retro_audio_t0027_replay_predictions.csv`
  - `data/audio/models/evaluations/playing_retro_audio_t0027_replay_eval.csv`
  - `data/audio/models/evaluations/playing_retro_audio_t0027_threshold_sweep.csv`
- Recommendation: proceed to T0028 export/build/install with selected T0026 model/settings.
- No app export, app code, APK build/install, `studs_live`, ordinary bounce, `audio_model.json`, `audio_contact_model.json`, or video model changes were made.

## Manual Verification

Love should be able to read the T0027 summary and understand:

- whether T0026 beats the app version currently installed on Motorola,
- which threshold settings would be exported in T0028,
- how 06-04 performs separately,
- and why the next ticket should or should not build a new APK.

## Automated Validation

- `python -m py_compile` for the T0027 replay script.
- T0027 replay command.
- `npm run validate`.
- `git diff --check` for touched docs/scripts.

Validation passed:

- `python -m py_compile skills\pingis-audio-classification\scripts\replay_playing_retro_audio_t0027.py`
- `python skills\pingis-audio-classification\scripts\replay_playing_retro_audio_t0027.py`

## Completion Report Expected

Codex should report:

- Baseline versus T0026 marker-level totals
- Selected racket/table thresholds and dedupe setting
- Per-session 06-04 outcome
- Recommendation for T0028
- Confirmation that no app export, APK build/install, `studs_live`, ordinary bounce, or video model changes were made
