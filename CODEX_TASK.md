# CODEX_TASK.md

Fill this in before asking Codex to implement project work. Use one active ticket per implementation pass, and keep the scope small enough to verify manually.

Quick read-only questions, repo exploration, and lightweight planning do not require a filled ticket. Code changes, infrastructure changes, dependency changes, and documentation updates that affect source-of-truth files should use a ticket.

## Ticket ID

`T0026`

## Branch

`codex/t0026-retrain-playing-retro-2026-06-04`

## Status

`Completed`

## Goal

Retrain a local `spel_retro_audio` candidate using `audio_session_2026-06-04_001` plus the historical playing-retro training data, because T0025 showed the remaining misses are primarily model classification and table-threshold failures rather than missing audio peaks.

This ticket should produce a candidate model and training/evaluation report only. It must not export app JSON, build or install an APK, change Review UX, change `playingRetroAudio.ts`, or affect `studs_live`.

## Dependencies

- T0025 completed audit for `audio_session_2026-06-04_001`.
- T0025 local raw data exists under `data/audio/raw/audio_session_2026-06-04_001.json` and `data/audio/raw/audio_session_2026-06-04_001/`.
- T0022 retrain script and T0023 replay script are available as implementation references.
- T0024 installed baseline is `playing_retro_audio_rf_v2026_06_03_t0022_multi_window_context`, racket threshold `0.0`, table threshold `0.5`, and 80 ms same-label dedupe.

## Allowed Areas

- `skills/pingis-audio-classification/scripts/`
- `data/audio/processed/`
- `data/audio/models/evaluations/`
- `data/audio/models/playing_retro_candidates/`
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

## Requirements

- Build a deterministic T0026 training script, preferably reusing the T0022 multi-window/context training shape.
- Include `audio_session_2026-06-04_001` as playing-retro training data together with historical dense playing sessions.
- Keep ordinary up/down bounce and `studs_live` out of the training/export path.
- Preserve the current feature family: tight `-60/+140 ms`, normal `-100/+200 ms`, wide `-160/+320 ms`, plus non-leaky candidate context.
- Use human-reviewed final markers as truth.
- Treat saved model candidates that do not match reviewed racket/table truth as `non_target` rows.
- Add manually reviewed missed racket/table markers as positive target rows.
- Do not use truth-derived close-event fields as model features.
- Report row counts by label, session, and row source.
- Report leave-one-session-out or equivalent holdout metrics on at least the promotion sessions: 05-28_002, 05-29_001, 05-29_002, 06-03_005, and 06-04_001.
- Save the trained candidate under `data/audio/models/playing_retro_candidates/`.
- Save evaluation summaries under `data/audio/models/evaluations/`.
- Recommend whether T0027 should replay/tune this candidate against the T0024 baseline.

## Non-Goals

- No app JSON export.
- No app code or UI changes.
- No APK build/install.
- No threshold selection for app runtime beyond analysis notes.
- No `studs_live`, ordinary bounce, `audio_model.json`, or `audio_contact_model.json` changes.
- No video-stroke work.

## Acceptance Criteria

- T0026 training script exists and is reproducible.
- T0026 candidate model artifacts exist locally.
- T0026 report shows dataset size, label balance, session coverage, and holdout/promotion-session metrics.
- Report explicitly compares whether 06-04 holdout behavior improved enough to justify T0027 replay.
- Docs record the selected T0026 candidate and next ticket.
- Root validation passes.

## Completion Notes

- Added deterministic training script `skills/pingis-audio-classification/scripts/train_playing_retro_audio_t0026.py`.
- Trained local-only candidate `playing_retro_audio_rf_v2026_06_04_t0026_multi_window_context`.
- Candidate artifacts were saved under `data/audio/models/playing_retro_candidates/playing_retro_audio_rf_v2026_06_04_t0026_multi_window_context/`.
- Evaluation/report outputs were saved under:
  - `data/audio/models/evaluations/playing_retro_audio_t0026_retrain_report.md`
  - `data/audio/models/evaluations/playing_retro_audio_t0026_retrain_eval.csv`
  - `data/audio/models/evaluations/playing_retro_audio_t0026_retrain_predictions.csv`
  - `data/audio/processed/playing_retro_audio_candidate_rows_t0026_2026_06_04.csv`
  - `data/audio/processed/playing_retro_audio_multi_window_dataset_t0026_2026_06_04.csv`
- Dataset: 4,598 rows across 18 sessions.
- Labels: 1,891 `racket_contact`, 1,966 `table_bounce`, and 741 `non_target`.
- 06-04 contribution: 274 rows, 80 racket, 85 table, 109 non-target, 268 candidate rows, and 6 `manual_missed_marker` rows. The rest of Love's manual additions entered training through corrected nearby candidate rows.
- Selected variant: `multi_window_context_racket_weighted`, 197 features.
- Leave-one-session-out 06-04 check: accuracy 0.854, racket recall 0.812, table recall 0.906, non-target recall 0.844.
- Final in-sample 06-04 focus check: corrected 26/30 baseline target rows that T0024 had effectively called `non_target`, and 6/6 `manual_missed_marker` rows.
- Older reference safety held versus T0022: 05-28 improved, 05-29_001 stayed equal, 05-29_002 changed only -0.009 racket / -0.008 table, and 06-03 changed -0.010 racket / -0.019 table with non-target +0.035.
- Recommendation: proceed to T0027 replay/tune against the installed T0024 marker-level baseline before any app export/build/install.
- Validation passed: Python compile for the T0026 script, T0026 training command, root `npm run validate`, and scoped `git diff --check`.
- No app export, app code, APK build/install, `studs_live`, ordinary bounce, `audio_model.json`, `audio_contact_model.json`, or video model changes were made.

## Manual Verification

Love should be able to read the T0026 summary and understand:

- what data was used,
- how many rows came from the 06-04 reviewed session,
- whether the model learned from the 35 manual additions,
- and why T0027 should or should not replay/tune this candidate.

## Automated Validation

- `python -m py_compile` for the T0026 training script.
- T0026 training command.
- `npm run validate`.
- `git diff --check` for touched docs/scripts.

## Completion Report Expected

Codex should report:

- Candidate model ID and artifact paths
- Total rows and label/session breakdown
- 06-04 contribution and how manual additions entered training
- Holdout/promotion-session metrics
- Recommendation for T0027
- Confirmation that no app export, APK build, `studs_live`, ordinary bounce, or video model changes were made
