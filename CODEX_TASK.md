# CODEX_TASK.md

Fill this in before asking Codex to implement project work. Use one active ticket per implementation pass, and keep the scope small enough to verify manually.

Quick read-only questions, repo exploration, and lightweight planning do not require a filled ticket. Code changes, infrastructure changes, dependency changes, and documentation updates that affect source-of-truth files should use a ticket.

## Ticket ID

`T0004`

## Branch

`codex/t0004-playing-retro-candidate-peaks`

## Goal

Build the first offline candidate-peak dataset/report step for `spel_retro_audio`, so training can learn from the exact peaks the app finds and misclassifies instead of only ideal human marker timestamps.

## Dependencies

- T0001 ticket workflow adoption is complete.
- T0002 documentation refresh removed IMU/AirHive from active project scope.
- T0003 cleanup removed retired IMU/AirHive workflow docs and skill scripts from the active repo workflow.
- Current audio source-of-truth lives in `PROJECT_CONTEXT.md`, `DECISIONS.md`, and `ITERATION_LOG.md`.
- Existing audio scripts and replay behavior live under `skills/pingis-audio-classification/scripts/`.

## Allowed areas

- `skills/pingis-audio-classification/scripts/`
- `skills/pingis-audio-classification/SKILL.md`
- `data/audio/processed/`
- `data/audio/models/evaluations/`
- `PROJECT_CONTEXT.md`
- `DECISIONS.md`
- `ITERATION_LOG.md`
- `REPO_CURRENT_STATE.md`
- `FOLLOWUPS.md`
- `CODEX_TASK.md`

## Do not touch

- `apps/collector/src/models/audio_model.json`
- `apps/collector/src/models/audio_contact_model.json`
- APK/build artifacts
- `studs_live` app behavior or live detector thresholds
- Video-stroke model files
- Raw reviewed session JSON labels, except for metadata fixes explicitly approved by Love

## Requirements

- Produce a candidate-centered report for dense play audio sessions.
- Match runtime/replay candidate peaks to human-reviewed markers with explicit offset and match status.
- Include positive examples, false positives, missed markers, ambiguous close events, and app-predicted labels when available.
- Keep ordinary up/down bounce separate from dense playing buckets.
- Do not train a model in this ticket.
- Do not export or install any app model in this ticket.

## Non-Goals

- No `spel_retro_audio` model training yet.
- No Collector UI integration yet.
- No video/FH-BH work yet.
- No changes to ordinary `studs_live` promotion rules.

## Acceptance criteria

- A deterministic command can generate the candidate-peak report from current local data.
- Report output includes per-session and per-bucket counts for matched racket, matched table, false positives, missed markers, and dense close-event cases.
- The report clearly shows where current runtime candidates differ from reviewed marker timestamps.
- Results are documented enough that T0005 can train from the generated candidate-centered rows.

## Manual verification

- Inspect generated report rows for `audio_session_2026-05-29_002`.
- Confirm close table/racket sequences are represented instead of collapsed into one event.

## Automated validation

- Run the new or updated report command.
- Run `npm run validate` if source-of-truth workflow files changed.
- Run a targeted Python syntax check for any changed Python script.

## Completion Report Expected

Codex should report:

- Summary of changes
- Files changed
- Commands run
- Report outputs generated
- Validation results
- Manual verification performed or still needed
- Docs updated or still needed
- Risks or unresolved questions
- Follow-up tickets for out-of-scope work
