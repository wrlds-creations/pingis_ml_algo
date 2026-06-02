# CODEX_TASK.md

Fill this in before asking Codex to implement project work. Use one active ticket per implementation pass, and keep the scope small enough to verify manually.

Quick read-only questions, repo exploration, and lightweight planning do not require a filled ticket. Code changes, infrastructure changes, dependency changes, and documentation updates that affect source-of-truth files should use a ticket.

## Ticket ID

`T0012`

## Branch

`codex/t0012-playing-retro-audio-review-apk-install`

## Goal

Build and install a Motorola test APK for the T0011 manual `Spel-retro audio` Review panel only after Love explicitly approves the APK step.

## Dependencies

- T0009 exported separate app model `apps/collector/src/models/playing_retro_audio_model.json`.
- T0010 replay passed on saved Tomas/Stiga Review candidates: 643 saved candidates, accuracy `0.978`, racket recall `0.987`, table recall `0.988`, non-target recall `0.948`.
- T0011 added a visible, manual `Spel-retro audio` Review panel that reclassifies current saved candidates and draws separate retro pins without mutating marker truth.
- T0011 TypeScript, export parity, replay, and root validation passed.
- Existing live/current app audio still uses `apps/collector/src/models/audio_model.json` through the existing normal path.

## Allowed Areas

- Android build commands and build outputs needed for release APK
- APK verification commands
- Device install commands only after Love approves
- `PROJECT_CONTEXT.md`
- `DECISIONS.md`
- `ITERATION_LOG.md`
- `REPO_CURRENT_STATE.md`
- `FOLLOWUPS.md`
- `CODEX_TASK.md`

## Do Not Touch

- `apps/collector/src/models/audio_model.json`
- `apps/collector/src/models/audio_contact_model.json`
- Existing `studs_live` app behavior or live detector thresholds
- Video-stroke model files
- Raw reviewed session JSON labels

## Requirements

- Ask/confirm with Love before building or installing if approval is not already explicit in the latest message.
- Build a release APK that includes the T0011 manual `Spel-retro audio` Review panel.
- Verify the APK bundle still contains `playing_retro_audio_model.json`.
- Verify normal `audio_model.json` and `audio_contact_model.json` are not changed by this ticket.
- Install on Motorola only after Love explicitly approves installation.
- Do not promote `spel_retro_audio` into `studs_live`.

## Non-Goals

- No candidate generation/surfacing for missed markers in this ticket.
- No ordinary up/down bounce model change.
- No video/FH-BH fusion.
- No broad UI redesign.

## Acceptance Criteria

- Release APK builds successfully, or build is intentionally deferred because Love has not approved.
- If installed, Motorola receives the APK and launches.
- APK verification confirms the separate `playing_retro_audio_model.json` is present.
- Docs record build/install status and APK hash if built.

## Manual Verification

- In Review for a playing-mode audio take, press `Kör retro` and confirm separate retro counts/pins appear.
- Confirm saving without pressing `Kör retro` behaves as normal.
- Confirm pressing `Kör retro` does not automatically create confirmed training markers.

## Automated Validation

- Run `cd apps/collector && npx tsc --noEmit`.
- Run `python skills/pingis-audio-classification/scripts/validate_playing_retro_audio_app_export.py`.
- Run `python skills/pingis-audio-classification/scripts/replay_playing_retro_audio_app_export.py`.
- Run root `npm run validate` if source-of-truth workflow files changed.
- Run the existing Gradle release build/verification commands if APK build is approved.

## Completion Report Expected

Codex should report:

- Summary of changes
- Files changed
- Commands run
- Validation results
- APK hash and install status if built
- Manual verification still needed on Motorola
- Docs updated or still needed
- Risks or unresolved questions
- Follow-up tickets for out-of-scope work
