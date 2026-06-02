# CODEX_TASK.md

Fill this in before asking Codex to implement project work. Use one active ticket per implementation pass, and keep the scope small enough to verify manually.

Quick read-only questions, repo exploration, and lightweight planning do not require a filled ticket. Code changes, infrastructure changes, dependency changes, and documentation updates that affect source-of-truth files should use a ticket.

## Ticket ID

`T0013`

## Branch

`codex/t0013-playing-retro-candidate-surfacing`

## Goal

Add post-recording `spel_retro_audio` candidate generation/surfacing for missed playing-mode racket/table events without changing live bounce behavior or marker truth.

## Dependencies

- T0009 exported separate app model `apps/collector/src/models/playing_retro_audio_model.json`.
- T0010 replay passed on saved Tomas/Stiga Review candidates: 643 saved candidates, accuracy `0.978`, racket recall `0.987`, table recall `0.988`, non-target recall `0.948`.
- T0011 added a visible, manual `Spel-retro audio` Review panel that reclassifies current saved candidates and draws separate retro pins without mutating marker truth.
- T0012 built and installed the Review-only APK on Motorola `ZY22L6NDHV`; APK SHA256 `3422CC2A34A0DAF31BDF03F89FF9CDE4BC2B0CDA7C972C0579D46B5B6C0D5A50`.
- T0010/T0012 still cannot recover 15 reviewed missed Tomas/Stiga markers because no saved app candidate exists at those timestamps.
- Existing live/current app audio still uses `apps/collector/src/models/audio_model.json` through the existing normal path.

## Allowed Areas

- `apps/collector/src/AudioTakeReviewScreen.tsx`
- `apps/collector/src/playingRetroAudio.ts`
- New clearly named helper/replay scripts under `skills/pingis-audio-classification/scripts/`
- Existing evaluation/report outputs under ignored `data/audio/models/evaluations/` or `data/audio/processed/`
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
- APK build/install unless Love explicitly asks for another build after this ticket

## Requirements

- Generate or surface additional post-recording retro candidates near peaks that the saved app candidate list missed.
- Keep generated candidates visibly separate from normal Review markers and the T0011 saved-candidate reclassification output.
- Report how many of the 15 known missed Tomas/Stiga markers are recovered, and how many extra false-positive candidates are introduced.
- Do not automatically create confirmed markers; Love must still confirm/correct/delete anything before it becomes training truth.
- Keep inference context based on app/generated candidate timestamps, never human truth timestamps.
- Do not promote `spel_retro_audio` into `studs_live`.

## Non-Goals

- No ordinary up/down bounce model change.
- No video/FH-BH fusion.
- No broad UI redesign.
- No APK install unless separately approved after replay/UI validation.

## Acceptance Criteria

- Replay/report shows recovered missed markers versus added false positives on Tomas/Stiga `05-28_002`, `05-29_001`, and `05-29_002`.
- Review UI can show generated retro candidates separately without mutating marker/save truth.
- Normal Review and live `studs_live` behavior remain unchanged until the separate retro action is invoked.
- TypeScript and relevant replay/parity checks pass.

## Manual Verification

- In Review for a playing-mode audio take, run retro analysis and confirm saved-candidate pins and generated retro candidates are visually distinguishable.
- Confirm generated candidates can be inspected without becoming confirmed training markers automatically.
- Confirm saving without retro interaction behaves as normal.

## Automated Validation

- Run `cd apps/collector && npx tsc --noEmit`.
- Run `python skills/pingis-audio-classification/scripts/validate_playing_retro_audio_app_export.py`.
- Run `python skills/pingis-audio-classification/scripts/replay_playing_retro_audio_app_export.py`.
- Run any new T0013 replay/report script on the three Tomas/Stiga target sessions.
- Run root `npm run validate` if source-of-truth workflow files changed.

## Completion Report Expected

Codex should report:

- Summary of changes
- Files changed
- Commands run
- Validation results
- Recovered-missed-marker and false-positive counts
- Manual verification still needed on Motorola
- Docs updated or still needed
- Risks or unresolved questions
- Follow-up tickets for out-of-scope work
