# CODEX_TASK.md

Fill this in before asking Codex to implement project work. Use one active ticket per implementation pass, and keep the scope small enough to verify manually.

Quick read-only questions, repo exploration, and lightweight planning do not require a filled ticket. Code changes, infrastructure changes, dependency changes, and documentation updates that affect source-of-truth files should use a ticket.

## Ticket ID

`T0135-fix-fhbh-audio-only-camera-dependency`

## Branch

`codex/t0057-fable-auto-improvement-loop`

## Status

`Completed`

## Goal

Fix the `Studs FH/BH LIVE` audio-only toggle so audio-only mode truly starts from microphone/audio without requiring camera readiness, and so switching back to camera mode can recover the camera preview.

## Dependencies

- T0134 added the first audio-only comparison toggle.
- Love reported that audio-only did not work and that toggling between on/off could leave the camera unable to recover.

## Allowed Areas

- `CODEX_TASK.md`
- `PROJECT_CONTEXT.md`
- `REPO_CURRENT_STATE.md`
- `ITERATION_LOG.md`
- `DECISIONS.md`
- `apps/collector/src/BounceSideLiveScreen.tsx`

## Do Not Touch

- Do not change native Android camera/audio modules.
- Do not retrain or replace any model JSON.
- Do not change `Bounce audio test`.
- Do not change STIGA app code.
- Do not delete raw/generated `data/` or `raw/`.
- Do not merge to `main`.
- Do not use cloud APIs or AWS.

## Requirements

- In audio-only mode, `STARTA` must not require `cameraReady`.
- In audio-only mode, starting a run must not call `startCameraForAiming` before audio starts.
- Switching audio-only off while not running should attempt to start/recover the camera preview.
- If the native camera is already started but React state lost `cameraReady`, recover the state.
- Preserve T0134 behavior: accepted audio bounces count immediately as `OSAKER`, skip tracker/crop side work, and save `decision_source=audio_only`.

## Non-Goals

- No new model or threshold changes.
- No new route.
- No production promotion.

## Acceptance Criteria

- `Audio only: ON` can start counting even if camera preview is not ready.
- Switching `Audio only` off while stopped attempts to restore the camera preview.
- Existing camera-side behavior remains unchanged when audio-only is off.
- TypeScript validates.
- Docs record the bug fix.

## Completion Notes

- Fixed audio-only startup so `STARTA` is enabled when `Audio only: ON` even if the camera is not ready.
- Audio-only runs no longer call `startCameraForAiming` before audio starts.
- Turning `Audio only` on stops/releases the camera preview.
- Turning `Audio only` off forces a camera stop/start recovery attempt.
- Preserved T0134 audio-only counting behavior: accepted audio bounces count immediately as `OSAKER` with `decision_source=audio_only`.
- No native Android code, model JSON, `Bounce audio test`, raw/generated data, release build, push, merge, or production promotion changed.

## Validation

- `cd apps/collector && npx tsc --noEmit`
- `npm run validate`
- `git diff --check` (passed with existing Windows LF-to-CRLF warnings only)
