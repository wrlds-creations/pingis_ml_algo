# CODEX_TASK.md

Fill this in before asking Codex to implement project work. Use one active ticket per implementation pass, and keep the scope small enough to verify manually.

Quick read-only questions, repo exploration, and lightweight planning do not require a filled ticket. Code changes, infrastructure changes, dependency changes, and documentation updates that affect source-of-truth files should use a ticket.

## Ticket ID

`T0131-studs-fhbh-live-v2-hybrid22`

## Branch

`codex/t0057-fable-auto-improvement-loop`

## Status

`Completed`

## Goal

Add a separate collector UI entry named `Studs FH/BH LIVE v2` that keeps the current camera/rubber-side flow but uses the STIGA `Hybrid 2.2` bounce-audio setup as the sound trigger.

## Dependencies

- The existing `Studs FH/BH LIVE` screen uses Fable audio as the bounce trigger and then uses the live racket color tracker for FH/BH side.
- The STIGA app has a tested `Hybrid 2.2` audio path:
  - PCM audio;
  - adaptive RMS/spectral candidate gate;
  - centered clip extraction;
  - binary racket-contact RF;
  - 4-class surface veto;
  - T0129 learned hard-negative veto;
  - Android defaults: contact `0.25`, veto `0.01`, bypass enabled at `0.61`, dedupe about `180-220 ms`.
- The collector already contains the binary contact model, 4-class surface model, RF runtime, and audio feature extractor.

## Allowed Areas

- `CODEX_TASK.md`
- `PROJECT_CONTEXT.md`
- `REPO_CURRENT_STATE.md`
- `ITERATION_LOG.md`
- `DECISIONS.md`
- `apps/collector/App.tsx`
- `apps/collector/src/SetupScreen.tsx`
- `apps/collector/src/BounceSideLiveScreen.tsx`
- `apps/collector/src/audioContactEngine.ts`
- `apps/collector/src/types.ts`
- `apps/collector/src/logisticRuntime.ts`
- `apps/collector/src/models/hybrid_hard_negative_veto_t0129_logreg.json`

## Do Not Touch

- Do not change the existing `Studs FH/BH LIVE` behavior.
- Do not change camera/rubber-side native tracking unless required for the new audio route.
- Do not replace `fable_audio_model.json`, `audio_model.json`, or `audio_contact_model.json`.
- Do not change `Bounce audio test`.
- Do not delete raw/generated `data/` or `raw/`.
- Do not merge to `main`.
- Do not use cloud APIs or AWS.

## Requirements

- Add a new visible setup entry: `Studs FH/BH LIVE v2`.
- Route v2 separately from the current `Studs FH/BH LIVE`.
- Reuse the existing camera tracker and side counting behavior.
- Use Hybrid 2.2-style audio decisions for v2:
  - contact RF raw `racket_contact` probability threshold `0.25`;
  - surface veto at confidence `0.75`;
  - T0129 learned hard-negative veto threshold `0.01`;
  - bypass learned veto when raw contact probability is greater than `0.61`;
  - app-side dedupe/grouping around the STIGA defaults.
- Native candidate gate should use the collector native RMS/spectral gate, not the old Fable bandpass-only setup.
- Keep v1 Fable audio path unchanged.
- Include useful debug metadata in v2 debug JSON candidates.

## Non-Goals

- No new model training.
- No TestFlight or STIGA app changes.
- No release APK unless requested separately.
- No user-facing production promotion beyond the explicit v2 test entry.

## Acceptance Criteria

- `Studs FH/BH LIVE` still opens the old path.
- `Studs FH/BH LIVE v2` opens a new path using Hybrid 2.2 audio.
- TypeScript validates.
- Source-of-truth docs record the new guarded test entry and remaining risks.

## Completion Notes

- Added `Studs FH/BH LIVE v2` as a separate setup entry and route.
- Kept the existing `Studs FH/BH LIVE` Fable-triggered route unchanged.
- Added portable logistic runtime support and bundled `hybrid_hard_negative_veto_t0129_logreg.json`.
- Extended the collector audio contact engine with `hybrid22`:
  - raw binary contact RF threshold `0.25`;
  - 4-class surface veto confidence `0.75`;
  - T0129 learned hard-negative veto threshold `0.01`;
  - learned-veto bypass when raw contact probability is greater than `0.61`;
  - app-side dedupe `180 ms`.
- V2 starts native audio with broadband RMS plus spectral gate (`abs_min_rms=0.003`, `retrigger_ms=220`) and keeps the same live racket tracker for FH/BH/uncertain side decisions.
- Debug dumps now include the selected audio trigger mode/config plus Hybrid 2.2 decision metadata.
- No native Android code, model retraining, release build, install, push, merge, raw data, or `main` promotion changed.

## Validation

- `cd apps/collector && npx tsc --noEmit`
- `npm run validate`
- `git diff --check` (passed with existing Windows LF-to-CRLF warnings only)
