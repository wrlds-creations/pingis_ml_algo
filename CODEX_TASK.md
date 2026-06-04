# CODEX_TASK.md

Fill this in before asking Codex to implement project work. Use one active ticket per implementation pass, and keep the scope small enough to verify manually.

Quick read-only questions, repo exploration, and lightweight planning do not require a filled ticket. Code changes, infrastructure changes, dependency changes, and documentation updates that affect source-of-truth files should use a ticket.

## Ticket ID

`T0029`

## Branch

`codex/t0029-playing-retro-review-start-profiling`

## Status

`Active`

## Goal

Profile why fresh `Ljud + video ML` playing imports take too long before the audio waveform/review markers appear, without changing model quality, marker output, training data, or app review semantics.

This ticket exists because Love reported that even a one-minute clip takes too long before review can start. Cache is explicitly out of scope because the same imported video is rarely reopened; the goal is to identify the real first-run bottleneck and produce a timing table that tells us which optimization ticket to do next.

## Dependencies

- T0028 is completed, merged to `main`, and installed on Motorola `ZY22L6NDHV`.
- The installed app uses `playing_retro_audio_rf_v2026_06_04_t0026_multi_window_context` with racket threshold `0.0`, table threshold `0.45`, and 80 ms same-label dedupe.
- Fresh playing imports currently hide the waveform while loading audio/video, generating peak candidates, running `spel_retro_audio`, and creating editable racket/table markers.
- Whole-video pose can start in the background after audio load and may add CPU pressure, but T0029 must measure before changing that behavior.

## Allowed Areas

- `apps/collector/src/AudioCollectionScreen.tsx`
- `apps/collector/src/AudioTakeReviewScreen.tsx`
- `apps/collector/src/playingRetroAudio.ts`
- `apps/collector/src/audioReview.ts`
- `apps/collector/src/types.ts` only if a typed timing/debug shape is needed
- `PROJECT_CONTEXT.md`
- `DECISIONS.md`
- `FOLLOWUPS.md`
- `ITERATION_LOG.md`
- `REPO_CURRENT_STATE.md`

## Do Not Touch

- `apps/collector/src/models/audio_model.json`
- `apps/collector/src/models/audio_contact_model.json`
- `apps/collector/src/models/playing_retro_audio_model.json`
- `apps/collector/src/models/video_stroke_model.json`
- Training scripts or generated model artifacts
- Raw reviewed labels or pulled session JSON
- `studs_live` behavior, thresholds, or ordinary bounce flows
- Cache implementation
- Model thresholds, recovery gates, candidate dedupe, or marker promotion logic

## Requirements

- Add low-overhead timing instrumentation for the fresh `Ljud + video ML` import/review-start path.
- Time these phases separately when available:
  - video import/copy
  - audio extraction to WAV
  - WAV decode/load
  - waveform/review-data generation
  - saved peak candidate generation/load
  - recovery candidate generation
  - `spel_retro_audio` feature extraction
  - `spel_retro_audio` RandomForest prediction
  - marker creation/merge
  - first waveform-ready render
  - pose analysis start/end if it starts before audio review is complete
- Surface or log the timing table in a way Love/Codex can capture from device testing.
- Preserve exact existing audio marker output for the same input/model/settings.
- Do not add caching in T0029.
- Do not optimize yet unless a change is needed only to measure safely.
- Document the measured bottleneck and confirm whether T0030, T0031, or T0032 should be next.

## Non-Goals

- No retraining.
- No threshold tuning.
- No candidate/recovery behavior change.
- No new APK model export.
- No cache.
- No video model changes.
- No UX redesign beyond timing/debug visibility.
- No data pull/audit of a newly reviewed clip.

## Acceptance Criteria

- A T0029 build can report timing for the full path from import selection to waveform-ready audio review.
- The report separates audio extraction/load, candidate generation, retro feature/RF work, marker creation, and pose work.
- A one-minute playing clip produces a concrete timing table.
- The same clip produces the same audio review markers before and after T0029 instrumentation.
- Docs identify the next optimization ticket:
  - T0030 if pose/background work delays audio start or makes the phone sluggish.
  - T0031 if JS `spel_retro_audio` feature/RF/recovery work dominates.
  - T0032 if JS optimization is insufficient and native/background execution is needed.

## Planned Follow-Up Tickets

| Ticket | Goal | Gate |
|---|---|---|
| `T0030` | Defer whole-video pose until after audio review starts or until motion review is opened | Do if T0029 shows pose starts too early or competes with audio start |
| `T0031` | Optimize JS `spel_retro_audio` while preserving exact output | Do if T0029 shows feature extraction/RF/recovery is the main bottleneck |
| `T0032` | Move heavy audio retro work to native/background execution if needed | Do only if T0031 cannot hit acceptable review-start speed |
| `T0033` | Resume one-reviewed-video model loop | Do when Love has a newly reviewed T0028/T0029+ clip ready for audit/retrain |

## Manual Verification

- Import a representative one-minute `Ljud + video ML` playing video on Motorola.
- Capture the timing table.
- Confirm the audio waveform appears with normal editable racket/table markers.
- Confirm marker counts are unchanged versus the same build path without instrumentation, or document the comparison blocker.

## Automated Validation

- `cd apps\collector && npx tsc --noEmit`
- `npm run validate`
- `git diff --check` for T0029 scoped files

## Completion Report Expected

Codex should report:

- Timing table for the tested clip
- Largest bottleneck
- Whether pose started before audio review was usable
- Confirmation that no model/training/cache behavior changed
- Recommended next ticket
