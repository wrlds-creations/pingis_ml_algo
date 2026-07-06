# CODEX_TASK.md

Fill this in before asking Codex to implement project work. Use one active ticket per implementation pass, and keep the scope small enough to verify manually.

Quick read-only questions, repo exploration, and lightweight planning do not require a filled ticket. Code changes, infrastructure changes, dependency changes, and documentation updates that affect source-of-truth files should use a ticket.

## Ticket ID

`T0138-fhbh-live-v6-post-session-side-processing`

## Branch

`codex/t0057-fable-auto-improvement-loop`

## Status

`Completed`

## Goal

Add `Studs FH/BH LIVE v6` as a guarded comparison screen focused on more accurate FH/BH detection after a run. V6 should collect accepted audio bounces and timestamped camera crop evidence while the user bounces, then process FH/BH/OSAKER after the user presses STOP instead of trying to show side results live.

## Dependencies

- T0137 added `Studs FH/BH LIVE v5`, which queues wrist-crop side jobs so crop/model latency does not block the audio listener.
- V5 already uses the native camera frame buffer and `captureCrop(targetTimeMs)` to select the frame nearest an audio timestamp.
- Love reported one-side bouncing works better than alternating; alternating needs better timestamp/crop evidence and can trade realtime feedback for final accuracy.

## Allowed Areas

- `CODEX_TASK.md`
- `PROJECT_CONTEXT.md`
- `REPO_CURRENT_STATE.md`
- `ITERATION_LOG.md`
- `DECISIONS.md`
- `apps/collector/App.tsx`
- `apps/collector/src/SetupScreen.tsx`
- `apps/collector/src/BounceSideLiveScreen.tsx`
- `apps/collector/android/app/src/main/java/com/collectorapp/BounceSideLiveModule.kt`

## Do Not Touch

- Do not retrain or replace any model JSON.
- Do not change `Bounce audio test`.
- Do not change STIGA app code.
- Do not delete raw/generated `data/` or `raw/`.
- Do not merge to `main`.
- Do not use cloud APIs or AWS.

## Requirements

- Add a new setup entry and route named `Studs FH/BH LIVE v6`.
- Base v6 on v5's Hybrid audio trigger and wrist-crop side model.
- During a run, accepted audio bounces should be collected as audio events, not immediately counted as FH/BH.
- During a run, capture a small timestamp window of wrist-crop evidence around each accepted audio timestamp before the native frame buffer rolls past it.
- After STOP, select the best crop per bounce, run/resolve the existing wrist-crop side model, and show final FH/BH/OSAKER counts.
- Improve timestamp debug: store selected crop delay relative to the audio timestamp and the per-crop candidate delays.
- Keep v1/v2/v3/v4/v5 behavior unchanged.

## Non-Goals

- No production promotion.
- No STIGA/iOS port.
- No new model training or threshold tuning.
- No native-side side model rewrite.
- No full video recording.

## Acceptance Criteria

- `Studs FH/BH LIVE v6` appears in setup and opens.
- While running, V6 collects accepted audio bounces and crop evidence without requiring immediate FH/BH classification.
- After STOP, V6 shows final FH/BH/OSAKER counts.
- Debug JSON contains audio events, selected crop metadata, and crop timing candidates.
- TypeScript validates.
- Android/Kotlin validates if native code changes are made.
- Docs record the new post-session comparison flow.

## Completion Notes

- Added `Studs FH/BH LIVE v6` setup card and route.
- V6 uses `audioTriggerMode="hybrid"` and `sideDecisionMode="wrist_crop_post"`.
- Accepted audio bounces are collected during the run without immediate FH/BH side counting.
- For each accepted audio bounce, V6 captures five timestamped wrist-crop candidates around the audio timestamp before the native frame buffer rolls past it.
- After STOP, V6 waits for crop evidence, scores candidates by confidence, timing, ROI source, and decision source, then shows final FH/BH/OSAKER counts.
- Debug JSON stores selected crop timing plus all crop candidate timing/score metadata for each bounce.
- V1/V2/V3/V4/V5, `Bounce audio test`, model JSONs, STIGA code, raw/generated data, release APK, push, merge, and `main` promotion remain unchanged.
- Installed and launched the debug/Metro Collector app on connected Motorola `ZY22KSPF5W`.

## Validation

- `cd apps/collector && npx tsc --noEmit`
- `cd apps/collector/android && .\gradlew.bat :app:compileDebugKotlin --no-daemon --console plain`
- `npm run validate`
- `git diff --check` (passed with existing Windows LF-to-CRLF warnings only)
- `.\install-android-dev.ps1`
- `adb devices` -> `ZY22KSPF5W device`
- `adb shell pidof com.collectorapp` -> `20010`
- `adb shell dumpsys package com.collectorapp` -> `lastUpdateTime=2026-07-06 22:26:40`
