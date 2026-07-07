# CODEX_TASK.md

Fill this in before asking Codex to implement project work. Use one active ticket per implementation pass, and keep the scope small enough to verify manually.

Quick read-only questions, repo exploration, and lightweight planning do not require a filled ticket. Code changes, infrastructure changes, dependency changes, and documentation updates that affect source-of-truth files should use a ticket.

## Ticket ID

`T0139-bounce-audio-edge-impulse-v4-test`

## Branch

`codex/t0057-fable-auto-improvement-loop`

## Status

`Completed`

## Goal

Add the downloaded Edge Impulse `pingpong-cpp-mcu-v4-impulse-#1.zip` model as a guarded Android-only comparison option inside `Bounce audio test`, so Love can test whether the external MFE/TFLite Micro model detects racket bounces better than the current local candidates.

## Dependencies

- Edge Impulse export exists locally at `C:\Development\Wrlds\edgeImpulse\pingpong-cpp-mcu-v4-impulse-#1.zip`.
- The zip is a generated C++ MCU SDK/model export, not a small JSON model. Generated SDK contents should not be committed.
- `Bounce audio test` already supports selectable diagnostic audio runtimes and debug JSON/WAV output.

## Allowed Areas

- `CODEX_TASK.md`
- `PROJECT_CONTEXT.md`
- `REPO_CURRENT_STATE.md`
- `ITERATION_LOG.md`
- `DECISIONS.md`
- `.gitignore`
- `apps/collector/src/NativeAudioStream.ts`
- `apps/collector/src/bounceAudioTestEngine.ts`
- `apps/collector/src/BounceAudioTestScreen.tsx`
- `apps/collector/android/app/build.gradle`
- `apps/collector/android/app/src/main/java/com/collectorapp/AudioStreamModule.kt`
- `apps/collector/android/app/src/main/java/com/collectorapp/EdgeImpulsePingpongBridge.kt`
- `apps/collector/android/app/src/main/cpp/CMakeLists.txt`
- `apps/collector/android/app/src/main/cpp/edge_impulse_pingpong_jni.cpp`

## Do Not Touch

- Do not retrain or replace any model JSON.
- Do not commit the generated Edge Impulse SDK/export folder.
- Do not promote Edge Impulse as the default production counter.
- Do not change `Studs FH/BH LIVE` entries.
- Do not change STIGA app code.
- Do not delete raw/generated `data/` or `raw/`.
- Do not merge to `main`.
- Do not use cloud APIs or AWS.

## Requirements

- Add a new `Bounce audio test` model option for Edge Impulse v4.
- Extract the Edge Impulse SDK locally into a gitignored folder for this machine only.
- Add a native Android wrapper that can compile with or without the local SDK folder.
- When the SDK is present, run the Edge Impulse classifier on a 16 kHz / 8000-sample centered window around each native peak candidate.
- Feed the native Edge Impulse probabilities into the existing `Bounce audio test` threshold/dedupe/debug flow.
- Save Edge Impulse probabilities, label, timing, threshold, and availability/error fields in debug JSON.
- Keep existing `T0103`, `T0104E`, `T0104E Soft`, and `RMS+Fable` options unchanged.

## Non-Goals

- No production promotion.
- No STIGA/iOS port.
- No new model training or Edge Impulse project changes.
- No commit of the generated SDK.
- No full video recording.

## Acceptance Criteria

- `Edge Impulse v4` appears in `Bounce audio test`.
- Existing Bounce audio test model options still work.
- With the local SDK present, Android emits Edge Impulse `Bounce`/`noise` probabilities for peak candidates.
- With the local SDK missing, the app still builds and the Edge Impulse option reports unavailable rather than breaking the app.
- Debug JSON contains Edge Impulse candidate metadata.
- TypeScript validates.
- Android/Kotlin/CMake validates.
- Docs record the diagnostic Edge Impulse integration and local-SDK caveat.

## Completion Notes

- Added a guarded `Edge Impulse v4` option to `Bounce audio test`.
- The generated Edge Impulse C++ SDK was extracted locally under the gitignored `apps/collector/android/app/src/main/cpp/edge_impulse_local/` folder and is not part of the repo.
- Added a small Android JNI/Kotlin bridge that builds in full mode when the local SDK exists and as a stub when it is missing.
- The local Edge Impulse `.so` is packaged for this machine from gitignored `apps/collector/android/app/src/main/jniLibs/**/libedge_impulse_pingpong.so`, so the app does not replace React Native's own native module CMake packaging.
- The Edge Impulse option reuses the existing `Bounce audio test` peak candidate/debug/session flow, but classifies a centered `16 kHz / 8000 sample` window with the Edge Impulse model and uses the `Bounce` probability as the typed `p` threshold.
- Existing `T0103`, `T0104E`, `T0104E Soft`, and `RMS+Fable` options remain unchanged.

## Validation

- `cd apps/collector && npx tsc --noEmit` passed.
- `cd apps/collector/android && .\gradlew.bat :app:compileDebugKotlin --no-daemon --console plain` passed.
- Direct Android CMake arm64 build of `edge_impulse_pingpong` passed with the local Edge Impulse SDK present.
- `npm run validate` passed.
- `git diff --check` passed with only existing Windows LF/CRLF warnings.
- `.\install-android-dev.ps1` passed and installed/launched `com.collectorapp` on connected Motorola `ZY22KSPF5W`; package `lastUpdateTime=2026-07-07 14:31:52`.
- Installed debug APK `C:\pcr\android\app\build\outputs\apk\debug\app-debug.apk` contains both `lib/arm64-v8a/libappmodules.so` and `lib/arm64-v8a/libedge_impulse_pingpong.so`.
- Clean relaunch after `adb logcat -c` loaded `libappmodules.so` and logged `Running "CollectorApp"` with no `PlatformConstants`, `runtime not ready`, or script-load crash.
