# CODEX_TASK.md

Fill this in before asking Codex to implement project work. Use one active ticket per implementation pass, and keep the scope small enough to verify manually.

Quick read-only questions, repo exploration, and lightweight planning do not require a filled ticket. Code changes, infrastructure changes, dependency changes, and documentation updates that affect source-of-truth files should use a ticket.

## Ticket ID

`T0113-android-java-gradle-toolchain-fix`

## Branch

`codex/t0057-fable-auto-improvement-loop`

## Status

`Complete`

## Goal

Make Android release builds work permanently on this Windows machine by moving Gradle off Oracle Java 8 and fixing the React Native Gradle plugin's Foojay resolver version for Gradle 9.

## Dependencies

- T0112 fixed the `Bounce audio test` storage fallback but a new release APK could not be built locally.
- Current default `java` resolves to Oracle Java 8 via `C:\Program Files (x86)\Common Files\Oracle\Java\javapath\java.exe`.
- Gradle wrapper is `9.0.0`, which requires Java 17+ to run.
- Running with Android Studio JBR 21 then exposed React Native Gradle plugin `foojay-resolver-convention` `0.5.0`, which references `JvmVendorSpec.IBM_SEMERU` removed in Gradle 9.
- Upstream Gradle/Foojay guidance and the React Native issue indicate `foojay-resolver-convention` `1.0.0` is the Gradle 9-compatible fix.

## Allowed Areas

- `CODEX_TASK.md`
- `PROJECT_CONTEXT.md`
- `DECISIONS.md`
- `REPO_CURRENT_STATE.md`
- `ITERATION_LOG.md`
- `apps/collector/patches/`
- `apps/collector/node_modules/@react-native/gradle-plugin/settings.gradle.kts` as the source for a `patch-package` patch
- local user Java/Gradle environment configuration commands
- Android release build validation/status commands

## Do Not Touch

- Do not merge to `main`.
- Do not push.
- Do not delete local or device data.
- Do not revert tracked or user changes.
- Do not uninstall Oracle Java 8 unless a later explicit approval says legacy Java can be removed.
- Do not replace or promote production Fable/studs/camera behavior.
- Do not change `audio_model.json`, `audio_contact_model.json`, `fable_audio_model.json`, `bounce_side_model.json`, T0103/T0104E JSON, or native peak-gate defaults.
- Do not move raw/generated data into git.

## Requirements

- Patch `@react-native/gradle-plugin` so its included build uses `org.gradle.toolchains.foojay-resolver-convention` `1.0.0` instead of `0.5.0`.
- Generate a `patch-package` patch so the fix survives `npm install`.
- Set this Windows user/machine build environment so Gradle wrapper uses a Java 17+ runtime instead of Oracle Java 8.
- Prefer using an existing modern JDK/JBR over uninstalling old Java.
- Re-run Android release build after the toolchain fix.
- Keep T0112 app behavior unchanged.

## Non-Goals

- No model export/retrain.
- No app runtime behavior change beyond already-completed T0112.
- No threshold, dedupe, or native gate behavior changes.
- No camera/racket-side changes.
- No new data pull or labeling.
- No push or main merge.

## Acceptance Criteria

- `gradlew.bat assembleRelease` no longer fails on Java 8 or `JvmVendorSpec IBM_SEMERU`.
- `patch-package` can reapply the React Native Gradle plugin patch.
- Collector TypeScript validation passes.
- Root validation passes.
- `git diff --check` passes.
- Final answer explains whether Java 8 was left installed and which Java Gradle now uses.

## Completion Notes

- Left Oracle Java 8 installed for legacy compatibility, but moved this user's Android build environment off it.
- Set user-level `JAVA_HOME` to `D:\Programs_Installed\Android\Android Studio\jbr`.
- Set user-level `ANDROID_HOME` and `ANDROID_SDK_ROOT` to `D:\Programs_Installed\Android\Sdk`.
- Patched `@react-native/gradle-plugin` via `patch-package` so its included build uses `org.gradle.toolchains.foojay-resolver-convention` `1.0.0` instead of `0.5.0`.
- Verified `patch-package` reapplies the new React Native Gradle plugin patch after install in the short build copy.
- Direct long-path release build now gets past Java 8, Foojay, and Android SDK configuration, but still fails in native Nitro/CMake path handling.
- A junction path (`C:\pma`) does not solve the CMake path issue because CMake resolves the real long target path.
- Created and used a real short physical build copy at `D:\pcr`.
- Built an arm64 release APK from `D:\pcr\apps\collector\android`.
- Installed and launched the release APK on connected Android `VOG_L29` (`EHT0219B01004275`).
- Release APK: `D:\pcr\apps\collector\android\app\build\outputs\apk\release\app-release.apk`.
- Release APK SHA256: `2304819D60FA1D3382D66486E78206D0CD6396779D6DDC1A3D2A67DF93B98231`.
- Installed package smoke: `com.collectorapp`, `versionName=1.0`, `versionCode=1`, `lastUpdateTime=2026-07-02 10:49:41`, PID `16598`, resumed `MainActivity`.
- No model JSON, native gate, production Fable/studs/camera behavior, raw data, push, or `main` merge changed.

## Validation

- `npx patch-package` in `D:\pcr\apps\collector`
- `cd D:\pcr\apps\collector\android && .\gradlew.bat clean --no-daemon --console plain`
- `cd D:\pcr\apps\collector\android && .\gradlew.bat assembleRelease -PreactNativeArchitectures=arm64-v8a --no-daemon --console plain`
- `adb install -r D:\pcr\apps\collector\android\app\build\outputs\apk\release\app-release.apk`
- `adb shell am start -n com.collectorapp/.MainActivity`
- `adb shell pidof com.collectorapp`
- `adb shell dumpsys package com.collectorapp`
- `adb shell dumpsys activity activities`
- Final source-tree `cd apps/collector && npx tsc --noEmit`: passed 2026-07-02.
- Final source-tree `npm run validate`: passed 2026-07-02.
- Final source-tree `git diff --check`: passed 2026-07-02 with Windows LF-to-CRLF warnings only.
