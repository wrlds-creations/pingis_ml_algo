# CODEX_TASK.md

Fill this in before asking Codex to implement project work. Use one active ticket per implementation pass, and keep the scope small enough to verify manually.

Quick read-only questions, repo exploration, and lightweight planning do not require a filled ticket. Code changes, infrastructure changes, dependency changes, and documentation updates that affect source-of-truth files should use a ticket.

## Ticket ID

`T0028`

## Branch

`codex/t0028-playing-retro-export-build-install-2026-06-04`

## Status

`Completed`

## Goal

Export the T0026 `spel_retro_audio` candidate selected by T0027 into the separate Collector app JSON, apply the T0027 review thresholds, validate parity/build, and install a release APK on Motorola.

This ticket promotes only the Review-only playing-retro audio path. It must not retrain, alter ordinary `studs_live`, replace `audio_model.json`, change `audio_contact_model.json`, or work on video models.

## Dependencies

- T0027 is completed and merged to `main`.
- T0027 selected `playing_retro_audio_rf_v2026_06_04_t0026_multi_window_context`.
- T0027 selected racket threshold `0.0`, table threshold `0.45`, and same-label dedupe `80 ms`.
- T0027 replay improved marker TP/wrong/FP/missed from `706/22/26/153` to `844/1/9/36`.
- T0026 local candidate artifacts exist under `data/audio/models/playing_retro_candidates/playing_retro_audio_rf_v2026_06_04_t0026_multi_window_context/`.
- Motorola `ZY22L6NDHV` should be connected by USB for install.

## Allowed Areas

- `apps/collector/src/models/playing_retro_audio_model.json`
- `apps/collector/src/playingRetroAudio.ts` only if needed for threshold metadata compatibility
- `skills/pingis-audio-classification/scripts/export_playing_retro_audio_model_json.py`
- `skills/pingis-audio-classification/scripts/validate_playing_retro_audio_app_export.py`
- `PROJECT_CONTEXT.md`
- `DECISIONS.md`
- `FOLLOWUPS.md`
- `ITERATION_LOG.md`
- `REPO_CURRENT_STATE.md`

## Do Not Touch

- `apps/collector/src/models/audio_model.json`
- `apps/collector/src/models/audio_contact_model.json`
- `studs_live` behavior or live detector thresholds
- Ordinary up/down bounce training/export paths
- Video-stroke model files
- Raw reviewed labels
- T0026 training artifacts
- T0027 replay artifacts
- Review UX except the already existing playing-retro model metadata consumption

## Requirements

- Update the playing-retro export script defaults to the T0026 model ID and T0027 thresholds.
- Update the playing-retro app export parity script to require the T0026 model ID and T0027 thresholds.
- Export `apps/collector/src/models/playing_retro_audio_model.json` from `playing_retro_audio_rf_v2026_06_04_t0026_multi_window_context`.
- Ensure exported metadata states app role `spel_retro_audio_review_only`, racket threshold `0.0`, table threshold `0.45`, same-label dedupe `80 ms`, and normal audio model unchanged.
- Validate exported app JSON against the local T0026 joblib model.
- Run Collector TypeScript validation.
- Run root validation.
- Force-regenerate the release Metro bundle.
- Build Android release APK.
- Verify the built bundle/APK contains the T0026 model/version and still contains `Ljud + video ML` and `Video FH/BH`.
- Install the release APK on connected Motorola and launch the app.
- Record APK SHA256, install time, and validation results in docs.

## Non-Goals

- No retraining.
- No new replay/tuning.
- No ordinary bounce or `studs_live` changes.
- No `audio_model.json` or `audio_contact_model.json` changes.
- No video-stroke changes.
- No Review UX redesign.
- No push/merge of T0028 unless Love asks after completion.

## Acceptance Criteria

- `playing_retro_audio_model.json` contains T0026 model metadata and T0027 thresholds.
- Export parity validation passes against the T0026 joblib artifacts.
- Collector TypeScript validation passes.
- Root validation passes.
- Release APK builds successfully.
- APK installs on Motorola and app launches.
- Docs record exact artifact IDs, SHA256, install time, and next recommended ticket.

## Completion Notes

- Updated `skills/pingis-audio-classification/scripts/export_playing_retro_audio_model_json.py` to default to T0026 model `playing_retro_audio_rf_v2026_06_04_t0026_multi_window_context`.
- Updated `skills/pingis-audio-classification/scripts/validate_playing_retro_audio_app_export.py` to require T0026 model metadata and T0027 thresholds.
- Exported `apps/collector/src/models/playing_retro_audio_model.json` from T0026.
- Exported metadata:
  - model version `playing_retro_audio_rf_v2026_06_04_t0026_multi_window_context`
  - app role `spel_retro_audio_review_only`
  - racket threshold `0.0`
  - table threshold `0.45`
  - same-label dedupe `80 ms`
  - source ticket `T0027`
  - `normal_audio_model_unchanged=true`
- Export size: 4,365 KB.
- Exported model shape: 197 features, 450 trees, 164,120 total nodes, labels `non_target`, `racket_contact`, `table_bounce`.
- Release bundle verification passed:
  - contained `playing_retro_audio_rf_v2026_06_04_t0026_multi_window_context`
  - contained `spel_retro_audio_review_only`
  - contained `Ljud + video ML`
  - contained `Video FH/BH`
  - did not contain `Ljudinsamling`
  - did not contain `Audio plus IMU`
- Built APK: `apps/collector/android/app/build/outputs/apk/release/app-release.apk`
- APK SHA256: `7AC97A6C4AA83A939941DD52E16F4C2C3627AD5AAA06666878F381290BB1D2AA`
- APK size: 164,665,426 bytes.
- Installed on Motorola `ZY22L6NDHV` via `adb install -r`.
- Install result: `Success`.
- App launch result: `pidof com.collectorapp` returned `17665`.
- Device package metadata: `versionCode=1`, `versionName=1.0`, `lastUpdateTime=2026-06-04 11:30:23`.
- No retraining, no `studs_live`, no ordinary bounce, no `audio_model.json`, no `audio_contact_model.json`, and no video model work was intentionally done in this ticket.

## Manual Verification

Love should open the installed app and review the next `Ljud + video ML` playing clip. Expected behavior:

- The app should look like the existing T0024/T0027 playing-retro flow.
- Fresh playing audio review should use the T0026 `spel_retro_audio` model.
- Racket/table markers should be normal editable review markers.
- Ordinary `Studsdetektor` behavior should be unchanged by this ticket.

## Automated Validation

- `python -m py_compile` for changed Python export/parity scripts.
- `python skills\pingis-audio-classification\scripts\export_playing_retro_audio_model_json.py`
- `python skills\pingis-audio-classification\scripts\validate_playing_retro_audio_app_export.py`
- `cd apps\collector && npx tsc --noEmit`
- `npm run validate`
- `cd apps\collector\android && .\gradlew.bat :app:createBundleReleaseJsAndAssets --rerun-tasks`
- `cd apps\collector\android && .\gradlew.bat assembleRelease`

Validation passed:

- `python -m py_compile skills\pingis-audio-classification\scripts\export_playing_retro_audio_model_json.py skills\pingis-audio-classification\scripts\validate_playing_retro_audio_app_export.py`
- `python skills\pingis-audio-classification\scripts\export_playing_retro_audio_model_json.py`
- `python skills\pingis-audio-classification\scripts\validate_playing_retro_audio_app_export.py`
- `cd apps\collector && npx tsc --noEmit`
- `npm run validate`
- `git diff --check` for T0028 scoped files
- `cd apps\collector\android && .\gradlew.bat :app:createBundleReleaseJsAndAssets --rerun-tasks`
- `cd apps\collector\android && .\gradlew.bat assembleRelease`
- Bundle string verification
- `adb devices`
- `adb install -r apps\collector\android\app\build\outputs\apk\release\app-release.apk`
- `adb shell monkey -p com.collectorapp -c android.intent.category.LAUNCHER 1`
- `adb shell pidof com.collectorapp`

## Completion Report Expected

Codex should report:

- Model ID exported
- Review thresholds exported
- APK path and SHA256
- Motorola install result
- Validation commands run
- Confirmation that ordinary `studs_live`, `audio_model.json`, `audio_contact_model.json`, and video model artifacts were not intentionally changed
