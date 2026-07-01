# CODEX_TASK.md

Fill this in before asking Codex to implement project work. Use one active ticket per implementation pass, and keep the scope small enough to verify manually.

Quick read-only questions, repo exploration, and lightweight planning do not require a filled ticket. Code changes, infrastructure changes, dependency changes, and documentation updates that affect source-of-truth files should use a ticket.

## Ticket ID

`T0111-colleague-test-release-t0104e-defaults`

## Branch

`codex/t0057-fable-auto-improvement-loop`

## Status

`Complete`

## Goal

Prepare a colleague-test release build by making `Bounce audio test` open on T0104E with default `p=0.25` and Fable noise veto `0.98`, then build/install a release APK on the connected phone.

## Dependencies

- T0109 evaluated T0104E `p=0.25`, noise veto `0.98` as a guarded phone-test setting.
- T0110 confirmed loud music without bounce false-counts at this setting, so this remains a colleague-test diagnostic build rather than a production merge.
- Love explicitly asked to use `0.25` and `0.98` as defaults and install a release version to the connected phone.
- Raw/generated `data/` remains ignored and must not be committed.

## Allowed Areas

- `CODEX_TASK.md`
- `PROJECT_CONTEXT.md`
- `DECISIONS.md`
- `REPO_CURRENT_STATE.md`
- `ITERATION_LOG.md`
- `apps/collector/src/bounceAudioTestEngine.ts`
- validation/status commands

## Do Not Touch

- Do not merge to `main`.
- Do not push.
- Do not delete local or device data.
- Do not revert tracked or user changes.
- Do not replace or promote production Fable/studs/camera behavior.
- Do not change `audio_model.json`, `audio_contact_model.json`, `fable_audio_model.json`, `bounce_side_model.json`, T0103/T0104E JSON, or native peak-gate defaults.
- Do not move raw/generated data into git.

## Requirements

- Set `Bounce audio test` default selected model to `T0104E`.
- Set T0104E's default typed runtime config to threshold `0.25` and Fable noise veto `0.98`.
- Keep T0103 and RMS+Fable available in the selector.
- Keep this as a diagnostic colleague-test build, not a `main` merge/promotion.
- Build and install a release APK on the connected Motorola.

## Non-Goals

- No model export/retrain.
- No camera/racket-side changes.
- No new data pull or labeling.
- No push or main merge.

## Acceptance Criteria

- Collector TypeScript validation passes.
- Root validation passes.
- `git diff --check` passes.
- Release build/install/launch succeeds on the connected phone.
- Final answer explains that this is installed for colleague testing but not merged to `main`.

## Completion Notes

- `Bounce audio test` now opens with `T0104E` selected.
- T0104E's default typed config is threshold `0.25` and Fable noise veto `0.98`.
- T0103 and `RMS+Fable` remain available in the same selector.
- This is a colleague-test diagnostic release, not a production promotion or `main` merge.
- Built, installed, and launched a release APK on Motorola `ZY22KSPF5W`.
- Release APK: `C:\pcr\android\app\build\outputs\apk\release\app-release.apk`.
- Release APK SHA256: `F7C9D1514D2E408ACFD7391070D68DB43EC2E5646DFAF59896AB6D0B66979F79`.
- Package smoke: `com.collectorapp`, `versionName=1.0`, `versionCode=1`, `lastUpdateTime=2026-07-01 20:25:36`, PID `32126`.

## Validation

- `cd apps/collector && npx tsc --noEmit`
- `npm run validate`
- `git diff --check`
- `.\build-android-local.ps1 -Install -Launch`
- release bundle string smoke for `T0104E candidate`, `colleague test default`, and `.98`.
