# Repo Current State

Use this file as the living snapshot of what actually exists in the repository. Update it after completed tickets, audits, meaningful dependency changes, validation changes, workflow changes, model/data/build state changes, or next-ticket changes.

## Snapshot

- Date: `2026-06-02`
- Current branch: `codex/t0011-playing-retro-audio-review-ui-apk`
- Current phase: `Audio-first playing-retro Review integration`
- Current status: `T0012 installed the Review-only spel-retro APK on Motorola; next work is generated retro candidate surfacing for missed markers`
- Current ticket: `T0013`
- Last completed ticket: `T0012`
- Recommended next ticket: `T0013-playing-retro-candidate-surfacing`

## Current Structure

```text
apps/collector/                         React Native Android collector/test app
data/audio/raw/                         local pulled audio session JSON/media, gitignored data source
data/audio/processed/                   local generated datasets and reports
data/audio/models/                      local trained model artifacts and evaluation reports
skills/pingis-audio-classification/     audio preprocess, replay, train, export workflow
skills/pingis-stroke-detection/         video stroke preprocess/train/export workflow
PROJECT_CONTEXT.md                      confirmed project facts and current model context
DECISIONS.md                            durable decision log
ITERATION_LOG.md                        detailed ML/data/build/device history
CODEX_TASK.md                           single active ticket
FOLLOWUPS.md                            out-of-scope issues and future tickets
```

## Known Validation Commands

| Command | Purpose | Last Known Result |
|---|---|---|
| `npm run validate` | Root WRLDS template, skill, and AWS metadata validation | `Passed 2026-06-02 after T0007 documentation update` |
| `cd apps/collector && npx tsc --noEmit` | Collector TypeScript validation | `Passed 2026-06-02 after T0011 playing-retro Review panel` |
| `python skills/pingis-audio-classification/scripts/build_playing_retro_candidate_report.py` | T0004 candidate-centered playing-retro audio report | `Passed 2026-06-02 on Tomas 05-28/05-29 sessions` |
| `python skills/pingis-audio-classification/scripts/train_playing_retro_audio.py` | T0005 local `spel_retro_audio` candidate train/eval | `Passed 2026-06-02; 4,028 rows, holdout accuracy 0.759` |
| `python skills/pingis-audio-classification/scripts/evaluate_playing_retro_audio_variants.py` | T0006 focused variant comparison | `Passed 2026-06-02; selected safe one-window candidate, holdout racket recall 0.623` |
| `python skills/pingis-audio-classification/scripts/evaluate_playing_retro_audio_multi_window.py` | T0007 multi-window/context variant comparison | `Passed 2026-06-02; selected multi-window/context candidate, holdout racket recall 0.896` |
| `python skills/pingis-audio-classification/scripts/evaluate_playing_retro_audio_cross_session.py` | T0008 cross-session validation | `Passed 2026-06-02; selected T0007 variant passes requested Tomas/Stiga holdouts` |
| `python skills/pingis-audio-classification/scripts/export_playing_retro_audio_model_json.py` | T0009 separate app JSON export for Review-only `spel_retro_audio` | `Passed 2026-06-02; exported 197-feature / 450-tree playing-retro model to apps/collector/src/models/playing_retro_audio_model.json` |
| `python skills/pingis-audio-classification/scripts/validate_playing_retro_audio_app_export.py` | T0009 export parity check against selected joblib model | `Passed 2026-06-02; feature order, labels, windows, and no-truth-field checks passed` |
| `python skills/pingis-audio-classification/scripts/replay_playing_retro_audio_app_export.py` | T0010 controlled replay of app export on saved Review candidates | `Passed 2026-06-02; 643 saved Tomas/Stiga candidates, acc 0.978, racket 0.987, table 0.988, non-target 0.948` |
| `cd apps/collector/android && ./gradlew assembleRelease` | T0012 clean release APK build from commit `52cfbb8` | `Passed 2026-06-02 in short worktree C:\st12; APK SHA256 3422CC2A34A0DAF31BDF03F89FF9CDE4BC2B0CDA7C972C0579D46B5B6C0D5A50` |
| `python -m py_compile <script>` | Targeted Python syntax check for changed scripts | `Run per ticket when Python changes` |

## Completed Tickets

| Ticket | Summary | Completed On | Notes |
|---|---|---|---|
| `T0001` | Adopt stricter WRLDS ticket workflow for this repo | `2026-06-02` | Added `REPO_CURRENT_STATE.md` and `FOLLOWUPS.md`, upgraded `CODEX_TASK.md`, and updated root workflow instructions |
| `T0002` | Refresh docs for audio/video-only scope | `2026-06-02` | Removed IMU/AirHive from active project context and marked sensor specs/workflows as legacy |
| `T0003` | Remove retired IMU/AirHive workflow files | `2026-06-02` | Deleted obsolete AirHive skills/specs and converted stroke skill to video-only workflow |
| `T0004` | Build candidate-centered `spel_retro_audio` report/dataset step | `2026-06-02` | Added deterministic report script and generated local row/summary outputs from saved app candidates plus replay peaks |
| `T0005` | Train and evaluate first local `spel_retro_audio` candidate | `2026-06-02` | Trained `playing_retro_audio_rf_v2026_06_02_app_candidates_100_200` from all matchable saved app candidates plus manually missed reviewed markers; no app export/APK |
| `T0006` | Improve `spel_retro_audio` through focused one-window variants | `2026-06-02` | Selected `playing_retro_audio_rf_v2026_06_02_safe_racket_weighted`; small safe gain only, no app export/APK |
| `T0007` | Improve `spel_retro_audio` with multi-window/context features | `2026-06-02` | Selected `playing_retro_audio_rf_v2026_06_02_multi_window_context`; holdout racket recall 0.896, table recall 0.933, non-target recall 0.833; no app export/APK |
| `T0008` | Cross-session validate T0007 `spel_retro_audio` candidate | `2026-06-02` | Selected T0007 variant passed requested Tomas/Stiga holdouts: racket recall 0.910 / 0.939 / 0.896 and no table/non-target regression versus T0006; no app export/APK |
| `T0009` | Export and stage separate Review-only `spel_retro_audio` app path | `2026-06-02` | Added `playing_retro_audio_model.json`, app helper `playingRetroAudio.ts`, export/parity scripts, and validation; normal `audio_model.json`, `audio_contact_model.json`, `studs_live`, and APK artifacts unchanged |
| `T0010` | Replay separate app export on saved Tomas/Stiga Review candidates | `2026-06-02` | Added deterministic replay script using `playing_retro_audio_model.json`: 643 saved candidates scored 0.978 accuracy, but 15 reviewed missed markers remain unclassifiable without candidate surfacing/generation |
| `T0011` | Add visible manual `spel_retro_audio` Review panel | `2026-06-02` | Review now has a separate `Spel-retro audio` button for playing-mode reviews. It reclassifies current saved candidates, shows racket/table/non-target counts, and draws separate retro pins without changing markers, save truth, `studs_live`, `audio_model.json`, or APK artifacts |
| `T0012` | Build and install Review-only `spel_retro_audio` APK | `2026-06-02` | Built clean release APK from commit `52cfbb8` in `C:\st12`, verified bundle strings `Spel-retro audio` and `spel_retro_audio_review_only`, installed on Motorola `ZY22L6NDHV`, app launched, APK SHA256 `3422CC2A34A0DAF31BDF03F89FF9CDE4BC2B0CDA7C972C0579D46B5B6C0D5A50` |

## Current Ticket

| Ticket | Goal | Status | Notes |
|---|---|---|---|
| `T0013` | Add playing-retro candidate generation/surfacing for missed markers | `Ready` | T0012 installed the saved-candidate reclassification panel. Next step is to recover the 15 known missed Tomas/Stiga markers without flooding false positives |

## Confirmed Next Tickets

| Ticket | Goal | Notes |
|---|---|---|
| `T0013` | Add playing-retro candidate generation/surfacing for missed markers | Separate from reclassification; target the 15 missed Tomas/Stiga markers without flooding false positives |
| `T0014` | Revisit video-assisted FH/BH fusion after audio retro is stable | Keep video paused until audio is useful |

## Dependencies

| Dependency | Purpose | Notes |
|---|---|---|
| `apps/collector` | Android app and bundled model runtime | Do not change app artifacts unless ticket allows it |
| `skills/pingis-audio-classification` | Audio ML/replay workflow | Primary area for T0004-T0008 |
| `data/audio/raw` | Local reviewed session source data | Keep out of git; do not alter labels without approval |
| `data/audio/processed` | Generated datasets/reports | Local artifacts may be regenerated |

## Validation Status

- Build: `T0012 release APK built from clean commit 52cfbb8 in C:\st12; installed on Motorola ZY22L6NDHV at 2026-06-02 15:27:29`
- Tests: `cd apps/collector && npx tsc --noEmit`, root `npm run validate`, forced `:app:createBundleReleaseJsAndAssets --rerun-tasks`, Gradle `assembleRelease`, APK bundle string verification, `adb install -r`, `adb shell monkey -p com.collectorapp 1`, and `adb shell pidof com.collectorapp` passed on 2026-06-02
- Lint: `git diff --check` passed for T0012 on 2026-06-02
- Manual verification: `App is installed and launched; Love still needs to open a playing-mode Review, press Spel-retro audio / Kör retro, and verify counts/pins do not mutate Save truth.`

## Known Issues Summary

- Current working tree already contains many pre-existing app/model/video changes from earlier work; ticket work must avoid staging or reverting unrelated files.
- `studs_live`, `spel_retro_audio`, and `video_stroke_retro` need separate ticket scopes to avoid model/config bleed.
- IMU/AirHive workflow docs and skill scripts were removed from active scope; remaining app labels/code paths should be treated as legacy unless a ticket explicitly removes or renames them.
- `spel_retro_audio` now has a separate app JSON export, opt-in helper, deterministic replay script, visible manual Review panel, and installed Review-only APK, but it still only reclassifies candidates that already exist.
- T0005 trains from all matchable saved app candidates plus manually missed reviewed markers; replay-generated T0004 candidates remain diagnostic, not training rows.
- Ordinary up/down bounce regression for T0007 is advisory only because the old ordinary rows do not preserve exact multi-window event timestamps; do not use it as promotion evidence for `studs_live`.

## Open Questions

- Should future ticket branches be created from current dirty `codex/video-stroke-test` or from a cleaned approved base?
- Should T0013 generate additional peak candidates in app Review, in a Python replay first, or both before the next APK?
- Should legacy IMU app surfaces be removed immediately or left as hidden/internal historical code until the next UI cleanup?
