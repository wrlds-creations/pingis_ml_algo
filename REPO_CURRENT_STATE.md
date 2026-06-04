# Repo Current State

Use this file as the living snapshot of what actually exists in the repository. Update it after completed tickets, audits, meaningful dependency changes, validation changes, workflow changes, model/data/build state changes, or next-ticket changes.

## Snapshot

- Date: `2026-06-04`
- Current branch: `codex/t0026-retrain-playing-retro-2026-06-04`
- Current phase: `Per-video playing-retro audio release loop`
- Current status: `T0026 retrain of spel_retro_audio with audio_session_2026-06-04_001 completed`
- Current ticket: `T0026`
- Last completed ticket: `T0026`
- Recommended next ticket: `T0027-replay-tune-playing-retro-2026-06-04`

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
| `npm run validate` | Root WRLDS template, skill, and AWS metadata validation | `Passed 2026-06-04 for T0025` |
| `cd apps/collector && npx tsc --noEmit` | Collector TypeScript validation | `Passed 2026-06-04 with T0024 retrained playing-retro export` |
| `python skills/pingis-audio-classification/scripts/build_playing_retro_candidate_report.py` | T0004 candidate-centered playing-retro audio report | `Passed 2026-06-02 on Tomas 05-28/05-29 sessions` |
| `python skills/pingis-audio-classification/scripts/train_playing_retro_audio.py` | T0005 local `spel_retro_audio` candidate train/eval | `Passed 2026-06-02; 4,028 rows, holdout accuracy 0.759` |
| `python skills/pingis-audio-classification/scripts/evaluate_playing_retro_audio_variants.py` | T0006 focused variant comparison | `Passed 2026-06-02; selected safe one-window candidate, holdout racket recall 0.623` |
| `python skills/pingis-audio-classification/scripts/evaluate_playing_retro_audio_multi_window.py` | T0007 multi-window/context variant comparison | `Passed 2026-06-02; selected multi-window/context candidate, holdout racket recall 0.896` |
| `python skills/pingis-audio-classification/scripts/evaluate_playing_retro_audio_cross_session.py` | T0008 cross-session validation | `Passed 2026-06-02; selected T0007 variant passes requested Tomas/Stiga holdouts` |
| `python skills/pingis-audio-classification/scripts/export_playing_retro_audio_model_json.py` | Separate app JSON export for Review-only `spel_retro_audio` | `Passed 2026-06-04; exported T0022 197-feature / 450-tree playing-retro model to apps/collector/src/models/playing_retro_audio_model.json` |
| `python skills/pingis-audio-classification/scripts/validate_playing_retro_audio_app_export.py` | Export parity check against selected joblib model | `Passed 2026-06-04; feature order, labels, windows, model version, and review thresholds passed` |
| `python skills/pingis-audio-classification/scripts/replay_playing_retro_audio_app_export.py` | T0010 controlled replay of app export on saved Review candidates | `Passed 2026-06-02; 643 saved Tomas/Stiga candidates, acc 0.978, racket 0.987, table 0.988, non-target 0.948` |
| `python skills/pingis-audio-classification/scripts/replay_playing_retro_recovery_candidates.py` | T0014 conservative dense recovery replay on Tomas/Stiga target sessions | `Passed 2026-06-03; baseline missed 15, visible recovery 6, recovered 6, wrong class 0, duplicate 0, visible FP 0` |
| `python skills/pingis-audio-classification/scripts/tune_playing_retro_recovery_thresholds.py` | T0015 threshold/gate sweep for T0016 APK readiness | `Passed 2026-06-03; swept 1,764 gates, selected current T0014 gate, recovered 6, wrong class 0, duplicate 0, visible FP 0` |
| `python skills/pingis-audio-classification/scripts/audit_playing_retro_review_session.py` | T0020 per-video playing-retro review audit | `Passed 2026-06-03 on audio_session_2026-06-03_005; T0020 dedupe reduces FP 11 -> 4 with TP unchanged at 186` |
| `python skills/pingis-audio-classification/scripts/analyze_playing_retro_misses.py` | T0021 row-level miss/correction analysis before retrain | `Passed 2026-06-03 on audio_session_2026-06-03_005; 13/20 manual additions were nearby candidates classified non_target` |
| `python skills/pingis-audio-classification/scripts/train_playing_retro_audio_t0022.py` | T0022 retrain local `spel_retro_audio` candidate with 2026-06-03 data | `Passed 2026-06-03; trained playing_retro_audio_rf_v2026_06_03_t0022_multi_window_context from 4,324 rows across 17 sessions` |
| `python skills/pingis-audio-classification/scripts/replay_playing_retro_audio_t0023.py` | T0023 replay/tune T0022 candidate versus T0020 baseline | `Passed 2026-06-03; selected racket threshold 0.0, table threshold 0.5, marker replay TP/wrong/FP/missed 576/22/24/118 -> 687/1/8/28` |
| `python skills/pingis-audio-classification/scripts/train_playing_retro_audio_t0026.py` | T0026 retrain local `spel_retro_audio` candidate with 2026-06-04 data | `Passed 2026-06-04; trained playing_retro_audio_rf_v2026_06_04_t0026_multi_window_context from 4,598 rows across 18 sessions` |
| `cd apps/collector/android && .\gradlew.bat :app:createBundleReleaseJsAndAssets --rerun-tasks` | Forced React Native release bundle regeneration | `Passed 2026-06-04 for T0024` |
| `cd apps/collector/android && .\gradlew.bat assembleRelease` | Android release APK build | `Passed 2026-06-04 for T0024; APK SHA256 5C03823425158A88A22A1CD4433C1B1ADE179B58E9B1E6634C966BCDC2C9DF61` |
| `python skills/pingis-audio-classification/scripts/audit_playing_retro_review_session_t0025.py` | T0025 audit of first T0024-reviewed 2026-06-04 playing session | `Passed 2026-06-04; 165 markers, 35 manual additions, manual buckets 21 non_target / 7 wrong-class / 6 table-threshold / 1 timing, 0 material candidate-generation gaps` |
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
| `T0012B` | Rebuild/install Review-only APK from current working app base | `2026-06-02` | Reinstalled from the current app working tree so `Ljud + video ML` and `Video FH/BH` were present; APK SHA256 `26041BDF11D3B48F8DF5163122AB38D56351B40F51670BD57F5DAEB30A6A9843` |
| `T0012C` | Remove legacy audio-only and Audio-plus-IMU entry points from installed APK | `2026-06-02` | Removed `Ljudinsamling` and `Audio plus IMU` setup/router paths and old user-facing labels; installed on Motorola at `2026-06-02 16:17:24`; APK SHA256 `95B7A08CE1B8325721A9F1DC6A9D829024FB91FF685C84873F17E6E30A380D95` |
| `T0013` | Make playing-retro primary review on existing saved candidates | `2026-06-03` | `Ljud + video ML` audio review auto-runs `spel_retro_audio`, turns retro racket/table predictions into normal editable pending review markers, keeps non-target candidates as analysis data, hides the manual `Kor retro` comparison panel for primary playing review, and leaves `Studsdetektor`/model JSON/APK unchanged |
| `T0014` | Add conservative playing-retro dense peak recovery | `2026-06-03` | Recovery scans the loaded WAV with a 28 ms recovery peak gap, classifies recovery anchors in the same `spel_retro_audio` context as saved candidates, and surfaces only conservative class/gap-filtered target suggestions. Replay recovered 6/15 missed truths, all table bounces, with 0 wrong-class, 0 duplicate-near-baseline, and 0 visible FP. No model retrain/APK/live change |
| `T0015` | Replay and tune playing-retro recovery gates | `2026-06-03` | Swept 1,764 racket/table confidence and nearest-saved-gap gates using T0014 recovery predictions. Selected the unchanged T0014 gate: racket >=0.80 with >=120 ms saved gap, table >=0.54 with >=60 ms saved gap. It recovers 6/15 missed truths with 0 wrong-class, 0 duplicate-near-baseline, and 0 visible FP, so T0016 APK build/install may proceed |
| `T0016` | Build and install playing-retro Review APK | `2026-06-03` | Built release APK with forced Metro bundle, verified expected playing-retro/current-app strings, installed on Motorola `ZY22L6NDHV`, launched app, and recorded APK SHA256 `C5BECE5093DDD109F8F300B6E41271E9E0F9222FA591772CCDCA5472C3363069` |
| `T0017` | Fix playing-retro marker/count wording and install APK | `2026-06-03` | Primary `spel_retro_audio` now leads the panel/status with editable `Review-markers`/`markers att granska`, keeps raw target/recovery counts diagnostic, preserves T0015 recovery gates, and installed APK SHA256 `A61496761AC76AF10DD8AC2A4E876D3AD3A8EFA1D35CBBCBCCF8BDCD6E52FE2A` on Motorola `ZY22L6NDHV` |
| `T0018` | Make playing-retro the actual primary marker source | `2026-06-03` | Primary `Ljud + video ML` playing review now discards old normal auto-markers when there is no saved human review, keeps their peak candidates only as retro input, blocks save while retro markers are pending, and installed APK SHA256 `A3D1D0AEFA507AFFCA3F02E08A7613FE89BD636A38C0CD07BCA0AADA2601E22D` on Motorola `ZY22L6NDHV` |
| `T0019` | Smooth playing-retro loading UX | `2026-06-03` | Fresh `Ljud + video ML` playing imports now stay on one `Analyserar spel-retro audio` preparation state until editable retro markers are ready; existing human reviews still open directly. Installed APK SHA256 `FA4081734AA1FFCAA76A81E124E27B1AA532349CEFE19D72019991B977377D9F` on Motorola `ZY22L6NDHV` |
| `T0020` | First per-video playing-retro audit/improve/install loop | `2026-06-03` | Pulled `audio_session_2026-06-03_005`, audited 212 reviewed markers and 204 review-relevant retro target candidates, added 80 ms same-label-only duplicate suppression plus blue-outline explanation, and installed APK SHA256 `12D2179D4EA260DF1AB0A556C857BF77B5795BB627FD0D989E023B19DDFCD1A6` on Motorola `ZY22L6NDHV` |
| `T0021` | Analyze T0020 misses before retrain | `2026-06-03` | Added `analyze_playing_retro_misses.py` and generated local T0021 reports for `audio_session_2026-06-03_005`; the main pattern is 13/20 manual additions near candidates classified as `non_target`, so T0022 should prioritize retraining the separate `spel_retro_audio` model |
| `T0022` | Retrain `spel_retro_audio` with 2026-06-03 data | `2026-06-03` | Added `train_playing_retro_audio_t0022.py` and trained local candidate `playing_retro_audio_rf_v2026_06_03_t0022_multi_window_context` from 4,324 rows across 17 playing sessions; no app JSON/APK/`studs_live` change |
| `T0023` | Replay/tune retrained `spel_retro_audio` model | `2026-06-03` | Added `replay_playing_retro_audio_t0023.py`; selected T0022 model with racket threshold `0.0`, table threshold `0.5`, and 80 ms same-label dedupe after replay improved marker TP/wrong/FP/missed from `576/22/24/118` to `687/1/8/28`; no app JSON/APK/`studs_live` change |
| `T0024` | Export, build, and install retrained playing-retro APK | `2026-06-04` | Exported T0022 `playing_retro_audio_rf_v2026_06_03_t0022_multi_window_context` into `playing_retro_audio_model.json`, applied racket threshold `0.0` and table threshold `0.5` in app runtime, built release APK, and installed SHA256 `5C03823425158A88A22A1CD4433C1B1ADE179B58E9B1E6634C966BCDC2C9DF61` on Motorola `ZY22L6NDHV` |
| `T0025` | Audit first T0024-reviewed 2026-06-04 playing session | `2026-06-04` | Pulled `audio_session_2026-06-04_001` and generated T0025 audit reports. Final truth is 165 markers (80 racket, 85 table), with 130 auto and 35 manual additions. Manual additions are mostly classification/threshold misses, not peak-generation gaps: 21 `non_target`, 7 wrong racket/table class, 6 table predictions below threshold, and 1 timing/dense case. No retrain/export/APK/`studs_live` change |
| `T0026` | Retrain `spel_retro_audio` with 2026-06-04 data | `2026-06-04` | Added `train_playing_retro_audio_t0026.py` and trained local candidate `playing_retro_audio_rf_v2026_06_04_t0026_multi_window_context` from 4,598 rows across 18 playing sessions. 06-04 contributes 274 rows; final in-sample focus corrects 26/30 baseline target rows called `non_target` and 6/6 manual-missed rows. No app JSON/APK/`studs_live` change |

## Current Ticket

| Ticket | Goal | Status | Notes |
|---|---|---|---|
| `T0027` | Replay/tune T0026 candidate against T0024 baseline | `Recommended Next` | Compare T0026 candidate marker replay against installed T0024 on 05-28, 05-29, 06-03, and 06-04 before any export/build/install |

## Confirmed Next Tickets

| Ticket | Goal | Notes |
|---|---|---|
| `T0026` | Retrain `spel_retro_audio` with 2026-06-04 data | Use T0025 audit buckets plus historical playing data; keep ordinary `studs_live` unaffected |
| `T0027` | Replay/tune T0026 candidate against T0024 baseline | Compare old/new on 05-28, 05-29, 06-03, and 06-04; sweep table/racket thresholds and same-label dedupe only if audit supports it |
| `T0028` | Export, build, and install improved playing-retro APK | Proceed only if T0027 beats T0024 safely; update only separate `playing_retro_audio_model.json` and app settings needed for Review |
| `T0029` | Revisit playing-retro candidate/peak recovery | Open only if T0025/T0027 show true candidate-generation gaps remain material after retraining |
| `T0030` | Revisit video-assisted FH/BH fusion after audio retro is stable | Keep video paused until audio review workload and dense hit recovery are good enough |

## Dependencies

| Dependency | Purpose | Notes |
|---|---|---|
| `apps/collector` | Android app and bundled model runtime | Do not change app artifacts unless ticket allows it |
| `skills/pingis-audio-classification` | Audio ML/replay workflow | Primary area for playing-retro audit/train/replay/export |
| `data/audio/raw` | Local reviewed session source data | Keep out of git; do not alter labels without approval |
| `data/audio/processed` | Generated datasets/reports | Local artifacts may be regenerated |

## Validation Status

- Build: `T0024 APK installed on Motorola ZY22L6NDHV at 2026-06-04 09:24:18; SHA256 5C03823425158A88A22A1CD4433C1B1ADE179B58E9B1E6634C966BCDC2C9DF61; app pid 4564`
- Tests: T0026 Python py_compile passed for `train_playing_retro_audio_t0026.py`; T0026 training command passed; root `npm run validate` passed. T0025 Python py_compile passed for `audit_playing_retro_review_session_t0025.py`; T0025 audit command passed. T0024 export, app export parity, Collector TypeScript, forced release bundle, and Gradle `assembleRelease` passed on 2026-06-04.
- Lint: `git diff --check` passed on 2026-06-04 for T0026 touched docs/scripts.
- Manual verification: `Bundle string verification passed: T0022 model version, spel_retro_audio_review_only, Ljud + video ML, and Video FH/BH were present; Ljudinsamling and Audio plus IMU were absent.`

## Known Issues Summary

- Current working tree already contains many pre-existing app/model/video changes from earlier work; ticket work must avoid staging or reverting unrelated files.
- `studs_live`, `spel_retro_audio`, and `video_stroke_retro` need separate ticket scopes to avoid model/config bleed.
- IMU/AirHive workflow docs and skill scripts were removed from active scope; remaining app labels/code paths should be treated as legacy unless a ticket explicitly removes or renames them.
- `spel_retro_audio` now has a separate app JSON export, app helper, deterministic replay scripts, primary Review integration, conservative dense recovery, T0015 replay gates, T0019 single-step loading UX, T0020 same-label 80 ms dedupe plus blue-outline explanation, and the T0024 installed T0022 model/settings. It still does not replace `studs_live`.
- T0021 showed that most remaining misses on `audio_session_2026-06-03_005` were model `non_target` calls near real events, not raw candidate absence. T0022 retrained a stronger model, T0023 replay selected thresholds, and T0024 installed that selected path.
- T0025 completed the next per-video audit with `audio_session_2026-06-04_001`: 165 audio markers, 80 racket, 85 table, 130 auto, 35 manual, and 268 model candidates. T0025 found 0 material true candidate-generation gaps; manual additions were mainly model `non_target`, wrong racket/table class, or table-threshold misses, so T0026 should prioritize retraining/classification and T0027 threshold replay before candidate/peak recovery work.
- T0026 trained local candidate `playing_retro_audio_rf_v2026_06_04_t0026_multi_window_context` from 4,598 rows across 18 sessions, adding `audio_session_2026-06-04_001` as playing-retro-only data. The selected variant remains `multi_window_context_racket_weighted` with 197 features. It is not exported; T0027 must replay/tune it marker-by-marker against the installed T0024 baseline before any APK work.
- T0005 trains from all matchable saved app candidates plus manually missed reviewed markers; replay-generated T0004 candidates remain diagnostic, not training rows.
- Ordinary up/down bounce regression for T0007 is advisory only because the old ordinary rows do not preserve exact multi-window event timestamps; do not use it as promotion evidence for `studs_live`.

## Open Questions

- Should future ticket branches be created from current dirty `codex/video-stroke-test` or from a cleaned approved base?
- Should remaining internal legacy IMU code paths be deleted or left as hidden historical fallback now that visible app entry points were removed in T0012C?
