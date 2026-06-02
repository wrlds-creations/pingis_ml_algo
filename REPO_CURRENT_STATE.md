# Repo Current State

Use this file as the living snapshot of what actually exists in the repository. Update it after completed tickets, audits, meaningful dependency changes, validation changes, workflow changes, model/data/build state changes, or next-ticket changes.

## Snapshot

- Date: `2026-06-02`
- Current branch: `codex/t0010-playing-retro-audio-review-replay-ui`
- Current phase: `Audio-first playing-retro Review integration`
- Current status: `T0010 app-export replay passed on saved Tomas/Stiga Review candidates; next step is visible Review candidate surfacing/generation before APK`
- Current ticket: `T0011`
- Last completed ticket: `T0010`
- Recommended next ticket: `T0011-playing-retro-audio-review-ui-apk`

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
| `cd apps/collector && npx tsc --noEmit` | Collector TypeScript validation | `Passed 2026-06-02 after T0009 playing-retro helper` |
| `python skills/pingis-audio-classification/scripts/build_playing_retro_candidate_report.py` | T0004 candidate-centered playing-retro audio report | `Passed 2026-06-02 on Tomas 05-28/05-29 sessions` |
| `python skills/pingis-audio-classification/scripts/train_playing_retro_audio.py` | T0005 local `spel_retro_audio` candidate train/eval | `Passed 2026-06-02; 4,028 rows, holdout accuracy 0.759` |
| `python skills/pingis-audio-classification/scripts/evaluate_playing_retro_audio_variants.py` | T0006 focused variant comparison | `Passed 2026-06-02; selected safe one-window candidate, holdout racket recall 0.623` |
| `python skills/pingis-audio-classification/scripts/evaluate_playing_retro_audio_multi_window.py` | T0007 multi-window/context variant comparison | `Passed 2026-06-02; selected multi-window/context candidate, holdout racket recall 0.896` |
| `python skills/pingis-audio-classification/scripts/evaluate_playing_retro_audio_cross_session.py` | T0008 cross-session validation | `Passed 2026-06-02; selected T0007 variant passes requested Tomas/Stiga holdouts` |
| `python skills/pingis-audio-classification/scripts/export_playing_retro_audio_model_json.py` | T0009 separate app JSON export for Review-only `spel_retro_audio` | `Passed 2026-06-02; exported 197-feature / 450-tree playing-retro model to apps/collector/src/models/playing_retro_audio_model.json` |
| `python skills/pingis-audio-classification/scripts/validate_playing_retro_audio_app_export.py` | T0009 export parity check against selected joblib model | `Passed 2026-06-02; feature order, labels, windows, and no-truth-field checks passed` |
| `python skills/pingis-audio-classification/scripts/replay_playing_retro_audio_app_export.py` | T0010 controlled replay of app export on saved Review candidates | `Passed 2026-06-02; 643 saved Tomas/Stiga candidates, acc 0.978, racket 0.987, table 0.988, non-target 0.948` |
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

## Current Ticket

| Ticket | Goal | Status | Notes |
|---|---|---|---|
| `T0011` | Wire visible Review path and APK gate for `spel_retro_audio` | `Ready` | T0010 shows reclassification is strong on saved candidates. Next step is Review UI/candidate surfacing so missed markers can be addressed before a Motorola APK test |

## Confirmed Next Tickets

| Ticket | Goal | Notes |
|---|---|---|
| `T0011` | Wire visible Review path and APK gate for `spel_retro_audio` | Keep separate from live/studs; decide whether to reclassify saved candidates only or add retro candidate generation for missed markers |
| `T0012` | Revisit video-assisted FH/BH fusion after audio retro is stable | Keep video paused until audio is useful |

## Dependencies

| Dependency | Purpose | Notes |
|---|---|---|
| `apps/collector` | Android app and bundled model runtime | Do not change app artifacts unless ticket allows it |
| `skills/pingis-audio-classification` | Audio ML/replay workflow | Primary area for T0004-T0008 |
| `data/audio/raw` | Local reviewed session source data | Keep out of git; do not alter labels without approval |
| `data/audio/processed` | Generated datasets/reports | Local artifacts may be regenerated |

## Validation Status

- Build: `Not run for T0010; no APK requested`
- Tests: `python .../validate_playing_retro_audio_app_export.py`, `python .../replay_playing_retro_audio_app_export.py`, targeted `py_compile`, and root `npm run validate` passed on 2026-06-02
- Lint: `git diff --check` passed for T0010 on 2026-06-02
- Manual verification: `T0010 replay loads apps/collector/src/models/playing_retro_audio_model.json, not joblib or audio_model.json. It evaluates saved app candidate timestamps and reports truth-derived close_event_bucket/neighbor_sequence only as metadata.`

## Known Issues Summary

- Current working tree already contains many pre-existing app/model/video changes from earlier work; ticket work must avoid staging or reverting unrelated files.
- `studs_live`, `spel_retro_audio`, and `video_stroke_retro` need separate ticket scopes to avoid model/config bleed.
- IMU/AirHive workflow docs and skill scripts were removed from active scope; remaining app labels/code paths should be treated as legacy unless a ticket explicitly removes or renames them.
- `spel_retro_audio` now has a separate app JSON export, opt-in helper, and deterministic replay script, but it is not wired into a visible Review UI or APK yet.
- T0005 trains from all matchable saved app candidates plus manually missed reviewed markers; replay-generated T0004 candidates remain diagnostic, not training rows.
- Ordinary up/down bounce regression for T0007 is advisory only because the old ordinary rows do not preserve exact multi-window event timestamps; do not use it as promotion evidence for `studs_live`.

## Open Questions

- Should future ticket branches be created from current dirty `codex/video-stroke-test` or from a cleaned approved base?
- Should T0011 install a Review-only reclassification APK first, or add retro candidate generation so the 15 missed Tomas/Stiga markers can also be surfaced?
- Should legacy IMU app surfaces be removed immediately or left as hidden/internal historical code until the next UI cleanup?
