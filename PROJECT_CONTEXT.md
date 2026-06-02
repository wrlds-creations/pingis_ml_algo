# Project Context

This file is the living project memory for Codex and WRLDS. Use it together with `ITERATION_LOG.md` for the detailed ML/build history.

## Project Identity

- Project name: `pingis_ml_algo`
- Repository: `https://github.com/LoveWRLDS/pingis_ml_algo`
- Client-facing name: `TBD`
- Internal codename: `Pingis ML App`
- Project type: React Native Android collector/test app plus local audio/video ML training pipeline
- Current phase: Audio/video model stabilization, with audio-first `spel_retro_audio` work before video fusion

## Business Objective

- Primary objective: Build a pingis training app that can collect labeled audio/video data, train reliable local ML models, and validate them on-device.
- Immediate success metric: Stable racket-contact audio detection in live app modes without floor/table false positives or duplicate counts.
- Longer-term success metric: Use retro audio analysis plus video FH/BH classification to detect in-play racket contacts, table bounces, and stroke type after a recording has been reviewed or imported.
- Business risks: Weak training data quality, unclear model evaluation, live behavior differing from offline metrics, and fragile handoff between AI agents.

## Users And Roles

- Primary user/tester: Love, using a Motorola Android device with microphone and camera.
- AI assistants: Codex and other agents maintaining app code, data pipeline, models, and documentation.
- External stakeholders: TBD.

## Scope

- In scope: Audio collection/review, video collection/review, data export/import, local preprocessing, RandomForest model training/export, Android test builds, live bounce validation, playing-mode retro audio analysis, and video FH/BH stroke classification.
- Current milestone scope: Make `spel_retro_audio` reliable before using video as supporting evidence.
- Future scope: Stroke classification, audio/video fusion, broader app product flows, possible backend/cloud sync.

## Non-Goals

- Do not plan new IMU/AirHive work; the current product direction is audio + video.
- Do not use existing audio-review video as a model input for the audio detector; the separate `Video stroke test` flow may use its own video as stroke-type input.
- Do not trust aggregate ML accuracy without scenario-level and live-device validation.

## Product Requirements

- Core user flows: Collect takes on-device, review markers, save session JSON/media, pull data to computer, train models, export app artifacts, install test APK, validate live.
- Required app surfaces: Audio collection, audio review, `Ljud + video ML`, `Video FH/BH`, `Studsdetektor`, `Studs fritt`, `Studs vaxla sida`, and setup/import flows needed for audio/video testing.
- Performance requirements: Live counting must avoid duplicate counts and keep latency low enough for interactive testing.
- Offline requirements: Local deterministic training must work from pulled session files without cloud services.
- Accessibility requirements: TBD.

## Technical Stack

- Languages: TypeScript/React Native, Kotlin Android native modules, Python ML scripts, JavaScript validation scripts.
- Frameworks: React Native CLI, Android native audio/camera modules, scikit-learn RandomForest training.
- Package manager: npm in `apps/collector`; root `package.json` is for WRLDS template validation only.
- Runtime versions: TBD.
- Architecture notes: App artifacts are JSON RandomForest exports consumed on-device; local `.pkl` and raw data are not intended for git.

## Frontend

- Frontend type: React Native mobile app.
- Platforms: Android is the active device target; iOS is not currently validated.
- UI system: Custom React Native screens with dark test UI; WRLDS UI skill is available for future reusable UI work.
- Navigation model: App-level mode screens in `apps/collector`.
- Design source: Iterative device feedback.

## Backend

- Backend type: None active for the current ML loop.
- API style: Not applicable.
- Services: TBD for future sync/backend.
- Background jobs: Local scripts only.
- Local development approach: Pull device data into `data/audio/raw`, run preprocessing/training/export scripts, build Android APK locally.

## AWS Account And Environments

- AWS account owner: TBD.
- AWS account ID: TBD.
- Regions: TBD.
- Environments: local only for current ML loop.
- Deployment model: GitHub branch/PR workflow, local Android release build for device testing.
- Required WRLDS tags: See `AWS_RESOURCES.md` before any AWS work.

## Data Model

- Main entities: Audio session files, video-stroke session files, takes/events, review markers, waveform/audio clips, MP4 files, model datasets, model artifacts.
- Audio review markers now carry explicit `review_status`, `contact_kind`, `not_racket_kind`, and `bounce_side` fields. `ignore` means skipped data, not a negative label.
- Playing review markers are intentionally compact for audio/video: human audio labels are racket contact, table bounce, or ignore; motion labels are `Forehand`, `Backhand`, or `Oklart`. Auto-candidates carry confidence and must not become training truth until Love confirms or corrects them.
- Audio session events now distinguish high-level `scenario` (`audio_sound`, `racket_bouncing`, `playing`) from detailed `scenario_id`. Legacy IMU-oriented scenarios may still exist in old raw JSON, but they are not active product scope and must not drive new model work.
- `Ljud + video ML` collection records or imports one whole WAV + MP4 event, without user-facing 30 s review splitting. Review is staged: first the normal audio waveform marks `Racketträff`, `Bordsstuds`, and `Ignorera`; during/after that audio pass, pose scans the whole video at 15 fps and Review 2 uses the waveform-like FH/BH motion timeline rather than anchoring analysis only to confirmed audio hits. Motion rows stay separate from audio rows with `Forehand`, `Backhand`, or `Oklart`. Imported iPhone/Drive videos use Android's video picker, copy the MP4 locally, and extract the embedded audio track to the same 22,050 Hz mono WAV format as normal collection. Per-take video sync stores `video_sync_offset_ms`, manual/auto anchor points, and `video_sync_source`; audio/contact training and pose-motion training remain separate data layers.
- `Video FH/BH` is the dedicated video-only data path for camera-only stroke detection. It imports a whole MP4, stores it under `/Download/pingis_video_stroke_sessions`, extracts the embedded audio to WAV only for visible waveform review, runs ML Kit pose at 15 fps, proposes FH/BH/unknown motion markers, and saves reviewed `video_stroke_session_*.json` files for the video-stroke training pipeline. Review v5 mirrors the normal audio review controls as closely as this video-only flow allows: video audio stays audible, the extracted WAV is shown as the main timeline, zoom supports 1x/2x/4x/8x/12x/16x, timeline and playhead drag/long-press behave like audio review, marker nav uses the same previous/next controls, marker nudging supports +/-10/20/50 ms, and selected stroke playback uses `-700 ms` to `+500 ms`.
- Audio-only collection now separates quiet, speech/music, other-bounce, and fast-racket scenarios. `Racket + musik` uses an explicit low/medium/high background level instead of exposing separate `music_low` presets in the UI. Reviewed marker preprocessing uses short event-centered clips and stores nearest-event spacing metadata so dense contact sequences can be evaluated separately from isolated sounds.
- New recordings store a detection config snapshot (`strict/normal/sensitive`, `hybrid/four_class_only/binary_only`, thresholds, merge window, model versions) plus model candidates separately from human review markers. Human-reviewed markers remain training truth; candidates are analysis data for misses and false positives.
- `spel_retro_audio` T0004 candidate reporting is available via `python skills/pingis-audio-classification/scripts/build_playing_retro_candidate_report.py`. It writes local ignored CSV/MD outputs under `data/audio/processed/` and matches saved app candidates plus replay-generated peaks against reviewed racket/table markers without training or exporting a model.
- `spel_retro_audio` T0005 local candidate training is available via `python skills/pingis-audio-classification/scripts/train_playing_retro_audio.py`. Candidate `playing_retro_audio_rf_v2026_06_02_app_candidates_100_200` trains from all matchable saved app candidate peaks plus manually reviewed missed racket/table markers, labels unmatched app candidates as `non_target`, and keeps replay-generated peaks as diagnostics only. It uses 4,028 candidate-centered rows across 16 playing sessions and writes ignored local artifacts under `data/audio/processed/`, `data/audio/models/evaluations/`, and `data/audio/models/playing_retro_candidates/`. Holdout `audio_session_2026-05-29_002` reaches 0.759 accuracy versus old app prediction 0.682, with racket recall 0.604, table recall 0.924, and non-target recall 0.625. Ordinary up/down bounce regression is reported separately at 10,353 rows / 0.714 accuracy; this candidate is not exported to Collector and must not affect `studs_live`.
- `spel_retro_audio` T0006 variant comparison is available via `python skills/pingis-audio-classification/scripts/evaluate_playing_retro_audio_variants.py`. It selected local candidate `playing_retro_audio_rf_v2026_06_02_safe_racket_weighted` after comparing focused one-window weighting variants. On holdout `audio_session_2026-05-29_002`, it improves accuracy 0.759 -> 0.771, racket recall 0.604 -> 0.623, table recall 0.924 -> 0.933, and keeps non-target recall at 0.625. The aggressive variant reached racket recall 0.679 but was rejected because non-target recall fell to 0.500. T0006 is local-only and not app-ready; next work should add true multi-window and non-leaky candidate-context features.
- Before pulling or training a new audio file/session, Codex must ask Love for missing intake metadata: whether it is trainable/diagnostic/holdout, the evaluation bucket (`ordinary_bounce`, `playing_dense`, `stiga_office_tomas`, table/floor/noise, etc.), impact style/force, environment/background, phone/mic placement, device/recorder/player, and whether it may affect the ordinary bounce detector or should stay domain-specific until cross-bucket replay passes.
- Current Collector audio artifact in the workspace is `collector_bounce_live_v2026_05_28_tomas_stiga_candidate_normal_4class_220_80_220`: `Normal / 4-klass`, `-100/+200 ms` model clips, confidence `0.65`, timing gates `Retrigger 220 ms`, `Group 80 ms`, `Merge 220 ms`, and 4-class model `collector_audio_4class_v2026_05_28_tomas_stiga_C_hybrid_window_candidate`. The current installed Motorola APK also contains `Video FH/BH` review v5, the whole-video `Ljud + video ML` motion gate aligned with `Video FH/BH`, and video model `collector_video_stroke_v2026_05_28_tomas_player_right_forehand_candidate`; it was installed on `ZY22L6NDHV` on 2026-05-29 with APK SHA256 `CE5965D72BD27723269383B88D6EFB87A7531F5579E20E6E875BE99BD4A5707F`.
- Review markers and review-relevant candidate pins are color-coded by detected/reviewed class: racket, table, floor, noise/music, other, and ignore. Non-review-relevant audio candidates stay in JSON as analysis data but are hidden from the normal waveform so they do not look like deleted, untouchable markers.
- Audio preprocessing skips unsafe noise/floor/other negative training windows that overlap a confirmed racket contact, because those clips can contain both sounds and should be analysis-only. Reviewed `table_bounce` markers are retained in dense racket/table play, even when close to racket markers, because that is the target real-game separation problem.
- Diagnostic-only sessions `audio_session_2026-05-26_002`, `audio_session_2026-05-26_003`, and `audio_session_2026-05-26_004` are excluded from audio and video-stroke preprocessing. Session 002 was saved without correction while the rejected 2026-05-26 audio candidate produced hundreds of false racket confirmations; sessions 003/004 are throwaway same-media comparison tests for the corrected candidate versus the rollback baseline.
- Reviewed session `audio_session_2026-05-26_001` is valid dense racket/table audio-video-pose data and is included in audio training. The first corrected candidate after restoring close table rows is `collector_audio_4class_v2026_05_26_dense_001_C_hybrid_window_candidate`; it underperformed in Love's Motorola feel-test and was rolled back from the installed app.
- Reviewed Stiga-office session `audio_session_2026-05-28_002` is valid Tomas audio/video dense data and is included in local candidate training. It has 145 reviewed audio markers: 67 racket contacts, 74 table bounces, and 4 not-racket markers without a more specific kind.
- Reviewed Stiga-office session `audio_session_2026-05-29_001` is valid trainable Tomas hard racket/table audio-video data and is included in local candidate training. It has 142 reviewed audio markers: 66 racket contacts, 72 table bounces, and 4 ignored markers. Local metadata tags it as `playing_dense_audio`, `stiga_office_tomas_hard`, and `hard_stroke_like`; it must remain bucket-evaluated before any ordinary bounce promotion. The first 05-29 retrain improved new-session live replay by only +1 true positive while adding +5 false positives, so it stays candidate-only.
- Reviewed Stiga-office session `audio_session_2026-05-29_002` is valid trainable Tomas backhand dense racket/table audio-video data, pulled from Motorola on 2026-06-01. It has 225 reviewed audio markers: 106 racket contacts and 119 table bounces, with many table-to-racket gaps under 120 ms. Local metadata tags it as `playing_dense_audio`, `stiga_office_tomas_backhand`, and `hard_backhand_stroke_like`; it must not influence ordinary up/down bounce promotion unless cross-bucket replay stays clean.
- Data ownership: TBD.
- Data retention: Raw training data is local and can be large; keep `/data/` gitignored.
- Data import/export requirements: Pull `audio_session_*.json` and matching media folders from device storage into local raw data folders.

## Hardware And Media

- Hardware involved: Motorola Android device, microphone, and camera.
- Media involved: WAV audio extracted or recorded at the Collector target format, plus MP4 video for `Ljud + video ML` and `Video FH/BH`.
- Sensor protocols: No active external sensor protocol in the current project scope.
- Firmware assumptions: Not applicable for the current audio/video scope.
- Calibration requirements: Audio/video sync and selected-player/camera metadata matter for review and evaluation; no AirHive/table-baseline calibration is active.

## Active Models

- `audio_model`: Four-class `racket_bounce / table_bounce / floor_bounce / noise`; current default live count engine in `Normal / 4-klass`.
- `audio_contact_model`: Binary `racket_contact` vs `not_racket_contact`; optional debug/comparison engine for binary and hybrid modes.
- `spel_retro_audio`: Local retro-analysis candidate family for post-recording playing-mode audio, separate from `studs_live`. Current local candidate is `playing_retro_audio_rf_v2026_06_02_safe_racket_weighted`; it is not bundled in `apps/collector/src/models/audio_model.json` and is not APK-ready.
- `video_stroke_model`: Android-only forehand/backhand camera model used by unified `Ljud + video ML` review and the separate `Video FH/BH` video-only review. Current installed APK contains `collector_video_stroke_v2026_05_28_tomas_player_right_forehand_candidate` after adding Tomas right-handed player-right forehand data. The model has a third raw `unknown` class for opponent/no-visible-player-stroke motion; app inference maps `unknown` to `uncertain`, not FH/BH. `Ljud + video ML` and `Video FH/BH` run whole-video pose at 15 fps, apply a motion gate before classification, auto-create visible motion markers only for concrete FH/BH candidates at 58%+ confidence, and let Love confirm/correct FH/BH/unknown markers before they become training rows. Pose features use the selected player's racket arm landmarks from `handedness`; `camera_side` tracks `player_left`/`player_right` for evaluation and balancing, but is not a RandomForest feature yet.
- Legacy IMU docs/scripts have been removed from the active workflow. Historical decisions/logs may still mention them, and old app code may still contain legacy surfaces until a focused cleanup ticket removes or renames them.

## Current Known Problems

- One physical racket contact can sometimes count twice, especially with catch/after-sound or same-peak duplicate auto-candidates. Live JS has `contact_group` debug and duplicate suppression; dense Review now also suppresses same-label auto-marker duplicates inside 180 ms while preserving close table/racket pairs.
- Collection modes are separated by data type: audio-only sound data, audio+video review data, and video-only FH/BH data. Legacy IMU-labeled app flows may still exist in code, but they are not active scope.
- Dense play collection is now first-class for audio/video: realistic alternating table/racket sequences can have contacts under 120-300 ms apart and must be evaluated separately from isolated racket-bounce takes.
- Floor/table/catch-after-sound rejection now requires reviewed hard-negative takes before primary contact training.
- Quiet audio is no longer enough for model progress; the next data gap is reviewed noisy/fast racket positives plus noisy table/floor/other-impact negatives.
- Review video/audio offset is handled in Review with saved `audio_origin_in_video_ms`, per-take `video_sync_offset_ms`, and an optional clap/tap sync event at the start of new takes. The separate `Video FH/BH` flow is shown under `DATA`, stores video-only imported MP4 files under `/Download/pingis_video_stroke_sessions`, and saves one whole-video review JSON without 30 s splitting. Its auto stroke detections must not enter training unless Love confirms or corrects them.

## Commands

- Root validation: `npm run validate`
- Collector install dependencies: `cd apps/collector && npm install`
- Android build: use the existing local Android build flow in `apps/collector`; check `ITERATION_LOG.md` for the latest working command and build status.
- Audio preprocessing/training: use `skills/pingis-audio-classification/scripts/`.
- Playing-retro candidate report: `python skills/pingis-audio-classification/scripts/build_playing_retro_candidate_report.py`
- Video stroke preprocessing/training: use `skills/pingis-stroke-detection/scripts/` only for the video-stroke pipeline.

## Definition Of Done

- Functional criteria: The requested app/data/model behavior works on the Motorola or in deterministic local scripts.
- Review criteria: Changes are scoped to one active `CODEX_TASK.md` ticket, performed on a feature branch, and not pushed directly to `main`.
- Validation criteria: Relevant validation/test command is run or the blocker is documented.
- Documentation criteria: Update `PROJECT_CONTEXT.md`, `DECISIONS.md`, `REPO_CURRENT_STATE.md`, `FOLLOWUPS.md`, and/or `ITERATION_LOG.md` when facts, decisions, tickets, follow-ups, models, or builds change.

## Testing Requirements

- Automated tests: Root template validation plus project-specific script tests where relevant.
- Manual tests: Device tests in `TEST_PLAN.md`.
- Device or browser coverage: Motorola Android device is the current required device for app behavior.
- Infrastructure validation: `AWS_RESOURCES.md` and AWS skill required before any AWS changes.

## Security And Privacy

- Data classification: TBD; assume raw audio/video data may be sensitive local user data.
- Secrets handling: Do not commit credentials or device-private exports.
- PII or sensitive data: Review media may contain face/voice/background environment.
- Compliance considerations: TBD.

## Open Questions

| Question | Why It Matters | Owner | Status |
|---|---|---|---|
| What is the final client-facing product name? | Needed for release/app branding | Love | Open |
| Which cloud/backend, if any, will own collected data later? | Determines AWS/security architecture | Love + WRLDS | Open |
| Should legacy `Audio plus IMU` app surfaces be removed or renamed now that the product scope is audio/video only? | Prevents future agents from collecting or training obsolete sensor data | Love + Codex | Open |
