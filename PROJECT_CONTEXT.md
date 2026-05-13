# Project Context

This file is the living project memory for Codex and WRLDS. Use it together with `ITERATION_LOG.md` for the detailed ML/build history.

## Project Identity

- Project name: `pingis_ml_algo`
- Repository: `https://github.com/LoveWRLDS/pingis_ml_algo`
- Client-facing name: `TBD`
- Internal codename: `Pingis ML App`
- Project type: React Native Android collector/test app plus local ML training pipeline
- Current phase: Audio-first model stabilization and device validation

## Business Objective

- Primary objective: Build a pingis training app that can collect labeled sensor data, train reliable local ML models, and validate them on-device.
- Immediate success metric: Stable racket-contact audio detection in live app modes without floor/table false positives or duplicate counts.
- Longer-term success metric: Combine racket-contact audio with AirHive IMU swing evidence for forehand/backhand and in-play stroke confidence.
- Business risks: Weak training data quality, unclear model evaluation, live behavior differing from offline metrics, and fragile handoff between AI agents.

## Users And Roles

- Primary user/tester: Love, using a Motorola device and AirHive IMU hardware.
- AI assistants: Codex and other agents maintaining app code, data pipeline, models, and documentation.
- External stakeholders: TBD.

## Scope

- In scope: Audio collection, review UI, data export/import, local preprocessing, RandomForest model training/export, Android test builds, live bounce validation, future AirHive IMU fusion.
- Current milestone scope: Make audio detection reliable before expanding IMU-driven behavior.
- Future scope: Stroke classification, FH/BH swing fusion, broader app product flows, possible backend/cloud sync.

## Non-Goals

- Do not make IMU the primary contact detector until audio is stable.
- Do not use review video as a model input yet; video is labeling support only.
- Do not trust aggregate ML accuracy without scenario-level and live-device validation.

## Product Requirements

- Core user flows: Collect takes on-device, review markers, save session JSON/media, pull data to computer, train models, export app artifacts, install test APK, validate live.
- Required app surfaces: Audio collection, audio review, `Audio plus IMU` with `Racketstuds` and `Playing` scenarios, `Studsdetektor`, `Studs fritt`, `Studs vaxla sida`, calibration/setup, and future IMU collection/test surfaces.
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

- Main entities: Audio session files, takes/events, review markers, waveform/audio clips, model datasets, model artifacts.
- Audio review markers now carry explicit `review_status`, `contact_kind`, `not_racket_kind`, and `bounce_side` fields. `ignore` means skipped data, not a negative label.
- Playing review markers are intentionally compact: human labels are `forehand_hit`, `backhand_hit`, or `table_bounce`; auto-candidates carry audio confidence and can be saved as `filtered` so they do not become training truth.
- Audio+IMU takes store AirHive `sensor_ts`, absolute `received_at_ms`, and take-relative `take_ts_ms` for IMU samples, plus per-take target/measured sample-rate and quality metadata.
- Audio session events now distinguish high-level `scenario` (`audio_sound`, `racket_bouncing`, `playing`) from detailed `scenario_id`; guided racket-bounce IMU now exposes FH-side, BH-side, and `racket_motion_no_bounce` for natural racket-arm motion without ball contact. Mixed racket-bounce is treated as legacy/free-review data rather than a guided preset.
- `racket_motion_no_bounce` review can be saved with zero markers; the whole reviewed take means "no bounce happened", and bounce-IMU preprocessing samples negative IMU windows from the saved sequence after the sync lead-in. Any explicit `no_bounce_motion` markers are IMU-only and must not be used as audio training examples.
- Audio-only collection now separates quiet, speech/music, other-bounce, and fast-racket scenarios. `Racket + musik` uses an explicit low/medium/high background level instead of exposing separate `music_low` presets in the UI. Reviewed marker preprocessing uses short event-centered clips and stores nearest-event spacing metadata so dense contact sequences can be evaluated separately from isolated sounds.
- New recordings store a detection config snapshot (`strict/normal/sensitive`, `hybrid/four_class_only/binary_only`, thresholds, merge window, model versions) plus model candidates separately from human review markers. Human-reviewed markers remain training truth; candidates are analysis data for misses and false positives.
- Current live baseline is `collector_bounce_live_v2026_05_13_normal_4class_220_80_220`: `Normal / 4-klass`, `-100/+200 ms` model clips, and timing gates `Retrigger 220 ms`, `Group 80 ms`, `Merge 220 ms`. This is the single source of truth for the Collector bounce detector until a later full-chain replay plus Motorola test beats it.
- Review markers and candidate pins are color-coded by detected/reviewed class: racket, table, floor, noise/music, other, and ignore.
- Audio preprocessing skips negative/noise/table/floor training windows that land within 300 ms of a confirmed racket contact, because those clips can contain both sounds and should be analysis-only.
- Data ownership: TBD.
- Data retention: Raw training data is local and can be large; keep `/data/` gitignored.
- Data import/export requirements: Pull `audio_session_*.json` and matching media folders from device storage into local raw data folders.

## Hardware And Sensors

- Hardware involved: Motorola Android device, microphone, camera for review support, AirHive IMU sensor for future fusion.
- Sensor protocols: AirHive BLE details live in local AirHive skills.
- BLE requirements: AirHive stream support exists; raw synced collection targets 150 Hz when the sensor/BLE path is stable, but the app reports measured rate instead of assuming it.
- Firmware assumptions: TBD.
- Calibration requirements: AirHive/table-baseline is required for synced IMU capture; FH/BH pose calibration is optional helper metadata for Audio plus IMU and is not a forehand/backhand ground-truth label.

## Active Models

- `audio_contact_model`: Binary `racket_contact` vs `not_racket_contact`; primary live count engine.
- `audio_model`: Four-class `racket_bounce / table_bounce / floor_bounce / noise`; secondary veto/debug model.
- `stroke_hit_model`: Future IMU hit/miss model; paused.
- `stroke_type_model`: Future IMU forehand/backhand model; paused.
- `bounce_imu_model`: Experimental synchronized bounce-motion model; paused.

## Current Known Problems

- One physical racket contact can sometimes count twice, especially with catch/after-sound; live JS now adds `contact_group` debug and duplicate suppression, but Motorola validation is still required.
- Collection modes are separated: `Ljudinsamling` is audio-only sound data for racket/table/floor/noise; `Audio plus IMU` contains `Racketstuds` for controlled racket-bounce IMU and natural no-bounce racket-arm motion, plus `Playing` for longer review-first sequences. The old separate `Fri inspelning` card is hidden from the startsida.
- Dense play collection is now first-class: `Spel: racket + bord` under `Ljudinsamling` and `Playing: racket + bord` under `Audio plus IMU` are for realistic alternating table/racket sequences where contacts can be 170-300 ms apart. These takes must be evaluated separately from isolated racket-bounce takes.
- Floor/table/catch-after-sound rejection now requires reviewed hard-negative takes before primary contact training.
- Quiet audio is no longer enough for model progress; the next data gap is reviewed noisy/fast racket positives plus noisy table/floor/other-impact negatives.
- Review video/audio offset is handled in Review with saved `audio_origin_in_video_ms`, per-take `video_sync_offset_ms`, and an optional clap/tap sync event at the start of new takes.

## Commands

- Root validation: `npm run validate`
- Collector install dependencies: `cd apps/collector && npm install`
- Android build: use the existing local Android build flow in `apps/collector`; check `ITERATION_LOG.md` for the latest working command and build status.
- Audio preprocessing/training: use `skills/pingis-audio-classification/scripts/`.
- IMU preprocessing/training: use `skills/pingis-stroke-detection/scripts/`.

## Definition Of Done

- Functional criteria: The requested app/data/model behavior works on the Motorola or in deterministic local scripts.
- Review criteria: Changes are on a feature branch, not pushed directly to `main`.
- Validation criteria: Relevant validation/test command is run or the blocker is documented.
- Documentation criteria: Update `PROJECT_CONTEXT.md`, `DECISIONS.md`, and/or `ITERATION_LOG.md` when facts, decisions, models, or builds change.

## Testing Requirements

- Automated tests: Root template validation plus project-specific script tests where relevant.
- Manual tests: Device tests in `TEST_PLAN.md`.
- Device or browser coverage: Motorola Android device is the current required device for app behavior.
- Infrastructure validation: `AWS_RESOURCES.md` and AWS skill required before any AWS changes.

## Security And Privacy

- Data classification: TBD; assume raw audio/video/IMU data may be sensitive local user data.
- Secrets handling: Do not commit credentials or device-private exports.
- PII or sensitive data: Review media may contain face/voice/background environment.
- Compliance considerations: TBD.

## Open Questions

| Question | Why It Matters | Owner | Status |
|---|---|---|---|
| What is the final client-facing product name? | Needed for release/app branding | Love | Open |
| Which cloud/backend, if any, will own collected data later? | Determines AWS/security architecture | Love + WRLDS | Open |
| What live thresholds define “audio stable enough” for IMU work to resume? | Prevents premature model fusion | Love + Codex | Open |
