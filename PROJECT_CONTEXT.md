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
- Required app surfaces: Audio collection, audio review, `Studsdetektor`, `Studs fritt`, `Studs vaxla sida`, calibration/setup, and future IMU collection/test surfaces.
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
- Data ownership: TBD.
- Data retention: Raw training data is local and can be large; keep `/data/` gitignored.
- Data import/export requirements: Pull `audio_session_*.json` and matching media folders from device storage into local raw data folders.

## Hardware And Sensors

- Hardware involved: Motorola Android device, microphone, camera for review support, AirHive IMU sensor for future fusion.
- Sensor protocols: AirHive BLE details live in local AirHive skills.
- BLE requirements: AirHive stream support exists but active product focus is audio.
- Firmware assumptions: TBD.
- Calibration requirements: AirHive calibration exists for synced collection, but IMU model work is paused.

## Active Models

- `audio_contact_model`: Binary `racket_contact` vs `not_racket_contact`; primary live count engine.
- `audio_model`: Four-class `racket_bounce / table_bounce / floor_bounce / noise`; secondary veto/debug model.
- `stroke_hit_model`: Future IMU hit/miss model; paused.
- `stroke_type_model`: Future IMU forehand/backhand model; paused.
- `bounce_imu_model`: Experimental synchronized bounce-motion model; paused.

## Current Known Problems

- One physical racket contact can sometimes count twice, especially with catch/after-sound.
- Forehand/backhand-style racket contacts are under-covered and can be missed.
- Floor bounce rejection is improved but not solved.
- Review video/audio offset exists but is not the current priority unless it blocks labeling.

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
