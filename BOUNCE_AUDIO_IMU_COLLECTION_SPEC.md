# Spec: Bounce Audio + IMU Collection

Date: 2026-04-22
Status: Proposed build spec
Owner: Codex + Love

## Summary

Build a new collector mode named `Bounce audio + IMU collection`.

This mode must:
- keep the current audio review and audio training flow intact
- record synchronized IMU data from the AirHive sensor during the same take
- let reviewed audio markers act as contact anchors for a future bounce-specific IMU model

This mode does not replace the current binary audio contact model.
It exists to create a second, separate bounce-motion dataset that we can train or ignore later.

## What Must Stay The Same

The current audio path stays intact:
- same review flow
- same binary labels:
  - `Racket` -> `racket_contact`
  - `Not racket` -> `not_racket_contact`
  - `Ignore` -> excluded
- same primary live role for the binary audio contact model

Nothing in this spec should force us to replace the audio model.

## What Is New

During the same take we will also collect:
- accelerometer
- gyroscope
- magnetometer
- timestamped IMU samples from the AirHive

The user still reviews only the audio.
The review is the source of truth for contact timing.

## Product Goal

Create a dataset that lets us train a separate IMU model for:
- `bounce_contact_motion`
- `not_bounce_contact`

This model is specifically for the short, repeated hand motion used when bouncing the ball on the racket.
It is not the same as the existing stroke models.

## Non-Goals

This iteration does not:
- replace the binary audio model
- train a combined audio + IMU end-to-end model
- solve partner rally
- add a fast double-bounce collection protocol
- add a forehand/backhand bounce-side ML model

Bounce side can continue to be handled separately by orientation and calibration.

## User Flow

1. User chooses `Bounce audio + IMU collection` from setup.
2. App asks for:
   - microphone permission
   - BLE / sensor connection
3. User connects AirHive sensor.
4. App shows the guided scenario list, same idea as current audio collection.
5. User starts a `15 s` take.
6. App records at the same time:
   - one WAV take
   - one IMU stream for that take
7. When the take ends:
   - if scenario requires review, open the same audio review screen
   - if scenario is auto-only, save directly
8. After review:
   - save reviewed audio markers
   - save IMU samples already attached to the same take
9. Offline preprocessing uses the review markers to cut out IMU windows around confirmed contacts.

## Scenarios

Use the same base scenarios as audio collection:

| Scenario | Audio role | IMU role |
|---|---|---|
| `racket_quiet` | positive | positive bounce motion |
| `racket_counting` | positive in speech noise | positive bounce motion in noisy condition |
| `racket_music_low` | positive in music noise | positive bounce motion in noisy condition |
| `racket_music_mid` | positive in heavier music noise | positive bounce motion in noisy condition |
| `speech_only` | negative | negative hand / body movement if sensor is worn |
| `desk_keyboard_only` | negative | negative hand movement if sensor is worn |
| `music_low_only` | negative | negative / idle reference |
| `music_mid_only` | negative | negative / idle reference |
| `table_quiet` | negative for binary contact | optional hard negative for bounce IMU |
| `floor_quiet` | negative for binary contact | optional hard negative for bounce IMU |

## Collection Protocol

For the base round:
- use about `0.5-1.0 s` between bounces
- do not intentionally collect fast double-bounces yet
- wear the IMU on the playing arm throughout the take

## Data Contract

## Session File

Create a new session type, for example:
- `bounce_fusion_session_YYYY-MM-DD_NNN.json`

Recommended schema:

```json
{
  "session_meta": {
    "recorder_name": "Love",
    "session_date": "2026-04-22T12:34:56.000Z",
    "app_version": "1.7",
    "collection_mode": "bounce_audio_imu",
    "target_duration_s": 15,
    "planned_takes": 36,
    "sensor_name": "AirHive",
    "sensor_sample_rate_hz": 50
  },
  "events": [
    {
      "scenario_id": "racket_counting",
      "background_condition": "speech",
      "label": "racket_bounce",
      "wav_filename": "racket_counting_001.wav",
      "duration_ms": 15020,
      "take_index": 1,
      "target_duration_s": 15,
      "take_start_unix_ms": 1776857696000,
      "imu_samples": [
        {
          "accel_x": 123,
          "accel_y": -44,
          "accel_z": 981,
          "gyro_x": 4.1,
          "gyro_y": -8.2,
          "gyro_z": 1.7,
          "mag_x": 12,
          "mag_y": -3,
          "mag_z": 28,
          "ts_ms": 0
        }
      ],
      "review": {
        "required": true,
        "anchor_rule": "attack_start",
        "completed_at": "2026-04-22T12:40:00.000Z",
        "markers": [
          {
            "id": "auto_000",
            "timestamp_ms": 1430,
            "source": "auto",
            "suggested_label": "racket_contact",
            "final_label": "racket_contact"
          }
        ]
      }
    }
  ]
}
```

## Timing Rule

The important sync rule is:
- review marker `timestamp_ms` is measured from take start
- IMU sample `ts_ms` is measured from the same take start

We do not need sample-perfect hardware sync.
We do need stable take-relative timing good enough for bounce windows.

Target sync quality:
- usable within about `+/-50 ms`

## Implementation Rule For Sync

At take start:
- record `take_start_unix_ms = Date.now()`
- start microphone session
- start BLE sample capture immediately

For IMU:
- every sample written into `imu_samples[]` gets `ts_ms` relative to take start

For audio review:
- markers remain `timestamp_ms` relative to take start

That gives us practical 1:1 alignment for dataset generation.

## Review Behavior

Review stays audio-first:
- user labels audio markers exactly as now
- user does not manually review IMU

Why:
- audio review already matches the actual contact event best
- IMU windows can be derived offline once contact timing is known

## Offline Preprocessing

Two outputs should continue to exist:

1. audio binary dataset
- unchanged
- still builds `audio_contact_dataset.csv`

2. new IMU bounce dataset
- new script or new preprocessing mode
- uses reviewed audio markers as anchors

## IMU Window Extraction

Recommended first baseline:
- window length: `400 ms`
- anchor: reviewed `attack_start`
- window: `-180 ms` to `+220 ms` around marker

Reason:
- bounce motion is short
- we want a little lead-in before contact
- we do not want the much larger `800 ms` stroke window by default

## IMU Labels

Primary new labels:
- `bounce_contact_motion`
- `not_bounce_contact`

Positive windows:
- markers with `final_label = racket_contact`

Negative windows:
- markers with `final_label = not_racket_contact`
- background windows sufficiently far from positive markers
- optional noise-only takes while wearing the sensor

Ignored markers:
- no IMU training window is created

## Model Plan

Train a separate model:
- name: `bounce_imu_model`
- task: binary bounce-contact motion

Recommended first version:
- RandomForest baseline first
- feature-based, same philosophy as the existing early IMU work

Possible later live inference:
- sliding window around `300-500 ms`
- step around `40-80 ms`

## Fusion Plan

Do not make IMU the new truth source.

Use it as a second signal:
- strong audio alone can still count
- weak / borderline audio can be rescued by strong bounce IMU evidence
- audio plus IMU together can reduce false positives in noise

Recommended first fusion logic later:
- `count if audio is strong`
- `count if audio is medium and bounce_imu_model is strong`
- `reject if audio is weak and IMU is weak`

## UI Requirements

New collector mode must show:
- sensor connection status
- live IMU sample rate
- current scenario
- current take number
- whether both audio and IMU are actively recording

If BLE disconnects during a take:
- mark the take as partial
- do not silently treat it as clean synchronized data

## Acceptance Criteria

The mode is acceptable when:
- a take saves both WAV and IMU samples together
- review markers remain relative to the same take start
- reviewed takes can still train the audio binary model exactly as before
- reviewed takes can also produce IMU training windows
- if we choose to drop the IMU path later, the audio pipeline still works unchanged

## Rollback Safety

This design must be easy to abandon.

That means:
- no changes to the meaning of current audio review labels
- no breaking changes to the binary audio training path
- the future IMU model is optional, separate, and removable

## Next Build After This Spec

If we implement this spec, the next build should do only:
1. add the new synchronized collector mode
2. save audio + IMU together
3. keep the same audio review flow
4. not yet train or deploy the IMU bounce model live

That keeps the change small and testable.
