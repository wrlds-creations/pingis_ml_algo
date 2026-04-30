---
name: berg-airhive-ble-imu
description: Preserve and reuse validated BERG AirHive BLE IMU integration knowledge, including BLE service and characteristic UUIDs, packet parsing, accelerometer, gyroscope, magnetometer unit contracts, magnetometer sign and scaling rules, sample emission, quaternion orientation baseline, pose calibration, renderer adapter, rolling chart debug behavior, and known pitfalls. Use when building, porting, debugging, or reviewing BERG AirHive sensor integrations.
---

# BERG AirHive BLE IMU

Use this skill to reproduce the validated BERG AirHive BLE IMU pipeline. Do not reduce it to a generic 3D view skill.

## Pipeline Contract

```text
BLE notification -> parsed IMU vector -> emitted ImuSample -> baseline orientation estimator -> pose calibration -> renderer adapter
```

## Required References

1. Read `references/sensor-protocol.md` for UUIDs, packet layout, unit contracts, magnetometer sign/scaling, and sample emission.
2. Read `references/orientation-baseline.md` for quaternion estimator, timing, pose calibration, and chart behavior.
3. Read `references/renderer-adapter.md` when connecting the display quaternion to a renderer.
4. Read `references/pitfalls.md` before changing AHRS, timing, magnetometer handling, or calibration.

## Baseline Rules

- Preserve baseline behavior unless the user explicitly asks to change it.
- Do not silently upgrade to a different AHRS.
- Do not silently switch to sensor-timestamp timing.
- Do not change magnetometer sign or scaling without calling out the behavioral change.
- Keep pose calibration separate from sensor calibration.
- Keep renderer-specific mesh correction out of the sensor parser and estimator.

## Debug Requirements

During integration, keep a way to inspect:

- Raw accelerometer, gyroscope, and magnetometer values
- Sensor timestamp
- Received timestamp
- Absolute quaternion
- Display quaternion after pose calibration
- Debug Euler angles for display only
- Rolling 10 second chart window

## Output Summary

Summarize:

- Which parts of the BLE pipeline were touched
- Whether baseline contracts were preserved
- Any intentional deviations
- Device or simulator validation performed
- Remaining sensor, timing, or renderer risks
