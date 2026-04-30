# Pitfalls

These are known failure modes from the validated BERG AirHive baseline.

## Do Not Upgrade AHRS Blindly

More advanced fusion can perform worse when scale, sign, timing, or synchronization assumptions are unresolved. Keep the baseline complementary filter unless the user explicitly asks to experiment.

## Do Not Reinterpret Magnetometer Values

The validated magnetometer mapping is:

```kotlin
x = -rawX / 10f
y = -rawY / 10f
z = -rawZ / 10f
```

Changing this means the integration no longer matches the baseline.

## Do Not Use Sensor Timestamp For Baseline Timing

The baseline uses `receivedAtMs` for estimator `dt`. A sensor timestamp synchronization model is a deliberate change, not a silent refactor.

## Do Not Confuse Pose Calibration With Sensor Calibration

Pose calibration stores a current orientation reference for display. It does not remove gyro bias, recalibrate accelerometer scale, or calibrate the magnetometer.

## Do Not Hide Debugging Too Early

Keep raw values, display quaternion, debug Euler angles, timestamps, and rolling charts available during integration.

## Do Not Treat Renderer Problems As Sensor Truth

If debug data is correct but the object looks wrong, fix renderer conversion or model correction first.
