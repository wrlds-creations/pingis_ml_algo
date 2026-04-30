# Orientation Baseline

This reference preserves the validated quaternion orientation behavior.

## Core Contracts

```kotlin
data class ImuSample(
    val accelerometer: Vector3f,
    val gyroscopeDps: Vector3f,
    val magnetometerUt: Vector3f,
    val sensorTimestamp: Int,
    val receivedAtMs: Long
)

interface OrientationEstimator {
    fun reset()
    fun update(sample: ImuSample): Quaternionf
}

interface PoseCalibration {
    fun calibrate(currentQuaternion: Quaternionf)
    fun reset()
    fun apply(absoluteQuaternion: Quaternionf): Quaternionf
}
```

## Estimator Type

The baseline estimator is a quaternion complementary filter, not a full AHRS.

Keep these behaviors:

- Integrate gyro each update.
- Compute measured orientation from accelerometer and magnetometer.
- Slerp predicted orientation toward measured orientation.
- Enforce quaternion sign continuity.

## Timing And Constants

- First frame `dt = 1 / 50`
- Later `dt = clamp((receivedAtMs - lastReceivedAtMs) / 1000, 0.001, 0.05)`
- Accelerometer low-pass alpha: `0.15`
- Magnetometer low-pass alpha: `0.10`
- Correction alpha: `1 - exp(-dt / 0.35)`
- Gyro deadband: `angularSpeed < 1e-5` means no delta rotation

Use `receivedAtMs` for `dt`, not the sensor timestamp, unless the user explicitly approves a model change.

## Gyro Integration

Convert degrees per second to radians per second inside the estimator:

```kotlin
val gyroRad = Vector3f(
    x = Math.toRadians(gyroDps.x.toDouble()).toFloat(),
    y = Math.toRadians(gyroDps.y.toDouble()).toFloat(),
    z = Math.toRadians(gyroDps.z.toDouble()).toFloat()
)
```

Integrate by axis-angle:

```kotlin
val angularSpeed = gyroRad.magnitude()
val delta = Quaternionf.fromAxisAngle(gyroRad / angularSpeed, angularSpeed * dt)
val predicted = (current * delta).normalized()
```

## Measured Orientation

1. Normalize accelerometer to get `up`.
2. Remove the `up` component from magnetometer to get horizontal magnetic field.
3. Build `east = horizontalMag.cross(up).normalized()`.
4. Build `north = up.cross(east).normalized()`.
5. Build a rotation matrix from `east`, `north`, and `up`.
6. Convert the matrix to a quaternion and invert it to get sensor orientation.
7. Fall back to gyro-predicted orientation if any vector collapses to zero.

## Complementary Correction

```kotlin
val orientation = predicted.slerp(measured, correctionAlpha)
```

Then keep sign continuity:

```kotlin
if (orientation.dot(previous) < 0f) {
    orientation = orientation.negated()
}
```

## Pose Calibration

Pose calibration is a display-space reference offset.

- Calibrate by storing the current absolute orientation as `referenceOrientation`.
- Display identity immediately after calibration.
- Apply calibration with `displayOrientation = inverse(referenceOrientation) * absoluteOrientation`.
- Reset by clearing `referenceOrientation`; keep estimator state intact.

## Rolling Chart Debug Behavior

Use a rolling 10 second sample window:

```text
append new sample -> drop samples older than latestReceivedAtMs - 10000
```

Charts are recommended during integration because they expose parser, axis, timing, and calibration errors quickly.
