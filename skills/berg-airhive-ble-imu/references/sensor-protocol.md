# Sensor Protocol

This reference preserves the validated BERG AirHive BLE and packet parsing contract.

## BLE UUIDs

| Purpose | UUID |
|---|---|
| Service | `07c80000-07c8-07c8-07c8-07c807c807c8` |
| Accelerometer characteristic | `07c80001-07c8-07c8-07c8-07c807c807c8` |
| Alternate accelerometer characteristic | `07c80203-07c8-07c8-07c8-07c807c807c8` |
| Gyroscope characteristic | `07c80004-07c8-07c8-07c8-07c807c807c8` |
| Magnetometer characteristic | `07c80010-07c8-07c8-07c8-07c807c807c8` |
| CCCD | `00002902-0000-1000-8000-00805f9b34fb` |

## Packet Layout

All three streams use a 9-byte payload with big-endian axis values:

| Bytes | Value |
|---|---|
| `0..1` | signed `Int16` X |
| `2..3` | signed `Int16` Y |
| `4..5` | signed `Int16` Z |
| `6..8` | unsigned 24-bit timestamp |

Example parser:

```kotlin
val buffer = ByteBuffer.wrap(payload).order(ByteOrder.BIG_ENDIAN)
val rawX = buffer.short.toFloat()
val rawY = buffer.short.toFloat()
val rawZ = buffer.short.toFloat()

val b0 = payload[6].toInt() and 0xFF
val b1 = payload[7].toInt() and 0xFF
val b2 = payload[8].toInt() and 0xFF
val timestamp = (b0 shl 16) or (b1 shl 8) or b2
```

## Unit Contracts

### Accelerometer

- Parse signed big-endian vector values.
- Preserve baseline values as emitted by the existing app.
- The baseline charts values directly and treats them as usable physical-ish acceleration values.

### Gyroscope

- Parse signed big-endian vector values.
- Treat values as degrees per second in app-level samples.
- Convert to radians per second only inside the orientation estimator.

### Magnetometer

The validated parser truth is:

```kotlin
Vector3f(
    x = -rawX / 10f,
    y = -rawY / 10f,
    z = -rawZ / 10f
)
```

Rules:

- Invert all three raw magnetometer axes.
- Divide by `10` to produce baseline microtesla values.
- Do not add further axis swaps in the parser.
- Do not compensate for magnetometer issues downstream without first proving the parser contract changed.

## Sample Emission Model

Emit one `ImuSample` every time any characteristic updates:

```text
notification -> update one latest sensor vector -> emit full sample with latest-known accel/gyro/mag
```

The baseline does not synchronize packets across sensor timestamps.

## Android BLE Baseline

- Use `BluetoothLeScanner`.
- Filter out devices resolved as `Unknown`.
- Sort named discovered devices by RSSI.
- Connect with `connectGatt(context, false, callback)` or an LE transport overload when appropriate.
- Discover services after `STATE_CONNECTED`.
- Subscribe to accelerometer, gyroscope, and magnetometer notifications.
- Enable notifications locally and write CCCD descriptors one at a time.
- Request high connection priority after services are discovered.
- Maintain connection ownership through a stable transport owner such as a foreground service when background stability is required.
