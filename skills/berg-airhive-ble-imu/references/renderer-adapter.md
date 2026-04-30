# Renderer Adapter

The renderer consumes an already-calibrated display quaternion. The BLE parser and orientation estimator must not depend on the rendered object.

## Contract

```text
displayQuaternion -> renderer adapter -> object transform
```

Use an adapter equivalent to:

```kotlin
interface RendererAdapter {
    fun render(displayQuaternion: Quaternionf)
}
```

## Separation

- `absoluteQuaternion` comes from the estimator.
- `displayQuaternion` comes from pose calibration.
- `modelCorrection` is renderer-specific.
- Mesh import corrections do not belong in BLE parsing or orientation math.

## Common Pattern

```text
finalObjectRotation = displayQuaternion * modelCorrection
```

If the object looks rotated incorrectly while debug quaternion values look correct, adjust `modelCorrection` or engine conversion code, not the sensor parser.

## Engine Notes

- OpenGL ES or Filament: convert quaternion to matrix and compose with translation and scale.
- Unity: convert handedness and axis conventions explicitly.
- SceneKit or RealityKit: convert to framework quaternion type and keep fixed corrections separate.
- Three.js: convert order and handedness carefully and keep mesh correction separate.
