---
name: pingis-stroke-detection
description: Work on pingis AirHive IMU collection, stroke detection, bounce-motion experiments, preprocessing, training, model export, and scripts under this skill.
---

# Skill: pingis-stroke-detection

## Trigger
Load this skill when working on:
- ML training or inference for pingpong stroke detection
- IMU data collection, preprocessing, or feature engineering for pingis
- `DataCollectionScreen` or any BLE data recording component
- Any file in `skills/pingis-stroke-detection/scripts/`

## Purpose
Train and deploy a machine learning model that detects table tennis strokes using the BERG AirHive IMU sensor. Two classification problems:

1. **Hit detection**: Did the player make contact with the ball? (hit vs swing_miss vs idle)
2. **Stroke type**: Forehand or backhand? (only relevant after a confirmed hit)

## Architecture Overview

```
BLE Sensor (50Hz)
    │
    ▼
ImuSample stream  ←── same contract as berg-airhive-imu-3d-view skill
    │
    ├─► [DATA COLLECTION] DataCollectionScreen.tsx
    │       Circular buffer (3s) → tap to label → save session JSON
    │
    └─► [INFERENCE] sliding window (800ms, step 200ms)
            │
            ▼
        Feature extraction OR raw window (CNN)
            │
            ▼
        RandomForest (Phase 1) / 1D CNN TFLite (Phase 2)
            │
            ▼
        "HIT forehand" / "MISS" / "IDLE"
```

## Phases

### Phase 1 — Random Forest (start here, 50-200 samples)
- Feature-engineered input: ~42 features per 800ms window
- Fast to train, interpretable, no GPU needed
- Deployed as server API (Flask) during development
- See: `scripts/train_rf.py`

### Phase 2 — 1D CNN + TFLite (500+ samples)
- Raw 40×9 input (no manual feature engineering)
- Exports to `.tflite` for on-device inference in React Native
- See: `scripts/train_cnn.py`

## Key Files
- `references/data-format.md` — Session JSON schema and label taxonomy
- `references/feature-engineering.md` — Feature definitions and formulas
- `references/model-selection.md` — RF vs CNN tradeoffs
- `scripts/preprocess.py` — Window extraction + feature engineering
- `scripts/train_rf.py` — Phase 1 model training
- `scripts/train_cnn.py` — Phase 2 CNN + TFLite export
- `scripts/preprocess_bounce_imu.py` — Build a binary bounce-contact IMU dataset from reviewed audio + IMU takes
- `scripts/train_rf_bounce_imu.py` — Train the first bounce-contact RandomForest
- `scripts/export_bounce_imu_model_json.py` — Export bounce IMU RF to app JSON when data exists
- `scripts/visualize_features.py` — Plot raw sensor data to verify labels
- `scripts/infer_test.py` — Offline inference test
- `assets/DataCollectionScreen.tsx` — React Native data collection component

## Sensor Data Contract
Always match the exact units from `skills/berg-airhive-imu-3d-view/references/sensor-protocol.md`:
- **Accelerometer**: raw Int16 values (no conversion)
- **Gyroscope**: degrees/second (convert to rad/s only in orientation estimator, not here)
- **Magnetometer**: `-rawX/10, -rawY/10, -rawZ/10` microtesla

## Data Storage
- Raw sessions: `data/raw/session_YYYY-MM-DD_NNN.json` (gitignored)
- Processed dataset: `data/processed/dataset.csv` (gitignored)
- Models: `data/models/` — `.tflite` files can be committed, `.pkl`/`.h5` are gitignored

## Python Dependencies
```
numpy pandas scikit-learn matplotlib scipy
tensorflow  # Phase 2 only
joblib      # model serialization
```
Install: `pip install numpy pandas scikit-learn matplotlib scipy joblib`
