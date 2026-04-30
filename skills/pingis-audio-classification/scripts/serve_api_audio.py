"""
serve_api_audio.py

Flask-API för att klassificera pingisbollsstudsar från ljuddata.
Kör på port 5001 (IMU-API kör på 5000).

Start: python skills/pingis-audio-classification/scripts/serve_api_audio.py

Endpoints:
  GET  /health_audio         → modellstatus
  GET  /labels_audio         → tillgängliga klasser
  POST /predict_audio        → klassificera ett base64-kodat m4a/wav-klipp
"""

import base64
import os
import tempfile
from pathlib import Path

import joblib
import numpy as np
from flask import Flask, jsonify, request

# Lazy-import: preprocess_audio behöver installerade deps
import sys
sys.path.insert(0, str(Path(__file__).parent))
from preprocess_audio import extract_features, load_audio

# ── Sökvägar & modell ─────────────────────────────────────────────────────────

ROOT_DIR  = Path(__file__).resolve().parents[3]
MODEL_DIR = ROOT_DIR / "data" / "audio" / "models"

app = Flask(__name__)

clf    = None
scaler = None
le     = None


def load_models() -> bool:
    global clf, scaler, le
    try:
        clf    = joblib.load(MODEL_DIR / "audio_rf_classifier.pkl")
        scaler = joblib.load(MODEL_DIR / "audio_feature_scaler.pkl")
        le     = joblib.load(MODEL_DIR / "audio_label_encoder.pkl")
        return True
    except FileNotFoundError as e:
        print(f"Modell saknas: {e}")
        return False


# ── Endpoints ─────────────────────────────────────────────────────────────────

@app.get("/health_audio")
def health():
    ok = clf is not None
    return jsonify({"status": "ok" if ok else "model_not_loaded", "model_loaded": ok})


@app.get("/labels_audio")
def labels():
    if le is None:
        return jsonify({"error": "model not loaded"}), 503
    return jsonify({"labels": le.classes_.tolist()})


@app.post("/predict_audio")
def predict():
    if clf is None:
        return jsonify({"error": "model not loaded"}), 503

    data = request.get_json(force=True, silent=True) or {}
    audio_b64 = data.get("audio_b64")
    if not audio_b64:
        return jsonify({"error": "audio_b64 saknas"}), 400

    # Avkoda base64 → tempfil → extrahera features
    try:
        audio_bytes = base64.b64decode(audio_b64)
    except Exception:
        return jsonify({"error": "Ogiltig base64"}), 400

    suffix = ".m4a"  # librosa hanterar m4a via soundfile/ffmpeg
    with tempfile.NamedTemporaryFile(suffix=suffix, delete=False) as tmp:
        tmp.write(audio_bytes)
        tmp_path = tmp.name

    try:
        y, sr = load_audio(tmp_path)
        feats = extract_features(y, sr)
    except Exception as e:
        return jsonify({"error": f"Feature-extraktion misslyckades: {e}"}), 422
    finally:
        os.unlink(tmp_path)

    # Bygg feature-vektor i samma ordning som träning (drop metadata-kolumner)
    feat_vals = np.array(list(feats.values()), dtype=np.float32).reshape(1, -1)

    feat_scaled = scaler.transform(feat_vals)
    proba       = clf.predict_proba(feat_scaled)[0]
    pred_idx    = int(np.argmax(proba))
    pred_label  = le.inverse_transform([pred_idx])[0]

    return jsonify({
        "label":         pred_label,
        "confidence":    round(float(proba[pred_idx]), 4),
        "probabilities": {
            str(le.inverse_transform([i])[0]): round(float(p), 4)
            for i, p in enumerate(proba)
        },
    })


# ── Main ──────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    if not load_models():
        print("Varning: inga modeller laddade — kör train_rf_audio.py först.")
    app.run(host="0.0.0.0", port=5001, debug=False)
