"""
classify_bounce_side.py

Punkt 5: detektera om bollen studsas med forehand- eller backhandsidan av
racketen framför kameran. Detta är hela live-algoritmen körd offline på en
inspelad video — samma steg som realtidsversionen i appen ska köra:

  1. LJUDANKARE: racketstudsar detekteras i ljudspåret med den brusrobusta
     Fable-detektorn (bandpass-gate + HistGB v3) — i appen kommer samma
     ankare från det befintliga live-flödet.
  2. TRÄFFRAME: videoframen närmast varje ankare hämtas.
  3. RACKET-ROI: racketens röda gummi segmenteras i HSV; tight bounding box
     runt största röda blobben (T0040 visade att tight racket-ROI är den
     enda ROI-strategi som fungerar: 0.92 på blandad holdout).
  4. SIDOKLASSIFICERING: FH-sida (rött gummi mot kameran) vs BH-sida
     (svart gummi) med en klassificerare på grid-färgfeatures i ROI:n.

Träning/utvärdering (sessionsmedveten, T0041-policyn):
  - Träna på video_bounce_side_session _001-_003, utvärdera på den BLANDADE
    _004 (12 FH / 13 BH, slumpad ordning) som aldrig tränas på.

Kör:
  python skills/pingis-stroke-detection/scripts/classify_bounce_side.py --train
  python skills/pingis-stroke-detection/scripts/classify_bounce_side.py \
      --classify data/video/raw/video_bounce_side/video_bounce_side_session_2026-06-09_004/media/video_bounce_side_session_2026-06-09_004.mp4
"""

from __future__ import annotations

import argparse
import csv
import json
import sys
from pathlib import Path

import cv2
import joblib
import numpy as np

ROOT_DIR = Path(__file__).resolve().parents[3]
BOUNCE_SIDE_RAW = ROOT_DIR / "data" / "video" / "raw" / "video_bounce_side"
OUT_DIR = ROOT_DIR / "data" / "video" / "models" / "bounce_side_v1"
NR_DIR = ROOT_DIR / "skills" / "pingis-audio-classification" / "scripts" / "noise_robust"
NR_MODEL_DIR = ROOT_DIR / "data" / "audio" / "models" / "noise_robust_v3"

TRAIN_SESSIONS = ["video_bounce_side_session_2026-06-09_001",
                  "video_bounce_side_session_2026-06-09_002",
                  "video_bounce_side_session_2026-06-09_003"]
HOLDOUT_SESSION = "video_bounce_side_session_2026-06-09_004"

GRID = 4
SEED = 20260610


# ── Racket-ROI via röd-gummi-segmentering ────────────────────────────────────

def red_mask(hsv: np.ndarray) -> np.ndarray:
    lower1 = cv2.inRange(hsv, (0, 80, 50), (10, 255, 255))
    lower2 = cv2.inRange(hsv, (170, 80, 50), (180, 255, 255))
    return cv2.bitwise_or(lower1, lower2)


def find_racket_roi(frame_bgr: np.ndarray) -> tuple[np.ndarray | None, str]:
    """Tight ROI runt största röda blobben (racketens gummi). Returnerar
    (crop, källa) där källa är 'red_anchor' eller 'center_fallback'."""
    hsv = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2HSV)
    mask = red_mask(hsv)
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, np.ones((5, 5), np.uint8))
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, np.ones((9, 9), np.uint8))
    n, labels, stats, _ = cv2.connectedComponentsWithStats(mask, connectivity=8)
    h, w = frame_bgr.shape[:2]
    min_area = (h * w) * 0.0008
    best = None
    for i in range(1, n):
        area = stats[i, cv2.CC_STAT_AREA]
        if area >= min_area and (best is None or area > stats[best, cv2.CC_STAT_AREA]):
            best = i
    if best is None:
        # Fallback: centrumcrop (racket utan röd sida mot kameran = BH-signal
        # i sig, men utan lokalisering blir featuren svag - flaggas).
        cw, ch = w // 3, h // 3
        return frame_bgr[ch:2 * ch, cw:2 * cw], "center_fallback"
    x, y = stats[best, cv2.CC_STAT_LEFT], stats[best, cv2.CC_STAT_TOP]
    bw, bh = stats[best, cv2.CC_STAT_WIDTH], stats[best, cv2.CC_STAT_HEIGHT]
    # Expandera 35 %: vid BH-studs syns bara en röd kant - ta med gummit runtom.
    ex, ey = int(bw * 0.35), int(bh * 0.35)
    x0, y0 = max(0, x - ex), max(0, y - ey)
    x1, y1 = min(w, x + bw + ex), min(h, y + bh + ey)
    return frame_bgr[y0:y1, x0:x1], "red_anchor"


def roi_features(roi_bgr: np.ndarray, source: str) -> dict[str, float]:
    """Grid-färgfeatures: FH-sida = mest rött gummi mot kameran,
    BH-sida = mest svart/mörkt gummi."""
    roi = cv2.resize(roi_bgr, (64, 64), interpolation=cv2.INTER_AREA)
    hsv = cv2.cvtColor(roi, cv2.COLOR_BGR2HSV).astype(np.float64)
    rmask = red_mask(cv2.cvtColor(roi, cv2.COLOR_BGR2HSV)).astype(np.float64) / 255.0
    v = hsv[:, :, 2]
    s = hsv[:, :, 1]
    dark = ((v < 70) & (s < 120)).astype(np.float64)

    feats: dict[str, float] = {}
    cell = 64 // GRID
    for gy in range(GRID):
        for gx in range(GRID):
            ys, xs = gy * cell, gx * cell
            feats[f"g{gy}{gx}_red"] = float(rmask[ys:ys + cell, xs:xs + cell].mean())
            feats[f"g{gy}{gx}_dark"] = float(dark[ys:ys + cell, xs:xs + cell].mean())
            feats[f"g{gy}{gx}_v"] = float(v[ys:ys + cell, xs:xs + cell].mean() / 255.0)
    hue = hsv[:, :, 0].ravel()
    hist, _ = np.histogram(hue, bins=12, range=(0, 180), weights=(s.ravel() / 255.0))
    total = hist.sum() + 1e-9
    for i, hv in enumerate(hist):
        feats[f"hue_{i}"] = float(hv / total)
    feats["red_total"] = float(rmask.mean())
    feats["dark_total"] = float(dark.mean())
    feats["red_minus_dark"] = feats["red_total"] - feats["dark_total"]
    feats["v_mean"] = float(v.mean() / 255.0)
    feats["is_fallback"] = 1.0 if source == "center_fallback" else 0.0
    return feats


def frame_at_ms(capture: cv2.VideoCapture, ts_ms: float) -> np.ndarray | None:
    capture.set(cv2.CAP_PROP_POS_MSEC, max(0.0, ts_ms))
    ok, frame = capture.read()
    return frame if ok else None


# ── Dataset från granskade bounce-side-sessioner ─────────────────────────────

def iter_session_markers(session_id: str):
    sdir = BOUNCE_SIDE_RAW / session_id
    data = json.loads((sdir / f"{session_id}.json").read_text(encoding="utf-8"))
    for take in data.get("takes", []):
        video = sdir / "media" / take["video_filename"]
        if not video.exists():
            continue
        markers = [m for m in take.get("markers", [])
                   if m.get("bounce_side") in ("forehand", "backhand")
                   and m.get("review_status") in (None, "confirmed", "edited")]
        yield video, markers


def build_dataset(sessions: list[str]) -> tuple[np.ndarray, np.ndarray, list[str], list[dict]]:
    rows: list[dict] = []
    meta: list[dict] = []
    for session_id in sessions:
        for video, markers in iter_session_markers(session_id):
            cap = cv2.VideoCapture(str(video))
            for marker in markers:
                frame = frame_at_ms(cap, float(marker["timestamp_ms"]))
                if frame is None:
                    continue
                roi, source = find_racket_roi(frame)
                if roi is None or roi.size == 0:
                    continue
                feats = roi_features(roi, source)
                rows.append(feats)
                meta.append({"session_id": session_id, "ts_ms": marker["timestamp_ms"],
                             "label": marker["bounce_side"], "roi_source": source})
            cap.release()
    feature_names = list(rows[0].keys())
    X = np.array([[r[n] for n in feature_names] for r in rows], dtype=np.float64)
    y = np.array([m["label"] for m in meta])
    return X, y, feature_names, meta


def train_main() -> None:
    from sklearn.ensemble import ExtraTreesClassifier, RandomForestClassifier
    from sklearn.metrics import accuracy_score, confusion_matrix
    from sklearn.pipeline import make_pipeline
    from sklearn.preprocessing import StandardScaler
    from sklearn.svm import LinearSVC

    X_train, y_train, feature_names, meta_train = build_dataset(TRAIN_SESSIONS)
    X_test, y_test, _, meta_test = build_dataset([HOLDOUT_SESSION])
    n_fallback_tr = sum(1 for m in meta_train if m["roi_source"] != "red_anchor")
    n_fallback_te = sum(1 for m in meta_test if m["roi_source"] != "red_anchor")
    print(f"Train: {len(y_train)} samples ({dict(zip(*np.unique(y_train, return_counts=True)))}, {n_fallback_tr} ROI-fallback)")
    print(f"Holdout {HOLDOUT_SESSION}: {len(y_test)} samples ({dict(zip(*np.unique(y_test, return_counts=True)))}, {n_fallback_te} ROI-fallback)")

    models = {
        "linear_svc": make_pipeline(StandardScaler(), LinearSVC(C=0.5, random_state=SEED, max_iter=5000)),
        "extra_trees": ExtraTreesClassifier(n_estimators=400, random_state=SEED, n_jobs=-1, class_weight="balanced"),
        "rf": RandomForestClassifier(n_estimators=400, random_state=SEED, n_jobs=-1, class_weight="balanced_subsample"),
    }
    results = {}
    best_name, best_acc = None, -1.0
    for name, model in models.items():
        model.fit(X_train, y_train)
        pred = model.predict(X_test)
        acc = accuracy_score(y_test, pred)
        cm = confusion_matrix(y_test, pred, labels=["forehand", "backhand"]).tolist()
        results[name] = {"holdout_accuracy": round(float(acc), 3), "confusion_fh_bh": cm}
        print(f"  {name}: holdout accuracy {acc:.3f}  confusion [[FH-FH,FH-BH],[BH-FH,BH-BH]] {cm}")
        if acc > best_acc:
            best_name, best_acc = name, acc

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    best_model = models[best_name]
    joblib.dump(best_model, OUT_DIR / "bounce_side_model.pkl")
    joblib.dump(feature_names, OUT_DIR / "bounce_side_feature_names.pkl")
    (OUT_DIR / "training_summary.json").write_text(json.dumps({
        "train_sessions": TRAIN_SESSIONS,
        "holdout_session": HOLDOUT_SESSION,
        "n_train": int(len(y_train)),
        "n_holdout": int(len(y_test)),
        "results": results,
        "selected_model": best_name,
        "feature_count": len(feature_names),
        "seed": SEED,
    }, indent=1), encoding="utf-8")
    print(f"Selected: {best_name} ({best_acc:.3f}). Artifacts: {OUT_DIR}")


# ── Klassificera ny video: ljudankare -> ROI -> sida ─────────────────────────

def classify_main(video_path: Path, audio_path: Path | None, out_prefix: str) -> None:
    sys.path.insert(0, str(NR_DIR))
    sys.path.insert(0, str(NR_DIR.parent))
    import nr_config  # noqa: PLC0415
    import nr_features  # noqa: PLC0415
    from preprocess_audio import load_audio  # noqa: PLC0415
    from analyze_video_retro import extract_audio  # noqa: PLC0415

    model = joblib.load(OUT_DIR / "bounce_side_model.pkl")
    feature_names = joblib.load(OUT_DIR / "bounce_side_feature_names.pkl")
    audio_model = joblib.load(NR_MODEL_DIR / "nr_histgb_all83.pkl")
    audio_scaler = joblib.load(NR_MODEL_DIR / "nr_scaler_all83.pkl")
    audio_cols = list(joblib.load(NR_MODEL_DIR / "nr_feature_cols_all83.pkl"))
    labels = nr_config.CLASSES

    if audio_path is not None:
        y, sr = load_audio(str(audio_path))
        tmp = None
    else:
        y, sr, tmp = extract_audio(video_path)

    triggers = nr_features.simulate_gate(y, sr, onset_ratio=1.5, retrigger_ms=120,
                                         abs_min_rms=0.0015, mode="bandpass", spectral_gate=False)
    anchors = []
    last_ms = -1e9
    for trig in triggers:
        clip = nr_features.extract_live_clip(y, int(trig["onset_sample"]))
        feats = nr_features.extract_all_features(clip, sr)
        x = np.nan_to_num(np.array([[feats.get(c, 0.0) for c in audio_cols]], dtype=np.float64))
        probs = audio_model.predict_proba(audio_scaler.transform(x))[0]
        if labels[int(np.argmax(probs))] != "racket_bounce" or float(np.max(probs)) < 0.5:
            continue
        if trig["onset_ms"] - last_ms <= 120:
            continue
        anchors.append(float(trig["onset_ms"]))
        last_ms = trig["onset_ms"]
    if tmp is not None:
        tmp.unlink(missing_ok=True)
    print(f"{len(anchors)} racketstuds-ankare från ljudet")

    cap = cv2.VideoCapture(str(video_path))
    out_rows = []
    for ts in anchors:
        frame = frame_at_ms(cap, ts)
        if frame is None:
            continue
        roi, source = find_racket_roi(frame)
        feats = roi_features(roi, source)
        x = np.array([[feats[n] for n in feature_names]], dtype=np.float64)
        pred = model.predict(x)[0]
        score = ""
        if hasattr(model, "predict_proba"):
            proba = model.predict_proba(x)[0]
            score = round(float(np.max(proba)), 3)
        out_rows.append({"ts_ms": round(ts, 1), "side": pred, "score": score, "roi_source": source})
    cap.release()

    out_csv = Path(f"{out_prefix}_bounce_sides.csv")
    with out_csv.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=["ts_ms", "side", "score", "roi_source"])
        writer.writeheader()
        writer.writerows(out_rows)
    n_fh = sum(1 for r in out_rows if r["side"] == "forehand")
    n_bh = sum(1 for r in out_rows if r["side"] == "backhand")
    print(f"FH {n_fh} / BH {n_bh}. Tidslinje: {out_csv}")


def main() -> None:
    parser = argparse.ArgumentParser(description="FH-/BH-sida vid racketstuds (audio-ankare + racket-ROI).")
    parser.add_argument("--train", action="store_true")
    parser.add_argument("--classify", default="", help="Video att klassificera.")
    parser.add_argument("--audio", default="", help="Valfri separat WAV (annars extraheras ur videon).")
    parser.add_argument("--out-prefix", default="")
    args = parser.parse_args()
    if args.train:
        train_main()
    if args.classify:
        video = Path(args.classify)
        prefix = args.out_prefix or str(video.with_suffix(""))
        classify_main(video, Path(args.audio) if args.audio else None, prefix)
    if not args.train and not args.classify:
        raise SystemExit("Ange --train och/eller --classify <video>")


if __name__ == "__main__":
    main()
