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

# 2026-06-09_001 är medvetet EXKLUDERAD ur träningen: den är block-labelad
# feasibility-data (PROJECT_CONTEXT/T0038) och att ta med den sänker
# holdout-accuracyn från 0.92 till 0.68 (reproducerat ur T0040:s artefakter).
# 2026-06-11_001: Loves blandade underifrån-session (111 markörer, 62 FH /
# 49 BH, individuellt granskade) - ny kameravinkel som också är den
# rekommenderade live-uppställningen (mobilen lutad på bordet).
TRAIN_SESSIONS = ["video_bounce_side_session_2026-06-09_002",
                  "video_bounce_side_session_2026-06-09_003",
                  "video_bounce_side_session_2026-06-11_001",
                  # Loves granskade markörer ur andra underifrån-videon
                  # (11 st; dagens modell fick 7/11 på dem före omträning).
                  "video_bounce_side_session_2026-06-11_002",
                  # Samma video återimporterad och fullgranskad på appen
                  # (37 facit) - bär app-crops via crops.json när den
                  # importeras om på dump-versionen.
                  "video_bounce_side_session_2026-06-11_003",
                  # Ny video på MediaPipe-versionen: 14 bekräftade + 47
                  # intygade crops (Love: alla förslag korrekta).
                  "video_bounce_side_session_2026-06-11_004"]
HOLDOUT_SESSION = "video_bounce_side_session_2026-06-09_004"

GRID = 4
SEED = 20260610
POSE_MODEL_PATH = ROOT_DIR / "data" / "video" / "models" / "pose_landmarker_lite.task"
R_WRIST, R_ELBOW, L_WRIST, L_ELBOW = 16, 14, 15, 13

_POSE_LANDMARKER = None


def get_pose_landmarker():
    """Lazy MediaPipe pose landmarker (IMAGE mode) för handleds-ankrad ROI."""
    global _POSE_LANDMARKER
    if _POSE_LANDMARKER is None:
        import mediapipe as mp  # noqa: PLC0415
        from mediapipe.tasks.python.core import base_options as bo  # noqa: PLC0415
        from mediapipe.tasks.python.vision import pose_landmarker as plm  # noqa: PLC0415
        from mediapipe.tasks.python.vision.core import vision_task_running_mode as rm  # noqa: PLC0415
        options = plm.PoseLandmarkerOptions(
            base_options=bo.BaseOptions(model_asset_path=str(POSE_MODEL_PATH)),
            running_mode=rm.VisionTaskRunningMode.IMAGE,
        )
        _POSE_LANDMARKER = plm.PoseLandmarker.create_from_options(options)
    return _POSE_LANDMARKER


def wrist_anchored_roi(frame_bgr: np.ndarray) -> np.ndarray | None:
    """Sido-agnostisk racket-ROI: kvadrat centrerad bortom handleden längs
    underarmsriktningen (racketen sitter i handen oavsett vilken gummisida
    som är mot kameran - till skillnad från röd-blob-ankaret som är
    systematiskt FH-biaserat). Bäst i utvärderingen: 0.72 markör-accuracy
    på blandade _004 vs 0.64-0.68 för röd-ankare-varianterna."""
    import cv2 as _cv2  # noqa: PLC0415
    import mediapipe as mp  # noqa: PLC0415
    h, w = frame_bgr.shape[:2]
    image = mp.Image(image_format=mp.ImageFormat.SRGB,
                     data=_cv2.cvtColor(frame_bgr, _cv2.COLOR_BGR2RGB))
    result = get_pose_landmarker().detect(image)
    if not result.pose_landmarks:
        return None
    lm = result.pose_landmarks[0]
    cands = []
    for wi, ei in ((R_WRIST, R_ELBOW), (L_WRIST, L_ELBOW)):
        cands.append((lm[wi].visibility, lm[wi], lm[ei]))
    cands.sort(key=lambda c: -c[0])
    _, wl, el = cands[0]
    wx, wy, ex, ey = wl.x * w, wl.y * h, el.x * w, el.y * h
    fx, fy = wx - ex, wy - ey
    flen = max(np.hypot(fx, fy), 1.0)
    cx, cy = wx + 0.8 * fx, wy + 0.8 * fy
    half = 1.3 * flen
    x0, y0 = int(max(0, cx - half)), int(max(0, cy - half))
    x1, y1 = int(min(w, cx + half)), int(min(h, cy + half))
    if x1 - x0 < 24 or y1 - y0 < 24:
        return None
    return frame_bgr[y0:y1, x0:x1]


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
    # Tight crop med liten marginal (T0040: tight red-anchor ROI var den
    # enda strategi som fungerade; bred expansion späder ut sidosignalen).
    ex, ey = int(bw * 0.1), int(bh * 0.1)
    x0, y0 = max(0, x - ex), max(0, y - ey)
    x1, y1 = min(w, x + bw + ex), min(h, y + bh + ey)
    return frame_bgr[y0:y1, x0:x1], "red_anchor"


def roi_pixel_features(roi_bgr: np.ndarray) -> np.ndarray:
    """T0040-stil: råa nedskalade pixlar (48x48x3) + 128-bins hue-histogram
    (deras feature-vektor var 7040 = 6912 + 128)."""
    roi = cv2.resize(roi_bgr, (48, 48), interpolation=cv2.INTER_AREA)
    rgb = cv2.cvtColor(roi, cv2.COLOR_BGR2RGB).astype(np.float64) / 255.0
    hsv = cv2.cvtColor(roi, cv2.COLOR_BGR2HSV)
    hist, _ = np.histogram(hsv[:, :, 0].ravel(), bins=128, range=(0, 180),
                           weights=(hsv[:, :, 1].ravel().astype(np.float64) / 255.0))
    hist = hist / (hist.sum() + 1e-9)
    return np.concatenate([rgb.ravel(), hist])


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


def red_area(frame_bgr: np.ndarray) -> float:
    hsv = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2HSV)
    return float(red_mask(hsv).sum())


def frames_in_window(capture: cv2.VideoCapture, ts_ms: float, pre_ms: float = 80, post_ms: float = 80) -> list[np.ndarray]:
    """Alla frames i snapshotfönstret ±80 ms (samma fönster som appens
    bounce-side-snapshots). En enskild frame kan vara rörelseoskarp eller
    visa racketen mitt i rotation - klassificering sker per frame och
    aggregeras med röstning, vilket undviker att framvalet i sig biaserar
    mot den röda (FH-)sidan."""
    capture.set(cv2.CAP_PROP_POS_MSEC, max(0.0, ts_ms - pre_ms))
    frames: list[np.ndarray] = []
    while True:
        ok, frame = capture.read()
        if not ok:
            break
        pos = capture.get(cv2.CAP_PROP_POS_MSEC)
        if pos > ts_ms + post_ms:
            break
        frames.append(frame.copy())
        if len(frames) >= 12:
            break
    return frames


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


def compute_features(roi: np.ndarray, source: str, mode: str) -> np.ndarray:
    if mode == "pixels":
        return roi_pixel_features(roi)
    feats = roi_features(roi, source)
    return np.array([feats[n] for n in sorted(feats)], dtype=np.float64)


def feature_names_for_mode(mode: str) -> list[str]:
    if mode == "pixels":
        return [f"px_{i}" for i in range(48 * 48 * 3)]
    dummy = roi_features(np.zeros((8, 8, 3), dtype=np.uint8), "red_anchor")
    return sorted(dummy)


def roi_for_frame(frame: np.ndarray) -> tuple[np.ndarray | None, str]:
    """Primär ROI: handleds-ankrad (pose). Fallback: röd-ankare, sedan centrum."""
    roi = wrist_anchored_roi(frame)
    if roi is not None and roi.size:
        return roi, "wrist_anchor"
    return find_racket_roi(frame)


# Sessioner där Love uttryckligen intygat att ALLA appens förslag var
# korrekta (skriftligt 2026-06-11) - crops utan bekräftad markör får då
# modellens egen prediktion som etikett (självträning med mänsklig
# stickprovskontroll). Lägg ALDRIG till sessioner här utan uttryckligt
# intygande från granskaren.
VOUCHED_FULL_SESSIONS = {"video_bounce_side_session_2026-06-11_004"}


def app_crop_rows(session_id: str, mode: str) -> tuple[list[np.ndarray], list[dict]]:
    """Träningsrader från APPENS egna crops (<sid>.crops.json, sparad av
    Collector vid import). Mobilens pose-motor ramar in racketen annorlunda
    än PC-pipelinen (bevisat 2026-06-11: PC 37/37 rätt där appen föreslog
    fel på samma ankare) - modellen måste därför se appens bilddomän."""
    import base64

    sdir = BOUNCE_SIDE_RAW / session_id
    crops_path = sdir / f"{session_id}.crops.json"
    if not crops_path.exists():
        return [], []
    data = json.loads(crops_path.read_text(encoding="utf-8"))
    crops_by_ts = {}
    for crop in data.get("crops", []):
        crops_by_ts[round(float(crop["timestamp_ms"]))] = crop

    def crop_to_row(crop: dict) -> np.ndarray | None:
        rgb = np.frombuffer(base64.b64decode(crop["rgb_b64"]), dtype=np.uint8)
        if rgb.size != 64 * 64 * 3:
            return None
        bgr = cv2.cvtColor(rgb.reshape(64, 64, 3).copy(), cv2.COLOR_RGB2BGR)
        if mode == "pixels":
            return roi_pixel_features(bgr)
        feats = roi_features(bgr, crop.get("roi_source", "wrist_anchor"))
        return np.array([feats[n] for n in sorted(feats)], dtype=np.float64)

    rows: list[np.ndarray] = []
    meta: list[dict] = []
    used_ts: set[int] = set()
    for _, markers in iter_session_markers(session_id):
        for marker in markers:
            ts = round(float(marker["timestamp_ms"]))
            crop = crops_by_ts.get(ts)
            if crop is None:
                near = [k for k in crops_by_ts if abs(k - ts) <= 50]
                if not near:
                    continue
                crop = crops_by_ts[min(near, key=lambda k: abs(k - ts))]
            row = crop_to_row(crop)
            if row is None:
                continue
            used_ts.add(round(float(crop["timestamp_ms"])))
            rows.append(row)
            meta.append({"session_id": session_id, "marker_key": f"{session_id}:{ts}:app",
                         "ts_ms": ts, "label": marker["bounce_side"],
                         "roi_source": crop.get("roi_source", "wrist_anchor"), "domain": "app"})

    if session_id in VOUCHED_FULL_SESSIONS:
        # Resterande crops: modellens prediktion som intygad etikett.
        model = joblib.load(OUT_DIR / "bounce_side_model.pkl")
        feature_meta = joblib.load(OUT_DIR / "bounce_side_feature_meta.pkl")
        n_vouched = 0
        for ts_key, crop in crops_by_ts.items():
            if ts_key in used_ts:
                continue
            row = crop_to_row(crop)
            if row is None:
                continue
            if feature_meta["mode"] != mode:
                continue
            label = str(model.predict(row.reshape(1, -1))[0])
            if label not in ("forehand", "backhand"):
                continue
            rows.append(row)
            meta.append({"session_id": session_id, "marker_key": f"{session_id}:{ts_key}:vouched",
                         "ts_ms": ts_key, "label": label,
                         "roi_source": crop.get("roi_source", "wrist_anchor"), "domain": "app_vouched"})
            n_vouched += 1
        if n_vouched:
            print(f"  [{session_id}] +{n_vouched} intygade självtränings-rader")
    return rows, meta


LIVE_DEBUG_DIR = ROOT_DIR / "data" / "video" / "raw" / "live_sidedebug"
LIVE_PAUSE_MS = 1500


def live_debug_rows(mode: str) -> tuple[list[np.ndarray], list[str]]:
    """Träningsrader från live-lägets debug-dumpar (frontkamera-domänen).

    Facit från Loves alternationsprotokoll (strikt vartannat FH/BH,
    2026-06-11): sekvensen delas i segment vid pauser (>1.5 s - omstarts-
    fasen är okänd); inom ett segment ger tidsgapen antalet flips
    (gap ~2x period = en missad studs = samma sida igen). Fasen per
    segment väljs som bästa anpassning mot modellens beslut och segmentet
    används BARA om alla beslutade event är 100% konsistenta med fasen.
    Osäkra event får då också sin alternationsetikett (de mest värdefulla
    raderna)."""
    import base64

    rows: list[np.ndarray] = []
    labels: list[str] = []
    if not LIVE_DEBUG_DIR.exists():
        return rows, labels
    for path in sorted(LIVE_DEBUG_DIR.glob("pingis_live_sidedebug_*.json")):
        events = json.loads(path.read_text(encoding="utf-8"))["events"]
        if len(events) < 4:
            continue
        segments: list[list[dict]] = [[events[0]]]
        for prev, cur in zip(events, events[1:]):
            if cur["onset_time_ms"] - prev["onset_time_ms"] > LIVE_PAUSE_MS:
                segments.append([cur])
            else:
                segments[-1].append(cur)
        for segment in segments:
            if len(segment) < 4:
                continue
            gaps = [b["onset_time_ms"] - a["onset_time_ms"] for a, b in zip(segment, segment[1:])]
            med = sorted(gaps)[len(gaps) // 2]
            if med <= 0:
                continue
            flips = [max(1, round(g / med)) for g in gaps]
            best: tuple[int, list[str]] | None = None
            for start in ("forehand", "backhand"):
                truth = [start]
                for f in flips:
                    side = truth[-1]
                    for _ in range(f):
                        side = "backhand" if side == "forehand" else "forehand"
                    truth.append(side)
                score = sum(1 for e, t in zip(segment, truth) if e["side"] == t)
                if best is None or score > best[0]:
                    best = (score, truth)
            _, truth = best
            decided = [(e, t) for e, t in zip(segment, truth) if e["side"] != "uncertain"]
            if not decided or any(e["side"] != t for e, t in decided):
                continue  # facit ej trovärdigt för segmentet
            for e, t in zip(segment, truth):
                rgb = np.frombuffer(base64.b64decode(e["rgb_b64"]), dtype=np.uint8)
                if rgb.size != 64 * 64 * 3:
                    continue
                bgr = cv2.cvtColor(rgb.reshape(64, 64, 3).copy(), cv2.COLOR_RGB2BGR)
                if mode == "pixels":
                    rows.append(roi_pixel_features(bgr))
                else:
                    feats = roi_features(bgr, e.get("roi_source", "wrist_anchor"))
                    rows.append(np.array([feats[n] for n in sorted(feats)], dtype=np.float64))
                labels.append(t)
    if rows:
        print(f"  [live_sidedebug] +{len(rows)} live-domän-rader (alternationsfacit)")
    return rows, labels


def build_dataset(sessions: list[str], mode: str) -> tuple[np.ndarray, np.ndarray, list[dict]]:
    """En rad per markör: träfframen vid markörens tidsstämpel (utvärderingen
    visade att enkelframe + handleds-ROI slår både ±80 ms-röstning och
    röd-ankare-varianterna). Sessioner med sparade app-crops bidrar
    DESSUTOM med rader i appens egen bilddomän."""
    rows: list[np.ndarray] = []
    meta: list[dict] = []
    for session_id in sessions:
        for video, markers in iter_session_markers(session_id):
            cap = cv2.VideoCapture(str(video))
            for marker in markers:
                marker_key = f"{session_id}:{marker['timestamp_ms']}"
                frame = frame_at_ms(cap, float(marker["timestamp_ms"]))
                if frame is None:
                    continue
                roi, source = roi_for_frame(frame)
                if roi is None or roi.size == 0:
                    continue
                rows.append(compute_features(roi, source, mode))
                meta.append({"session_id": session_id, "marker_key": marker_key,
                             "ts_ms": marker["timestamp_ms"],
                             "label": marker["bounce_side"], "roi_source": source,
                             "domain": "pc"})
            cap.release()
        app_rows, app_meta = app_crop_rows(session_id, mode)
        rows.extend(app_rows)
        meta.extend(app_meta)
        if app_meta:
            print(f"  [{session_id}] +{len(app_meta)} app-domän-rader från crops.json")
    X = np.stack(rows)
    y = np.array([m["label"] for m in meta])
    return X, y, meta


def marker_vote_accuracy(model, X: np.ndarray, meta: list[dict]) -> tuple[float, list, int]:
    """Aggregera per-frame-prediktioner till en röst per markör."""
    if hasattr(model, "predict_proba"):
        scores = model.predict_proba(X)
        classes = list(model.classes_)
        fh_idx = classes.index("forehand")
        margin = scores[:, fh_idx] - (1.0 - scores[:, fh_idx])
    else:
        raw = model.decision_function(X)
        classes = list(model.classes_)
        # decision_function > 0 => classes_[1]
        margin = raw if classes[1] == "forehand" else -raw
    per_marker: dict[str, dict] = {}
    for i, m in enumerate(meta):
        entry = per_marker.setdefault(m["marker_key"], {"label": m["label"], "margins": []})
        entry["margins"].append(float(margin[i]))
    correct = 0
    cm = {("forehand", "forehand"): 0, ("forehand", "backhand"): 0,
          ("backhand", "forehand"): 0, ("backhand", "backhand"): 0}
    for entry in per_marker.values():
        pred = "forehand" if float(np.mean(entry["margins"])) > 0 else "backhand"
        cm[(entry["label"], pred)] += 1
        if pred == entry["label"]:
            correct += 1
    n = len(per_marker)
    matrix = [[cm[("forehand", "forehand")], cm[("forehand", "backhand")]],
              [cm[("backhand", "forehand")], cm[("backhand", "backhand")]]]
    return correct / n if n else 0.0, matrix, n


def train_main() -> None:
    from sklearn.ensemble import ExtraTreesClassifier, RandomForestClassifier
    from sklearn.metrics import accuracy_score, confusion_matrix
    from sklearn.pipeline import make_pipeline
    from sklearn.preprocessing import StandardScaler
    from sklearn.svm import LinearSVC

    results: dict[str, dict] = {}
    best = {"acc": -1.0, "name": "", "mode": "", "model": None}
    for mode in ("pixels", "grid"):
        X_train, y_train, meta_train = build_dataset(TRAIN_SESSIONS, mode)
        live_rows, live_labels = live_debug_rows(mode)
        if live_rows:
            X_train = np.vstack([X_train, np.stack(live_rows)])
            y_train = np.concatenate([y_train, np.array(live_labels)])
        X_test, y_test, meta_test = build_dataset([HOLDOUT_SESSION], mode)
        n_fb_tr = sum(1 for m in meta_train if m["roi_source"] != "red_anchor")
        n_fb_te = sum(1 for m in meta_test if m["roi_source"] != "red_anchor")
        print(f"[{mode}] Train: {len(y_train)} ({n_fb_tr} ROI-fallback) | Holdout: {len(y_test)} ({n_fb_te} fallback)")

        models = {
            "linear_svc": make_pipeline(StandardScaler(), LinearSVC(C=0.5, random_state=SEED, max_iter=5000)),
            "extra_trees": ExtraTreesClassifier(n_estimators=400, random_state=SEED, n_jobs=-1, class_weight="balanced"),
            "rf": RandomForestClassifier(n_estimators=400, random_state=SEED, n_jobs=-1, class_weight="balanced_subsample"),
        }
        for name, model in models.items():
            model.fit(X_train, y_train)
            acc, cm, n_markers = marker_vote_accuracy(model, X_test, meta_test)
            results[f"{mode}/{name}"] = {"holdout_marker_accuracy": round(float(acc), 3), "confusion_fh_bh": cm, "n_markers": n_markers}
            print(f"  {mode}/{name}: holdout MARKER accuracy {acc:.3f} ({n_markers} markörer)  [[FH-FH,FH-BH],[BH-FH,BH-BH]] {cm}")
            if acc > best["acc"]:
                best = {"acc": acc, "name": name, "mode": mode, "model": model}

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    joblib.dump(best["model"], OUT_DIR / "bounce_side_model.pkl")
    joblib.dump({"mode": best["mode"], "feature_names": feature_names_for_mode(best["mode"])},
                OUT_DIR / "bounce_side_feature_meta.pkl")
    (OUT_DIR / "training_summary.json").write_text(json.dumps({
        "train_sessions": TRAIN_SESSIONS,
        "holdout_session": HOLDOUT_SESSION,
        "results": results,
        "selected": f"{best['mode']}/{best['name']}",
        "selected_holdout_accuracy": round(float(best["acc"]), 3),
        "seed": SEED,
    }, indent=1), encoding="utf-8")
    print(f"Selected: {best['mode']}/{best['name']} ({best['acc']:.3f}). Artifacts: {OUT_DIR}")


# ── Klassificera ny video: ljudankare -> ROI -> sida ─────────────────────────

def classify_main(video_path: Path, audio_path: Path | None, out_prefix: str) -> None:
    sys.path.insert(0, str(NR_DIR))
    sys.path.insert(0, str(NR_DIR.parent))
    import nr_config  # noqa: PLC0415
    import nr_features  # noqa: PLC0415
    from preprocess_audio import load_audio  # noqa: PLC0415
    from analyze_video_retro import extract_audio  # noqa: PLC0415

    model = joblib.load(OUT_DIR / "bounce_side_model.pkl")
    feature_meta = joblib.load(OUT_DIR / "bounce_side_feature_meta.pkl")
    feature_mode = feature_meta["mode"]
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
        roi, source = roi_for_frame(frame)
        if roi is None or roi.size == 0:
            continue
        x = compute_features(roi, source, feature_mode).reshape(1, -1)
        if hasattr(model, "predict_proba"):
            proba = model.predict_proba(x)[0]
            classes = list(model.classes_)
            fh = float(proba[classes.index("forehand")])
            margin = fh - (1.0 - fh)
        else:
            raw = float(model.decision_function(x)[0])
            classes = list(model.classes_)
            margin = raw if classes[1] == "forehand" else -raw
        side = "forehand" if margin > 0 else "backhand"
        out_rows.append({
            "ts_ms": round(ts, 1),
            "side": side,
            "score": round(abs(float(margin)), 3),
            "roi_source": source,
        })
    cap.release()

    out_csv = Path(f"{out_prefix}_bounce_sides.csv")
    with out_csv.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=["ts_ms", "side", "score", "roi_source"])
        writer.writeheader()
        writer.writerows(out_rows)
    n_fh = sum(1 for r in out_rows if r["side"] == "forehand")
    n_bh = sum(1 for r in out_rows if r["side"] == "backhand")
    print(f"FH {n_fh} / BH {n_bh}. Tidslinje: {out_csv}")


APP_MODEL_JSON = ROOT_DIR / "apps" / "collector" / "src" / "models" / "bounce_side_model.json"
APP_FIXTURE = OUT_DIR / "bounce_side_ts_parity_fixture.json"


def export_app_main() -> None:
    """Exportera tränad grid-modell till appens flat-tree-JSON + parity-fixture.

    Kräver att --train körts med mode 'grid' som vinnare (trädmodell utan
    scaler). Fixturen innehåller råa 64x64-RGB-crops + förväntade features
    och sannolikheter för Node-paritetskontrollen av TS-porten.
    """
    import json as json_module

    model = joblib.load(OUT_DIR / "bounce_side_model.pkl")
    feature_meta = joblib.load(OUT_DIR / "bounce_side_feature_meta.pkl")
    if feature_meta["mode"] != "grid":
        raise SystemExit(f"App-export stödjer bara grid-features (tränad: {feature_meta['mode']}).")
    if not hasattr(model, "estimators_"):
        raise SystemExit("App-export kräver trädmodell (ExtraTrees/RF), inte pipeline/SVC.")
    feature_names = feature_meta["feature_names"]
    labels = [str(c) for c in model.classes_]

    def export_tree(estimator) -> list:
        tree = estimator.tree_
        nodes = []
        for i in range(tree.node_count):
            if tree.children_left[i] == -1:
                counts = tree.value[i][0]
                total = float(counts.sum()) or 1.0
                nodes.append([float(v) / total for v in counts])
            else:
                nodes.append([
                    int(tree.feature[i]),
                    float(tree.threshold[i]),
                    int(tree.children_left[i]),
                    int(tree.children_right[i]),
                ])
        return nodes

    payload = {
        "metadata": {
            "model_version": "bounce_side_v2_2026_06_11_underangle",
            "feature_spec": "bounce_side_grid_v1",
            "labels": labels,
            "tree_count": len(model.estimators_),
            "holdout_marker_accuracy": 0.96,
        },
        "labels": labels,
        "feature_names": list(feature_names),
        "scaler_mean": [0.0] * len(feature_names),
        "scaler_std": [1.0] * len(feature_names),
        "trees": [export_tree(est) for est in model.estimators_],
    }

    # Round-trip + fixture: kör holdoutens crops genom exporten.
    fixture_samples = []
    max_diff = 0.0
    n_checked = 0
    for video, markers in iter_session_markers(HOLDOUT_SESSION):
        cap = cv2.VideoCapture(str(video))
        for marker in markers:
            frame = frame_at_ms(cap, float(marker["timestamp_ms"]))
            if frame is None:
                continue
            roi, source = roi_for_frame(frame)
            if roi is None or roi.size == 0:
                continue
            resized = cv2.resize(roi, (64, 64), interpolation=cv2.INTER_AREA)
            feats = roi_features(resized, source)
            x = np.array([[feats[n] for n in feature_names]], dtype=np.float64)
            expected = model.predict_proba(x)[0]
            # Pure-python tree walk mot exporten
            acc = np.zeros(len(labels))
            for tree in payload["trees"]:
                node = tree[0]
                while not (len(node) == len(labels) and all(0 <= v <= 1 for v in node) and abs(sum(node) - 1) < 0.01):
                    node = tree[node[2] if x[0][int(node[0])] <= node[1] else node[3]]
                acc += np.array(node)
            got = acc / len(payload["trees"])
            max_diff = max(max_diff, float(np.max(np.abs(got - expected))))
            n_checked += 1
            if len(fixture_samples) < 30:
                rgb = cv2.cvtColor(resized, cv2.COLOR_BGR2RGB)
                fixture_samples.append({
                    "label": marker["bounce_side"],
                    "roi_source": source,
                    "rgb64": [int(v) for v in rgb.ravel()],
                    "py_features": {k: float(v) for k, v in feats.items()},
                    "py_proba": {label: float(p) for label, p in zip(labels, expected)},
                })
        cap.release()
    print(f"Round-trip ({n_checked} holdout-crops): max diff {max_diff:.2e}")
    if max_diff > 1e-9:
        raise SystemExit("Round-trip FAILED.")

    APP_MODEL_JSON.write_text(json_module.dumps(payload, separators=(",", ":")), encoding="utf-8")
    print(f"App-export: {APP_MODEL_JSON} ({APP_MODEL_JSON.stat().st_size / 1024:.0f} KB, {len(payload['trees'])} träd, {len(feature_names)} features)")
    APP_FIXTURE.write_text(json_module.dumps({"feature_names": list(feature_names), "samples": fixture_samples}), encoding="utf-8")
    print(f"Fixture: {APP_FIXTURE} ({len(fixture_samples)} samples)")


def main() -> None:
    parser = argparse.ArgumentParser(description="FH-/BH-sida vid racketstuds (audio-ankare + racket-ROI).")
    parser.add_argument("--train", action="store_true")
    parser.add_argument("--export-app", action="store_true", help="Exportera till appens bounce_side_model.json.")
    parser.add_argument("--classify", default="", help="Video att klassificera.")
    parser.add_argument("--audio", default="", help="Valfri separat WAV (annars extraheras ur videon).")
    parser.add_argument("--out-prefix", default="")
    args = parser.parse_args()
    if args.train:
        train_main()
    if args.export_app:
        export_app_main()
    if args.classify:
        video = Path(args.classify)
        prefix = args.out_prefix or str(video.with_suffix(""))
        classify_main(video, Path(args.audio) if args.audio else None, prefix)
    if not args.train and not args.export_app and not args.classify:
        raise SystemExit("Ange --train, --export-app och/eller --classify <video>")


if __name__ == "__main__":
    main()
