"""
train_video_stroke_v2.py

Punkt 4: förbättrad forehand/backhand-klassificering (video).

Diagnos av dagens svaghet:
  - De stora spelsessionerna är ENKLASSIGA (05-22_001/25_003/25_007 = bara
    forehand, 05-22_003/26_001 = bara backhand). Med tagnings-/radsplit kan
    modellen lära sig session/kamera i stället för slagmotorik.
  - Dagens 30 features är fönster-aggregat (mean/std/min/max över hela
    -700/+500 ms) utan tidsupplösning, utan kroppsram-normalisering och
    utan spegel-invarians för fattning.

v2-features (från sparade per-frame-pose-serier i
data/video/processed/landmarks/<session>/<video>.pose.json):
  - Kroppsram per frame: axelcentrum + axelbredd; racketarmens handled/
    armbåge uttrycks som (p - centrum)/axelbredd.
  - Spegel-normalisering: x-axeln multipliceras med -1 för vänsterhänta så
    att forehand alltid är "utåt samma håll".
  - Tidsupplösning: 4 lika tidsbin över -700..+500 ms med medel/std av
    normaliserad handled x/y + signerad hastighet vx/vy + armbågsvinkel.
  - Globala: läge/hastighet vid träffögonblicket, min/max-tidpunkter,
    banlängd, krökning, cross-body-andel, z-djup, synlighet.

Utvärdering (ärlig):
  - Holdout A: video_stroke_session_2026-05-18_007 (enda stora BLANDADE
    sessionen: 38 BH / 37 FH) - tränas aldrig på.
  - Holdout B: cross-session-test på 05-22_001 (FH) + 05-22_003 (BH)
    tillsammans, tränat utan dem - mäter om modellen generaliserar till
    nya enklassiga sessioner i stället för att memorera dem.
  - GroupKFold(5) per session på träningspoolen.
  - Baseline: exakt samma splits med dagens 30 features ur dataset-CSV:n.

Kör:
  python skills/pingis-stroke-detection/scripts/train_video_stroke_v2.py
"""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingClassifier, RandomForestClassifier
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.model_selection import GroupKFold, cross_val_score
from sklearn.preprocessing import LabelEncoder

ROOT_DIR = Path(__file__).resolve().parents[3]
DATASET_CSV = ROOT_DIR / "data" / "video" / "processed" / "video_stroke_dataset.csv"
LANDMARK_DIR = ROOT_DIR / "data" / "video" / "processed" / "landmarks"
OUT_DIR = ROOT_DIR / "data" / "video" / "models" / "video_stroke_v2"

WINDOW_PRE_MS = 700
WINDOW_POST_MS = 500
N_TIME_BINS = 4
MIN_FRAMES = 5
SEED = 20260610

LEFT_SHOULDER, RIGHT_SHOULDER = 11, 12
LEFT_ELBOW, RIGHT_ELBOW = 13, 14
LEFT_WRIST, RIGHT_WRIST = 15, 16

HOLDOUT_MIXED = "video_stroke_session_2026-05-18_007"
HOLDOUT_CROSS = {"audio_session_2026-05-22_001", "audio_session_2026-05-22_003"}

CLASSES = ["backhand", "forehand", "unknown"]


def elbow_angle(sx, sy, ex, ey, wx, wy) -> float:
    ux, uy = sx - ex, sy - ey
    fx, fy = wx - ex, wy - ey
    nu = math.hypot(ux, uy)
    nf = math.hypot(fx, fy)
    if nu < 1e-9 or nf < 1e-9:
        return 180.0
    c = max(-1.0, min(1.0, (ux * fx + uy * fy) / (nu * nf)))
    return math.degrees(math.acos(c))


def extract_v2_features(frames: list[dict], marker_ms: float, handedness: str) -> dict | None:
    """Tidsupplösta kroppsram-features för ett slagfönster."""
    start = marker_ms - WINDOW_PRE_MS
    end = marker_ms + WINDOW_POST_MS
    mirror = -1.0 if handedness == "left" else 1.0
    wrist_i, elbow_i, shoulder_i = (
        (LEFT_WRIST, LEFT_ELBOW, LEFT_SHOULDER) if handedness == "left"
        else (RIGHT_WRIST, RIGHT_ELBOW, RIGHT_SHOULDER)
    )

    rows = []
    for frame in frames:
        ts = float(frame["timestamp_ms"])
        if ts < start or ts > end or not frame.get("pose_detected"):
            continue
        lm = {p["type"]: p for p in frame["landmarks"]}
        need = (wrist_i, elbow_i, shoulder_i, LEFT_SHOULDER, RIGHT_SHOULDER)
        if any(i not in lm for i in need):
            continue
        ls, rs = lm[LEFT_SHOULDER], lm[RIGHT_SHOULDER]
        cx, cy = (ls["x"] + rs["x"]) / 2, (ls["y"] + rs["y"]) / 2
        width = abs(rs["x"] - ls["x"]) + 1e-6
        w, e, s = lm[wrist_i], lm[elbow_i], lm[shoulder_i]
        rows.append({
            "t": ts - marker_ms,
            "nx": mirror * (w["x"] - cx) / width,
            "ny": (w["y"] - cy) / width,
            "nz": mirror * w.get("z", 0.0) / max(width, 1e-6),
            "ex": mirror * (e["x"] - cx) / width,
            "ey": (e["y"] - cy) / width,
            "angle": elbow_angle(s["x"], s["y"], e["x"], e["y"], w["x"], w["y"]),
            "vis": float(w.get("visibility", 0.0)),
        })

    if len(rows) < MIN_FRAMES:
        return None
    rows.sort(key=lambda r: r["t"])
    t = np.array([r["t"] for r in rows])
    nx = np.array([r["nx"] for r in rows])
    ny = np.array([r["ny"] for r in rows])
    nz = np.array([r["nz"] for r in rows])
    ang = np.array([r["angle"] for r in rows])
    vis = np.array([r["vis"] for r in rows])

    dt = np.diff(t) / 1000.0
    dt[dt <= 0] = 1e-3
    vx = np.diff(nx) / dt
    vy = np.diff(ny) / dt
    vt = t[1:]

    feats: dict[str, float] = {}
    edges = np.linspace(-WINDOW_PRE_MS, WINDOW_POST_MS, N_TIME_BINS + 1)
    for b in range(N_TIME_BINS):
        m = (t >= edges[b]) & (t < edges[b + 1])
        mv = (vt >= edges[b]) & (vt < edges[b + 1])
        feats[f"bin{b}_nx_mean"] = float(nx[m].mean()) if m.any() else 0.0
        feats[f"bin{b}_nx_std"] = float(nx[m].std()) if m.any() else 0.0
        feats[f"bin{b}_ny_mean"] = float(ny[m].mean()) if m.any() else 0.0
        feats[f"bin{b}_ny_std"] = float(ny[m].std()) if m.any() else 0.0
        feats[f"bin{b}_vx_mean"] = float(vx[mv].mean()) if mv.any() else 0.0
        feats[f"bin{b}_vy_mean"] = float(vy[mv].mean()) if mv.any() else 0.0
        feats[f"bin{b}_angle_mean"] = float(ang[m].mean()) if m.any() else 180.0

    impact_idx = int(np.argmin(np.abs(t)))
    feats["impact_nx"] = float(nx[impact_idx])
    feats["impact_ny"] = float(ny[impact_idx])
    feats["impact_nz"] = float(nz[impact_idx])
    vi = int(np.argmin(np.abs(vt))) if len(vt) else 0
    feats["impact_vx"] = float(vx[vi]) if len(vx) else 0.0
    feats["impact_vy"] = float(vy[vi]) if len(vy) else 0.0
    feats["nx_min"] = float(nx.min())
    feats["nx_max"] = float(nx.max())
    feats["nx_argmin_ms"] = float(t[int(np.argmin(nx))])
    feats["nx_argmax_ms"] = float(t[int(np.argmax(nx))])
    feats["ny_min"] = float(ny.min())
    feats["ny_max"] = float(ny.max())
    feats["path_len"] = float(np.sum(np.hypot(np.diff(nx), np.diff(ny))))
    feats["curvature"] = float(np.sum(np.abs(np.diff(vx))) + np.sum(np.abs(np.diff(vy)))) if len(vx) > 1 else 0.0
    feats["cross_body_ratio"] = float((nx < 0).mean())
    feats["angle_min"] = float(ang.min())
    feats["angle_max"] = float(ang.max())
    feats["angle_delta"] = float(ang[-1] - ang[0])
    feats["nz_mean"] = float(nz.mean())
    feats["vis_mean"] = float(vis.mean())
    feats["n_frames"] = float(len(rows))
    return feats


def load_v2_dataset(meta: pd.DataFrame) -> tuple[pd.DataFrame, list[str]]:
    cache: dict[Path, list[dict]] = {}
    rows = []
    skipped = 0
    for _, r in meta.iterrows():
        stem = Path(str(r["video_filename"])).stem
        pose_path = LANDMARK_DIR / str(r["session_id"]) / f"{stem}.pose.json"
        if not pose_path.exists():
            skipped += 1
            continue
        if pose_path not in cache:
            cache[pose_path] = json.loads(pose_path.read_text(encoding="utf-8"))
        feats = extract_v2_features(cache[pose_path], float(r["timestamp_ms"]), str(r["handedness"]))
        if feats is None:
            skipped += 1
            continue
        rows.append({
            "session_id": r["session_id"],
            "stroke_type": r["stroke_type"],
            **feats,
        })
    df = pd.DataFrame(rows)
    feature_cols = [c for c in df.columns if c not in ("session_id", "stroke_type")]
    print(f"v2 dataset: {len(df)} rader ({skipped} skippade), {len(feature_cols)} features")
    return df, feature_cols


def evaluate(name: str, df: pd.DataFrame, feature_cols: list[str], le: LabelEncoder) -> dict:
    """Träna på allt utom holdouts; rapportera CV + båda holdouts."""
    results: dict[str, dict] = {}
    train_mask = ~df["session_id"].isin(HOLDOUT_CROSS | {HOLDOUT_MIXED})
    train = df[train_mask]
    X = train[feature_cols].to_numpy(dtype=np.float64)
    X = np.nan_to_num(X, nan=0.0, posinf=0.0, neginf=0.0)
    y = le.transform(train["stroke_type"].astype(str))
    groups = train["session_id"].to_numpy()

    models = {
        "rf": RandomForestClassifier(n_estimators=300, class_weight="balanced_subsample", random_state=SEED, n_jobs=-1),
        "histgb": HistGradientBoostingClassifier(max_iter=300, learning_rate=0.08, class_weight="balanced", random_state=SEED),
    }
    for model_name, model in models.items():
        cv = GroupKFold(n_splits=min(5, train["session_id"].nunique()))
        cv_scores = cross_val_score(model, X, y, cv=cv, groups=groups, scoring="f1_macro", n_jobs=1)
        model.fit(X, y)
        entry = {"cv_f1_macro": float(np.mean(cv_scores))}
        for hold_name, sessions in (("mixed_0518_007", {HOLDOUT_MIXED}), ("cross_0522", HOLDOUT_CROSS)):
            hold = df[df["session_id"].isin(sessions)]
            if hold.empty:
                continue
            Xh = np.nan_to_num(hold[feature_cols].to_numpy(dtype=np.float64), nan=0.0, posinf=0.0, neginf=0.0)
            yh = le.transform(hold["stroke_type"].astype(str))
            pred = model.predict(Xh)
            report = classification_report(yh, pred, labels=range(len(le.classes_)), target_names=le.classes_, output_dict=True, zero_division=0)
            fhbh = hold["stroke_type"].isin(["forehand", "backhand"]).to_numpy()
            fb_acc = float((pred[fhbh] == yh[fhbh]).mean()) if fhbh.any() else None
            entry[hold_name] = {
                "n": int(len(hold)),
                "macro_f1": round(float(report["macro avg"]["f1-score"]), 3),
                "fh_recall": round(float(report.get("forehand", {}).get("recall", 0)), 3),
                "bh_recall": round(float(report.get("backhand", {}).get("recall", 0)), 3),
                "fhbh_accuracy": round(fb_acc, 3) if fb_acc is not None else None,
                "confusion": confusion_matrix(yh, pred, labels=range(len(le.classes_))).tolist(),
            }
        results[model_name] = entry
        if model_name == "histgb":
            results[f"_fitted_{model_name}"] = model  # type: ignore[assignment]
    print(f"\n=== {name} ===")
    for model_name in models:
        e = results[model_name]
        line = f"  {model_name}: cv_f1={e['cv_f1_macro']:.3f}"
        for hold_name in ("mixed_0518_007", "cross_0522"):
            if hold_name in e:
                h = e[hold_name]
                line += f" | {hold_name}: FH/BH-acc={h['fhbh_accuracy']} (FH {h['fh_recall']}, BH {h['bh_recall']}, macroF1 {h['macro_f1']})"
        print(line)
    return results


# Features som utesluts ur APP-exporten: z-semantiken skiljer mellan
# MediaPipe (träning, höft-normaliserad) och ML Kit (appen, pixelskala);
# övriga features är enhetsinvarianta via axelbredd-normaliseringen.
APP_EXCLUDED_FEATURES = {"impact_nz", "nz_mean"}
APP_MODEL_JSON = ROOT_DIR / "apps" / "collector" / "src" / "models" / "video_stroke_model.json"


def export_app_model(df: pd.DataFrame, feature_cols: list[str], le: LabelEncoder, out_json: Path) -> None:
    """Träna RF på ALLA rader (utvärderingsprotokollet körs separat innan)
    och exportera till appens video_stroke_model.json-format."""
    from sklearn.preprocessing import StandardScaler

    app_cols = [c for c in feature_cols if c not in APP_EXCLUDED_FEATURES]
    X = np.nan_to_num(df[app_cols].to_numpy(dtype=np.float64), nan=0.0, posinf=0.0, neginf=0.0)
    y = le.transform(df["stroke_type"].astype(str))
    scaler = StandardScaler().fit(X)
    clf = RandomForestClassifier(
        n_estimators=300, class_weight="balanced_subsample", random_state=SEED, n_jobs=-1,
    ).fit(scaler.transform(X), y)

    def export_tree(estimator) -> list:
        tree = estimator.tree_
        nodes = []
        for i in range(tree.node_count):
            if tree.children_left[i] == -1:
                counts = tree.value[i][0]
                total = float(counts.sum()) or 1.0
                nodes.append([round(float(v) / total, 8) for v in counts])
            else:
                nodes.append([
                    int(tree.feature[i]),
                    round(float(tree.threshold[i]), 8),
                    int(tree.children_left[i]),
                    int(tree.children_right[i]),
                ])
        return nodes

    payload = {
        "trained": True,
        "model_version": "collector_video_stroke_v2_2026_06_11_timebins",
        "feature_spec": "video_stroke_features_v2",
        "labels": [str(c) for c in le.classes_],
        "feature_names": app_cols,
        "scaler_mean": [float(v) for v in scaler.mean_],
        "scaler_std": [float(v) for v in scaler.scale_],
        "trees": [export_tree(est) for est in clf.estimators_],
    }

    # Round-trip-kontroll: gå igenom exporten i ren Python mot sklearn.
    rng = np.random.default_rng(7)
    idx = rng.choice(len(X), size=min(100, len(X)), replace=False)
    Xs = scaler.transform(X[idx])
    expected = clf.predict_proba(Xs)
    n_classes = len(le.classes_)
    max_diff = 0.0
    for row_i, row in enumerate(Xs):
        acc = np.zeros(n_classes)
        for tree in payload["trees"]:
            node = tree[0]
            pos = 0
            while not (len(node) == n_classes and all(0 <= v <= 1 for v in node) and abs(sum(node) - 1) < 0.01):
                pos = node[2] if row[int(node[0])] <= node[1] else node[3]
                node = tree[int(pos)]
            acc += np.array(node)
        got = acc / len(payload["trees"])
        max_diff = max(max_diff, float(np.max(np.abs(got - expected[row_i]))))
    print(f"App-export round-trip ({len(idx)} rader): max diff {max_diff:.2e}")
    if max_diff > 1e-6:
        raise SystemExit("Round-trip FAILED - exporterar inte.")

    out_json.write_text(json.dumps(payload, separators=(",", ":")), encoding="utf-8")
    size_kb = out_json.stat().st_size / 1024
    print(f"App-export: {out_json} ({size_kb:.0f} KB, {len(payload['trees'])} träd, {len(app_cols)} features)")


def main() -> None:
    parser = argparse.ArgumentParser(description="Train improved FH/BH video stroke model (v2).")
    parser.add_argument("--out-dir", default=str(OUT_DIR))
    parser.add_argument("--export-app", action="store_true",
                        help="Exportera RF (utan z-features) till appens video_stroke_model.json.")
    args = parser.parse_args()

    meta = pd.read_csv(DATASET_CSV)
    meta = meta[meta["stroke_type"].isin(CLASSES)].reset_index(drop=True)
    le = LabelEncoder().fit(CLASSES)

    # Baseline: dagens 30 features direkt ur CSV:n, samma splits.
    old_cols = [c for c in meta.columns if c not in (
        "session_id", "player_name", "handedness", "camera_facing", "camera_angle",
        "camera_side", "video_filename", "take_index", "marker_id", "timestamp_ms",
        "stroke_type", "feature_spec",
    )]
    baseline = evaluate("BASELINE (30 gamla features)", meta.rename(columns=str), old_cols, le)

    v2_df, v2_cols = load_v2_dataset(meta)
    v2 = evaluate(f"V2 ({len(v2_cols)} tidsupplösta features)", v2_df, v2_cols, le)

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    fitted = v2.pop("_fitted_histgb", None)
    baseline.pop("_fitted_histgb", None)
    if fitted is not None:
        joblib.dump(fitted, out_dir / "video_stroke_v2_histgb.pkl")
        joblib.dump(v2_cols, out_dir / "video_stroke_v2_feature_cols.pkl")
        joblib.dump(le, out_dir / "video_stroke_v2_label_encoder.pkl")
    summary = {"baseline_30feat": baseline, "v2": v2, "n_v2_features": len(v2_cols), "seed": SEED}
    (out_dir / "training_summary.json").write_text(json.dumps(summary, indent=1, default=str), encoding="utf-8")
    print(f"\nArtifacts: {out_dir}")

    if args.export_app:
        export_app_model(v2_df, v2_cols, le, APP_MODEL_JSON)


if __name__ == "__main__":
    main()
