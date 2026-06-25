"""
analyze_nr_val_errors.py

Per-detection error analysis of the noise-robust val replays (Task B).
Consumes the `--dump-detections` CSVs written by replay_nr_live.py:

    data/audio/processed/noise_robust/diag_hgb_c65_r120_detections.csv
    data/audio/processed/noise_robust/diag_rf_c65_r120_detections.csv

Prints, per model:
  1. FP origin: which WAVs / buckets produce counted false positives and what
     the nearest trainable truth marker was (label + gap).
  2. p_racket and margin (p_racket - max(other)) distributions for TP vs FP
     per bucket, plus threshold / margin sweeps (TPs lost vs FPs removed).
  3. Miss taxonomy for every trainable racket truth marker in the val
     sessions: gate_miss (no classified trigger within tolerance),
     counted_dup (a counted detection nearby was absorbed by another truth),
     merge_window (qualifying detection killed by merge/group window),
     rejected_conf (predicted racket below confidence), model_miss
     (all nearby triggers classified not-racket).
  4. Speech-bucket FP detail and floor-take FP concentration.

Read-only with respect to session data; writes nothing (the markdown report
is authored separately from this output).

Run:
  python skills/pingis-audio-classification/scripts/noise_robust/analyze_nr_val_errors.py
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

SCRIPT_DIR = Path(__file__).resolve().parent
SCRIPTS_DIR = SCRIPT_DIR.parent
for _path in (str(SCRIPT_DIR), str(SCRIPTS_DIR)):
    if _path not in sys.path:
        sys.path.insert(0, _path)

import nr_config  # noqa: E402
from preprocess_audio import (  # noqa: E402
    is_trainable_racket_marker,
    is_trainable_review_marker,
)

ROOT_DIR = Path(__file__).resolve().parents[4]
if not (ROOT_DIR / "data" / "audio" / "raw").exists():
    raise RuntimeError(f"Repo root resolution failed: {ROOT_DIR}")

PROCESSED = ROOT_DIR / "data" / "audio" / "processed"
INVENTORY_PATH = PROCESSED / "audio_inventory_2026_06_10.json"
TOLERANCE_MS = nr_config.MATCH_TOLERANCE_MS

RUNS = {
    "histgb": PROCESSED / "noise_robust" / "diag_hgb_c65_r120_detections.csv",
    "rf": PROCESSED / "noise_robust" / "diag_rf_c65_r120_detections.csv",
}

THRESHOLDS = [0.65, 0.70, 0.75, 0.80, 0.85, 0.90]
MARGINS = [0.0, 0.10, 0.20, 0.30, 0.40, 0.50]


def load_truth() -> pd.DataFrame:
    """All trainable truth markers for val + val_fp_bed sessions."""
    inventory = json.loads(INVENTORY_PATH.read_text(encoding="utf-8"))
    rows = []
    for record in inventory["sessions"]:
        session_id = record["session_id"]
        split = nr_config.split_for_session(session_id)
        if split not in {"val", "val_fp_bed"}:
            continue
        session = json.loads((ROOT_DIR / record["json_path"]).read_text(encoding="utf-8"))
        for event in session.get("events") or []:
            wav = event.get("wav_filename") or ""
            markers = (event.get("review") or {}).get("markers") or []
            for marker in markers:
                if not is_trainable_review_marker(marker):
                    continue
                rows.append({
                    "session_id": session_id,
                    "split": split,
                    "wav_filename": wav,
                    "scenario_id": str(event.get("scenario_id") or ""),
                    "background_condition": str(event.get("background_condition") or ""),
                    "timestamp_ms": int(round(float(marker.get("timestamp_ms") or 0))),
                    "is_racket": is_trainable_racket_marker(marker),
                })
    return pd.DataFrame(rows)


def prob_stats(series: pd.Series) -> str:
    if series.empty:
        return "n=0"
    qs = series.quantile([0.1, 0.25, 0.5, 0.75, 0.9])
    return (
        f"n={len(series)} mean={series.mean():.3f} min={series.min():.3f} "
        f"p10={qs[0.1]:.3f} p25={qs[0.25]:.3f} p50={qs[0.5]:.3f} "
        f"p75={qs[0.75]:.3f} p90={qs[0.9]:.3f} max={series.max():.3f}"
    )


def classify_miss(near: pd.DataFrame) -> str:
    if near.empty:
        return "gate_miss"
    decisions = set(near["decision"])
    if "counted" in decisions:
        return "counted_dup"  # counted detection nearby but matched to another truth / dup
    if {"rejected_merge", "rejected_group"} & decisions:
        return "merge_window"
    if "rejected_conf" in decisions:
        return "rejected_conf"
    return "model_miss"


def analyze(model_name: str, csv_path: Path, truth: pd.DataFrame) -> None:
    df = pd.read_csv(csv_path)
    df["margin"] = df["p_racket"] - df[["p_table", "p_floor", "p_noise"]].max(axis=1)
    counted = df[df["decision"] == "counted"]
    fps = counted[counted["match_kind"] == "fp"]
    tps = counted[counted["match_kind"] == "tp"]

    print(f"\n{'=' * 78}\nMODEL {model_name}  ({csv_path.name}; {len(df)} classified triggers)\n{'=' * 78}")
    print(f"counted={len(counted)} tp={len(tps)} fp={len(fps)} dup={(counted['match_kind'] == 'dup').sum()}")

    print("\n--- 1. Counted FPs by bucket / wav / nearest truth ---")
    if fps.empty:
        print("no counted FPs")
    else:
        fps = fps.copy()
        fps["near"] = fps.apply(
            lambda r: f"{r['nearest_truth_label']}@{r['nearest_truth_gap_ms']:+.0f}ms"
            if isinstance(r["nearest_truth_label"], str) and r["nearest_truth_label"]
            else "none",
            axis=1,
        )
        for (bucket, sid, wav), grp in fps.groupby(["bucket", "session_id", "wav_filename"]):
            nears = grp["near"].value_counts().to_dict()
            labels = grp["nearest_truth_label"].fillna("none").value_counts().to_dict()
            print(f"[{bucket}] {sid}/{wav}: {len(grp)} FP | nearest-label {labels}")
            for _, r in grp.iterrows():
                print(
                    f"    onset={r['onset_ms']:.0f}ms pred={r['predicted_label']} "
                    f"p_racket={r['p_racket']:.3f} margin={r['margin']:.3f} nearest={r['near']}"
                )

    print("\n--- 2. p_racket / margin distributions: TP vs FP per bucket ---")
    for bucket in sorted(set(counted["bucket"])):
        tp_b = tps[tps["bucket"] == bucket]
        fp_b = fps[fps["bucket"] == bucket] if not fps.empty else fps
        print(f"[{bucket}] TP p_racket: {prob_stats(tp_b['p_racket'])}")
        if not fp_b.empty:
            print(f"[{bucket}] FP p_racket: {prob_stats(fp_b['p_racket'])}")
            print(f"[{bucket}] TP margin : {prob_stats(tp_b['margin'])}")
            print(f"[{bucket}] FP margin : {prob_stats(fp_b['margin'])}")

    print("\n--- 2b. Threshold sweep on counted detections (TP kept / FP kept) ---")
    for thr in THRESHOLDS:
        tp_keep = (tps["p_racket"] >= thr).sum()
        fp_keep = (fps["p_racket"] >= thr).sum() if not fps.empty else 0
        print(f"p_racket>={thr:.2f}: TP {tp_keep}/{len(tps)} FP {fp_keep}/{len(fps)}")
    for mar in MARGINS:
        tp_keep = (tps["margin"] >= mar).sum()
        fp_keep = (fps["margin"] >= mar).sum() if not fps.empty else 0
        print(f"margin >={mar:.2f}: TP {tp_keep}/{len(tps)} FP {fp_keep}/{len(fps)}")

    print("\n--- 3. Miss taxonomy (val racket truth markers) ---")
    racket_truth = truth[(truth["split"] == "val") & truth["is_racket"]]
    matched_ts = set(
        zip(tps["session_id"], tps["wav_filename"], tps["matched_truth"].astype(int))
    )
    miss_rows = []
    for _, t in racket_truth.iterrows():
        key = (t["session_id"], t["wav_filename"], t["timestamp_ms"])
        if key in matched_ts:
            continue
        near = df[
            (df["session_id"] == t["session_id"])
            & (df["wav_filename"] == t["wav_filename"])
            & ((df["onset_ms"] - t["timestamp_ms"]).abs() <= TOLERANCE_MS)
        ]
        cause = classify_miss(near)
        best = near.loc[near["p_racket"].idxmax()] if not near.empty else None
        miss_rows.append({
            "session_id": t["session_id"],
            "wav_filename": t["wav_filename"],
            "timestamp_ms": t["timestamp_ms"],
            "background": t["background_condition"],
            "cause": cause,
            "n_near": len(near),
            "best_p_racket": None if best is None else round(float(best["p_racket"]), 3),
            "best_pred": None if best is None else best["predicted_label"],
        })
    misses = pd.DataFrame(miss_rows)
    print(f"truth={len(racket_truth)} matched={len(racket_truth) - len(misses)} missed={len(misses)}")
    if not misses.empty:
        print(misses["cause"].value_counts().to_string())
        print("\nby session/background:")
        print(misses.groupby(["session_id", "background", "cause"]).size().to_string())
        print("\nmiss detail:")
        for _, m in misses.iterrows():
            print(
                f"  {m['session_id']}/{m['wav_filename']} t={m['timestamp_ms']}ms "
                f"[{m['background']}] cause={m['cause']} n_near={m['n_near']} "
                f"best_pred={m['best_pred']} best_p_racket={m['best_p_racket']}"
            )

    print("\n--- 4. Speech-bucket FPs: voice transient vs near-marker ---")
    sp = fps[fps["bucket"] == "speech"] if not fps.empty else fps
    if sp.empty:
        print("no speech-bucket FPs")
    else:
        for _, r in sp.iterrows():
            gap = r["nearest_truth_gap_ms"]
            tag = "no-marker-near" if (pd.isna(gap) or abs(gap) > 500) else f"near {r['nearest_truth_label']} ({gap:+.0f}ms)"
            print(f"  {r['wav_filename']} onset={r['onset_ms']:.0f} p_racket={r['p_racket']:.3f} -> {tag}")

    print("\n--- 5. Floor-take FP concentration (val_fp_bed) ---")
    bed = fps[fps["split"] == "val_fp_bed"] if not fps.empty else fps
    if bed.empty:
        print("no val_fp_bed FPs")
    else:
        print(bed.groupby("wav_filename").size().to_string())
        print("nearest-label mix:", bed["nearest_truth_label"].fillna("none").value_counts().to_dict())


def main() -> None:
    truth = load_truth()
    print(f"Truth markers loaded: {len(truth)} ({truth['is_racket'].sum()} racket)")
    for model_name, csv_path in RUNS.items():
        if not csv_path.exists():
            print(f"SKIP {model_name}: {csv_path} missing")
            continue
        analyze(model_name, csv_path, truth)


if __name__ == "__main__":
    main()
