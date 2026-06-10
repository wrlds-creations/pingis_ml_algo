"""
replay_nr_live.py

Module 4 of the noise-robust racket bounce detector (see NR_SPEC.md).

Replays the live detection cascade on saved session WAVs and scores the
counted events against reviewed racket markers:

    10 ms frames -> adaptive onset gate -> spectral gate -> 300 ms clip
                -> features -> 4-class model -> confidence/merge/group -> count

Outputs a per-event CSV (`<out-prefix>_events.csv`) and a per-bucket summary
as JSON + markdown (`<out-prefix>_summary.json` / `.md`).

Run:
  python skills/pingis-audio-classification/scripts/noise_robust/replay_nr_live.py --split val
  python skills/pingis-audio-classification/scripts/noise_robust/replay_nr_live.py \
      --split test --split test_fp_bed --model rf --feature-set all83 --gate bandpass
  python skills/pingis-audio-classification/scripts/noise_robust/replay_nr_live.py --self-test
"""

from __future__ import annotations

import argparse
import csv
import json
import sys
import time
from pathlib import Path
from typing import Any

import joblib
import numpy as np

SCRIPT_DIR = Path(__file__).resolve().parent
SCRIPTS_DIR = SCRIPT_DIR.parent
for _path in (str(SCRIPT_DIR), str(SCRIPTS_DIR)):
    if _path not in sys.path:
        sys.path.insert(0, _path)

import nr_config  # noqa: E402
from preprocess_audio import (  # noqa: E402
    TARGET_SR,
    contact_kind_for,
    extract_features,
    is_trainable_racket_marker,
    load_audio,
)

ROOT_DIR = Path(__file__).resolve().parents[4]
if not (ROOT_DIR / "data" / "audio" / "raw").exists():
    raise RuntimeError(
        f"Repo root resolution failed: {ROOT_DIR / 'data' / 'audio' / 'raw'} does not exist"
    )

INVENTORY_PATH = ROOT_DIR / "data" / "audio" / "processed" / "audio_inventory_2026_06_10.json"
DEFAULT_MODEL_DIR = ROOT_DIR / "data" / "audio" / "models" / "noise_robust_v1"
DEFAULT_OUT_PREFIX = ROOT_DIR / "data" / "audio" / "processed" / "noise_robust" / "replay_nr_live"
APP_EXPORT_JSON = "nr_audio_model.json"

# Per-gate-mode default for the absolute RMS floor (NR_SPEC Module 1).
ABS_MIN_RMS_BY_GATE = {"broadband": 0.003, "bandpass": 0.0015}

# Latency model: feature+predict wall clock + 200 ms post-window wait +
# 10 ms onset frame quantization (NR_SPEC Module 4 step 5).
POST_WINDOW_MS = 200.0
FRAME_QUANT_MS = 10.0

BUCKET_ORDER = ["quiet", "music_low", "music_mid", "music_high", "speech", "mixed", "impact", "crowd"]
DIRECT_BUCKETS = {"quiet", "music_low", "music_mid", "music_high", "speech", "mixed", "impact"}
CROWD_SESSION_PREFIX = "audio_session_2026-04-09_"

VALID_SPLITS = ["val", "test", "val_fp_bed", "test_fp_bed"]
FP_BED_SPLITS = {"val_fp_bed", "test_fp_bed"}


# ---------------------------------------------------------------------------
# nr_features access (lazy so --help / --self-test work before Module 1 lands)
# ---------------------------------------------------------------------------

def load_nr_features() -> Any:
    import nr_features  # noqa: PLC0415

    return nr_features


# ---------------------------------------------------------------------------
# Bucket mapping
# ---------------------------------------------------------------------------

def bucket_for_event(session_id: str, background_condition: Any) -> str:
    """Map an event's background_condition to a reporting bucket.

    The crowd bucket is defined as the 04-09 bed sessions (their events have
    no background_condition). Unknown conditions fall back to `mixed`, except
    `desk` (keyboard impacts -> impact) and missing values outside the crowd
    sessions (-> quiet).
    """
    if session_id.startswith(CROWD_SESSION_PREFIX):
        return "crowd"
    condition = str(background_condition or "").strip().lower()
    if condition in DIRECT_BUCKETS:
        return condition
    if condition in {"", "(none)", "none", "null"}:
        return "quiet"
    if condition == "desk":
        return "impact"
    return "mixed"


# ---------------------------------------------------------------------------
# Scoring
# ---------------------------------------------------------------------------

def score_matches(detections_ms: list[float], truth_ms: list[int], tolerance_ms: int) -> dict[str, int]:
    """Greedy nearest-truth matching; each truth marker is matchable once.

    A detection within tolerance of an already-matched truth marker is a
    duplicate (not a false positive).
    """
    matched: set[int] = set()
    true_positive = 0
    false_positive = 0
    duplicates = 0
    for detection in sorted(detections_ms):
        nearest_idx: int | None = None
        nearest_delta = tolerance_ms + 1
        for idx, truth in enumerate(truth_ms):
            delta = abs(detection - truth)
            if delta <= tolerance_ms and delta < nearest_delta:
                nearest_idx = idx
                nearest_delta = delta
        if nearest_idx is None:
            false_positive += 1
        elif nearest_idx in matched:
            duplicates += 1
        else:
            matched.add(nearest_idx)
            true_positive += 1
    return {
        "true_positive": true_positive,
        "false_positive": false_positive,
        "duplicates": duplicates,
        "missed": len(truth_ms) - true_positive,
    }


def truth_gate_hits(truth_ms: list[int], trigger_ms: list[float], tolerance_ms: int) -> int:
    """Number of truth markers with at least one trigger within tolerance."""
    return sum(
        1 for truth in truth_ms if any(abs(truth - trig) <= tolerance_ms for trig in trigger_ms)
    )


def apply_count_logic(
    decision_ms: list[float], merge_ms: int, group_ms: int
) -> tuple[list[float], dict[str, int]]:
    """Merge window (vs last counted) then group window (vs group start)."""
    counted: list[float] = []
    rejects = {"merge_window": 0, "group_window": 0}
    last_counted_ms: float | None = None
    group_start_ms: float | None = None
    for event_ms in sorted(decision_ms):
        if last_counted_ms is not None and event_ms - last_counted_ms <= merge_ms:
            rejects["merge_window"] += 1
            continue
        if group_start_ms is not None and event_ms - group_start_ms <= group_ms:
            rejects["group_window"] += 1
            continue
        counted.append(event_ms)
        last_counted_ms = event_ms
        group_start_ms = event_ms
    return counted, rejects


# ---------------------------------------------------------------------------
# Feature plumbing
# ---------------------------------------------------------------------------

def base62_feature_names() -> list[str]:
    dummy = np.zeros(nr_config.FEATURE_BUFFER_SAMPLES, dtype=np.float32)
    return list(extract_features(dummy, TARGET_SR).keys())


def robust_feature_names(nr_mod: Any) -> list[str]:
    dummy = np.random.default_rng(0).normal(0.0, 1e-3, nr_config.CLIP_SAMPLES).astype(np.float32)
    feats = nr_mod.extract_all_features(dummy, TARGET_SR)
    return [name for name in feats if name.startswith("nr_")]


def compute_clip_features(
    clip: np.ndarray, feature_set: str, nr_mod: Any, robust_fn: Any
) -> dict[str, float]:
    """Compute only what the chosen feature set needs (fair latency numbers)."""
    if feature_set == "base62":
        padded = np.zeros(nr_config.FEATURE_BUFFER_SAMPLES, dtype=np.float32)
        padded[: len(clip)] = clip
        return extract_features(padded, TARGET_SR)
    if feature_set == "robust21" and robust_fn is not None:
        return robust_fn(clip, TARGET_SR)
    feats = nr_mod.extract_all_features(clip, TARGET_SR)
    if feature_set == "robust21":
        return {name: value for name, value in feats.items() if name.startswith("nr_")}
    return feats


# ---------------------------------------------------------------------------
# Model loading
# ---------------------------------------------------------------------------

def _unpack_joblib(payload: Any) -> tuple[Any, Any, list[str] | None, list[str] | None]:
    """Return (classifier, scaler, classes, feature_names) from a joblib payload.

    Supports bare estimators, sklearn Pipelines and dict bundles.
    """
    if isinstance(payload, dict):
        classifier = payload.get("model") or payload.get("classifier") or payload.get("estimator")
        if classifier is None:
            raise ValueError("joblib dict bundle has no 'model'/'classifier'/'estimator' key")
        scaler = payload.get("scaler")
        classes: list[str] | None = None
        encoder = payload.get("label_encoder")
        if encoder is not None and hasattr(encoder, "classes_"):
            classes = [str(c) for c in encoder.classes_]
        elif payload.get("classes") is not None:
            classes = [str(c) for c in payload["classes"]]
        elif payload.get("labels") is not None:
            classes = [str(c) for c in payload["labels"]]
        feature_names: list[str] | None = None
        for key in ("feature_names", "feature_cols", "feature_columns", "features"):
            if payload.get(key) is not None:
                feature_names = [str(c) for c in payload[key]]
                break
        return classifier, scaler, classes, feature_names
    return payload, None, None, None


def resolve_model_path(model_dir: Path, model_name: str, feature_set: str, model_file: str) -> Path:
    if model_file:
        path = Path(model_file)
        if not path.is_absolute():
            path = model_dir / path
        if not path.exists():
            raise SystemExit(f"--model-file not found: {path}")
        return path
    candidates = [
        model_dir / f"nr_{model_name}_{feature_set}.pkl",  # train_nr_model.py's actual name
        model_dir / f"{model_name}_{feature_set}.joblib",
        model_dir / f"{model_name}_{feature_set}.pkl",
        model_dir / f"nr_{model_name}_{feature_set}.joblib",
        model_dir / f"{model_name}_{feature_set}_model.joblib",
    ]
    for candidate in candidates:
        if candidate.exists():
            return candidate
    raise SystemExit(
        "No model artifact found. Tried: "
        + ", ".join(str(c) for c in candidates)
        + " (use --model-file to point at the joblib saved by train_nr_model.py)"
    )


def resolve_classes(classifier: Any, classes: list[str] | None) -> list[str]:
    if classes:
        return classes
    raw = getattr(classifier, "classes_", None)
    if raw is None:
        raise SystemExit("Cannot determine class labels: classifier has no classes_ attribute")
    arr = np.asarray(raw)
    if arr.dtype.kind in ("U", "S", "O"):
        return [str(c) for c in arr]
    # Numeric classes: the trainer label-encoded alphabetically (NR_SPEC),
    # which is exactly nr_config.CLASSES order.
    if len(arr) != len(nr_config.CLASSES):
        raise SystemExit(
            f"Numeric classifier classes_ has {len(arr)} entries; expected {len(nr_config.CLASSES)}"
        )
    return list(nr_config.CLASSES)


def resolve_feature_names(
    classifier: Any, bundle_names: list[str] | None, feature_set: str, nr_mod: Any
) -> list[str]:
    if bundle_names:
        return bundle_names
    names = getattr(classifier, "feature_names_in_", None)
    if names is not None:
        return [str(n) for n in names]
    base = base62_feature_names()
    if feature_set == "base62":
        return base
    robust = robust_feature_names(nr_mod)
    if feature_set == "robust21":
        return robust
    return base + robust


def resolve_scaler(
    model_dir: Path,
    model_name: str,
    feature_set: str,
    scaler_file: str,
    bundle_scaler: Any,
    feature_names: list[str],
) -> tuple[np.ndarray | None, np.ndarray | None, str]:
    """Return (mean, std, source). RF prefers the app JSON export for parity."""
    if scaler_file:
        path = Path(scaler_file)
        if not path.is_absolute():
            path = model_dir / path
        if not path.exists():
            raise SystemExit(f"--scaler-file not found: {path}")
        scaler = joblib.load(path)
        return (
            np.asarray(scaler.mean_, dtype=np.float64),
            np.asarray(scaler.scale_, dtype=np.float64),
            f"file:{path.name}",
        )
    if model_name == "rf":
        app_path = model_dir / APP_EXPORT_JSON
        if app_path.exists():
            data = json.loads(app_path.read_text(encoding="utf-8"))
            app_names = [str(n) for n in (data.get("feature_names") or [])]
            if app_names == list(feature_names):
                return (
                    np.asarray(data["scaler_mean"], dtype=np.float64),
                    np.asarray(data["scaler_std"], dtype=np.float64),
                    f"app_json:{APP_EXPORT_JSON}",
                )
    if bundle_scaler is not None and hasattr(bundle_scaler, "mean_"):
        return (
            np.asarray(bundle_scaler.mean_, dtype=np.float64),
            np.asarray(bundle_scaler.scale_, dtype=np.float64),
            "bundle",
        )
    for candidate in (
        f"nr_scaler_{feature_set}.pkl",  # train_nr_model.py's actual name
        f"scaler_{feature_set}.joblib",
        f"{model_name}_{feature_set}_scaler.joblib",
        f"scaler_{feature_set}.pkl",
        "scaler.joblib",
    ):
        path = model_dir / candidate
        if path.exists():
            scaler = joblib.load(path)
            return (
                np.asarray(scaler.mean_, dtype=np.float64),
                np.asarray(scaler.scale_, dtype=np.float64),
                f"joblib:{candidate}",
            )
    return None, None, "none"


def load_model(
    model_dir: Path, model_name: str, feature_set: str, model_file: str, scaler_file: str, nr_mod: Any,
    allow_unscaled: bool = False,
) -> dict[str, Any]:
    model_path = resolve_model_path(model_dir, model_name, feature_set, model_file)
    payload = joblib.load(model_path)
    classifier, bundle_scaler, classes, bundle_names = _unpack_joblib(payload)
    classes = resolve_classes(classifier, classes)
    feature_names = resolve_feature_names(classifier, bundle_names, feature_set, nr_mod)
    if hasattr(classifier, "named_steps"):
        scaler_mean, scaler_std, scaler_source = None, None, "pipeline_internal"
    else:
        scaler_mean, scaler_std, scaler_source = resolve_scaler(
            model_dir, model_name, feature_set, scaler_file, bundle_scaler, feature_names
        )
        if scaler_mean is None:
            # train_nr_model.py always fits on StandardScaler-transformed
            # features; predicting unscaled silently produces garbage.
            msg = (
                "No scaler artifact found for this model/feature set; the trainer "
                "fits on scaled features, so unscaled replay would be invalid. "
                "Pass --scaler-file, or --allow-unscaled if the model truly "
                "was trained without scaling."
            )
            if not allow_unscaled:
                raise SystemExit(msg)
            print("WARNING: " + msg + " Proceeding because --allow-unscaled was set.")
    if "racket_bounce" not in classes:
        raise SystemExit(f"Model classes do not include racket_bounce: {classes}")
    return {
        "classifier": classifier,
        "classes": classes,
        "feature_names": feature_names,
        "scaler_mean": scaler_mean,
        "scaler_std": scaler_std,
        "scaler_source": scaler_source,
        "model_path": str(model_path),
    }


# ---------------------------------------------------------------------------
# Replay core
# ---------------------------------------------------------------------------

def replay_event_wav(
    y: np.ndarray,
    nr_mod: Any,
    robust_fn: Any,
    model: dict[str, Any],
    args: argparse.Namespace,
    abs_min_rms: float,
    latencies_ms: list[float],
) -> dict[str, Any]:
    triggers = nr_mod.simulate_gate(
        y,
        TARGET_SR,
        onset_ratio=args.onset_ratio,
        retrigger_ms=args.retrigger_ms,
        abs_min_rms=abs_min_rms,
        mode=args.gate,
        spectral_gate=args.spectral_gate,
    )
    classifier = model["classifier"]
    classes: list[str] = model["classes"]
    feature_names: list[str] = model["feature_names"]
    scaler_mean = model["scaler_mean"]
    scaler_std = model["scaler_std"]
    racket_idx = classes.index("racket_bounce")

    raw_ms = [float(t["onset_ms"]) for t in triggers]
    passed = [t for t in triggers if bool(t.get("passed_spectral", True))]
    passed_ms = [float(t["onset_ms"]) for t in passed]

    decision_ms: list[float] = []
    event_latencies: list[float] = []
    for trigger in passed:
        clip = nr_mod.extract_live_clip(y, int(trigger["onset_sample"]))
        started = time.perf_counter()
        feats = compute_clip_features(clip, args.feature_set, nr_mod, robust_fn)
        x = np.array([[float(feats.get(name, 0.0)) for name in feature_names]], dtype=np.float64)
        x = np.nan_to_num(x, nan=0.0, posinf=0.0, neginf=0.0)
        if scaler_mean is not None:
            std_safe = np.where(scaler_std == 0.0, 1.0, scaler_std)
            x = (x - scaler_mean) / std_safe
        probs = classifier.predict_proba(x)[0]
        elapsed_ms = (time.perf_counter() - started) * 1000.0
        latencies_ms.append(elapsed_ms)
        event_latencies.append(elapsed_ms)
        best_idx = int(np.argmax(probs))
        confidence = float(probs[best_idx])
        p_racket = float(probs[racket_idx])
        if args.decision == "argmax":
            qualifies = classes[best_idx] == "racket_bounce" and confidence >= args.confidence
        else:
            qualifies = p_racket >= args.confidence
        if qualifies:
            decision_ms.append(float(trigger["onset_ms"]))

    counted_ms, count_rejects = apply_count_logic(decision_ms, args.merge_ms, args.group_ms)
    return {
        "raw_ms": raw_ms,
        "passed_ms": passed_ms,
        "decision_ms": decision_ms,
        "counted_ms": counted_ms,
        "count_rejects": count_rejects,
        "n_classified": len(passed),
        "mean_feature_predict_ms": float(np.mean(event_latencies)) if event_latencies else None,
    }


# ---------------------------------------------------------------------------
# Session selection
# ---------------------------------------------------------------------------

def load_inventory() -> dict[str, Any]:
    if not INVENTORY_PATH.exists():
        raise SystemExit(f"Inventory not found: {INVENTORY_PATH}")
    return json.loads(INVENTORY_PATH.read_text(encoding="utf-8"))


def select_sessions(
    inventory: dict[str, Any], splits: list[str], session_ids: list[str]
) -> list[tuple[dict[str, Any], str]]:
    by_id = {record["session_id"]: record for record in inventory["sessions"]}
    selected: list[tuple[dict[str, Any], str]] = []
    if session_ids:
        for session_id in session_ids:
            if session_id not in by_id:
                raise SystemExit(f"Session not in inventory: {session_id}")
            selected.append((by_id[session_id], nr_config.split_for_session(session_id)))
        return selected
    for session_id in sorted(by_id):
        record = by_id[session_id]
        if record.get("group") != "main":
            continue
        split = nr_config.split_for_session(session_id)
        if split not in splits:
            continue
        if record.get("is_diagnostic") and session_id != "audio_session_2026-05-29_001":
            continue
        selected.append((record, split))
    return selected


# ---------------------------------------------------------------------------
# Aggregation + reporting
# ---------------------------------------------------------------------------

def new_accumulator() -> dict[str, float]:
    return {
        "events": 0,
        "audio_s": 0.0,
        "n_truth": 0,
        "n_triggers_raw": 0,
        "n_classified": 0,
        "n_counted": 0,
        "true_positive": 0,
        "false_positive": 0,
        "duplicates": 0,
        "missed": 0,
        "gate_truth_hits_pre": 0,
        "gate_truth_hits_post": 0,
    }


def accumulate(acc: dict[str, float], row: dict[str, Any]) -> None:
    acc["events"] += 1
    acc["audio_s"] += float(row["duration_s"])
    acc["n_truth"] += int(row["n_truth"])
    acc["n_triggers_raw"] += int(row["n_triggers_raw"])
    acc["n_classified"] += int(row["n_classified"])
    acc["n_counted"] += int(row["n_counted"])
    acc["true_positive"] += int(row["true_positive"])
    acc["false_positive"] += int(row["false_positive"])
    acc["duplicates"] += int(row["duplicates"])
    acc["missed"] += int(row["missed"])
    acc["gate_truth_hits_pre"] += int(row["gate_truth_hits_pre"] or 0)
    acc["gate_truth_hits_post"] += int(row["gate_truth_hits_post"] or 0)


def derived_metrics(acc: dict[str, float]) -> dict[str, Any]:
    truth = acc["n_truth"]
    tp = acc["true_positive"]
    fp = acc["false_positive"]
    minutes = acc["audio_s"] / 60.0
    return {
        **acc,
        "audio_min": round(minutes, 3),
        "recall": round(tp / truth, 4) if truth else None,
        "precision": round(tp / (tp + fp), 4) if (tp + fp) else None,
        "fp_per_min": round(fp / minutes, 4) if minutes > 0 else None,
        "gate_recall_pre": round(acc["gate_truth_hits_pre"] / truth, 4) if truth else None,
        "gate_recall_post": round(acc["gate_truth_hits_post"] / truth, 4) if truth else None,
    }


def fmt(value: Any) -> str:
    if value is None:
        return "-"
    if isinstance(value, float):
        return f"{value:.3f}"
    return str(value)


def summary_markdown(
    config_line: str,
    per_bucket: dict[str, dict[str, Any]],
    overall: dict[str, Any],
    latency: dict[str, Any],
) -> str:
    lines = ["# replay_nr_live summary", "", config_line, ""]
    header = (
        "| bucket | events | audio_min | truth | counted | tp | fp | dup | missed "
        "| recall | precision | fp_per_min | gate_recall_pre | gate_recall_post |"
    )
    lines.append(header)
    lines.append("|" + "---|" * 14)
    ordered = [b for b in BUCKET_ORDER if b in per_bucket] + sorted(
        b for b in per_bucket if b not in BUCKET_ORDER
    )
    for bucket in ordered:
        m = per_bucket[bucket]
        lines.append(
            f"| {bucket} | {m['events']} | {fmt(m['audio_min'])} | {m['n_truth']} "
            f"| {m['n_counted']} | {m['true_positive']} | {m['false_positive']} "
            f"| {m['duplicates']} | {m['missed']} | {fmt(m['recall'])} | {fmt(m['precision'])} "
            f"| {fmt(m['fp_per_min'])} | {fmt(m['gate_recall_pre'])} | {fmt(m['gate_recall_post'])} |"
        )
    m = overall
    lines.append(
        f"| ALL | {m['events']} | {fmt(m['audio_min'])} | {m['n_truth']} "
        f"| {m['n_counted']} | {m['true_positive']} | {m['false_positive']} "
        f"| {m['duplicates']} | {m['missed']} | {fmt(m['recall'])} | {fmt(m['precision'])} "
        f"| {fmt(m['fp_per_min'])} | {fmt(m['gate_recall_pre'])} | {fmt(m['gate_recall_post'])} |"
    )
    lines.append("")
    lines.append(
        f"Latency over {latency['n_clips']} clips: feature+predict "
        f"p50={fmt(latency['feature_predict_ms_p50'])} ms, "
        f"p95={fmt(latency['feature_predict_ms_p95'])} ms; "
        f"latency_est_ms_p50={fmt(latency['latency_est_ms_p50'])}, "
        f"latency_est_ms_p95={fmt(latency['latency_est_ms_p95'])} "
        f"(+{POST_WINDOW_MS:.0f} ms post-window, +{FRAME_QUANT_MS:.0f} ms frame quantization)."
    )
    lines.append("")
    return "\n".join(lines)


EVENT_CSV_FIELDS = [
    "session_id",
    "split",
    "event_index",
    "wav_filename",
    "scenario_id",
    "background_condition",
    "bucket",
    "is_fp_bed",
    "duration_s",
    "n_truth",
    "n_triggers_raw",
    "n_triggers_passed_spectral",
    "n_classified",
    "n_decision_pass",
    "n_counted",
    "true_positive",
    "false_positive",
    "duplicates",
    "missed",
    "merge_rejects",
    "group_rejects",
    "gate_truth_hits_pre",
    "gate_truth_hits_post",
    "gate_recall_pre",
    "gate_recall_post",
    "fp_per_min",
    "mean_feature_predict_ms",
]


# ---------------------------------------------------------------------------
# Main replay loop
# ---------------------------------------------------------------------------

def run_replay(args: argparse.Namespace) -> None:
    nr_mod = load_nr_features()
    robust_fn = getattr(nr_mod, "extract_robust_features", None)
    abs_min_rms = args.abs_min_rms if args.abs_min_rms is not None else ABS_MIN_RMS_BY_GATE[args.gate]
    tolerance_ms = args.match_tolerance_ms

    model = load_model(
        Path(args.model_dir), args.model, args.feature_set, args.model_file, args.scaler_file, nr_mod,
        allow_unscaled=args.allow_unscaled,
    )
    print(f"Model: {model['model_path']}")
    print(f"Scaler source: {model['scaler_source']}; n_features={len(model['feature_names'])}")
    print(f"Classes: {model['classes']}")

    session_ids = [s for s in (args.sessions or "").split(",") if s.strip()]
    splits = args.split or ([] if session_ids else ["val"])
    inventory = load_inventory()
    selected = select_sessions(inventory, splits, [s.strip() for s in session_ids])
    if not selected:
        raise SystemExit("No sessions selected (check --split / --sessions).")
    print(f"Selected {len(selected)} session(s): {[record['session_id'] for record, _ in selected]}")

    rows: list[dict[str, Any]] = []
    latencies_ms: list[float] = []
    skipped = {
        "missing_wav": 0,
        "missing_wav_filename": 0,
        "unreviewed_event": 0,
        "fp_bed_racket_event": 0,
    }
    per_bucket: dict[str, dict[str, float]] = {}
    per_split: dict[str, dict[str, float]] = {}
    overall = new_accumulator()

    for record, split in selected:
        session_id = record["session_id"]
        is_fp_bed = split in FP_BED_SPLITS
        session_json_path = ROOT_DIR / record["json_path"]
        if not session_json_path.exists():
            print(f"WARNING: session JSON missing, skipping session: {session_json_path}")
            continue
        session = json.loads(session_json_path.read_text(encoding="utf-8"))
        media_dir = ROOT_DIR / record["media_dir"]
        events = session.get("events") or []
        for event_index, event in enumerate(events):
            wav_filename = event.get("wav_filename") or ""
            if not wav_filename:
                skipped["missing_wav_filename"] += 1
                continue
            wav_path = media_dir / wav_filename
            if not wav_path.exists():
                skipped["missing_wav"] += 1
                print(f"WARNING: missing WAV {wav_path}")
                continue
            label = str(event.get("label") or "")
            scenario_id = str(event.get("scenario_id") or "")
            markers = (event.get("review") or {}).get("markers") or []
            if is_fp_bed and contact_kind_for(label, scenario_id) == "racket_bounce":
                # Real racket content inside an FP bed session cannot be
                # scored as pure false positives -> excluded.
                skipped["fp_bed_racket_event"] += 1
                continue
            if not is_fp_bed and not markers:
                # Unreviewed event in a truth session: no scoring basis.
                skipped["unreviewed_event"] += 1
                continue

            truth = sorted(
                int(round(float(m.get("timestamp_ms") or 0)))
                for m in markers
                if is_trainable_racket_marker(m)
            )
            if is_fp_bed:
                truth = []

            y, _sr = load_audio(str(wav_path))
            duration_s = len(y) / float(TARGET_SR)
            replay = replay_event_wav(y, nr_mod, robust_fn, model, args, abs_min_rms, latencies_ms)

            if is_fp_bed:
                metrics = {
                    "true_positive": 0,
                    "false_positive": len(replay["counted_ms"]),
                    "duplicates": 0,
                    "missed": 0,
                }
            else:
                metrics = score_matches(replay["counted_ms"], truth, tolerance_ms)

            if truth:
                gate_pre = truth_gate_hits(truth, replay["raw_ms"], tolerance_ms)
                gate_post = truth_gate_hits(truth, replay["passed_ms"], tolerance_ms)
                gate_recall_pre: float | None = round(gate_pre / len(truth), 4)
                gate_recall_post: float | None = round(gate_post / len(truth), 4)
            else:
                gate_pre = 0
                gate_post = 0
                gate_recall_pre = None
                gate_recall_post = None

            minutes = duration_s / 60.0
            bucket = bucket_for_event(session_id, event.get("background_condition"))
            row: dict[str, Any] = {
                "session_id": session_id,
                "split": split,
                "event_index": event_index,
                "wav_filename": wav_filename,
                "scenario_id": scenario_id,
                "background_condition": str(event.get("background_condition") or ""),
                "bucket": bucket,
                "is_fp_bed": is_fp_bed,
                "duration_s": round(duration_s, 3),
                "n_truth": len(truth),
                "n_triggers_raw": len(replay["raw_ms"]),
                "n_triggers_passed_spectral": len(replay["passed_ms"]),
                "n_classified": replay["n_classified"],
                "n_decision_pass": len(replay["decision_ms"]),
                "n_counted": len(replay["counted_ms"]),
                **metrics,
                "merge_rejects": replay["count_rejects"]["merge_window"],
                "group_rejects": replay["count_rejects"]["group_window"],
                "gate_truth_hits_pre": gate_pre if truth else None,
                "gate_truth_hits_post": gate_post if truth else None,
                "gate_recall_pre": gate_recall_pre,
                "gate_recall_post": gate_recall_post,
                "fp_per_min": round(metrics["false_positive"] / minutes, 4) if minutes > 0 else None,
                "mean_feature_predict_ms": (
                    round(replay["mean_feature_predict_ms"], 3)
                    if replay["mean_feature_predict_ms"] is not None
                    else None
                ),
            }
            rows.append(row)
            accumulate(per_bucket.setdefault(bucket, new_accumulator()), row)
            accumulate(per_split.setdefault(split, new_accumulator()), row)
            accumulate(overall, row)

    if latencies_ms:
        p50 = float(np.percentile(latencies_ms, 50))
        p95 = float(np.percentile(latencies_ms, 95))
        latency = {
            "n_clips": len(latencies_ms),
            "feature_predict_ms_p50": round(p50, 3),
            "feature_predict_ms_p95": round(p95, 3),
            "latency_est_ms_p50": round(p50 + POST_WINDOW_MS + FRAME_QUANT_MS, 3),
            "latency_est_ms_p95": round(p95 + POST_WINDOW_MS + FRAME_QUANT_MS, 3),
        }
    else:
        latency = {
            "n_clips": 0,
            "feature_predict_ms_p50": None,
            "feature_predict_ms_p95": None,
            "latency_est_ms_p50": None,
            "latency_est_ms_p95": None,
        }

    out_prefix = Path(args.out_prefix)
    out_prefix.parent.mkdir(parents=True, exist_ok=True)
    events_csv = Path(f"{out_prefix}_events.csv")
    summary_json = Path(f"{out_prefix}_summary.json")
    summary_md = Path(f"{out_prefix}_summary.md")

    with events_csv.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=EVENT_CSV_FIELDS)
        writer.writeheader()
        for row in rows:
            writer.writerow({key: ("" if row.get(key) is None else row.get(key)) for key in EVENT_CSV_FIELDS})

    per_bucket_out = {bucket: derived_metrics(acc) for bucket, acc in per_bucket.items()}
    per_split_out = {split: derived_metrics(acc) for split, acc in per_split.items()}
    overall_out = derived_metrics(overall)
    config = {
        "model_dir": str(args.model_dir),
        "model": args.model,
        "feature_set": args.feature_set,
        "model_path": model["model_path"],
        "scaler_source": model["scaler_source"],
        "classes": model["classes"],
        "splits": splits,
        "sessions": session_ids,
        "gate": args.gate,
        "onset_ratio": args.onset_ratio,
        "abs_min_rms": abs_min_rms,
        "spectral_gate": args.spectral_gate,
        "confidence": args.confidence,
        "decision": args.decision,
        "retrigger_ms": args.retrigger_ms,
        "merge_ms": args.merge_ms,
        "group_ms": args.group_ms,
        "match_tolerance_ms": tolerance_ms,
    }
    summary = {
        "config": config,
        "overall": overall_out,
        "per_bucket": per_bucket_out,
        "per_split": per_split_out,
        "latency": latency,
        "skipped": skipped,
        "n_event_rows": len(rows),
    }
    summary_json.write_text(json.dumps(summary, indent=2), encoding="utf-8")

    config_line = (
        f"model={args.model} feature_set={args.feature_set} gate={args.gate} "
        f"onset_ratio={args.onset_ratio} abs_min_rms={abs_min_rms} "
        f"spectral_gate={args.spectral_gate} decision={args.decision} "
        f"confidence={args.confidence} retrigger={args.retrigger_ms} "
        f"merge={args.merge_ms} group={args.group_ms} "
        f"splits={','.join(splits) if splits else '-'} sessions={','.join(session_ids) if session_ids else '-'}"
    )
    markdown = summary_markdown(config_line, per_bucket_out, overall_out, latency)
    summary_md.write_text(markdown, encoding="utf-8")

    print()
    print(markdown)
    print(f"Skipped: {skipped}")
    print(f"Wrote {events_csv}")
    print(f"Wrote {summary_json}")
    print(f"Wrote {summary_md}")


# ---------------------------------------------------------------------------
# Self test (scorer + count logic, no audio or model needed)
# ---------------------------------------------------------------------------

def run_self_test() -> None:
    result = score_matches([1010, 1950, 2950, 5000], [1000, 2000, 3000], 140)
    expected = {"true_positive": 3, "false_positive": 1, "duplicates": 0, "missed": 0}
    assert result == expected, f"scorer case 1 failed: {result} != {expected}"

    result = score_matches([1010, 1090], [1000], 140)
    expected = {"true_positive": 1, "false_positive": 0, "duplicates": 1, "missed": 0}
    assert result == expected, f"scorer case 2 (duplicate) failed: {result} != {expected}"

    result = score_matches([1010, 1090], [1000, 2000, 3000], 140)
    expected = {"true_positive": 1, "false_positive": 0, "duplicates": 1, "missed": 2}
    assert result == expected, f"scorer case 3 (duplicate, multi-truth) failed: {result} != {expected}"

    counted, rejects = apply_count_logic([1000.0, 1100.0, 1500.0], 220, 80)
    assert counted == [1000.0, 1500.0], f"count logic counted wrong: {counted}"
    assert rejects == {"merge_window": 1, "group_window": 0}, f"count logic rejects wrong: {rejects}"

    print("self-test OK: greedy scorer and merge/group count logic behave as specified")


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Replay the noise-robust live racket-bounce cascade on reviewed sessions."
    )
    parser.add_argument("--model-dir", default=str(DEFAULT_MODEL_DIR))
    parser.add_argument("--model", choices=["rf", "histgb"], default="rf")
    parser.add_argument("--feature-set", choices=["base62", "robust21", "all83"], default="all83")
    parser.add_argument(
        "--split",
        action="append",
        choices=VALID_SPLITS,
        default=None,
        help="Repeatable. Default: val (ignored when --sessions is given).",
    )
    parser.add_argument("--sessions", default="", help="Comma-separated session ids (overrides --split).")
    parser.add_argument("--gate", choices=["broadband", "bandpass"], default="broadband")
    parser.add_argument("--onset-ratio", type=float, default=1.5)
    parser.add_argument(
        "--abs-min-rms",
        type=float,
        default=None,
        help="Default depends on --gate: broadband 0.003, bandpass 0.0015.",
    )
    parser.add_argument(
        "--spectral-gate",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Enable/disable the 256-pt spectral gate (--no-spectral-gate to disable).",
    )
    parser.add_argument("--confidence", type=float, default=0.5)
    parser.add_argument(
        "--decision",
        choices=["argmax", "prob"],
        default="argmax",
        help="argmax: count when argmax==racket_bounce AND conf>=threshold; "
        "prob: count when P(racket_bounce)>=threshold.",
    )
    parser.add_argument("--retrigger-ms", type=int, default=220)
    parser.add_argument("--merge-ms", type=int, default=220)
    parser.add_argument("--group-ms", type=int, default=80)
    parser.add_argument("--out-prefix", default=str(DEFAULT_OUT_PREFIX))
    parser.add_argument(
        "--model-file",
        default="",
        help="Explicit model joblib (absolute, or relative to --model-dir). "
        "Default: <model-dir>/<model>_<feature-set>.joblib.",
    )
    parser.add_argument(
        "--scaler-file",
        default="",
        help="Explicit StandardScaler joblib. Default: app JSON export (rf), bundle scaler, "
        "or <model-dir>/scaler_<feature-set>.joblib.",
    )
    parser.add_argument("--match-tolerance-ms", type=int, default=nr_config.MATCH_TOLERANCE_MS)
    parser.add_argument(
        "--allow-unscaled",
        action="store_true",
        help="Permit replay without a scaler artifact (only valid for models trained on unscaled features).",
    )
    parser.add_argument("--self-test", action="store_true", help="Run the scorer unit tests and exit.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.self_test:
        run_self_test()
        return
    run_replay(args)


if __name__ == "__main__":
    main()
