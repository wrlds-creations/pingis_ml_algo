"""
T0050 targeted Fable debug round audit.

Evaluation-only script for Love's 2026-06-28 Fable-algoritm A/B/C blocks.
It reads saved app debug JSON as the complete count source and computes clip
features only where the debug file contains audio_b64 snippets.
"""

from __future__ import annotations

import base64
import csv
import json
import math
import sys
from collections import Counter
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np

SCRIPT_DIR = Path(__file__).resolve().parent
SCRIPTS_DIR = SCRIPT_DIR.parent
for _path in (str(SCRIPT_DIR), str(SCRIPTS_DIR)):
    if _path not in sys.path:
        sys.path.insert(0, _path)

import nr_features  # noqa: E402

ROOT_DIR = Path(__file__).resolve().parents[4]
DEFAULT_DEBUG_DIR = ROOT_DIR / "data" / "audio" / "raw" / "t0050_fable_targeted_round" / "fable_live_debug"
DEFAULT_OUT_DIR = ROOT_DIR / "data" / "audio" / "models" / "evaluations" / "t0050_fable_targeted_round"

PEAK_VETO_THRESHOLD = 0.222991943359375
PARTIAL_VETO_THRESHOLD = 0.76144


@dataclass(frozen=True)
class BlockSpec:
    block: str
    description: str
    kind: str
    expected: int | None
    reported_app: int | None
    filename: str


BLOCKS = [
    BlockSpec("A1", "normal racket", "real", 25, 26, "fable_live_session_2026-06-28T14-42-47-846Z.json"),
    BlockSpec("A2", "high/slower racket", "real", 31, 25, "fable_live_session_2026-06-28T14-44-22-436Z.json"),
    BlockSpec("A3", "fast racket", "real", 40, 23, "fable_live_session_2026-06-28T14-45-07-067Z.json"),
    BlockSpec("A4", "messy kid-style racket", "real", 30, 32, "fable_live_session_2026-06-28T14-45-42-972Z.json"),
    BlockSpec("B1", "talking", "negative", 0, 1, "fable_live_session_2026-06-28T14-46-29-673Z.json"),
    BlockSpec("EXTRA", "short unmapped extra session", "extra", None, 1, "fable_live_session_2026-06-28T14-46-47-703Z.json"),
    BlockSpec("B2", "loud talking", "negative", 0, 0, "fable_live_session_2026-06-28T14-47-53-073Z.json"),
    BlockSpec("B3", "racket handling, no ball", "negative", 0, 3, "fable_live_session_2026-06-28T14-49-08-360Z.json"),
    BlockSpec("B4", "talking + racket movement, no ball", "negative", 0, 1, "fable_live_session_2026-06-28T14-50-18-211Z.json"),
    BlockSpec("C1", "bounce while talking", "real", 30, 26, "fable_live_session_2026-06-28T14-51-09-058Z.json"),
    BlockSpec("C2", "messy failed practice", "real", 40, 0, "fable_live_session_2026-06-28T14-52-19-037Z.json"),
]


def boolish(value: Any) -> bool:
    return bool(value is True or str(value).lower() == "true")


def decode_audio_b64(audio_b64: str) -> np.ndarray:
    pcm_i16 = np.frombuffer(base64.b64decode(audio_b64), dtype="<i2")
    return pcm_i16.astype(np.float32) / 32768.0


def finite_float(value: Any) -> float | None:
    try:
        out = float(value)
    except Exception:
        return None
    return out if math.isfinite(out) else None


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        path.write_text("", encoding="utf-8")
        return
    fields: list[str] = []
    for row in rows:
        for key in row:
            if key not in fields:
                fields.append(key)
    with path.open("w", newline="", encoding="utf-8") as fh:
        writer = csv.DictWriter(fh, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def counter_json(counter: Counter[str]) -> str:
    return json.dumps(dict(sorted(counter.items())), ensure_ascii=False, sort_keys=True)


def load_events(debug_dir: Path) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    event_rows: list[dict[str, Any]] = []
    summary_rows: list[dict[str, Any]] = []
    for spec in BLOCKS:
        path = debug_dir / spec.filename
        if not path.exists():
            summary_rows.append({
                "block": spec.block,
                "filename": spec.filename,
                "missing": True,
                "expected": spec.expected,
                "reported_app": spec.reported_app,
            })
            continue
        payload = json.loads(path.read_text(encoding="utf-8"))
        events = list(payload.get("events") or [])
        labels: Counter[str] = Counter()
        reasons: Counter[str] = Counter()
        counted_conf: list[float] = []
        audio_clips = 0
        counted_with_audio = 0
        counted_without_audio = 0
        peak_veto_counted = 0
        partial_veto_counted = 0
        rejected_racket_label = 0
        rejected_noise_label = 0
        rejected_table_label = 0

        for event in events:
            label = str(event.get("model_label") or "")
            reason = str(event.get("reject_reason") or "")
            counted = boolish(event.get("counted"))
            has_audio = bool(event.get("audio_b64"))
            labels[label or "-"] += 1
            reasons[reason or "counted"] += 1
            conf = finite_float(event.get("model_confidence"))
            if counted and conf is not None:
                counted_conf.append(conf)
            if has_audio:
                audio_clips += 1
            if counted and has_audio:
                counted_with_audio += 1
            if counted and not has_audio:
                counted_without_audio += 1
            if not counted and label == "racket_bounce":
                rejected_racket_label += 1
            if not counted and label == "noise":
                rejected_noise_label += 1
            if not counted and label == "table_bounce":
                rejected_table_label += 1

            row: dict[str, Any] = {
                "block": spec.block,
                "description": spec.description,
                "kind": spec.kind,
                "filename": spec.filename,
                "event_index": event.get("index", ""),
                "onset_ms": event.get("native_onset_time_ms", ""),
                "saved_counted": counted,
                "saved_label": label,
                "saved_confidence": event.get("model_confidence", ""),
                "saved_reject_reason": reason,
                "native_rms": event.get("native_rms", ""),
                "native_background_rms": event.get("native_background_rms", ""),
                "has_audio_b64": has_audio,
            }
            if has_audio:
                clip = decode_audio_b64(str(event["audio_b64"]))
                clip_abs_peak = float(np.max(np.abs(clip))) if len(clip) else 0.0
                features = nr_features.extract_all_features(clip)
                nr_bp_peak_ratio = float(features.get("nr_bp_peak_ratio", 0.0))
                peak_veto = clip_abs_peak < PEAK_VETO_THRESHOLD
                partial_veto = nr_bp_peak_ratio > PARTIAL_VETO_THRESHOLD
                row.update({
                    "clip_abs_peak": clip_abs_peak,
                    "nr_bp_peak_ratio": nr_bp_peak_ratio,
                    "peak_veto_lt_0_22299": peak_veto,
                    "partial_veto_ratio_gt_0_76144": partial_veto,
                })
                if counted and peak_veto:
                    peak_veto_counted += 1
                if counted and partial_veto:
                    partial_veto_counted += 1
            event_rows.append(row)

        saved_counted = sum(1 for e in events if boolish(e.get("counted")))
        expected_delta = "" if spec.expected is None else saved_counted - int(spec.expected)
        summary_rows.append({
            "block": spec.block,
            "description": spec.description,
            "kind": spec.kind,
            "filename": spec.filename,
            "started_at": payload.get("started_at", ""),
            "stopped_at": payload.get("stopped_at", ""),
            "json_events": len(events),
            "audio_b64_events": audio_clips,
            "expected": "" if spec.expected is None else spec.expected,
            "reported_app": "" if spec.reported_app is None else spec.reported_app,
            "saved_counted_json": saved_counted,
            "saved_minus_expected": expected_delta,
            "counted_with_audio": counted_with_audio,
            "counted_without_audio_due_cap": counted_without_audio,
            "counted_conf_min": "" if not counted_conf else min(counted_conf),
            "counted_conf_mean": "" if not counted_conf else sum(counted_conf) / len(counted_conf),
            "counted_conf_max": "" if not counted_conf else max(counted_conf),
            "rejected_racket_label": rejected_racket_label,
            "rejected_noise_label": rejected_noise_label,
            "rejected_table_label": rejected_table_label,
            "peak_veto_counted_with_audio": peak_veto_counted,
            "partial_veto_counted_with_audio": partial_veto_counted,
            "label_counts": counter_json(labels),
            "reject_reason_counts": counter_json(reasons),
        })
    return event_rows, summary_rows


def write_report(out_dir: Path, summary_rows: list[dict[str, Any]]) -> None:
    mapped = [row for row in summary_rows if row.get("kind") != "extra" and not row.get("missing")]
    real = [row for row in mapped if row["kind"] == "real"]
    neg = [row for row in mapped if row["kind"] == "negative"]
    real_expected = sum(int(row["expected"]) for row in real if row["expected"] != "")
    real_counted = sum(int(row["saved_counted_json"]) for row in real)
    neg_false = sum(int(row["saved_counted_json"]) for row in neg)

    lines = [
        "# T0050 Fable Targeted Round Audit",
        "",
        "## Scope",
        "",
        "- Evaluation only; no model/app/APK/runtime change.",
        "- App count source is the complete saved Fable debug JSON event list.",
        "- Clip-feature analysis is limited to events with `audio_b64`; the Fable screen caps saved clips at 150 per session.",
        "- Expected counts are block-level notes from Love, so missed-contact timing cannot be assigned exactly without a reviewed timeline.",
        "",
        "## Overall",
        "",
        f"- Mapped real racket expected/count: `{real_expected}` / `{real_counted}`.",
        f"- Mapped negative false counts: `{neg_false}`.",
        "",
        "## Block Summary",
        "",
        "| Block | Description | Kind | Expected | App/Saved | Delta | Events | Audio Clips | Key Rejections | Veto: Peak / Partial |",
        "|---|---|---|---:|---:|---:|---:|---:|---|---|",
    ]
    for row in summary_rows:
        reasons = json.loads(str(row.get("reject_reason_counts") or "{}"))
        key_reasons = ", ".join(f"{k}:{v}" for k, v in sorted(reasons.items(), key=lambda kv: kv[1], reverse=True)[:3])
        expected = row.get("expected", "")
        delta = row.get("saved_minus_expected", "")
        lines.append(
            f"| `{row['block']}` | {row['description']} | {row['kind']} | {expected} | "
            f"{row.get('saved_counted_json', '')} | {delta} | {row.get('json_events', '')} | "
            f"{row.get('audio_b64_events', '')} | {key_reasons} | "
            f"{row.get('peak_veto_counted_with_audio', '')} / {row.get('partial_veto_counted_with_audio', '')} |"
        )

    lines += [
        "",
        "## Interpretation",
        "",
        "- The biggest issue in this round is not just speech false positives. Real-racket recall failed badly in `A3 fast`, `C1 bounce while talking`, and especially `C2 messy failed practice`.",
        "- `C2` produced native/model events but saved `0` counts; most events were rejected as `not_racket`, so this looks like model/domain rejection more than a complete native-onset absence.",
        "- Negative blocks still matter: `B1`, `B3`, and `B4` false-counted, with `B3 racket handling` worst at `3` false counts.",
        "- The conservative partial veto is not enough as the next product fix if it only removes false counts while the current model also misses many real bounces.",
        "",
        "## Recommended Next Step",
        "",
        "- Do not ship a gate from this round alone.",
        "- Treat T0050 as holdout/replay evidence first. If Love confirms these blocks are trainable, use the real missed/low-confidence positive clips plus handling/talking negatives in a full candidate retrain/replay ticket.",
        "- Before collecting more, verify whether `C2` truly contained about `40` audible racket contacts; if yes, it is high-value failure data.",
        "",
    ]
    (out_dir / "t0050_fable_targeted_round_report.md").write_text("\n".join(lines), encoding="utf-8")


def main() -> None:
    out_dir = DEFAULT_OUT_DIR
    out_dir.mkdir(parents=True, exist_ok=True)
    event_rows, summary_rows = load_events(DEFAULT_DEBUG_DIR)
    write_csv(out_dir / "t0050_event_features.csv", event_rows)
    write_csv(out_dir / "t0050_block_summary.csv", summary_rows)
    write_report(out_dir, summary_rows)
    print(f"wrote {out_dir}")
    for row in summary_rows:
        print(f"{row['block']} {row.get('saved_counted_json', 'missing')} / expected {row.get('expected', '')}")


if __name__ == "__main__":
    main()
