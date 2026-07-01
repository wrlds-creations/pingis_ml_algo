#!/usr/bin/env python3
"""T0073 bad-case pack for the selected T0072 Fable candidate.

Evaluation-only. This script reads existing T0072 probability outputs, extracts
the bad cases for the selected policy, writes short WAV snippets, and produces
CSV/Markdown/HTML reports for review. It does not train or export a model and
does not change app runtime behavior.
"""

from __future__ import annotations

import argparse
import csv
import html
import json
import math
import re
import shutil
import sys
import wave
from collections import Counter, defaultdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np

ROOT = Path(__file__).resolve().parents[4]
SCRIPT_DIR = Path(__file__).resolve().parent
if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))

from evaluate_t0067_peak_gate_replay import match_predictions, read_wav, write_csv  # noqa: E402
from evaluate_t0069_peak_fable_hybrid_replay import DEFAULT_HELDOUT_WAV, HELDOUT_SESSION_ID, finite_float, intish  # noqa: E402
from evaluate_t0070_peak_candidate_classifier_veto import accepted_after_dedupe, md_table, read_csv_dicts  # noqa: E402

DEFAULT_RAW_DIR = ROOT / "data/audio/raw/t0065_fable_training_audio_round_a/fable_training_audio"
DEFAULT_T0071_DIR = ROOT / "data/audio/models/evaluations/t0071_round_a_scenario_label_expansion"
DEFAULT_T0072_DIR = ROOT / "data/audio/models/evaluations/t0072_round_a_reviewed_classifier_replay"
DEFAULT_T0063_LABELS = ROOT / "data/audio/models/evaluations/t0063_t0060_heldout_label_ingest/t0063_exact_heldout_labels.csv"
DEFAULT_OUT_DIR = ROOT / "data/audio/models/evaluations/t0073_fable_candidate_bad_case_export_prep"

SELECTED_CLASSIFIER_ID = "extra_leaf4"
SELECTED_PIPELINE_ID = "extra_leaf4_thr0p5_smart220"
SELECTED_PIPELINE_LABEL = "ExtraTrees leaf4 p>=0.5 smart220"
SELECTED_THRESHOLD = 0.5
SELECTED_DEDUPE_MS = 220.0
MATCH_TOLERANCE_MS = 140.0
SNIPPET_PRE_MS = 450.0
SNIPPET_POST_MS = 750.0

MANUAL_REVIEW_BY_CASE = {
    "round_fp_021": {
        "manual_review": "acceptable_bounce_like",
        "manual_note": "Love 2026-06-30: OK; sounds like a ball bounce as well, despite the recorder scenario being racket handling/no bounce.",
    },
    "round_fp_022": {
        "manual_review": "acceptable_bounce_like",
        "manual_note": "Love 2026-06-30: OK; sounds like a ball bounce as well, despite the recorder scenario being racket handling/no bounce.",
    },
    "round_fp_023": {
        "manual_review": "acceptable_bounce_like",
        "manual_note": "Love 2026-06-30: OK; sounds like a ball bounce as well, despite the recorder scenario being racket handling/no bounce.",
    },
    "round_fp_024": {
        "manual_review": "reject_unsafe_false_positive",
        "manual_note": "Love 2026-06-30: Should not count; floor/table/other impact, no racket.",
    },
    "round_fp_025": {
        "manual_review": "reject_unsafe_false_positive",
        "manual_note": "Love 2026-06-30: Should not count; floor/table/other impact, no racket.",
    },
}


def read_csv_rows(path: Path) -> list[dict[str, str]]:
    if not path.exists():
        return []
    with path.open("r", newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def truth_by_session(t0071_dir: Path) -> dict[str, list[dict[str, Any]]]:
    rows = read_csv_rows(t0071_dir / "t0071_reviewed_positive_labels.csv")
    out: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        sid = row["session_id"]
        out[sid].append(
            {
                "session_id": sid,
                "scenario_id": row.get("scenario_id", ""),
                "scenario_title": row.get("scenario_title", ""),
                "label_index": intish(row.get("label_index")),
                "time_ms": finite_float(row.get("reviewed_time_ms"), 0.0),
                "source": row.get("source", ""),
                "note": row.get("note", ""),
            }
        )
    for labels in out.values():
        labels.sort(key=lambda row: finite_float(row["time_ms"]))
    return dict(out)


def heldout_truth_rows(path: Path) -> list[dict[str, Any]]:
    rows = read_csv_rows(path)
    out: list[dict[str, Any]] = []
    for row in rows:
        if row.get("label") in {"racket", "racket_bounce"} or row.get("review_label") in {"racket", "racket_bounce"}:
            out.append(
                {
                    "session_id": HELDOUT_SESSION_ID,
                    "scenario_id": "heldout_c2_speaking_background",
                    "scenario_title": "Held-out C2 speaking/background",
                    "label_index": intish(row.get("label_index")),
                    "time_ms": finite_float(row.get("reviewed_time_s"), finite_float(row.get("time_s"), 0.0)) * 1000.0,
                    "source": row.get("source", ""),
                    "note": "",
                }
            )
    return sorted(out, key=lambda row: finite_float(row["time_ms"]))


def rows_by_session(rows: list[dict[str, Any]]) -> dict[str, list[dict[str, Any]]]:
    out: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        out[str(row["session_id"])].append(row)
    for values in out.values():
        values.sort(key=lambda row: finite_float(row.get("time_ms"), 0.0))
    return dict(out)


def selected_oof_rows(t0072_dir: Path) -> list[dict[str, Any]]:
    return [
        dict(row)
        for row in read_csv_dicts(t0072_dir / "t0072_oof_predictions.csv")
        if row.get("classifier_id") == SELECTED_CLASSIFIER_ID
    ]


def selected_heldout_rows(t0072_dir: Path) -> list[dict[str, Any]]:
    return [
        dict(row)
        for row in read_csv_dicts(t0072_dir / "t0072_final_predictions.csv")
        if row.get("classifier_id") == SELECTED_CLASSIFIER_ID and row.get("session_id") == HELDOUT_SESSION_ID
    ]


def nearest_row(rows: list[dict[str, Any]], target_ms: float) -> tuple[dict[str, Any] | None, float]:
    if not rows:
        return None, float("nan")
    row = min(rows, key=lambda item: abs(finite_float(item.get("time_ms"), 0.0) - target_ms))
    delta = finite_float(row.get("time_ms"), 0.0) - target_ms
    return row, delta


def accepted_rows_for_sessions(
    rows_by_sid: dict[str, list[dict[str, Any]]],
    *,
    prob_key: str,
) -> dict[str, list[dict[str, Any]]]:
    out: dict[str, list[dict[str, Any]]] = {}
    for sid, session_rows in rows_by_sid.items():
        out[sid] = accepted_after_dedupe(session_rows, prob_key, SELECTED_THRESHOLD, SELECTED_DEDUPE_MS)
    return out


def matched_truth_indices(pred_ms: list[float], truth_ms: list[float]) -> set[int]:
    matched = match_predictions(pred_ms, truth_ms, MATCH_TOLERANCE_MS)
    return {truth_idx for _, truth_idx, _ in matched["matches"]}


def accepted_identity(row: dict[str, Any]) -> tuple[str, int]:
    return (str(row.get("session_id", "")), intish(row.get("candidate_index")))


def classify_miss(
    *,
    truth_time_ms: float,
    nearest_candidate: dict[str, Any] | None,
    nearest_candidate_delta_ms: float,
    nearest_accepted_delta_ms: float,
    prob_key: str,
    accepted_ids: set[tuple[str, int]],
) -> str:
    if nearest_candidate is None or abs(nearest_candidate_delta_ms) > MATCH_TOLERANCE_MS:
        return "candidate_timing_gap"
    prob = finite_float(nearest_candidate.get(prob_key), 0.0)
    if prob < SELECTED_THRESHOLD:
        if prob >= 0.35:
            return "borderline_classifier_rejection"
        return "classifier_rejection"
    if accepted_identity(nearest_candidate) not in accepted_ids:
        return "dedupe_suppressed_candidate"
    if abs(nearest_accepted_delta_ms) > MATCH_TOLERANCE_MS:
        return "accepted_candidate_outside_tolerance"
    return "unclassified_miss"


def feature_summary(row: dict[str, Any] | None, prob_key: str) -> dict[str, Any]:
    if not row:
        return {
            "candidate_index": "",
            "candidate_time_ms": "",
            "candidate_delta_ms": "",
            "candidate_prob": "",
            "candidate_model_label": "",
            "candidate_model_prob_racket": "",
            "candidate_peak_ratio": "",
            "candidate_peak_z": "",
            "candidate_frame_rms": "",
            "candidate_prev_gap_ms": "",
            "candidate_next_gap_ms": "",
            "candidate_neighbor_count_500ms": "",
        }
    return {
        "candidate_index": intish(row.get("candidate_index")),
        "candidate_time_ms": round(finite_float(row.get("time_ms"), 0.0), 3),
        "candidate_prob": round(finite_float(row.get(prob_key), 0.0), 4),
        "candidate_model_label": row.get("model_label", ""),
        "candidate_model_prob_racket": round(finite_float(row.get("prob_racket_bounce"), 0.0), 4),
        "candidate_peak_ratio": round(finite_float(row.get("peak_ratio"), 0.0), 3),
        "candidate_peak_z": round(finite_float(row.get("peak_z"), 0.0), 3),
        "candidate_frame_rms": round(finite_float(row.get("frame_rms"), 0.0), 5),
        "candidate_prev_gap_ms": round(finite_float(row.get("prev_gap_ms"), 0.0), 1),
        "candidate_next_gap_ms": round(finite_float(row.get("next_gap_ms"), 0.0), 1),
        "candidate_neighbor_count_500ms": intish(row.get("neighbor_count_500ms")),
    }


def make_case_row(
    *,
    case_id: str,
    case_type: str,
    issue_bucket: str,
    source_set: str,
    truth: dict[str, Any] | None,
    candidate: dict[str, Any] | None,
    accepted: dict[str, Any] | None,
    prob_key: str,
    snippet_relpath: str,
) -> dict[str, Any]:
    time_ms = finite_float((truth or candidate or {}).get("time_ms"), 0.0)
    candidate_delta = ""
    accepted_delta = ""
    if truth and candidate:
        candidate_delta = round(finite_float(candidate.get("time_ms"), 0.0) - finite_float(truth.get("time_ms"), 0.0), 3)
    if truth and accepted:
        accepted_delta = round(finite_float(accepted.get("time_ms"), 0.0) - finite_float(truth.get("time_ms"), 0.0), 3)
    base = {
        "case_id": case_id,
        "case_type": case_type,
        "issue_bucket": issue_bucket,
        "source_set": source_set,
        "session_id": (truth or candidate or {}).get("session_id", ""),
        "scenario_id": (truth or candidate or {}).get("scenario_id", ""),
        "scenario_title": (truth or candidate or {}).get("scenario_title", ""),
        "label_index": (truth or {}).get("label_index", ""),
        "truth_time_ms": round(finite_float((truth or {}).get("time_ms"), float("nan")), 3) if truth else "",
        "event_time_ms": round(time_ms, 3),
        "nearest_candidate_delta_ms": candidate_delta,
        "nearest_accepted_delta_ms": accepted_delta,
        "selected_pipeline": SELECTED_PIPELINE_LABEL,
        "threshold": SELECTED_THRESHOLD,
        "dedupe_ms": SELECTED_DEDUPE_MS,
        "snippet": snippet_relpath,
        "manual_review": "",
        "manual_note": "",
    }
    base.update(feature_summary(candidate, prob_key))
    return base


def write_wav_snippet(
    *,
    wav_path: Path,
    out_path: Path,
    center_ms: float,
    pre_ms: float = SNIPPET_PRE_MS,
    post_ms: float = SNIPPET_POST_MS,
) -> None:
    y, sr = read_wav(wav_path)
    start = max(0, int(round((center_ms - pre_ms) * sr / 1000.0)))
    end = min(len(y), int(round((center_ms + post_ms) * sr / 1000.0)))
    if end <= start:
        end = min(len(y), start + max(1, int(sr * 0.1)))
    pcm = np.clip(y[start:end], -1.0, 1.0)
    data = (pcm * 32767.0).astype("<i2").tobytes()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with wave.open(str(out_path), "wb") as wav:
        wav.setnchannels(1)
        wav.setsampwidth(2)
        wav.setframerate(sr)
        wav.writeframes(data)


def wav_for_session(session_id: str, raw_dir: Path, heldout_wav: Path) -> Path:
    if session_id == HELDOUT_SESSION_ID:
        return heldout_wav
    return raw_dir / f"{session_id}.wav"


def safe_file_stem(value: str) -> str:
    stem = re.sub(r"[^A-Za-z0-9_.-]+", "_", value).strip("._")
    return stem or "session"


def waveform_envelope(y: np.ndarray, *, target_bins: int = 2400) -> list[float]:
    if len(y) == 0:
        return []
    bins = min(target_bins, max(1, len(y)))
    edges = np.linspace(0, len(y), bins + 1, dtype=np.int64)
    envelope: list[float] = []
    for start, end in zip(edges[:-1], edges[1:]):
        if end <= start:
            end = min(len(y), start + 1)
        envelope.append(round(float(np.max(np.abs(y[start:end]))), 5))
    peak = max(envelope) if envelope else 0.0
    if peak > 0:
        envelope = [round(value / peak, 5) for value in envelope]
    return envelope


def js_json(value: Any) -> str:
    return json.dumps(value, ensure_ascii=True).replace("</", "<\\/")


def render_session_wave_page(
    *,
    session_id: str,
    wav_relpath: str,
    duration_s: float,
    waveform: list[float],
    cases: list[dict[str, Any]],
    generated_at: str,
) -> str:
    case_payload = [
        {
            "case_id": row.get("case_id", ""),
            "case_type": row.get("case_type", ""),
            "scenario_title": row.get("scenario_title", ""),
            "issue_bucket": row.get("issue_bucket", ""),
            "time_s": round(finite_float(row.get("event_time_ms"), 0.0) / 1000.0, 4),
            "candidate_prob": row.get("candidate_prob", ""),
            "snippet": row.get("snippet", ""),
        }
        for row in cases
    ]
    table_rows = []
    for row in cases:
        case_id = html.escape(str(row.get("case_id", "")))
        table_rows.append(
            "<tr>"
            f"<td><button type=\"button\" data-case-id=\"{case_id}\">Jump</button></td>"
            f"<td>{case_id}</td>"
            f"<td>{html.escape(str(row.get('case_type', '')))}</td>"
            f"<td>{html.escape(str(row.get('scenario_title', '')))}</td>"
            f"<td>{html.escape(str(row.get('event_time_ms', '')))}</td>"
            f"<td>{html.escape(str(row.get('issue_bucket', '')))}</td>"
            f"<td>{html.escape(str(row.get('candidate_prob', '')))}</td>"
            "</tr>"
        )
    audio_src = "../" + wav_relpath
    return f"""<!doctype html>
<html lang="en">
<head>
  <meta charset="utf-8" />
  <title>T0073 Full Wave - {html.escape(session_id)}</title>
  <style>
    :root {{
      color-scheme: light;
      --ink: #16202c;
      --muted: #627184;
      --line: #d7dde5;
      --target: #e11d48;
      --playhead: #0369a1;
      --wave: #43515f;
      --soft: #f5f7fa;
    }}
    body {{
      font-family: Arial, sans-serif;
      margin: 24px;
      color: var(--ink);
      background: white;
    }}
    header {{
      display: flex;
      align-items: flex-start;
      justify-content: space-between;
      gap: 16px;
      margin-bottom: 16px;
    }}
    h1 {{
      margin: 0 0 6px;
      font-size: 24px;
    }}
    .meta, .hint {{
      color: var(--muted);
      font-size: 13px;
      line-height: 1.45;
    }}
    .panel {{
      border: 1px solid var(--line);
      border-radius: 8px;
      padding: 14px;
      margin: 14px 0;
      background: #fff;
    }}
    audio {{
      width: 100%;
      margin-top: 8px;
    }}
    canvas {{
      display: block;
      width: 100%;
      height: 320px;
      border: 1px solid var(--line);
      border-radius: 8px;
      background: #fbfcfd;
      cursor: crosshair;
    }}
    .toolbar {{
      display: flex;
      flex-wrap: wrap;
      gap: 8px;
      align-items: center;
      margin: 10px 0;
    }}
    button, a.button {{
      border: 1px solid #aeb9c6;
      border-radius: 6px;
      background: white;
      color: var(--ink);
      font-size: 13px;
      padding: 7px 10px;
      text-decoration: none;
      cursor: pointer;
    }}
    button:hover, a.button:hover {{
      border-color: #6b7a8a;
      background: var(--soft);
    }}
    table {{
      border-collapse: collapse;
      width: 100%;
      margin-top: 12px;
    }}
    th, td {{
      border: 1px solid var(--line);
      padding: 8px;
      vertical-align: top;
      font-size: 13px;
    }}
    th {{
      background: var(--soft);
      text-align: left;
    }}
    .active-case {{
      background: #fff1f2;
    }}
    .target-label {{
      color: var(--target);
      font-weight: 700;
    }}
  </style>
</head>
<body>
  <header>
    <div>
      <h1>Full Wave Review</h1>
      <div class="meta">Session: <code>{html.escape(session_id)}</code></div>
      <div class="meta">Generated at {html.escape(generated_at)}. Duration {duration_s:.2f}s.</div>
    </div>
    <a class="button" href="../index.html">Back to index</a>
  </header>

  <section class="panel">
    <div class="meta">The red vertical marker is the bad-case timestamp to check. Click a marker or a row to jump near it, then press Play to hear the original continuous recording around it.</div>
    <audio id="audio" controls preload="metadata" src="{html.escape(audio_src)}"></audio>
    <div class="toolbar">
      <button type="button" id="jumpBefore">Jump 1s before active marker</button>
      <button type="button" id="jumpExact">Jump exactly to active marker</button>
      <span class="meta" id="activeMeta"></span>
    </div>
    <canvas id="waveform"></canvas>
    <div class="hint">Click anywhere on the waveform to move the audio playhead. Red lines are bad-case markers; the active one is thicker and has a red triangle.</div>
  </section>

  <section class="panel">
    <h2>Cases in this full WAV</h2>
    <table>
      <thead>
        <tr><th></th><th>Case</th><th>Type</th><th>Scenario</th><th>Time ms</th><th>Issue</th><th>Prob</th></tr>
      </thead>
      <tbody>
        {''.join(table_rows)}
      </tbody>
    </table>
  </section>

  <script>
    const durationS = {duration_s:.6f};
    const waveform = {js_json(waveform)};
    const cases = {js_json(case_payload)};
    const audio = document.getElementById('audio');
    const canvas = document.getElementById('waveform');
    const ctx = canvas.getContext('2d');
    const activeMeta = document.getElementById('activeMeta');
    let activeCaseId = (location.hash || '').replace('#', '') || (cases[0] && cases[0].case_id) || '';

    function activeCase() {{
      return cases.find((item) => item.case_id === activeCaseId) || cases[0] || null;
    }}

    function xForTime(timeS, width) {{
      if (!durationS) return 0;
      return Math.max(0, Math.min(width, (timeS / durationS) * width));
    }}

    function timeForX(x, width) {{
      if (!width) return 0;
      return Math.max(0, Math.min(durationS, (x / width) * durationS));
    }}

    function draw() {{
      const width = canvas.width;
      const height = canvas.height;
      const dpr = window.devicePixelRatio || 1;
      ctx.clearRect(0, 0, width, height);
      ctx.fillStyle = '#fbfcfd';
      ctx.fillRect(0, 0, width, height);

      const mid = height * 0.52;
      const amp = height * 0.42;
      ctx.strokeStyle = '#e1e7ee';
      ctx.lineWidth = 1 * dpr;
      ctx.beginPath();
      ctx.moveTo(0, mid);
      ctx.lineTo(width, mid);
      ctx.stroke();

      const gridStep = durationS > 90 ? 10 : durationS > 35 ? 5 : 1;
      ctx.fillStyle = '#8a96a5';
      ctx.font = `${{12 * dpr}}px Arial`;
      ctx.textAlign = 'center';
      for (let t = 0; t <= durationS; t += gridStep) {{
        const x = xForTime(t, width);
        ctx.strokeStyle = t % (gridStep * 2) === 0 ? '#d7dde5' : '#edf1f5';
        ctx.beginPath();
        ctx.moveTo(x, 0);
        ctx.lineTo(x, height);
        ctx.stroke();
        if (t > 0 && t < durationS) {{
          ctx.fillText(`${{t.toFixed(0)}}s`, x, height - 8 * dpr);
        }}
      }}

      ctx.strokeStyle = '#43515f';
      ctx.lineWidth = 1 * dpr;
      const binWidth = width / Math.max(1, waveform.length);
      for (let i = 0; i < waveform.length; i += 1) {{
        const x = i * binWidth;
        const value = Math.max(0.002, waveform[i] || 0);
        ctx.beginPath();
        ctx.moveTo(x, mid - value * amp);
        ctx.lineTo(x, mid + value * amp);
        ctx.stroke();
      }}

      cases.forEach((item) => {{
        const x = xForTime(item.time_s, width);
        const isActive = item.case_id === activeCaseId;
        ctx.strokeStyle = 'rgba(225, 29, 72, 0.82)';
        ctx.lineWidth = (isActive ? 4 : 2) * dpr;
        ctx.beginPath();
        ctx.moveTo(x, 0);
        ctx.lineTo(x, height);
        ctx.stroke();
        if (isActive) {{
          ctx.fillStyle = '#e11d48';
          ctx.beginPath();
          ctx.moveTo(x, 0);
          ctx.lineTo(x - 8 * dpr, 18 * dpr);
          ctx.lineTo(x + 8 * dpr, 18 * dpr);
          ctx.closePath();
          ctx.fill();
        }}
      }});

      const playX = xForTime(audio.currentTime || 0, width);
      ctx.strokeStyle = '#0369a1';
      ctx.lineWidth = 2 * dpr;
      ctx.beginPath();
      ctx.moveTo(playX, 0);
      ctx.lineTo(playX, height);
      ctx.stroke();

      const current = activeCase();
      if (current) {{
        activeMeta.innerHTML = `Active: <span class="target-label">${{current.case_id}}</span> at ${{current.time_s.toFixed(3)}}s`;
      }}
      document.querySelectorAll('tbody tr').forEach((tr) => {{
        const button = tr.querySelector('button[data-case-id]');
        tr.classList.toggle('active-case', button && button.dataset.caseId === activeCaseId);
      }});
    }}

    function resizeCanvas() {{
      const dpr = window.devicePixelRatio || 1;
      const rect = canvas.getBoundingClientRect();
      canvas.width = Math.max(800, Math.floor(rect.width * dpr));
      canvas.height = Math.floor(320 * dpr);
      draw();
    }}

    function selectCase(caseId, seekMode) {{
      activeCaseId = caseId;
      if (location.hash !== `#${{caseId}}`) {{
        history.replaceState(null, '', `#${{caseId}}`);
      }}
      const current = activeCase();
      if (current && seekMode) {{
        audio.currentTime = seekMode === 'exact' ? current.time_s : Math.max(0, current.time_s - 1.0);
      }}
      draw();
    }}

    document.querySelectorAll('button[data-case-id]').forEach((button) => {{
      button.addEventListener('click', () => selectCase(button.dataset.caseId, 'before'));
    }});

    document.getElementById('jumpBefore').addEventListener('click', () => {{
      const current = activeCase();
      if (current) audio.currentTime = Math.max(0, current.time_s - 1.0);
      draw();
    }});

    document.getElementById('jumpExact').addEventListener('click', () => {{
      const current = activeCase();
      if (current) audio.currentTime = current.time_s;
      draw();
    }});

    canvas.addEventListener('click', (event) => {{
      const rect = canvas.getBoundingClientRect();
      const xCss = event.clientX - rect.left;
      const timeS = timeForX(xCss * (window.devicePixelRatio || 1), canvas.width);
      audio.currentTime = timeS;
      let nearest = null;
      let nearestDelta = Infinity;
      cases.forEach((item) => {{
        const delta = Math.abs(item.time_s - timeS);
        if (delta < nearestDelta) {{
          nearest = item;
          nearestDelta = delta;
        }}
      }});
      if (nearest && nearestDelta <= Math.max(0.25, durationS * 0.01)) {{
        selectCase(nearest.case_id, null);
      }}
      draw();
    }});

    audio.addEventListener('timeupdate', draw);
    audio.addEventListener('seeked', draw);
    window.addEventListener('resize', resizeCanvas);
    window.addEventListener('hashchange', () => {{
      const id = (location.hash || '').replace('#', '');
      if (id) selectCase(id, 'before');
    }});

    resizeCanvas();
    const initial = activeCase();
    if (initial) {{
      audio.currentTime = Math.max(0, initial.time_s - 1.0);
      draw();
    }}
  </script>
</body>
</html>
"""


def write_full_wave_pages(
    cases: list[dict[str, Any]],
    *,
    raw_dir: Path,
    heldout_wav: Path,
    out_dir: Path,
    generated_at: str,
) -> None:
    by_session: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in cases:
        by_session[str(row.get("session_id", ""))].append(row)

    full_wavs_dir = out_dir / "full_wavs"
    pages_dir = out_dir / "full_wave_pages"
    full_wavs_dir.mkdir(parents=True, exist_ok=True)
    pages_dir.mkdir(parents=True, exist_ok=True)

    used_stems: dict[str, str] = {}
    for session_id, session_cases in sorted(by_session.items()):
        stem = safe_file_stem(session_id)
        if stem in used_stems and used_stems[stem] != session_id:
            stem = f"{stem}_{len(used_stems) + 1}"
        used_stems[stem] = session_id

        wav_path = wav_for_session(session_id, raw_dir, heldout_wav)
        full_wav_path = full_wavs_dir / f"{stem}.wav"
        shutil.copy2(wav_path, full_wav_path)
        y, sr = read_wav(wav_path)
        duration_s = len(y) / float(sr) if sr else 0.0
        page_path = pages_dir / f"{stem}.html"
        wav_relpath = full_wav_path.relative_to(out_dir).as_posix()
        page_relpath = page_path.relative_to(out_dir).as_posix()
        for row in session_cases:
            row["full_wave"] = f"{page_relpath}#{row.get('case_id', '')}"
            row["full_wav"] = wav_relpath
        page_path.write_text(
            render_session_wave_page(
                session_id=session_id,
                wav_relpath=wav_relpath,
                duration_s=duration_s,
                waveform=waveform_envelope(y),
                cases=sorted(session_cases, key=lambda item: finite_float(item.get("event_time_ms"), 0.0)),
                generated_at=generated_at,
            ),
            encoding="utf-8",
        )


def extract_bad_cases(
    *,
    t0071_dir: Path,
    t0072_dir: Path,
    heldout_labels: Path,
    raw_dir: Path,
    heldout_wav: Path,
    out_dir: Path,
) -> list[dict[str, Any]]:
    snippets_dir = out_dir / "snippets"
    cases: list[dict[str, Any]] = []

    round_truth = truth_by_session(t0071_dir)
    round_rows = selected_oof_rows(t0072_dir)
    round_by_sid = rows_by_session(round_rows)
    accepted_by_sid = accepted_rows_for_sessions(round_by_sid, prob_key="oof_prob")
    accepted_ids = {accepted_identity(row) for rows in accepted_by_sid.values() for row in rows}

    case_counter = 1
    for sid, truth_rows in sorted(round_truth.items()):
        candidates = round_by_sid.get(sid, [])
        accepted = accepted_by_sid.get(sid, [])
        pred_ms = [finite_float(row.get("time_ms"), 0.0) for row in accepted]
        truth_ms = [finite_float(row.get("time_ms"), 0.0) for row in truth_rows]
        matched_truth = matched_truth_indices(pred_ms, truth_ms)
        for idx, truth in enumerate(truth_rows):
            if idx in matched_truth:
                continue
            truth_time = finite_float(truth["time_ms"])
            nearest_candidate, nearest_candidate_delta = nearest_row(candidates, truth_time)
            nearest_accepted, nearest_accepted_delta = nearest_row(accepted, truth_time)
            issue = classify_miss(
                truth_time_ms=truth_time,
                nearest_candidate=nearest_candidate,
                nearest_candidate_delta_ms=nearest_candidate_delta,
                nearest_accepted_delta_ms=nearest_accepted_delta,
                prob_key="oof_prob",
                accepted_ids=accepted_ids,
            )
            case_id = f"round_miss_{case_counter:03d}"
            snippet_path = snippets_dir / f"{case_id}_{sid}_{int(round(truth_time)):06d}ms.wav"
            write_wav_snippet(wav_path=wav_for_session(sid, raw_dir, heldout_wav), out_path=snippet_path, center_ms=truth_time)
            row = make_case_row(
                case_id=case_id,
                case_type="missed_round_a_positive",
                issue_bucket=issue,
                source_set="round_a_oof",
                truth=truth,
                candidate=nearest_candidate,
                accepted=nearest_accepted,
                prob_key="oof_prob",
                snippet_relpath=snippet_path.relative_to(out_dir).as_posix(),
            )
            if nearest_candidate:
                row["nearest_candidate_delta_ms"] = round(nearest_candidate_delta, 3)
            cases.append(row)
            case_counter += 1

    for sid, accepted in sorted(accepted_by_sid.items()):
        if sid in round_truth:
            continue
        for candidate in accepted:
            case_id = f"round_fp_{case_counter:03d}"
            event_time = finite_float(candidate.get("time_ms"), 0.0)
            snippet_path = snippets_dir / f"{case_id}_{sid}_{int(round(event_time)):06d}ms.wav"
            write_wav_snippet(wav_path=wav_for_session(sid, raw_dir, heldout_wav), out_path=snippet_path, center_ms=event_time)
            cases.append(
                make_case_row(
                    case_id=case_id,
                    case_type="hard_negative_false_count",
                    issue_bucket="unsafe_false_positive",
                    source_set="round_a_oof",
                    truth=None,
                    candidate=candidate,
                    accepted=candidate,
                    prob_key="oof_prob",
                    snippet_relpath=snippet_path.relative_to(out_dir).as_posix(),
                )
            )
            case_counter += 1

    heldout_truth = heldout_truth_rows(heldout_labels)
    heldout_rows = selected_heldout_rows(t0072_dir)
    heldout_by_sid = rows_by_session(heldout_rows)
    heldout_accepted_by_sid = accepted_rows_for_sessions(heldout_by_sid, prob_key="clf_prob")
    heldout_accepted_ids = {accepted_identity(row) for rows in heldout_accepted_by_sid.values() for row in rows}
    heldout_candidates = heldout_by_sid.get(HELDOUT_SESSION_ID, [])
    heldout_accepted = heldout_accepted_by_sid.get(HELDOUT_SESSION_ID, [])
    pred_ms = [finite_float(row.get("time_ms"), 0.0) for row in heldout_accepted]
    truth_ms = [finite_float(row.get("time_ms"), 0.0) for row in heldout_truth]
    matched_truth = matched_truth_indices(pred_ms, truth_ms)
    for idx, truth in enumerate(heldout_truth):
        if idx in matched_truth:
            continue
        truth_time = finite_float(truth["time_ms"])
        nearest_candidate, nearest_candidate_delta = nearest_row(heldout_candidates, truth_time)
        nearest_accepted, nearest_accepted_delta = nearest_row(heldout_accepted, truth_time)
        issue = classify_miss(
            truth_time_ms=truth_time,
            nearest_candidate=nearest_candidate,
            nearest_candidate_delta_ms=nearest_candidate_delta,
            nearest_accepted_delta_ms=nearest_accepted_delta,
            prob_key="clf_prob",
            accepted_ids=heldout_accepted_ids,
        )
        case_id = f"heldout_miss_{case_counter:03d}"
        snippet_path = snippets_dir / f"{case_id}_{int(round(truth_time)):06d}ms.wav"
        write_wav_snippet(wav_path=heldout_wav, out_path=snippet_path, center_ms=truth_time)
        row = make_case_row(
            case_id=case_id,
            case_type="missed_heldout_c2_positive",
            issue_bucket=issue,
            source_set="heldout_c2_final_fit",
            truth=truth,
            candidate=nearest_candidate,
            accepted=nearest_accepted,
            prob_key="clf_prob",
            snippet_relpath=snippet_path.relative_to(out_dir).as_posix(),
        )
        if nearest_candidate:
            row["nearest_candidate_delta_ms"] = round(nearest_candidate_delta, 3)
        cases.append(row)
        case_counter += 1

    return cases


def count_rows(rows: list[dict[str, Any]], *keys: str) -> list[dict[str, Any]]:
    counter: Counter[tuple[Any, ...]] = Counter(tuple(row.get(key, "") for key in keys) for row in rows)
    out = []
    for values, count in counter.most_common():
        out.append({key: value for key, value in zip(keys, values)} | {"count": count})
    return out


def apply_manual_review(cases: list[dict[str, Any]]) -> None:
    for row in cases:
        review = MANUAL_REVIEW_BY_CASE.get(str(row.get("case_id", "")))
        if not review:
            continue
        row["manual_review"] = review["manual_review"]
        row["manual_note"] = review["manual_note"]


def recommendation(cases: list[dict[str, Any]]) -> tuple[str, list[str]]:
    false_counts = [row for row in cases if row["case_type"] == "hard_negative_false_count"]
    talking_fp = [row for row in false_counts if row.get("scenario_id") == "talking_only_no_bounce"]
    rejected_fp = [row for row in false_counts if row.get("manual_review") == "reject_unsafe_false_positive"]
    accepted_like_fp = [row for row in false_counts if row.get("manual_review") == "acceptable_bounce_like"]
    heldout_miss = [row for row in cases if row["case_type"] == "missed_heldout_c2_positive"]
    round_miss = [row for row in cases if row["case_type"] == "missed_round_a_positive"]
    background_miss = [row for row in round_miss if row.get("scenario_id") == "racket_bounce_background_sound"]
    reasons = [
        f"Round A misses: {len(round_miss)} total, {len(background_miss)} background-sound.",
        f"Hard-negative false counts: {len(false_counts)} total, {len(talking_fp)} talking-only.",
        f"Manual hard-negative review: {len(accepted_like_fp)} acceptable bounce-like, {len(rejected_fp)} rejected unsafe.",
        f"Held-out C2 misses: {len(heldout_miss)}.",
    ]
    if rejected_fp:
        return "app_style_parity_plus_threshold_or_safety_gate_before_apk", reasons
    if len(false_counts) <= 5 and not talking_fp and len(heldout_miss) <= 5:
        return "proceed_to_app_style_export_parity_after_manual_false_positive_review", reasons
    if talking_fp or len(false_counts) > 10:
        return "do_not_export_until_hard_negative_safety_improves", reasons
    return "collect_or_review_more_data_before_export", reasons


def render_markdown(cases: list[dict[str, Any]], summary: dict[str, Any]) -> str:
    rec, reasons = recommendation(cases)
    false_counts = [row for row in cases if row["case_type"] == "hard_negative_false_count"]
    manual_rows = [row for row in cases if row["case_type"] in {"hard_negative_false_count", "missed_heldout_c2_positive"}]
    lines = [
        "# T0073 Fable Candidate Bad-Case Pack",
        "",
        f"Generated at: `{summary['generated_at']}`",
        f"Selected policy: `{SELECTED_PIPELINE_LABEL}`",
        "",
        "## Recommendation",
        "",
        f"- Recommendation: `{rec}`",
        *[f"- {reason}" for reason in reasons],
        "",
        "This is still evaluation-only. No model JSON, app runtime, APK, cloud/API, or camera behavior changed.",
        "",
        "## Bad-Case Counts",
        "",
        *md_table(count_rows(cases, "case_type"), ["case_type", "count"], ["Case Type", "Count"]),
        "",
        "## Issue Buckets",
        "",
        *md_table(count_rows(cases, "case_type", "issue_bucket"), ["case_type", "issue_bucket", "count"], ["Case Type", "Issue", "Count"]),
        "",
        "## Scenario Counts",
        "",
        *md_table(count_rows(cases, "case_type", "scenario_title"), ["case_type", "scenario_title", "count"], ["Case Type", "Scenario", "Count"]),
        "",
        "## Hard-Negative False Counts",
        "",
        *md_table(
            false_counts,
            [
                "case_id",
                "scenario_title",
                "event_time_ms",
                "candidate_prob",
                "candidate_model_label",
                "candidate_model_prob_racket",
                "candidate_peak_ratio",
                "manual_review",
                "full_wave",
                "snippet",
            ],
            ["Case", "Scenario", "Time ms", "Clf Prob", "Fable Label", "Fable Racket", "Peak Ratio", "Manual Review", "Full Wave", "Snippet"],
        ),
        "",
        "Manual result: `round_fp_021`-`round_fp_023` are acceptable/ambiguous because they sound ball-bounce-like to Love; `round_fp_024` and `round_fp_025` should not count and must be treated as hard-negative safety rows before any APK install.",
        "",
        "## Manual Review Priority",
        "",
        "Listen to these first: all hard-negative false counts, then the held-out C2 misses. Use the full-wave links when the short snippet is too ambiguous. Round A background misses are lower priority unless export parity looks good.",
        "",
        *md_table(
            manual_rows,
            ["case_id", "case_type", "scenario_title", "event_time_ms", "issue_bucket", "candidate_prob", "manual_review", "manual_note", "full_wave", "snippet"],
            ["Case", "Type", "Scenario", "Time ms", "Issue", "Clf Prob", "Manual Review", "Manual Note", "Full Wave", "Snippet"],
        ),
        "",
        "## Outputs",
        "",
        "- `t0073_bad_cases.csv`",
        "- `t0073_bad_case_summary.json`",
        "- `t0073_bad_case_report.md`",
        "- `index.html`",
        "- `snippets/*.wav`",
        "- `full_wavs/*.wav`",
        "- `full_wave_pages/*.html`",
    ]
    return "\n".join(lines) + "\n"


def render_html(cases: list[dict[str, Any]], summary: dict[str, Any]) -> str:
    rows = []
    for row in cases:
        snippet = html.escape(str(row.get("snippet", "")))
        full_wave = html.escape(str(row.get("full_wave", "")))
        rows.append(
            "<tr>"
            f"<td>{html.escape(str(row.get('case_id', '')))}</td>"
            f"<td>{html.escape(str(row.get('case_type', '')))}</td>"
            f"<td>{html.escape(str(row.get('scenario_title', '')))}</td>"
            f"<td>{html.escape(str(row.get('event_time_ms', '')))}</td>"
            f"<td>{html.escape(str(row.get('issue_bucket', '')))}</td>"
            f"<td>{html.escape(str(row.get('candidate_prob', '')))}</td>"
            f"<td>{html.escape(str(row.get('manual_review', '')))}<br><span class=\"note\">{html.escape(str(row.get('manual_note', '')))}</span></td>"
            f"<td><a href=\"{full_wave}\">full wave</a></td>"
            f"<td><audio controls src=\"{snippet}\"></audio></td>"
            "<td>accept / bad / unsure</td>"
            "</tr>"
        )
    return f"""<!doctype html>
<html lang="en">
<head>
  <meta charset="utf-8" />
  <title>T0073 Fable Bad Cases</title>
  <style>
    body {{ font-family: Arial, sans-serif; margin: 24px; color: #18202a; }}
    table {{ border-collapse: collapse; width: 100%; }}
    th, td {{ border: 1px solid #d7dde5; padding: 8px; vertical-align: top; font-size: 13px; }}
    th {{ background: #f3f6f9; text-align: left; }}
    audio {{ width: 220px; }}
    .meta {{ color: #596673; }}
    .note {{ color: #596673; font-size: 12px; }}
  </style>
</head>
<body>
  <h1>T0073 Fable Bad Cases</h1>
  <p class="meta">Generated at {html.escape(summary['generated_at'])}. Selected policy: {html.escape(SELECTED_PIPELINE_LABEL)}.</p>
  <p>Manual note: listen to hard-negative false counts first, then held-out C2 misses. Use <strong>full wave</strong> when the snippet is too short to tell whether the sound belongs to a real bounce.</p>
  <table>
    <thead>
      <tr><th>Case</th><th>Type</th><th>Scenario</th><th>Time ms</th><th>Issue</th><th>Prob</th><th>Manual Review</th><th>Full Wave</th><th>Snippet</th><th>Your note</th></tr>
    </thead>
    <tbody>
      {''.join(rows)}
    </tbody>
  </table>
</body>
</html>
"""


def run(args: argparse.Namespace) -> dict[str, Any]:
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    cases = extract_bad_cases(
        t0071_dir=Path(args.t0071_dir),
        t0072_dir=Path(args.t0072_dir),
        heldout_labels=Path(args.heldout_labels),
        raw_dir=Path(args.raw_dir),
        heldout_wav=Path(args.heldout_wav),
        out_dir=out_dir,
    )
    apply_manual_review(cases)
    generated_at = datetime.now(timezone.utc).isoformat()
    write_full_wave_pages(
        cases,
        raw_dir=Path(args.raw_dir),
        heldout_wav=Path(args.heldout_wav),
        out_dir=out_dir,
        generated_at=generated_at,
    )
    rec, reasons = recommendation(cases)
    summary = {
        "generated_at": generated_at,
        "ticket": "T0073-fable-candidate-bad-case-export-prep",
        "selected_pipeline_id": SELECTED_PIPELINE_ID,
        "selected_pipeline_label": SELECTED_PIPELINE_LABEL,
        "threshold": SELECTED_THRESHOLD,
        "dedupe_ms": SELECTED_DEDUPE_MS,
        "match_tolerance_ms": MATCH_TOLERANCE_MS,
        "case_count": len(cases),
        "case_counts": count_rows(cases, "case_type"),
        "issue_counts": count_rows(cases, "case_type", "issue_bucket"),
        "scenario_counts": count_rows(cases, "case_type", "scenario_title"),
        "manual_review_counts": count_rows(cases, "manual_review"),
        "recommendation": rec,
        "recommendation_reasons": reasons,
    }
    write_csv(out_dir / "t0073_bad_cases.csv", cases)
    (out_dir / "t0073_bad_case_summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    (out_dir / "t0073_bad_case_report.md").write_text(render_markdown(cases, summary), encoding="utf-8")
    (out_dir / "index.html").write_text(render_html(cases, summary), encoding="utf-8")
    return summary


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--raw-dir", default=str(DEFAULT_RAW_DIR))
    parser.add_argument("--t0071-dir", default=str(DEFAULT_T0071_DIR))
    parser.add_argument("--t0072-dir", default=str(DEFAULT_T0072_DIR))
    parser.add_argument("--heldout-labels", default=str(DEFAULT_T0063_LABELS))
    parser.add_argument("--heldout-wav", default=str(DEFAULT_HELDOUT_WAV))
    parser.add_argument("--out-dir", default=str(DEFAULT_OUT_DIR))
    print(json.dumps(run(parser.parse_args()), indent=2))


if __name__ == "__main__":
    main()
