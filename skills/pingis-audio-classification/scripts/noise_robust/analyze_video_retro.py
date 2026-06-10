"""
analyze_video_retro.py

Punkt 3: retrospektiv analys av en inspelad videosekvens (eller WAV).

1. Extraherar ljudspåret med ffmpeg till 22 050 Hz mono om input är video.
2. Kör den brusrobusta detektorkaskaden (bandpass-gate 1.5-7 kHz, ingen
   spektralgate, retrigger 120 ms) och klassificerar varje trigger med
   noise_robust v3-modellen (HistGB all83): racket / bord / golv / brus.
3. Mappar ut varje bordsstuds och racketstuds på tidslinjen.
4. Räknar slag över nät med alternerings- och rimlighetslogik:
   - En bollväxling (rally) består av alternerande racket- och bordsstudsar.
   - Ett "slag över nät" räknas för en racketträff som följs av en
     bordsstuds inom MAX_FLIGHT_MS (bollen landade på andra sidan).
   - Om gapet mellan två händelser överstiger RALLY_TIMEOUT_MS är bollen
     ute / bollväxlingen bruten och en ny rally startar vid nästa händelse.

Output (prefix väljs med --out-prefix):
  <prefix>_timeline.csv   en rad per detekterad händelse
  <prefix>_rallies.csv    en rad per bollväxling
  <prefix>_report.md      sammanfattning

Validering mot facit: ge --session-json <reviewed session.json> så matchas
tidslinjen mot granskade racket/bordsmarkörer (140 ms tolerans) och
precision/recall per klass rapporteras.

Exempel:
  python .../analyze_video_retro.py --input match.mp4
  python .../analyze_video_retro.py --input data/audio/raw/audio_session_2026-06-04_006/free_recording_001.wav \
      --session-json data/audio/raw/audio_session_2026-06-04_006.json
"""

from __future__ import annotations

import argparse
import csv
import json
import subprocess
import sys
import tempfile
from pathlib import Path

import joblib
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import nr_config  # noqa: E402
import nr_features  # noqa: E402
from preprocess_audio import FFMPEG, load_audio, is_trainable_review_marker  # noqa: E402

ROOT_DIR = Path(__file__).resolve().parents[4]
MODEL_DIR = ROOT_DIR / "data" / "audio" / "models" / "noise_robust_v3"

# Detektorprofil (samma som Fable-läget/replayens känsliga profil).
GATE_KWARGS = dict(onset_ratio=1.5, retrigger_ms=120, abs_min_rms=0.0015, mode="bandpass", spectral_gate=False)
RACKET_CONF_MIN = 0.5
TABLE_CONF_MIN = 0.5
MERGE_MS = 120
ECHO_MS = 300
ECHO_RATIO = 0.6

# Rimlighetslogik för bollväxlingar.
RALLY_TIMEOUT_MS = 2_000   # längre tystnad än så = bollen ute / växling bruten
MAX_FLIGHT_MS = 1_200      # racket -> bord måste ske inom rimlig flygtid
MIN_RALLY_EVENTS = 2       # en ensam smäll är ingen bollväxling

VIDEO_SUFFIXES = {".mp4", ".mov", ".mkv", ".avi", ".m4v", ".webm", ".m4a", ".aac", ".mp3", ".flac", ".ogg"}


def extract_audio(input_path: Path) -> tuple[np.ndarray, int, Path | None]:
    """Ladda ljud; konvertera via ffmpeg om input inte är WAV @ rätt format."""
    if input_path.suffix.lower() in VIDEO_SUFFIXES:
        tmp = Path(tempfile.mkstemp(suffix=".wav")[1])
        cmd = [FFMPEG, "-y", "-i", str(input_path), "-vn", "-ac", "1", "-ar", "22050",
               "-acodec", "pcm_s16le", str(tmp)]
        proc = subprocess.run(cmd, capture_output=True, text=True)
        if proc.returncode != 0:
            raise SystemExit(f"ffmpeg failed: {proc.stderr[-800:]}")
        y, sr = load_audio(str(tmp))
        return y, sr, tmp
    y, sr = load_audio(str(input_path))
    return y, sr, None


def detect_events(y: np.ndarray, sr: int, model, scaler, feature_cols, labels) -> list[dict]:
    """Gate + klassificering + merge/eko-dedup -> tidslinje av events."""
    triggers = nr_features.simulate_gate(y, sr, **GATE_KWARGS)
    events: list[dict] = []
    last_counted: dict | None = None
    for trig in triggers:
        clip = nr_features.extract_live_clip(y, int(trig["onset_sample"]))
        feats = nr_features.extract_all_features(clip, sr)
        x = np.array([[float(feats.get(c, 0.0)) for c in feature_cols]], dtype=np.float64)
        x = np.nan_to_num(x, nan=0.0, posinf=0.0, neginf=0.0)
        probs = model.predict_proba(scaler.transform(x))[0]
        best = int(np.argmax(probs))
        label = labels[best]
        conf = float(probs[best])

        if label == "racket_bounce" and conf < RACKET_CONF_MIN:
            continue
        if label == "table_bounce" and conf < TABLE_CONF_MIN:
            continue
        if label not in ("racket_bounce", "table_bounce"):
            continue

        ts = float(trig["onset_ms"])
        if last_counted is not None:
            gap = ts - last_counted["ts_ms"]
            if gap <= MERGE_MS:
                continue
            if (
                gap <= ECHO_MS
                and trig["frame_rms"] <= ECHO_RATIO * last_counted["frame_rms"]
                and label == last_counted["label"]
            ):
                continue
        event = {
            "ts_ms": ts,
            "label": label,
            "confidence": round(conf, 4),
            "frame_rms": float(trig["frame_rms"]),
            "p_racket": round(float(probs[labels.index("racket_bounce")]), 4),
            "p_table": round(float(probs[labels.index("table_bounce")]), 4),
        }
        events.append(event)
        last_counted = event
    return events


def segment_rallies(events: list[dict]) -> list[dict]:
    """Dela upp tidslinjen i bollväxlingar och räkna slag över nät."""
    rallies: list[dict] = []
    current: list[dict] = []

    def close_rally(reason: str) -> None:
        if len(current) < MIN_RALLY_EVENTS:
            current.clear()
            return
        strokes_over_net = 0
        rackets = [e for e in current if e["label"] == "racket_bounce"]
        for i, e in enumerate(current):
            if e["label"] != "racket_bounce":
                continue
            nxt = current[i + 1] if i + 1 < len(current) else None
            if nxt is not None and nxt["label"] == "table_bounce" and nxt["ts_ms"] - e["ts_ms"] <= MAX_FLIGHT_MS:
                strokes_over_net += 1
        rallies.append({
            "start_ms": current[0]["ts_ms"],
            "end_ms": current[-1]["ts_ms"],
            "duration_s": round((current[-1]["ts_ms"] - current[0]["ts_ms"]) / 1000, 2),
            "n_events": len(current),
            "n_racket": len(rackets),
            "n_table": sum(1 for e in current if e["label"] == "table_bounce"),
            "strokes_over_net": strokes_over_net,
            "ended_by": reason,
        })
        current.clear()

    for event in events:
        if current and event["ts_ms"] - current[-1]["ts_ms"] > RALLY_TIMEOUT_MS:
            close_rally("timeout_ball_out")
        current.append(event)
    close_rally("end_of_recording")
    return rallies


def score_against_truth(events: list[dict], session_json: Path, wav_filename: str | None) -> dict:
    session = json.loads(session_json.read_text(encoding="utf-8"))
    racket_truth: list[int] = []
    table_truth: list[int] = []
    for ev in session.get("events", []):
        if wav_filename and ev.get("wav_filename") != wav_filename:
            continue
        for marker in (ev.get("review") or {}).get("markers", []):
            if not is_trainable_review_marker(marker):
                continue
            ts = int(marker.get("timestamp_ms", 0))
            final = marker.get("final_label")
            kind = str(marker.get("class_label") or marker.get("contact_kind") or marker.get("not_racket_kind") or "")
            if final == "racket_contact":
                racket_truth.append(ts)
            elif final == "not_racket_contact" and kind == "table_bounce":
                table_truth.append(ts)

    def match(dets: list[float], truth: list[int], tol: int = 140) -> dict:
        matched: set[int] = set()
        tp = 0
        fp = 0
        for det in sorted(dets):
            best_idx, best_gap = None, tol + 1
            for idx, ts in enumerate(truth):
                gap = abs(det - ts)
                if gap <= tol and gap < best_gap and idx not in matched:
                    best_idx, best_gap = idx, gap
            if best_idx is None:
                fp += 1
            else:
                matched.add(best_idx)
                tp += 1
        return {
            "truth": len(truth), "tp": tp, "fp": fp, "missed": len(truth) - tp,
            "recall": round(tp / len(truth), 3) if truth else None,
            "precision": round(tp / (tp + fp), 3) if (tp + fp) else None,
        }

    return {
        "racket": match([e["ts_ms"] for e in events if e["label"] == "racket_bounce"], racket_truth),
        "table": match([e["ts_ms"] for e in events if e["label"] == "table_bounce"], table_truth),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Retrospektiv video/ljud-analys: studstidslinje + bollväxlingar.")
    parser.add_argument("--input", required=True, help="Video- eller ljudfil.")
    parser.add_argument("--out-prefix", default="", help="Default: <input>_retro under samma katalog.")
    parser.add_argument("--session-json", default="", help="Granskad session-JSON för validering mot facit.")
    parser.add_argument("--wav-filename", default="", help="Begränsa facit till detta wav_filename i sessionen.")
    parser.add_argument("--model-dir", default=str(MODEL_DIR))
    args = parser.parse_args()

    input_path = Path(args.input)
    if not input_path.exists():
        raise SystemExit(f"Input saknas: {input_path}")
    out_prefix = Path(args.out_prefix) if args.out_prefix else input_path.with_suffix("")
    out_prefix = Path(str(out_prefix) + "_retro")

    model_dir = Path(args.model_dir)
    model = joblib.load(model_dir / "nr_histgb_all83.pkl")
    scaler = joblib.load(model_dir / "nr_scaler_all83.pkl")
    feature_cols = list(joblib.load(model_dir / "nr_feature_cols_all83.pkl"))
    labels = nr_config.CLASSES

    y, sr, tmp = extract_audio(input_path)
    print(f"Ljud: {len(y) / sr:.1f} s @ {sr} Hz")
    events = detect_events(y, sr, model, scaler, feature_cols, labels)
    rallies = segment_rallies(events)
    if tmp is not None:
        tmp.unlink(missing_ok=True)

    timeline_path = Path(f"{out_prefix}_timeline.csv")
    with timeline_path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=["ts_ms", "label", "confidence", "frame_rms", "p_racket", "p_table"])
        writer.writeheader()
        writer.writerows(events)

    rallies_path = Path(f"{out_prefix}_rallies.csv")
    with rallies_path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=["start_ms", "end_ms", "duration_s", "n_events", "n_racket", "n_table", "strokes_over_net", "ended_by"])
        writer.writeheader()
        writer.writerows(rallies)

    n_racket = sum(1 for e in events if e["label"] == "racket_bounce")
    n_table = sum(1 for e in events if e["label"] == "table_bounce")
    total_strokes = sum(r["strokes_over_net"] for r in rallies)

    lines = [
        "# Retrospektiv studs-/slaganalys",
        "",
        f"Input: `{input_path.name}` ({len(y) / sr:.1f} s). Modell: {model_dir.name} (HistGB all83).",
        "",
        f"- Racketstudsar: **{n_racket}**",
        f"- Bordsstudsar: **{n_table}**",
        f"- Bollväxlingar: **{len(rallies)}** (timeout {RALLY_TIMEOUT_MS} ms = bruten växling)",
        f"- Slag över nät totalt: **{total_strokes}**",
        "",
        "| växling | start (s) | längd (s) | racket | bord | slag över nät | slut |",
        "|---|---|---|---|---|---|---|",
    ]
    for i, r in enumerate(rallies, 1):
        lines.append(
            f"| {i} | {r['start_ms'] / 1000:.1f} | {r['duration_s']} | {r['n_racket']} | {r['n_table']} | {r['strokes_over_net']} | {r['ended_by']} |"
        )

    if args.session_json:
        scores = score_against_truth(events, Path(args.session_json), args.wav_filename or None)
        lines += [
            "",
            "## Validering mot granskat facit",
            "",
            f"- Racket: recall {scores['racket']['recall']}, precision {scores['racket']['precision']} "
            f"(TP {scores['racket']['tp']} / FP {scores['racket']['fp']} / missade {scores['racket']['missed']} av {scores['racket']['truth']})",
            f"- Bord: recall {scores['table']['recall']}, precision {scores['table']['precision']} "
            f"(TP {scores['table']['tp']} / FP {scores['table']['fp']} / missade {scores['table']['missed']} av {scores['table']['truth']})",
        ]
        print(json.dumps(scores, indent=1))

    report_path = Path(f"{out_prefix}_report.md")
    report_path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(f"Tidslinje: {timeline_path}")
    print(f"Bollväxlingar: {rallies_path}")
    print(f"Rapport: {report_path}")
    print(f"\n{n_racket} racket / {n_table} bord / {len(rallies)} växlingar / {total_strokes} slag över nät")


if __name__ == "__main__":
    main()
