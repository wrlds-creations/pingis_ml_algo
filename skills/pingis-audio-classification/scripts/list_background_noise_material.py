#!/usr/bin/env python3
"""Inventory all background-noise material usable for noise augmentation.

Scans data/audio/raw (top level + archive_m4a + device_pull) session JSONs and
lists every event whose scenario_id / background_condition / label suggests
background, noise, music, or speech content. For each event it reports the wav
path, measured duration (wav header via stdlib `wave`, falling back to metadata
duration_ms), and how many reviewed racket-contact markers it has.

Roles:
  noise_bed        -> no racket contacts; clean continuous background material
  noisy_positive   -> racket contacts recorded WITH music/speech background
  impact_negative  -> impact-type noise takes (ball-like transients, not beds)

Stdlib only. Read-only with respect to session files; writes a CSV inventory to
 data/audio/processed/background_noise_inventory.csv
"""
from __future__ import annotations

import csv
import json
import wave
from pathlib import Path

REPO = Path(__file__).resolve().parents[3]
RAW = REPO / "data" / "audio" / "raw"
OUT_CSV = REPO / "data" / "audio" / "processed" / "background_noise_inventory.csv"

MUSIC_BG = {"music_low", "music_mid", "music_high"}
SPEECH_BG = {"speech"}

# scenario ids that are pure background recordings (no ball/racket impacts)
PURE_BED_SCENARIOS = {
    "speech_only", "music_low_only", "music_mid_only", "music_high_only",
    "desk_keyboard_only", "speech_music_noise",
}
# scenario ids with racket contacts under noise
NOISY_RACKET_SCENARIOS = {
    "racket_music", "racket_music_low", "racket_music_mid", "racket_music_high",
    "racket_speech", "racket_counting",
}
# impact-style noise scenarios (transients, useful as negatives, not beds)
IMPACT_NOISE_SCENARIOS = {"other_bounce_noise", "floor_noisy"}


def wav_duration_s(path: Path) -> float | None:
    try:
        with wave.open(str(path), "rb") as w:
            fr = w.getframerate()
            if fr <= 0:
                return None
            return w.getnframes() / float(fr)
    except Exception:
        return None


def categorize(scenario: str, bg: str, label: str) -> str:
    s = scenario or ""
    if "music" in s or bg in MUSIC_BG:
        return "music"
    if "speech" in s or "counting" in s or bg in SPEECH_BG:
        return "speech"
    if label == "noise" or s in IMPACT_NOISE_SCENARIOS or "keyboard" in s or bg in {"desk", "impact"}:
        return "other"
    return ""


def role_for(scenario: str, label: str, n_racket: int, has_markers: bool) -> str:
    if scenario in PURE_BED_SCENARIOS:
        return "noise_bed"
    if scenario in NOISY_RACKET_SCENARIOS:
        return "noisy_positive"
    if scenario in IMPACT_NOISE_SCENARIOS:
        return "impact_negative"
    if label == "noise":
        return "noise_bed"
    if has_markers:
        return "noisy_positive" if n_racket > 0 else "noise_bed"
    if label in {"racket_bounce", "racket_contact"}:
        return "noisy_positive"
    return "impact_negative" if label in {"table_bounce", "floor_bounce"} else "unknown"


def iter_session_jsons():
    for base, origin in ((RAW, "raw"), (RAW / "archive_m4a", "archive_m4a"), (RAW / "device_pull", "device_pull")):
        if not base.is_dir():
            continue
        for j in sorted(base.glob("audio_session_*.json")):
            yield j, origin


def main() -> None:
    rows = []
    for json_path, origin in iter_session_jsons():
        session_id = json_path.stem
        session_dir = json_path.with_suffix("")
        try:
            meta = json.loads(json_path.read_text(encoding="utf-8"))
        except Exception as exc:  # noqa: BLE001
            print(f"WARN cannot parse {json_path}: {exc}")
            continue
        for ev in meta.get("events", []):
            label = ev.get("label", "")
            scenario = ev.get("scenario_id", "")
            bg = ev.get("background_condition", "")
            cat = categorize(scenario, bg, label)
            if not cat:
                continue  # not noise/music/speech related
            fname = ev.get("wav_filename", "")
            audio_path = session_dir / fname
            exists = audio_path.is_file()
            if not exists and fname.endswith(".wav"):
                alt = audio_path.with_suffix(".m4a")
                if alt.is_file():
                    audio_path, exists = alt, True
            meta_dur = (ev.get("duration_ms") or 0) / 1000.0
            measured = wav_duration_s(audio_path) if exists and audio_path.suffix == ".wav" else None
            dur = measured if measured is not None else meta_dur

            markers = (ev.get("review") or {}).get("markers", [])
            n_markers = len(markers)
            n_racket = sum(1 for m in markers if m.get("final_label") == "racket_contact")
            n_confirmed_racket = sum(
                1 for m in markers
                if m.get("final_label") == "racket_contact" and m.get("review_status") in {"confirmed", None, ""}
            )
            review_stage = (ev.get("review") or {}).get("review_stage", "")

            rows.append({
                "origin": origin,
                "session_id": session_id,
                "wav_path": str(audio_path.relative_to(REPO)) if exists else f"MISSING:{fname}",
                "exists": exists,
                "label": label,
                "scenario_id": scenario,
                "background_condition": bg,
                "duration_s": round(dur, 2),
                "duration_source": "wav_header" if measured is not None else "metadata",
                "n_markers": n_markers,
                "n_racket_contacts": n_racket,
                "n_confirmed_racket": n_confirmed_racket,
                "review_stage": review_stage,
                "noise_category": cat,
                "role": role_for(scenario, label, n_racket, n_markers > 0),
            })

    OUT_CSV.parent.mkdir(parents=True, exist_ok=True)
    with OUT_CSV.open("w", newline="", encoding="utf-8") as fh:
        writer = csv.DictWriter(fh, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)

    # ---- summaries ----
    def fmt_min(sec: float) -> str:
        return f"{sec / 60.0:6.2f} min"

    print(f"Inventory rows: {len(rows)}  ->  {OUT_CSV}")
    print("\n=== NOISE BEDS (no racket contacts) by category ===")
    for cat in ("music", "speech", "other"):
        beds = [r for r in rows if r["role"] == "noise_bed" and r["noise_category"] == cat]
        total = sum(r["duration_s"] for r in beds)
        print(f"\n[{cat}] {len(beds)} events, total {fmt_min(total)}")
        for r in beds:
            print(f"  {r['session_id']:38s} {r['scenario_id'] or r['label']:22s} bg={r['background_condition']:10s} "
                  f"{r['duration_s']:7.1f}s racket_markers={r['n_racket_contacts']} {r['wav_path']}")

    print("\n=== NOISY POSITIVES (racket contacts WITH background) ===")
    by_level: dict[str, list] = {}
    for r in rows:
        if r["role"] != "noisy_positive":
            continue
        key = r["background_condition"] or r["scenario_id"]
        by_level.setdefault(key, []).append(r)
    for key in sorted(by_level):
        evs = by_level[key]
        total = sum(r["duration_s"] for r in evs)
        contacts = sum(r["n_racket_contacts"] for r in evs)
        print(f"\n[bg={key}] {len(evs)} events, {fmt_min(total)}, reviewed racket contacts={contacts}")
        for r in evs:
            print(f"  {r['session_id']:38s} {r['scenario_id']:22s} {r['duration_s']:7.1f}s "
                  f"markers={r['n_markers']} racket={r['n_racket_contacts']} stage={r['review_stage'] or '-'} {r['wav_path']}")

    print("\n=== IMPACT NEGATIVES with noise flavor (not beds) ===")
    imps = [r for r in rows if r["role"] == "impact_negative"]
    total = sum(r["duration_s"] for r in imps)
    print(f"{len(imps)} events, total {fmt_min(total)}")
    for r in imps:
        print(f"  {r['session_id']:38s} {r['scenario_id'] or r['label']:22s} bg={r['background_condition']:10s} "
              f"{r['duration_s']:7.1f}s {r['wav_path']}")

    print("\n=== Sessions contributing noisy material ===")
    sess: dict[str, dict] = {}
    for r in rows:
        d = sess.setdefault(r["session_id"], {"beds": 0.0, "pos": 0.0, "contacts": 0})
        if r["role"] == "noise_bed":
            d["beds"] += r["duration_s"]
        elif r["role"] == "noisy_positive":
            d["pos"] += r["duration_s"]
            d["contacts"] += r["n_racket_contacts"]
    for sid in sorted(sess):
        d = sess[sid]
        print(f"  {sid:40s} beds={d['beds']:7.1f}s noisy_pos={d['pos']:7.1f}s racket_contacts={d['contacts']}")


if __name__ == "__main__":
    main()
