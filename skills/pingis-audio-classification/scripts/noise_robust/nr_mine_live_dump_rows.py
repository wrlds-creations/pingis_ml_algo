"""Mina träningsrader ur Fable-lägets live-debug-dumpar (med audio_b64).

Loves 2026-06-12-pass: tät studsrytm (~400 ms period) där v3 felklassar
20-35 äkta studsar per pass som 'noise'. Dumparna innehåller 300 ms-klippen
exakt som appen såg dem, så raderna hamnar i appens egen ljuddomän.

Etikettlogik (rytm-facit, konservativ — osäkra kandidater hoppas över):
  racket_bounce: kandidaten ingår i en studsrytm-körning, dvs. har gap
      250-750 ms till en grannkandidat åt minst ett håll OCH ingår i en
      kedja om >= 3 sådana kandidater. Modellens egen label ignoreras —
      det är just felklassningarna vi vill fånga.
  noise: svans-retrigger — gap < 200 ms till föregående kandidat och
      svagare frame-RMS (efterklang/skrammel efter träffen).

Skriver CSV i samma schema som nr_train.csv (alla 83 features + label +
session) för train_nr_model.py --extra-train-csv.
"""
from __future__ import annotations

import argparse
import base64
import json
from pathlib import Path

import numpy as np
import pandas as pd

import nr_features

RHYTHM_MIN_MS = 250
RHYTHM_MAX_MS = 750
RUN_MIN_LEN = 3
TAIL_MAX_MS = 200


def decode_clip(b64: str) -> np.ndarray:
    pcm = np.frombuffer(base64.b64decode(b64), dtype="<i2")
    return (pcm.astype(np.float32) / 32768.0)


def label_events(events: list[dict]) -> list[tuple[dict, str]]:
    """[(event, label)] för kandidater med säker rytm-/svansetikett."""
    evs = [e for e in events if e.get("audio_b64") and e.get("native_onset_time_ms")]
    evs.sort(key=lambda e: e["native_onset_time_ms"])
    ts = [e["native_onset_time_ms"] for e in evs]

    in_rhythm_link = [False] * len(evs)  # har rytm-gap åt minst ett håll
    for i in range(len(evs) - 1):
        gap = ts[i + 1] - ts[i]
        if RHYTHM_MIN_MS <= gap <= RHYTHM_MAX_MS:
            in_rhythm_link[i] = True
            in_rhythm_link[i + 1] = True

    # kedjor om >= RUN_MIN_LEN rytm-länkade kandidater
    labelled: list[tuple[dict, str]] = []
    run: list[int] = []

    def flush(run_idx: list[int]) -> None:
        if len(run_idx) >= RUN_MIN_LEN:
            for j in run_idx:
                labelled.append((evs[j], "racket_bounce"))

    for i, linked in enumerate(in_rhythm_link):
        if linked:
            run.append(i)
        else:
            flush(run)
            run = []
    flush(run)

    rhythm_set = {id(e) for e, _ in labelled}
    for i in range(1, len(evs)):
        gap = ts[i] - ts[i - 1]
        rms = evs[i].get("native_rms") or 0.0
        prev_rms = evs[i - 1].get("native_rms") or 0.0
        if gap < TAIL_MAX_MS and rms < prev_rms and id(evs[i]) not in rhythm_set:
            labelled.append((evs[i], "noise"))
    return labelled


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dump-dir", type=Path, required=True,
                        help="Katalog med fable_live_session_*.json (med audio_b64).")
    parser.add_argument("--out-csv", type=Path, required=True)
    args = parser.parse_args()

    rows: list[dict] = []
    for path in sorted(args.dump_dir.glob("fable_live_session_*.json")):
        data = json.loads(path.read_text(encoding="utf-8"))
        events = data.get("events", [])
        if not any(e.get("audio_b64") for e in events):
            print(f"{path.name}: inga ljudklipp - hoppar")
            continue
        session = path.stem
        labelled = label_events(events)
        n_racket = sum(1 for _, lbl in labelled if lbl == "racket_bounce")
        n_noise = sum(1 for _, lbl in labelled if lbl == "noise")
        n_flipped = sum(1 for e, lbl in labelled
                        if lbl == "racket_bounce" and e.get("model_label") != "racket_bounce")
        print(f"{path.name}: {len(events)} kandidater -> {n_racket} racket "
              f"(varav {n_flipped} v3-felklassade), {n_noise} noise")
        for i, (event, label) in enumerate(labelled):
            clip = decode_clip(event["audio_b64"])
            features = nr_features.extract_all_features(clip)
            # Samma metadataschema som nr_train_mined_v3.csv så att
            # train_nr_model.py --extra-train-csv validerar och GroupKFold
            # grupperar per livesession (läckagefritt).
            features.update({
                "clip_id": f"{session}:live{i:04d}",
                "split": "train",
                "session_id": session,
                "wav_filename": "",
                "scenario_id": "fable_live",
                "background_condition": "live_self_bounce",
                "label": label,
                "source": "live_dump_mined",
                "anchor_ms": float(event["native_onset_time_ms"]),
                "jitter_ms": 0,
                "augment": "none",
                "aug_bed": None,
                "group_id": session,
                "close_event_bucket": None,
            })
            rows.append(features)

    df = pd.DataFrame(rows)
    args.out_csv.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(args.out_csv, index=False)
    print(f"\n{len(df)} rader -> {args.out_csv}")
    print(df["label"].value_counts().to_string())


if __name__ == "__main__":
    main()
