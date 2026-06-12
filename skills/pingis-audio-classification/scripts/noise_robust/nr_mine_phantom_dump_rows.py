"""Mina BRUS-rader ur en Fable-dump där användaren intygar att inga studsar
förekom (Love 2026-06-12: satt och skrev vid datorn, helt tyst rum, appen
räknade 13 spökstudsar med konfidens upp till 0.96). Varje kandidat med
ljudklipp blir en noise-rad — inklusive de felräknade, som är de viktigaste.
Samma schema som nr_train_mined_v3.csv."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import pandas as pd

import nr_features
from nr_mine_live_dump_rows import decode_clip


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dump", type=Path, required=True, action="append",
                        help="Repeterbar: dump-fil där ALLT är brus (inga studsar).")
    parser.add_argument("--out-csv", type=Path, required=True)
    parser.add_argument("--schema-csv", type=Path, required=True,
                        help="CSV vars kolumnordning ska matchas (nr_train.csv).")
    args = parser.parse_args()

    rows: list[dict] = []
    for dump_path in args.dump:
        events = json.loads(dump_path.read_text(encoding="utf-8")).get("events", [])
        session = dump_path.stem
        n_counted = 0
        for i, event in enumerate(events):
            if not event.get("audio_b64"):
                continue
            n_counted += 1 if event.get("counted") else 0
            features = nr_features.extract_all_features(decode_clip(event["audio_b64"]))
            features.update({
                "clip_id": f"{session}:phantom{i:04d}",
                "split": "train",
                "session_id": session,
                "wav_filename": "",
                "scenario_id": "fable_live_phantom",
                "background_condition": "quiet_typing",
                "label": "noise",
                "source": "phantom_dump_mined",
                "anchor_ms": float(event.get("native_onset_time_ms") or 0),
                "jitter_ms": 0,
                "augment": "none",
                "aug_bed": None,
                "group_id": session,
                "close_event_bucket": None,
            })
            rows.append(features)
        print(f"{dump_path.name}: {len(events)} kandidater -> noise-rader (varav {n_counted} var felräknade)")

    ref_cols = list(pd.read_csv(args.schema_csv, nrows=1).columns)
    df = pd.DataFrame(rows)[ref_cols]
    df.to_csv(args.out_csv, index=False)
    print(f"{len(df)} noise-rader -> {args.out_csv}")


if __name__ == "__main__":
    main()
