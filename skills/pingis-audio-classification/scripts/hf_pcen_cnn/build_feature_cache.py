"""Build ignored frontend tensors for every row in one HF candidate manifest."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
from numpy.lib.format import open_memmap

from .audio_io import load_mono_audio
from .frontends import DEFAULT_FRONTEND_CONFIG, frontend_pair
from .hf_gate import extract_full_band_clip


def _base_session_id(session_id: str) -> str:
    return str(session_id).split(":", 1)[0]


def build_cache(manifest_path: Path, output_dir: Path, batch_size: int) -> dict:
    rows = pd.read_csv(manifest_path).reset_index(drop=True)
    if rows.empty:
        raise ValueError("Candidate manifest is empty")

    config = DEFAULT_FRONTEND_CONFIG
    probe = np.zeros((1, config.clip_samples), dtype=np.float32)
    probe_logmel, _ = frontend_pair(probe, config)
    feature_shape = tuple(int(value) for value in probe_logmel.shape[1:])

    output_dir.mkdir(parents=True, exist_ok=True)
    logmel_out = open_memmap(
        output_dir / "logmel.npy",
        mode="w+",
        dtype=np.float16,
        shape=(len(rows), *feature_shape),
    )
    pcen_out = open_memmap(
        output_dir / "pcen.npy",
        mode="w+",
        dtype=np.float16,
        shape=(len(rows), *feature_shape),
    )

    errors: list[dict[str, str]] = []
    completed = np.zeros(len(rows), dtype=bool)
    for audio_number, (audio_path, group) in enumerate(rows.groupby("audio_path", sort=False), start=1):
        try:
            pcm, _ = load_mono_audio(Path(audio_path), config.sample_rate)
        except Exception as exc:
            errors.append({"audio_path": str(audio_path), "error": f"{type(exc).__name__}: {exc}"})
            continue
        indexes = group.index.to_numpy(dtype=np.int64)
        onset_samples = group["onset_sample"].to_numpy(dtype=np.int64)
        for start in range(0, len(indexes), batch_size):
            batch_indexes = indexes[start : start + batch_size]
            batch_onsets = onset_samples[start : start + batch_size]
            clips = np.stack(
                [
                    extract_full_band_clip(
                        pcm,
                        int(onset_sample),
                        config.sample_rate,
                        pre_ms=config.pre_onset_ms,
                        total_ms=config.clip_ms,
                    )
                    for onset_sample in batch_onsets
                ]
            )
            logmel, pcen = frontend_pair(clips, config)
            logmel_out[batch_indexes] = logmel.astype(np.float16)
            pcen_out[batch_indexes] = pcen.astype(np.float16)
            completed[batch_indexes] = True
        if audio_number % 20 == 0:
            print(f"Processed {audio_number} audio files")

    logmel_out.flush()
    pcen_out.flush()
    rows["cache_row"] = np.arange(len(rows), dtype=np.int64)
    rows["base_session_id"] = rows["session_id"].map(_base_session_id)
    rows["cache_complete"] = completed
    rows.to_csv(output_dir / "metadata.csv", index=False)
    contract = {
        "manifest": str(manifest_path.resolve()),
        "rows": int(len(rows)),
        "complete_rows": int(completed.sum()),
        "feature_shape": list(feature_shape),
        "storage_dtype": "float16",
        "frontend": config.to_dict(),
        "errors": errors,
    }
    (output_dir / "cache_contract.json").write_text(
        json.dumps(contract, indent=2, sort_keys=True), encoding="utf-8"
    )
    return contract


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--batch-size", type=int, default=64)
    args = parser.parse_args()
    print(json.dumps(build_cache(args.manifest, args.output_dir, args.batch_size), indent=2))


if __name__ == "__main__":
    main()
