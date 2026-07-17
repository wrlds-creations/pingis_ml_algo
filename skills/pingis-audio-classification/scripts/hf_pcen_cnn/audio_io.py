"""Small deterministic audio reader for candidate-manifest generation."""

from __future__ import annotations

from math import gcd
from pathlib import Path
import subprocess

import numpy as np
import soundfile as sf
from scipy.signal import resample_poly


def _load_with_ffmpeg(path: Path, target_sample_rate: int) -> tuple[np.ndarray, int]:
    probe = subprocess.run(
        [
            "ffprobe",
            "-v",
            "error",
            "-select_streams",
            "a:0",
            "-show_entries",
            "stream=sample_rate",
            "-of",
            "default=noprint_wrappers=1:nokey=1",
            str(path),
        ],
        check=True,
        capture_output=True,
        text=True,
    )
    input_sample_rate = int(probe.stdout.strip())
    decoded = subprocess.run(
        [
            "ffmpeg",
            "-v",
            "error",
            "-i",
            str(path),
            "-vn",
            "-ac",
            "1",
            "-ar",
            str(target_sample_rate),
            "-f",
            "f32le",
            "pipe:1",
        ],
        check=True,
        capture_output=True,
    )
    signal = np.frombuffer(decoded.stdout, dtype="<f4").copy()
    return signal, input_sample_rate


def load_mono_audio(path: Path, target_sample_rate: int = 22_050) -> tuple[np.ndarray, int]:
    try:
        pcm, input_sample_rate = sf.read(str(path), always_2d=False, dtype="float32")
    except (RuntimeError, sf.LibsndfileError):
        return _load_with_ffmpeg(path, target_sample_rate)
    signal = np.asarray(pcm, dtype=np.float32)
    if signal.ndim == 2:
        signal = signal.mean(axis=1, dtype=np.float32)
    if input_sample_rate != target_sample_rate:
        divisor = gcd(int(input_sample_rate), int(target_sample_rate))
        signal = resample_poly(
            signal.astype(np.float64),
            target_sample_rate // divisor,
            input_sample_rate // divisor,
        ).astype(np.float32)
    return signal.reshape(-1), int(input_sample_rate)
