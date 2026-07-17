"""Causal high-frequency candidate gate and onset-centered clip extraction.

The high-pass signal is used only to choose candidate timestamps. Downstream
classifiers always receive clips from the original full-band PCM.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd
from scipy.signal import butter, sosfilt


TARGET_SAMPLE_RATE = 22_050


@dataclass(frozen=True)
class HFGateConfig:
    cutoff_hz: float = 7_000.0
    frame_ms: float = 1.0
    background_ms: float = 250.0
    min_history_ms: float = 30.0
    onset_ratio: float = 3.0
    mad_multiplier: float = 6.0
    cooldown_ms: float = 80.0
    filter_order: int = 6
    absolute_floor: float = 1e-7

    def validate(self, sample_rate: int) -> None:
        nyquist = sample_rate / 2.0
        if not 0.0 < self.cutoff_hz < nyquist:
            raise ValueError(f"cutoff_hz must be between 0 and Nyquist ({nyquist})")
        if self.frame_ms <= 0 or self.background_ms <= 0 or self.min_history_ms <= 0:
            raise ValueError("frame and history durations must be positive")
        if self.onset_ratio <= 1.0:
            raise ValueError("onset_ratio must be greater than 1")
        if self.filter_order <= 0:
            raise ValueError("filter_order must be positive")


@dataclass(frozen=True)
class PreparedHFGate:
    cutoff_hz: float
    frame_samples: int
    frame_rms: np.ndarray
    background_rms: np.ndarray
    robust_sigma: np.ndarray


def _frame_count(duration_ms: float, frame_ms: float) -> int:
    return max(1, int(round(duration_ms / frame_ms)))


def highpass_pcm(pcm: np.ndarray, sample_rate: int, config: HFGateConfig) -> np.ndarray:
    """Apply the causal IIR high-pass filter intended for runtime parity."""
    config.validate(sample_rate)
    signal = np.asarray(pcm, dtype=np.float64).reshape(-1)
    if signal.size == 0:
        return np.empty(0, dtype=np.float64)
    sos = butter(
        config.filter_order,
        config.cutoff_hz,
        btype="highpass",
        fs=sample_rate,
        output="sos",
    )
    return sosfilt(sos, signal)


def _causal_percentile(values: np.ndarray, window: int, percentile: float) -> np.ndarray:
    return (
        pd.Series(values, copy=False)
        .rolling(window=max(1, window), min_periods=1)
        .quantile(percentile / 100.0)
        .to_numpy(dtype=np.float64)
    )


def prepare_hf_gate(
    pcm: np.ndarray,
    sample_rate: int = TARGET_SAMPLE_RATE,
    config: HFGateConfig = HFGateConfig(),
    *,
    filtered_pcm: np.ndarray | None = None,
) -> PreparedHFGate:
    config.validate(sample_rate)
    signal = np.asarray(pcm, dtype=np.float32).reshape(-1)
    filtered = (
        np.asarray(filtered_pcm, dtype=np.float64).reshape(-1)
        if filtered_pcm is not None
        else highpass_pcm(signal, sample_rate, config)
    )
    if len(filtered) != len(signal):
        raise ValueError("filtered_pcm must have the same length as pcm")
    frame_samples = max(1, int(round(sample_rate * config.frame_ms / 1000.0)))
    usable_samples = len(filtered) // frame_samples * frame_samples
    if usable_samples == 0:
        empty = np.empty(0, dtype=np.float64)
        return PreparedHFGate(config.cutoff_hz, frame_samples, empty, empty, empty)
    frames = filtered[:usable_samples].reshape(-1, frame_samples)
    frame_rms = np.sqrt(np.mean(frames * frames, axis=1))
    background_frames = _frame_count(config.background_ms, config.frame_ms)
    background_rms = _causal_percentile(frame_rms, background_frames, 50.0)
    q25 = _causal_percentile(frame_rms, background_frames, 25.0)
    q75 = _causal_percentile(frame_rms, background_frames, 75.0)
    robust_sigma = np.maximum(0.0, q75 - q25) / 1.349
    return PreparedHFGate(
        config.cutoff_hz,
        frame_samples,
        frame_rms,
        background_rms,
        robust_sigma,
    )


def detect_hf_candidates(
    pcm: np.ndarray,
    sample_rate: int = TARGET_SAMPLE_RATE,
    config: HFGateConfig = HFGateConfig(),
    *,
    filtered_pcm: np.ndarray | None = None,
    prepared: PreparedHFGate | None = None,
) -> list[dict[str, float | int]]:
    """Return adaptive HF onset candidates from mono full-band PCM.

    Causal rolling quartiles provide a robust background and spread estimate.
    Only rising threshold crossings are eligible, then a refractory period
    suppresses duplicate candidates from the same transient.
    """
    config.validate(sample_rate)
    signal = np.asarray(pcm, dtype=np.float32).reshape(-1)
    if signal.size == 0:
        return []

    gate = prepared or prepare_hf_gate(
        signal,
        sample_rate,
        config,
        filtered_pcm=filtered_pcm,
    )
    if gate.cutoff_hz != config.cutoff_hz:
        raise ValueError("prepared gate cutoff does not match config cutoff")
    frame_samples = gate.frame_samples
    minimum_history = _frame_count(config.min_history_ms, config.frame_ms)
    cooldown_frames = _frame_count(config.cooldown_ms, config.frame_ms)
    frame_rms = gate.frame_rms
    background_rms = gate.background_rms
    robust_sigma = gate.robust_sigma
    if frame_rms.size == 0:
        return []
    thresholds = np.maximum.reduce(
        (
            np.full_like(frame_rms, config.absolute_floor),
            background_rms * config.onset_ratio,
            background_rms + config.mad_multiplier * robust_sigma,
        )
    )
    above = frame_rms >= thresholds
    above[:minimum_history] = False
    crossings = above & ~np.r_[False, above[:-1]]

    candidates: list[dict[str, float | int]] = []
    cooldown_until = -1
    for frame_index in np.flatnonzero(crossings):
        if frame_index < cooldown_until:
            continue
        start = int(frame_index * frame_samples)
        full_band_frame = signal[start : start + frame_samples]
        candidates.append(
            {
                "onset_sample": start,
                "onset_ms": float(start / sample_rate * 1000.0),
                "hf_rms": float(frame_rms[frame_index]),
                "background_rms": float(background_rms[frame_index]),
                "threshold": float(thresholds[frame_index]),
                "full_band_peak": float(np.max(np.abs(full_band_frame), initial=0.0)),
            }
        )
        cooldown_until = int(frame_index + cooldown_frames)

    return candidates


def extract_full_band_clip(
    pcm: np.ndarray,
    onset_sample: int,
    sample_rate: int = TARGET_SAMPLE_RATE,
    *,
    pre_ms: float = 100.0,
    total_ms: float = 400.0,
) -> np.ndarray:
    """Extract a zero-padded full-band clip with a stable onset position."""
    if not 0.0 <= pre_ms < total_ms:
        raise ValueError("pre_ms must be non-negative and smaller than total_ms")
    signal = np.asarray(pcm, dtype=np.float32).reshape(-1)
    pre_samples = int(round(sample_rate * pre_ms / 1000.0))
    total_samples = int(round(sample_rate * total_ms / 1000.0))
    start = int(onset_sample) - pre_samples
    end = start + total_samples
    left_pad = max(0, -start)
    right_pad = max(0, end - len(signal))
    core = signal[max(0, start) : min(len(signal), end)]
    if left_pad or right_pad:
        core = np.pad(core, (left_pad, right_pad))
    return core[:total_samples].astype(np.float32, copy=False)
