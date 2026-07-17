"""Deterministic log-mel and PCEN frontends for onset-anchored clips."""

from __future__ import annotations

from dataclasses import asdict, dataclass

import librosa
import numpy as np


@dataclass(frozen=True)
class FrontendConfig:
    sample_rate: int = 22_050
    clip_ms: float = 400.0
    pre_onset_ms: float = 100.0
    n_fft: int = 512
    hop_length: int = 128
    n_mels: int = 64
    fmin_hz: float = 40.0
    fmax_hz: float = 11_025.0
    pcen_gain: float = 0.98
    pcen_bias: float = 2.0
    pcen_power: float = 0.5
    pcen_time_constant: float = 0.4
    pcen_eps: float = 1e-6

    @property
    def clip_samples(self) -> int:
        return int(round(self.sample_rate * self.clip_ms / 1000.0))

    def to_dict(self) -> dict:
        return asdict(self)


DEFAULT_FRONTEND_CONFIG = FrontendConfig()


def _as_clip_batch(clips: np.ndarray, config: FrontendConfig) -> np.ndarray:
    values = np.asarray(clips, dtype=np.float32)
    if values.ndim == 1:
        values = values[np.newaxis, :]
    if values.ndim != 2 or values.shape[1] != config.clip_samples:
        raise ValueError(
            f"Expected [batch, {config.clip_samples}] PCM, got {values.shape}"
        )
    return values


def mel_magnitude(
    clips: np.ndarray,
    config: FrontendConfig = DEFAULT_FRONTEND_CONFIG,
) -> np.ndarray:
    """Return a full-band mel-magnitude tensor without altering the PCM."""
    values = _as_clip_batch(clips, config)
    mel = librosa.feature.melspectrogram(
        y=values,
        sr=config.sample_rate,
        n_fft=config.n_fft,
        hop_length=config.hop_length,
        n_mels=config.n_mels,
        fmin=config.fmin_hz,
        fmax=config.fmax_hz,
        power=1.0,
        center=False,
    )
    return np.asarray(mel, dtype=np.float32)


def _instance_standardize(features: np.ndarray) -> np.ndarray:
    mean = features.mean(axis=(-2, -1), keepdims=True)
    std = features.std(axis=(-2, -1), keepdims=True)
    return np.asarray((features - mean) / np.maximum(std, 1e-5), dtype=np.float32)


def frontend_pair(
    clips: np.ndarray,
    config: FrontendConfig = DEFAULT_FRONTEND_CONFIG,
) -> tuple[np.ndarray, np.ndarray]:
    """Compute matched log-mel and PCEN views from the same mel magnitudes."""
    mel = mel_magnitude(clips, config)
    logmel = librosa.amplitude_to_db(np.maximum(mel, 1e-10), ref=1.0, top_db=80.0)
    pcen = librosa.pcen(
        mel * (2.0**31),
        sr=config.sample_rate,
        hop_length=config.hop_length,
        gain=config.pcen_gain,
        bias=config.pcen_bias,
        power=config.pcen_power,
        time_constant=config.pcen_time_constant,
        eps=config.pcen_eps,
        axis=-1,
    )
    return _instance_standardize(logmel), _instance_standardize(pcen)
