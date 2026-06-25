"""
nr_features.py

Module 1 of the noise-robust racket bounce detector (see NR_SPEC.md).

Provides:
  - simulate_gate(...)          frame-exact offline replica of the native
                                adaptive onset gate (AudioStreamModule.kt)
  - extract_live_clip(...)      100 ms pre / 200 ms post live clip geometry
  - extract_robust_features(...) 21 noise-robust `nr_` features on the raw
                                300 ms live clip
  - extract_all_features(...)   base 62 features (preprocess_audio) + 21
                                robust features = 83 features
  - all_feature_names()         stable 83-name column order

Run a tiny self-check:
  python skills/pingis-audio-classification/scripts/noise_robust/nr_features.py <wav>
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import librosa
import numpy as np
from scipy.signal import butter, sosfiltfilt

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import preprocess_audio  # noqa: E402  (shared loaders + base 62 features)
import nr_config  # noqa: E402

TARGET_SR = nr_config.TARGET_SR
CLIP_PRE_SAMPLES = nr_config.CLIP_PRE_SAMPLES
CLIP_POST_SAMPLES = nr_config.CLIP_POST_SAMPLES
CLIP_SAMPLES = nr_config.CLIP_SAMPLES
FEATURE_BUFFER_SAMPLES = nr_config.FEATURE_BUFFER_SAMPLES

EPS = 1e-10
GATE_FRAME_SAMPLES = 220  # ~10 ms at 22050 Hz, matches the native module
GATE_BG_FRAMES = 30
GATE_FFT_SIZE = 256
GATE_BALL_BAND_HZ = (200.0, 6000.0)
GATE_BALL_RATIO_MIN = 0.55
GATE_FLATNESS_MAX = 0.6
BP_LOW_HZ = 1500.0
BP_HIGH_HZ = 7000.0
ENV_SMOOTH_SAMPLES = 110  # ~5 ms moving average

BACKGROUND_END_SAMPLE = 1764  # first 80 ms of the 300 ms clip
STFT_N_FFT = 512
STFT_HOP = 128
ROBUST_BANDS_HZ = [
    ("low", 200.0, 800.0),
    ("mid", 800.0, 2500.0),
    ("high", 2500.0, 6000.0),
    ("vhigh", 6000.0, 11025.0),
]

# Stable key order for the robust feature dict. NR_SPEC.md says "21 features"
# but only enumerates 20 names; `nr_bp_peak_db` (band-passed envelope peak
# level in dB) is the added 21st so that extract_all_features returns exactly
# 83 keys as required by the spec's output contract.
ROBUST_FEATURE_NAMES = [
    "nr_band_delta_low",
    "nr_band_delta_mid",
    "nr_band_delta_high",
    "nr_band_delta_vhigh",
    "nr_band_delta_max",
    "nr_band_delta_argmax",
    "nr_snr_db_est",
    "nr_bg_rms_db",
    "nr_flux_onset",
    "nr_bg_flatness",
    "nr_impact_flatness",
    "nr_bp_attack_ms",
    "nr_bp_decay50_ms",
    "nr_bp_crest",
    "nr_bp_peak_ratio",
    "nr_post_decay_db_50ms",
    "nr_post_decay_db_100ms",
    "nr_pcen_max",
    "nr_pcen_mean",
    "nr_pcen_std",
    "nr_bp_peak_db",
]

_BASE_FEATURE_NAMES: list[str] | None = None


def base_feature_names() -> list[str]:
    """Base 62 feature names in preprocess_audio.extract_features order."""
    global _BASE_FEATURE_NAMES
    if _BASE_FEATURE_NAMES is None:
        probe = preprocess_audio.extract_features(np.zeros(FEATURE_BUFFER_SAMPLES, dtype=np.float32), TARGET_SR)
        _BASE_FEATURE_NAMES = list(probe.keys())
    return list(_BASE_FEATURE_NAMES)


def all_feature_names() -> list[str]:
    """All 83 feature names: base 62 first, then nr_ in spec order."""
    return base_feature_names() + list(ROBUST_FEATURE_NAMES)


def _bandpass_sos(sr: int, low_hz: float, high_hz: float):
    return butter(4, [low_hz, high_hz], btype="bandpass", fs=sr, output="sos")


def _spectral_flatness(power_spectrum: np.ndarray) -> float:
    """Geometric over arithmetic mean of a power spectrum."""
    spec = np.asarray(power_spectrum, dtype=np.float64) + EPS
    geo = float(np.exp(np.mean(np.log(spec))))
    arith = float(np.mean(spec))
    return float(geo / (arith + EPS))


def _gate_spectral_check(frame: np.ndarray, sr: int) -> tuple[bool, float, float]:
    """Spectral gate on a raw 220-sample trigger frame: Hann window, zero-pad
    to 256, power spectrum; ball-band ratio and flatness, DC excluded."""
    windowed = frame * np.hanning(len(frame))
    padded = np.zeros(GATE_FFT_SIZE)
    padded[: len(windowed)] = windowed
    spectrum = np.abs(np.fft.rfft(padded)) ** 2
    freqs = np.fft.rfftfreq(GATE_FFT_SIZE, 1.0 / sr)

    non_dc = spectrum[1:]
    total = float(np.sum(non_dc))
    ball_mask = (freqs >= GATE_BALL_BAND_HZ[0]) & (freqs <= GATE_BALL_BAND_HZ[1])
    ball = float(np.sum(spectrum[ball_mask]))
    ball_ratio = float(ball / (total + EPS))
    flatness = _spectral_flatness(non_dc)
    passed = ball_ratio >= GATE_BALL_RATIO_MIN and flatness <= GATE_FLATNESS_MAX
    return passed, ball_ratio, flatness


def simulate_gate(
    y: np.ndarray,
    sr: int,
    *,
    onset_ratio: float = 1.5,
    retrigger_ms: float = 220,
    abs_min_rms: float = 0.003,
    mode: str = "broadband",
    bp_low: float = 1500.0,
    bp_high: float = 7000.0,
    spectral_gate: bool = True,
) -> list[dict]:
    """Frame-exact offline replica of the native adaptive onset gate
    (AudioStreamModule.kt streamLoop).

    Frames of 220 samples, non-overlapping. Background = rolling mean of the
    RMS of the last 30 non-triggering frames (empty buffer falls back to the
    current frame RMS, like the Kotlin `?: rms`). Trigger when
    `frame_rms >= max(bg_mean * onset_ratio, abs_min_rms)`. Frames within
    `retrigger_ms` of the last ACCEPTED trigger are skipped entirely.

    Spectral-gate semantics match native exactly: a spectrally REJECTED
    trigger does NOT start a retrigger cooldown and does NOT reset the
    background; its frame RMS is appended to the background buffer (this is
    what keeps the gate from re-firing forever on sustained music/speech).
    Only an ACCEPTED trigger sets the cooldown and resets the background
    buffer to `[bg_mean * 2.0] * 30`.

    `mode="bandpass"` computes the frame RMS on a band-passed copy of `y`
    (the spectral gate still sees the raw frame). In bandpass mode callers
    should pass `abs_min_rms=0.0015`.

    Returns a list of dicts: onset_sample, onset_ms, frame_rms, bg_rms,
    passed_spectral, ball_ratio, flatness.
    """
    if mode not in ("broadband", "bandpass"):
        raise ValueError(f"Unknown gate mode: {mode}")
    y = np.asarray(y, dtype=np.float64)
    if mode == "bandpass":
        rms_signal = sosfiltfilt(_bandpass_sos(sr, bp_low, bp_high), y)
    else:
        rms_signal = y

    triggers: list[dict] = []
    bg_buffer: list[float] = []
    last_trigger_ms: float | None = None
    n_frames = len(y) // GATE_FRAME_SAMPLES

    for i in range(n_frames):
        onset_sample = i * GATE_FRAME_SAMPLES
        onset_ms = onset_sample / sr * 1000.0
        if last_trigger_ms is not None and (onset_ms - last_trigger_ms) < retrigger_ms:
            continue  # skipped entirely: not classified, not added to background

        frame_rms = float(np.sqrt(np.mean(rms_signal[onset_sample:onset_sample + GATE_FRAME_SAMPLES] ** 2)))
        # Kotlin: empty background buffer falls back to the current frame RMS
        # (`?: rms`), which makes the very first frame unable to trigger.
        bg_mean = float(np.mean(bg_buffer)) if bg_buffer else frame_rms
        threshold = max(bg_mean * onset_ratio, abs_min_rms)

        if frame_rms >= threshold:
            raw_frame = y[onset_sample:onset_sample + GATE_FRAME_SAMPLES]
            passed, ball_ratio, flatness = _gate_spectral_check(raw_frame, sr)
            if not spectral_gate:
                passed = True
            triggers.append({
                "onset_sample": onset_sample,
                "onset_ms": onset_ms,
                "frame_rms": frame_rms,
                "bg_rms": bg_mean,
                "passed_spectral": bool(passed),
                "ball_ratio": float(ball_ratio),
                "flatness": float(flatness),
            })
            if passed:
                # Accepted trigger: cooldown + elevated background reset.
                last_trigger_ms = onset_ms
                bg_buffer = [bg_mean * 2.0] * GATE_BG_FRAMES
            else:
                # Native: rejected trigger only feeds the background buffer
                # (no cooldown, no reset) and scanning continues.
                bg_buffer.append(frame_rms)
                if len(bg_buffer) > GATE_BG_FRAMES:
                    bg_buffer.pop(0)
        else:
            bg_buffer.append(frame_rms)
            if len(bg_buffer) > GATE_BG_FRAMES:
                bg_buffer.pop(0)

    return triggers


def extract_live_clip(y: np.ndarray, onset_sample: int) -> np.ndarray:
    """100 ms before + 200 ms after the onset sample (6615 samples),
    zero-padded at the edges so the onset always lands at sample 2205."""
    y = np.asarray(y, dtype=np.float32)
    start = int(onset_sample) - CLIP_PRE_SAMPLES
    end = int(onset_sample) + CLIP_POST_SAMPLES
    left_pad = max(0, -start)
    right_pad = max(0, end - len(y))
    core = y[max(0, start):min(len(y), end)]
    if left_pad or right_pad:
        core = np.concatenate([
            np.zeros(left_pad, dtype=np.float32),
            core,
            np.zeros(right_pad, dtype=np.float32),
        ])
    return core.astype(np.float32)


def _band_masks(freqs: np.ndarray) -> list[np.ndarray]:
    masks = []
    for idx, (_, lo, hi) in enumerate(ROBUST_BANDS_HZ):
        if idx == len(ROBUST_BANDS_HZ) - 1:
            masks.append((freqs >= lo) & (freqs <= hi))
        else:
            masks.append((freqs >= lo) & (freqs < hi))
    return masks


def extract_robust_features(clip: np.ndarray, sr: int = TARGET_SR) -> dict:
    """21 noise-robust features on the RAW 6615-sample live clip."""
    clip = np.asarray(clip, dtype=np.float32)
    if len(clip) != CLIP_SAMPLES:
        clip = librosa.util.fix_length(clip, size=CLIP_SAMPLES)

    features: dict = {}

    stft = librosa.stft(clip, n_fft=STFT_N_FFT, hop_length=STFT_HOP, center=False)
    S = (np.abs(stft) ** 2).astype(np.float64)  # (257, n_frames)
    n_stft_frames = S.shape[1]
    freqs = librosa.fft_frequencies(sr=sr, n_fft=STFT_N_FFT)
    band_masks = _band_masks(freqs)
    total_mask = freqs >= 200.0

    frame_starts = np.arange(n_stft_frames) * STFT_HOP
    bg_frames = np.where(frame_starts + STFT_N_FFT <= BACKGROUND_END_SAMPLE)[0]
    impact_frames = np.where(frame_starts >= BACKGROUND_END_SAMPLE)[0]

    band_energy = np.stack([S[mask].sum(axis=0) for mask in band_masks])  # (4, n_frames)
    total_energy = S[total_mask].sum(axis=0)  # (n_frames,)

    bg_band = np.median(band_energy[:, bg_frames], axis=1) if len(bg_frames) else np.zeros(4)
    bg_total = float(np.median(total_energy[bg_frames])) if len(bg_frames) else 0.0

    if len(impact_frames):
        p = int(impact_frames[int(np.argmax(total_energy[impact_frames]))])
    else:
        p = int(np.argmax(total_energy))

    band_delta = np.log10(band_energy[:, p] + EPS) - np.log10(bg_band + EPS)
    for idx, (name, _, _) in enumerate(ROBUST_BANDS_HZ):
        features[f"nr_band_delta_{name}"] = float(band_delta[idx])
    features["nr_band_delta_max"] = float(np.max(band_delta))
    features["nr_band_delta_argmax"] = float(np.argmax(band_delta))

    features["nr_snr_db_est"] = float(10.0 * np.log10((total_energy[p] + EPS) / (bg_total + EPS)))

    bg_rms = float(np.sqrt(np.mean(clip[:BACKGROUND_END_SAMPLE].astype(np.float64) ** 2)))
    features["nr_bg_rms_db"] = float(20.0 * np.log10(bg_rms + EPS))

    # Spectral flux around the onset: frames starting in 90..150 ms.
    mag = np.sqrt(S)
    flux_lo = int(0.090 * sr)
    flux_hi = int(0.150 * sr)
    flux_frames = [
        k for k in range(1, n_stft_frames)
        if flux_lo <= frame_starts[k] <= flux_hi
    ]
    if flux_frames:
        flux_vals = [float(np.sum(np.maximum(0.0, mag[:, k] - mag[:, k - 1]))) for k in flux_frames]
        max_flux = max(flux_vals)
    else:
        max_flux = 0.0
    features["nr_flux_onset"] = float(np.log10(EPS + max_flux))

    if len(bg_frames):
        features["nr_bg_flatness"] = _spectral_flatness(S[:, bg_frames].mean(axis=1))
    else:
        features["nr_bg_flatness"] = 1.0
    features["nr_impact_flatness"] = _spectral_flatness(S[:, p])

    # Band-passed envelope features.
    bp = sosfiltfilt(_bandpass_sos(sr, BP_LOW_HZ, BP_HIGH_HZ), clip.astype(np.float64))
    abs_bp = np.abs(bp)
    env = np.convolve(abs_bp, np.ones(ENV_SMOOTH_SAMPLES) / ENV_SMOOTH_SAMPLES, mode="same")
    env_peak_idx = int(np.argmax(env))
    env_peak = float(env[env_peak_idx])

    attack_ms = 0.0
    if env_peak > 0.0:
        lookback = int(0.050 * sr)
        seg_start = max(0, env_peak_idx - lookback)
        seg = env[seg_start:env_peak_idx + 1]
        idx_90 = None
        idx_10 = None
        for j in range(len(seg) - 1, -1, -1):
            if idx_90 is None and seg[j] <= 0.9 * env_peak:
                idx_90 = j
            if seg[j] <= 0.1 * env_peak:
                idx_10 = j
                break
        if idx_90 is not None and idx_10 is not None and idx_90 >= idx_10:
            attack_ms = float((idx_90 - idx_10) / sr * 1000.0)
    features["nr_bp_attack_ms"] = attack_ms

    decay_cap = int(0.150 * sr)
    decay_region = env[env_peak_idx:env_peak_idx + decay_cap]
    if env_peak > 0.0 and len(decay_region):
        below = np.where(decay_region < 0.5 * env_peak)[0]
        decay_samples = int(below[0]) if len(below) else len(decay_region)
    else:
        decay_samples = len(decay_region)
    features["nr_bp_decay50_ms"] = float(decay_samples / sr * 1000.0)

    bp_peak = float(np.max(abs_bp))
    bp_rms = float(np.sqrt(np.mean(bp ** 2)))
    features["nr_bp_crest"] = float(bp_peak / (bp_rms + EPS))
    features["nr_bp_peak_ratio"] = float(bp_peak / (float(np.max(np.abs(clip))) + EPS))

    idx_50 = min(env_peak_idx + int(0.050 * sr), len(env) - 1)
    idx_100 = min(env_peak_idx + int(0.100 * sr), len(env) - 1)
    features["nr_post_decay_db_50ms"] = float(20.0 * np.log10((env[idx_50] + EPS) / (env_peak + EPS)))
    features["nr_post_decay_db_100ms"] = float(20.0 * np.log10((env[idx_100] + EPS) / (env_peak + EPS)))

    mel = librosa.feature.melspectrogram(
        y=clip, sr=sr, n_fft=STFT_N_FFT, hop_length=STFT_HOP, n_mels=40, center=False,
    )
    pcen = librosa.pcen(mel * (2 ** 31), sr=sr, hop_length=STFT_HOP)
    t = pcen.mean(axis=0)
    features["nr_pcen_max"] = float(np.max(t))
    features["nr_pcen_mean"] = float(np.mean(t))
    features["nr_pcen_std"] = float(np.std(t))

    features["nr_bp_peak_db"] = float(20.0 * np.log10(env_peak + EPS))

    return {name: features[name] for name in ROBUST_FEATURE_NAMES}


def extract_all_features(clip: np.ndarray, sr: int = TARGET_SR) -> dict:
    """Base 62 features (clip zero-padded at END to 1 s) merged with the 21
    robust features computed on the raw 300 ms clip. Returns 83 features."""
    clip = np.asarray(clip, dtype=np.float32)
    padded = librosa.util.fix_length(clip, size=FEATURE_BUFFER_SAMPLES)
    base = preprocess_audio.extract_features(padded, sr)
    robust = extract_robust_features(clip, sr)
    merged = dict(base)
    merged.update(robust)
    return merged


def main() -> None:
    parser = argparse.ArgumentParser(description="Self-check the noise-robust feature module on one WAV.")
    parser.add_argument("wav", help="Path to a WAV file")
    parser.add_argument("--mode", choices=["broadband", "bandpass"], default="broadband")
    args = parser.parse_args()

    y, sr = preprocess_audio.load_audio(args.wav)
    abs_min = 0.003 if args.mode == "broadband" else 0.0015
    triggers = simulate_gate(y, sr, mode=args.mode, abs_min_rms=abs_min)
    passed = sum(1 for t in triggers if t["passed_spectral"])
    print(f"{args.wav}: {len(triggers)} triggers ({passed} passed spectral gate), mode={args.mode}")
    if triggers:
        clip = extract_live_clip(y, triggers[0]["onset_sample"])
    else:
        clip = extract_live_clip(y, CLIP_PRE_SAMPLES)
    feats = extract_all_features(clip, sr)
    n_nan = sum(1 for v in feats.values() if not np.isfinite(v))
    print(f"extract_all_features: {len(feats)} features, {n_nan} non-finite")


if __name__ == "__main__":
    main()
