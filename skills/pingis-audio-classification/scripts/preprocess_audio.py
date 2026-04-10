"""
preprocess_audio.py

Läser alla audio_session_*.json från data/audio/raw/ och skapar
data/audio/processed/audio_dataset.csv med 35 features per studs-klipp.

Session-läge (clip_duration_ms == 0):
  Laddar hela den långa .m4a-filen, kör onset-detection för att hitta
  enskilda studsar, klipper ut ett 1-sekunders fönster runt varje onset.
  För noise-etiketten delas filen upp i 1-sekunders bitar utan onset-detection.

Gammalt klipp-läge (clip_duration_ms == 1000):
  Laddar 1-sekunders klipp direkt och extraherar features (bakåtkompatibilitet).

Kör: python skills/pingis-audio-classification/scripts/preprocess_audio.py
"""

import json
import os
import subprocess
import sys
import tempfile
from pathlib import Path

import librosa
import numpy as np
import pandas as pd

# ── ffmpeg-sökväg (funkar även om ffmpeg inte är i PATH) ──────────────────────

_FFMPEG_CANDIDATES = [
    "ffmpeg",
    r"C:\Users\lovea\AppData\Local\Microsoft\WinGet\Packages\Gyan.FFmpeg_Microsoft.Winget.Source_8wekyb3d8bbwe\ffmpeg-8.1-full_build\bin\ffmpeg.exe",
]

def _find_ffmpeg() -> str:
    for candidate in _FFMPEG_CANDIDATES:
        try:
            subprocess.run([candidate, "-version"], capture_output=True, check=True)
            return candidate
        except (FileNotFoundError, subprocess.CalledProcessError):
            continue
    raise RuntimeError("ffmpeg hittades inte. Installera med: winget install ffmpeg")

FFMPEG = _find_ffmpeg()


def load_audio(path: str) -> tuple[np.ndarray, int]:
    """Laddar en ljudfil (.m4a, .wav, etc.) via ffmpeg → wav → librosa."""
    suffix = Path(path).suffix.lower()
    if suffix in (".wav", ".flac", ".ogg"):
        return librosa.load(path, sr=TARGET_SR, mono=True)

    # Konvertera till temporär wav-fil
    with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as tmp:
        tmp_path = tmp.name
    try:
        subprocess.run(
            [FFMPEG, "-y", "-i", path, "-ar", str(TARGET_SR), "-ac", "1", tmp_path],
            capture_output=True, check=True,
        )
        y, sr = librosa.load(tmp_path, sr=TARGET_SR, mono=True)
    finally:
        os.unlink(tmp_path)
    return y, sr

# ── Sökvägar ──────────────────────────────────────────────────────────────────

ROOT_DIR = Path(__file__).resolve().parents[3]
RAW_DIR  = ROOT_DIR / "data" / "audio" / "raw"
OUT_DIR  = ROOT_DIR / "data" / "audio" / "processed"
OUT_FILE = OUT_DIR / "audio_dataset.csv"

TARGET_SR   = 22050   # intern samplingsfrekvens
CLIP_FRAMES = TARGET_SR  # 1 sekund

# onset-detection: ignorera studsar som är < 300 ms isär (falska positiver)
MIN_ONSET_GAP_S = 0.30
# fönster: 300 ms före onset, 700 ms efter
WINDOW_BEFORE_S = 0.30
WINDOW_AFTER_S  = 0.70
# RMS-tröskel: onset-klipp under detta värde kastas (brus/falska onsets)
MIN_CLIP_RMS    = 0.005

# ── Feature-extraktion ────────────────────────────────────────────────────────

def extract_features(y: np.ndarray, sr: int = TARGET_SR) -> dict:
    """Extraherar 35 features från ett 1-sekunders ljud-array."""
    y = librosa.util.fix_length(y, size=CLIP_FRAMES)

    features: dict = {}

    mfccs = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=13)
    for i in range(13):
        features[f"mfcc_{i}_mean"] = float(np.mean(mfccs[i]))
        features[f"mfcc_{i}_std"]  = float(np.std(mfccs[i]))

    centroid = librosa.feature.spectral_centroid(y=y, sr=sr)[0]
    features["spectral_centroid_mean"] = float(np.mean(centroid))
    features["spectral_centroid_std"]  = float(np.std(centroid))

    rolloff = librosa.feature.spectral_rolloff(y=y, sr=sr, roll_percent=0.85)[0]
    features["spectral_rolloff_mean"] = float(np.mean(rolloff))
    features["spectral_rolloff_std"]  = float(np.std(rolloff))

    zcr = librosa.feature.zero_crossing_rate(y)[0]
    features["zcr_mean"] = float(np.mean(zcr))
    features["zcr_std"]  = float(np.std(zcr))

    rms = librosa.feature.rms(y=y)[0]
    features["rms_mean"] = float(np.mean(rms))
    features["rms_std"]  = float(np.std(rms))

    onset_env = librosa.onset.onset_strength(y=y, sr=sr)
    features["onset_strength_max"] = float(np.max(onset_env))

    return features


# ── Onset-detection: studs-klipp ──────────────────────────────────────────────

def extract_clips_onset(y: np.ndarray, sr: int) -> tuple[list[np.ndarray], int]:
    """
    Hittar enskilda studsar i en lång inspelning via onset-detection.
    Returnerar (klipp-lista, antal kastade klipp med för låg energi).
    """
    onset_frames = librosa.onset.onset_detect(
        y=y, sr=sr,
        units='frames',
        hop_length=512,
        backtrack=True,   # flytta tillbaka onset till exakt energitopp
    )
    onset_times = librosa.frames_to_time(onset_frames, sr=sr, hop_length=512)

    if len(onset_times) == 0:
        return [], 0

    # Filtrera onsets som är för nära varandra
    filtered: list[float] = [onset_times[0]]
    for t in onset_times[1:]:
        if t - filtered[-1] >= MIN_ONSET_GAP_S:
            filtered.append(t)

    clips = []
    dropped = 0
    for onset_time in filtered:
        start = max(0, int((onset_time - WINDOW_BEFORE_S) * sr))
        end   = min(len(y), start + CLIP_FRAMES)
        clip  = y[start:end]
        # Filtrera bort svaga onsets (brus/falska positiver)
        rms = np.sqrt(np.mean(clip ** 2))
        if rms < MIN_CLIP_RMS:
            dropped += 1
            continue
        clips.append(clip)

    return clips, dropped


# ── Noise: dela upp i jämna bitar ─────────────────────────────────────────────

def extract_clips_chunks(y: np.ndarray, sr: int) -> list[np.ndarray]:
    """Delar upp en lång inspelning i 1-sekunders bitar utan onset-detection."""
    n_chunks = len(y) // sr
    return [y[i * sr : (i + 1) * sr] for i in range(n_chunks)]


# ── Data augmentation: blanda studs med brus ──────────────────────────────────

# SNR-nivåer (Signal-to-Noise Ratio i dB) som genereras per studs-klipp.
# 20 dB = svagt brus, 10 dB = måttligt, 3 dB = starkt (studs knappt hörbar).
AUGMENT_SNR_DB   = [20.0, 10.0, 6.0]
# Gain-augmentation: simulerar olika avstånd till mikrofonen
AUGMENT_GAINS    = [0.5, 0.8, 1.5]

def mix_with_noise(signal: np.ndarray, noise_pool: list[np.ndarray], snr_db: float, rng: np.random.Generator) -> np.ndarray:
    """Blandar signal med ett slumpmässigt noise-klipp vid given SNR."""
    if not noise_pool:
        return signal

    noise = noise_pool[rng.integers(len(noise_pool))].copy()
    noise = librosa.util.fix_length(noise, size=len(signal))

    sig_rms   = np.sqrt(np.mean(signal ** 2)) + 1e-9
    noise_rms = np.sqrt(np.mean(noise ** 2))  + 1e-9
    target_noise_rms = sig_rms / (10 ** (snr_db / 20.0))
    noise_scaled = noise * (target_noise_rms / noise_rms)

    mixed = signal + noise_scaled
    # Normalisera så att inga klipp clippar
    peak = np.max(np.abs(mixed))
    if peak > 1.0:
        mixed /= peak
    return mixed.astype(np.float32)


# ── Huvudloop ─────────────────────────────────────────────────────────────────

def main() -> None:
    session_files = sorted(RAW_DIR.glob("audio_session_*.json"))
    if not session_files:
        print(f"Inga sessioner hittades i {RAW_DIR}")
        sys.exit(1)

    rng  = np.random.default_rng(42)
    rows: list[dict] = []
    errors      = 0
    total_clips = 0

    # Första passet: samla in alla rena klipp grupperade per etikett
    bounce_clips: dict[str, list[np.ndarray]] = {}  # label → [clip, ...]
    noise_clips:  list[np.ndarray] = []

    for session_path in session_files:
        with open(session_path, encoding="utf-8") as f:
            session = json.load(f)

        session_dir   = RAW_DIR / session_path.stem
        recorder      = session["session_meta"].get("recorder_name", "unknown")
        session_mode  = session["session_meta"].get("clip_duration_ms", 1000) == 0

        for event in session["events"]:
            audio_path = session_dir / event["wav_filename"]
            if not audio_path.exists():
                print(f"  Saknas: {audio_path}")
                errors += 1
                continue

            label = event["label"]

            try:
                y, sr = load_audio(str(audio_path))
            except Exception as e:
                print(f"  Fel vid laddning av {audio_path.name}: {e}")
                errors += 1
                continue

            # Välj segmenteringsstrategi
            if session_mode:
                if label == "noise":
                    clips = extract_clips_chunks(y, sr)
                    dropped = 0
                else:
                    clips, dropped = extract_clips_onset(y, sr)
                    if not clips:
                        print(f"  Inga onsets hittades i {audio_path.name} — hoppar över")
                        continue
            else:
                clips = [y]
                dropped = 0

            for clip_idx, clip in enumerate(clips):
                try:
                    feats = extract_features(clip, sr)
                except Exception as e:
                    print(f"  Feature-fel i {audio_path.name} klipp {clip_idx}: {e}")
                    errors += 1
                    continue

                feats["label"]         = label
                feats["recorder_name"] = recorder
                feats["source_file"]   = event["wav_filename"]
                rows.append(feats)
                total_clips += 1

                # Spara rå klipp för augmentation
                c = librosa.util.fix_length(clip.copy(), size=TARGET_SR)
                if label == "noise":
                    noise_clips.append(c)
                else:
                    bounce_clips.setdefault(label, []).append(c)

            if session_mode:
                drop_info = f" ({dropped} kastade, låg energi)" if dropped else ""
                print(f"  {audio_path.name}: {len(clips)} klipp ({label}){drop_info}")

    # Andra passet: augmentera studs-klipp
    aug_count = 0

    # 2a. Gain-augmentation (simulerar olika avstånd)
    for label, clips in bounce_clips.items():
        for clip in clips:
            for gain in AUGMENT_GAINS:
                scaled = np.clip(clip * gain, -1.0, 1.0).astype(np.float32)
                try:
                    feats = extract_features(scaled, TARGET_SR)
                except Exception:
                    continue
                feats["label"]         = label
                feats["recorder_name"] = "augmented"
                feats["source_file"]   = f"aug_gain{gain}"
                rows.append(feats)
                aug_count += 1
    print(f"\n  Gain-augmenterat: {aug_count} extra klipp (gains {AUGMENT_GAINS})")

    # 2b. Noise-augmentation (blanda studs med brus)
    if noise_clips:
        snr_count = 0
        for label, clips in bounce_clips.items():
            for clip in clips:
                for snr in AUGMENT_SNR_DB:
                    mixed = mix_with_noise(clip, noise_clips, snr, rng)
                    try:
                        feats = extract_features(mixed, TARGET_SR)
                    except Exception:
                        continue
                    feats["label"]         = label
                    feats["recorder_name"] = "augmented"
                    feats["source_file"]   = f"aug_snr{int(snr)}db"
                    rows.append(feats)
                    snr_count += 1
        aug_count += snr_count
        print(f"  SNR-augmenterat: {snr_count} extra klipp (SNR {AUGMENT_SNR_DB} dB)")
    else:
        print("  Ingen brus-data — hoppar över SNR-augmentation")

    print(f"  Totalt augmenterat: {aug_count} extra klipp")

    if not rows:
        print("Inga klipp bearbetades — avbryter.")
        sys.exit(1)

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    df = pd.DataFrame(rows)
    df.to_csv(OUT_FILE, index=False)

    label_counts = df["label"].value_counts().to_dict()
    print(f"\nDatasetet sparat: {OUT_FILE}")
    print(f"  {len(df)} rader totalt  ({total_clips} rena + augmenterade klipp)")
    print(f"  Etikettfördelning: {label_counts}")
    if errors:
        print(f"  {errors} filer/klipp kunde inte bearbetas")


if __name__ == "__main__":
    main()
