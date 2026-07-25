"""Export frozen CNN finalists as feature-input ONNX QA artifacts.

The ONNX graphs intentionally start after the deterministic audio frontend.
The accompanying contract and PCM parity fixture let the phone runtime prove
that its log-mel/PCEN implementation matches the Python training pipeline.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from dataclasses import asdict
from datetime import UTC, datetime
from pathlib import Path

import numpy as np
import onnx
import onnxruntime as ort
import pandas as pd
import torch

from .audio_io import load_mono_audio
from .candidate_labels import FOUR_CLASSES
from .evaluate_cross_day_cnn import MODEL_FACTORIES
from .frontends import DEFAULT_FRONTEND_CONFIG, frontend_pair
from .hf_gate import HFGateConfig, extract_full_band_clip


EXPORT_SCHEMA = "hf_pcen_cnn_feature_onnx_v1"
SELECTED_MODELS = ("dual_residual", "pcen_compact")


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _softmax(logits: np.ndarray) -> np.ndarray:
    shifted = logits - logits.max(axis=1, keepdims=True)
    exponentials = np.exp(shifted)
    return exponentials / exponentials.sum(axis=1, keepdims=True)


def stack_model_input(
    feature_names: tuple[str, ...],
    *,
    logmel: np.ndarray,
    pcen: np.ndarray,
) -> np.ndarray:
    available = {"logmel": logmel, "pcen": pcen}
    unknown = [name for name in feature_names if name not in available]
    if unknown:
        raise ValueError(f"Unsupported frontend names: {unknown}")
    selected = [np.asarray(available[name], dtype=np.float32) for name in feature_names]
    if any(values.ndim != 3 for values in selected):
        raise ValueError("Frontend arrays must have shape [batch, mel, frame]")
    return np.stack(selected, axis=1).astype(np.float32, copy=False)


def select_parity_rows(metadata: pd.DataFrame, limit: int = 4) -> list[int]:
    """Choose deterministic, diverse cached candidates for runtime parity."""
    eligible = metadata.loc[metadata["cache_complete"].astype(bool)].copy()
    eligible["_row_index"] = eligible.index
    eligible = eligible.sort_values(
        ["device_alias", "label", "session_id", "candidate_index"],
        kind="stable",
    )
    selected: list[int] = []
    seen_devices: set[str] = set()
    seen_labels: set[str] = set()
    for _, row in eligible.iterrows():
        device = str(row["device_alias"])
        label = str(row["label"])
        if device in seen_devices and label in seen_labels:
            continue
        selected.append(int(row["_row_index"]))
        seen_devices.add(device)
        seen_labels.add(label)
        if len(selected) >= limit:
            return selected
    for value in eligible["_row_index"].tolist():
        row_index = int(value)
        if row_index not in selected:
            selected.append(row_index)
        if len(selected) >= limit:
            break
    if not selected:
        raise ValueError("No cache-complete rows are available for parity fixtures")
    return selected


def _load_parity_pcm(metadata: pd.DataFrame, indexes: list[int]) -> np.ndarray:
    clips: list[np.ndarray] = []
    for row_index in indexes:
        row = metadata.iloc[row_index]
        pcm, _ = load_mono_audio(
            Path(str(row["audio_path"])),
            DEFAULT_FRONTEND_CONFIG.sample_rate,
        )
        clips.append(
            extract_full_band_clip(
                pcm,
                int(row["onset_sample"]),
                DEFAULT_FRONTEND_CONFIG.sample_rate,
                pre_ms=DEFAULT_FRONTEND_CONFIG.pre_onset_ms,
                total_ms=DEFAULT_FRONTEND_CONFIG.clip_ms,
            )
        )
    return np.stack(clips).astype(np.float32, copy=False)


def _checkpoint_model(checkpoint_path: Path) -> tuple[torch.nn.Module, dict]:
    checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=True)
    model_name = checkpoint_path.stem
    if model_name not in MODEL_FACTORIES:
        raise ValueError(f"No model factory is registered for {model_name}")
    if tuple(checkpoint["classes"]) != FOUR_CLASSES:
        raise ValueError(f"Class contract mismatch in {checkpoint_path}")
    model = MODEL_FACTORIES[model_name]()
    model.load_state_dict(checkpoint["state_dict"])
    model.eval()
    return model, checkpoint


def _export_model(
    checkpoint_path: Path,
    output_dir: Path,
    *,
    logmel: np.ndarray,
    pcen: np.ndarray,
) -> dict[str, object]:
    model, checkpoint = _checkpoint_model(checkpoint_path)
    feature_names = tuple(checkpoint["feature_names"])
    inputs = stack_model_input(feature_names, logmel=logmel, pcen=pcen)
    onnx_path = output_dir / f"{checkpoint_path.stem}.onnx"
    torch.onnx.export(
        model,
        torch.from_numpy(inputs[:1]),
        onnx_path,
        input_names=["features"],
        output_names=["logits"],
        dynamic_axes={"features": {0: "batch"}, "logits": {0: "batch"}},
        opset_version=17,
        do_constant_folding=True,
    )
    exported = onnx.load(str(onnx_path))
    onnx.checker.check_model(exported)
    with torch.inference_mode():
        torch_logits = model(torch.from_numpy(inputs)).cpu().numpy()
    session = ort.InferenceSession(
        str(onnx_path),
        providers=["CPUExecutionProvider"],
    )
    onnx_logits = session.run(["logits"], {"features": inputs})[0]
    max_logit_error = float(np.max(np.abs(torch_logits - onnx_logits)))
    max_probability_error = float(
        np.max(np.abs(_softmax(torch_logits) - _softmax(onnx_logits)))
    )
    if max_logit_error > 1e-4 or max_probability_error > 1e-5:
        raise RuntimeError(
            f"ONNX parity failed for {checkpoint_path.stem}: "
            f"logits={max_logit_error}, probabilities={max_probability_error}"
        )
    return {
        "model": checkpoint_path.stem,
        "file": onnx_path.name,
        "sha256": _sha256(onnx_path),
        "bytes": onnx_path.stat().st_size,
        "classes": list(checkpoint["classes"]),
        "feature_names": list(feature_names),
        "input_shape": ["batch", len(feature_names), 64, 65],
        "output": "four_class_logits",
        "racket_probability_index": FOUR_CLASSES.index("racket_bounce"),
        "racket_probability_threshold": float(checkpoint["threshold"]),
        "parity": {
            "rows": int(len(inputs)),
            "max_abs_logit_error": max_logit_error,
            "max_abs_probability_error": max_probability_error,
        },
    }


def export_finalists(
    cache_dir: Path,
    final_selection_dir: Path,
    output_dir: Path,
    *,
    parity_rows: int = 4,
) -> dict[str, object]:
    output_dir.mkdir(parents=True, exist_ok=True)
    metadata_path = cache_dir / "metadata.csv"
    metadata = pd.read_csv(metadata_path)
    fixture_indexes = select_parity_rows(metadata, parity_rows)
    pcm = _load_parity_pcm(metadata, fixture_indexes)
    logmel, pcen = frontend_pair(pcm)

    artifact_dir = final_selection_dir / "artifacts"
    models = [
        _export_model(
            artifact_dir / f"{model_name}.pt",
            output_dir,
            logmel=logmel,
            pcen=pcen,
        )
        for model_name in SELECTED_MODELS
    ]
    gate_config = HFGateConfig(
        cutoff_hz=float(metadata["cutoff_hz"].iloc[0]),
        onset_ratio=float(metadata["onset_ratio"].iloc[0]),
        mad_multiplier=float(metadata["mad_multiplier"].iloc[0]),
    )
    fixture = {
        "schema": "hf_pcen_frontend_parity_v1",
        "frontend": {
            **DEFAULT_FRONTEND_CONFIG.to_dict(),
            "mel_power": 1.0,
            "mel_scale": "slaney",
            "mel_norm": "slaney",
            "stft_center": False,
            "logmel_ref": 1.0,
            "logmel_top_db": 80.0,
            "pcen_input_scale": float(2.0**31),
            "instance_std_floor": 1e-5,
        },
        "rows": [
            {
                "session_id": str(metadata.iloc[index]["session_id"]),
                "device_alias": str(metadata.iloc[index]["device_alias"]),
                "label": str(metadata.iloc[index]["label"]),
                "onset_ms": float(metadata.iloc[index]["onset_ms"]),
                "pcm": pcm[position].tolist(),
                "logmel": logmel[position].tolist(),
                "pcen": pcen[position].tolist(),
            }
            for position, index in enumerate(fixture_indexes)
        ],
    }
    fixture_path = output_dir / "frontend_parity_fixture.json"
    fixture_path.write_text(
        json.dumps(fixture, separators=(",", ":")),
        encoding="utf-8",
    )
    final_report_path = final_selection_dir / "final_holdout_report.json"
    contract = {
        "schema": EXPORT_SCHEMA,
        "generated_at": datetime.now(UTC).isoformat(),
        "source": {
            "final_holdout_report": str(final_report_path.resolve()),
            "final_holdout_report_sha256": _sha256(final_report_path),
        },
        "candidate_gate": asdict(gate_config),
        "clip": {
            "source": "original_full_band_pcm",
            "sample_rate": DEFAULT_FRONTEND_CONFIG.sample_rate,
            "samples": DEFAULT_FRONTEND_CONFIG.clip_samples,
            "duration_ms": DEFAULT_FRONTEND_CONFIG.clip_ms,
            "pre_onset_ms": DEFAULT_FRONTEND_CONFIG.pre_onset_ms,
        },
        "frontend_parity_fixture": {
            "file": fixture_path.name,
            "sha256": _sha256(fixture_path),
            "rows": len(fixture_indexes),
        },
        "timing": {
            "policy": "existing_fable_timing",
            "same_bounce_ms": 250.0,
            "fast_rebound_min_ms": 150.0,
            "echo_ms": 300.0,
            "echo_ratio": 0.6,
            "stronger_anchor_ratio": 1.1,
            "fast_rebound_probability": 0.9,
        },
        "models": models,
    }
    contract_path = output_dir / "deployment_contract.json"
    contract_path.write_text(
        json.dumps(contract, indent=2),
        encoding="utf-8",
    )
    return contract


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--cache-dir", type=Path, required=True)
    parser.add_argument("--final-selection-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--parity-rows", type=int, default=4)
    args = parser.parse_args()
    contract = export_finalists(
        args.cache_dir,
        args.final_selection_dir,
        args.output_dir,
        parity_rows=args.parity_rows,
    )
    print(json.dumps(contract, indent=2))


if __name__ == "__main__":
    main()
