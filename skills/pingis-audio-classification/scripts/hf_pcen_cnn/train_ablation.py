"""Train matched four-class CNNs on log-mel and PCEN candidate features."""

from __future__ import annotations

import argparse
import copy
import json
import random
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from sklearn.metrics import classification_report, confusion_matrix, f1_score
from sklearn.model_selection import StratifiedGroupKFold
from torch import nn
from torch.utils.data import DataLoader, Dataset

from .candidate_labels import FOUR_CLASSES
from .model import BounceCandidateCnn


CLASS_TO_INDEX = {label: index for index, label in enumerate(FOUR_CLASSES)}


class CachedFeatureDataset(Dataset):
    def __init__(self, features: np.ndarray, labels: np.ndarray, indexes: np.ndarray) -> None:
        self.features = features
        self.labels = labels
        self.indexes = np.asarray(indexes, dtype=np.int64)

    def __len__(self) -> int:
        return len(self.indexes)

    def __getitem__(self, item: int) -> tuple[torch.Tensor, torch.Tensor]:
        index = int(self.indexes[item])
        feature = torch.from_numpy(
            np.array(self.features[index], dtype=np.float32, copy=True)
        ).unsqueeze(0)
        return feature, torch.tensor(int(self.labels[index]), dtype=torch.long)


def _seed_everything(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


def _choose_grouped_split(
    labels: np.ndarray,
    groups: np.ndarray,
    indexes: np.ndarray,
    seed: int,
) -> tuple[np.ndarray, np.ndarray]:
    splitter = StratifiedGroupKFold(n_splits=5, shuffle=True, random_state=seed)
    overall = np.bincount(labels[indexes], minlength=len(FOUR_CLASSES)) / len(indexes)
    best: tuple[float, np.ndarray, np.ndarray] | None = None
    for train_local, val_local in splitter.split(indexes, labels[indexes], groups[indexes]):
        train_indexes = indexes[train_local]
        val_indexes = indexes[val_local]
        if np.any(np.bincount(labels[train_indexes], minlength=len(FOUR_CLASSES)) == 0):
            continue
        if np.any(np.bincount(labels[val_indexes], minlength=len(FOUR_CLASSES)) == 0):
            continue
        val_distribution = np.bincount(labels[val_indexes], minlength=len(FOUR_CLASSES)) / len(val_indexes)
        score = float(np.abs(val_distribution - overall).sum())
        if best is None or score < best[0]:
            best = (score, train_indexes, val_indexes)
    if best is None:
        raise RuntimeError("Could not create a grouped split containing all four classes")
    return best[1], best[2]


def _class_weights(labels: np.ndarray) -> torch.Tensor:
    counts = np.bincount(labels, minlength=len(FOUR_CLASSES)).astype(np.float64)
    weights = counts.sum() / (len(FOUR_CLASSES) * np.maximum(counts, 1.0))
    return torch.tensor(weights, dtype=torch.float32)


@torch.no_grad()
def _predict(
    model: nn.Module,
    features: np.ndarray,
    labels: np.ndarray,
    indexes: np.ndarray,
    batch_size: int,
) -> tuple[np.ndarray, np.ndarray]:
    loader = DataLoader(
        CachedFeatureDataset(features, labels, indexes),
        batch_size=batch_size,
        shuffle=False,
        num_workers=0,
    )
    model.eval()
    probabilities: list[np.ndarray] = []
    predictions: list[np.ndarray] = []
    for inputs, _ in loader:
        logits = model(inputs)
        probs = torch.softmax(logits, dim=1)
        probabilities.append(probs.cpu().numpy())
        predictions.append(probs.argmax(dim=1).cpu().numpy())
    return np.concatenate(predictions), np.concatenate(probabilities)


def _metric_block(labels: np.ndarray, predictions: np.ndarray) -> dict:
    return {
        "rows": int(len(labels)),
        "macro_f1": float(f1_score(labels, predictions, average="macro", zero_division=0)),
        "confusion_matrix": confusion_matrix(
            labels, predictions, labels=np.arange(len(FOUR_CLASSES))
        ).tolist(),
        "classification_report": classification_report(
            labels,
            predictions,
            labels=np.arange(len(FOUR_CLASSES)),
            target_names=FOUR_CLASSES,
            output_dict=True,
            zero_division=0,
        ),
    }


def _train_one(
    frontend_name: str,
    features: np.ndarray,
    labels: np.ndarray,
    train_indexes: np.ndarray,
    val_indexes: np.ndarray,
    external_indexes: np.ndarray,
    *,
    epochs: int,
    batch_size: int,
    seed: int,
) -> tuple[BounceCandidateCnn, dict, np.ndarray, np.ndarray]:
    _seed_everything(seed)
    model = BounceCandidateCnn(num_classes=len(FOUR_CLASSES))
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3, weight_decay=1e-4)
    criterion = nn.CrossEntropyLoss(weight=_class_weights(labels[train_indexes]))
    train_loader = DataLoader(
        CachedFeatureDataset(features, labels, train_indexes),
        batch_size=batch_size,
        shuffle=True,
        num_workers=0,
    )

    best_state = copy.deepcopy(model.state_dict())
    best_f1 = -1.0
    best_epoch = 0
    patience = 5
    stale_epochs = 0
    history: list[dict] = []
    for epoch in range(1, epochs + 1):
        model.train()
        losses: list[float] = []
        for inputs, targets in train_loader:
            optimizer.zero_grad(set_to_none=True)
            loss = criterion(model(inputs), targets)
            loss.backward()
            optimizer.step()
            losses.append(float(loss.item()))
        val_predictions, _ = _predict(model, features, labels, val_indexes, batch_size)
        val_f1 = float(
            f1_score(labels[val_indexes], val_predictions, average="macro", zero_division=0)
        )
        history.append(
            {"epoch": epoch, "train_loss": float(np.mean(losses)), "val_macro_f1": val_f1}
        )
        print(
            f"{frontend_name} epoch {epoch:02d}: "
            f"loss={np.mean(losses):.4f} val_macro_f1={val_f1:.4f}"
        )
        if val_f1 > best_f1 + 1e-4:
            best_f1 = val_f1
            best_epoch = epoch
            best_state = copy.deepcopy(model.state_dict())
            stale_epochs = 0
        else:
            stale_epochs += 1
            if stale_epochs >= patience:
                break

    model.load_state_dict(best_state)
    val_predictions, val_probabilities = _predict(
        model, features, labels, val_indexes, batch_size
    )
    external_predictions, external_probabilities = _predict(
        model, features, labels, external_indexes, batch_size
    )
    report = {
        "frontend": frontend_name,
        "best_epoch": best_epoch,
        "best_val_macro_f1": best_f1,
        "history": history,
        "validation": _metric_block(labels[val_indexes], val_predictions),
        "external": _metric_block(labels[external_indexes], external_predictions),
    }
    return model, report, val_probabilities, external_probabilities


def run_ablation(
    cache_dir: Path,
    output_dir: Path,
    *,
    epochs: int,
    batch_size: int,
    seed: int,
) -> dict:
    metadata = pd.read_csv(cache_dir / "metadata.csv")
    label_values = metadata["label"].map(CLASS_TO_INDEX)
    labels = label_values.fillna(-1).to_numpy(dtype=np.int64)
    trainable = (
        metadata["cache_complete"].astype(bool)
        & metadata["disposition"].isin(("positive", "hard_negative"))
        & label_values.notna()
    ).to_numpy()
    colleague = metadata["source"].eq("legacy:colleague_legacy").to_numpy()
    development_indexes = np.flatnonzero(trainable & colleague)
    external_indexes = np.flatnonzero(trainable & ~colleague)
    groups = metadata["base_session_id"].astype(str).to_numpy()
    train_indexes, val_indexes = _choose_grouped_split(
        labels, groups, development_indexes, seed
    )

    output_dir.mkdir(parents=True, exist_ok=True)
    split_rows = metadata[["cache_row", "source", "device_id", "base_session_id", "label"]].copy()
    split_rows["split"] = "not_trainable"
    split_rows.loc[train_indexes, "split"] = "train"
    split_rows.loc[val_indexes, "split"] = "validation"
    split_rows.loc[external_indexes, "split"] = "external_device"
    split_rows.to_csv(output_dir / "split.csv", index=False)

    report: dict = {
        "classes": list(FOUR_CLASSES),
        "seed": seed,
        "split_policy": {
            "development_source": "legacy:colleague_legacy",
            "group_key": "base_session_id",
            "validation": "one balanced fold from five-fold StratifiedGroupKFold",
            "external": "all trainable Motorola and STIGA native-recorder candidates",
            "limitation": (
                "External data has no table/floor labels and STIGA positives have counts only; "
                "this is not a full four-class leave-device-out validation."
            ),
        },
        "row_counts": {
            "train": int(len(train_indexes)),
            "validation": int(len(val_indexes)),
            "external": int(len(external_indexes)),
        },
        "label_counts": {
            split: {
                FOUR_CLASSES[index]: int(count)
                for index, count in enumerate(
                    np.bincount(labels[indexes], minlength=len(FOUR_CLASSES))
                )
            }
            for split, indexes in (
                ("train", train_indexes),
                ("validation", val_indexes),
                ("external", external_indexes),
            )
        },
        "frontends": {},
    }

    complete_indexes = np.flatnonzero(metadata["cache_complete"].astype(bool).to_numpy())
    dummy_labels = labels.copy()
    dummy_labels[dummy_labels < 0] = 0
    for frontend_name in ("logmel", "pcen"):
        features = np.load(cache_dir / f"{frontend_name}.npy", mmap_mode="r")
        model, frontend_report, _, _ = _train_one(
            frontend_name,
            features,
            labels,
            train_indexes,
            val_indexes,
            external_indexes,
            epochs=epochs,
            batch_size=batch_size,
            seed=seed,
        )
        report["frontends"][frontend_name] = frontend_report
        torch.save(
            {"state_dict": model.state_dict(), "classes": FOUR_CLASSES},
            output_dir / f"{frontend_name}_model.pt",
        )
        all_predictions, all_probabilities = _predict(
            model, features, dummy_labels, complete_indexes, batch_size
        )
        prediction_rows = metadata.loc[complete_indexes, [
            "cache_row",
            "session_id",
            "source",
            "device_id",
            "scenario_id",
            "onset_ms",
            "hf_rms",
            "disposition",
            "label",
            "expected_bounce_count",
        ]].copy()
        prediction_rows["predicted_label"] = [FOUR_CLASSES[index] for index in all_predictions]
        for class_index, class_name in enumerate(FOUR_CLASSES):
            prediction_rows[f"prob_{class_name}"] = all_probabilities[:, class_index]
        prediction_rows.to_csv(
            output_dir / f"{frontend_name}_candidate_predictions.csv", index=False
        )

    (output_dir / "ablation_report.json").write_text(
        json.dumps(report, indent=2, sort_keys=True), encoding="utf-8"
    )
    return report


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--cache-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--epochs", type=int, default=20)
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()
    report = run_ablation(
        args.cache_dir,
        args.output_dir,
        epochs=args.epochs,
        batch_size=args.batch_size,
        seed=args.seed,
    )
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
