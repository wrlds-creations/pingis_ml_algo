"""
train_rf_audio.py

Tränar en RandomForest-klassificerare på audio_dataset.csv och sparar
modell + scaler + label-encoder under data/audio/models/.

Kör: python skills/pingis-audio-classification/scripts/train_rf_audio.py
"""

import warnings
from pathlib import Path

import joblib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, confusion_matrix, ConfusionMatrixDisplay
from sklearn.model_selection import cross_val_score, train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler

warnings.filterwarnings("ignore")

# ── Sökvägar ──────────────────────────────────────────────────────────────────

ROOT_DIR  = Path(__file__).resolve().parents[3]
DATASET   = ROOT_DIR / "data" / "audio" / "processed" / "audio_dataset.csv"
MODEL_DIR = ROOT_DIR / "data" / "audio" / "models"

TARGET_LABELS = ["racket_bounce", "floor_bounce", "noise"]  # table_bounce läggs till när data finns
META_COLS     = {"label", "recorder_name", "source_file", "onset_index"}

# ── Träning ───────────────────────────────────────────────────────────────────

def main() -> None:
    if not DATASET.exists():
        print(f"Dataset saknas: {DATASET}")
        print("Kör preprocess_audio.py först.")
        return

    df = pd.read_csv(DATASET)
    df = df[df["label"].isin(TARGET_LABELS)].copy()

    # Ta bort klasser med för få prov för stratifierad split (kräver min 2)
    MIN_SAMPLES = 5
    counts = df["label"].value_counts()
    drop = counts[counts < MIN_SAMPLES].index.tolist()
    if drop:
        print(f"  Hoppar över klasser med < {MIN_SAMPLES} prov: {drop}")
        df = df[~df["label"].isin(drop)].copy()

    print(f"Laddade {len(df)} rader  ·  etiketter: {df['label'].value_counts().to_dict()}")

    feature_cols = [c for c in df.columns if c not in META_COLS]
    X = df[feature_cols].values.astype(np.float32)
    y_raw = df["label"].values

    le = LabelEncoder()
    y  = le.fit_transform(y_raw)

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    scaler  = StandardScaler()
    X_train = scaler.fit_transform(X_train)
    X_test  = scaler.transform(X_test)

    clf = RandomForestClassifier(
        n_estimators=300,
        class_weight='balanced_subsample',
        random_state=42,
        n_jobs=-1,
    )
    clf.fit(X_train, y_train)

    # ── Utvärdering ──────────────────────────────────────────────────────────

    cv_scores = cross_val_score(clf, X_train, y_train, cv=5, scoring="f1_macro")
    print(f"\nKorsvalidering F1 (macro): {cv_scores.mean():.3f} ± {cv_scores.std():.3f}")

    y_pred = clf.predict(X_test)
    print("\nTestset:\n")
    print(classification_report(y_test, y_pred, target_names=le.classes_))

    # ── Spara modell ─────────────────────────────────────────────────────────

    MODEL_DIR.mkdir(parents=True, exist_ok=True)
    joblib.dump(clf,    MODEL_DIR / "audio_rf_classifier.pkl")
    joblib.dump(scaler, MODEL_DIR / "audio_feature_scaler.pkl")
    joblib.dump(le,     MODEL_DIR / "audio_label_encoder.pkl")
    print(f"\nModell sparad i {MODEL_DIR}")

    # ── Konfusionsmatris ─────────────────────────────────────────────────────

    cm = confusion_matrix(y_test, y_pred)
    disp = ConfusionMatrixDisplay(cm, display_labels=le.classes_)
    disp.plot(colorbar=False)
    plt.title("Audio bounce — konfusionsmatris")
    plt.tight_layout()
    out_fig = MODEL_DIR / "confusion_matrix.png"
    plt.savefig(out_fig, dpi=120)
    print(f"Konfusionsmatris sparad: {out_fig}")


if __name__ == "__main__":
    main()
