# -*- coding: utf-8 -*-
"""01_data_preparation.ipynb

Simple local data preparation script.
Run from repository root:
    python notebooks/01_data_preparation.py
"""

from pathlib import Path
import os
import glob
import joblib
from collections import Counter
from sklearn.model_selection import train_test_split

DATA_DIR = Path("data/pump_sound_data")
PROCESSED_DIR = Path("artifacts/processed")
PROCESSED_DIR.mkdir(parents=True, exist_ok=True)

files = glob.glob(os.path.join(DATA_DIR, "**", "*.wav"), recursive=True)

print("Data directory:", DATA_DIR)
print("Number of audio files found:", len(files))
print("Example files:", files[:5])

if not files:
    raise FileNotFoundError(
        f"No .wav files found under {DATA_DIR}. "
        "Run notebooks/data_download.ipynb first or place data there."
    )

labels = [os.path.basename(os.path.dirname(f)) for f in files]
unique_labels = sorted(list(set(labels)))
label_to_int = {label: i for i, label in enumerate(unique_labels)}
int_labels = [label_to_int[label] for label in labels]

print(f"Unique labels found: {unique_labels}")

X_train, X_temp, y_train, y_temp = train_test_split(
    files, int_labels, test_size=0.3, random_state=42, stratify=int_labels
)

X_val, X_test, y_val, y_test = train_test_split(
    X_temp, y_temp, test_size=0.5, random_state=42, stratify=y_temp
)

print(f"\nTotal audio files: {len(files)}")
print(f"Training set size: {len(X_train)} ({len(X_train)/len(files):.2%})")
print(f"Validation set size: {len(X_val)} ({len(X_val)/len(files):.2%})")
print(f"Test set size: {len(X_test)} ({len(X_test)/len(files):.2%})")

train_label_counts = Counter([unique_labels[i] for i in y_train])
val_label_counts = Counter([unique_labels[i] for i in y_val])
test_label_counts = Counter([unique_labels[i] for i in y_test])

print(f"\nTraining set label distribution: {train_label_counts}")
print(f"Validation set label distribution: {val_label_counts}")
print(f"Test set label distribution: {test_label_counts}")

data_splits = {
    "X_train": X_train,
    "y_train": y_train,
    "X_val": X_val,
    "y_val": y_val,
    "X_test": X_test,
    "y_test": y_test,
    "unique_labels": unique_labels,
    "label_to_int": label_to_int,
}

save_path = PROCESSED_DIR / "processed_audio_splits.joblib"
joblib.dump(data_splits, save_path)
print(f"Data splits saved successfully to: {save_path}")
