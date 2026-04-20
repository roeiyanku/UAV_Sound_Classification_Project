# -*- coding: utf-8 -*-
"""Prepare train/val/test splits from local audio dataset."""

from pathlib import Path
from collections import Counter
import glob
import joblib
from sklearn.model_selection import train_test_split

BASE_DIR = Path(__file__).resolve().parents[1]
DATA_DIR = BASE_DIR / "data" / "pump_sound_data"
PROCESSED_DIR = BASE_DIR / "artifacts" / "processed"
PROCESSED_DIR.mkdir(parents=True, exist_ok=True)

files = glob.glob(str(DATA_DIR / "**" / "*.wav"), recursive=True)
print("Data dir:", DATA_DIR)
print("Files found:", len(files))

if not files:
    raise FileNotFoundError(f"No wav files found in {DATA_DIR}")

labels = [Path(file).parent.name for file in files]
unique_labels = sorted(set(labels))
label_to_int = {label: i for i, label in enumerate(unique_labels)}
int_labels = [label_to_int[label] for label in labels]

X_train, X_temp, y_train, y_temp = train_test_split(
    files, int_labels, test_size=0.3, random_state=42, stratify=int_labels
)
X_val, X_test, y_val, y_test = train_test_split(
    X_temp, y_temp, test_size=0.5, random_state=42, stratify=y_temp
)

print("Train labels:", Counter(y_train))

joblib.dump(
    {
        "X_train": X_train,
        "y_train": y_train,
        "X_val": X_val,
        "y_val": y_val,
        "X_test": X_test,
        "y_test": y_test,
        "unique_labels": unique_labels,
        "label_to_int": label_to_int,
    },
    PROCESSED_DIR / "processed_audio_splits.joblib",
)

print("Saved:", PROCESSED_DIR / "processed_audio_splits.joblib")
