# -*- coding: utf-8 -*-
"""02_feature_extraction.ipynb

Local (non-Colab) feature extraction notebook script.
"""

from pathlib import Path
import time
import joblib
import numpy as np
import librosa
import torch

from transformers import (
    AutoFeatureExtractor,
    ASTForAudioClassification,
    Wav2Vec2Processor,
    Wav2Vec2Model,
)


def find_project_root() -> Path:
    if "__file__" in globals():
        start = Path(__file__).resolve().parent
    else:
        start = Path.cwd().resolve()

    for candidate in [start, *start.parents]:
        if (candidate / "requirements.txt").exists():
            return candidate
    return start


PROJECT_ROOT = find_project_root()
PROCESSED_DIR = PROJECT_ROOT / "artifacts" / "processed"
PROCESSED_DIR.mkdir(parents=True, exist_ok=True)

splits_path = PROCESSED_DIR / "processed_audio_splits.joblib"
data = joblib.load(splits_path)

data = joblib.load(PROCESSED_DIR / "processed_audio_splits.joblib")
X_train = data["X_train"]
y_train = data["y_train"]
X_val = data["X_val"]
y_val = data["y_val"]
X_test = data["X_test"]
y_test = data["y_test"]

RANDOM_STATE = 42
SAMPLE_RATE = 16000
CLIP_SECONDS = 5
EMBEDDING_BATCH_SIZE = 4

device = "cuda" if torch.cuda.is_available() else "cpu"
print("Using device:", device)


def load_waveform(path, sr=SAMPLE_RATE, seconds=CLIP_SECONDS):
    audio, _ = librosa.load(path, sr=sr)
    target_len = sr * seconds
    if len(audio) < target_len:
        audio = np.pad(audio, (0, target_len - len(audio)))
    else:
        audio = audio[:target_len]
    return audio.astype(np.float32)


def extract_ast_features(paths, batch_size=EMBEDDING_BATCH_SIZE):
    feature_extractor = AutoFeatureExtractor.from_pretrained("MIT/ast-finetuned-audioset-10-10-0.4593")
    ast_model = ASTForAudioClassification.from_pretrained("MIT/ast-finetuned-audioset-10-10-0.4593").to(device)
    ast_model.eval()

    features = []
    for start_idx in range(0, len(paths), batch_size):
        batch_paths = paths[start_idx:start_idx + batch_size]
        batch_audio = [load_waveform(path) for path in batch_paths]
        inputs = feature_extractor(batch_audio, sampling_rate=SAMPLE_RATE, return_tensors="pt")
        inputs = {key: value.to(device) for key, value in inputs.items()}

        with torch.no_grad():
            outputs = ast_model(**inputs, output_hidden_states=True)
            hidden_states = outputs.hidden_states[-1]
            batch_features = hidden_states.mean(dim=1).detach().cpu().numpy()
        features.append(batch_features)

    return np.vstack(features)


def extract_wav2vec_features(paths, batch_size=EMBEDDING_BATCH_SIZE):
    processor = Wav2Vec2Processor.from_pretrained("facebook/wav2vec2-base")
    wav2vec_model = Wav2Vec2Model.from_pretrained("facebook/wav2vec2-base").to(device)
    wav2vec_model.eval()

    features = []
    for start_idx in range(0, len(paths), batch_size):
        batch_paths = paths[start_idx:start_idx + batch_size]
        batch_audio = [load_waveform(path) for path in batch_paths]

        inputs = processor(batch_audio, sampling_rate=SAMPLE_RATE, return_tensors="pt", padding=True)
        input_values = inputs.input_values.to(device)
        attention_mask = inputs.attention_mask.to(device) if hasattr(inputs, "attention_mask") else None

        with torch.no_grad():
            outputs = wav2vec_model(input_values=input_values, attention_mask=attention_mask)
            hidden_states = outputs.last_hidden_state
            batch_features = hidden_states.mean(dim=1).detach().cpu().numpy()
        features.append(batch_features)

    return np.vstack(features)


ast_start = time.time()
ast_train_features = extract_ast_features(X_train)
ast_val_features = extract_ast_features(X_val)
ast_test_features = extract_ast_features(X_test)
print("AST extraction time (s):", round(time.time() - ast_start, 2))

wav2vec_start = time.time()
wav2vec_train_features = extract_wav2vec_features(X_train)
wav2vec_val_features = extract_wav2vec_features(X_val)
wav2vec_test_features = extract_wav2vec_features(X_test)
wav2vec_train_time = time.time() - wav2vec_start
print("Wav2Vec2 extraction time (s):", round(wav2vec_train_time, 2))
print("Wav2Vec2 feature shapes:", wav2vec_train_features.shape, wav2vec_val_features.shape, wav2vec_test_features.shape)

ast_feature_data = {
    "ast_train_features": ast_train_features,
    "ast_val_features": ast_val_features,
    "ast_test_features": ast_test_features,
}
joblib.dump(ast_feature_data, PROCESSED_DIR / "extracted_ast_features.joblib")

wav2vec_feature_data = {
    "wav2vec_train_features": wav2vec_train_features,
    "wav2vec_val_features": wav2vec_val_features,
    "wav2vec_test_features": wav2vec_test_features,
}
joblib.dump(wav2vec_feature_data, PROCESSED_DIR / "extracted_wav2vec_features.joblib")

labels = {"train": y_train, "val": y_val, "test": y_test}
joblib.dump(labels, PROCESSED_DIR / "labels.joblib")

print("Saved extracted features and labels to:", PROCESSED_DIR)
