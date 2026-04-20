# -*- coding: utf-8 -*-
"""Extract AST and Wav2Vec2 features."""

from pathlib import Path
import time
import joblib
import librosa
import numpy as np
import torch
from transformers import AutoFeatureExtractor, ASTForAudioClassification, Wav2Vec2Processor, Wav2Vec2Model

BASE_DIR = Path(__file__).resolve().parents[1]
PROCESSED_DIR = BASE_DIR / "artifacts" / "processed"
PROCESSED_DIR.mkdir(parents=True, exist_ok=True)

SAMPLE_RATE = 16000
CLIP_SECONDS = 5
BATCH_SIZE = 4


def load_waveform(path):
    audio, _ = librosa.load(path, sr=SAMPLE_RATE)
    target_len = SAMPLE_RATE * CLIP_SECONDS
    if len(audio) < target_len:
        audio = np.pad(audio, (0, target_len - len(audio)))
    return audio[:target_len].astype(np.float32)


def extract_ast_features(paths, device):
    extractor = AutoFeatureExtractor.from_pretrained("MIT/ast-finetuned-audioset-10-10-0.4593")
    model = ASTForAudioClassification.from_pretrained("MIT/ast-finetuned-audioset-10-10-0.4593").to(device)
    model.eval()

    all_features = []
    for i in range(0, len(paths), BATCH_SIZE):
        batch = [load_waveform(p) for p in paths[i : i + BATCH_SIZE]]
        inputs = extractor(batch, sampling_rate=SAMPLE_RATE, return_tensors="pt")
        inputs = {k: v.to(device) for k, v in inputs.items()}
        with torch.no_grad():
            hidden = model(**inputs, output_hidden_states=True).hidden_states[-1]
        all_features.append(hidden.mean(dim=1).cpu().numpy())

    return np.vstack(all_features)


def extract_wav2vec_features(paths, device):
    processor = Wav2Vec2Processor.from_pretrained("facebook/wav2vec2-base")
    model = Wav2Vec2Model.from_pretrained("facebook/wav2vec2-base").to(device)
    model.eval()

    all_features = []
    for i in range(0, len(paths), BATCH_SIZE):
        batch = [load_waveform(p) for p in paths[i : i + BATCH_SIZE]]
        inputs = processor(batch, sampling_rate=SAMPLE_RATE, return_tensors="pt", padding=True)
        with torch.no_grad():
            outputs = model(
                input_values=inputs.input_values.to(device),
                attention_mask=inputs.attention_mask.to(device) if hasattr(inputs, "attention_mask") else None,
            )
        all_features.append(outputs.last_hidden_state.mean(dim=1).cpu().numpy())

    return np.vstack(all_features)


data = joblib.load(PROCESSED_DIR / "processed_audio_splits.joblib")
device = "cuda" if torch.cuda.is_available() else "cpu"
print("Using:", device)

start = time.time()
ast = {
    "ast_train_features": extract_ast_features(data["X_train"], device),
    "ast_val_features": extract_ast_features(data["X_val"], device),
    "ast_test_features": extract_ast_features(data["X_test"], device),
}
print("AST time:", round(time.time() - start, 2), "sec")

start = time.time()
wav2vec = {
    "wav2vec_train_features": extract_wav2vec_features(data["X_train"], device),
    "wav2vec_val_features": extract_wav2vec_features(data["X_val"], device),
    "wav2vec_test_features": extract_wav2vec_features(data["X_test"], device),
}
print("Wav2Vec2 time:", round(time.time() - start, 2), "sec")

joblib.dump(ast, PROCESSED_DIR / "extracted_ast_features.joblib")
joblib.dump(wav2vec, PROCESSED_DIR / "extracted_wav2vec_features.joblib")
joblib.dump({"train": data["y_train"], "val": data["y_val"], "test": data["y_test"]}, PROCESSED_DIR / "labels.joblib")
print("Saved extracted features in", PROCESSED_DIR)
