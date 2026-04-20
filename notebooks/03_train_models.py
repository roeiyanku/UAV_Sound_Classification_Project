# -*- coding: utf-8 -*-
"""03_train_models.ipynb

Simple local model training script.
Run from repository root:
    python notebooks/03_train_models.py
"""

from pathlib import Path
from datetime import datetime
import joblib
import numpy as np
import pandas as pd
from sklearn.decomposition import PCA
from sklearn.ensemble import IsolationForest
from sklearn.metrics import classification_report, accuracy_score, roc_auc_score
from sklearn.neural_network import MLPRegressor
from sklearn.svm import OneClassSVM


PROCESSED_DIR = Path("artifacts/processed")
RESULTS_DIR = Path("artifacts/results")
RESULTS_DIR.mkdir(parents=True, exist_ok=True)


def train_teacher_student(X_train_normal, random_state=42):
    n_features = X_train_normal.shape[1]
    n_components = max(8, min(64, n_features // 2))

    teacher = PCA(n_components=n_components, random_state=random_state)
    teacher_targets = teacher.fit_transform(X_train_normal)

    student = MLPRegressor(
        hidden_layer_sizes=(128, 64),
        activation="relu",
        solver="adam",
        max_iter=200,
        random_state=random_state,
    )
    student.fit(X_train_normal, teacher_targets)

    train_pred = student.predict(X_train_normal)
    train_scores = np.mean((teacher_targets - train_pred) ** 2, axis=1)
    threshold = np.percentile(train_scores, 95)
    return teacher, student, threshold


def predict_teacher_student(teacher, student, threshold, X):
    teacher_targets = teacher.transform(X)
    student_pred = student.predict(X)
    scores = np.mean((teacher_targets - student_pred) ** 2, axis=1)
    preds = (scores > threshold).astype(int)
    return preds, scores


wav2vec_features = joblib.load(PROCESSED_DIR / "extracted_wav2vec_features.joblib")
ast_features = joblib.load(PROCESSED_DIR / "extracted_ast_features.joblib")
labels = joblib.load(PROCESSED_DIR / "labels.joblib")

y_train = np.array(labels["train"])
y_test = np.array(labels["test"])

features = {
    "wav2vec": wav2vec_features,
    "ast": ast_features,
}

model_factories = {
    "ocsvm": lambda: OneClassSVM(nu=0.1, kernel="rbf", gamma="scale"),
    "iso_forest": lambda: IsolationForest(n_estimators=100, contamination=0.1, random_state=42),
}

results = {}
normal_mask = y_train == 0

for feature_name in ["wav2vec", "ast"]:
    X_train_data = features[feature_name][f"{feature_name}_train_features"]
    X_test_data = features[feature_name][f"{feature_name}_test_features"]
    X_train_normal = X_train_data[normal_mask]

    results[feature_name] = {}

    for model_name in ["ocsvm", "iso_forest", "teacher_student"]:
        print(f"\nTraining {model_name} on {feature_name}...")

        if model_name == "teacher_student":
            teacher, student, threshold = train_teacher_student(X_train_normal)
            preds, scores = predict_teacher_student(teacher, student, threshold, X_test_data)
            auc = roc_auc_score(y_test, scores)
        else:
            model = model_factories[model_name]()
            model.fit(X_train_normal)
            raw_preds = model.predict(X_test_data)
            preds = np.array([0 if p == 1 else 1 for p in raw_preds])

            auc = None
            if hasattr(model, "decision_function"):
                try:
                    scores = -model.decision_function(X_test_data)
                    auc = roc_auc_score(y_test, scores)
                except Exception:
                    auc = None

        report = classification_report(
            y_test,
            preds,
            target_names=["Normal", "Anomaly"],
            output_dict=True,
            zero_division=0,
        )
        acc = accuracy_score(y_test, preds)

        results[feature_name][model_name] = {
            "accuracy": acc,
            "precision": report["Anomaly"]["precision"],
            "recall": report["Anomaly"]["recall"],
            "f1": report["Anomaly"]["f1-score"],
            "auc": auc,
        }

current_date = datetime.now().strftime("%Y-%m-%d")
rows = []
for feature_name, model_results in results.items():
    for model_name, metrics in model_results.items():
        rows.append(
            {
                "date": current_date,
                "feature_type": feature_name,
                "model_trained": model_name,
                "accuracy": metrics.get("accuracy"),
                "precision": metrics.get("precision"),
                "recall": metrics.get("recall"),
                "f1_score": metrics.get("f1"),
                "auc": metrics.get("auc"),
            }
        )

df_results = pd.DataFrame(rows)
results_filename = RESULTS_DIR / f"model_training_results_{current_date}.csv"
df_results.to_csv(results_filename, index=False)

print(f"Model training results saved to: {results_filename}")
print(df_results)
