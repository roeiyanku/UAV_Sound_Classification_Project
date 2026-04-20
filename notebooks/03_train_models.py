# -*- coding: utf-8 -*-
"""Train anomaly detection models and save results."""

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

BASE_DIR = Path(__file__).resolve().parents[1]
PROCESSED_DIR = BASE_DIR / "artifacts" / "processed"
RESULTS_DIR = BASE_DIR / "artifacts" / "results"
RESULTS_DIR.mkdir(parents=True, exist_ok=True)


def train_teacher_student(X_normal):
    n_components = max(8, min(64, X_normal.shape[1] // 2))

    teacher = PCA(n_components=n_components, random_state=42)
    teacher_targets = teacher.fit_transform(X_normal)

    student = MLPRegressor(hidden_layer_sizes=(128, 64), max_iter=200, random_state=42)
    student.fit(X_normal, teacher_targets)

    train_pred = student.predict(X_normal)
    train_scores = np.mean((teacher_targets - train_pred) ** 2, axis=1)
    threshold = np.percentile(train_scores, 95)

    return teacher, student, threshold


def predict_teacher_student(teacher, student, threshold, X):
    teacher_targets = teacher.transform(X)
    student_pred = student.predict(X)
    scores = np.mean((teacher_targets - student_pred) ** 2, axis=1)
    preds = (scores > threshold).astype(int)
    return preds, scores


def metrics(y_true, y_pred, scores):
    report = classification_report(y_true, y_pred, target_names=["Normal", "Anomaly"], output_dict=True, zero_division=0)
    auc = roc_auc_score(y_true, scores) if scores is not None else None
    return {
        "accuracy": accuracy_score(y_true, y_pred),
        "precision": report["Anomaly"]["precision"],
        "recall": report["Anomaly"]["recall"],
        "f1": report["Anomaly"]["f1-score"],
        "auc": auc,
    }


wav2vec = joblib.load(PROCESSED_DIR / "extracted_wav2vec_features.joblib")
ast = joblib.load(PROCESSED_DIR / "extracted_ast_features.joblib")
labels = joblib.load(PROCESSED_DIR / "labels.joblib")

y_train = np.array(labels["train"])
y_test = np.array(labels["test"])
normal_mask = y_train == 0

features = {"wav2vec": wav2vec, "ast": ast}
all_results = []

for feature_name in ["wav2vec", "ast"]:
    X_train = features[feature_name][f"{feature_name}_train_features"]
    X_test = features[feature_name][f"{feature_name}_test_features"]
    X_train_normal = X_train[normal_mask]

    ocsvm = OneClassSVM(nu=0.1, kernel="rbf", gamma="scale")
    ocsvm.fit(X_train_normal)
    ocsvm_preds = np.where(ocsvm.predict(X_test) == 1, 0, 1)
    ocsvm_scores = -ocsvm.decision_function(X_test)

    iso = IsolationForest(n_estimators=100, contamination=0.1, random_state=42)
    iso.fit(X_train_normal)
    iso_preds = np.where(iso.predict(X_test) == 1, 0, 1)
    iso_scores = -iso.decision_function(X_test)

    teacher, student, threshold = train_teacher_student(X_train_normal)
    ts_preds, ts_scores = predict_teacher_student(teacher, student, threshold, X_test)

    model_results = {
        "ocsvm": metrics(y_test, ocsvm_preds, ocsvm_scores),
        "iso_forest": metrics(y_test, iso_preds, iso_scores),
        "teacher_student": metrics(y_test, ts_preds, ts_scores),
    }

    for model_name, m in model_results.items():
        all_results.append(
            {
                "date": datetime.now().strftime("%Y-%m-%d"),
                "feature_type": feature_name,
                "model_trained": model_name,
                "accuracy": m["accuracy"],
                "precision": m["precision"],
                "recall": m["recall"],
                "f1_score": m["f1"],
                "auc": m["auc"],
            }
        )

results_df = pd.DataFrame(all_results)
out_file = RESULTS_DIR / f"model_training_results_{datetime.now().strftime('%Y-%m-%d')}.csv"
results_df.to_csv(out_file, index=False)

print("Saved:", out_file)
print(results_df)
