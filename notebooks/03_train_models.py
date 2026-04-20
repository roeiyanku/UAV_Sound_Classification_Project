# -*- coding: utf-8 -*-
"""03_train_models.ipynb

Local (non-Colab) model training notebook script.
"""

from pathlib import Path
from datetime import datetime
import joblib
import numpy as np
import pandas as pd
from sklearn.ensemble import IsolationForest
from sklearn.metrics import classification_report, accuracy_score, roc_auc_score
from sklearn.svm import OneClassSVM


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
RESULTS_DIR = PROJECT_ROOT / "artifacts" / "results"
RESULTS_DIR.mkdir(parents=True, exist_ok=True)

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

SELECTED_FEATURES = list(features.keys())
SELECTED_MODELS = list(model_factories.keys())

results = {}
normal_mask = y_train == 0

for feature_name in SELECTED_FEATURES:
    X_train_data = features[feature_name][f"{feature_name}_train_features"]
    X_test_data = features[feature_name][f"{feature_name}_test_features"]

    results[feature_name] = {}
    X_train_normal = X_train_data[normal_mask]

    for model_name in SELECTED_MODELS:
        print(f"\nTraining {model_name} on {feature_name}...")

        model = model_factories[model_name]()
        model.fit(X_train_normal)

        raw_preds = model.predict(X_test_data)
        preds = np.array([0 if p == 1 else 1 for p in raw_preds])

        report = classification_report(
            y_test,
            preds,
            target_names=["Normal", "Anomaly"],
            output_dict=True,
            zero_division=0,
        )
        acc = accuracy_score(y_test, preds)

        auc = None
        if hasattr(model, "decision_function"):
            try:
                scores = model.decision_function(X_test_data)
                auc = roc_auc_score(y_test, -scores)
            except Exception:
                auc = None

        results[feature_name][model_name] = {
            "accuracy": acc,
            "precision": report["Anomaly"]["precision"],
            "recall": report["Anomaly"]["recall"],
            "f1": report["Anomaly"]["f1-score"],
            "auc": auc,
        }

        print(f"Accuracy: {acc:.3f}, F1: {report['Anomaly']['f1-score']:.3f}, AUC: {auc}")

current_date = datetime.now().strftime("%Y-%m-%d")

all_results_data = []
for feature_name, model_results in results.items():
    for model_name, metrics in model_results.items():
        row = {
            "date": current_date,
            "feature_type": feature_name,
            "model_trained": model_name,
            "accuracy": metrics.get("accuracy"),
            "precision": metrics.get("precision"),
            "recall": metrics.get("recall"),
            "f1_score": metrics.get("f1"),
            "auc": metrics.get("auc"),
        }
        all_results_data.append(row)

df_results = pd.DataFrame(all_results_data)
results_filename = RESULTS_DIR / f"model_training_results_{current_date}.csv"
df_results.to_csv(results_filename, index=False)

print(f"Model training results saved to: {results_filename}")
print(df_results)
