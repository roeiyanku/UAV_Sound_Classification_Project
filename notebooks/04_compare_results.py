# -*- coding: utf-8 -*-
"""04_compare_results.ipynb

Simple local results comparison script.
Run from repository root:
    python notebooks/04_compare_results.py
"""

from pathlib import Path
import glob
import os
import pandas as pd
import matplotlib.pyplot as plt

RESULTS_DIR = Path("artifacts/results")
FIGURES_DIR = Path("artifacts/figures")
FIGURES_DIR.mkdir(parents=True, exist_ok=True)


csv_files = glob.glob(os.path.join(RESULTS_DIR, "*.csv"))
if not csv_files:
    raise FileNotFoundError("No results CSV files found in artifacts/results.")

latest_file = max(csv_files, key=os.path.getmtime)
print("Loading:", latest_file)

df = pd.read_csv(latest_file)
df_sorted = df.sort_values(by=["auc", "f1_score"], ascending=False).reset_index(drop=True)
best_row = df_sorted.iloc[0]

print("Best feature set:", best_row["feature_type"])
print("Best model:", best_row["model_trained"])
print("Best AUC:", round(best_row["auc"], 4))
print("Best F1:", round(best_row["f1_score"], 4))
print("Best Accuracy:", round(best_row["accuracy"], 4))

df_sorted["label"] = df_sorted["feature_type"] + " + " + df_sorted["model_trained"]

plt.figure(figsize=(10, 5))
plt.bar(df_sorted["label"], df_sorted["auc"])
plt.xticks(rotation=45, ha="right")
plt.ylabel("AUC")
plt.title("Model Comparison by AUC")
plt.tight_layout()
plt.savefig(FIGURES_DIR / "auc_comparison.png", bbox_inches="tight")
plt.show()

plt.figure(figsize=(10, 5))
plt.bar(df_sorted["label"], df_sorted["f1_score"])
plt.xticks(rotation=45, ha="right")
plt.ylabel("F1 Score")
plt.title("Model Comparison by F1 Score")
plt.tight_layout()
plt.savefig(FIGURES_DIR / "f1_comparison.png", bbox_inches="tight")
plt.show()
