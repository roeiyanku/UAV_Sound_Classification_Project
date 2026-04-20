# -*- coding: utf-8 -*-
"""Compare model results and save AUC/F1 plots."""

from pathlib import Path
import glob
import pandas as pd
import matplotlib.pyplot as plt

BASE_DIR = Path(__file__).resolve().parents[1]
RESULTS_DIR = BASE_DIR / "artifacts" / "results"
FIGURES_DIR = BASE_DIR / "artifacts" / "figures"
FIGURES_DIR.mkdir(parents=True, exist_ok=True)

csv_files = glob.glob(str(RESULTS_DIR / "*.csv"))
if not csv_files:
    raise FileNotFoundError(f"No CSV results found in {RESULTS_DIR}")

latest_file = max(csv_files, key=lambda x: Path(x).stat().st_mtime)
print("Loading:", latest_file)

df = pd.read_csv(latest_file)
df_sorted = df.sort_values(by=["auc", "f1_score"], ascending=False).reset_index(drop=True)

best = df_sorted.iloc[0]
print("Best feature:", best["feature_type"])
print("Best model:", best["model_trained"])
print("Best AUC:", round(best["auc"], 4))
print("Best F1:", round(best["f1_score"], 4))

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
