# -*- coding: utf-8 -*-
"""04_compare_results.ipynb

Local (non-Colab) results comparison notebook script.
"""

from pathlib import Path
import glob
import os
import pandas as pd
import matplotlib.pyplot as plt


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
RESULTS_DIR = PROJECT_ROOT / "artifacts" / "results"
FIGURES_DIR = PROJECT_ROOT / "artifacts" / "figures"
FIGURES_DIR.mkdir(parents=True, exist_ok=True)

print("Results dir:", RESULTS_DIR)
print("Figures dir:", FIGURES_DIR)

csv_files = glob.glob(os.path.join(RESULTS_DIR, "*.csv"))
if not csv_files:
    raise FileNotFoundError("No results CSV files found in artifacts/results.")

latest_file = max(csv_files, key=os.path.getmtime)
print("Loading:", latest_file)

df = pd.read_csv(latest_file)
print(df.columns.tolist())
print(df.shape)

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
auc_plot_path = FIGURES_DIR / "auc_comparison.png"
plt.savefig(auc_plot_path, bbox_inches="tight")
plt.show()
print("Saved plot to:", auc_plot_path)

plt.figure(figsize=(10, 5))
plt.bar(df_sorted["label"], df_sorted["f1_score"])
plt.xticks(rotation=45, ha="right")
plt.ylabel("F1 Score")
plt.title("Model Comparison by F1 Score")
plt.tight_layout()
f1_plot_path = FIGURES_DIR / "f1_comparison.png"
plt.savefig(f1_plot_path, bbox_inches="tight")
plt.show()
print("Saved plot to:", f1_plot_path)

summary = f"""
Best overall combination:
- Feature type: {best_row['feature_type']}
- Model: {best_row['model_trained']}
- AUC: {best_row['auc']:.4f}
- F1: {best_row['f1_score']:.4f}
- Accuracy: {best_row['accuracy']:.4f}
"""
print(summary)

feature_summary = (
    df.groupby("feature_type")[["auc", "f1_score", "accuracy"]]
    .mean()
    .sort_values(by="auc", ascending=False)
)
print(feature_summary)

plt.figure(figsize=(8, 4))
plt.bar(feature_summary.index, feature_summary["auc"])
plt.xticks(rotation=45, ha="right")
plt.ylabel("Mean AUC")
plt.title("Average AUC by Feature Type")
plt.tight_layout()
plt.show()
