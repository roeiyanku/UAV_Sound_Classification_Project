"""
Training report builder for the audio anomaly-detection pipeline.

Drop this file into your project (e.g. alongside `models.py` in MODELS_DIR,
or in the same folder as your notebooks) and import:

    from reporting import TrainingReport

Each call to `TrainingReport(...)` creates a timestamped folder containing:

    run_<TIMESTAMP>_<DATASET>[_aug]/
        report.pdf               <- viewable directly in Google Drive
        results.csv              <- per-(feature, model) metrics
        results.json             <- same data, machine-readable
        metadata.json            <- run config, hyperparameters, env info
        figures/
            cm_<feature>_<model>.png
            roc_<feature>_<model>.png
            pr_<feature>_<model>.png
            roc_overlay.png
            pr_overlay.png

Requires weasyprint:
    !pip install weasyprint

Designed to keep working with the existing pump/valve MIMII pipeline
without changing the upstream code.
"""

from __future__ import annotations

import os
import json
import platform
import sys
import datetime as dt
from dataclasses import dataclass, field, asdict
from typing import Any, Sequence

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import (
    accuracy_score,
    average_precision_score,
    classification_report,
    confusion_matrix,
    precision_recall_curve,
    roc_auc_score,
    roc_curve,
)


# ---------------------------------------------------------------------------
# Result records
# ---------------------------------------------------------------------------


@dataclass
class ModelResult:
    feature_name: str
    model_name: str
    accuracy: float
    precision_anomaly: float
    recall_anomaly: float
    f1_anomaly: float
    precision_normal: float
    recall_normal: float
    f1_normal: float
    auc: float | None
    average_precision: float | None
    train_time_s: float | None
    latency_total_s: float | None
    latency_ms_sample: float | None
    n_test: int
    confusion_matrix: list[list[int]]  # rows = true [anom, norm], cols = pred [anom, norm]
    cm_path: str
    roc_path: str | None
    pr_path: str | None
    notes: str = ""
    model_hyperparams: dict[str, Any] = field(default_factory=dict)


# ---------------------------------------------------------------------------
# TrainingReport
# ---------------------------------------------------------------------------


class TrainingReport:
    """Per-run report builder.

    Parameters
    ----------
    root_dir : str
        Directory under which a timestamped run folder will be created.
    dataset : str
        Name of the dataset (e.g. "pump", "valve"). Stored in metadata
        and embedded in the run folder name.
    augmented : bool
        Whether feature extraction used augmented training data.
    anomaly_label : int, default 0
        Which integer label corresponds to the anomaly class. With the
        current pipeline this is 0 (because labels were sorted alphabetically:
        abnormal -> 0, normal -> 1).
    label_names : sequence of str, default ("anomaly", "normal")
        Human-readable name for each integer label, indexed by label.
        ``label_names[0]`` is the name for class 0, ``label_names[1]`` for class 1.
    hyperparams : dict, optional
        Pipeline-wide hyperparameters (epochs, batch sizes, etc.).
    extra_metadata : dict, optional
        Anything else worth recording (random seed, git commit, ...).
    """

    def __init__(
        self,
        root_dir: str,
        dataset: str,
        augmented: bool,
        anomaly_label: int = 0,
        label_names: Sequence[str] = ("anomaly", "normal"),
        hyperparams: dict | None = None,
        extra_metadata: dict | None = None,
    ):
        if anomaly_label not in (0, 1):
            raise ValueError("anomaly_label must be 0 or 1 for binary classification")
        if len(label_names) != 2:
            raise ValueError("label_names must have exactly 2 entries (binary task)")

        self.timestamp = dt.datetime.now()
        ts_str = self.timestamp.strftime("%Y-%m-%d_%H-%M-%S")
        suffix = f"_{dataset}" + ("_aug" if augmented else "")
        self.run_id = f"run_{ts_str}{suffix}"
        self.run_dir = os.path.join(root_dir, self.run_id)
        self.fig_dir = os.path.join(self.run_dir, "figures")
        os.makedirs(self.fig_dir, exist_ok=True)

        self.dataset = dataset
        self.augmented = augmented
        self.anomaly_label = int(anomaly_label)
        self.normal_label = 1 - self.anomaly_label
        self.label_names = tuple(label_names)
        self.anomaly_name = self.label_names[self.anomaly_label]
        self.normal_name = self.label_names[self.normal_label]

        self.hyperparams = dict(hyperparams or {})
        self.extra_metadata = dict(extra_metadata or {})
        self.results: list[ModelResult] = []
        self.data_summary: dict | None = None
        self.extraction_times: dict[str, float] = {}
        # cache for overlay plots
        self._roc_curves: list[tuple[str, np.ndarray, np.ndarray, float]] = []
        self._pr_curves: list[tuple[str, np.ndarray, np.ndarray, float]] = []

    # ------------------------------------------------------------------
    # Logging hooks
    # ------------------------------------------------------------------

    def log_data_summary(
        self,
        y_train: Sequence[int],
        y_val: Sequence[int],
        y_test: Sequence[int],
    ) -> None:
        """Record train/val/test sizes and class balance."""

        def _summary(y):
            y = np.asarray(y)
            unique, counts = np.unique(y, return_counts=True)
            counts_by_label = {int(u): int(c) for u, c in zip(unique, counts)}
            return {
                "n_samples": int(len(y)),
                "n_anomaly": int(counts_by_label.get(self.anomaly_label, 0)),
                "n_normal": int(counts_by_label.get(self.normal_label, 0)),
            }

        self.data_summary = {
            "train": _summary(y_train),
            "val": _summary(y_val),
            "test": _summary(y_test),
        }

    def log_extraction_time(self, feature_name: str, seconds: float) -> None:
        self.extraction_times[feature_name] = float(seconds)

    def add_result(
        self,
        feature_name: str,
        model_name: str,
        y_true: Sequence[int],
        y_pred: Sequence[int],
        scores: Sequence[float] | None = None,
        scores_higher_is_anomaly: bool = True,
        train_time_s: float | None = None,
        latency_total_s: float | None = None,
        model_hyperparams: dict | None = None,
        notes: str = "",
    ) -> ModelResult:
        """Compute metrics for one (feature, model) combination and persist
        confusion matrix and ROC/PR curve images.

        Parameters
        ----------
        scores : array-like or None
            Continuous decision scores for AUC and PR-AUC. If ``None``, those
            metrics are skipped.
        scores_higher_is_anomaly : bool
            ``True`` if higher score => more anomalous (e.g. teacher-student
            distance). ``False`` if higher score => more normal (e.g.
            ``OCSVM.decision_function``, sigmoid output of the CNN where the
            positive class is "normal", logistic-regression P(class=1)).

            Set this correctly. The report does **not** silently flip the AUC
            when it lands below 0.5; it reports it honestly so direction bugs
            don't get hidden.
        """

        y_true = np.asarray(y_true).astype(int)
        y_pred = np.asarray(y_pred).astype(int)
        if y_true.shape != y_pred.shape:
            raise ValueError(f"y_true and y_pred shapes differ: {y_true.shape} vs {y_pred.shape}")

        # Confusion matrix with explicit row/col order: [anomaly, normal]
        labels_in_display_order = [self.anomaly_label, self.normal_label]
        cm = confusion_matrix(y_true, y_pred, labels=labels_in_display_order)

        # Per-class metrics. Use labels=[0, 1] (default sorted), and pull out by
        # int label so we are immune to target_names ordering surprises.
        report = classification_report(
            y_true,
            y_pred,
            labels=[0, 1],
            output_dict=True,
            zero_division=0,
        )
        acc = accuracy_score(y_true, y_pred)

        anom_metrics = report[str(self.anomaly_label)]
        norm_metrics = report[str(self.normal_label)]

        # AUC / PR
        auc_val: float | None = None
        ap_val: float | None = None
        roc_path = None
        pr_path = None
        if scores is not None:
            scores = np.asarray(scores).astype(float)
            anomaly_score = scores if scores_higher_is_anomaly else -scores
            is_anom = (y_true == self.anomaly_label).astype(int)
            try:
                if len(np.unique(is_anom)) >= 2:
                    auc_val = float(roc_auc_score(is_anom, anomaly_score))
                    ap_val = float(average_precision_score(is_anom, anomaly_score))
                    fpr, tpr, _ = roc_curve(is_anom, anomaly_score)
                    prec, rec, _ = precision_recall_curve(is_anom, anomaly_score)
                    roc_path = self._save_roc(feature_name, model_name, fpr, tpr, auc_val)
                    pr_path = self._save_pr(feature_name, model_name, prec, rec, ap_val)
                    label_for_overlay = f"{feature_name} / {model_name}"
                    self._roc_curves.append((label_for_overlay, fpr, tpr, auc_val))
                    self._pr_curves.append((label_for_overlay, prec, rec, ap_val))
            except Exception as e:  # pragma: no cover
                print(f"[reporting] AUC/PR computation failed for "
                      f"{feature_name}/{model_name}: {e}")

        cm_path = self._save_cm(feature_name, model_name, cm)

        n_test = int(len(y_true))
        latency_ms_sample = (
            (latency_total_s / n_test * 1000.0) if latency_total_s is not None else None
        )

        result = ModelResult(
            feature_name=feature_name,
            model_name=model_name,
            accuracy=float(acc),
            precision_anomaly=float(anom_metrics["precision"]),
            recall_anomaly=float(anom_metrics["recall"]),
            f1_anomaly=float(anom_metrics["f1-score"]),
            precision_normal=float(norm_metrics["precision"]),
            recall_normal=float(norm_metrics["recall"]),
            f1_normal=float(norm_metrics["f1-score"]),
            auc=auc_val,
            average_precision=ap_val,
            train_time_s=train_time_s,
            latency_total_s=latency_total_s,
            latency_ms_sample=latency_ms_sample,
            n_test=n_test,
            confusion_matrix=cm.tolist(),
            cm_path=os.path.relpath(cm_path, self.run_dir),
            roc_path=os.path.relpath(roc_path, self.run_dir) if roc_path else None,
            pr_path=os.path.relpath(pr_path, self.run_dir) if pr_path else None,
            notes=notes,
            model_hyperparams=dict(model_hyperparams or {}),
        )
        self.results.append(result)
        return result

    # ------------------------------------------------------------------
    # Plot helpers
    # ------------------------------------------------------------------

    def _save_cm(self, feature_name: str, model_name: str, cm: np.ndarray) -> str:
        """Confusion matrix with rows/cols ordered [anomaly, normal]."""
        fig, ax = plt.subplots(figsize=(3.2, 2.8))
        sns.heatmap(
            cm,
            annot=True,
            fmt="d",
            cmap="Blues",
            cbar=False,
            xticklabels=[self.anomaly_name, self.normal_name],
            yticklabels=[self.anomaly_name, self.normal_name],
            ax=ax,
        )
        ax.set_xlabel("Predicted")
        ax.set_ylabel("True")
        ax.set_title(f"{model_name} on {feature_name}")
        fig.tight_layout()
        path = os.path.join(self.fig_dir, f"cm_{feature_name}_{model_name}.png")
        fig.savefig(path, dpi=120, bbox_inches="tight")
        plt.close(fig)
        return path

    def _save_roc(self, feature_name, model_name, fpr, tpr, auc_val) -> str:
        fig, ax = plt.subplots(figsize=(3.2, 2.8))
        ax.plot(fpr, tpr, label=f"AUC = {auc_val:.3f}")
        ax.plot([0, 1], [0, 1], linestyle="--", color="grey", alpha=0.6)
        ax.set_xlabel("False positive rate")
        ax.set_ylabel("True positive rate")
        ax.set_title(f"ROC — {model_name} on {feature_name}")
        ax.legend(loc="lower right")
        ax.set_xlim(0, 1); ax.set_ylim(0, 1.02)
        fig.tight_layout()
        path = os.path.join(self.fig_dir, f"roc_{feature_name}_{model_name}.png")
        fig.savefig(path, dpi=120, bbox_inches="tight")
        plt.close(fig)
        return path

    def _save_pr(self, feature_name, model_name, prec, rec, ap_val) -> str:
        fig, ax = plt.subplots(figsize=(3.2, 2.8))
        ax.plot(rec, prec, label=f"AP = {ap_val:.3f}")
        ax.set_xlabel("Recall (anomaly)")
        ax.set_ylabel("Precision (anomaly)")
        ax.set_title(f"PR — {model_name} on {feature_name}")
        ax.legend(loc="lower left")
        ax.set_xlim(0, 1); ax.set_ylim(0, 1.02)
        fig.tight_layout()
        path = os.path.join(self.fig_dir, f"pr_{feature_name}_{model_name}.png")
        fig.savefig(path, dpi=120, bbox_inches="tight")
        plt.close(fig)
        return path

    def _save_overlays(self) -> tuple[str | None, str | None]:
        roc_overlay_path = None
        pr_overlay_path = None
        if self._roc_curves:
            fig, ax = plt.subplots(figsize=(7.0, 5.5))
            for label, fpr, tpr, auc_val in self._roc_curves:
                ax.plot(fpr, tpr, label=f"{label} (AUC={auc_val:.3f})")
            ax.plot([0, 1], [0, 1], linestyle="--", color="grey", alpha=0.6)
            ax.set_xlabel("False positive rate")
            ax.set_ylabel("True positive rate")
            ax.set_title("ROC curves (all models)")
            ax.legend(loc="lower right", fontsize=8)
            ax.set_xlim(0, 1); ax.set_ylim(0, 1.02)
            fig.tight_layout()
            roc_overlay_path = os.path.join(self.fig_dir, "roc_overlay.png")
            fig.savefig(roc_overlay_path, dpi=120, bbox_inches="tight")
            plt.close(fig)
        if self._pr_curves:
            fig, ax = plt.subplots(figsize=(7.0, 5.5))
            for label, prec, rec, ap_val in self._pr_curves:
                ax.plot(rec, prec, label=f"{label} (AP={ap_val:.3f})")
            ax.set_xlabel("Recall (anomaly)")
            ax.set_ylabel("Precision (anomaly)")
            ax.set_title("Precision–Recall curves (all models)")
            ax.legend(loc="lower left", fontsize=8)
            ax.set_xlim(0, 1); ax.set_ylim(0, 1.02)
            fig.tight_layout()
            pr_overlay_path = os.path.join(self.fig_dir, "pr_overlay.png")
            fig.savefig(pr_overlay_path, dpi=120, bbox_inches="tight")
            plt.close(fig)
        return roc_overlay_path, pr_overlay_path

    # ------------------------------------------------------------------
    # Output
    # ------------------------------------------------------------------

    def to_dataframe(self) -> pd.DataFrame:
        rows = []
        for r in self.results:
            d = asdict(r)
            d.pop("confusion_matrix", None)
            d["timestamp"] = self.timestamp.isoformat(timespec="seconds")
            d["dataset"] = self.dataset
            d["augmented"] = self.augmented
            rows.append(d)
        cols = [
            "timestamp", "dataset", "augmented",
            "feature_name", "model_name",
            "accuracy",
            "precision_anomaly", "recall_anomaly", "f1_anomaly",
            "precision_normal", "recall_normal", "f1_normal",
            "auc", "average_precision",
            "train_time_s", "latency_total_s", "latency_ms_sample",
            "n_test", "cm_path", "roc_path", "pr_path",
            "notes",
        ]
        df = pd.DataFrame(rows)
        present = [c for c in cols if c in df.columns]
        return df[present]

    def _metadata_dict(self) -> dict:
        return {
            "run_id": self.run_id,
            "timestamp": self.timestamp.isoformat(timespec="seconds"),
            "dataset": self.dataset,
            "augmented": self.augmented,
            "anomaly_label": self.anomaly_label,
            "anomaly_name": self.anomaly_name,
            "normal_label": self.normal_label,
            "normal_name": self.normal_name,
            "hyperparams": self.hyperparams,
            "extraction_times_s": self.extraction_times,
            "data_summary": self.data_summary,
            "extra_metadata": self.extra_metadata,
            "environment": {
                "python": sys.version.split()[0],
                "platform": platform.platform(),
                "numpy": np.__version__,
                "pandas": pd.__version__,
            },
        }

    def finalize(self) -> str:
        """Write CSV + JSON + PDF. Returns path to the PDF file."""
        try:
            from weasyprint import HTML as WeasyprintHTML
        except ImportError:
            raise ImportError(
                "weasyprint is required to generate PDF reports.\n"
                "Install it with:  !pip install weasyprint"
            )

        roc_overlay_path, pr_overlay_path = self._save_overlays()

        df = self.to_dataframe()
        csv_path = os.path.join(self.run_dir, "results.csv")
        df.to_csv(csv_path, index=False)

        json_path = os.path.join(self.run_dir, "results.json")
        with open(json_path, "w") as f:
            json.dump([asdict(r) for r in self.results], f, indent=2)

        meta_path = os.path.join(self.run_dir, "metadata.json")
        with open(meta_path, "w") as f:
            json.dump(self._metadata_dict(), f, indent=2)

        # base_url lets weasyprint resolve relative figure paths (figures/*.png)
        html_string = self._build_html(df, roc_overlay_path, pr_overlay_path)
        pdf_path = os.path.join(self.run_dir, "report.pdf")
        WeasyprintHTML(string=html_string, base_url=self.run_dir).write_pdf(pdf_path)

        return pdf_path

    
    # ------------------------------------------------------------------
    # HTML builder
    # ------------------------------------------------------------------

    def _build_html(self, df: pd.DataFrame, roc_overlay_path, pr_overlay_path) -> str:
        meta = self._metadata_dict()
        ds = meta["data_summary"] or {}

        def fmt(x, n=3):
            if x is None or (isinstance(x, float) and np.isnan(x)):
                return "—"
            if isinstance(x, float):
                return f"{x:.{n}f}"
            return str(x)

        # Sort headline table by AUC desc, F1-anom desc
        df_sorted = df.sort_values(
            by=["auc", "f1_anomaly"], ascending=False, na_position="last"
        ).reset_index(drop=True)

        # --- Headline table ---
        table_rows = []
        for _, row in df_sorted.iterrows():
            table_rows.append(
                "<tr>"
                f"<td>{row['feature_name']}</td>"
                f"<td>{row['model_name']}</td>"
                f"<td>{fmt(row['accuracy'])}</td>"
                f"<td>{fmt(row['precision_anomaly'])}</td>"
                f"<td>{fmt(row['recall_anomaly'])}</td>"
                f"<td><b>{fmt(row['f1_anomaly'])}</b></td>"
                f"<td>{fmt(row['auc'])}</td>"
                f"<td>{fmt(row['average_precision'])}</td>"
                f"<td>{fmt(row.get('train_time_s'), 1)}</td>"
                f"<td>{fmt(row.get('latency_ms_sample'))}</td>"
                "</tr>"
            )
        headline_table = (
            "<table class='headline'><thead><tr>"
            "<th>Feature</th><th>Model</th><th>Acc</th>"
            "<th>Prec†</th><th>Rec†</th><th>F1†</th>"
            "<th>AUC</th><th>AP</th><th>Train s</th><th>ms/samp</th>"
            "</tr></thead><tbody>" + "".join(table_rows) + "</tbody></table>"
            + "<p style='font-size:10px;color:#666;margin:4px 0 0;'>" + f"† {self.anomaly_name} class" + "</p>"
        )

        # --- Per-model sections ---
        def fig_or_blank(rel):
            if not rel:
                return "<div class='note'>No score-based curve (model did not provide scores).</div>"
            return f"<img src='{rel}' alt=''>"

        per_model = []
        for r in self.results:
            cm = np.array(r.confusion_matrix)
            tp = int(cm[0, 0]); fn = int(cm[0, 1])
            fp = int(cm[1, 0]); tn = int(cm[1, 1])
            per_model.append(f"""
            <div class='model-block'>
              <h3>{r.feature_name} &middot; {r.model_name}</h3>
              <table class='figs-table'><tr>
                <td><img src='{r.cm_path}' alt=''></td>
                <td>{fig_or_blank(r.roc_path)}</td>
                <td>{fig_or_blank(r.pr_path)}</td>
              </tr></table>
              <table class='metrics'>
                <tr><th>Accuracy</th><td>{fmt(r.accuracy)}</td>
                    <th>AUC</th><td>{fmt(r.auc)}</td>
                    <th>Average precision</th><td>{fmt(r.average_precision)}</td></tr>
                <tr><th>Precision ({self.anomaly_name})</th><td>{fmt(r.precision_anomaly)}</td>
                    <th>Recall ({self.anomaly_name})</th><td>{fmt(r.recall_anomaly)}</td>
                    <th>F1 ({self.anomaly_name})</th><td>{fmt(r.f1_anomaly)}</td></tr>
                <tr><th>Precision ({self.normal_name})</th><td>{fmt(r.precision_normal)}</td>
                    <th>Recall ({self.normal_name})</th><td>{fmt(r.recall_normal)}</td>
                    <th>F1 ({self.normal_name})</th><td>{fmt(r.f1_normal)}</td></tr>
                <tr><th>True {self.anomaly_name}</th><td>{tp}</td>
                    <th>False {self.normal_name}</th><td>{fn}</td>
                    <th>n test</th><td>{r.n_test}</td></tr>
                <tr><th>False {self.anomaly_name}</th><td>{fp}</td>
                    <th>True {self.normal_name}</th><td>{tn}</td>
                    <th>Train time (s)</th><td>{fmt(r.train_time_s, 2)}</td></tr>
                <tr><th>Inference total (s)</th><td>{fmt(r.latency_total_s, 3)}</td>
                    <th>ms / sample</th><td>{fmt(r.latency_ms_sample)}</td>
                    <th>Notes</th><td>{r.notes or '—'}</td></tr>
              </table>
            </div>
            """)

        # --- Overlays ---
        overlays = []
        if roc_overlay_path:
            rel = os.path.relpath(roc_overlay_path, self.run_dir)
            overlays.append(f"<div><img src='{rel}' alt=''></div>")
        if pr_overlay_path:
            rel = os.path.relpath(pr_overlay_path, self.run_dir)
            overlays.append(f"<div><img src='{rel}' alt=''></div>")
        overlay_block = (
            "<section><h2>All models</h2>"
            "<table class='figs-table'><tr>"
            + "".join(f"<td>{o}</td>" for o in overlays)
            + "</tr></table></section>"
            if overlays else ""
        )

        # --- Data summary ---
        def split_row(name, s):
            if not s:
                return ""
            return (f"<tr><td>{name}</td><td>{s['n_samples']}</td>"
                    f"<td>{s['n_anomaly']}</td><td>{s['n_normal']}</td></tr>")

        data_rows = "".join([
            split_row("Train", ds.get("train")),
            split_row("Validation", ds.get("val")),
            split_row("Test", ds.get("test")),
        ])
        data_table = (
            f"<table class='small'><thead><tr><th>Split</th><th>n</th>"
            f"<th>{self.anomaly_name}</th><th>{self.normal_name}</th>"
            f"</tr></thead><tbody>{data_rows}</tbody></table>"
            if data_rows else "<p class='note'>No data summary recorded.</p>"
        )

        # --- Hyperparams + extraction times + env ---
        def kv_table(d, key_label="Key", val_label="Value"):
            if not d:
                return "<p class='note'>None recorded.</p>"
            rows = "".join(
                f"<tr><td>{k}</td><td>{json.dumps(v) if not isinstance(v, str) else v}</td></tr>"
                for k, v in d.items()
            )
            return (f"<table class='small'><thead><tr><th>{key_label}</th>"
                    f"<th>{val_label}</th></tr></thead><tbody>{rows}</tbody></table>")

        ext_times_table = (
            "<table class='small'><thead><tr><th>Feature</th><th>Seconds</th>"
            "</tr></thead><tbody>"
            + "".join(f"<tr><td>{k}</td><td>{fmt(v, 1)}</td></tr>"
                      for k, v in self.extraction_times.items())
            + "</tbody></table>"
            if self.extraction_times else "<p class='note'>Not recorded.</p>"
        )

        # --- Best model summary ---
        if not df_sorted.empty:
            best = df_sorted.iloc[0]
            best_block = f"""
              <div class='best'>
                <div><span class='best-label'>Best by AUC</span></div>
                <div class='best-headline'>
                  {best['feature_name']} &middot; {best['model_name']}
                </div>
                <div class='best-stats'>
                  AUC {fmt(best['auc'])} &nbsp;·&nbsp;
                  F1 ({self.anomaly_name}) {fmt(best['f1_anomaly'])} &nbsp;·&nbsp;
                  Accuracy {fmt(best['accuracy'])}
                </div>
              </div>
            """
        else:
            best_block = ""

        env = meta["environment"]
        ts_human = self.timestamp.strftime("%Y-%m-%d %H:%M:%S")

        css = """
        @page{margin:0.5in;}
        body{font-family:-apple-system,BlinkMacSystemFont,"Segoe UI",Roboto,sans-serif;
             margin:0;padding:0;color:#1a1a1a;background:#fafafa;}
        .container{max-width:1100px;margin:0 auto;padding:32px 24px 80px;}
        h1{font-size:28px;margin:0 0 4px;}
        h2{font-size:20px;margin:36px 0 12px;border-bottom:1px solid #e5e5e5;padding-bottom:6px;}
        h3{font-size:16px;margin:24px 0 8px;}
        .meta{color:#555;font-size:14px;margin-bottom:24px;}
        .meta code{background:#eee;padding:1px 6px;border-radius:4px;font-size:13px;}
        table{border-collapse:collapse;width:100%;background:white;font-size:14px;}
        table.headline{font-size:11px;}table.headline th, table.headline td{border:1px solid #e1e1e1;padding:4px 5px;text-align:right;}
        table.headline th:first-child, table.headline td:first-child,
        table.headline th:nth-child(2), table.headline td:nth-child(2){text-align:left;}
        table.headline thead{background:#f0f0f0;}
        table.headline tbody tr:nth-child(even){background:#fafafa;}
        table.metrics{margin-top:8px;font-size:13px;}
        table.metrics th{background:#f7f7f7;padding:6px 10px;text-align:left;font-weight:600;color:#444;}
        table.metrics td{padding:6px 10px;border-bottom:1px solid #f0f0f0;}
        table.small{font-size:13px;}
        table.small th{background:#f7f7f7;padding:6px 10px;text-align:left;}
        table.small td{padding:6px 10px;border-bottom:1px solid #f0f0f0;}
        .figs-table{width:100%;border-collapse:separate;border-spacing:6px;margin:8px 0 12px;}
        .figs-table td{width:33%;vertical-align:top;}
        .figs-table img{width:100%;height:auto;border:1px solid #eee;border-radius:4px;background:white;}
        .figs-table .note{color:#888;font-style:italic;font-size:12px;padding:8px;}
        .model-block{background:white;border:1px solid #e5e5e5;border-radius:8px;padding:12px;margin-bottom:12px;page-break-inside:avoid;}
        .model-block h3{page-break-after:avoid;margin:0 0 8px;}
        h2{page-break-after:avoid;}
        .note{color:#888;font-style:italic;font-size:13px;}
        .best{background:white;border-left:4px solid #2563eb;padding:14px 18px;border-radius:4px;margin:12px 0 24px;}
        .best-label{font-size:11px;letter-spacing:0.08em;text-transform:uppercase;color:#2563eb;}
        .best-headline{font-size:18px;font-weight:600;margin:4px 0;}
        .best-stats{color:#555;font-size:14px;}
        section{margin-bottom:8px;}
        """

        return f"""<!doctype html>
<html lang='en'><head><meta charset='utf-8'>
<title>Training report — {self.run_id}</title>
<style>{css}</style></head><body><div class='container'>

  <h1>Training report</h1>
  <div class='meta'>
    <div><b>Run:</b> <code>{self.run_id}</code></div>
    <div><b>Timestamp:</b> {ts_human}</div>
    <div><b>Dataset:</b> {self.dataset} &middot;
         <b>Augmented:</b> {self.augmented} &middot;
         <b>Class encoding:</b> {self.anomaly_label}={self.anomaly_name}, {self.normal_label}={self.normal_name}</div>
  </div>

  {best_block}

  <section><h2>Headline results</h2>{headline_table}</section>

  <section><h2>Data summary</h2>{data_table}</section>

  <section><h2>Per-model results</h2>
    {''.join(per_model) if per_model else "<p class='note'>No model results recorded.</p>"}
  </section>

  {overlay_block}

  <section><h2>Pipeline hyperparameters</h2>{kv_table(self.hyperparams)}</section>
  <section><h2>Feature-extraction times</h2>{ext_times_table}</section>
  <section><h2>Environment</h2>{kv_table(env)}</section>
  <section><h2>Extra metadata</h2>{kv_table(self.extra_metadata)}</section>

</div></body></html>"""
