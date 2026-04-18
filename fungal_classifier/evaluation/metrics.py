"""
fungal_classifier/evaluation/metrics.py

Classification metrics, reporting, and visualization for multi-class fungal genome classifiers.

Provides:
  - Per-class and macro metrics (accuracy, F1, precision, recall, MCC)
  - Confusion matrix plots
  - CV score summaries with confidence intervals
  - Block-level performance comparison table
"""

from __future__ import annotations

import logging
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.metrics import (
    accuracy_score,
    balanced_accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
    matthews_corrcoef,
    precision_score,
    recall_score,
)

logger = logging.getLogger(__name__)


# ── core metrics ──────────────────────────────────────────────────────────────

def compute_metrics(
    y_true: pd.Series | np.ndarray,
    y_pred: pd.Series | np.ndarray,
    prefix: str = "",
) -> dict[str, float]:
    """
    Compute a comprehensive set of classification metrics.

    Parameters
    ----------
    y_true  : Ground-truth labels.
    y_pred  : Predicted labels.
    prefix  : Optional prefix for metric names (e.g. 'val_').

    Returns
    -------
    Dict of metric_name -> value.
    """
    metrics = {
        f"{prefix}accuracy": accuracy_score(y_true, y_pred),
        f"{prefix}balanced_accuracy": balanced_accuracy_score(y_true, y_pred),
        f"{prefix}f1_macro": f1_score(y_true, y_pred, average="macro", zero_division=0),
        f"{prefix}f1_weighted": f1_score(y_true, y_pred, average="weighted", zero_division=0),
        f"{prefix}precision_macro": precision_score(y_true, y_pred, average="macro", zero_division=0),
        f"{prefix}recall_macro": recall_score(y_true, y_pred, average="macro", zero_division=0),
        f"{prefix}mcc": matthews_corrcoef(y_true, y_pred),
    }
    return metrics


def per_class_metrics(
    y_true: pd.Series | np.ndarray,
    y_pred: pd.Series | np.ndarray,
) -> pd.DataFrame:
    """
    Return a DataFrame of per-class precision, recall, F1, and support.
    """
    report = classification_report(y_true, y_pred, output_dict=True, zero_division=0)
    df = pd.DataFrame(report).T
    # Drop 'accuracy', 'macro avg', 'weighted avg' for the per-class view
    drop_rows = ["accuracy", "macro avg", "weighted avg"]
    df = df.drop(index=[r for r in drop_rows if r in df.index])
    return df[["precision", "recall", "f1-score", "support"]].sort_values(
        "f1-score", ascending=False
    )


# ── CV summary ────────────────────────────────────────────────────────────────

def cv_summary(
    fold_metrics: list[dict],
    ci: float = 0.95,
) -> pd.DataFrame:
    """
    Summarize cross-validation fold metrics with mean and confidence intervals.

    Parameters
    ----------
    fold_metrics : List of metric dicts, one per fold.
    ci           : Confidence interval level (default 95%).

    Returns
    -------
    pd.DataFrame with columns: metric, mean, std, ci_lower, ci_upper.
    """
    df = pd.DataFrame(fold_metrics)
    numeric = df.select_dtypes(include=np.number)
    z = 1.96 if ci == 0.95 else 2.576  # 95% or 99%
    n = len(numeric)

    summary_rows = []
    for col in numeric.columns:
        vals = numeric[col].dropna()
        mean = vals.mean()
        std = vals.std()
        se = std / np.sqrt(len(vals))
        summary_rows.append({
            "metric": col,
            "mean": mean,
            "std": std,
            "ci_lower": mean - z * se,
            "ci_upper": mean + z * se,
        })
    return pd.DataFrame(summary_rows).set_index("metric")


def block_comparison_table(
    block_cv_scores: dict[str, pd.DataFrame],
    metric: str = "f1_macro",
) -> pd.DataFrame:
    """
    Build a comparison table of CV performance across feature blocks.

    Parameters
    ----------
    block_cv_scores : Dict block_name -> DataFrame of per-fold scores.
    metric          : Metric column to compare.

    Returns
    -------
    pd.DataFrame with block names and mean ± std for the metric.
    """
    rows = []
    for block_name, scores_df in block_cv_scores.items():
        if metric not in scores_df.columns:
            continue
        vals = scores_df[metric]
        rows.append({
            "block": block_name,
            f"mean_{metric}": vals.mean(),
            f"std_{metric}": vals.std(),
        })
    df = pd.DataFrame(rows).sort_values(f"mean_{metric}", ascending=False)
    df["mean ± std"] = df.apply(
        lambda r: f"{r[f'mean_{metric}']:.3f} ± {r[f'std_{metric}']:.3f}", axis=1
    )
    return df


# ── visualization ─────────────────────────────────────────────────────────────

def plot_confusion_matrix(
    y_true: pd.Series | np.ndarray,
    y_pred: pd.Series | np.ndarray,
    labels: list[str] | None = None,
    normalize: bool = True,
    title: str = "Confusion Matrix",
    save_path: Path | None = None,
    figsize: tuple[int, int] | None = None,
) -> plt.Figure:
    """
    Plot a confusion matrix heatmap.

    Parameters
    ----------
    normalize : If True, normalize rows to sum to 1 (shows recall per class).
    """
    import seaborn as sns

    cm = confusion_matrix(y_true, y_pred, labels=labels)
    if normalize:
        cm = cm.astype(float) / cm.sum(axis=1, keepdims=True)
        fmt = ".2f"
    else:
        fmt = "d"

    n_classes = cm.shape[0]
    if figsize is None:
        figsize = (max(8, n_classes * 0.6), max(6, n_classes * 0.5))

    fig, ax = plt.subplots(figsize=figsize)
    sns.heatmap(
        cm, annot=True, fmt=fmt, cmap="Blues",
        xticklabels=labels, yticklabels=labels,
        ax=ax, linewidths=0.5,
        annot_kws={"size": max(6, 10 - n_classes // 5)},
    )
    ax.set_xlabel("Predicted")
    ax.set_ylabel("True")
    ax.set_title(title)
    plt.tight_layout()

    if save_path:
        fig.savefig(save_path, dpi=150, bbox_inches="tight")
        logger.info(f"Saved confusion matrix to {save_path}")
    return fig


def plot_cv_scores(
    block_cv_scores: dict[str, pd.DataFrame],
    metric: str = "f1_macro",
    title: str = "Cross-Validation Performance by Feature Block",
    save_path: Path | None = None,
) -> plt.Figure:
    """Box plot of CV metric distributions across blocks."""
    data = {
        block: scores[metric].values
        for block, scores in block_cv_scores.items()
        if metric in scores.columns
    }
    blocks = sorted(data.keys(), key=lambda b: np.mean(data[b]), reverse=True)

    fig, ax = plt.subplots(figsize=(max(6, len(blocks) * 1.2), 5))
    bp = ax.boxplot(
        [data[b] for b in blocks],
        labels=blocks,
        patch_artist=True,
        medianprops={"color": "black", "linewidth": 2},
    )
    colors = ["#1a6fa8", "#e8722a", "#2ca02c", "#d62728", "#9467bd", "#8c564b"]
    for patch, color in zip(bp["boxes"], colors * 10):
        patch.set_facecolor(color)
        patch.set_alpha(0.7)

    ax.set_ylabel(metric.replace("_", " ").title())
    ax.set_title(title)
    ax.grid(axis="y", alpha=0.3)
    plt.tight_layout()

    if save_path:
        fig.savefig(save_path, dpi=150, bbox_inches="tight")
    return fig


# ── reporting ─────────────────────────────────────────────────────────────────

def print_evaluation_report(
    y_true: pd.Series,
    y_pred: pd.Series,
    model_name: str = "Model",
) -> None:
    """Print a formatted evaluation report to stdout."""
    metrics = compute_metrics(y_true, y_pred)
    print(f"\n{'='*50}")
    print(f"  {model_name} Evaluation Report")
    print(f"{'='*50}")
    for name, val in metrics.items():
        print(f"  {name:<30} {val:.4f}")
    print(f"\nPer-class breakdown:")
    print(per_class_metrics(y_true, y_pred).to_string())
    print(f"{'='*50}\n")
