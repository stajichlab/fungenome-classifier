"""
fungal_classifier/evaluation/shap_analysis.py

SHAP-based feature importance and interpretation.

Provides:
  - Per-block SHAP value computation (TreeExplainer for XGBoost/LightGBM)
  - Top-feature summaries per class
  - Block-level contribution scores (which feature type matters most?)
  - Visualization helpers (summary plots, force plots, block contribution bars)
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Literal

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)


# ── SHAP computation ──────────────────────────────────────────────────────────


def compute_shap_values(
    classifier,
    X: pd.DataFrame,
    explainer_type: Literal["tree", "linear", "kernel"] = "tree",
    n_background_samples: int = 100,
) -> np.ndarray:
    """
    Compute SHAP values for a trained classifier.

    Parameters
    ----------
    classifier        : Fitted BlockClassifier or sklearn estimator.
    X                 : Feature matrix.
    explainer_type    : SHAP explainer type.
    n_background_samples : For KernelExplainer only.

    Returns
    -------
    np.ndarray of shape (n_samples, n_features) for binary,
    or (n_classes, n_samples, n_features) for multiclass.
    """
    import shap

    model = classifier._model if hasattr(classifier, "_model") else classifier

    if explainer_type == "tree":
        explainer = shap.TreeExplainer(model)
        shap_values = explainer.shap_values(X.values)
    elif explainer_type == "linear":
        explainer = shap.LinearExplainer(model, X.values)
        shap_values = explainer.shap_values(X.values)
    else:
        background = shap.sample(X.values, n_background_samples)
        explainer = shap.KernelExplainer(model.predict_proba, background)
        shap_values = explainer.shap_values(X.values, nsamples=100)

    return shap_values


def mean_absolute_shap(
    shap_values: np.ndarray,
    feature_names: list[str],
) -> pd.Series:
    """
    Compute mean |SHAP| per feature (global importance).

    For multiclass (list of arrays), averages across classes.

    Returns
    -------
    pd.Series sorted descending by importance.
    """
    if isinstance(shap_values, list):
        # Multiclass: list of (n_samples, n_features) arrays
        abs_shap = np.mean([np.abs(sv) for sv in shap_values], axis=0)
    else:
        abs_shap = np.abs(shap_values)

    mean_abs = abs_shap.mean(axis=0)
    return pd.Series(mean_abs, index=feature_names).sort_values(ascending=False)


def per_class_shap_summary(
    shap_values: list[np.ndarray],
    feature_names: list[str],
    class_names: list[str],
    top_n: int = 20,
) -> pd.DataFrame:
    """
    Build a DataFrame of top-N features per class by mean |SHAP|.

    Returns
    -------
    pd.DataFrame with columns: class, rank, feature, mean_abs_shap.
    """
    rows = []
    for cls_idx, cls_name in enumerate(class_names):
        if isinstance(shap_values, list):
            sv = shap_values[cls_idx]
        else:
            sv = shap_values[:, :, cls_idx] if shap_values.ndim == 3 else shap_values

        mean_abs = np.abs(sv).mean(axis=0)
        ranked = np.argsort(mean_abs)[::-1][:top_n]
        for rank, feat_idx in enumerate(ranked):
            rows.append(
                {
                    "class": cls_name,
                    "rank": rank + 1,
                    "feature": feature_names[feat_idx],
                    "mean_abs_shap": float(mean_abs[feat_idx]),
                }
            )
    return pd.DataFrame(rows)


# ── block-level attribution ───────────────────────────────────────────────────


def block_level_importance(
    mean_abs_shap: pd.Series,
    block_prefix_sep: str = "__",
) -> pd.Series:
    """
    Aggregate per-feature SHAP importance to block level.

    Expects feature names with format: blockname__featurename
    (as produced by BlockFusionPipeline.concat_fusion with prefix_cols=True).

    Returns
    -------
    pd.Series: block_name -> total mean |SHAP|, sorted descending.
    """
    block_totals: dict[str, float] = {}
    for feat, val in mean_abs_shap.items():
        if block_prefix_sep in str(feat):
            block = str(feat).split(block_prefix_sep)[0]
        else:
            block = "unknown"
        block_totals[block] = block_totals.get(block, 0.0) + val

    return pd.Series(block_totals).sort_values(ascending=False)


# ── visualization ─────────────────────────────────────────────────────────────


def plot_shap_summary(
    shap_values: np.ndarray,
    X: pd.DataFrame,
    max_display: int = 20,
    title: str = "SHAP Feature Importance",
    save_path: Path | None = None,
) -> plt.Figure:
    """
    Beeswarm SHAP summary plot.

    If multiclass, plots mean |SHAP| bar chart instead.
    """
    import shap

    fig, ax = plt.subplots(figsize=(10, max_display * 0.35 + 2))

    if isinstance(shap_values, list):
        # Multiclass: plot global mean |SHAP| as bar chart
        mean_abs = mean_absolute_shap(shap_values, X.columns.tolist())
        top = mean_abs.head(max_display)
        top.sort_values().plot(kind="barh", ax=ax, color="#1a6fa8")
        ax.set_xlabel("Mean |SHAP|")
        ax.set_title(title)
    else:
        shap.summary_plot(shap_values, X, max_display=max_display, show=False, plot_type="dot")
        fig = plt.gcf()
        fig.suptitle(title)

    plt.tight_layout()
    if save_path:
        fig.savefig(save_path, dpi=150, bbox_inches="tight")
        logger.info(f"Saved SHAP summary plot to {save_path}")
    return fig


def plot_block_contributions(
    block_importance: pd.Series,
    title: str = "Feature Block Contributions",
    save_path: Path | None = None,
) -> plt.Figure:
    """Bar chart of total SHAP importance per feature block."""
    colors = ["#1a6fa8", "#e8722a", "#2ca02c", "#d62728", "#9467bd"]
    fig, ax = plt.subplots(figsize=(8, 4))
    block_importance.sort_values().plot(
        kind="barh",
        ax=ax,
        color=colors[: len(block_importance)],
    )
    ax.set_xlabel("Total Mean |SHAP|")
    ax.set_title(title)
    plt.tight_layout()
    if save_path:
        fig.savefig(save_path, dpi=150, bbox_inches="tight")
    return fig


def plot_per_class_heatmap(
    class_shap_df: pd.DataFrame,
    top_n: int = 15,
    save_path: Path | None = None,
) -> plt.Figure:
    """
    Heatmap of top-N features per class.
    Rows = features, columns = classes.
    """
    pivot = class_shap_df[class_shap_df["rank"] <= top_n].pivot_table(
        index="feature", columns="class", values="mean_abs_shap", fill_value=0
    )
    fig, ax = plt.subplots(figsize=(max(8, len(pivot.columns) * 1.2), max(6, len(pivot) * 0.4)))
    import seaborn as sns

    sns.heatmap(pivot, ax=ax, cmap="YlOrRd", linewidths=0.5)
    ax.set_title("SHAP Importance by Class and Feature")
    plt.tight_layout()
    if save_path:
        fig.savefig(save_path, dpi=150, bbox_inches="tight")
    return fig


# ── full SHAP analysis pipeline ───────────────────────────────────────────────


def run_shap_analysis(
    block_classifiers: dict,  # block_name -> BlockClassifier
    feature_blocks: dict,  # block_name -> pd.DataFrame
    class_names: list[str],
    output_dir: Path,
    top_n: int = 20,
) -> dict:
    """
    Run full SHAP analysis for all blocks and save plots + CSVs.

    Parameters
    ----------
    block_classifiers : Dict of fitted BlockClassifier instances.
    feature_blocks    : Dict of feature DataFrames.
    class_names       : List of class label strings.
    output_dir        : Directory to save outputs.
    top_n             : Top features per class to report.

    Returns
    -------
    Dict block_name -> {mean_abs_shap, per_class_summary}.
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    results = {}

    for block_name, clf in block_classifiers.items():
        if block_name not in feature_blocks:
            continue

        logger.info(f"Computing SHAP values for block: {block_name}")
        X = feature_blocks[block_name]

        try:
            shap_values = compute_shap_values(clf, X, explainer_type="tree")
        except Exception as e:
            logger.warning(f"SHAP computation failed for {block_name}: {e}")
            continue

        mean_abs = mean_absolute_shap(shap_values, X.columns.tolist())
        mean_abs.to_csv(output_dir / f"{block_name}_mean_abs_shap.csv")

        if isinstance(shap_values, list):
            class_summary = per_class_shap_summary(
                shap_values, X.columns.tolist(), class_names, top_n
            )
            class_summary.to_csv(output_dir / f"{block_name}_per_class_shap.csv", index=False)

        plot_shap_summary(
            shap_values,
            X,
            max_display=top_n,
            title=f"SHAP: {block_name}",
            save_path=output_dir / f"{block_name}_shap_summary.svg",
        )
        plt.close("all")

        results[block_name] = {"mean_abs_shap": mean_abs}

    return results
