"""
fungal_classifier/evaluation/embeddings.py

Export and visualise genome embeddings from trained models.

Two embedding sources:
  1. Feature-space PCoA/UMAP directly from feature blocks (no model needed)
  2. Learned tower embeddings from the deep fusion model (richer representations)

Both can be exported to TSV for downstream analysis (e.g. in R, iTOL tree annotation).
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Literal

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches

logger = logging.getLogger(__name__)


# ── dimensionality reduction ──────────────────────────────────────────────────

def compute_pca_embedding(
    X: pd.DataFrame,
    n_components: int = 2,
    scale: bool = True,
) -> pd.DataFrame:
    """PCA embedding of a feature matrix."""
    from sklearn.decomposition import PCA
    from sklearn.preprocessing import StandardScaler

    arr = StandardScaler().fit_transform(X.fillna(0).values) if scale else X.fillna(0).values
    pca = PCA(n_components=n_components, random_state=42)
    coords = pca.fit_transform(arr)
    ev = pca.explained_variance_ratio_

    cols = [f"PC{i+1}_({ev[i]:.1%})" for i in range(n_components)]
    df = pd.DataFrame(coords, index=X.index, columns=cols, dtype=np.float32)
    logger.info(f"PCA: {n_components} components, {ev[:2].sum():.1%} variance explained")
    return df


def compute_umap_embedding(
    X: pd.DataFrame,
    n_components: int = 2,
    n_neighbors: int = 15,
    min_dist: float = 0.1,
    metric: str = "euclidean",
    scale: bool = True,
    random_state: int = 42,
) -> pd.DataFrame:
    """UMAP embedding of a feature matrix."""
    try:
        from umap import UMAP
    except ImportError:
        raise ImportError("umap-learn required: pip install umap-learn")

    from sklearn.preprocessing import StandardScaler

    arr = StandardScaler().fit_transform(X.fillna(0).values) if scale else X.fillna(0).values
    reducer = UMAP(
        n_components=n_components,
        n_neighbors=n_neighbors,
        min_dist=min_dist,
        metric=metric,
        random_state=random_state,
        verbose=False,
    )
    coords = reducer.fit_transform(arr)
    cols = [f"UMAP{i+1}" for i in range(n_components)]
    return pd.DataFrame(coords, index=X.index, columns=cols, dtype=np.float32)


def compute_tsne_embedding(
    X: pd.DataFrame,
    n_components: int = 2,
    perplexity: float = 30.0,
    scale: bool = True,
    random_state: int = 42,
) -> pd.DataFrame:
    """t-SNE embedding. For large n (>5000), reduce with PCA first."""
    from sklearn.manifold import TSNE
    from sklearn.preprocessing import StandardScaler

    arr = StandardScaler().fit_transform(X.fillna(0).values) if scale else X.fillna(0).values

    # Pre-reduce with PCA for speed
    if arr.shape[1] > 50:
        from sklearn.decomposition import PCA
        arr = PCA(n_components=50, random_state=random_state).fit_transform(arr)

    tsne = TSNE(
        n_components=n_components,
        perplexity=perplexity,
        random_state=random_state,
        n_jobs=-1,
    )
    coords = tsne.fit_transform(arr)
    cols = [f"tSNE{i+1}" for i in range(n_components)]
    return pd.DataFrame(coords, index=X.index, columns=cols, dtype=np.float32)


# ── deep model tower embeddings ───────────────────────────────────────────────

def extract_tower_embeddings(
    model,
    feature_blocks: dict[str, pd.DataFrame],
    device: str = "cpu",
) -> dict[str, pd.DataFrame]:
    """
    Extract per-block tower embeddings from a trained DeepFusionClassifier.

    Parameters
    ----------
    model         : Trained DeepFusionClassifier (PyTorch).
    feature_blocks: Dict block_name -> feature DataFrame (after SVD reduction).
    device        : 'cpu' or 'cuda'.

    Returns
    -------
    Dict block_name -> embedding DataFrame of shape (n_genomes, embedding_dim).
    """
    try:
        import torch
        import torch.nn.functional as F
    except ImportError:
        raise ImportError("PyTorch required for tower embedding extraction")

    model.eval()
    device = torch.device(device)
    model.to(device)

    genome_ids = list(next(iter(feature_blocks.values())).index)
    block_tensors = {
        name: torch.tensor(df.loc[genome_ids].values.astype(np.float32), device=device)
        for name, df in feature_blocks.items()
        if name in [n for n in model.block_names]
    }

    with torch.no_grad():
        out = model(block_tensors, return_embeddings=True)

    embeddings = {}
    for name, emb_tensor in out["embeddings"].items():
        arr = emb_tensor.cpu().numpy()
        cols = [f"emb_{i}" for i in range(arr.shape[1])]
        embeddings[name] = pd.DataFrame(arr, index=genome_ids, columns=cols, dtype=np.float32)

    logger.info(f"Extracted embeddings for {len(embeddings)} blocks")
    return embeddings


def compute_fused_embedding(
    model,
    feature_blocks: dict[str, pd.DataFrame],
    method: Literal["pca", "umap", "tsne"] = "umap",
    device: str = "cpu",
    **method_kwargs,
) -> pd.DataFrame:
    """
    Extract fused embedding from deep model towers, then reduce to 2D for visualisation.
    """
    tower_embeddings = extract_tower_embeddings(model, feature_blocks, device)

    # Concatenate all tower embeddings
    fused = pd.concat(list(tower_embeddings.values()), axis=1)

    if method == "pca":
        return compute_pca_embedding(fused, n_components=2, **method_kwargs)
    elif method == "umap":
        return compute_umap_embedding(fused, n_components=2, **method_kwargs)
    elif method == "tsne":
        return compute_tsne_embedding(fused, n_components=2, **method_kwargs)
    else:
        raise ValueError(f"Unknown method: {method}")


# ── visualisation ─────────────────────────────────────────────────────────────

def plot_embedding(
    coords: pd.DataFrame,
    metadata: pd.DataFrame,
    color_by: str,
    title: str = "",
    save_path: Path | None = None,
    point_size: int = 12,
    alpha: float = 0.7,
    max_legend_items: int = 25,
) -> plt.Figure:
    """
    Scatter plot of 2D embedding coloured by a metadata column.

    Parameters
    ----------
    coords      : DataFrame with 2 columns (embedding dimensions), genome_id index.
    metadata    : DataFrame with metadata columns, genome_id index.
    color_by    : Column name in metadata to use for colouring.
    """
    common = coords.index.intersection(metadata.index)
    coords_c = coords.loc[common]
    labels = metadata.loc[common, color_by].fillna("Unknown")

    unique_labels = sorted(labels.unique())
    n_labels = len(unique_labels)
    palette = (
        plt.cm.tab20(np.linspace(0, 1, n_labels))
        if n_labels <= 20
        else plt.cm.hsv(np.linspace(0, 1, n_labels, endpoint=False))
    )
    color_map = dict(zip(unique_labels, palette))
    colors = [color_map[l] for l in labels]

    xcol, ycol = coords_c.columns[0], coords_c.columns[1]

    fig, ax = plt.subplots(figsize=(10, 8))
    ax.scatter(
        coords_c[xcol], coords_c[ycol],
        c=colors, s=point_size, alpha=alpha, linewidths=0,
    )
    ax.set_xlabel(xcol)
    ax.set_ylabel(ycol)
    ax.set_title(title or f"Embedding coloured by {color_by}")
    ax.set_aspect("equal", "datalim")

    if n_labels <= max_legend_items:
        handles = [
            mpatches.Patch(color=color_map[l], label=l)
            for l in unique_labels
        ]
        ax.legend(
            handles=handles,
            bbox_to_anchor=(1.02, 1),
            loc="upper left",
            fontsize=max(6, 10 - n_labels // 5),
            title=color_by,
            framealpha=0.8,
        )

    plt.tight_layout()
    if save_path:
        fig.savefig(save_path, dpi=150, bbox_inches="tight")
        logger.info(f"Saved embedding plot to {save_path}")
    return fig


def plot_embedding_grid(
    coords: pd.DataFrame,
    metadata: pd.DataFrame,
    color_cols: list[str],
    save_path: Path | None = None,
) -> plt.Figure:
    """
    Grid of embedding plots, one per metadata column.
    Useful for comparing how taxonomy vs ecology structures the space.
    """
    n = len(color_cols)
    ncols = min(3, n)
    nrows = (n + ncols - 1) // ncols
    fig, axes = plt.subplots(nrows, ncols, figsize=(6 * ncols, 5 * nrows))
    axes = np.array(axes).flatten()

    xcol, ycol = coords.columns[0], coords.columns[1]
    common = coords.index.intersection(metadata.index)

    for ax, col in zip(axes, color_cols):
        if col not in metadata.columns:
            ax.set_visible(False)
            continue
        labels = metadata.loc[common, col].fillna("Unknown")
        unique_labels = sorted(labels.unique())
        palette = plt.cm.tab20(np.linspace(0, 1, len(unique_labels)))
        color_map = dict(zip(unique_labels, palette))
        colors = [color_map[l] for l in labels]

        ax.scatter(coords.loc[common, xcol], coords.loc[common, ycol],
                   c=colors, s=8, alpha=0.6, linewidths=0)
        ax.set_title(col, fontsize=10)
        ax.set_xlabel(xcol, fontsize=8)
        ax.set_ylabel(ycol, fontsize=8)
        ax.tick_params(labelsize=7)

    for ax in axes[n:]:
        ax.set_visible(False)

    plt.suptitle("Genome Embedding — Multiple Annotations", y=1.01, fontsize=13)
    plt.tight_layout()
    if save_path:
        fig.savefig(save_path, dpi=150, bbox_inches="tight")
    return fig


# ── export pipeline ───────────────────────────────────────────────────────────

def run_embedding_export(
    feature_blocks: dict[str, pd.DataFrame],
    metadata: pd.DataFrame,
    output_dir: Path,
    methods: list[Literal["pca", "umap", "tsne"]] = ["pca", "umap"],
    color_cols: list[str] = ["taxonomy_order", "ecological_niche", "lifestyle"],
    deep_model=None,
    deep_feature_blocks: dict | None = None,
) -> dict[str, pd.DataFrame]:
    """
    Run full embedding export pipeline for all blocks and methods.

    Parameters
    ----------
    feature_blocks       : Dict block_name -> feature DataFrame.
    metadata             : Genome metadata DataFrame.
    output_dir           : Output directory.
    methods              : Embedding methods to compute.
    color_cols           : Metadata columns for colouring plots.
    deep_model           : Optional trained DeepFusionClassifier.
    deep_feature_blocks  : Feature blocks after SVD reduction (for deep model).

    Returns
    -------
    Dict of embedding_name -> coords DataFrame.
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    all_embeddings: dict[str, pd.DataFrame] = {}
    color_cols = [c for c in color_cols if c in metadata.columns]

    for block_name, X in feature_blocks.items():
        for method in methods:
            embed_name = f"{block_name}_{method}"
            logger.info(f"Computing {method.upper()} for block: {block_name}...")
            try:
                if method == "pca":
                    coords = compute_pca_embedding(X)
                elif method == "umap":
                    coords = compute_umap_embedding(X)
                elif method == "tsne":
                    coords = compute_tsne_embedding(X)
                else:
                    continue

                all_embeddings[embed_name] = coords

                # Save coordinates as TSV
                coords.to_csv(output_dir / f"{embed_name}_coords.tsv", sep="\t")

                # Plot grid
                fig = plot_embedding_grid(
                    coords, metadata, color_cols,
                    save_path=output_dir / f"{embed_name}_grid.svg",
                )
                plt.close(fig)

            except Exception as e:
                logger.warning(f"Embedding failed for {embed_name}: {e}")

    # Deep model embeddings
    if deep_model is not None and deep_feature_blocks is not None:
        logger.info("Extracting deep tower embeddings...")
        try:
            for method in ["umap", "pca"]:
                coords = compute_fused_embedding(deep_model, deep_feature_blocks, method=method)
                all_embeddings[f"deep_fusion_{method}"] = coords
                coords.to_csv(output_dir / f"deep_fusion_{method}_coords.tsv", sep="\t")
                fig = plot_embedding_grid(
                    coords, metadata, color_cols,
                    save_path=output_dir / f"deep_fusion_{method}_grid.svg",
                )
                plt.close(fig)
        except Exception as e:
            logger.warning(f"Deep model embedding failed: {e}")

    logger.info(f"Embeddings saved to {output_dir}")
    return all_embeddings
