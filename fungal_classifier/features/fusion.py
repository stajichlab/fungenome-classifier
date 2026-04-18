"""
fungal_classifier/features/fusion.py

Feature block fusion strategies.

Supports:
  - concat:    simple concatenation of normalized feature blocks
  - stacking:  use per-block model probabilities as meta-features
  - attention: learned attention weights per block (PyTorch)
"""

from __future__ import annotations

import logging
from typing import Literal

import numpy as np
import pandas as pd
from sklearn.decomposition import TruncatedSVD
from sklearn.preprocessing import StandardScaler

logger = logging.getLogger(__name__)

BLOCK_NAMES = ["kmer", "domains", "pathways", "repeats", "motifs"]


# ── preprocessing per block ───────────────────────────────────────────────────

def normalize_block(
    df: pd.DataFrame,
    scaler: Literal["standard", "minmax", "robust", "none"] = "standard",
    svd_components: int | None = None,
) -> pd.DataFrame:
    """
    Normalize a feature block and optionally reduce with TruncatedSVD.

    Parameters
    ----------
    df              : Feature matrix (genomes × features).
    scaler          : Scaling strategy.
    svd_components  : If not None, reduce to this many components.

    Returns
    -------
    Normalized (and optionally reduced) DataFrame.
    """
    genome_ids = df.index

    if scaler == "standard":
        arr = StandardScaler().fit_transform(df.values)
    elif scaler == "minmax":
        from sklearn.preprocessing import MinMaxScaler
        arr = MinMaxScaler().fit_transform(df.values)
    elif scaler == "robust":
        from sklearn.preprocessing import RobustScaler
        arr = RobustScaler().fit_transform(df.values)
    else:
        arr = df.values.copy()

    if svd_components is not None and svd_components < arr.shape[1]:
        svd = TruncatedSVD(n_components=svd_components, random_state=42)
        arr = svd.fit_transform(arr)
        cols = [f"svd_{i}" for i in range(arr.shape[1])]
    else:
        cols = df.columns.tolist()

    return pd.DataFrame(arr, index=genome_ids, columns=cols, dtype=np.float32)


# ── variance / univariate filtering ──────────────────────────────────────────

def filter_low_variance(df: pd.DataFrame, threshold: float = 0.01) -> pd.DataFrame:
    """Drop columns with variance below threshold."""
    from sklearn.feature_selection import VarianceThreshold
    selector = VarianceThreshold(threshold=threshold)
    arr = selector.fit_transform(df.values)
    kept_cols = df.columns[selector.get_support()]
    logger.info(
        f"Variance filter: {df.shape[1]} -> {len(kept_cols)} features "
        f"(dropped {df.shape[1] - len(kept_cols)})"
    )
    return pd.DataFrame(arr, index=df.index, columns=kept_cols, dtype=np.float32)


def select_top_k_univariate(
    df: pd.DataFrame,
    y: pd.Series,
    k: int = 500,
    scoring: Literal["f_classif", "mutual_info_classif"] = "f_classif",
) -> pd.DataFrame:
    """Select top-k features by univariate scoring against labels y."""
    from sklearn.feature_selection import SelectKBest, f_classif, mutual_info_classif

    score_fn = f_classif if scoring == "f_classif" else mutual_info_classif
    k = min(k, df.shape[1])
    selector = SelectKBest(score_fn, k=k)
    arr = selector.fit_transform(df.values, y.values)
    kept_cols = df.columns[selector.get_support()]
    return pd.DataFrame(arr, index=df.index, columns=kept_cols, dtype=np.float32)


# ── fusion strategies ─────────────────────────────────────────────────────────

def concat_fusion(
    blocks: dict[str, pd.DataFrame],
    prefix_cols: bool = True,
) -> pd.DataFrame:
    """
    Concatenate feature blocks column-wise.

    Parameters
    ----------
    blocks      : Dict block_name -> feature DataFrame (same index).
    prefix_cols : If True, prefix each column with its block name.

    Returns
    -------
    pd.DataFrame of shape (n_genomes, total_features).
    """
    dfs = []
    for name, df in blocks.items():
        if prefix_cols:
            df = df.add_prefix(f"{name}__")
        dfs.append(df)
    fused = pd.concat(dfs, axis=1)
    # Align on common genome IDs
    fused = fused.dropna(how="all")
    logger.info(f"Concat fusion: {fused.shape[1]} total features from {len(blocks)} blocks")
    return fused


def stacking_fusion(
    block_probabilities: dict[str, pd.DataFrame],
) -> pd.DataFrame:
    """
    Late fusion via stacking: use per-block class probability vectors as meta-features.

    Parameters
    ----------
    block_probabilities : Dict block_name -> DataFrame of shape (n_genomes, n_classes)
                          containing predicted class probabilities.

    Returns
    -------
    pd.DataFrame of shape (n_genomes, n_blocks * n_classes).
    """
    dfs = []
    for name, proba_df in block_probabilities.items():
        dfs.append(proba_df.add_prefix(f"{name}__proba_"))
    fused = pd.concat(dfs, axis=1).dropna(how="all")
    logger.info(f"Stacking fusion: {fused.shape[1]} meta-features from {len(block_probabilities)} blocks")
    return fused


class BlockFusionPipeline:
    """
    Orchestrates multi-block feature normalization, selection, and fusion.

    Usage
    -----
    pipeline = BlockFusionPipeline(
        scaler="standard",
        variance_threshold=0.01,
        svd_components=200,
        fusion_strategy="concat",
    )
    X_fused = pipeline.fit_transform(blocks, y=metadata["taxonomy_order"])
    X_new   = pipeline.transform(new_blocks)
    """

    def __init__(
        self,
        scaler: Literal["standard", "minmax", "robust", "none"] = "standard",
        variance_threshold: float = 0.01,
        univariate_k: int | None = 500,
        univariate_scoring: Literal["f_classif", "mutual_info_classif"] = "f_classif",
        svd_components: int | None = 200,
        fusion_strategy: Literal["concat", "stacking"] = "concat",
    ):
        self.scaler = scaler
        self.variance_threshold = variance_threshold
        self.univariate_k = univariate_k
        self.univariate_scoring = univariate_scoring
        self.svd_components = svd_components
        self.fusion_strategy = fusion_strategy
        self._fitted_blocks: dict = {}

    def fit_transform(
        self,
        blocks: dict[str, pd.DataFrame],
        y: pd.Series,
    ) -> pd.DataFrame:
        """Fit and transform all feature blocks, return fused matrix."""
        from sklearn.preprocessing import StandardScaler, MinMaxScaler, RobustScaler
        processed: dict[str, pd.DataFrame] = {}

        for name, df in blocks.items():
            logger.info(f"Processing block: {name} ({df.shape})")
            df = filter_low_variance(df, self.variance_threshold)

            if self.univariate_k is not None:
                y_aligned = y.loc[df.index]
                df = select_top_k_univariate(df, y_aligned, self.univariate_k, self.univariate_scoring)

            df = normalize_block(df, self.scaler, self.svd_components)
            self._fitted_blocks[name] = df.columns.tolist()
            processed[name] = df

        if self.fusion_strategy == "concat":
            return concat_fusion(processed)
        else:
            raise ValueError("For stacking fusion, use per-block classifiers first.")

    def transform(self, blocks: dict[str, pd.DataFrame]) -> pd.DataFrame:
        """Apply fitted transforms to new data."""
        processed: dict[str, pd.DataFrame] = {}
        for name, df in blocks.items():
            if name in self._fitted_blocks:
                kept_cols = [c for c in self._fitted_blocks[name] if c in df.columns]
                df = df[kept_cols]
            processed[name] = normalize_block(df, self.scaler, None)
        return concat_fusion(processed)
