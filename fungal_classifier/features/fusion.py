"""
fungal_classifier/features/fusion.py

Feature block fusion strategies.

Supports:
  - concat:    simple concatenation of normalized feature blocks
  - stacking:  use per-block model probabilities as meta-features
  - attention: learned attention weights per block (PyTorch)
"""
__all__ = ["normalize_block", "filter_low_variance", "select_top_k_univariate", "concat_fusion", "stacking_fusion", "BlockFusionPipeline"]

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
    return_fitters: bool = False,
) -> pd.DataFrame | tuple[pd.DataFrame, Any, "TruncatedSVD | None"]:
    """
    Normalize a feature block and optionally reduce with TruncatedSVD.

    Parameters
    ----------
    df              : Feature matrix (genomes × features).
    scaler          : Scaling strategy.
    svd_components  : If not None, reduce to this many components.
    return_fitters  : If True, also return the fitted scaler and SVD.

    Returns
    -------
    Normalized DataFrame, or (DataFrame, scaler, svd) if return_fitters=True.
    The scaler is None when scaler="none".
    The SVD is None when svd_components is None or >= n_features.
    """
    genome_ids = df.index
    fitted_scaler: Any = None

    if scaler == "standard":
        fitted_scaler = StandardScaler()
        arr = fitted_scaler.fit_transform(df.values)
    elif scaler == "minmax":
        from sklearn.preprocessing import MinMaxScaler
        fitted_scaler = MinMaxScaler()
        arr = fitted_scaler.fit_transform(df.values)
    elif scaler == "robust":
        from sklearn.preprocessing import RobustScaler
        fitted_scaler = RobustScaler()
        arr = fitted_scaler.fit_transform(df.values)
    else:
        arr = df.values.copy()

    fitted_svd: Any = None
    if svd_components is not None and svd_components < arr.shape[1]:
        fitted_svd = TruncatedSVD(n_components=svd_components, random_state=42)
        arr = fitted_svd.fit_transform(arr)
        cols = [f"svd_{i}" for i in range(arr.shape[1])]
    else:
        cols = df.columns.tolist()

    result = pd.DataFrame(arr, index=genome_ids, columns=cols, dtype=np.float32)
    if return_fitters:
        return result, fitted_scaler, fitted_svd
    return result


# ── variance / univariate filtering ──────────────────────────────────────────

def filter_low_variance(
    df: pd.DataFrame,
    threshold: float = 0.01,
    return_selector: bool = False,
) -> pd.DataFrame | tuple[pd.DataFrame, "VarianceThreshold"]:
    """Drop columns with variance below threshold.

    Parameters
    ----------
    df            : Feature DataFrame.
    threshold     : Minimum variance.
    return_selector : If True, also returns the fitted selector.

    Returns
    -------
    Filtered DataFrame, or (DataFrame, selector) if return_selector=True.
    """
    from sklearn.feature_selection import VarianceThreshold

    selector = VarianceThreshold(threshold=threshold)
    arr = selector.fit_transform(df.values)
    kept_cols = df.columns[selector.get_support()]
    result = pd.DataFrame(arr, index=df.index, columns=kept_cols, dtype=np.float32)
    logger.info(
        f"Variance filter: {df.shape[1]} -> {len(kept_cols)} features "
        f"(dropped {df.shape[1] - len(kept_cols)})"
    )
    if return_selector:
        return result, selector
    return result


def select_top_k_univariate(
    df: pd.DataFrame,
    y: pd.Series,
    k: int = 500,
    scoring: Literal["f_classif", "mutual_info_classif"] = "f_classif",
    return_selector: bool = False,
) -> pd.DataFrame | tuple[pd.DataFrame, "SelectKBest"]:
    """Select top-k features by univariate scoring against labels y.

    Parameters
    ----------
    df            : Feature DataFrame.
    y             : Label Series.
    k             : Number of features to keep.
    scoring       : Scoring function.
    return_selector : If True, also returns the fitted selector.

    Returns
    -------
    Reduced DataFrame, or (DataFrame, selector) if return_selector=True.
    """
    from sklearn.feature_selection import SelectKBest, f_classif, mutual_info_classif

    score_fn = f_classif if scoring == "f_classif" else mutual_info_classif
    k = min(k, df.shape[1])
    selector: SelectKBest = SelectKBest(score_fn, k=k)
    arr = selector.fit_transform(df.values, y.values)
    kept_cols = df.columns[selector.get_support()]
    result = pd.DataFrame(arr, index=df.index, columns=kept_cols, dtype=np.float32)
    if return_selector:
        return result, selector
    return result


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
    total_input_ids = sum(len(df) for df in blocks.values())
    fused = fused.dropna(how="all")
    n_dropped = total_input_ids - len(fused)
    if n_dropped > 0:
        logger.warning(
            f"concat_fusion dropped {n_dropped} genomes that were absent from "
            f"all blocks (remaining: {len(fused)})."
        )
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
    logger.info(
        f"Stacking fusion: {fused.shape[1]} meta-features from {len(block_probabilities)} blocks"
    )
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
        # Per-block fitted objects for use in transform()
        self._var_selectors: dict[str, "VarianceThreshold"] = {}
        self._univ_selectors: dict[str, "SelectKBest"] = {}
        self._scalers: dict[str, "StandardScaler | MinMaxScaler | RobustScaler"] = {}
        self._svds: dict[str, "TruncatedSVD"] = {}
        self._block_cols: dict[str, list[str]] = {}  # final column names per block

    def fit_transform(
        self,
        blocks: dict[str, pd.DataFrame],
        y: pd.Series,
    ) -> pd.DataFrame:
        """Fit and transform all feature blocks, return fused matrix."""
        processed: dict[str, pd.DataFrame] = {}

        for name, df in blocks.items():
            logger.info(f"Processing block: {name} ({df.shape})")
            orig_cols = df.columns.tolist()

            # 1. Variance filtering
            if self.variance_threshold > 0:
                df, var_sel = filter_low_variance(
                    df, self.variance_threshold, return_selector=True
                )
                self._var_selectors[name] = var_sel
            else:
                self._var_selectors[name] = None

            # 2. Univariate feature selection
            # 2. Univariate feature selection
            if self.univariate_k is not None:
                y_aligned = y.loc[df.index]
                df, univ_sel = select_top_k_univariate(
                    df, y_aligned, self.univariate_k,
                    self.univariate_scoring, return_selector=True
                )
                self._univ_selectors[name] = univ_sel
            else:
                self._univ_selectors[name] = None

            # 3. Normalise / SVD
            df_out, scaler, svd = normalize_block(
                df, self.scaler, self.svd_components, return_fitters=True
            )
            self._scalers[name] = scaler
            self._svds[name] = svd
            self._block_cols[name] = df_out.columns.tolist()
            processed[name] = df_out

        if self.fusion_strategy == "concat":
            return concat_fusion(processed)
        else:
            raise ValueError("For stacking fusion, use per-block classifiers first.")

    def transform(self, blocks: dict[str, pd.DataFrame]) -> pd.DataFrame:
        """Apply fitted variance filter, univariate selection, scaling, and SVD to new data.

        All fit-time selectors (VarianceThreshold, SelectKBest), scalers, and SVD
        projectors stored during fit_transform are applied to new input blocks.
        """
        processed: dict[str, pd.DataFrame] = {}

        for name, df in blocks.items():
            if name not in self._block_cols:
                logger.warning(
                    f"transform: block '{name}' was not seen during fit_transform — skipping."
                )
                continue

            # 1. Variance filtering (use stored selector on original columns)
            if self._var_selectors[name] is not None:
                var_sel = self._var_selectors[name]
                var_mask = var_sel.get_support()
                var_cols = [c for c, keep in zip(df.columns, var_mask) if keep]
                df = df[var_cols]

            # 2. Univariate selection (use stored selector)
            if self._univ_selectors[name] is not None:
                univ_sel = self._univ_selectors[name]
                # Only pass columns the selector was trained on
                trained_cols = df.columns.tolist()
                # SelectKBest.transform keeps only the selected features from its training set
                # We need to apply the transform using the original training features
                # that the selector saw. Since selectors work on arrays, we apply the mask:
                univ_mask = univ_sel.get_support()
                # Map mask back to current df columns (same order as after var filter)
                univ_cols = [c for c, keep in zip(df.columns, univ_mask) if keep]
                df = df[univ_cols]

            # 3. Scale and optionally SVD-reduce
            df_out = self._transform_block(df, name)
            processed[name] = df_out

        return concat_fusion(processed)

    def _transform_block(self, df: pd.DataFrame, name: str) -> pd.DataFrame:
        """Apply fitted scaler and SVD to a single block's DataFrame."""
        scaler = self._scalers[name]
        svd = self._svds[name]

        if scaler is not None:
            arr = scaler.transform(df.values)
        else:
            arr = df.values

        if svd is not None:
            arr = svd.transform(arr)
            cols = [f"svd_{i}" for i in range(arr.shape[1])]
        else:
            cols = df.columns.tolist()

        return pd.DataFrame(arr, index=df.index, columns=cols, dtype=np.float32)
