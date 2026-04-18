"""
fungal_classifier/utils/preprocessing.py

Shared preprocessing utilities:
  - Sparse feature handling
  - Log and CLR transforms for compositional data
  - Class imbalance strategies (SMOTE, class weights)
  - Label encoding helpers
"""

from __future__ import annotations

import logging

import numpy as np
import pandas as pd
from sklearn.preprocessing import LabelEncoder

logger = logging.getLogger(__name__)


# ── transforms ────────────────────────────────────────────────────────────────

def log1p_transform(df: pd.DataFrame) -> pd.DataFrame:
    """Apply log(1 + x) transform — appropriate for count data (domains, CAZymes, BGC)."""
    return np.log1p(df).astype(np.float32)


def clr_transform(df: pd.DataFrame, pseudocount: float = 1e-6) -> pd.DataFrame:
    """
    Centered log-ratio (CLR) transform for compositional data (k-mer frequencies,
    pathway proportions).

    CLR(x) = log(x / geometric_mean(x))

    Appropriate when features sum to a constant (e.g. relative k-mer frequencies).
    """
    arr = df.values.astype(np.float64) + pseudocount
    geo_mean = np.exp(np.log(arr).mean(axis=1, keepdims=True))
    clr = np.log(arr / geo_mean)
    return pd.DataFrame(clr, index=df.index, columns=df.columns, dtype=np.float32)


def binarize_threshold(df: pd.DataFrame, threshold: float = 0.0) -> pd.DataFrame:
    """Convert to binary presence/absence at given threshold."""
    return (df > threshold).astype(np.float32)


# ── imputation ────────────────────────────────────────────────────────────────

def impute_missing(
    df: pd.DataFrame,
    strategy: str = "median",
) -> pd.DataFrame:
    """
    Impute missing values.

    Parameters
    ----------
    strategy : 'median' | 'mean' | 'zero' | 'knn'
    """
    if strategy == "zero":
        return df.fillna(0.0).astype(np.float32)
    elif strategy in ("median", "mean"):
        from sklearn.impute import SimpleImputer
        imp = SimpleImputer(strategy=strategy)
        arr = imp.fit_transform(df.values)
        return pd.DataFrame(arr, index=df.index, columns=df.columns, dtype=np.float32)
    elif strategy == "knn":
        from sklearn.impute import KNNImputer
        imp = KNNImputer(n_neighbors=5)
        arr = imp.fit_transform(df.values)
        return pd.DataFrame(arr, index=df.index, columns=df.columns, dtype=np.float32)
    else:
        raise ValueError(f"Unknown imputation strategy: {strategy}")


# ── class imbalance ───────────────────────────────────────────────────────────

def compute_class_weights(y: pd.Series) -> dict:
    """
    Compute balanced class weights for imbalanced classification.

    Returns dict suitable for XGBoost sample_weight or sklearn class_weight.
    """
    from sklearn.utils.class_weight import compute_class_weight
    classes = np.unique(y)
    weights = compute_class_weight("balanced", classes=classes, y=y)
    return dict(zip(classes, weights))


def apply_smote(
    X: pd.DataFrame,
    y: pd.Series,
    random_state: int = 42,
    strategy: str = "auto",
) -> tuple[pd.DataFrame, pd.Series]:
    """
    Apply SMOTE oversampling to balance classes.

    Only applies to training data — never call on validation/test splits.
    """
    try:
        from imblearn.over_sampling import SMOTE
    except ImportError:
        raise ImportError("imbalanced-learn required: pip install imbalanced-learn")

    sm = SMOTE(sampling_strategy=strategy, random_state=random_state)
    X_res, y_res = sm.fit_resample(X.values, y.values)

    logger.info(
        f"SMOTE: {len(y)} -> {len(y_res)} samples  |  "
        f"before: {dict(zip(*np.unique(y, return_counts=True)))}  |  "
        f"after: {dict(zip(*np.unique(y_res, return_counts=True)))}"
    )

    # Synthetic samples get integer indices
    n_orig = len(X)
    new_index = list(X.index) + [f"smote_{i}" for i in range(len(y_res) - n_orig)]
    return (
        pd.DataFrame(X_res, index=new_index, columns=X.columns, dtype=np.float32),
        pd.Series(y_res, index=new_index, name=y.name),
    )


# ── label encoding helpers ────────────────────────────────────────────────────

def encode_labels(
    y: pd.Series,
    min_class_size: int = 5,
    other_label: str = "Other",
) -> tuple[pd.Series, LabelEncoder]:
    """
    Encode labels to integers, collapsing rare classes into 'Other'.

    Parameters
    ----------
    min_class_size : Classes with fewer samples are collapsed to other_label.

    Returns
    -------
    (encoded_series, fitted_LabelEncoder)
    """
    counts = y.value_counts()
    rare = counts[counts < min_class_size].index
    if len(rare) > 0:
        logger.info(f"Collapsing {len(rare)} rare classes to '{other_label}': {list(rare)}")
        y = y.copy()
        y[y.isin(rare)] = other_label

    le = LabelEncoder()
    y_enc = pd.Series(le.fit_transform(y), index=y.index, name=y.name)
    return y_enc, le


# ── genome-size correction ────────────────────────────────────────────────────

def correct_for_genome_size(
    df: pd.DataFrame,
    genome_sizes: pd.Series,
) -> pd.DataFrame:
    """
    Divide each feature by the genome size (in Mb) to normalize for genome size.

    Useful for raw count features (domain copy numbers, BGC counts) where
    larger genomes tend to have more of everything.

    Parameters
    ----------
    df           : Feature DataFrame (genome_id index).
    genome_sizes : Series (genome_id -> size in bp).
    """
    common = df.index.intersection(genome_sizes.index)
    size_mb = genome_sizes.loc[common] / 1e6
    corrected = df.loc[common].div(size_mb, axis=0)
    return corrected.astype(np.float32)
