"""
tests/test_preprocessing.py

Unit tests for preprocessing utilities.
"""

import numpy as np
import pandas as pd

from fungal_classifier.utils.preprocessing import (
    binarize_threshold,
    clr_transform,
    compute_class_weights,
    correct_for_genome_size,
    encode_labels,
    impute_missing,
    log1p_transform,
)


def make_df(n=20, d=10, seed=0):
    rng = np.random.default_rng(seed)
    ids = [f"g{i:04d}" for i in range(n)]
    return pd.DataFrame(
        np.abs(rng.standard_normal((n, d))).astype("float32"),
        index=ids,
        columns=[f"f{j}" for j in range(d)],
    )


# ── transforms ────────────────────────────────────────────────────────────────


def test_log1p_non_negative():
    df = make_df()
    result = log1p_transform(df)
    assert (result >= 0).all().all()
    assert result.shape == df.shape


def test_log1p_zero_stays_zero():
    df = pd.DataFrame({"a": [0.0, 1.0]}, index=["g0", "g1"])
    result = log1p_transform(df)
    assert result.loc["g0", "a"] == 0.0


def test_clr_row_sums_to_zero():
    df = make_df()
    df = df + 0.001  # ensure no zeros
    result = clr_transform(df)
    row_sums = result.sum(axis=1)
    np.testing.assert_allclose(row_sums, 0.0, atol=1e-4)


def test_clr_shape_preserved():
    df = make_df()
    result = clr_transform(df)
    assert result.shape == df.shape


def test_binarize():
    df = pd.DataFrame({"a": [0.0, 0.5, 2.0]}, index=["g0", "g1", "g2"])
    result = binarize_threshold(df, threshold=0.1)
    assert result.loc["g0", "a"] == 0.0
    assert result.loc["g1", "a"] == 1.0
    assert result.loc["g2", "a"] == 1.0


# ── imputation ────────────────────────────────────────────────────────────────


def test_impute_zero():
    df = make_df()
    df.iloc[0, 0] = np.nan
    result = impute_missing(df, strategy="zero")
    assert result.iloc[0, 0] == 0.0
    assert not result.isna().any().any()


def test_impute_median():
    df = make_df()
    df.iloc[0, 0] = np.nan
    col_median = df.iloc[:, 0].median()
    result = impute_missing(df, strategy="median")
    assert not result.isna().any().any()
    assert abs(result.iloc[0, 0] - col_median) < 1e-3


# ── class weights ──────────────────────────────────────────────────────────────


def test_compute_class_weights_balanced():
    y = pd.Series(["A"] * 90 + ["B"] * 10)
    weights = compute_class_weights(y)
    assert weights["A"] < weights["B"], "Minority class should have higher weight"


def test_compute_class_weights_keys():
    y = pd.Series(["X", "Y", "Z", "X", "Y"])
    weights = compute_class_weights(y)
    assert set(weights.keys()) == {"X", "Y", "Z"}


# ── label encoding ────────────────────────────────────────────────────────────


def test_encode_labels_basic():
    y = pd.Series(["cat", "dog", "cat", "bird"] * 10)
    y_enc, le = encode_labels(y, min_class_size=2)
    assert set(y_enc.unique()).issubset(set(range(len(le.classes_))))


def test_encode_labels_collapses_rare():
    y = pd.Series(["common"] * 50 + ["rare"] * 2)
    y_enc, le = encode_labels(y, min_class_size=5, other_label="Other")
    assert "Other" in le.classes_
    assert "rare" not in le.classes_


# ── genome size correction ────────────────────────────────────────────────────


def test_correct_for_genome_size():
    df = pd.DataFrame({"domain_A": [100.0, 200.0]}, index=["g0", "g1"])
    sizes = pd.Series([10_000_000, 20_000_000], index=["g0", "g1"])  # 10 Mb, 20 Mb
    result = correct_for_genome_size(df, sizes)
    # Both should give same density (100 / 10Mb = 200 / 20Mb = 10 per Mb)
    assert abs(result.loc["g0", "domain_A"] - result.loc["g1", "domain_A"]) < 1e-4
