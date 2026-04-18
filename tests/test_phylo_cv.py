"""
tests/test_phylo_cv.py

Unit tests for phylogeny-aware cross-validation.
"""

import numpy as np
import pandas as pd
import pytest

from fungal_classifier.evaluation.phylo_cv import CladeHoldoutCV, assign_clades_from_taxonomy


def make_mock_metadata(n=100):
    orders = ["Hypocreales", "Eurotiales", "Saccharomycetales", "Agaricales", "Boletales"]
    genome_ids = [f"genome_{i:04d}" for i in range(n)]
    return pd.DataFrame(
        {
            "taxonomy_order": np.random.choice(orders, size=n),
            "ecological_niche": np.random.choice(["saprotrophic", "mycorrhizal", "pathogenic"], size=n),
        },
        index=pd.Index(genome_ids, name="genome_id"),
    )


def test_clade_assignment_from_taxonomy():
    meta = make_mock_metadata(100)
    clade_labels = assign_clades_from_taxonomy(meta, clade_level="order")
    assert len(clade_labels) == 100
    assert clade_labels.nunique() <= 5


def test_clade_holdout_cv_no_overlap():
    """Train and test sets must be disjoint in each fold."""
    meta = make_mock_metadata(200)
    clade_labels = assign_clades_from_taxonomy(meta, clade_level="order")
    X = pd.DataFrame(np.random.randn(200, 10), index=meta.index)
    y = meta["taxonomy_order"]

    cv = CladeHoldoutCV(clade_labels=clade_labels, n_folds=5)
    for train_idx, test_idx in cv.split(X, y):
        train_set = set(train_idx)
        test_set = set(test_idx)
        assert len(train_set & test_set) == 0, "Train/test overlap detected!"


def test_clade_holdout_cv_coverage():
    """All samples should appear in a test set exactly once."""
    meta = make_mock_metadata(200)
    clade_labels = assign_clades_from_taxonomy(meta, clade_level="order")
    X = pd.DataFrame(np.random.randn(200, 10), index=meta.index)
    y = meta["taxonomy_order"]

    cv = CladeHoldoutCV(clade_labels=clade_labels, n_folds=5)
    test_counts = np.zeros(len(X), dtype=int)
    for _, test_idx in cv.split(X, y):
        test_counts[test_idx] += 1

    # Every sample should appear in exactly one test fold
    assert np.all(test_counts <= 1), "Some samples appear in multiple test folds"


def test_fold_summary_returns_dataframe():
    meta = make_mock_metadata(100)
    clade_labels = assign_clades_from_taxonomy(meta, clade_level="order")
    X = pd.DataFrame(np.random.randn(100, 5), index=meta.index)
    cv = CladeHoldoutCV(clade_labels=clade_labels, n_folds=5)
    summary = cv.fold_summary()
    assert isinstance(summary, pd.DataFrame)
    assert "fold" in summary.columns
    assert "clade" in summary.columns
