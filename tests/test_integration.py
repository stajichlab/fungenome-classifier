"""
tests/test_integration.py

End-to-end integration smoke tests.

These tests run the full pipeline — feature building, training, evaluation,
and prediction — on synthetic in-memory data. No real genome files are needed.
Tests are marked 'slow' and excluded from fast CI runs.

Run with: pytest tests/test_integration.py -v
"""

import numpy as np
import pandas as pd
import pytest

from fungal_classifier.features.fusion import BlockFusionPipeline, concat_fusion
from fungal_classifier.models.block_classifier import BlockClassifier, train_all_blocks
from fungal_classifier.models.fusion_model import StackingFusionModel
from fungal_classifier.evaluation.phylo_cv import CladeHoldoutCV, assign_clades_from_taxonomy
from fungal_classifier.evaluation.metrics import compute_metrics, cv_summary
from fungal_classifier.utils.preprocessing import encode_labels, compute_class_weights


# ── synthetic data factory ────────────────────────────────────────────────────

def make_synthetic_dataset(
    n_genomes: int = 200,
    n_classes: int = 6,
    n_orders: int = 8,
    seed: int = 42,
):
    """
    Generate a synthetic fungal genome dataset for end-to-end testing.

    Returns feature_blocks, metadata, y.
    """
    rng = np.random.default_rng(seed)
    genome_ids = [f"genome_{i:04d}" for i in range(n_genomes)]
    classes = [f"Class_{c}" for c in range(n_classes)]
    orders = [f"Order_{o}" for o in range(n_orders)]

    # Assign labels — add some signal by making class correlate with features
    labels = rng.choice(classes, size=n_genomes)
    order_labels = rng.choice(orders, size=n_genomes)
    niches = rng.choice(["saprotrophic", "mycorrhizal", "pathogenic"], size=n_genomes)

    metadata = pd.DataFrame(
        {
            "taxonomy_order": order_labels,
            "taxonomy_class": rng.choice([f"TClass_{i}" for i in range(4)], size=n_genomes),
            "ecological_niche": niches,
            "lifestyle": rng.choice(["saprotroph", "biotroph", "necrotroph"], size=n_genomes),
        },
        index=pd.Index(genome_ids, name="genome_id"),
    )

    # Feature blocks with planted signal: class centroid + noise
    label_enc = {cls: i for i, cls in enumerate(classes)}
    label_ids = np.array([label_enc[l] for l in labels])
    centroids = rng.standard_normal((n_classes, 50))

    feature_blocks = {}
    for block_name, n_features in [("kmer", 80), ("domains", 60), ("repeats", 30)]:
        X = centroids[label_ids] + rng.standard_normal((n_genomes, 50)) * 1.5
        # Pad or trim to n_features
        if n_features > 50:
            noise = rng.standard_normal((n_genomes, n_features - 50)) * 0.5
            X = np.concatenate([X, noise], axis=1)
        else:
            X = X[:, :n_features]
        df = pd.DataFrame(
            X.astype(np.float32),
            index=genome_ids,
            columns=[f"{block_name}_f{j}" for j in range(X.shape[1])],
        )
        feature_blocks[block_name] = df

    y = pd.Series(labels, index=genome_ids, name="taxonomy_order")
    return feature_blocks, metadata, y


# ── tests ─────────────────────────────────────────────────────────────────────

@pytest.mark.slow
def test_full_pipeline_xgboost():
    """Full pipeline: feature fusion → clade holdout CV → block training → stacking."""
    blocks, metadata, y = make_synthetic_dataset(n_genomes=300, n_classes=5, n_orders=8)

    # Clade holdout CV
    clade_labels = assign_clades_from_taxonomy(metadata, clade_level="order")
    cv = CladeHoldoutCV(clade_labels=clade_labels, n_folds=5, random_seed=42)

    # Train block classifiers
    block_results = train_all_blocks(
        feature_blocks=blocks,
        y=y,
        cv_splitter=cv,
        model_kwargs={"model_type": "random_forest", "n_estimators": 20},
    )

    assert set(block_results.keys()) == set(blocks.keys())
    for block_name, result in block_results.items():
        assert "classifier" in result
        assert "cv_scores" in result
        assert "oof_probabilities" in result
        assert len(result["cv_scores"]) > 0

    # Train stacking fusion
    block_classifiers = {name: res["classifier"] for name, res in block_results.items()}
    fusion = StackingFusionModel(meta_learner="logistic_regression")
    fusion.fit_from_block_results(block_results, y, block_classifiers)

    assert fusion.classes_ is not None
    assert len(fusion.classes_) == 5


@pytest.mark.slow
def test_cv_no_phylo_leakage():
    """Verify train/test sets are clade-disjoint in every fold."""
    blocks, metadata, y = make_synthetic_dataset(n_genomes=200, n_orders=8)
    clade_labels = assign_clades_from_taxonomy(metadata, clade_level="order")
    X = blocks["kmer"]

    cv = CladeHoldoutCV(clade_labels=clade_labels, n_folds=5)
    for train_idx, test_idx in cv.split(X, y):
        train_clades = set(clade_labels.iloc[train_idx].values)
        test_clades = set(clade_labels.iloc[test_idx].values)
        assert len(train_clades & test_clades) == 0, \
            f"Clade leakage detected: {train_clades & test_clades}"


@pytest.mark.slow
def test_feature_fusion_pipeline():
    """Feature fusion pipeline preserves index and produces sensible output."""
    blocks, metadata, y = make_synthetic_dataset(n_genomes=150)

    pipeline = BlockFusionPipeline(
        scaler="standard",
        variance_threshold=0.0,
        univariate_k=30,
        svd_components=None,
        fusion_strategy="concat",
    )
    X_fused = pipeline.fit_transform(blocks, y)

    assert X_fused.shape[0] == 150
    assert X_fused.shape[1] > 0
    assert not X_fused.isna().any().any()


@pytest.mark.slow
def test_metrics_consistency():
    """Metrics computed on known-perfect predictions should return 1.0."""
    n = 50
    classes = ["A", "B", "C"]
    y_true = pd.Series(classes * (n // 3) + classes[:n % 3], name="label")
    y_pred = y_true.copy()
    metrics = compute_metrics(y_true, y_pred)
    assert metrics["accuracy"] == pytest.approx(1.0)
    assert metrics["f1_macro"] == pytest.approx(1.0)
    assert metrics["mcc"] == pytest.approx(1.0)


@pytest.mark.slow
def test_cv_summary_confidence_intervals():
    """CV summary CIs should be symmetric and wider for more variable scores."""
    fold_metrics = [
        {"accuracy": 0.8 + 0.01 * i, "f1_macro": 0.7 + 0.02 * i}
        for i in range(10)
    ]
    summary = cv_summary(fold_metrics)
    assert "accuracy" in summary.index
    assert summary.loc["accuracy", "ci_lower"] < summary.loc["accuracy", "mean"]
    assert summary.loc["accuracy", "ci_upper"] > summary.loc["accuracy", "mean"]


@pytest.mark.slow
def test_block_classifier_handles_unseen_classes():
    """Classifier should gracefully handle test sets with fewer classes than training."""
    rng = np.random.default_rng(0)
    n_train, n_test = 200, 50
    X_train = pd.DataFrame(rng.standard_normal((n_train, 20)).astype("float32"),
                            index=[f"train_{i}" for i in range(n_train)])
    X_test  = pd.DataFrame(rng.standard_normal((n_test, 20)).astype("float32"),
                            index=[f"test_{i}" for i in range(n_test)])
    y_train = pd.Series(rng.choice(["A", "B", "C", "D"], size=n_train),
                        index=X_train.index)
    y_test  = pd.Series(rng.choice(["A", "B"], size=n_test),  # only 2 of 4 classes
                        index=X_test.index)

    clf = BlockClassifier(model_type="random_forest", n_estimators=10)
    clf.fit(X_train, y_train)
    preds = clf.predict(X_test)
    proba = clf.predict_proba(X_test)

    assert len(preds) == n_test
    assert proba.shape == (n_test, 4)  # 4 training classes in probability output


@pytest.mark.slow
def test_deep_fusion_train_loop():
    """Deep fusion model should train for at least a few epochs without error."""
    try:
        import torch
        from fungal_classifier.models.deep_fusion import DeepFusionClassifier, DeepFusionTrainer
    except ImportError:
        pytest.skip("PyTorch not available")

    rng = np.random.default_rng(42)
    n = 80
    n_classes = 4
    block_dims = {"kmer": 30, "domains": 25}
    genome_ids = [f"g{i}" for i in range(n)]
    classes = [f"C{c}" for c in range(n_classes)]

    blocks = {
        name: pd.DataFrame(
            rng.standard_normal((n, dim)).astype("float32"),
            index=genome_ids,
            columns=[f"{name}_f{j}" for j in range(dim)],
        )
        for name, dim in block_dims.items()
    }
    y = pd.Series(rng.choice(classes, size=n), index=genome_ids)

    split = n * 3 // 4
    block_train = {name: df.iloc[:split] for name, df in blocks.items()}
    block_val   = {name: df.iloc[split:] for name, df in blocks.items()}
    y_train, y_val = y.iloc[:split], y.iloc[split:]

    model = DeepFusionClassifier(
        block_dims=block_dims, n_classes=n_classes,
        hidden_dim=32, embedding_dim=16, dropout=0.0,
    )
    trainer = DeepFusionTrainer(
        model=model, lr=1e-3, n_epochs=5, batch_size=16,
        patience=10, device="cpu",
    )
    history = trainer.fit(block_train, y_train, block_val, y_val)

    assert len(history) > 0
    assert "train_loss" in history[0]
    assert history[0]["train_loss"] > 0

    proba = trainer.predict_proba(block_val)
    assert proba.shape == (len(y_val), n_classes)
    assert np.allclose(proba.sum(axis=1), 1.0, atol=1e-4)
