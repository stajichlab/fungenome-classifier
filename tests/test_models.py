"""
tests/test_models.py

Unit tests for block classifiers, fusion model, and deep model.
"""

import numpy as np
import pandas as pd
import pytest

from fungal_classifier.models.block_classifier import BlockClassifier
from fungal_classifier.models.fusion_model import StackingFusionModel


# ── fixtures ──────────────────────────────────────────────────────────────────

def make_feature_block(n=100, d=50, seed=0):
    rng = np.random.default_rng(seed)
    genome_ids = [f"g{i:04d}" for i in range(n)]
    X = pd.DataFrame(rng.standard_normal((n, d)).astype("float32"),
                     index=genome_ids,
                     columns=[f"feat_{j}" for j in range(d)])
    return X


def make_labels(n=100, n_classes=5, seed=0):
    rng = np.random.default_rng(seed)
    classes = [f"Class_{c}" for c in range(n_classes)]
    genome_ids = [f"g{i:04d}" for i in range(n)]
    return pd.Series(rng.choice(classes, size=n), index=genome_ids, name="label")


# ── BlockClassifier ───────────────────────────────────────────────────────────

@pytest.mark.parametrize("model_type", ["random_forest"])
def test_block_classifier_fit_predict(model_type):
    X = make_feature_block()
    y = make_labels()
    clf = BlockClassifier(model_type=model_type, n_estimators=10, random_seed=42)
    clf.fit(X, y)
    preds = clf.predict(X)
    assert len(preds) == len(X)
    assert set(preds).issubset(set(y.unique()))


def test_block_classifier_predict_proba_shape():
    X = make_feature_block()
    y = make_labels(n_classes=4)
    clf = BlockClassifier(model_type="random_forest", n_estimators=10)
    clf.fit(X, y)
    proba = clf.predict_proba(X)
    assert proba.shape == (len(X), 4)
    assert np.allclose(proba.sum(axis=1), 1.0, atol=1e-5)


def test_block_classifier_save_load(tmp_path):
    X = make_feature_block()
    y = make_labels()
    clf = BlockClassifier(model_type="random_forest", n_estimators=5)
    clf.fit(X, y)
    preds_before = clf.predict(X)

    save_path = tmp_path / "clf.pkl"
    clf.save(save_path)
    clf2 = BlockClassifier.load(save_path)
    preds_after = clf2.predict(X)
    assert list(preds_before) == list(preds_after)


def test_feature_importances():
    X = make_feature_block(d=20)
    y = make_labels()
    clf = BlockClassifier(model_type="random_forest", n_estimators=10)
    clf.fit(X, y)
    imp = clf.feature_importances(X.columns.tolist())
    assert len(imp) == 20
    assert imp.sum() > 0


# ── StackingFusionModel ───────────────────────────────────────────────────────

def make_block_results(n=100, n_classes=4, n_blocks=3):
    """Mock block_results as produced by train_all_blocks."""
    classes = [f"Class_{c}" for c in range(n_classes)]
    genome_ids = [f"g{i:04d}" for i in range(n)]
    results = {}
    for b in range(n_blocks):
        rng = np.random.default_rng(b)
        proba = rng.dirichlet(np.ones(n_classes), size=n).astype("float32")
        oof = pd.DataFrame(proba, index=genome_ids, columns=classes)

        X = make_feature_block(n, 30, seed=b)
        y = make_labels(n, n_classes, seed=b)
        clf = BlockClassifier(model_type="random_forest", n_estimators=5)
        clf.fit(X, y)

        results[f"block_{b}"] = {
            "classifier": clf,
            "cv_scores": pd.DataFrame([{"fold": 0, "accuracy": 0.8, "f1_macro": 0.75}]),
            "oof_probabilities": oof,
        }
    return results, genome_ids, classes


def test_stacking_fusion_fit_predict():
    block_results, genome_ids, classes = make_block_results()
    y = make_labels(len(genome_ids), len(classes))
    y.index = genome_ids

    fusion = StackingFusionModel(meta_learner="logistic_regression")
    block_classifiers = {name: res["classifier"] for name, res in block_results.items()}
    fusion.fit_from_block_results(block_results, y, block_classifiers)

    # Predict requires feature_blocks matching block_classifiers
    X_blocks = {name: make_feature_block(50, 30, seed=i)
                for i, name in enumerate(block_classifiers)}
    preds = fusion.predict(X_blocks)
    assert len(preds) == 50


# ── Deep fusion model (shape tests only — no CUDA required) ──────────────────

def test_deep_fusion_forward_shape():
    try:
        import torch
        from fungal_classifier.models.deep_fusion import DeepFusionClassifier

        block_dims = {"kmer": 50, "domains": 80, "pathways": 40}
        n_classes = 6
        model = DeepFusionClassifier(
            block_dims=block_dims, n_classes=n_classes,
            hidden_dim=64, embedding_dim=32, fusion="attention", dropout=0.0
        )
        model.eval()

        batch = {
            name: torch.randn(8, dim)
            for name, dim in block_dims.items()
        }
        with torch.no_grad():
            out = model(batch)

        assert out["logits"].shape == (8, n_classes)
        assert len(out["aux_logits"]) == 3
        for aux in out["aux_logits"].values():
            assert aux.shape == (8, n_classes)

    except ImportError:
        pytest.skip("PyTorch not available")
