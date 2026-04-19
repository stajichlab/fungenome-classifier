"""
fungal_classifier/models/block_classifier.py

Per-feature-block classifiers.

Each block (kmer, domains, pathways, repeats, motifs) gets its own classifier.
Results from block-wise training are used to:
  1. Understand which feature types carry the most signal.
  2. Generate block-level probability vectors for stacking/late fusion.
"""

from __future__ import annotations

import logging
import pickle
from pathlib import Path
from typing import Literal

import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, ClassifierMixin
from sklearn.preprocessing import LabelEncoder

logger = logging.getLogger(__name__)


# ── block classifier ──────────────────────────────────────────────────────────


class BlockClassifier(BaseEstimator, ClassifierMixin):
    """
    Wraps XGBoost or LightGBM as a per-block classifier.

    Parameters
    ----------
    model_type      : 'xgboost' or 'lightgbm'.
    n_estimators    : Number of boosting rounds.
    max_depth       : Maximum tree depth.
    learning_rate   : Shrinkage rate.
    subsample       : Row subsampling ratio.
    colsample       : Column subsampling ratio.
    early_stopping  : Rounds for early stopping (requires eval set).
    random_seed     : Reproducibility seed.
    """

    def __init__(
        self,
        model_type: Literal["xgboost", "lightgbm", "random_forest"] = "xgboost",
        n_estimators: int = 500,
        max_depth: int = 6,
        learning_rate: float = 0.05,
        subsample: float = 0.8,
        colsample: float = 0.8,
        early_stopping: int = 50,
        random_seed: int = 42,
    ):
        self.model_type = model_type
        self.n_estimators = n_estimators
        self.max_depth = max_depth
        self.learning_rate = learning_rate
        self.subsample = subsample
        self.colsample = colsample
        self.early_stopping = early_stopping
        self.random_seed = random_seed
        self._model = None
        self._label_encoder = LabelEncoder()
        self.classes_: np.ndarray | None = None

    def _build_model(self, n_classes: int):
        if self.model_type == "xgboost":
            import xgboost as xgb

            objective = "multi:softprob" if n_classes > 2 else "binary:logistic"
            return xgb.XGBClassifier(
                n_estimators=self.n_estimators,
                max_depth=self.max_depth,
                learning_rate=self.learning_rate,
                subsample=self.subsample,
                colsample_bytree=self.colsample,
                objective=objective,
                eval_metric="mlogloss",
                early_stopping_rounds=self.early_stopping,
                use_label_encoder=False,
                random_state=self.random_seed,
                verbosity=0,
                n_jobs=-1,
            )
        elif self.model_type == "lightgbm":
            import lightgbm as lgb

            return lgb.LGBMClassifier(
                n_estimators=self.n_estimators,
                max_depth=self.max_depth,
                learning_rate=self.learning_rate,
                subsample=self.subsample,
                colsample_bytree=self.colsample,
                num_leaves=2**self.max_depth - 1,
                random_state=self.random_seed,
                verbose=-1,
                n_jobs=-1,
            )
        elif self.model_type == "random_forest":
            from sklearn.ensemble import RandomForestClassifier

            return RandomForestClassifier(
                n_estimators=self.n_estimators,
                max_depth=self.max_depth if self.max_depth > 0 else None,
                random_state=self.random_seed,
                n_jobs=-1,
            )
        else:
            raise ValueError(f"Unknown model_type: {self.model_type}")

    def fit(
        self,
        X: pd.DataFrame,
        y: pd.Series,
        eval_set: tuple[pd.DataFrame, pd.Series] | None = None,
    ) -> "BlockClassifier":
        """
        Fit the classifier.

        Parameters
        ----------
        X        : Feature matrix.
        y        : Label series.
        eval_set : Optional (X_val, y_val) for early stopping.
        """
        y_enc = self._label_encoder.fit_transform(y)
        self.classes_ = self._label_encoder.classes_
        n_classes = len(self.classes_)

        self._model = self._build_model(n_classes)

        if eval_set is not None and self.model_type in ("xgboost", "lightgbm"):
            X_val, y_val = eval_set
            y_val_enc = self._label_encoder.transform(y_val)
            self._model.fit(
                X.values,
                y_enc,
                eval_set=[(X_val.values, y_val_enc)],
                verbose=False,
            )
        else:
            self._model.fit(X.values, y_enc)

        logger.info(
            f"Trained {self.model_type} on {X.shape[0]} samples, "
            f"{X.shape[1]} features, {n_classes} classes"
        )
        return self

    def predict(self, X: pd.DataFrame) -> np.ndarray:
        y_enc = self._model.predict(X.values)
        return self._label_encoder.inverse_transform(y_enc)

    def predict_proba(self, X: pd.DataFrame) -> pd.DataFrame:
        """Return probability DataFrame with class labels as columns."""
        proba = self._model.predict_proba(X.values)
        if proba.ndim == 1:
            proba = np.column_stack([1 - proba, proba])
        return pd.DataFrame(proba, index=X.index, columns=self.classes_)

    def feature_importances(self, feature_names: list[str]) -> pd.Series:
        """Return feature importances as a sorted Series."""
        if hasattr(self._model, "feature_importances_"):
            imp = self._model.feature_importances_
        else:
            raise AttributeError(f"Model {self.model_type} has no feature_importances_.")
        return pd.Series(imp, index=feature_names).sort_values(ascending=False)

    def save(self, path: Path) -> None:
        with open(path, "wb") as fh:
            pickle.dump(self, fh)
        logger.info(f"Saved BlockClassifier to {path}")

    @classmethod
    def load(cls, path: Path) -> "BlockClassifier":
        with open(path, "rb") as fh:
            return pickle.load(fh)


# ── block-wise training loop ──────────────────────────────────────────────────


def train_all_blocks(
    feature_blocks: dict[str, pd.DataFrame],
    y: pd.Series,
    cv_splitter,
    model_kwargs: dict | None = None,
) -> dict[str, dict]:
    """
    Train a BlockClassifier for each feature block using the provided CV splitter.

    Parameters
    ----------
    feature_blocks : Dict block_name -> feature DataFrame.
    y              : Label series aligned with feature DataFrames.
    cv_splitter    : Scikit-learn compatible CV splitter (e.g. CladeHoldoutCV).
    model_kwargs   : Keyword arguments passed to BlockClassifier.

    Returns
    -------
    Dict block_name -> {
        "classifier": fitted BlockClassifier,
        "cv_scores": list of per-fold accuracy,
        "oof_probabilities": out-of-fold probability DataFrame,
    }
    """
    from sklearn.metrics import accuracy_score, f1_score

    model_kwargs = model_kwargs or {}
    results: dict[str, dict] = {}

    for block_name, X_block in feature_blocks.items():
        logger.info(f"\n{'=' * 50}\nBlock: {block_name}")

        # Align labels to block genome IDs
        common_ids = X_block.index.intersection(y.index)
        X = X_block.loc[common_ids]
        y_block = y.loc[common_ids]

        fold_scores = []
        oof_proba_rows = []

        for fold, (train_idx, test_idx) in enumerate(cv_splitter.split(X, y_block)):
            X_train, X_test = X.iloc[train_idx], X.iloc[test_idx]
            y_train, y_test = y_block.iloc[train_idx], y_block.iloc[test_idx]

            clf = BlockClassifier(**model_kwargs)
            clf.fit(X_train, y_train, eval_set=(X_test, y_test))

            proba_df = clf.predict_proba(X_test)
            oof_proba_rows.append(proba_df)

            y_pred = clf.predict(X_test)
            acc = accuracy_score(y_test, y_pred)
            f1 = f1_score(y_test, y_pred, average="macro", zero_division=0)
            fold_scores.append({"fold": fold, "accuracy": acc, "f1_macro": f1})
            logger.info(f"  Fold {fold}: acc={acc:.3f}, f1={f1:.3f}")

        # Fit final model on all data
        final_clf = BlockClassifier(**model_kwargs)
        final_clf.fit(X, y_block)

        oof_probabilities = pd.concat(oof_proba_rows, axis=0).sort_index()

        results[block_name] = {
            "classifier": final_clf,
            "cv_scores": pd.DataFrame(fold_scores),
            "oof_probabilities": oof_probabilities,
        }

        mean_acc = pd.DataFrame(fold_scores)["accuracy"].mean()
        logger.info(f"  Mean accuracy: {mean_acc:.3f}")

    return results
