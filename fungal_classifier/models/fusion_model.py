"""
fungal_classifier/models/fusion_model.py

Late fusion model that stacks predictions from individual block classifiers.

Strategy:
  1. Each block produces out-of-fold (OOF) class probabilities.
  2. OOF probabilities are concatenated as meta-features.
  3. A meta-learner (logistic regression or XGBoost) is trained on meta-features.

This is a standard stacking ensemble. It avoids leakage by using OOF predictions
produced during cross-validation, not predictions from models trained on the
full training set.
"""

from __future__ import annotations

import logging
import pickle
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, f1_score
from sklearn.preprocessing import LabelEncoder

logger = logging.getLogger(__name__)


class StackingFusionModel:
    """
    Meta-learner that combines block-level probability outputs.

    Parameters
    ----------
    meta_learner : 'logistic_regression' or 'xgboost'.
    use_oof      : If True, trains meta-learner on OOF block probabilities
                   (recommended to prevent leakage). If False, uses training
                   set block probabilities (faster but biased).
    """

    def __init__(
        self,
        meta_learner: str = "logistic_regression",
        use_oof: bool = True,
        random_seed: int = 42,
    ):
        self.meta_learner = meta_learner
        self.use_oof = use_oof
        self.random_seed = random_seed
        self._meta_model = None
        self._block_classifiers: dict = {}
        self._label_encoder = LabelEncoder()
        self.classes_: np.ndarray | None = None

    def _build_meta_learner(self):
        if self.meta_learner == "logistic_regression":
            return LogisticRegression(
                max_iter=1000,
                random_state=self.random_seed,
                C=1.0,
                solver="lbfgs",
            )
        elif self.meta_learner == "xgboost":
            import xgboost as xgb

            return xgb.XGBClassifier(
                n_estimators=200,
                max_depth=3,
                learning_rate=0.1,
                use_label_encoder=False,
                eval_metric="mlogloss",
                random_state=self.random_seed,
                verbosity=0,
            )
        else:
            raise ValueError(f"Unknown meta_learner: {self.meta_learner}")

    def fit_from_block_results(
        self,
        block_results: dict[str, dict],
        y: pd.Series,
        block_classifiers: dict | None = None,
    ) -> "StackingFusionModel":
        """
        Fit the meta-learner using OOF probabilities from block training.

        Parameters
        ----------
        block_results      : Output of train_all_blocks().
        y                  : Ground-truth labels.
        block_classifiers  : Optional dict of fitted classifiers for inference.
        """
        oof_frames = []
        for block_name, result in block_results.items():
            oof_proba = result["oof_probabilities"]
            oof_proba = oof_proba.add_prefix(f"{block_name}__")
            oof_frames.append(oof_proba)

        meta_X = pd.concat(oof_frames, axis=1).fillna(0.0)
        common_ids = meta_X.index.intersection(y.index)
        meta_X = meta_X.loc[common_ids]
        y_aligned = y.loc[common_ids]

        y_enc = self._label_encoder.fit_transform(y_aligned)
        self.classes_ = self._label_encoder.classes_
        self._meta_model = self._build_meta_learner()
        self._meta_model.fit(meta_X.values, y_enc)

        if block_classifiers:
            self._block_classifiers = block_classifiers

        logger.info(
            f"Stacking meta-learner trained on {meta_X.shape[0]} samples, "
            f"{meta_X.shape[1]} meta-features"
        )
        return self

    def predict(
        self,
        feature_blocks: dict[str, pd.DataFrame],
    ) -> pd.Series:
        """
        Predict labels for new genomes.

        Parameters
        ----------
        feature_blocks : Dict block_name -> feature DataFrame.

        Returns
        -------
        pd.Series of predicted labels indexed by genome_id.
        """
        meta_X = self._build_meta_features(feature_blocks)
        y_enc = self._meta_model.predict(meta_X.values)
        labels = self._label_encoder.inverse_transform(y_enc)
        return pd.Series(labels, index=meta_X.index)

    def predict_proba(
        self,
        feature_blocks: dict[str, pd.DataFrame],
    ) -> pd.DataFrame:
        """Return class probability DataFrame."""
        meta_X = self._build_meta_features(feature_blocks)
        proba = self._meta_model.predict_proba(meta_X.values)
        return pd.DataFrame(proba, index=meta_X.index, columns=self.classes_)

    def _build_meta_features(
        self,
        feature_blocks: dict[str, pd.DataFrame],
    ) -> pd.DataFrame:
        """Get block-level probabilities for new data."""
        frames = []
        for block_name, clf in self._block_classifiers.items():
            if block_name not in feature_blocks:
                logger.warning(f"Block {block_name} missing from input — filling with zeros.")
                continue
            proba = clf.predict_proba(feature_blocks[block_name])
            proba = proba.add_prefix(f"{block_name}__")
            frames.append(proba)
        return pd.concat(frames, axis=1).fillna(0.0)

    def evaluate(
        self,
        feature_blocks: dict[str, pd.DataFrame],
        y_true: pd.Series,
    ) -> dict[str, float]:
        """Compute accuracy and macro-F1 on a held-out set."""
        y_pred = self.predict(feature_blocks)
        common = y_true.index.intersection(y_pred.index)
        acc = accuracy_score(y_true.loc[common], y_pred.loc[common])
        f1 = f1_score(
            y_true.loc[common],
            y_pred.loc[common],
            average="macro",
            zero_division=0,
        )
        return {"accuracy": acc, "f1_macro": f1}

    def save(self, path: Path) -> None:
        with open(path, "wb") as fh:
            pickle.dump(self, fh)

    @classmethod
    def load(cls, path: Path) -> "StackingFusionModel":
        with open(path, "rb") as fh:
            return pickle.load(fh)
