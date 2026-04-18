#!/usr/bin/env python3
"""
scripts/03_evaluate.py

Detailed evaluation of trained models including confusion matrices,
per-class metrics, and phylogenetic signal analysis.

Usage:
    python scripts/03_evaluate.py \\
        --model-dir results/taxonomy_order/ \\
        --features-dir data/features/ \\
        --metadata data/raw/metadata.tsv \\
        --tree data/raw/phylogeny.nwk \\
        --target taxonomy_order
"""

import argparse
import logging
import sys
from pathlib import Path

import pandas as pd
import matplotlib
matplotlib.use("Agg")

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from fungal_classifier.models.fusion_model import StackingFusionModel
from fungal_classifier.models.block_classifier import BlockClassifier
from fungal_classifier.evaluation.metrics import (
    plot_confusion_matrix, plot_cv_scores, print_evaluation_report,
    cv_summary, block_comparison_table
)
from fungal_classifier.evaluation.phylo_cv import (
    load_tree, get_patristic_distances, blombergs_k
)
from fungal_classifier.utils.io import load_metadata, load_feature_blocks

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--model-dir", type=Path, required=True)
    p.add_argument("--features-dir", type=Path, default=Path("data/features"))
    p.add_argument("--metadata", type=Path, required=True)
    p.add_argument("--target", required=True)
    p.add_argument("--tree", type=Path)
    p.add_argument("--output-dir", type=Path)
    return p.parse_args()


def main():
    args = parse_args()
    out_dir = args.output_dir or args.model_dir / "evaluation"
    out_dir.mkdir(parents=True, exist_ok=True)

    metadata = load_metadata(args.metadata)
    y = metadata[args.target].dropna()

    feature_blocks = load_feature_blocks(args.features_dir)
    common_ids = sorted(set(y.index) & set(next(iter(feature_blocks.values())).index))
    y = y.loc[common_ids]
    feature_blocks = {name: df.loc[common_ids] for name, df in feature_blocks.items()}

    # Load fusion model
    fusion_model = StackingFusionModel.load(args.model_dir / "models" / "fusion_model.pkl")

    # Load block classifiers
    block_classifiers = {}
    for pkl in (args.model_dir / "models").glob("block_*.pkl"):
        block_name = pkl.stem.replace("block_", "")
        block_classifiers[block_name] = BlockClassifier.load(pkl)

    # Predictions
    y_pred = fusion_model.predict(feature_blocks)
    proba = fusion_model.predict_proba(feature_blocks)

    print_evaluation_report(y, y_pred, model_name=f"Fusion ({args.target})")

    # Confusion matrix
    labels = sorted(y.unique())
    plot_confusion_matrix(
        y, y_pred, labels=labels, normalize=True,
        title=f"Confusion Matrix — {args.target}",
        save_path=out_dir / "confusion_matrix.svg",
    )

    # Phylogenetic signal test
    if args.tree:
        logger.info("Computing phylogenetic signal (Blomberg's K) for predictions...")
        tree = load_tree(str(args.tree))
        D = get_patristic_distances(tree, common_ids)
        # Encode labels as integers for K calculation
        from sklearn.preprocessing import LabelEncoder
        le = LabelEncoder()
        y_enc = pd.Series(le.fit_transform(y), index=y.index)
        K = blombergs_k(y_enc, D)
        logger.info(f"Blomberg's K for '{args.target}': {K:.4f}")
        with open(out_dir / "phylo_signal.txt", "w") as fh:
            fh.write(f"Target: {args.target}\n")
            fh.write(f"Blomberg's K: {K:.4f}\n")
            fh.write("(K>>1: strong phylogenetic signal; K≈0: no signal)\n")

    logger.info(f"Evaluation outputs saved to {out_dir}")


if __name__ == "__main__":
    main()
