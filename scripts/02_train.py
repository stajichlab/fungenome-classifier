#!/usr/bin/env python3
"""
scripts/02_train.py

Train block classifiers and fusion model.

Usage:
    # Gradient boosting with clade-holdout CV
    python scripts/02_train.py \\
        --features-dir data/features/ \\
        --metadata data/raw/metadata.tsv \\
        --tree data/raw/phylogeny.nwk \\
        --target taxonomy_order \\
        --config configs/default.yaml \\
        --output-dir results/taxonomy_order/

    # Deep fusion model
    python scripts/02_train.py \\
        --config configs/deep_fusion.yaml \\
        --target ecological_niche \\
        --model-type deep
"""

import argparse
import json
import logging
import sys
from pathlib import Path

import pandas as pd
import yaml

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from fungal_classifier.features.fusion import BlockFusionPipeline
from fungal_classifier.models.block_classifier import BlockClassifier, train_all_blocks
from fungal_classifier.models.fusion_model import StackingFusionModel
from fungal_classifier.evaluation.phylo_cv import (
    CladeHoldoutCV,
    assign_clades_from_taxonomy,
    assign_clades_from_tree,
    phylogenetic_eigenvectors,
    get_patristic_distances,
    load_tree,
)
from fungal_classifier.evaluation.metrics import (
    cv_summary, block_comparison_table, print_evaluation_report
)
from fungal_classifier.evaluation.shap_analysis import run_shap_analysis
from fungal_classifier.utils.io import load_metadata, load_feature_blocks, save_predictions

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
)
logger = logging.getLogger(__name__)


def parse_args():
    p = argparse.ArgumentParser(description="Train FungalClassifier models")
    p.add_argument("--features-dir", type=Path, default=Path("data/features"))
    p.add_argument("--metadata", type=Path, required=True)
    p.add_argument("--tree", type=Path, help="Newick phylogenetic tree file")
    p.add_argument("--target", required=True,
                   help="Metadata column to use as classification target")
    p.add_argument("--config", type=Path, default=Path("configs/default.yaml"))
    p.add_argument("--output-dir", type=Path, default=Path("results"))
    p.add_argument("--model-type", choices=["xgboost", "lightgbm", "deep"], default="xgboost")
    p.add_argument("--cv-strategy", choices=["clade_holdout", "random"], default="clade_holdout")
    p.add_argument("--no-shap", action="store_true", help="Skip SHAP analysis")
    p.add_argument("--phylo-eigenvectors", action="store_true",
                   help="Include phylogenetic eigenvectors as features")
    return p.parse_args()


def main():
    args = parse_args()

    with open(args.config) as fh:
        config = yaml.safe_load(fh)

    args.output_dir.mkdir(parents=True, exist_ok=True)

    # ── Load data ─────────────────────────────────────────────────────────────
    metadata = load_metadata(args.metadata)
    if args.target not in metadata.columns:
        raise ValueError(f"Target column '{args.target}' not found in metadata. "
                         f"Available: {list(metadata.columns)}")

    y = metadata[args.target].dropna()
    logger.info(f"Target: {args.target} | Classes: {y.nunique()} | Samples: {len(y)}")

    feature_blocks = load_feature_blocks(args.features_dir)
    if not feature_blocks:
        raise RuntimeError(f"No feature blocks found in {args.features_dir}")

    # Align all blocks and labels to common genome IDs
    common_ids = set(y.index)
    for block_df in feature_blocks.values():
        common_ids &= set(block_df.index)
    common_ids = sorted(common_ids)
    logger.info(f"Common genome IDs across all blocks: {len(common_ids)}")

    y = y.loc[common_ids]
    feature_blocks = {name: df.loc[common_ids] for name, df in feature_blocks.items()}

    # ── Optional: phylogenetic eigenvectors as features ───────────────────────
    if args.phylo_eigenvectors and args.tree:
        logger.info("Computing phylogenetic eigenvectors...")
        tree = load_tree(str(args.tree))
        D = get_patristic_distances(tree, common_ids)
        phylo_eig = phylogenetic_eigenvectors(D, n_components=20)
        feature_blocks["phylo_eigenvectors"] = phylo_eig

    # ── Build CV splitter ─────────────────────────────────────────────────────
    cv_cfg = config["cross_validation"]

    if args.cv_strategy == "clade_holdout":
        if args.tree:
            logger.info("Assigning clades from phylogenetic tree...")
            tree = load_tree(str(args.tree))
            clade_labels = assign_clades_from_tree(tree, common_ids, n_clades=cv_cfg["n_folds"] * 2)
        else:
            logger.info(f"Assigning clades from taxonomy level: {cv_cfg['clade_level']}")
            clade_labels = assign_clades_from_taxonomy(
                metadata.loc[common_ids], clade_level=cv_cfg["clade_level"]
            )
        cv = CladeHoldoutCV(
            clade_labels=clade_labels,
            n_folds=cv_cfg["n_folds"],
            random_seed=cv_cfg["random_seed"],
        )
        fold_summary = cv.fold_summary()
        fold_summary.to_csv(args.output_dir / "cv_fold_summary.csv", index=False)
        logger.info(f"\nCV fold summary:\n{fold_summary.to_string()}")
    else:
        from sklearn.model_selection import StratifiedKFold
        cv = StratifiedKFold(
            n_splits=cv_cfg["n_folds"],
            shuffle=True,
            random_state=cv_cfg["random_seed"],
        )

    # ── Feature fusion ────────────────────────────────────────────────────────
    prep_cfg = config["preprocessing"]
    fs_cfg = config["feature_selection"]

    if args.model_type != "deep":
        fusion_pipeline = BlockFusionPipeline(
            scaler=prep_cfg["scaler"],
            variance_threshold=fs_cfg["variance_threshold"],
            univariate_k=fs_cfg.get("univariate_k"),
            univariate_scoring=fs_cfg["univariate_scoring"],
            svd_components=prep_cfg.get("svd_components"),
        )
        X_fused = fusion_pipeline.fit_transform(feature_blocks, y)
        logger.info(f"Fused feature matrix: {X_fused.shape}")

    # ── Train block classifiers ───────────────────────────────────────────────
    model_cfg = config["models"]["block_classifier"]
    block_model_kwargs = {
        "model_type": args.model_type if args.model_type != "deep" else "xgboost",
        **config["models"]["block_classifier"].get(args.model_type, {}),
    }

    logger.info("\nTraining per-block classifiers with clade-holdout CV...")
    block_results = train_all_blocks(
        feature_blocks=feature_blocks,
        y=y,
        cv_splitter=cv,
        model_kwargs=block_model_kwargs,
    )

    # ── Save block CV scores ──────────────────────────────────────────────────
    block_cv_scores = {name: res["cv_scores"] for name, res in block_results.items()}
    comparison = block_comparison_table(block_cv_scores, metric="f1_macro")
    comparison.to_csv(args.output_dir / "block_comparison.csv", index=False)
    logger.info(f"\nBlock performance comparison:\n{comparison.to_string()}")

    # ── Train stacking fusion model ───────────────────────────────────────────
    logger.info("\nTraining stacking fusion model...")
    block_classifiers = {name: res["classifier"] for name, res in block_results.items()}
    fusion_model = StackingFusionModel(
        meta_learner=config["models"]["fusion"]["meta_learner"],
        random_seed=cv_cfg["random_seed"],
    )
    fusion_model.fit_from_block_results(
        block_results=block_results,
        y=y,
        block_classifiers=block_classifiers,
    )

    # ── Save models ───────────────────────────────────────────────────────────
    models_dir = args.output_dir / "models"
    models_dir.mkdir(exist_ok=True)

    for block_name, clf in block_classifiers.items():
        clf.save(models_dir / f"block_{block_name}.pkl")

    fusion_model.save(models_dir / "fusion_model.pkl")
    logger.info(f"Models saved to {models_dir}")

    # ── SHAP analysis ─────────────────────────────────────────────────────────
    if not args.no_shap:
        logger.info("\nRunning SHAP analysis...")
        run_shap_analysis(
            block_classifiers=block_classifiers,
            feature_blocks=feature_blocks,
            class_names=list(fusion_model.classes_),
            output_dir=args.output_dir / "shap",
            top_n=20,
        )

    # ── Summary report ────────────────────────────────────────────────────────
    summary = {
        "target": args.target,
        "n_genomes": len(common_ids),
        "n_classes": y.nunique(),
        "cv_strategy": args.cv_strategy,
        "n_folds": cv_cfg["n_folds"],
        "blocks": list(feature_blocks.keys()),
        "block_cv_summary": {
            name: cv_summary(res["cv_scores"].to_dict("records")).to_dict()
            for name, res in block_results.items()
        },
    }
    with open(args.output_dir / "training_summary.json", "w") as fh:
        json.dump(summary, fh, indent=2, default=str)

    logger.info(f"\nTraining complete. Results saved to {args.output_dir}")


if __name__ == "__main__":
    main()
