#!/usr/bin/env python3
"""
scripts/04_predict.py

Predict taxonomy / ecological niche for new fungal genomes.

Usage:
    python scripts/04_predict.py \\
        --model-dir results/taxonomy_order/ \\
        --genome-dir data/new_genomes/ \\
        --annotation-dir data/new_annotations/ \\
        --output predictions.tsv
"""

import argparse
import logging
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from fungal_classifier.features.domains import build_domain_matrix
from fungal_classifier.features.kmer import build_kmer_matrix
from fungal_classifier.features.repeats import build_repeat_matrix
from fungal_classifier.models.block_classifier import BlockClassifier
from fungal_classifier.models.fusion_model import StackingFusionModel
from fungal_classifier.utils.io import (
    discover_annotation_files,
    discover_genome_files,
    save_predictions,
)

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--model-dir", type=Path, required=True)
    p.add_argument("--genome-dir", type=Path, required=True)
    p.add_argument("--annotation-dir", type=Path)
    p.add_argument("--output", type=Path, default=Path("predictions.tsv"))
    p.add_argument("--n-jobs", type=int, default=4)
    return p.parse_args()


def main():
    args = parse_args()

    genome_paths = discover_genome_files(args.genome_dir)
    logger.info(f"Found {len(genome_paths)} new genomes")

    feature_blocks = {}

    # Build k-mer features
    logger.info("Computing k-mer features...")
    feature_blocks["kmer"] = build_kmer_matrix(
        fasta_paths=genome_paths,
        k_values=[1, 2, 3, 4, 5, 6],
        normalize="relative_abundance",
        n_jobs=args.n_jobs,
    )

    # Build domain features if annotations available
    if args.annotation_dir:
        domain_paths = discover_annotation_files(
            args.annotation_dir / "pfam",
            suffix=".domtblout",
            genome_ids=list(genome_paths.keys()),
        )
        if domain_paths:
            feature_blocks["domains"] = build_domain_matrix(domain_paths)

        repeat_paths = discover_annotation_files(
            args.annotation_dir / "repeatmasker",
            suffix=".out",
            genome_ids=list(genome_paths.keys()),
        )
        if repeat_paths:
            feature_blocks["repeats"] = build_repeat_matrix(repeat_paths)

    # Load models
    fusion_model = StackingFusionModel.load(args.model_dir / "models" / "fusion_model.pkl")
    block_classifiers = {}
    for pkl in (args.model_dir / "models").glob("block_*.pkl"):
        block_name = pkl.stem.replace("block_", "")
        block_classifiers[block_name] = BlockClassifier.load(pkl)
    fusion_model._block_classifiers = block_classifiers

    # Predict
    predictions = fusion_model.predict(feature_blocks)
    probabilities = fusion_model.predict_proba(feature_blocks)

    save_predictions(predictions, probabilities, args.output)
    logger.info(f"Predictions saved to {args.output}")

    # Print summary
    print(f"\nPredictions for {len(predictions)} genomes:")
    print(predictions.value_counts().to_string())


if __name__ == "__main__":
    main()
