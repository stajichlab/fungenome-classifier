#!/usr/bin/env python3
"""
scripts/06_export_embeddings.py

Compute and export 2D embeddings (PCA, UMAP, t-SNE) for all feature blocks.
Optionally extracts learned tower embeddings from a trained deep fusion model.

Outputs per block per method:
  - {block}_{method}_coords.tsv   ← genome_id + 2D coordinates (for iTOL, R, etc.)
  - {block}_{method}_grid.svg     ← faceted scatter plot coloured by metadata columns

Usage:
    python scripts/06_export_embeddings.py \\
        --model-dir results/taxonomy_order \\
        --features-dir data/features/ \\
        --metadata data/raw/metadata.tsv \\
        --output-dir results/embeddings \\
        --methods pca umap \\
        --color-by taxonomy_order ecological_niche lifestyle
"""

import argparse
import logging
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from fungal_classifier.evaluation.embeddings import run_embedding_export
from fungal_classifier.utils.io import load_feature_blocks, load_metadata

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)


def parse_args():
    p = argparse.ArgumentParser(description="Export genome embeddings")
    p.add_argument("--features-dir", type=Path, default=Path("data/features"))
    p.add_argument("--metadata", type=Path, required=True)
    p.add_argument("--output-dir", type=Path, default=Path("results/embeddings"))
    p.add_argument("--model-dir", type=Path, help="Path to results dir (for deep model)")
    p.add_argument(
        "--methods",
        nargs="+",
        default=["pca", "umap"],
        choices=["pca", "umap", "tsne"],
        help="Embedding methods to compute",
    )
    p.add_argument(
        "--blocks", nargs="+", default=None, help="Subset of blocks to embed (default: all)"
    )
    p.add_argument(
        "--color-by",
        nargs="+",
        default=["taxonomy_order", "taxonomy_class", "ecological_niche", "lifestyle"],
        help="Metadata columns for colouring scatter plots",
    )
    p.add_argument(
        "--deep", action="store_true", help="Also extract tower embeddings from trained deep model"
    )
    return p.parse_args()


def main():
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)

    metadata = load_metadata(args.metadata)
    blocks = load_feature_blocks(args.features_dir, block_names=args.blocks)

    if not blocks:
        logger.error(f"No feature blocks found in {args.features_dir}")
        sys.exit(1)

    logger.info(f"Loaded {len(blocks)} feature blocks: {list(blocks.keys())}")

    # Optionally load deep model
    deep_model = None
    deep_blocks = None
    if args.deep and args.model_dir:
        deep_pt = args.model_dir / "models" / "deep_fusion_model.pt"
        if deep_pt.exists():
            try:
                import torch
                from sklearn.decomposition import TruncatedSVD
                from sklearn.preprocessing import StandardScaler

                from fungal_classifier.models.deep_fusion import DeepFusionClassifier

                # Rebuild reduced blocks for deep model input
                svd_k = 150
                deep_blocks = {}
                block_dims = {}
                for name, df in blocks.items():
                    arr = StandardScaler().fit_transform(df.fillna(0).values)
                    k = min(svd_k, arr.shape[1] - 1)
                    arr_r = TruncatedSVD(n_components=k, random_state=42).fit_transform(arr)
                    deep_blocks[name] = __import__("pandas").DataFrame(
                        arr_r, index=df.index, columns=[f"svd_{i}" for i in range(k)]
                    )
                    block_dims[name] = k

                checkpoint = torch.load(deep_pt, map_location="cpu")
                # We need n_classes from checkpoint — infer from label encoder
                le = checkpoint.get("label_encoder")
                n_classes = len(le.classes_) if le else 10
                deep_model = DeepFusionClassifier(
                    block_dims=block_dims,
                    n_classes=n_classes,
                    hidden_dim=256,
                    embedding_dim=128,
                    fusion="attention",
                )
                deep_model.load_state_dict(checkpoint["model_state"])
                deep_model.eval()
                logger.info("Loaded deep fusion model for tower embedding extraction")
            except Exception as e:
                logger.warning(f"Could not load deep model: {e}")
        else:
            logger.warning(f"Deep model checkpoint not found at {deep_pt}")

    run_embedding_export(
        feature_blocks=blocks,
        metadata=metadata,
        output_dir=args.output_dir,
        methods=args.methods,
        color_cols=args.color_by,
        deep_model=deep_model,
        deep_feature_blocks=deep_blocks,
    )

    logger.info(f"\nEmbedding export complete. Files saved to: {args.output_dir}")
    logger.info(
        "Coordinate TSVs can be loaded directly into R (ggplot2) or used for iTOL tree annotation."
    )


if __name__ == "__main__":
    main()
