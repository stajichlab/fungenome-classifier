"""
fungal_classifier/utils/io.py

Data loading, saving, and path management utilities.
"""

from __future__ import annotations

import logging
from pathlib import Path

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)


# ── metadata ──────────────────────────────────────────────────────────────────

def load_metadata(path: Path) -> pd.DataFrame:
    """
    Load genome metadata TSV.

    Expected columns:
      genome_id, taxonomy_phylum, taxonomy_class, taxonomy_order,
      taxonomy_family, taxonomy_genus, ecological_niche, lifestyle, tree_label
    """
    df = pd.read_csv(path, sep="\t", index_col="genome_id", low_memory=False)
    logger.info(f"Loaded metadata: {df.shape[0]} genomes, {df.shape[1]} columns")
    return df


# ── feature matrices ──────────────────────────────────────────────────────────

def save_feature_matrix(df: pd.DataFrame, path: Path, format: str = "parquet") -> None:
    """Save a feature matrix to Parquet or HDF5."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    if format == "parquet":
        df.to_parquet(path, compression="snappy")
    elif format == "hdf5":
        df.to_hdf(path, key="features", mode="w", complevel=4)
    else:
        raise ValueError(f"Unknown format: {format}")
    logger.info(f"Saved {df.shape} feature matrix to {path}")


def load_feature_matrix(path: Path) -> pd.DataFrame:
    """Load a feature matrix from Parquet or HDF5."""
    path = Path(path)
    if path.suffix in (".parquet",):
        return pd.read_parquet(path)
    elif path.suffix in (".h5", ".hdf5"):
        return pd.read_hdf(path, key="features")
    else:
        raise ValueError(f"Unrecognized file extension: {path.suffix}")


def load_feature_blocks(feature_dir: Path, block_names: list[str] | None = None) -> dict[str, pd.DataFrame]:
    """
    Load all feature block matrices from a directory.

    Expects files named: {block_name}.parquet or {block_name}.h5
    """
    feature_dir = Path(feature_dir)
    blocks: dict[str, pd.DataFrame] = {}
    suffixes = {".parquet", ".h5", ".hdf5"}

    for path in sorted(feature_dir.iterdir()):
        if path.suffix not in suffixes:
            continue
        block_name = path.stem
        if block_names is not None and block_name not in block_names:
            continue
        try:
            blocks[block_name] = load_feature_matrix(path)
            logger.info(f"Loaded block '{block_name}': {blocks[block_name].shape}")
        except Exception as e:
            logger.warning(f"Failed to load {path}: {e}")

    return blocks


# ── path discovery helpers ────────────────────────────────────────────────────

def discover_genome_files(
    genome_dir: Path,
    extensions: tuple[str, ...] = (".fna", ".fa", ".fasta", ".fna.gz"),
) -> dict[str, Path]:
    """
    Discover genome FASTA files in a directory.

    Returns dict: genome_id (stem without extension) -> Path.
    """
    genome_dir = Path(genome_dir)
    paths: dict[str, Path] = {}
    for path in sorted(genome_dir.iterdir()):
        name = path.name
        for ext in extensions:
            if name.endswith(ext):
                genome_id = name[: -len(ext)]
                paths[genome_id] = path
                break
    logger.info(f"Discovered {len(paths)} genome files in {genome_dir}")
    return paths


def discover_annotation_files(
    annotation_dir: Path,
    suffix: str,
    genome_ids: list[str] | None = None,
) -> dict[str, Path]:
    """
    Discover annotation files matching a suffix pattern.

    Returns dict: genome_id -> Path.
    """
    annotation_dir = Path(annotation_dir)
    paths: dict[str, Path] = {}
    for path in sorted(annotation_dir.rglob(f"*{suffix}")):
        genome_id = path.stem.replace(suffix.lstrip("."), "").rstrip(".")
        if genome_ids is None or genome_id in genome_ids:
            paths[genome_id] = path
    logger.info(f"Discovered {len(paths)} annotation files (*{suffix}) in {annotation_dir}")
    return paths


# ── results persistence ───────────────────────────────────────────────────────

def save_predictions(predictions: pd.Series, probabilities: pd.DataFrame, output_path: Path) -> None:
    """Save prediction labels and probabilities to TSV."""
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    df = probabilities.copy()
    df.insert(0, "predicted_label", predictions)
    df.to_csv(output_path, sep="\t")
    logger.info(f"Saved predictions for {len(df)} genomes to {output_path}")
