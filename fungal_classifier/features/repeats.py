"""
fungal_classifier/features/repeats.py

Repeat content feature extraction from RepeatMasker output.

Computes:
  - TE class proportions (normalized by genome size or total repeat bp)
  - TE family-level proportions
  - Simple repeat and low-complexity region density
  - Repeat landscape summaries (Kimura distance distributions)
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Literal

import numpy as np
import pandas as pd
from tqdm import tqdm

logger = logging.getLogger(__name__)

# Standard RepeatMasker class hierarchy (top-level)
REPEAT_CLASSES = [
    "LTR", "DNA", "LINE", "SINE",
    "RC",                          # Rolling circle
    "Satellite",
    "Simple_repeat",
    "Low_complexity",
    "rRNA", "snRNA", "srpRNA",
    "Unknown",
    "Other",
]


# ── parsers ───────────────────────────────────────────────────────────────────

def parse_rmout(path: Path) -> pd.DataFrame:
    """
    Parse RepeatMasker .out file into a tidy DataFrame.

    RepeatMasker .out format (whitespace-delimited):
      SW_score, perc_div, perc_del, perc_ins, query_seq, q_begin, q_end,
      q_left, strand, repeat_name, repeat_class/family, r_begin, r_end,
      r_left, ID, [*]
    """
    records = []
    with open(path) as fh:
        for _ in range(3):  # skip 3 header lines
            next(fh)
        for line in fh:
            line = line.strip()
            if not line:
                continue
            parts = line.split()
            try:
                query_seq = parts[4]
                q_begin = int(parts[5])
                q_end = int(parts[6])
                strand = parts[8]
                repeat_name = parts[9]
                class_family = parts[10]
                perc_div = float(parts[1])
                length_bp = q_end - q_begin + 1

                if "/" in class_family:
                    repeat_class, repeat_family = class_family.split("/", 1)
                else:
                    repeat_class, repeat_family = class_family, class_family

                records.append(
                    {
                        "query_seq": query_seq,
                        "start": q_begin,
                        "end": q_end,
                        "length_bp": length_bp,
                        "strand": strand,
                        "repeat_name": repeat_name,
                        "repeat_class": repeat_class,
                        "repeat_family": repeat_family,
                        "perc_div": perc_div,
                    }
                )
            except (IndexError, ValueError):
                continue

    return pd.DataFrame(records)


def get_genome_size_from_fai(fai_path: Path) -> int:
    """Read total genome size (bp) from a FASTA .fai index file."""
    total = 0
    with open(fai_path) as fh:
        for line in fh:
            parts = line.strip().split("\t")
            if len(parts) >= 2:
                total += int(parts[1])
    return total


# ── feature building ──────────────────────────────────────────────────────────

def compute_repeat_features(
    rmout_path: Path,
    genome_size_bp: int | None = None,
    normalize_by: Literal["genome_size", "total_repeat_bp", "none"] = "genome_size",
    include_families: bool = False,
    classes_to_include: list[str] = REPEAT_CLASSES,
) -> pd.Series:
    """
    Compute repeat content feature vector for one genome.

    Parameters
    ----------
    rmout_path       : Path to RepeatMasker .out file.
    genome_size_bp   : Total genome size in bp (required if normalize_by='genome_size').
    normalize_by     : Normalization strategy.
    include_families : If True, also include family-level features.
    classes_to_include: List of TE classes to include.

    Returns
    -------
    pd.Series of repeat features.
    """
    df = parse_rmout(rmout_path)
    features: dict[str, float] = {}

    # Aggregate by class
    class_bp = df.groupby("repeat_class")["length_bp"].sum()
    total_repeat_bp = df["length_bp"].sum()

    for cls in classes_to_include:
        bp = float(class_bp.get(cls, 0))
        if normalize_by == "genome_size":
            denom = genome_size_bp or 1
        elif normalize_by == "total_repeat_bp":
            denom = total_repeat_bp or 1
        else:
            denom = 1.0
        features[f"repeat_class_{cls}"] = bp / denom

    # Total repeat fraction
    features["repeat_total_fraction"] = total_repeat_bp / (genome_size_bp or total_repeat_bp or 1)

    # Mean divergence per class (repeat landscape proxy)
    for cls in classes_to_include:
        sub = df[df["repeat_class"] == cls]
        features[f"repeat_meandiv_{cls}"] = float(sub["perc_div"].mean()) if len(sub) > 0 else 0.0

    # Family-level features (optional — expands feature space significantly)
    if include_families:
        family_bp = df.groupby("repeat_family")["length_bp"].sum()
        denom = genome_size_bp or total_repeat_bp or 1
        for fam, bp in family_bp.items():
            features[f"repeat_family_{fam}"] = float(bp) / denom

    return pd.Series(features, dtype=np.float32)


def build_repeat_matrix(
    rmout_paths: dict[str, Path],
    genome_sizes: dict[str, int] | None = None,
    normalize_by: Literal["genome_size", "total_repeat_bp", "none"] = "genome_size",
    include_families: bool = False,
    classes_to_include: list[str] = REPEAT_CLASSES,
) -> pd.DataFrame:
    """
    Build genome × repeat-feature matrix.

    Parameters
    ----------
    rmout_paths     : Dict genome_id -> path to .out file.
    genome_sizes    : Dict genome_id -> genome size in bp.
    normalize_by    : Normalization strategy.
    include_families: Include TE family-level features.

    Returns
    -------
    pd.DataFrame of shape (n_genomes, n_repeat_features).
    """
    rows: dict[str, pd.Series] = {}
    for genome_id, path in tqdm(rmout_paths.items(), desc="Repeat features"):
        try:
            gsize = (genome_sizes or {}).get(genome_id)
            vec = compute_repeat_features(
                path,
                genome_size_bp=gsize,
                normalize_by=normalize_by,
                include_families=include_families,
                classes_to_include=classes_to_include,
            )
            rows[genome_id] = vec
        except Exception as e:
            logger.warning(f"Failed to process repeat file for {genome_id}: {e}")

    matrix = pd.DataFrame(rows).T.fillna(0.0).astype(np.float32)
    matrix.index.name = "genome_id"
    logger.info(f"Repeat matrix: {matrix.shape}")
    return matrix
