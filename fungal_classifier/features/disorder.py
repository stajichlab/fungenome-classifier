"""
fungal_classifier/features/disorder.py

Intrinsic disorder feature extraction from AIUPred output.

AIUPred produces per-residue disorder scores (0–1) for each protein.
Genome-level features summarise the disorder landscape across all proteins.

Reference:
  Erdos & Dosztanyi (2024) Nucleic Acids Res. 52(W1):W176-W181.
"""

from __future__ import annotations

import gzip
import logging
from pathlib import Path

import numpy as np
import pandas as pd
from tqdm import tqdm

logger = logging.getLogger(__name__)

DISORDER_THRESHOLD = 0.5  # residues above this are considered disordered
IDR_MIN_LENGTH = 10  # consecutive disordered residues to call a long IDR


# ── parser ────────────────────────────────────────────────────────────────────


def parse_aiupred(path: Path) -> pd.DataFrame:
    """
    Parse AIUPred output file.

    Format:
        # ... header comment lines ...
        #>protein_id [extra tokens]
        position<TAB>residue<TAB>disorder_score
        ...

    Returns tidy DataFrame with columns: protein_id, position, residue, disorder.
    """
    records = []
    current_protein: str | None = None
    opener = gzip.open if Path(path).suffix == ".gz" else open
    with opener(path, "rt") as fh:
        for line in fh:
            line = line.rstrip("\n")
            if line.startswith("#>"):
                current_protein = line[2:].split()[0]
                continue
            if line.startswith("#") or not line.strip():
                continue
            parts = line.split("\t")
            if len(parts) < 3 or current_protein is None:
                continue
            try:
                records.append(
                    {
                        "protein_id": current_protein,
                        "position": int(parts[0]),
                        "residue": parts[1],
                        "disorder": float(parts[2]),
                    }
                )
            except (IndexError, ValueError):
                continue
    return pd.DataFrame(records)


# ── feature aggregation ───────────────────────────────────────────────────────


def _has_long_idr(scores: pd.Series, threshold: float, min_len: int) -> bool:
    """Return True if the protein has a run of >= min_len disordered residues."""
    run = 0
    for s in scores:
        if s >= threshold:
            run += 1
            if run >= min_len:
                return True
        else:
            run = 0
    return False


def aiupred_to_features(
    df: pd.DataFrame,
    threshold: float = DISORDER_THRESHOLD,
    idr_min_length: int = IDR_MIN_LENGTH,
) -> pd.Series:
    """
    Aggregate AIUPred per-residue data to genome-level features.

    Features
    --------
    aiupred_mean_disorder      : mean disorder score across all residues
    aiupred_median_disorder    : median disorder score
    aiupred_frac_disordered    : fraction of residues above threshold
    aiupred_frac_proteins_idr  : fraction of proteins with any disordered residue
    aiupred_frac_proteins_long_idr : fraction with a long IDR (>= idr_min_length)
    aiupred_mean_protein_disorder  : mean per-protein mean disorder score
    """
    if df.empty:
        return pd.Series(dtype=np.float32)

    all_scores = df["disorder"]
    features: dict[str, float] = {
        "aiupred_mean_disorder": float(all_scores.mean()),
        "aiupred_median_disorder": float(all_scores.median()),
        "aiupred_frac_disordered": float((all_scores >= threshold).mean()),
    }

    protein_groups = df.groupby("protein_id")["disorder"]
    n_proteins = protein_groups.ngroups

    has_any_idr = protein_groups.apply(lambda s: (s >= threshold).any()).sum()
    has_long_idr = protein_groups.apply(lambda s: _has_long_idr(s, threshold, idr_min_length)).sum()
    per_protein_mean = protein_groups.mean()

    features["aiupred_frac_proteins_idr"] = float(has_any_idr) / max(n_proteins, 1)
    features["aiupred_frac_proteins_long_idr"] = float(has_long_idr) / max(n_proteins, 1)
    features["aiupred_mean_protein_disorder"] = float(per_protein_mean.mean())

    return pd.Series(features, dtype=np.float32)


def build_disorder_matrix(
    annotation_paths: dict[str, Path],
    threshold: float = DISORDER_THRESHOLD,
    idr_min_length: int = IDR_MIN_LENGTH,
) -> pd.DataFrame:
    """
    Build genome × disorder feature matrix from AIUPred output files.

    Parameters
    ----------
    annotation_paths : Dict genome_id -> path to .aiupred.txt[.gz] file.
    threshold        : Disorder score cutoff for 'disordered' residue.
    idr_min_length   : Minimum consecutive disordered residues for a long IDR.
    """
    rows: dict[str, pd.Series] = {}
    for genome_id, path in tqdm(annotation_paths.items(), desc="Disorder features"):
        try:
            df = parse_aiupred(path)
            rows[genome_id] = aiupred_to_features(df, threshold, idr_min_length)
        except Exception as e:
            logger.warning(f"Failed AIUPred parse for {genome_id}: {e}")
    matrix = pd.DataFrame(rows).T.fillna(0.0).astype(np.float32)
    matrix.index.name = "genome_id"
    logger.info(f"Disorder matrix: {matrix.shape}")
    return matrix
