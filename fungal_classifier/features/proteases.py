"""
fungal_classifier/features/proteases.py

Protease repertoire feature extraction from MEROPS BLAST results.

Expected input: BLAST tabular output (-outfmt 6) searched against the
MEROPS peptidase database.  Subject IDs are MEROPS identifiers of the form
<FamilyID>.<subfamily> (e.g. "S01.001", "A01.009").

Features capture:
  - Count / presence per MEROPS family (e.g. S01, A01, C19 …)
  - Count / presence per MEROPS clan
  - Total secreted protease proxy (families that act extracellularly)

Download MEROPS BLAST DB:
    https://www.ebi.ac.uk/merops/download_list.shtml  (pepunit.lib)
"""

from __future__ import annotations

import gzip
import logging
import re
from pathlib import Path

import numpy as np
import pandas as pd
from tqdm import tqdm

logger = logging.getLogger(__name__)

# Minimum BLAST identity (%) and coverage to trust a hit
MIN_IDENTITY = 30.0
MIN_ALIGN_LEN = 50
MAX_EVALUE = 1e-5

# MEROPS family-to-clan mapping (partial; extend as needed)
FAMILY_TO_CLAN: dict[str, str] = {
    "A01": "AA",
    "A02": "AA",
    "A11": "AA",
    "A22": "AA",
    "C01": "CA",
    "C02": "CA",
    "C19": "CA",
    "M01": "MA",
    "M02": "MA",
    "M04": "MA",
    "M10": "MA",
    "S01": "SA",
    "S08": "SB",
    "S09": "SC",
    "S10": "SC",
    "T01": "PB",
}


# ── parser ────────────────────────────────────────────────────────────────────


def parse_merops_blast(
    path: Path,
    min_identity: float = MIN_IDENTITY,
    min_align_len: int = MIN_ALIGN_LEN,
    max_evalue: float = MAX_EVALUE,
) -> pd.DataFrame:
    """
    Parse BLAST tabular (-outfmt 6) results against MEROPS pepunit database.

    Expected columns:
        qseqid sseqid pident length mismatch gapopen
        qstart qend sstart send evalue bitscore

    Returns tidy DataFrame with columns: protein_id, merops_id,
    family, clan, identity, evalue.
    """
    records = []
    opener = gzip.open if Path(path).suffix == ".gz" else open
    with opener(path, "rt") as fh:
        for line in fh:
            line = line.strip()
            if not line or line.startswith("#"):
                continue
            parts = line.split("\t")
            if len(parts) < 12:
                continue
            try:
                identity = float(parts[2])
                align_len = int(parts[3])
                evalue = float(parts[10])
            except (IndexError, ValueError):
                continue
            if identity < min_identity or align_len < min_align_len or evalue > max_evalue:
                continue

            merops_id = parts[1].split("|")[0] if "|" in parts[1] else parts[1]
            # Extract family code: e.g. "S01.001" → "S01"
            m = re.match(r"([A-Z]\d{2})", merops_id)
            family = m.group(1) if m else "unknown"
            clan = FAMILY_TO_CLAN.get(family, "unassigned")

            records.append(
                {
                    "protein_id": parts[0],
                    "merops_id": merops_id,
                    "family": family,
                    "clan": clan,
                    "identity": identity,
                    "evalue": evalue,
                }
            )

    df = pd.DataFrame(records)
    if df.empty:
        return df
    # Keep best hit per protein
    return df.sort_values("evalue").drop_duplicates(subset="protein_id")


# ── feature building ──────────────────────────────────────────────────────────


def merops_to_features(df: pd.DataFrame, n_proteins: int | None = None) -> pd.Series:
    """
    Aggregate MEROPS BLAST hits to genome-level features.

    Features
    --------
    merops_total_protease_frac  : fraction of proteome with a MEROPS hit
    merops_family_{F}           : copy number of family F
    merops_clan_{C}             : copy number of clan C
    """
    if df.empty:
        return pd.Series(dtype=np.float32)

    features: dict[str, float] = {}
    total = n_proteins or len(df)

    features["merops_total_protease_frac"] = len(df) / max(total, 1)

    family_counts = df["family"].value_counts()
    for fam, cnt in family_counts.items():
        features[f"merops_family_{fam}"] = float(cnt)

    clan_counts = df["clan"].value_counts()
    for clan, cnt in clan_counts.items():
        features[f"merops_clan_{clan}"] = float(cnt)

    return pd.Series(features, dtype=np.float32)


def build_merops_matrix(
    annotation_paths: dict[str, Path],
    min_identity: float = MIN_IDENTITY,
    min_align_len: int = MIN_ALIGN_LEN,
    max_evalue: float = MAX_EVALUE,
    min_genome_freq: float = 0.01,
) -> pd.DataFrame:
    """
    Build genome × MEROPS feature matrix.

    Parameters
    ----------
    annotation_paths : Dict genome_id -> path to BLAST tabular output.
    min_identity     : Minimum % identity to accept a hit.
    min_align_len    : Minimum alignment length.
    max_evalue       : Maximum e-value.
    min_genome_freq  : Drop families/clans present in < this fraction of genomes.
    """
    rows: dict[str, pd.Series] = {}
    for genome_id, path in tqdm(annotation_paths.items(), desc="MEROPS features"):
        try:
            df = parse_merops_blast(path, min_identity, min_align_len, max_evalue)
            rows[genome_id] = merops_to_features(df)
        except Exception as e:
            logger.warning(f"Failed MEROPS parse for {genome_id}: {e}")

    matrix = pd.DataFrame(rows).T.fillna(0.0).astype(np.float32)
    matrix.index.name = "genome_id"

    if not matrix.empty and min_genome_freq > 0:
        freq = (matrix > 0).mean(axis=0)
        matrix = matrix.loc[:, freq >= min_genome_freq]

    logger.info(f"MEROPS matrix: {matrix.shape}")
    return matrix
