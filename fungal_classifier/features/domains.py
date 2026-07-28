"""
fungal_classifier/features/domains.py

Protein domain feature extraction from Pfam/InterPro annotations.

Expects hmmer domtblout format (from pfam_scan.pl or hmmscan).
Produces a genome × domain matrix of copy numbers or binary presence/absence.
"""
__all__ = ["parse_domtblout", "parse_interpro_tsv", "build_domain_matrix"]

from __future__ import annotations

import gzip
import logging
from pathlib import Path
from typing import Literal

import numpy as np
import pandas as pd
from tqdm import tqdm

logger = logging.getLogger(__name__)


# ── parsers ───────────────────────────────────────────────────────────────────


def parse_domtblout(path: Path, e_value_threshold: float = 1e-5) -> pd.DataFrame:
    """
    Parse hmmer --domtblout file into a tidy DataFrame.

    HMMER domtblout columns (whitespace-delimited, 0-indexed):
      0  : target name        (protein/gene ID)
      1  : target accession   (protein accession, may be '-')
      2  : target description
      3  : query name         (HMM/Pfam family name)
      4  : query accession    (Pfam accession, e.g. PF00001)
      5  : qlen
      6  : full E-value
      7  : full score
      8  : full bias
      9  : domain E-value     (conditional E-value for this domain)
      10 : domain score
      11 : domain bias
      ...

    We use: protein_id=parts[0], domain_name=parts[3],
            domain_acc=parts[4], e_value=parts[6], score=parts[7].

    The Pfam accession (parts[4]) may include a dot-version suffix
    (e.g. PF00001.17) — this is stripped to the base family ID.
    """
    records = []
    opener = gzip.open if Path(path).suffix == ".gz" else open
    with opener(path, "rt") as fh:
        for line in fh:
            line = line.strip()
            if line.startswith("#") or not line:
                continue
            parts = line.split()
            try:
                protein_id = parts[0]
                # Query accession (Pfam ID) — may be '-' if not in Pfam
                raw_acc = parts[4]
                # Strip version suffix (e.g. PF00001.17 -> PF00001); fall back to domain name if absent
                domain_acc = raw_acc.split(".")[0] if raw_acc != "-" else parts[3]
                domain_name = parts[3]
                e_value = float(parts[6])
                score = float(parts[7])
            except (IndexError, ValueError):
                continue
            if e_value <= e_value_threshold:
                records.append(
                    {
                        "protein_id": protein_id,
                        "domain_acc": domain_acc,
                        "domain_name": domain_name,
                        "e_value": e_value,
                        "score": score,
                    }
                )
    return pd.DataFrame(records)


def parse_interpro_tsv(path: Path) -> pd.DataFrame:
    """
    Parse InterProScan TSV output into a tidy DataFrame.

    InterProScan TSV columns (tab-separated):
      protein_id, md5, length, analysis, signature_acc, signature_desc,
      start, end, score, status, date, ipr_acc, ipr_desc, go_terms, pathways
    """
    cols = [
        "protein_id",
        "md5",
        "length",
        "analysis",
        "domain_acc",
        "domain_name",
        "start",
        "end",
        "e_value",
        "status",
        "date",
        "ipr_acc",
        "ipr_desc",
        "go_terms",
        "pathways",
    ]
    df = pd.read_csv(path, sep="\t", header=None, names=cols, low_memory=False)
    df = df[df["e_value"].apply(lambda x: str(x) != "-")]
    df["e_value"] = pd.to_numeric(df["e_value"], errors="coerce")
    return df[["protein_id", "domain_acc", "domain_name", "e_value"]]


# ── feature building ──────────────────────────────────────────────────────────


def _domains_to_vector(
    domain_df: pd.DataFrame,
    all_domains: list[str],
    representation: Literal["binary", "copy_number"],
) -> pd.Series:
    """Convert a per-genome domain DataFrame to a feature vector."""
    counts = domain_df["domain_acc"].value_counts()
    vec = pd.Series(0, index=all_domains, dtype=np.float32)
    for domain, count in counts.items():
        if domain in vec.index:
            vec[domain] = 1.0 if representation == "binary" else float(count)
    return vec


def build_domain_matrix(
    annotation_paths: dict[str, Path],
    format: Literal["domtblout", "interpro"] = "domtblout",
    representation: Literal["binary", "copy_number"] = "copy_number",
    e_value_threshold: float = 1e-5,
    min_genome_freq: float = 0.01,
) -> pd.DataFrame:
    """
    Build genome × domain feature matrix.

    Parameters
    ----------
    annotation_paths : Dict mapping genome_id -> path to hmmer/interproscan output.
    format           : Input format.
    representation   : 'binary' or 'copy_number'.
    e_value_threshold: E-value cutoff for domain hits.
    min_genome_freq  : Drop domains present in fewer than this fraction of genomes.

    Returns
    -------
    pd.DataFrame of shape (n_genomes, n_domains).
    """
    parser = parse_domtblout if format == "domtblout" else parse_interpro_tsv

    # First pass: collect all domain accessions
    per_genome: dict[str, pd.DataFrame] = {}
    all_domains: set[str] = set()

    for genome_id, path in tqdm(annotation_paths.items(), desc="Parsing domain files"):
        try:
            df = parser(path)
            if format == "domtblout":
                df = df[df["e_value"] <= e_value_threshold]
            per_genome[genome_id] = df
            all_domains.update(df["domain_acc"].unique())
        except Exception as e:
            logger.warning(f"Failed to parse {genome_id}: {e}")

    all_domains_list = sorted(all_domains)

    # Second pass: build matrix
    rows = {}
    for genome_id, df in per_genome.items():
        rows[genome_id] = _domains_to_vector(df, all_domains_list, representation)

    matrix = pd.DataFrame(rows).T
    matrix.index.name = "genome_id"
    matrix = matrix.fillna(0.0).astype(np.float32)

    # Filter rare domains
    freq = (matrix > 0).mean(axis=0)
    keep = freq[freq >= min_genome_freq].index
    matrix = matrix[keep]

    if matrix.shape[1] == 0:
        logger.warning(
            f"min_genome_freq={min_genome_freq} is too high — all domains filtered out. "
            f"Consider lowering min_genome_freq (default: 0.01)."
        )

    logger.info(
        f"Domain matrix: {matrix.shape[0]} genomes × {matrix.shape[1]} domains "
        f"(dropped {len(all_domains_list) - len(keep)} rare domains)"
    )
    return matrix
