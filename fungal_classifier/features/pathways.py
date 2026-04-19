"""
fungal_classifier/features/pathways.py

Functional pathway feature extraction.

Handles:
  - KEGG ortholog (KO) counts aggregated to pathway level
  - GO term counts aggregated to GO-slim or level-2 categories
  - CAZyme family profiles (from dbCAN output)
  - Biosynthetic gene cluster (BGC) counts (from antiSMASH output)
"""

from __future__ import annotations

import gzip
import json
import logging
from pathlib import Path

import numpy as np
import pandas as pd
from tqdm import tqdm

logger = logging.getLogger(__name__)


# ── KEGG ──────────────────────────────────────────────────────────────────────


def parse_kegg_annotation(path: Path) -> pd.Series:
    """
    Parse a KEGG KO annotation file (tab-separated: gene_id, KO_term).
    Returns Series: pathway_id -> count of genes mapped to that pathway.

    Requires a local KO-to-pathway mapping (ko_to_pathway.tsv).
    This file can be downloaded from KEGG FTP or via the KEGG REST API.
    """
    try:
        ko_df = pd.read_csv(path, sep="\t", header=None, names=["gene_id", "ko_term"])
        ko_df = ko_df.dropna(subset=["ko_term"])
        ko_counts = ko_df["ko_term"].value_counts()
        return ko_counts.rename_axis("ko_term").rename("count")
    except Exception as e:
        logger.warning(f"Failed to parse KEGG file {path}: {e}")
        return pd.Series(dtype=float)


def build_kegg_matrix(
    annotation_paths: dict[str, Path],
    ko_to_pathway_map: dict[str, list[str]] | None = None,
    min_genome_freq: float = 0.01,
) -> pd.DataFrame:
    """
    Build genome × KEGG-pathway feature matrix.

    Parameters
    ----------
    annotation_paths  : Dict genome_id -> path to KO annotation file.
    ko_to_pathway_map : Optional pre-loaded KO -> [pathway_id, ...] mapping.
                        If None, uses raw KO term counts.
    min_genome_freq   : Drop features in < this fraction of genomes.
    """
    rows: dict[str, pd.Series] = {}

    for genome_id, path in tqdm(annotation_paths.items(), desc="KEGG features"):
        ko_counts = parse_kegg_annotation(path)
        if ko_to_pathway_map is None:
            rows[genome_id] = ko_counts
        else:
            pathway_counts: dict[str, float] = {}
            for ko, count in ko_counts.items():
                for pathway in ko_to_pathway_map.get(ko, []):
                    pathway_counts[pathway] = pathway_counts.get(pathway, 0) + count
            rows[genome_id] = pd.Series(pathway_counts, dtype=np.float32)

    matrix = pd.DataFrame(rows).T.fillna(0.0).astype(np.float32)
    matrix.index.name = "genome_id"
    freq = (matrix > 0).mean(axis=0)
    matrix = matrix.loc[:, freq >= min_genome_freq]
    logger.info(f"KEGG matrix: {matrix.shape}")
    return matrix


# ── CAZyme ────────────────────────────────────────────────────────────────────


def parse_dbcan_output(path: Path, min_tools: int = 2) -> pd.Series:
    """
    Parse dbCAN overview.txt output.

    Columns: Gene ID, EC#, HMMER, Hotpep, DIAMOND, Signalp, #ofTools, CAZyme family.
    Filters to annotations supported by >= min_tools tools.
    Returns Series: cazyme_family -> count.
    """
    try:
        df = pd.read_csv(path, sep="\t")
        df.columns = [c.strip() for c in df.columns]
        # Filter by number of supporting tools
        df = df[df["#ofTools"] >= min_tools]
        # The last column is the predicted CAZyme family
        family_col = df.columns[-1]
        # Families may be semicolon-delimited (e.g. "GH5;CBM1")
        families = df[family_col].dropna().str.split(";").explode()
        families = families.str.strip().str.extract(r"([A-Z]+\d+)")[0].dropna()
        return families.value_counts().rename("count").rename_axis("cazyme_family")
    except Exception as e:
        logger.warning(f"Failed to parse dbCAN file {path}: {e}")
        return pd.Series(dtype=float)


def build_cazyme_matrix(
    annotation_paths: dict[str, Path],
    min_tools: int = 2,
    min_genome_freq: float = 0.01,
) -> pd.DataFrame:
    """Build genome × CAZyme-family feature matrix."""
    rows = {
        gid: parse_dbcan_output(path, min_tools)
        for gid, path in tqdm(annotation_paths.items(), desc="CAZyme features")
    }
    matrix = pd.DataFrame(rows).T.fillna(0.0).astype(np.float32)
    matrix.index.name = "genome_id"
    freq = (matrix > 0).mean(axis=0)
    matrix = matrix.loc[:, freq >= min_genome_freq]
    logger.info(f"CAZyme matrix: {matrix.shape}")
    return matrix


# ── BGC (antiSMASH) ───────────────────────────────────────────────────────────


def parse_antismash_json(path: Path) -> pd.Series:
    """
    Parse antiSMASH JSON output file.
    Returns Series: bgc_type -> count of clusters.
    """
    try:
        opener = gzip.open if Path(path).suffix == ".gz" else open
        with opener(path, "rt") as fh:
            data = json.load(fh)
        bgc_counts: dict[str, int] = {}
        for record in data.get("records", []):
            for region in record.get("areas", []):
                for product in region.get("products", []):
                    bgc_counts[product] = bgc_counts.get(product, 0) + 1
        return pd.Series(bgc_counts, dtype=np.float32)
    except Exception as e:
        logger.warning(f"Failed to parse antiSMASH JSON {path}: {e}")
        return pd.Series(dtype=float)


def build_bgc_matrix(
    annotation_paths: dict[str, Path],
    min_genome_freq: float = 0.01,
) -> pd.DataFrame:
    """Build genome × BGC-type feature matrix from antiSMASH JSON files."""
    rows = {
        gid: parse_antismash_json(path)
        for gid, path in tqdm(annotation_paths.items(), desc="BGC features")
    }
    matrix = pd.DataFrame(rows).T.fillna(0.0).astype(np.float32)
    matrix.index.name = "genome_id"
    freq = (matrix > 0).mean(axis=0)
    matrix = matrix.loc[:, freq >= min_genome_freq]
    logger.info(f"BGC matrix: {matrix.shape}")
    return matrix


# ── GO terms ──────────────────────────────────────────────────────────────────


def aggregate_go_terms(
    go_term_counts: pd.Series,
    go_slim_terms: set[str] | None = None,
    level: int = 2,
) -> pd.Series:
    """
    Aggregate raw GO term counts to GO-slim or a fixed ontology level.

    Parameters
    ----------
    go_term_counts : Series of GO:XXXXXXX -> count.
    go_slim_terms  : Set of GO slim terms to keep. If None, keeps all.
    level          : Not used if go_slim_terms is provided; otherwise
                     applies a simple depth-based filter (requires goatools).
    """
    if go_slim_terms is not None:
        return go_term_counts[go_term_counts.index.isin(go_slim_terms)]
    return go_term_counts


def build_go_matrix(
    annotation_paths: dict[str, Path],
    go_slim_terms: set[str] | None = None,
    min_genome_freq: float = 0.01,
) -> pd.DataFrame:
    """
    Build genome × GO-term feature matrix.

    Expects annotation files as tab-separated: gene_id, GO:XXXXXXX
    (one row per gene-term association, as produced by InterProScan).
    """
    rows: dict[str, pd.Series] = {}
    for genome_id, path in tqdm(annotation_paths.items(), desc="GO features"):
        try:
            df = pd.read_csv(path, sep="\t", header=None, names=["gene_id", "go_term"])
            counts = df["go_term"].value_counts()
            counts = aggregate_go_terms(counts, go_slim_terms)
            rows[genome_id] = counts
        except Exception as e:
            logger.warning(f"Failed to parse GO file {path}: {e}")

    matrix = pd.DataFrame(rows).T.fillna(0.0).astype(np.float32)
    matrix.index.name = "genome_id"
    freq = (matrix > 0).mean(axis=0)
    matrix = matrix.loc[:, freq >= min_genome_freq]
    logger.info(f"GO matrix: {matrix.shape}")
    return matrix
