"""
fungal_classifier/utils/io.py

Data loading, saving, and path management utilities.
"""

from __future__ import annotations

import gzip
import logging
import os
import re
from pathlib import Path

import pandas as pd

logger = logging.getLogger(__name__)


def open_text(path: Path):
    """Open a file for text reading, transparently decompressing .gz files."""
    path = Path(path)
    if path.suffix == ".gz":
        return gzip.open(path, "rt")
    return open(path)


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


# ── taxonomy ──────────────────────────────────────────────────────────────────


def load_taxonomy(path: Path) -> pd.DataFrame:
    """
    Load the samples.csv taxonomy table.

    Expected columns (comma-separated):
      ASMID, SPECIESIN, STRAIN, BIOPROJECT, NCBI_TAXONID, BUSCO_LINEAGE,
      PHYLUM, SUBPHYLUM, CLASS, SUBCLASS, ORDER, FAMILY, GENUS, SPECIES,
      LOCUSTAG

    Returns a DataFrame indexed by ASMID with standardised lower-case column
    names.  A 'locustag' column is preserved for joining to annotation files
    whose protein IDs are prefixed with the locus tag.
    """
    df = pd.read_csv(path, dtype=str)
    df.columns = [c.strip().lower() for c in df.columns]
    df = df.rename(columns={"asmid": "genome_id"})
    df = df.set_index("genome_id")
    logger.info(
        f"Loaded taxonomy: {df.shape[0]} assemblies, "
        f"{df.shape[1]} columns; "
        f"{df['phylum'].nunique()} phyla"
    )
    return df


def validate_species_prefixes(
    taxonomy_path: Path,
    annotation_dir: Path,
) -> dict[str, bool]:
    """
    Check that each entry in samples.csv has annotation files in annotation_dir.
    The expected prefix is built as ``{SPECIES}_{STRAIN}`` with whitespace
    replaced by '_' in each column.

    Returns a dict mapping expected_prefix -> found (bool).  Logs a warning for
    every prefix with no matching files.
    """
    annotation_dir = Path(annotation_dir)

    # Collect all entry names present under each annotation subdirectory.
    # Species prefixes may contain dots (e.g. "CBS_148.51"), so we cannot
    # split on the first dot; instead we test startswith per expected prefix.
    annotation_names: list[str] = []
    for subdir in annotation_dir.iterdir():
        if not subdir.is_dir():
            continue
        for entry in subdir.iterdir():
            annotation_names.append(entry.name)

    df = pd.read_csv(taxonomy_path, dtype=str)
    df.columns = [c.strip().lower() for c in df.columns]

    results: dict[str, bool] = {}
    for _, row in df.iterrows():
        species = str(row["species"]).strip()
        strain = str(row["strain"]).strip()
        expected = re.sub(r"\s+", "_", species) + "_" + re.sub(r"\s+", "_", strain)
        # File names equal the prefix exactly (bare directories) or start with
        # the prefix followed by '.' (e.g. "Genus_species_STRAIN.ext").
        found = any(n == expected or n.startswith(expected + ".") for n in annotation_names)
        results[expected] = found
        if not found:
            logger.warning(
                "No annotation files found for expected prefix '%s' (SPECIES: '%s', STRAIN: '%s')",
                expected,
                species,
                strain,
            )

    missing = [p for p, ok in results.items() if not ok]
    if missing:
        logger.warning(
            "%d/%d species prefix(es) have no annotation files: %s",
            len(missing),
            len(results),
            missing,
        )
    else:
        logger.info(
            "All %d species prefixes matched annotation files in %s",
            len(results),
            annotation_dir,
        )

    return results


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


def load_feature_blocks(
    feature_dir: Path, block_names: list[str] | None = None
) -> dict[str, pd.DataFrame]:
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
    extensions: tuple[str, ...] = (
        ".scaffolds.fa",
        ".scaffolds.fa.gz",
        ".scaffolds.fasta",
        ".scaffolds.fna",
        ".fna.gz",
        ".fna",
        ".fa.gz",
        ".fa",
        ".fasta.gz",
        ".fasta",
    ),
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
    Discover annotation files matching a suffix pattern, including .gz variants.

    Follows symlinks so that per-genome subdirectories that are symlinks to
    external locations are traversed correctly.

    Returns dict: genome_id -> Path. Uncompressed files take priority over .gz.
    """
    annotation_dir = Path(annotation_dir)
    paths: dict[str, Path] = {}
    bare_suffix = suffix.lstrip(".")  # e.g. ".tmhmm_short.tsv" → "tmhmm_short.tsv"

    # os.walk with followlinks=True handles symlinked subdirectories;
    # pathlib.rglob does not follow symlinks on Python < 3.13.
    # Sorting filenames ensures uncompressed ("overview.tsv") sorts before
    # the gz variant ("overview.tsv.gz"), so setdefault keeps uncompressed.
    for dirpath, _dirnames, filenames in os.walk(annotation_dir, followlinks=True):
        parent = Path(dirpath)
        for filename in sorted(filenames):
            is_gz = filename.endswith(".gz")
            name = filename[:-3] if is_gz else filename
            if not (filename.endswith(suffix) or filename.endswith(suffix + ".gz")):
                continue
            if not name.endswith(bare_suffix):
                continue
            stem = name[: -len(bare_suffix)].rstrip("._")
            # Fall back to parent directory name for per-genome subdir layout
            # e.g. dbcan/{genome_id}/overview.tsv.gz → genome_id from parent
            genome_id = stem if stem else parent.name
            if genome_ids is None or genome_id in genome_ids:
                paths.setdefault(genome_id, parent / filename)

    logger.info(f"Discovered {len(paths)} annotation files (*{suffix}[.gz]) in {annotation_dir}")
    return paths


# ── results persistence ───────────────────────────────────────────────────────


def save_predictions(
    predictions: pd.Series, probabilities: pd.DataFrame, output_path: Path
) -> None:
    """Save prediction labels and probabilities to TSV."""
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    df = probabilities.copy()
    df.insert(0, "predicted_label", predictions)
    df.to_csv(output_path, sep="\t")
    logger.info(f"Saved predictions for {len(df)} genomes to {output_path}")
