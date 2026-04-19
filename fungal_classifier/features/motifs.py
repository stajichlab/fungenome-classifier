"""
fungal_classifier/features/motifs.py

Promoter motif enrichment feature extraction.

Workflow:
  1. Extract upstream sequences (default 1 kb) from genome + GFF annotation
  2. Scan with FIMO against JASPAR fungal PWMs
  3. Aggregate per-genome: motif enrichment score or binary presence

Requires:
  - MEME Suite (fimo): https://meme-suite.org/
  - JASPAR 2024 fungal motif database (or custom PWMs in MEME format)

JASPAR fungal PWM download:
    wget https://jaspar.elixir.lu/download/data/2024/CORE/JASPAR2024_CORE_fungi_non-redundant_pfms_meme.txt
"""

from __future__ import annotations

import gzip
import logging
import subprocess
import tempfile
from pathlib import Path
from typing import Literal

import numpy as np
import pandas as pd
from tqdm import tqdm

logger = logging.getLogger(__name__)


# ── upstream sequence extraction ──────────────────────────────────────────────


def extract_upstream_sequences(
    genome_fasta: Path,
    gff_path: Path,
    upstream_bp: int = 1000,
    output_fasta: Path | None = None,
    feature_type: str = "gene",
) -> Path:
    """
    Extract upstream (promoter) sequences for all genes using bedtools.

    Requires: bedtools in PATH.

    Parameters
    ----------
    genome_fasta : Path to genome FASTA.
    gff_path     : Path to GFF3/GTF annotation.
    upstream_bp  : Bases upstream of TSS to extract.
    output_fasta : Output path (tmp file if None).
    feature_type : GFF feature type to use as TSS anchor (default: 'gene').

    Returns
    -------
    Path to output promoter FASTA.
    """
    if output_fasta is None:
        tmp = tempfile.NamedTemporaryFile(suffix="_promoters.fasta", delete=False)
        output_fasta = Path(tmp.name)
        tmp.close()

    # Step 1: Extract gene features and compute upstream windows
    cmd_flank = [
        "bedtools",
        "flank",
        "-i",
        str(gff_path),
        "-g",
        f"{genome_fasta}.fai",  # requires samtools faidx genome.fasta first
        "-l",
        str(upstream_bp),
        "-r",
        "0",
        "-s",  # strand-aware
    ]

    cmd_getfasta = [
        "bedtools",
        "getfasta",
        "-fi",
        str(genome_fasta),
        "-bed",
        "stdin",
        "-s",  # strand-aware
        "-name",
        "-fo",
        str(output_fasta),
    ]

    try:
        p1 = subprocess.Popen(cmd_flank, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
        p2 = subprocess.Popen(
            cmd_getfasta, stdin=p1.stdout, stdout=subprocess.PIPE, stderr=subprocess.PIPE
        )
        p1.stdout.close()
        _, err2 = p2.communicate()

        if p2.returncode != 0:
            raise RuntimeError(f"bedtools getfasta failed: {err2.decode()}")

        logger.debug(f"Extracted promoter sequences to {output_fasta}")
    except FileNotFoundError:
        raise EnvironmentError(
            "bedtools not found in PATH. Install with: conda install -c bioconda bedtools"
        )

    return output_fasta


# ── FIMO scanning ─────────────────────────────────────────────────────────────


def run_fimo(
    pwm_database: Path,
    promoter_fasta: Path,
    output_dir: Path,
    p_value_threshold: float = 1e-4,
    extra_args: list[str] | None = None,
) -> Path:
    """
    Run FIMO to scan promoter sequences against a PWM database.

    Parameters
    ----------
    pwm_database       : Path to MEME-format PWM file (e.g. JASPAR fungi).
    promoter_fasta     : Path to promoter sequences FASTA.
    output_dir         : Directory for FIMO output.
    p_value_threshold  : P-value cutoff for reporting motif hits.

    Returns
    -------
    Path to fimo.tsv output file.
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    fimo_tsv = output_dir / "fimo.tsv"

    cmd = [
        "fimo",
        "--thresh",
        str(p_value_threshold),
        "--oc",
        str(output_dir),
        "--no-qvalue",
        "--text",
        str(pwm_database),
        str(promoter_fasta),
    ]
    if extra_args:
        cmd.extend(extra_args)

    try:
        result = subprocess.run(cmd, capture_output=True, text=True)
        if result.returncode != 0:
            raise RuntimeError(f"FIMO failed: {result.stderr}")
    except FileNotFoundError:
        raise EnvironmentError(
            "fimo not found in PATH. Install MEME Suite: https://meme-suite.org/meme/doc/install.html"
        )

    return fimo_tsv


# ── FIMO output parsing ───────────────────────────────────────────────────────


def parse_fimo_tsv(path: Path, p_value_threshold: float = 1e-4) -> pd.DataFrame:
    """
    Parse FIMO TSV output file.

    FIMO TSV columns:
        motif_id, motif_alt_id, sequence_name, start, stop, strand,
        score, p-value, q-value, matched_sequence

    Returns tidy DataFrame of significant hits.
    """
    try:
        df = pd.read_csv(
            path,
            sep="\t",
            comment="#",
            dtype={"p-value": float, "score": float},
        )
        # Drop footer lines FIMO sometimes writes
        df = df.dropna(subset=["motif_id", "sequence_name"])
        df = df[df["p-value"].astype(float) <= p_value_threshold]
        return df
    except Exception as e:
        logger.warning(f"Failed to parse FIMO output {path}: {e}")
        return pd.DataFrame()


# ── feature aggregation ───────────────────────────────────────────────────────


def fimo_hits_to_enrichment(
    fimo_df: pd.DataFrame,
    n_promoters: int,
    representation: Literal["count", "binary", "score_sum"] = "count",
) -> pd.Series:
    """
    Aggregate FIMO hits to per-motif enrichment features for one genome.

    Parameters
    ----------
    fimo_df      : Parsed FIMO hits for one genome (all sequences).
    n_promoters  : Total number of promoters scanned (for normalization).
    representation : How to represent motif hits.

    Returns
    -------
    pd.Series: motif_id -> enrichment value.
    """
    if fimo_df.empty:
        return pd.Series(dtype=np.float32)

    if representation == "binary":
        return (fimo_df.groupby("motif_id")["sequence_name"].nunique() > 0).astype(np.float32)

    elif representation == "count":
        # Fraction of promoters containing at least one hit for this motif
        hits_per_promoter = fimo_df.groupby("motif_id")["sequence_name"].nunique()
        return (hits_per_promoter / max(n_promoters, 1)).astype(np.float32)

    elif representation == "score_sum":
        return fimo_df.groupby("motif_id")["score"].sum().astype(np.float32)

    else:
        raise ValueError(f"Unknown representation: {representation}")


def build_motif_matrix(
    fimo_result_paths: dict[str, Path],
    n_promoters_per_genome: dict[str, int] | None = None,
    representation: Literal["count", "binary", "score_sum"] = "count",
    p_value_threshold: float = 1e-4,
    min_genome_freq: float = 0.01,
) -> pd.DataFrame:
    """
    Build genome × motif feature matrix from pre-computed FIMO outputs.

    Parameters
    ----------
    fimo_result_paths    : Dict genome_id -> path to fimo.tsv file.
    n_promoters_per_genome : Dict genome_id -> number of promoters scanned.
    representation       : Feature type per motif.
    p_value_threshold    : Additional p-value filter on FIMO hits.
    min_genome_freq      : Drop motifs found in < this fraction of genomes.

    Returns
    -------
    pd.DataFrame of shape (n_genomes, n_motifs).
    """
    rows: dict[str, pd.Series] = {}

    for genome_id, path in tqdm(fimo_result_paths.items(), desc="Motif features"):
        try:
            fimo_df = parse_fimo_tsv(path, p_value_threshold)
            n_prom = (n_promoters_per_genome or {}).get(genome_id, 1000)
            vec = fimo_hits_to_enrichment(fimo_df, n_prom, representation)
            rows[genome_id] = vec
        except Exception as e:
            logger.warning(f"Failed motif features for {genome_id}: {e}")

    matrix = pd.DataFrame(rows).T.fillna(0.0).astype(np.float32)
    matrix.index.name = "genome_id"

    freq = (matrix > 0).mean(axis=0)
    matrix = matrix.loc[:, freq >= min_genome_freq]
    logger.info(f"Motif matrix: {matrix.shape}")
    return matrix


# ── per-genome pipeline ───────────────────────────────────────────────────────


def compute_motif_features_for_genome(
    genome_id: str,
    genome_fasta: Path,
    gff_path: Path,
    pwm_database: Path,
    work_dir: Path,
    upstream_bp: int = 1000,
    p_value_threshold: float = 1e-4,
) -> pd.Series:
    """
    Full pipeline: extract promoters → FIMO scan → feature vector for one genome.

    Parameters
    ----------
    work_dir : Writable directory for intermediate files (one subdir per genome).
    """
    genome_work = work_dir / genome_id
    genome_work.mkdir(parents=True, exist_ok=True)

    # Extract promoters
    promoter_fasta = genome_work / "promoters.fasta"
    if not promoter_fasta.exists():
        extract_upstream_sequences(genome_fasta, gff_path, upstream_bp, promoter_fasta)

    # Count promoters
    opener = gzip.open if Path(promoter_fasta).suffix == ".gz" else open
    with opener(promoter_fasta, "rt") as _fh:
        n_promoters = sum(1 for line in _fh if line.startswith(">"))

    # Run FIMO
    fimo_dir = genome_work / "fimo_out"
    fimo_tsv = run_fimo(pwm_database, promoter_fasta, fimo_dir, p_value_threshold)

    # Parse and aggregate
    fimo_df = parse_fimo_tsv(fimo_tsv, p_value_threshold)
    return fimo_hits_to_enrichment(fimo_df, n_promoters, representation="count")


def build_motif_matrix_from_genomes(
    genome_fastas: dict[str, Path],
    gff_paths: dict[str, Path],
    pwm_database: Path,
    work_dir: Path,
    upstream_bp: int = 1000,
    p_value_threshold: float = 1e-4,
    min_genome_freq: float = 0.01,
    n_jobs: int = 4,
) -> pd.DataFrame:
    """
    Build motif feature matrix from raw genomes + annotations.

    Runs the full extract-promoters → FIMO → aggregate pipeline.
    Results are cached in work_dir so partially completed runs can resume.

    Parameters
    ----------
    genome_fastas : Dict genome_id -> genome FASTA path.
    gff_paths     : Dict genome_id -> GFF3 annotation path.
    pwm_database  : Path to JASPAR MEME-format PWM file.
    work_dir      : Working directory for intermediate FIMO files.
    n_jobs        : Parallel workers.
    """
    from joblib import Parallel, delayed

    def _process(genome_id: str) -> tuple[str, pd.Series]:
        try:
            vec = compute_motif_features_for_genome(
                genome_id,
                genome_fastas[genome_id],
                gff_paths[genome_id],
                pwm_database,
                work_dir,
                upstream_bp,
                p_value_threshold,
            )
            return genome_id, vec
        except Exception as e:
            logger.warning(f"Motif pipeline failed for {genome_id}: {e}")
            return genome_id, pd.Series(dtype=np.float32)

    genome_ids = sorted(set(genome_fastas) & set(gff_paths))
    results = Parallel(n_jobs=n_jobs)(
        delayed(_process)(gid) for gid in tqdm(genome_ids, desc="Motif pipeline")
    )

    rows = {gid: vec for gid, vec in results if not vec.empty}
    matrix = pd.DataFrame(rows).T.fillna(0.0).astype(np.float32)
    matrix.index.name = "genome_id"

    freq = (matrix > 0).mean(axis=0)
    matrix = matrix.loc[:, freq >= min_genome_freq]
    logger.info(f"Motif matrix: {matrix.shape}")
    return matrix
