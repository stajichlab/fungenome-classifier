"""
fungal_classifier/features/kmer.py

K-mer and oligonucleotide composition feature extraction.

Computes:
  - k-mer frequency vectors (k=1..6) from genomic or CDS sequences
  - Dinucleotide relative abundance (obs/exp ratios), correcting for base composition
  - Trinucleotide relative abundance

Normalization options:
  - count:               raw counts
  - relative_abundance:  frequency / total_kmers
  - obs_exp:             observed / expected under independence assumption
                         (only defined for k >= 2; uses lower-order frequencies)
"""

from __future__ import annotations

import gzip
import itertools
import logging
from pathlib import Path
from typing import Literal

import numpy as np
import pandas as pd
from Bio import SeqIO
from tqdm import tqdm

logger = logging.getLogger(__name__)

NUCLEOTIDES = list("ACGT")


# ── helpers ───────────────────────────────────────────────────────────────────


def _all_kmers(k: int) -> list[str]:
    """Return all canonical k-mer strings for alphabet ACGT."""
    return ["".join(p) for p in itertools.product(NUCLEOTIDES, repeat=k)]


def _count_kmers(seq: str, k: int) -> dict[str, int]:
    """Slide a window of length k over seq and count occurrences."""
    seq = seq.upper().replace("N", "")  # drop ambiguous bases
    counts: dict[str, int] = {kmer: 0 for kmer in _all_kmers(k)}
    for i in range(len(seq) - k + 1):
        kmer = seq[i : i + k]
        if kmer in counts:
            counts[kmer] += 1
    return counts


def _obs_exp_ratio(counts_k: dict[str, int], counts_k1: dict[str, int]) -> dict[str, float]:
    """
    Compute observed/expected ratio for each k-mer.

    Expected frequency of XY...Z = f(XY...)/f(Y...Z) * f(Y...) corrected form.
    For k=2: obs_exp(XY) = f(XY) / (f(X) * f(Y)) * genome_length
    For k>2: obs_exp(XY...Z) = f(XY...Z) / (f(XY...) * f(Y...Z) / f(Y...))
    """
    total_k = sum(counts_k.values()) or 1
    total_k1 = sum(counts_k1.values()) or 1
    freq_k = {kmer: c / total_k for kmer, c in counts_k.items()}
    freq_k1 = {kmer: c / total_k1 for kmer, c in counts_k1.items()}

    result: dict[str, float] = {}
    for kmer, obs in freq_k.items():
        prefix = kmer[:-1]
        suffix = kmer[1:]
        denom = freq_k1.get(prefix, 0) * freq_k1.get(suffix, 0)
        result[kmer] = obs / denom if denom > 0 else 0.0
    return result


def _load_sequences(
    fasta_path: Path,
    seq_type: Literal["genomic", "cds", "both"] = "genomic",
) -> str:
    """
    Load sequences from a FASTA file and concatenate into one string.
    For 'both', concatenates genomic and CDS sequences with an N separator.
    """
    sequences = []
    fasta_path = Path(fasta_path)
    opener = gzip.open if fasta_path.suffix == ".gz" else open
    with opener(fasta_path, "rt") as fh:
        for record in SeqIO.parse(fh, "fasta"):
            sequences.append(str(record.seq))
    return "N".join(sequences)


# ── public API ────────────────────────────────────────────────────────────────


def compute_kmer_features(
    fasta_path: Path,
    k_values: list[int] = [1, 2, 3, 4, 5, 6],
    normalize: Literal["count", "relative_abundance", "obs_exp"] = "relative_abundance",
    seq_type: Literal["genomic", "cds", "both"] = "genomic",
) -> pd.Series:
    """
    Compute k-mer feature vector for a single genome FASTA.

    Parameters
    ----------
    fasta_path : Path to input FASTA file.
    k_values   : List of k values to compute.
    normalize  : Normalization strategy.
    seq_type   : Which sequences to use.

    Returns
    -------
    pd.Series with index = kmer labels, values = features.
    """
    seq = _load_sequences(fasta_path, seq_type)
    features: dict[str, float] = {}

    counts_cache: dict[int, dict[str, int]] = {}

    for k in sorted(k_values):
        counts_k = _count_kmers(seq, k)
        counts_cache[k] = counts_k
        total = sum(counts_k.values()) or 1

        if normalize == "count":
            for kmer, c in counts_k.items():
                features[f"kmer_{k}_{kmer}"] = float(c)

        elif normalize == "relative_abundance":
            for kmer, c in counts_k.items():
                features[f"kmer_{k}_{kmer}"] = c / total

        elif normalize == "obs_exp":
            if k == 1:
                for kmer, c in counts_k.items():
                    features[f"kmer_{k}_{kmer}"] = c / total
            else:
                counts_k1 = counts_cache[k - 1]
                oe = _obs_exp_ratio(counts_k, counts_k1)
                for kmer, val in oe.items():
                    features[f"kmer_{k}_{kmer}"] = val

    return pd.Series(features, dtype=np.float32)


def build_kmer_matrix(
    fasta_paths: dict[str, Path],
    k_values: list[int] = [1, 2, 3, 4, 5, 6],
    normalize: Literal["count", "relative_abundance", "obs_exp"] = "relative_abundance",
    seq_type: Literal["genomic", "cds", "both"] = "genomic",
    n_jobs: int = 4,
) -> pd.DataFrame:
    """
    Compute k-mer feature matrix for a collection of genomes.

    Parameters
    ----------
    fasta_paths : Dict mapping genome_id -> Path to FASTA.
    k_values    : k values to compute.
    normalize   : Normalization strategy.
    seq_type    : Sequence type to use.
    n_jobs      : Parallelism (uses joblib).

    Returns
    -------
    pd.DataFrame of shape (n_genomes, n_features).
    """
    from joblib import Parallel, delayed

    def _process(genome_id: str, path: Path) -> tuple[str, pd.Series]:
        try:
            vec = compute_kmer_features(path, k_values, normalize, seq_type)
            return genome_id, vec
        except Exception as e:
            logger.warning(f"Failed to process {genome_id}: {e}")
            return genome_id, pd.Series(dtype=np.float32)

    results = Parallel(n_jobs=n_jobs, batch_size=1)(
        delayed(_process)(gid, path)
        for gid, path in tqdm(fasta_paths.items(), desc="K-mer features")
    )

    rows = {gid: vec for gid, vec in results if not vec.empty}
    df = pd.DataFrame(rows).T
    df.index.name = "genome_id"
    df = df.fillna(0.0).astype(np.float32)
    logger.info(f"K-mer matrix: {df.shape[0]} genomes × {df.shape[1]} features")
    return df
