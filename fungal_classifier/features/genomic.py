"""
fungal_classifier/features/genomic.py

Genome-level size and protein length features.

Features produced
-----------------
genome_length_total      : total assembly size in bp (sum of all scaffolds/contigs)
genome_scaffold_count    : number of scaffolds/contigs
genome_n50               : N50 scaffold length in bp
genome_gc_content        : GC fraction across entire assembly
protein_count            : number of protein records
protein_length_mean      : mean protein length (aa)
protein_length_median    : median protein length (aa)
protein_length_std       : standard deviation of protein lengths (aa)
protein_length_min       : shortest protein (aa)
protein_length_max       : longest protein (aa)
protein_total_aa         : total amino acids across all proteins (proxy for proteome size)
"""

from __future__ import annotations

import gzip
import logging
from pathlib import Path

import numpy as np
import pandas as pd
from Bio import SeqIO
from tqdm import tqdm

logger = logging.getLogger(__name__)


def _open(path: Path):
    path = Path(path)
    return gzip.open(path, "rt") if path.suffix == ".gz" else open(path)


def _genome_stats(fasta_path: Path) -> dict[str, float]:
    lengths = []
    gc_total = 0
    base_total = 0

    with _open(fasta_path) as fh:
        for record in SeqIO.parse(fh, "fasta"):
            seq = str(record.seq).upper()
            lengths.append(len(seq))
            gc_total += seq.count("G") + seq.count("C")
            base_total += sum(seq.count(b) for b in "ACGT")

    if not lengths:
        return {}

    lengths.sort(reverse=True)
    cumsum = 0
    half = sum(lengths) / 2
    n50 = 0
    for ln in lengths:
        cumsum += ln
        if cumsum >= half:
            n50 = ln
            break

    return {
        "genome_length_total": float(sum(lengths)),
        "genome_scaffold_count": float(len(lengths)),
        "genome_n50": float(n50),
        "genome_gc_content": gc_total / max(base_total, 1),
    }


def _protein_stats(fasta_path: Path) -> dict[str, float]:
    lengths = []

    with _open(fasta_path) as fh:
        for record in SeqIO.parse(fh, "fasta"):
            lengths.append(len(record.seq))

    if not lengths:
        return {}

    arr = np.array(lengths, dtype=np.float32)
    return {
        "protein_count": float(len(lengths)),
        "protein_length_mean": float(arr.mean()),
        "protein_length_median": float(np.median(arr)),
        "protein_length_std": float(arr.std()),
        "protein_length_min": float(arr.min()),
        "protein_length_max": float(arr.max()),
        "protein_total_aa": float(arr.sum()),
    }


def build_genomic_matrix(
    genome_fasta_paths: dict[str, Path],
    protein_fasta_paths: dict[str, Path] | None = None,
    n_jobs: int = 1,
) -> pd.DataFrame:
    """
    Build genome × genomic-size feature matrix.

    Parameters
    ----------
    genome_fasta_paths   : genome_id -> genome assembly FASTA
    protein_fasta_paths  : genome_id -> protein FASTA (optional)
    n_jobs               : parallel workers

    Returns
    -------
    pd.DataFrame of shape (n_genomes, n_features)
    """
    from joblib import Parallel, delayed

    def _process(genome_id: str, genome_path: Path, protein_path: Path | None):
        try:
            features: dict[str, float] = {}
            features.update(_genome_stats(genome_path))
            if protein_path is not None:
                features.update(_protein_stats(protein_path))
            return genome_id, pd.Series(features, dtype=np.float32)
        except Exception as e:
            logger.warning(f"Failed genomic features for {genome_id}: {e}")
            return genome_id, pd.Series(dtype=np.float32)

    tasks = [
        (gid, path, (protein_fasta_paths or {}).get(gid))
        for gid, path in genome_fasta_paths.items()
    ]

    results = Parallel(n_jobs=n_jobs)(
        delayed(_process)(gid, gpath, ppath)
        for gid, gpath, ppath in tqdm(tasks, desc="Genomic size features")
    )

    rows = {gid: vec for gid, vec in results if not vec.empty}
    matrix = pd.DataFrame(rows).T.fillna(0.0).astype(np.float32)
    matrix.index.name = "genome_id"
    logger.info(f"Genomic matrix: {matrix.shape[0]} genomes × {matrix.shape[1]} features")
    return matrix
