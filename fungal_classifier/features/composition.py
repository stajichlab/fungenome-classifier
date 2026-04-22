"""
fungal_classifier/features/composition.py

Codon usage, amino acid frequency, and gene count features.

Three input sources are supported (in order of preference):
  1. Pre-computed codon frequency CSV  ({genome_id}.cds-transcripts.codon_freq.csv)
     Columns: species_prefix, codon, frequency
  2. CDS nucleotide FASTA  ({genome_id}.cds-transcripts.fa[.gz])
     Codon/AA frequencies and gene count are computed directly.
  3. Protein FASTA  ({genome_id}.{faa,pep,aa}[.gz])
     Only AA frequencies and gene count are computed (no codon data).

Features produced
-----------------
gene_count                    : number of CDS/protein records
codon_{XYZ}                   : relative usage of each sense codon (0–1 sum to 1)
aa_{A..Y}                     : relative amino acid frequency (0–1, sum to 1)
gc1, gc2, gc3                 : GC content at each codon position
gc_genome_coding              : overall GC of coding sequence
"""

from __future__ import annotations

import gzip
import itertools
import logging
from pathlib import Path

import numpy as np
import pandas as pd
from Bio import SeqIO
from Bio.Data import CodonTable
from tqdm import tqdm

logger = logging.getLogger(__name__)

# Standard genetic code (NCBI table 1)
_STD_TABLE = CodonTable.unambiguous_dna_by_id[1]
SENSE_CODONS = sorted(
    c for c in itertools.product("ACGT", repeat=3) if "".join(c) not in _STD_TABLE.stop_codons
)
SENSE_CODONS = ["".join(c) for c in SENSE_CODONS]  # 61 codons
AMINO_ACIDS = sorted(set(_STD_TABLE.forward_table.values()))  # 20 AAs


# ── helpers ───────────────────────────────────────────────────────────────────


def _open(path: Path):
    path = Path(path)
    return gzip.open(path, "rt") if path.suffix == ".gz" else open(path)


def _count_codons(seq: str) -> dict[str, int]:
    """Slide a codon window over a CDS sequence (must be in-frame)."""
    seq = seq.upper()
    counts: dict[str, int] = {c: 0 for c in SENSE_CODONS}
    for i in range(0, len(seq) - 2, 3):
        codon = seq[i : i + 3]
        if codon in counts:
            counts[codon] += 1
    return counts


def _codons_to_aa(codon_counts: dict[str, int]) -> dict[str, float]:
    aa_counts: dict[str, float] = {aa: 0.0 for aa in AMINO_ACIDS}
    for codon, cnt in codon_counts.items():
        aa = _STD_TABLE.forward_table.get(codon)
        if aa and aa in aa_counts:
            aa_counts[aa] += cnt
    total = sum(aa_counts.values()) or 1
    return {aa: v / total for aa, v in aa_counts.items()}


def _gc_positions(codon_counts: dict[str, int]) -> tuple[float, float, float]:
    """Return GC fraction at positions 1, 2, 3 of codons."""
    gc = [0, 0, 0]
    total = [0, 0, 0]
    for codon, cnt in codon_counts.items():
        for i, nt in enumerate(codon):
            total[i] += cnt
            if nt in "GC":
                gc[i] += cnt
    return tuple(g / max(t, 1) for g, t in zip(gc, total))


# ── parsers ───────────────────────────────────────────────────────────────────


def parse_codon_freq_csv(path: Path) -> pd.Series:
    """
    Load pre-computed codon frequency CSV.

    Expected columns: species_prefix, codon, frequency.
    Returns Series: codon -> relative frequency.
    """
    df = pd.read_csv(path)
    df.columns = [c.strip().lower() for c in df.columns]
    if "codon" not in df.columns or "frequency" not in df.columns:
        raise ValueError(f"Expected 'codon' and 'frequency' columns in {path}")
    s = df.set_index("codon")["frequency"]
    return s.rename_axis("codon")


def compute_features_from_cds_fasta(
    path: Path,
) -> pd.Series:
    """
    Compute codon usage, AA frequency, gene count and GC features from a CDS FASTA.

    The FASTA must contain in-frame CDS sequences (start at codon position 1).
    """
    total_codons: dict[str, int] = {c: 0 for c in SENSE_CODONS}
    gene_count = 0

    with _open(path) as fh:
        for record in SeqIO.parse(fh, "fasta"):
            gene_count += 1
            cds = str(record.seq)
            for codon, cnt in _count_codons(cds).items():
                total_codons[codon] += cnt

    features: dict[str, float] = {"gene_count": float(gene_count)}

    total_sense = sum(total_codons.values()) or 1

    # Codon relative usage
    for codon, cnt in total_codons.items():
        features[f"codon_{codon}"] = cnt / total_sense

    # Amino acid frequencies
    aa_freqs = _codons_to_aa(total_codons)
    for aa, freq in aa_freqs.items():
        features[f"aa_{aa}"] = freq

    # GC at codon positions
    gc1, gc2, gc3 = _gc_positions(total_codons)
    features["gc1"] = gc1
    features["gc2"] = gc2
    features["gc3"] = gc3
    features["gc_genome_coding"] = (gc1 + gc2 + gc3) / 3.0

    return pd.Series(features, dtype=np.float32)


def compute_features_from_protein_fasta(path: Path) -> pd.Series:
    """
    Compute amino acid frequencies and gene count from a protein FASTA.

    No codon data is produced; codon_* and gc* features will be absent.
    """
    aa_counts: dict[str, float] = {aa: 0.0 for aa in AMINO_ACIDS}
    gene_count = 0

    with _open(path) as fh:
        for record in SeqIO.parse(fh, "fasta"):
            gene_count += 1
            for aa in str(record.seq).upper():
                if aa in aa_counts:
                    aa_counts[aa] += 1.0

    total = sum(aa_counts.values()) or 1
    features: dict[str, float] = {"gene_count": float(gene_count)}
    for aa, cnt in aa_counts.items():
        features[f"aa_{aa}"] = cnt / total

    return pd.Series(features, dtype=np.float32)


# ── matrix builders ───────────────────────────────────────────────────────────


def build_composition_matrix_from_csvs(
    codon_csv_paths: dict[str, Path],
    cds_fasta_paths: dict[str, Path] | None = None,
) -> pd.DataFrame:
    """
    Build genome × composition feature matrix from pre-computed codon frequency CSVs.

    If cds_fasta_paths is also provided, gene_count and AA frequencies are
    computed from the FASTA (the CSV does not contain these).
    """
    rows: dict[str, pd.Series] = {}
    for genome_id, csv_path in tqdm(codon_csv_paths.items(), desc="Codon freq (CSV)"):
        try:
            codon_series = parse_codon_freq_csv(csv_path)
            features: dict = {f"codon_{c}": codon_series.get(c, 0.0) for c in SENSE_CODONS}

            # Derive AA frequencies from codon table
            codon_counts = {c: codon_series.get(c, 0.0) for c in SENSE_CODONS}
            codon_counts_int = {c: codon_counts[c] * 1000 for c in codon_counts}  # pseudo-counts
            aa_freqs = _codons_to_aa(codon_counts_int)
            for aa, freq in aa_freqs.items():
                features[f"aa_{aa}"] = freq

            gc1, gc2, gc3 = _gc_positions(codon_counts_int)
            features["gc1"] = gc1
            features["gc2"] = gc2
            features["gc3"] = gc3
            features["gc_genome_coding"] = (gc1 + gc2 + gc3) / 3.0

            # Gene count from FASTA if available
            if cds_fasta_paths and genome_id in cds_fasta_paths:
                with _open(cds_fasta_paths[genome_id]) as fh:
                    features["gene_count"] = float(sum(1 for r in SeqIO.parse(fh, "fasta")))

            rows[genome_id] = pd.Series(features, dtype=np.float32)
        except Exception as e:
            logger.warning(f"Failed composition (CSV) for {genome_id}: {e}")

    matrix = pd.DataFrame(rows).T.fillna(0.0).astype(np.float32)
    matrix.index.name = "genome_id"
    logger.info(f"Composition matrix (from CSVs): {matrix.shape}")
    return matrix


def build_composition_matrix_from_fasta(
    cds_fasta_paths: dict[str, Path] | None = None,
    protein_fasta_paths: dict[str, Path] | None = None,
    n_jobs: int = 1,
) -> pd.DataFrame:
    """
    Build genome × composition feature matrix by computing from FASTA files.

    CDS FASTAs are preferred; protein FASTAs are used as fallback (no codon data).
    """
    from joblib import Parallel, delayed

    all_paths: dict[str, tuple[Path, str]] = {}
    for gid, p in (protein_fasta_paths or {}).items():
        all_paths[gid] = (p, "protein")
    for gid, p in (cds_fasta_paths or {}).items():
        all_paths[gid] = (p, "cds")  # CDS overrides protein

    def _process(genome_id: str, path: Path, kind: str) -> tuple[str, pd.Series]:
        try:
            if kind == "cds":
                vec = compute_features_from_cds_fasta(path)
            else:
                vec = compute_features_from_protein_fasta(path)
            return genome_id, vec
        except Exception as e:
            logger.warning(f"Failed composition (FASTA) for {genome_id}: {e}")
            return genome_id, pd.Series(dtype=np.float32)

    results = Parallel(n_jobs=n_jobs, batch_size=1)(
        delayed(_process)(gid, path, kind)
        for gid, (path, kind) in tqdm(all_paths.items(), desc="Composition (FASTA)")
    )

    rows = {gid: vec for gid, vec in results if not vec.empty}
    matrix = pd.DataFrame(rows).T.fillna(0.0).astype(np.float32)
    matrix.index.name = "genome_id"
    logger.info(f"Composition matrix (from FASTA): {matrix.shape}")
    return matrix
