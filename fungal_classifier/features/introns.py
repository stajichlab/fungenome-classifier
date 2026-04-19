"""
fungal_classifier/features/introns.py

Intron structure and splice site features from GFF3 annotation + genome FASTA.

Introns are inferred as the gaps between consecutive exons (or CDS features) of
the same mRNA/transcript.  Splice site dinucleotides are extracted directly from
the genome sequence, with strand-aware reverse complementation.

Features produced
-----------------
intron_count              : total introns across all multi-exon genes
intron_per_gene           : mean introns per gene (counting only multi-exon genes)
intron_genes_fraction     : fraction of genes with ≥ 1 intron
intron_length_mean        : mean intron length (bp)
intron_length_median      : median intron length (bp)
intron_length_std         : std dev of intron lengths (bp)
intron_length_min         : shortest intron (bp)
intron_length_max         : longest intron (bp)

Splice site features (computed when genome FASTA is supplied):
  donor_GT / donor_GC / donor_AT / donor_other
      fraction of 5' donor sites with each dinucleotide class
  acceptor_AG / acceptor_AC / acceptor_other
      fraction of 3' acceptor sites with each dinucleotide class
  canonical_GTAG_fraction   : fraction of introns with GT donor + AG acceptor
  atac_fraction             : fraction of AT-AC introns (U12 minor spliceosome)
  polypyrimidine_score      : mean (C+T) fraction in the 20 bp immediately
                              upstream of the 3' acceptor (polypyrimidine tract)
"""

from __future__ import annotations

import gzip
import logging
from collections import defaultdict
from pathlib import Path

import numpy as np
import pandas as pd
from Bio import SeqIO
from tqdm import tqdm

logger = logging.getLogger(__name__)

_COMP = str.maketrans("ACGTacgtNn", "TGCAtgcaNn")


def _rc(seq: str) -> str:
    return seq.translate(_COMP)[::-1]


# ── GFF3 parsing ──────────────────────────────────────────────────────────────

def _parse_attributes(attr_str: str) -> dict[str, str]:
    """Parse GFF3 attribute column into a dict."""
    attrs: dict[str, str] = {}
    for part in attr_str.strip().rstrip(";").split(";"):
        part = part.strip()
        if "=" in part:
            key, _, val = part.partition("=")
            attrs[key.strip()] = val.strip()
    return attrs


def _open(path: Path):
    path = Path(path)
    return gzip.open(path, "rt") if path.suffix == ".gz" else open(path)


def _parse_gff3_introns(
    gff_path: Path,
    feature_types: tuple[str, ...] = ("exon",),
    fallback_types: tuple[str, ...] = ("CDS",),
) -> list[tuple[str, int, int, str]]:
    """
    Parse GFF3 and return inferred intron coordinates.

    Introns are the gaps between consecutive exon (or CDS) features that share
    the same Parent transcript, on the same chromosome and strand.

    Parameters
    ----------
    gff_path       : Path to GFF3 file (plain or .gz).
    feature_types  : Primary feature type(s) to use as exon proxies.
    fallback_types : Used only if no primary features are found.

    Returns
    -------
    List of (chrom, intron_start, intron_end, strand) in 0-based half-open
    coordinates, i.e. the intron sequence is genome[intron_start:intron_end].
    """
    # transcript_id -> list of (chrom, start, end, strand) in 0-based half-open
    exon_groups: dict[str, list[tuple[str, int, int, str]]] = defaultdict(list)
    types_seen: set[str] = set()

    with _open(gff_path) as fh:
        for line in fh:
            if line.startswith("#") or not line.strip():
                continue
            parts = line.rstrip("\n").split("\t")
            if len(parts) < 9:
                continue
            chrom, _, ftype, gff_start, gff_end, _, strand, _, attr_str = parts[:9]
            types_seen.add(ftype)
            if ftype not in feature_types:
                continue
            attrs = _parse_attributes(attr_str)
            parent = attrs.get("Parent", attrs.get("transcript_id", ""))
            if not parent:
                continue
            # GFF3 is 1-based inclusive → 0-based half-open
            start = int(gff_start) - 1
            end = int(gff_end)
            exon_groups[parent].append((chrom, start, end, strand))

    # Fall back to CDS features if no primary features found
    if not exon_groups:
        with _open(gff_path) as fh:
            for line in fh:
                if line.startswith("#") or not line.strip():
                    continue
                parts = line.rstrip("\n").split("\t")
                if len(parts) < 9:
                    continue
                chrom, _, ftype, gff_start, gff_end, _, strand, _, attr_str = parts[:9]
                if ftype not in fallback_types:
                    continue
                attrs = _parse_attributes(attr_str)
                parent = attrs.get("Parent", attrs.get("transcript_id", ""))
                if not parent:
                    continue
                start = int(gff_start) - 1
                end = int(gff_end)
                exon_groups[parent].append((chrom, start, end, strand))

    if not exon_groups:
        logger.warning(
            f"No exon/CDS features found in {gff_path}. "
            f"Feature types seen: {sorted(types_seen)}"
        )
        return []

    introns: list[tuple[str, int, int, str]] = []
    for _tid, exons in exon_groups.items():
        if len(exons) < 2:
            continue
        # All exons should be on the same chrom/strand; take from first record
        chrom = exons[0][0]
        strand = exons[0][3]
        # Sort by start coordinate
        exons_sorted = sorted(exons, key=lambda e: e[1])
        for i in range(len(exons_sorted) - 1):
            e1_end = exons_sorted[i][2]      # 0-based half-open end of left exon
            e2_start = exons_sorted[i + 1][1]  # 0-based start of right exon
            if e2_start > e1_end:            # genuine gap (not overlapping/adjacent)
                introns.append((chrom, e1_end, e2_start, strand))

    return introns


# ── genome loading ─────────────────────────────────────────────────────────────

def _load_genome_index(fasta_path: Path) -> dict[str, str]:
    """Load all scaffold sequences into a dict: id -> sequence string."""
    index: dict[str, str] = {}
    with _open(fasta_path) as fh:
        for record in SeqIO.parse(fh, "fasta"):
            index[record.id] = str(record.seq).upper()
    return index


# ── splice site extraction ────────────────────────────────────────────────────

def _donor_acceptor(
    genome: dict[str, str],
    chrom: str,
    intron_start: int,
    intron_end: int,
    strand: str,
    ppt_window: int = 20,
) -> tuple[str, str, float | None]:
    """
    Extract donor dinucleotide, acceptor dinucleotide, and polypyrimidine score.

    Returns (donor_2nt, acceptor_2nt, ppt_score) where ppt_score is None if
    the intron is too short to compute.  All sequences are returned in 5'→3'
    mRNA orientation.
    """
    seq = genome.get(chrom, "")
    intron_len = intron_end - intron_start
    if intron_len < 4 or intron_start < 0 or intron_end > len(seq):
        return ("", "", None)

    if strand == "+":
        donor = seq[intron_start: intron_start + 2]
        acceptor = seq[intron_end - 2: intron_end]
        if intron_len >= ppt_window + 2:
            tract = seq[intron_end - ppt_window - 2: intron_end - 2]
            ppt = (tract.count("C") + tract.count("T")) / max(len(tract), 1)
        else:
            ppt = None
    else:  # minus strand — donor is at the HIGH coordinate end
        donor = _rc(seq[intron_end - 2: intron_end])
        acceptor = _rc(seq[intron_start: intron_start + 2])
        if intron_len >= ppt_window + 2:
            tract = _rc(seq[intron_start + 2: intron_start + 2 + ppt_window])
            ppt = (tract.count("C") + tract.count("T")) / max(len(tract), 1)
        else:
            ppt = None

    return (donor, acceptor, ppt)


# ── per-genome feature computation ───────────────────────────────────────────

def compute_intron_features(
    gff_path: Path,
    genome_fasta: Path | None = None,
    feature_types: tuple[str, ...] = ("exon",),
    ppt_window: int = 20,
) -> pd.Series:
    """
    Compute intron structure and splice site features for one genome.

    Parameters
    ----------
    gff_path       : GFF3 annotation file (plain or .gz).
    genome_fasta   : Genome FASTA for splice site extraction (optional).
                     When omitted, only length/count features are computed.
    feature_types  : GFF3 feature type(s) to treat as exons.
    ppt_window     : Number of bp upstream of 3' acceptor for PPT scoring.

    Returns
    -------
    pd.Series of float32 features.
    """
    introns = _parse_gff3_introns(gff_path, feature_types=feature_types)

    features: dict[str, float] = {}

    if not introns:
        return pd.Series(features, dtype=np.float32)

    # ── length and count features ─────────────────────────────────────────────
    lengths = np.array([end - start for _, start, end, _ in introns], dtype=np.float32)

    # Count multi-exon genes: number of distinct parent transcripts with ≥1 intron
    # (We count at the intron level; for per-gene stats we need transcript grouping)
    # Re-parse to count transcripts — reuse exon_groups indirectly via intron count
    # Since we already have the flat intron list, approximate gene count from GFF
    # by counting transcripts with ≥ 2 exons.  We do this by re-running the parse
    # with counting only, which is fast.
    n_multigene, n_total_genes = _count_gene_stats(gff_path, feature_types)

    features["intron_count"] = float(len(introns))
    features["intron_per_gene"] = float(len(introns)) / max(n_multigene, 1)
    features["intron_genes_fraction"] = n_multigene / max(n_total_genes, 1)
    features["intron_length_mean"] = float(lengths.mean())
    features["intron_length_median"] = float(np.median(lengths))
    features["intron_length_std"] = float(lengths.std())
    features["intron_length_min"] = float(lengths.min())
    features["intron_length_max"] = float(lengths.max())

    # ── splice site features (require genome) ─────────────────────────────────
    if genome_fasta is not None:
        genome = _load_genome_index(genome_fasta)

        donors: list[str] = []
        acceptors: list[str] = []
        ppt_scores: list[float] = []

        for chrom, start, end, strand in introns:
            d, a, ppt = _donor_acceptor(genome, chrom, start, end, strand, ppt_window)
            if d and a:
                donors.append(d)
                acceptors.append(a)
                if ppt is not None:
                    ppt_scores.append(ppt)

        n = max(len(donors), 1)

        donor_counts = {"GT": 0, "GC": 0, "AT": 0}
        for d in donors:
            d2 = d[:2].upper()
            if d2 in donor_counts:
                donor_counts[d2] += 1
        features["donor_GT"] = donor_counts["GT"] / n
        features["donor_GC"] = donor_counts["GC"] / n
        features["donor_AT"] = donor_counts["AT"] / n
        features["donor_other"] = (n - sum(donor_counts.values())) / n

        acceptor_counts = {"AG": 0, "AC": 0}
        for a in acceptors:
            a2 = a[-2:].upper()
            if a2 in acceptor_counts:
                acceptor_counts[a2] += 1
        features["acceptor_AG"] = acceptor_counts["AG"] / n
        features["acceptor_AC"] = acceptor_counts["AC"] / n
        features["acceptor_other"] = (n - sum(acceptor_counts.values())) / n

        # Canonical GT-AG and minor AT-AC
        gtag = sum(
            1 for d, a in zip(donors, acceptors)
            if d[:2].upper() == "GT" and a[-2:].upper() == "AG"
        )
        atac = sum(
            1 for d, a in zip(donors, acceptors)
            if d[:2].upper() == "AT" and a[-2:].upper() == "AC"
        )
        features["canonical_GTAG_fraction"] = gtag / n
        features["atac_fraction"] = atac / n
        features["noncanonical_fraction"] = (n - gtag - atac) / n

        features["polypyrimidine_score"] = (
            float(np.mean(ppt_scores)) if ppt_scores else 0.0
        )

    return pd.Series(features, dtype=np.float32)


def _count_gene_stats(
    gff_path: Path,
    feature_types: tuple[str, ...] = ("exon",),
) -> tuple[int, int]:
    """
    Return (n_multi_exon_transcripts, n_total_transcripts).

    Runs a lightweight pass over the GFF without storing sequences.
    """
    transcript_exon_counts: dict[str, int] = defaultdict(int)
    found = False

    with _open(gff_path) as fh:
        for line in fh:
            if line.startswith("#") or not line.strip():
                continue
            parts = line.rstrip("\n").split("\t")
            if len(parts) < 9:
                continue
            ftype = parts[2]
            if ftype not in feature_types:
                continue
            attrs = _parse_attributes(parts[8])
            parent = attrs.get("Parent", attrs.get("transcript_id", ""))
            if parent:
                transcript_exon_counts[parent] += 1
                found = True

    if not found:
        # Fallback to CDS
        with _open(gff_path) as fh:
            for line in fh:
                if line.startswith("#") or not line.strip():
                    continue
                parts = line.rstrip("\n").split("\t")
                if len(parts) < 9:
                    continue
                if parts[2] != "CDS":
                    continue
                attrs = _parse_attributes(parts[8])
                parent = attrs.get("Parent", attrs.get("transcript_id", ""))
                if parent:
                    transcript_exon_counts[parent] += 1

    total = len(transcript_exon_counts)
    multi = sum(1 for c in transcript_exon_counts.values() if c >= 2)
    return multi, total


# ── matrix builder ────────────────────────────────────────────────────────────

def build_intron_matrix(
    gff_paths: dict[str, Path],
    genome_fasta_paths: dict[str, Path] | None = None,
    feature_types: tuple[str, ...] = ("exon",),
    ppt_window: int = 20,
    n_jobs: int = 1,
) -> pd.DataFrame:
    """
    Build genome × intron feature matrix from GFF3 + genome FASTAs.

    Parameters
    ----------
    gff_paths           : genome_id -> GFF3 file path.
    genome_fasta_paths  : genome_id -> genome FASTA (optional; enables splice
                          site and PPT features when provided).
    feature_types       : GFF3 feature type(s) to treat as exons (default: exon).
    ppt_window          : Polypyrimidine tract window size (bp upstream of 3' ss).
    n_jobs              : Parallel workers.

    Returns
    -------
    pd.DataFrame of shape (n_genomes, n_features).
    """
    from joblib import Parallel, delayed

    def _process(genome_id: str) -> tuple[str, pd.Series]:
        gff = gff_paths[genome_id]
        fasta = (genome_fasta_paths or {}).get(genome_id)
        try:
            vec = compute_intron_features(
                gff_path=gff,
                genome_fasta=fasta,
                feature_types=feature_types,
                ppt_window=ppt_window,
            )
            return genome_id, vec
        except Exception as e:
            logger.warning(f"Failed intron features for {genome_id}: {e}")
            return genome_id, pd.Series(dtype=np.float32)

    results = Parallel(n_jobs=n_jobs)(
        delayed(_process)(gid)
        for gid in tqdm(sorted(gff_paths), desc="Intron features")
    )

    rows = {gid: vec for gid, vec in results if not vec.empty}
    matrix = pd.DataFrame(rows).T.fillna(0.0).astype(np.float32)
    matrix.index.name = "genome_id"
    logger.info(f"Intron matrix: {matrix.shape[0]} genomes × {matrix.shape[1]} features")
    return matrix
