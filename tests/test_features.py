"""
tests/test_features.py

Unit tests for feature extraction modules.
"""

import gzip
from pathlib import Path

import pandas as pd

from fungal_classifier.features.fusion import concat_fusion, filter_low_variance
from fungal_classifier.features.kmer import _count_kmers, _obs_exp_ratio, compute_kmer_features
from fungal_classifier.features.pathways import parse_dbcan_output, parse_dbcan_substrate
from fungal_classifier.utils.io import discover_annotation_files

# ── k-mer tests ───────────────────────────────────────────────────────────────


def test_count_kmers_k1():
    seq = "AACGT"
    counts = _count_kmers(seq, k=1)
    assert counts["A"] == 2
    assert counts["C"] == 1
    assert counts["G"] == 1
    assert counts["T"] == 1


def test_count_kmers_handles_n():
    """N bases should be dropped; remaining characters are all counted."""
    seq = "AANCGT"
    counts = _count_kmers(seq, k=1)
    # "AANCGT" → replace N → "AACGT": A=2, C=1, G=1, T=1 = 5 total
    assert sum(counts.values()) == 5
    assert counts["A"] == 2


def test_count_kmers_k2():
    seq = "ACGT"
    counts = _count_kmers(seq, k=2)
    assert counts["AC"] == 1
    assert counts["CG"] == 1
    assert counts["GT"] == 1


def test_obs_exp_ratio_uniform():
    """For a random sequence with uniform base composition, obs/exp should be ≈1."""
    import random

    rng = random.Random(42)
    seq = "".join(rng.choice("ACGT") for _ in range(10_000))
    counts_2 = _count_kmers(seq, k=2)
    counts_1 = _count_kmers(seq, k=1)
    oe = _obs_exp_ratio(counts_2, counts_1)
    for kmer, val in oe.items():
        assert 0.5 < val < 2.0, f"Unexpected obs/exp {val} for {kmer}"


def test_kmer_feature_vector_length():
    """Feature vector should have 4^k entries per k value."""
    import tempfile
    from pathlib import Path

    seq = "ACGTACGTACGT" * 100
    with tempfile.NamedTemporaryFile(suffix=".fasta", mode="w", delete=False) as f:
        f.write(f">test_genome\n{seq}\n")
        fasta_path = Path(f.name)

    vec = compute_kmer_features(fasta_path, k_values=[1, 2, 3], normalize="relative_abundance")
    expected_len = 4 + 16 + 64
    assert len(vec) == expected_len, f"Expected {expected_len}, got {len(vec)}"
    fasta_path.unlink()


# ── fusion tests ──────────────────────────────────────────────────────────────


def test_filter_low_variance_removes_constant():
    df = pd.DataFrame(
        {
            "a": [1, 1, 1, 1],
            "b": [1, 2, 3, 4],
            "c": [0, 0, 0, 0],
        }
    )
    filtered = filter_low_variance(df, threshold=0.01)
    assert "b" in filtered.columns
    assert "a" not in filtered.columns or filtered["a"].var() > 0.01


def test_concat_fusion_alignment():
    """concat_fusion should produce a matrix aligned on common genome IDs."""
    block1 = pd.DataFrame({"f1": [1, 2, 3]}, index=["g1", "g2", "g3"])
    block2 = pd.DataFrame({"f2": [4, 5, 6]}, index=["g1", "g2", "g3"])
    fused = concat_fusion({"kmer": block1, "domains": block2})
    assert fused.shape == (3, 2)
    assert "kmer__f1" in fused.columns
    assert "domains__f2" in fused.columns


def test_concat_fusion_handles_missing_genomes():
    """Genomes absent from one block should result in NaN rows."""
    block1 = pd.DataFrame({"f1": [1, 2, 3]}, index=["g1", "g2", "g3"])
    block2 = pd.DataFrame({"f2": [4, 5]}, index=["g1", "g2"])  # g3 missing
    fused = concat_fusion({"kmer": block1, "domains": block2})
    assert "g3" in fused.index
    assert pd.isna(fused.loc["g3", "domains__f2"])


# ── dbCAN parsing tests ───────────────────────────────────────────────────────

_OVERVIEW_TSV_OLD = (
    "Gene ID\tEC#\tHMMER\tHotpep\tDIAMOND\t#ofTools\tCAZyme\n"
    "gene1\t-\tGH5\t-\tGH5\t2\tGH5\n"
    "gene2\t-\tCBM1\t-\t-\t1\tCBM1\n"
    "gene3\t-\tGH18\tGH18\tGH18\t3\tGH18\n"
)

# dbCAN3 format: cazyme_fam column with position info and sub-families
_OVERVIEW_TSV = (
    "Gene_ID\tEC\tcazyme_fam\tsub_fam\tdiamond_fam\tSubstrate\t#ofTools\n"
    "gene1\t-\tGH5_7(54-357)\tGH5_e260\t-\tbeta-mannan\t2\n"
    "gene2\t-\tCBM1(1-50)\tCBM1_e1\t-\t-\t1\n"
    "gene3\t-\tGH18(21-300)\tGH18_e5\t-\tchitin\t3\n"
)

_SUBSTRATE_TSV = (
    "Gene_ID\tSubstrate\tScore\ngene1\tcellulose\t0.9\ngene3\tchitin\t0.8\ngene4\tchitin\t0.7\n"
)


def _write_plain(tmp_path: Path, name: str, content: str) -> Path:
    p = tmp_path / name
    p.write_text(content)
    return p


def _write_gz(tmp_path: Path, name: str, content: str) -> Path:
    p = tmp_path / name
    p.write_bytes(gzip.compress(content.encode()))
    return p


def test_parse_dbcan_output_new_format(tmp_path):
    p = _write_plain(tmp_path, "overview.tsv", _OVERVIEW_TSV)
    counts = parse_dbcan_output(p, min_tools=2)
    assert counts["GH5_7"] == 1
    assert counts["GH18"] == 1
    assert "CBM1" not in counts  # only 1 tool, filtered out


def test_parse_dbcan_output_old_format(tmp_path):
    p = _write_plain(tmp_path, "overview.tsv", _OVERVIEW_TSV_OLD)
    counts = parse_dbcan_output(p, min_tools=2)
    assert counts["GH5"] == 1
    assert counts["GH18"] == 1
    assert "CBM1" not in counts


def test_parse_dbcan_output_gz(tmp_path):
    p = _write_gz(tmp_path, "overview.tsv.gz", _OVERVIEW_TSV)
    counts = parse_dbcan_output(p, min_tools=2)
    assert counts["GH5_7"] == 1
    assert counts["GH18"] == 1
    assert "CBM1" not in counts


def test_parse_dbcan_substrate_plain(tmp_path):
    p = _write_plain(tmp_path, "substrates.tsv", _SUBSTRATE_TSV)
    counts = parse_dbcan_substrate(p)
    assert counts["substrate_chitin"] == 2
    assert counts["substrate_cellulose"] == 1


def test_parse_dbcan_substrate_gz(tmp_path):
    p = _write_gz(tmp_path, "substrates.tsv.gz", _SUBSTRATE_TSV)
    counts = parse_dbcan_substrate(p)
    assert counts["substrate_chitin"] == 2
    assert counts["substrate_cellulose"] == 1


def test_discover_dbcan_subdir_gz(tmp_path):
    """Per-genome subdir layout: dbcan/{genome_id}/overview.tsv.gz"""
    for gid in ("genomeA", "genomeB"):
        subdir = tmp_path / gid
        subdir.mkdir()
        _write_gz(subdir, "overview.tsv.gz", _OVERVIEW_TSV)
    paths = discover_annotation_files(tmp_path, suffix="overview.tsv")
    assert set(paths.keys()) == {"genomeA", "genomeB"}


def test_discover_dbcan_subdir_substrate_gz(tmp_path):
    """Per-genome subdir layout: dbcan/{genome_id}/substrates.tsv.gz"""
    for gid in ("genomeA", "genomeB"):
        subdir = tmp_path / gid
        subdir.mkdir()
        _write_gz(subdir, "substrates.tsv.gz", _SUBSTRATE_TSV)
    paths = discover_annotation_files(tmp_path, suffix="substrates.tsv")
    assert set(paths.keys()) == {"genomeA", "genomeB"}


def test_discover_dbcan_uncompressed_takes_priority(tmp_path):
    """Uncompressed file should win over .gz when both exist."""
    subdir = tmp_path / "genomeA"
    subdir.mkdir()
    plain = _write_plain(subdir, "overview.tsv", _OVERVIEW_TSV)
    _write_gz(subdir, "overview.tsv.gz", _OVERVIEW_TSV)
    paths = discover_annotation_files(tmp_path, suffix="overview.tsv")
    assert paths["genomeA"] == plain
