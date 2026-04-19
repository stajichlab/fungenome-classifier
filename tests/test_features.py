"""
tests/test_features.py

Unit tests for feature extraction modules.
"""

import pandas as pd

from fungal_classifier.features.fusion import concat_fusion, filter_low_variance
from fungal_classifier.features.kmer import _count_kmers, _obs_exp_ratio, compute_kmer_features

# ── k-mer tests ───────────────────────────────────────────────────────────────


def test_count_kmers_k1():
    seq = "AACGT"
    counts = _count_kmers(seq, k=1)
    assert counts["A"] == 2
    assert counts["C"] == 1
    assert counts["G"] == 1
    assert counts["T"] == 1


def test_count_kmers_handles_n():
    """N bases should be dropped."""
    seq = "AANCGT"
    counts = _count_kmers(seq, k=1)
    assert sum(counts.values()) == 4  # N dropped, AN and NC windows also invalid


def test_count_kmers_k2():
    seq = "ACGT"
    counts = _count_kmers(seq, k=2)
    assert counts["AC"] == 1
    assert counts["CG"] == 1
    assert counts["GT"] == 1


def test_obs_exp_ratio_uniform():
    """For uniform sequence, obs/exp should be ≈1 for all dimers."""
    seq = "ACGT" * 100
    counts_2 = _count_kmers(seq, k=2)
    counts_1 = _count_kmers(seq, k=1)
    oe = _obs_exp_ratio(counts_2, counts_1)
    for val in oe.values():
        assert 0.5 < val < 2.0, f"Unexpected obs/exp {val} for uniform sequence"


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
