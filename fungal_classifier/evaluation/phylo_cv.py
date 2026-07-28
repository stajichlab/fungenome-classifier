"""
fungal_classifier/evaluation/phylo_cv.py

Phylogeny-aware cross-validation.

Standard random CV inflates accuracy when data are phylogenetically structured:
a model can 'cheat' by learning that similar organisms have similar labels,
which is trivially true for taxonomy classification.

Solution: clade holdout CV
  - Cut the tree at a given taxonomic level (order, family, class).
  - In each fold, all members of one or more clades are held out as the test set.
  - The model never sees any close relative of a test genome during training.

Also implements:
  - PhyloSignal test (Blomberg's K) to quantify phylogenetic signal in labels.
  - Phylogenetic eigenvector features (PCoA on patristic distances) to include
    phylogenetic context as covariates in the model.
"""
__all__ = ["load_tree", "get_patristic_distances", "phylogenetic_eigenvectors", "assign_clades_from_taxonomy", "assign_clades_from_tree", "CladeHoldoutCV", "blombergs_k"]

from __future__ import annotations

import logging
from collections import defaultdict
from typing import Iterator

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)


# ── tree utilities ────────────────────────────────────────────────────────────


def load_tree(newick_path: str):
    """Load a phylogenetic tree from a Newick file using ete3."""
    try:
        from ete3 import Tree

        return Tree(newick_path, format=1)
    except ImportError:
        import dendropy

        return dendropy.Tree.get(path=newick_path, schema="newick")


def get_patristic_distances(tree, genome_ids: list[str]) -> pd.DataFrame:
    """
    Compute pairwise patristic (branch-length) distances between tips.

    Returns
    -------
    pd.DataFrame of shape (n, n) with genome_ids as index and columns.
    """
    try:
        tips = {leaf.name: leaf for leaf in tree.iter_leaves()}
        n = len(genome_ids)
        D = np.zeros((n, n))
        for i, gid_i in enumerate(genome_ids):
            for j, gid_j in enumerate(genome_ids[i + 1 :], start=i + 1):
                if gid_i in tips and gid_j in tips:
                    dist = tree.get_distance(tips[gid_i], tips[gid_j])
                    D[i, j] = D[j, i] = dist
        return pd.DataFrame(D, index=genome_ids, columns=genome_ids)
    except ImportError:
        pdm = tree.phylogenetic_distance_matrix()
        taxa = {t.label: t for t in tree.taxon_namespace}
        n = len(genome_ids)
        D = np.zeros((n, n))
        for i, gid_i in enumerate(genome_ids):
            for j, gid_j in enumerate(genome_ids[i + 1 :], start=i + 1):
                if gid_i in taxa and gid_j in taxa:
                    d = pdm(taxa[gid_i], taxa[gid_j])
                    D[i, j] = D[j, i] = d
        return pd.DataFrame(D, index=genome_ids, columns=genome_ids)


def phylogenetic_eigenvectors(
    distance_matrix: pd.DataFrame,
    n_components: int = 20,
) -> pd.DataFrame:
    """
    Compute phylogenetic eigenvectors via PCoA (Principal Coordinates Analysis)
    on the patristic distance matrix.

    These can be included as covariates to control for phylogenetic structure.

    Returns
    -------
    pd.DataFrame of shape (n_genomes, n_components).
    """
    from sklearn.manifold import MDS

    D = distance_matrix.values
    genome_ids = distance_matrix.index.tolist()

    mds = MDS(
        n_components=n_components,
        dissimilarity="precomputed",
        random_state=42,
        normalized_stress=False,
    )
    coords = mds.fit_transform(D)
    cols = [f"phylo_pc{i + 1}" for i in range(n_components)]
    return pd.DataFrame(coords, index=genome_ids, columns=cols, dtype=np.float32)


# ── clade assignment ──────────────────────────────────────────────────────────


def assign_clades_from_taxonomy(
    metadata: pd.DataFrame,
    clade_level: str = "order",
) -> pd.Series:
    """
    Assign each genome to a clade based on taxonomy metadata.

    Parameters
    ----------
    metadata    : DataFrame with genome_id index and taxonomy columns.
    clade_level : Taxonomic level to use for clade assignment
                  (phylum | class | order | family | genus).

    Returns
    -------
    pd.Series mapping genome_id -> clade label.
    """
    col = f"taxonomy_{clade_level}"
    if col not in metadata.columns:
        raise ValueError(
            f"Column '{col}' not found in metadata. Available: {list(metadata.columns)}"
        )
    return metadata[col].fillna("Unknown")


def assign_clades_from_tree(
    tree,
    genome_ids: list[str],
    n_clades: int = 10,
) -> pd.Series:
    """
    Assign genomes to clades by cutting the phylogenetic tree into n_clades
    subtrees using the top internal nodes.

    Returns
    -------
    pd.Series mapping genome_id -> clade_id (integer).
    """
    try:
        # Collect internal nodes sorted by number of leaves
        internal_nodes = sorted(
            [n for n in tree.traverse() if not n.is_leaf()],
            key=lambda n: len(n.get_leaves()),
            reverse=True,
        )
        # Pick top n_clades nodes as clade roots
        clade_assignments: dict[str, int] = {}
        for clade_id, node in enumerate(internal_nodes[:n_clades]):
            for leaf in node.get_leaves():
                if leaf.name not in clade_assignments:
                    clade_assignments[leaf.name] = clade_id
        # Assign unassigned genomes to clade -1
        for gid in genome_ids:
            if gid not in clade_assignments:
                clade_assignments[gid] = -1
        return pd.Series(clade_assignments)
    except ImportError:
        raise ImportError(
            "ete3 is required for tree-based clade assignment. Install with: pip install ete3"
        )


# ── cross-validation splitter ─────────────────────────────────────────────────


class CladeHoldoutCV:
    """
    Scikit-learn compatible cross-validator that holds out entire clades.

    Each fold holds out one or more clades as the test set, ensuring
    no close relatives appear in both train and test.

    Parameters
    ----------
    clade_labels : pd.Series mapping genome_id -> clade label.
    n_folds      : Number of CV folds (clades are distributed across folds).
    random_seed  : For reproducibility.

    Usage
    -----
    cv = CladeHoldoutCV(clade_labels=clade_series, n_folds=10)
    for train_idx, test_idx in cv.split(X, y):
        ...
    """

    def __init__(
        self,
        clade_labels: pd.Series,
        n_folds: int = 10,
        random_seed: int = 42,
    ):
        self.clade_labels = clade_labels
        self.n_folds = n_folds
        self.random_seed = random_seed
        self._fold_assignments: dict[str, int] = {}
        self._build_fold_assignments()

    def _build_fold_assignments(self) -> None:
        """Distribute clades across folds (approximately equal genome count)."""
        rng = np.random.default_rng(self.random_seed)
        # Sort by size descending, then distribute round-robin for balance
        clade_sizes = self.clade_labels.value_counts()
        sorted_clades = clade_sizes.index.tolist()
        rng.shuffle(sorted_clades)

        fold_sizes = defaultdict(int)
        clade_to_fold: dict[str, int] = {}

        for clade in sorted_clades:
            # Assign to smallest current fold
            fold = min(range(self.n_folds), key=lambda f: fold_sizes[f])
            clade_to_fold[clade] = fold
            fold_sizes[fold] += clade_sizes[clade]

        self._clade_to_fold = clade_to_fold

    def split(
        self,
        X: pd.DataFrame,
        y: pd.Series | None = None,
        groups=None,
    ) -> Iterator[tuple[np.ndarray, np.ndarray]]:
        """
        Generate (train_indices, test_indices) pairs.

        Parameters
        ----------
        X : Feature matrix with genome_id index.

        Yields
        ------
        train_idx, test_idx : numpy integer arrays.
        """
        genome_ids = X.index.tolist()
        id_to_pos = {gid: i for i, gid in enumerate(genome_ids)}

        for fold in range(self.n_folds):
            test_genomes = [
                gid
                for gid in genome_ids
                if self._clade_to_fold.get(self.clade_labels.get(gid, "Unknown"), -1) == fold
            ]
            train_genomes = [gid for gid in genome_ids if gid not in set(test_genomes)]

            test_idx = np.array([id_to_pos[gid] for gid in test_genomes if gid in id_to_pos])
            train_idx = np.array([id_to_pos[gid] for gid in train_genomes if gid in id_to_pos])

            if len(test_idx) == 0:
                logger.warning(f"Fold {fold} has no test samples — skipping.")
                continue

            logger.info(f"Fold {fold}: {len(train_idx)} train, {len(test_idx)} test genomes")
            yield train_idx, test_idx

    def get_n_splits(self, X=None, y=None, groups=None) -> int:
        return self.n_folds

    def fold_summary(self) -> pd.DataFrame:
        """Return a summary DataFrame of clades and their fold assignments."""
        rows = [
            {"clade": clade, "fold": fold, "n_genomes": (self.clade_labels == clade).sum()}
            for clade, fold in self._clade_to_fold.items()
        ]
        return pd.DataFrame(rows).sort_values(["fold", "clade"])


# ── phylogenetic signal test ──────────────────────────────────────────────────


def blombergs_k(
    trait: pd.Series,
    tree = None,
    /,
    *,
    distance_matrix: pd.DataFrame | None = None,
    random_seed: int = 42,
) -> float:
    """
    Estimate Blomberg's K statistic for phylogenetic signal in a continuous trait.

    K > 1: more phylogenetic signal than expected under Brownian motion.
    K < 1: less signal (convergence or homoplasy).
    K ≈ 0: no phylogenetic signal.

    Uses phylogenetic independent contrasts (Felsenstein 1985) via ape.pic
    to compute the expected variance under a pure Brownian-motion model,
    then compares it to the observed trait variance.

    K = Var(trait) / Σ(contrasts_i²)

    Parameters
    ----------
    trait          : Continuous trait values, indexed by tip label (genome_id).
    tree           : An ete3 Tree or dendropy Tree. Tip labels must match trait index.
    distance_matrix : **Deprecated.** Ignored with a warning. Will be removed in a
                      future version. Pass tree instead.
    random_seed    : Seed passed to np.random.default_rng (reserved for future use).

    Returns
    -------
    float: Blomberg's K, or np.nan if fewer than 4 taxa are common.
    """
    if distance_matrix is not None:
        import warnings
        warnings.warn(
            "distance_matrix argument is deprecated and has no effect. "
            "blombergs_k now requires a tree object (ete3 or dendropy). "
            "The previous placeholder implementation always returned a near-zero K. "
            "Pass tree=<ete3.Tree> or tree=<dendropy.Tree> instead.",
            DeprecationWarning,
            stacklevel=2,
        )

    if tree is None:
        raise TypeError(
            "blombergs_k() requires 'tree' as a positional argument (ete3 or dendropy Tree). "
            "The distance_matrix argument is no longer supported."
        )

    if distance_matrix is not None:
        import warnings
        warnings.warn(
            "distance_matrix argument is deprecated and has no effect. "
            "blombergs_k now requires a tree object (ete3 or dendropy). "
            "Pass tree=<ete3.Tree> or tree=<dendropy.Tree> instead.",
            DeprecationWarning,
            stacklevel=2,
        )

    try:
        import ape
    except ImportError:
        raise ImportError(
            "ape is required for blombergs_k: pip install ape\n"
            "(also requires dendropy for some tree formats)"
        )

    # ── collect tip names from the tree ────────────────────────────────────
    try:
        # ete3 path
        tip_names = [leaf.name for leaf in tree.iter_leaves()]
    except AttributeError:
        # dendropy path
        tip_names = [str(t.label) for t in tree.taxon_namespace]

    common = trait.index.intersection(tip_names)
    if len(common) < 4:
        logger.warning(
            f"blombergs_k: only {len(common)} matching tips — need ≥4 for reliable estimate. "
            f"Returning NaN."
        )
        return np.nan

    trait_aligned = trait.loc[list(common)].astype(float)
    y = trait_aligned.values

    # ── prune tree to common tips and serialise to Newick ──────────────────
    try:
        # ete3 path
        sub_tree = tree.copy("newick")
        for leaf in sub_tree.iter_leaves():
            if leaf.name not in common:
                leaf.delete(prevent_nested=True)
        newick_str = sub_tree.write(format=1)
    except (AttributeError, TypeError):
        # dendropy path — only catch AttributeError/TypeError from ete3 operations
        import dendropy
        taxa_to_keep = [
            tree.taxon_namespace.get_taxon(gid)
            for gid in common
            if tree.taxon_namespace.get_taxon(gid) is not None
        ]
        pruned = tree.extract_tree_with_taxa(taxa_to_keep)
        newick_str = pruned.as_string(schema="newick").strip()

    # ── load into ape ───────────────────────────────────────────────────────
    try:
        phy = ape.read(newick=newick_str)
    except Exception as exc:
        raise ValueError(
            f"blombergs_k: failed to parse pruned tree as Newick: {exc}. "
            f"Ensure the tree is valid Newick format with named tips."
        ) from exc

    # ── align trait vector to ape tip order ────────────────────────────────
    ape_tips = [str(taxon.label) for taxon in phy.taxon_set]
    # y_in_ape_order must match ape tip order
    try:
        y_in_ape_order = np.array([float(trait_aligned.loc[tip]) for tip in ape_tips])
    except KeyError as exc:
        raise ValueError(
            f"blombergs_k: ape tree tips {ape_tips} do not match trait index. "
            f"Missing: {exc}"
        ) from exc

    # ── compute phylogenetic independent contrasts via ape.pic ────────────
    try:
        pic_result = ape.pic(y_in_ape_order, phy)
        pic_vals = np.asarray(pic_result).flatten()
    except Exception as exc:
        raise RuntimeError(
            f"blombergs_k: ape.pic computation failed: {exc}. "
            f"Ensure the tree is rooted and branch lengths are non-negative."
        ) from exc

    n_pic = len(pic_vals)
    if n_pic == 0:
        return np.nan

    # Blomberg's K = observed trait variance / sum of squared contrasts
    # Under Brownian motion, Σ(contrast²) is the MLE of the BM rate σ²
    sum_sq_contrasts = max(float(np.sum(pic_vals ** 2)), 1e-12)
    var_trait = float(np.var(y, ddof=1))

    return var_trait / sum_sq_contrasts
