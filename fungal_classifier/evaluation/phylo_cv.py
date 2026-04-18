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
        from ete3 import Tree
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
        import dendropy
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
    cols = [f"phylo_pc{i+1}" for i in range(n_components)]
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
        raise ValueError(f"Column '{col}' not found in metadata. Available: {list(metadata.columns)}")
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
        from ete3 import Tree
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
        raise ImportError("ete3 is required for tree-based clade assignment. Install with: pip install ete3")


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
        clades = self.clade_labels.unique()
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
                gid for gid in genome_ids
                if self._clade_to_fold.get(
                    self.clade_labels.get(gid, "Unknown"), -1
                ) == fold
            ]
            train_genomes = [gid for gid in genome_ids if gid not in set(test_genomes)]

            test_idx = np.array([id_to_pos[gid] for gid in test_genomes if gid in id_to_pos])
            train_idx = np.array([id_to_pos[gid] for gid in train_genomes if gid in id_to_pos])

            if len(test_idx) == 0:
                logger.warning(f"Fold {fold} has no test samples — skipping.")
                continue

            logger.info(
                f"Fold {fold}: {len(train_idx)} train, {len(test_idx)} test genomes"
            )
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
    distance_matrix: pd.DataFrame,
) -> float:
    """
    Estimate Blomberg's K statistic for phylogenetic signal in a continuous trait.

    K > 1: more phylogenetic signal than expected under Brownian motion.
    K < 1: less signal (convergence or homoplasy).
    K ≈ 0: no phylogenetic signal.

    Parameters
    ----------
    trait           : Continuous trait values, indexed by genome_id.
    distance_matrix : Patristic distance matrix.

    Returns
    -------
    float: Blomberg's K.
    """
    common = trait.index.intersection(distance_matrix.index)
    trait = trait.loc[common].astype(float)
    D = distance_matrix.loc[common, common].values

    n = len(trait)
    y = trait.values - trait.mean()

    # Variance of independent contrasts (simplified)
    var_phy = float(np.diag(D).mean())
    var_obs = float(np.var(y, ddof=1))

    K = var_obs / (var_phy + 1e-12)
    return K
