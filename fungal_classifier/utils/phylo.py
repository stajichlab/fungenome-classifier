"""
fungal_classifier/utils/phylo.py

Phylogenetic utility functions:
  - Tree pruning to genome set
  - Ancestral state reconstruction helpers
  - Taxonomic lineage parsing
  - Tree-to-distance matrix conversion caching
"""

from __future__ import annotations

import logging
from pathlib import Path

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)


def prune_tree_to_genomes(tree, genome_ids: list[str]):
    """
    Prune tree to only include tips present in genome_ids.

    Works with both ete3 and dendropy Tree objects.
    Returns pruned tree of the same type.
    """
    try:
        from ete3 import Tree
        if isinstance(tree, Tree):
            tree = tree.copy()
            tip_names = {l.name for l in tree.iter_leaves()}
            to_keep = set(genome_ids) & tip_names
            to_prune = tip_names - to_keep
            if to_prune:
                tree.prune(list(to_keep), preserve_branch_length=True)
            logger.info(f"Pruned tree: {len(to_keep)} tips retained, {len(to_prune)} removed")
            return tree
    except ImportError:
        pass

    try:
        import dendropy
        if isinstance(tree, dendropy.Tree):
            taxa_to_keep = {
                t for t in tree.taxon_namespace if t.label in set(genome_ids)
            }
            tree.retain_taxa(taxa_to_keep)
            return tree
    except ImportError:
        pass

    raise TypeError(f"Unrecognized tree type: {type(tree)}")


def get_tree_tip_names(tree) -> list[str]:
    """Return list of tip names from ete3 or dendropy tree."""
    try:
        from ete3 import Tree
        if isinstance(tree, Tree):
            return [l.name for l in tree.iter_leaves()]
    except ImportError:
        pass
    try:
        import dendropy
        if isinstance(tree, dendropy.Tree):
            return [t.label for t in tree.taxon_namespace]
    except ImportError:
        pass
    raise TypeError(f"Unrecognized tree type: {type(tree)}")


def cache_distance_matrix(
    tree,
    genome_ids: list[str],
    cache_path: Path,
) -> pd.DataFrame:
    """
    Compute and cache patristic distance matrix as Parquet.

    Patristic distance computation is O(n^2) and slow for n>2000.
    Caching avoids recomputation across notebook/script runs.
    """
    from fungal_classifier.evaluation.phylo_cv import get_patristic_distances

    cache_path = Path(cache_path)
    if cache_path.exists():
        logger.info(f"Loading cached distance matrix from {cache_path}")
        return pd.read_parquet(cache_path)

    logger.info(f"Computing patristic distance matrix for {len(genome_ids)} genomes...")
    D = get_patristic_distances(tree, genome_ids)
    cache_path.parent.mkdir(parents=True, exist_ok=True)
    D.to_parquet(cache_path)
    logger.info(f"Cached to {cache_path}")
    return D


def parse_taxonomy_string(
    taxonomy_str: str,
    delimiter: str = ";",
    levels: list[str] | None = None,
) -> dict[str, str]:
    """
    Parse a delimited taxonomy string into a dict of level -> name.

    Example:
        "k__Fungi;p__Ascomycota;c__Sordariomycetes;o__Hypocreales"
        -> {"kingdom": "Fungi", "phylum": "Ascomycota", ...}
    """
    if levels is None:
        levels = ["kingdom", "phylum", "class", "order", "family", "genus", "species"]

    parts = [p.strip() for p in taxonomy_str.split(delimiter) if p.strip()]
    result: dict[str, str] = {}
    for i, part in enumerate(parts):
        level = levels[i] if i < len(levels) else f"level_{i}"
        # Strip common prefixes like k__, p__, c__, o__, f__, g__, s__
        if len(part) >= 3 and part[1:3] == "__":
            result[level] = part[3:]
        else:
            result[level] = part
    return result


def expand_taxonomy_column(
    metadata: pd.DataFrame,
    taxonomy_col: str = "taxonomy_string",
    delimiter: str = ";",
) -> pd.DataFrame:
    """
    Expand a delimited taxonomy string column into individual level columns.

    Returns metadata with new columns: taxonomy_kingdom, taxonomy_phylum, etc.
    """
    parsed = metadata[taxonomy_col].apply(
        lambda x: parse_taxonomy_string(str(x), delimiter) if pd.notna(x) else {}
    )
    tax_df = pd.DataFrame(list(parsed), index=metadata.index)
    tax_df.columns = [f"taxonomy_{c}" for c in tax_df.columns]
    return pd.concat([metadata, tax_df], axis=1)


def get_clade_members(tree, clade_name: str, metadata: pd.DataFrame, level: str = "order") -> list[str]:
    """
    Return genome IDs belonging to a named clade based on taxonomy metadata.
    """
    col = f"taxonomy_{level}"
    if col not in metadata.columns:
        raise ValueError(f"Column {col} not found in metadata")
    return metadata[metadata[col] == clade_name].index.tolist()
