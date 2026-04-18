# Phylogeny-Aware Cross-Validation

## The Problem: Phylogenetic Leakage

When classifying fungi by taxonomy or ecological niche, genomic features are not independent across genomes — closely related organisms share similar features by descent. Standard random cross-validation does not account for this, and the resulting accuracy estimates are inflated.

**Example:** If *Aspergillus fumigatus* ends up in the test set, and *Aspergillus niger* is in the training set, the model has already "seen" an organism with essentially the same features. This inflates accuracy estimates compared to what you'd observe on a truly novel genus.

This is sometimes called **phylogenetic leakage** or **clade leakage**.

---

## The Solution: Clade Holdout CV

In clade holdout cross-validation, entire clades (groups of related organisms on the phylogenetic tree) are held out as test sets. This ensures that no close relative of a test genome was seen during training.

### How it works

1. Assign each genome to a clade based on either:
   - **Taxonomy** (e.g., all genomes in the same order form a clade)
   - **Tree-based partitioning** (cut the tree into N subtrees)

2. Distribute clades across K folds (balancing by genome count)

3. In each fold, all genomes from held-out clades are the test set; all others are training

### Implementation

```python
from fungal_classifier.evaluation.phylo_cv import (
    CladeHoldoutCV, assign_clades_from_taxonomy, assign_clades_from_tree
)

# Option 1: taxonomy-based clade assignment
clade_labels = assign_clades_from_taxonomy(metadata, clade_level="order")

# Option 2: tree-based assignment
tree = load_tree("data/raw/phylogeny.nwk")
clade_labels = assign_clades_from_tree(tree, genome_ids, n_clades=30)

# Create CV splitter (scikit-learn compatible)
cv = CladeHoldoutCV(clade_labels=clade_labels, n_folds=10)

# Use in sklearn cross_val_score
from sklearn.model_selection import cross_val_score
scores = cross_val_score(model, X, y, cv=cv, scoring="f1_macro")
```

---

## Choosing the Right Clade Level

| Clade Level | Use When |
|---|---|
| `order` | Classifying at phylum/class level; large dataset (>2000 genomes) |
| `family` | Classifying at order/family level; moderate dataset |
| `genus` | Most stringent; tests generalization to new genera |

A rough guideline: the clade holdout level should be **one level above** the target classification level. If predicting ecological niche (which has less clear phylogenetic signal), use `family`-level holdout.

---

## Quantifying Phylogenetic Signal

Before choosing a CV strategy, it's useful to know how much phylogenetic signal exists in your labels using **Blomberg's K** statistic.

```python
from fungal_classifier.evaluation.phylo_cv import (
    get_patristic_distances, blombergs_k
)
from sklearn.preprocessing import LabelEncoder

D = get_patristic_distances(tree, genome_ids)
le = LabelEncoder()
y_enc = pd.Series(le.fit_transform(y), index=y.index)
K = blombergs_k(y_enc, D)
print(f"Blomberg's K = {K:.3f}")
```

**Interpreting K:**

| K value | Interpretation | CV recommendation |
|---|---|---|
| K >> 1 | Strong phylogenetic signal | Use clade holdout at high level |
| K ≈ 1 | Signal consistent with Brownian motion | Standard clade holdout |
| K < 1 | Weak signal (convergence/homoplasy) | Family-level holdout may suffice |
| K ≈ 0 | No phylogenetic signal | Random stratified CV acceptable |

---

## Phylogenetic Eigenvectors as Covariates

Even with clade holdout CV, you may want to include phylogenetic context in your model — particularly for ecological niche prediction where the trait may show moderate phylogenetic signal.

Phylogenetic eigenvectors are computed via PCoA (Principal Coordinates Analysis) on the patristic distance matrix. Including them as additional features lets the model learn how much phylogeny explains the labels.

```python
from fungal_classifier.evaluation.phylo_cv import (
    get_patristic_distances, phylogenetic_eigenvectors
)

D = get_patristic_distances(tree, genome_ids)
phylo_features = phylogenetic_eigenvectors(D, n_components=20)
# Add as a feature block
feature_blocks["phylo_eigenvectors"] = phylo_features
```

**When to use this:**
- When you want to separate "what is explained by phylogeny" from "what is explained by ecology"
- When building a classifier for ecological niche and want to avoid falsely attributing phylogenetic signal to functional features
- Not recommended when phylogenetic leakage is already well-controlled by clade holdout

---

## Comparison: CV Strategies

| Strategy | Accuracy Inflation | Recommended Use |
|---|---|---|
| Random CV | High | Never for phylogenetically structured data |
| Stratified CV | Moderate–High | Only if K ≈ 0 |
| Clade holdout (family) | Low | Default recommendation |
| Clade holdout (order) | Very low | Strict generalization testing |
| Leave-one-clade-out | Minimal | Small datasets, conservative estimate |

---

## References

- Blomberg, S.P., Garland, T. & Ives, A.R. (2003). Testing for phylogenetic signal in comparative data. *Evolution* 57(4): 717–745.
- Valdar, W. et al. (2006). Genome-wide genetic association of complex traits in heterogeneous stock mice. *Nature Genetics* 38(8): 879–887.
- Washburne, A.D. et al. (2018). Phylogenetic factorization of compositional data yields lineage-level associations in microbiome datasets. *PeerJ* 6: e2969.
