# Model Architecture

## Overview

FungalClassifier uses a two-stage architecture: independent per-block classifiers followed by a fusion model that combines their outputs.

```
┌──────────────┐  ┌──────────────┐  ┌──────────────┐  ┌──────────────┐  ┌──────────────┐
│  k-mer block │  │ domain block │  │pathway block │  │ repeat block │  │ motif block  │
│   XGBoost    │  │   XGBoost    │  │   XGBoost    │  │   XGBoost    │  │   XGBoost    │
└──────┬───────┘  └──────┬───────┘  └──────┬───────┘  └──────┬───────┘  └──────┬───────┘
       │ P(class)        │ P(class)        │ P(class)        │ P(class)        │ P(class)
       └─────────────────┴─────────────────┴─────────────────┴─────────────────┘
                                           │
                                    ┌──────▼──────┐
                                    │  Stacking   │
                                    │  meta-learner│
                                    │  (LogReg /  │
                                    │   XGBoost)  │
                                    └──────┬──────┘
                                           │
                                    ┌──────▼──────┐
                                    │ Prediction  │
                                    └─────────────┘
```

## Stage 1: Block Classifiers

Each feature block gets its own independent gradient boosted classifier (XGBoost by default, LightGBM optionally). Training is done with clade-holdout cross-validation to produce out-of-fold (OOF) probability vectors without leakage.

**Why separate classifiers?**
- Allows direct comparison of predictive power per feature type
- Enables SHAP analysis at the block level
- Missing annotation data for one block doesn't break the whole pipeline
- Provides natural interpretability: "The domain block alone achieves 0.82 F1 for order-level classification, while the k-mer block achieves 0.71"

**XGBoost settings:**
- `max_depth=6`, `n_estimators=500`, `learning_rate=0.05`
- Early stopping against a validation fold (last clade holdout fold)
- Multiclass: `objective=multi:softprob` produces proper probability estimates

## Stage 2: Stacking Fusion

The meta-learner receives as input the **OOF probability vectors** from all block classifiers concatenated. For 5 blocks and 20 classes, this is a 100-dimensional meta-feature vector.

```
meta_X = [P_kmer(c1..c20) | P_domains(c1..c20) | P_pathways(c1..c20) | P_repeats(c1..c20) | P_motifs(c1..c20)]
```

The meta-learner (logistic regression) learns which blocks to trust for which classes. For example, it might learn that domain-block predictions are highly reliable for Ascomycota classes but that k-mer predictions add signal for Basidiomycota.

**Using OOF probabilities is critical.** If you train block classifiers on the full training set and then use their in-sample predictions as meta-features, the meta-learner sees overfit block predictions and will overweight them. OOF probabilities produced during clade-holdout CV are the correct inputs.

---

## Deep Fusion Alternative

For the deep model, the architecture replaces stacking with learned multi-modal embedding:

```
Block inputs
    │
    ├── kmer (dim=150) ──► Tower(256→128) ──► embedding_kmer    ┐
    ├── domains (dim=150) ► Tower(256→128) ──► embedding_domains │
    ├── pathways (dim=150)► Tower(256→128) ──► embedding_pathways├──► Attention ──► fused(128) ──► Head ──► P(class)
    ├── repeats (dim=150) ► Tower(256→128) ──► embedding_repeats │
    └── motifs (dim=150) ─► Tower(256→128) ──► embedding_motifs  ┘
                                                      │
                                              Auxiliary heads (per block)
                                              ──► P_aux(class)
```

**Block Towers** compress each feature block from its input dimension (after SVD) into a shared 128-dimensional embedding space via two linear layers with BatchNorm, ReLU, and Dropout.

**Block Attention** computes a soft attention weight per block using a linear layer over the concatenated embeddings, then produces an attention-weighted sum. This lets the model learn, per sample, which feature types to rely on most.

**Auxiliary heads** apply a linear classifier to each block embedding independently and contribute an auxiliary cross-entropy loss (weighted 0.2 × main loss). This forces each tower to learn a meaningful embedding even if one block dominates the main head.

---

## Phylogenetic Correction

Two approaches are available and can be combined:

### Approach 1: Clade Holdout CV (evaluation-time correction)
Ensures test sets never contain close relatives of training genomes. This doesn't modify the model but gives unbiased accuracy estimates.

### Approach 2: Phylogenetic Eigenvectors (model-time correction)
Adds the first 20 PCoA eigenvectors of the patristic distance matrix as an additional feature block (`phylo_eigenvectors`). This allows the model to explicitly represent phylogenetic relatedness, separating genomic features that co-vary with phylogeny from those that explain ecological niche independently.

**When to use eigenvectors:** For ecological niche / lifestyle prediction where you want to understand "what predicts ecology beyond phylogeny." Not needed for taxonomy classification (where phylogeny is the signal of interest).

---

## Handling Class Imbalance

Fungal genome datasets often have highly imbalanced classes (many Aspergillus, few Talaromyces). Options:

1. **Class weights** (default): pass `scale_pos_weight` (binary) or `sample_weight` (multiclass) proportional to inverse class frequency. Set automatically from `compute_class_weights()`.

2. **SMOTE oversampling**: use `apply_smote()` on the training fold only. Appropriate when minority classes have <50 samples.

3. **Evaluation metric**: use `balanced_accuracy` or `f1_macro` rather than raw accuracy to avoid misleading metrics.

---

## SHAP Analysis Pipeline

After training, SHAP values are computed using `shap.TreeExplainer` (fast exact computation for tree models).

For multiclass XGBoost, SHAP returns a list of `(n_samples, n_features)` arrays — one per class. The mean |SHAP| aggregation averages across classes to give global importance.

**Block attribution:** When using concatenated fusion, feature names are prefixed (`kmer__`, `domains__` etc.), allowing SHAP values to be summed within each block prefix to produce a block-level attribution score. This answers: "What fraction of total predictive information comes from sequence composition vs. domain repertoire?"

**Caveats:**
- SHAP values are model-specific: TreeSHAP for gradient boosting, LinearSHAP for logistic regression meta-learner
- For the deep model, use `shap.DeepExplainer` or `shap.GradientExplainer`
- SHAP interaction values (pairwise) are computationally expensive at n=6000 but can be computed for subsets

---

## Output Files

After training, `results/` contains:

```
results/
├── models/
│   ├── block_kmer.pkl
│   ├── block_domains.pkl
│   ├── block_pathways.pkl
│   ├── block_repeats.pkl
│   ├── block_motifs.pkl
│   └── fusion_model.pkl
├── shap/
│   ├── kmer_mean_abs_shap.csv
│   ├── domains_mean_abs_shap.csv
│   ├── domains_per_class_shap.csv
│   ├── domains_shap_summary.svg
│   └── ...
├── block_comparison.csv       ← per-block CV F1 scores
├── cv_fold_summary.csv        ← clade assignment per fold
├── training_summary.json
└── evaluation/
    ├── confusion_matrix.svg
    └── phylo_signal.txt
```
