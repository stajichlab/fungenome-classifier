# Changelog

All notable changes to FungalClassifier are documented here.
Format follows [Keep a Changelog](https://keepachangelog.com/en/1.0.0/).

---

## [Unreleased]

### Planned
- Secretome feature block (SignalP + GPI anchor composition)
- Multi-label classification for genomes spanning multiple niches
- Active learning module for sequencing prioritisation
- NCBI datasets API integration for automated annotation download

---

## [0.1.0] — Initial release

### Added

**Feature blocks**
- `features/kmer.py` — k-mer (k=1–6) and oligonucleotide composition; obs/expected ratio normalisation for base composition correction
- `features/domains.py` — Pfam/InterPro domain copy-number vectors from hmmer domtblout and InterProScan TSV
- `features/pathways.py` — KEGG ortholog counts, CAZyme family profiles (dbCAN), biosynthetic gene cluster types (antiSMASH), GO slim aggregation
- `features/repeats.py` — RepeatMasker TE class proportions, mean divergence (repeat landscape), genome-size normalisation
- `features/motifs.py` — JASPAR fungi PWM scanning via FIMO; upstream promoter extraction via bedtools; per-genome motif enrichment fractions
- `features/fusion.py` — Variance filtering, univariate selection, StandardScaler/SVD per block; concat and stacking fusion strategies; `BlockFusionPipeline` class

**Models**
- `models/block_classifier.py` — XGBoost, LightGBM, RandomForest block classifiers with out-of-fold probability output
- `models/fusion_model.py` — Stacking meta-learner (logistic regression or XGBoost) trained on OOF block probabilities
- `models/deep_fusion.py` — PyTorch multi-modal classifier: per-block `BlockTower` embeddings, soft `BlockAttention` fusion, auxiliary classification heads, `DeepFusionTrainer` with early stopping and cosine LR scheduling

**Evaluation**
- `evaluation/phylo_cv.py` — `CladeHoldoutCV` (scikit-learn compatible), taxonomy and tree-based clade assignment, patristic distance computation, phylogenetic eigenvectors (PCoA), Blomberg's K phylogenetic signal test
- `evaluation/metrics.py` — Accuracy, balanced accuracy, F1 macro/weighted, precision, recall, MCC; confusion matrix plots; CV summary with 95% CIs; block comparison tables
- `evaluation/shap_analysis.py` — TreeExplainer SHAP values, mean |SHAP| global importance, per-class SHAP summaries, block-level attribution, summary/heatmap/force plots
- `evaluation/embeddings.py` — PCA, UMAP, t-SNE embeddings per feature block; deep tower embedding extraction; faceted scatter plots; TSV export for iTOL/R

**Utilities**
- `utils/io.py` — Parquet/HDF5 feature matrix I/O, genome/annotation file discovery, metadata loading, prediction saving
- `utils/preprocessing.py` — log1p, CLR transforms, median/KNN imputation, SMOTE, class weights, label encoding with rare-class collapsing, genome-size correction
- `utils/phylo.py` — Tree pruning, patristic distance caching, taxonomy string parsing, clade member lookup

**Scripts**
- `01_build_features.py` — Full feature matrix construction for all blocks
- `02_train.py` — Block classifiers + stacking fusion; deep fusion model with --model-type deep
- `03_evaluate.py` — Confusion matrices, per-class metrics, Blomberg's K
- `04_predict.py` — Prediction on new genomes
- `05_compare_targets.py` — Multi-target block × target heatmap and performance comparison
- `06_export_embeddings.py` — PCA/UMAP/t-SNE embedding export for all blocks

**Notebooks**
- `01_feature_exploration.ipynb` — Sparsity heatmaps, PCA/UMAP, class balance, inter-block correlation
- `02_phylo_signal_analysis.ipynb` — Blomberg's K, clade holdout visualisation, random-vs-clade CV comparison, PCoA eigenvectors
- `03_shap_interpretation.ipynb` — Global and per-class SHAP, block attribution, attention weights, force plots

**Configuration**
- `configs/default.yaml` — XGBoost + stacking pipeline
- `configs/deep_fusion.yaml` — PyTorch attention fusion pipeline

**Documentation**
- `docs/data_preparation.md` — Annotation pipeline, directory structure, QC, batch processing guide
- `docs/feature_engineering.md` — Biological rationale per feature block
- `docs/phylo_cv.md` — Clade holdout CV methodology, Blomberg's K, eigenvectors
- `docs/model_architecture.md` — Block classifier + stacking + deep fusion architecture, SHAP pipeline

**Infrastructure**
- `Makefile` — Pipeline orchestration targets
- `environment.yml` — Conda environment with all dependencies
- `pyproject.toml` — Package configuration
- Tests: unit tests for features, models, preprocessing, phylo CV, and full integration smoke tests
