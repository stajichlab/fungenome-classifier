# Changelog

All notable changes to FungalClassifier are documented here.
Format follows [Keep a Changelog](https://keepachangelog.com/en/1.0.0/).

---

## [Unreleased]

### Added

**New feature blocks**
- `features/subcellular.py` — protein subcellular localisation and membrane topology features combining four predictors:
  - TMHMM2 short format: transmembrane helix count histogram, fraction of TM proteins, mean ExpAA
  - SignalP-6: fraction of proteins with signal peptide, mean SP probability
  - WolfPSORT: fraction and mean kNN score per predicted compartment (nucl, mito, cyto, extr, plas, E.R., golg, vacu, cysk, pero, cyto\_nucl, cyto\_pero)
  - TargetP-2: fraction noTP/SP/mTP, mean SP and mTP probabilities
  - `build_subcellular_matrix()` concatenates all available sub-blocks into one matrix
- `features/disorder.py` — intrinsic protein disorder features from AIUPred output:
  - Per-genome mean and median residue disorder score
  - Fraction of disordered residues (threshold 0.5)
  - Fraction of proteins containing any IDR / a long IDR (≥10 consecutive disordered residues)
  - Mean per-protein disorder score
- `features/proteases.py` — protease repertoire features from MEROPS BLAST tabular results:
  - Family-level copy numbers (S01, A01, C19, …)
  - Clan-level copy numbers (SA, CA, MA, …)
  - Total protease fraction of proteome
  - Filters by identity, alignment length, and e-value; rare families dropped by `min_genome_freq`
- `features/composition.py` — codon usage, amino acid frequency, and gene count features:
  - 61 sense-codon relative frequencies
  - 20 amino acid relative frequencies (derived from codons or directly from protein FASTA)
  - GC content at each codon position (GC1, GC2, GC3) and overall coding GC
  - Gene count (number of CDS/protein records)
  - Accepts pre-computed `.codon_freq.csv` files (fast) or computes directly from CDS/protein FASTA

**New utility**
- `utils/io.py`: `load_taxonomy()` — loads `annotations/taxonomy/samples.csv` (ASMID, SPECIESIN, STRAIN, BIOPROJECT, NCBI\_TAXONID, BUSCO\_LINEAGE, PHYLUM–SPECIES hierarchy, LOCUSTAG); indexed by assembly ID with standardised lower-case columns

**Compressed input support (.gz)**
- All annotation parsers now transparently read gzip-compressed files:
  - `features/domains.py` `parse_domtblout()` — `.domtblout.gz`
  - `features/pathways.py` `parse_antismash_json()` — `.json.gz`
  - `features/repeats.py` `parse_rmout()`, `get_genome_size_from_fai()` — `.out.gz`, `.fai.gz`
  - `features/motifs.py` promoter FASTA counter — `.fasta.gz`
  - `features/kmer.py` `_load_sequences()` — genome FASTA `.gz` via BioPython file handle
  - `features/composition.py` — all FASTA and CSV inputs
  - `features/subcellular.py`, `features/disorder.py`, `features/proteases.py` — all inputs
- `utils/io.py`: added `open_text(path)` helper that returns a `gzip.open` or plain `open` handle based on the `.gz` suffix
- `utils/io.py`: `discover_annotation_files()` now searches both `*{suffix}` and `*{suffix}.gz` patterns; genome ID extraction updated to correctly strip compound suffixes (e.g. `.tmhmm_short.tsv`, `.signalp.results.txt`, `_summary.targetp2`)

**Script and configuration updates**
- `scripts/01_build_features.py`:
  - New `--taxonomy` argument to load and export `samples.csv` alongside feature matrices
  - New blocks: `subcellular`, `disorder`, `proteases`, `composition` (all included in default block list)
  - Annotation subdirectories: `tmhmm/`, `signalp/`, `wolfpsort/`, `targetp/`, `aiupred/`, `merops/`, `cds/`
- `configs/default.yaml`: added `subcellular`, `disorder`, `proteases`, and `composition` sections with tunable thresholds and filters

**dbCAN substrate prediction support**
- `features/pathways.py`: added `parse_dbcan_substrate()` to parse dbCAN-sub `substrate.out` files; extracts per-genome substrate-class gene counts prefixed with `substrate_` (e.g. `substrate_cellulose`)
- `features/pathways.py`: `build_cazyme_matrix()` accepts an optional `substrate_paths` dict; substrate features are joined into the CAZyme matrix before frequency filtering
- `scripts/01_build_features.py`: automatically discovers `substrate.out` files in `data/raw/annotations/dbcan/` alongside `overview.txt` and passes them to `build_cazyme_matrix`
- `utils/io.py`: `discover_annotation_files()` now falls back to the parent directory name as `genome_id` when the file name alone yields an empty stem — supports per-genome subdirectory layouts used by dbCAN3 (e.g. `dbcan/{genome_id}/overview.txt` and `dbcan/{genome_id}/substrate.out`)

### Planned
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
