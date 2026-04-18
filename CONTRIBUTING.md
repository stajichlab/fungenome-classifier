# Contributing to FungalClassifier

Thank you for contributing. This document covers development setup, code standards, testing, and how to add new feature blocks or model types.

---

## Development Setup

```bash
git clone https://github.com/your-org/fungal-classifier.git
cd fungal-classifier
conda env create -f environment.yml
conda activate fungal-classifier
pip install -e ".[dev]"
```

Install pre-commit hooks:
```bash
pip install pre-commit
pre-commit install
```

---

## Code Standards

- **Python 3.11+**
- **Formatter:** `ruff format` (enforced by pre-commit)
- **Linter:** `ruff check` — no warnings allowed in CI
- **Type hints:** required for all public functions
- **Docstrings:** NumPy-style for all public classes and functions
- **Line length:** 100 characters

Run checks locally:
```bash
make lint
make format
```

---

## Testing

```bash
# Fast unit tests (no slow marker)
make test-fast

# All tests including integration
make test

# Single test file
pytest tests/test_features.py -v
```

All PRs must pass the full test suite. Integration tests (`@pytest.mark.slow`) run on CI but not on every local commit.

**Test coverage target:** >80% for `fungal_classifier/` core modules.

---

## Adding a New Feature Block

1. Create `fungal_classifier/features/{block_name}.py`
2. Implement a `build_{block_name}_matrix(paths, ...) -> pd.DataFrame` function
3. Add it to `fungal_classifier/features/__init__.py`
4. Add the block handler to `scripts/01_build_features.py`
5. Add default config section to `configs/default.yaml`
6. Write unit tests in `tests/test_features.py`
7. Add a section to `docs/feature_engineering.md`

**Convention:** All feature matrices must:
- Have `genome_id` as the index name
- Return `np.float32` dtype
- Accept a `min_genome_freq` parameter and filter rare features
- Log their shape at INFO level

---

## Adding a New Model Type

1. Create `fungal_classifier/models/{model_name}.py`
2. Implement sklearn-compatible `fit(X, y)` and `predict(X)` / `predict_proba(X)`
3. Add to `fungal_classifier/models/__init__.py`
4. Add a `--model-type` option to `scripts/02_train.py` if appropriate
5. Write tests in `tests/test_models.py`

---

## Pull Request Process

1. Branch from `main`: `git checkout -b feature/my-feature`
2. Write code and tests
3. Run `make test && make lint`
4. Update docs if you changed behaviour or added features
5. Open a PR with a clear description of what changed and why
6. A maintainer will review within a week

---

## Reporting Issues

Please include:
- Python version and OS
- Conda environment (`conda list`)
- Minimal reproducible example
- Full error traceback

---

## Roadmap

Planned additions:
- [ ] `features/secretome.py` — signal peptide and GPI anchor features
- [ ] `features/synteny.py` — gene cluster synteny features
- [ ] `models/graph_model.py` — GNN over genome-scale metabolic networks
- [ ] Multi-label classification (genomes with multiple ecological roles)
- [ ] Active learning module for prioritising new genome sequencing
- [ ] Integration with NCBI datasets API for auto-downloading annotation
