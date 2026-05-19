# HDC Table Representation

Compact research code for High-Dimensional Computing (HDC) table representation and projection experiments.

## Overview

This repository contains data, preprocessing, experiment scripts, and notebooks used to evaluate table/column representation and projection accuracy for HDC experiments (movie / dblp datasets used in analysis).

## Requirements

- Python 3.9+ (recommended)
- See [requirements.txt](requirements.txt) for exact packages

## Setup

1. Create and activate a Python virtual environment (example):

```bash
python -m venv env
source env/bin/activate
pip install -r requirements.txt
```

2. Place raw or interim datasets under the `data/` tree. Processed outputs are written to `data/processed/` and `outputs/`.

## Quick start

- Preprocess raw files: `python scripts/preprocessing.py`
- Generate queries: `python scripts/query_generation.py` (queries for the experiments in the paper are included already)
- Run experiments: `python scripts/relational_exp.py` or use `scripts/experimentconfig.py` to configure runs.

Adjust parameters in `config.yaml` or in the scripts' CLI where applicable.

## Data layout

- `data/raw/` — original source files
- `data/interim/` — intermediate extracted embeddings and CSVs
- `data/processed/` — processed tables and experiment inputs
- `data/queries/` — JSON query sets used for evaluation

## Scripts

Core scripts live in [scripts/](scripts/):
- `preprocessing.py` — data cleaning and embedding handling
- `query_generation.py` — create equality/nonequality query sets
- `relational_exp.py` — run relational/experiment workflows
- `embdi_database.py`, `hdc_database.py` — dataset/DB helpers

The generation of EmbDI embeddings is not included as part of our code.

## Notebooks

Interactive analysis and plotting notebooks are in [notebooks/](notebooks/). Example: [notebooks/projection_accuracy_analysis_movie.ipynb](notebooks/projection_accuracy_analysis_movie.ipynb)


## Outputs

- Figures and reports are under `outputs/` (figures).

## Contributing

Issues and pull requests welcome. For reproducibility, include dataset references and `config.yaml` used for experiments. Seed 0 correspond to our experiments on the queries that are included in the `data/queries/` folder.

## License

See [LICENSE](LICENSE).

## Contact

Repo maintained for research; open an issue for questions or reproducibility requests.
Main contact: <sebastian.bugedo@uhasselt.be>
