# Data Science Portfolio Project Template

Use this repo structure for any end-to-end data science project you want to showcase on GitHub.

## What to include in a portfolio project

- A clear problem statement + success metric
- Reproducible environment setup
- Clean notebook(s) for EDA (and **saved outputs** in `reports/`)
- A small, testable Python package in `src/` (feature engineering, training, inference)
- A short results section with charts + takeaways

## Repo structure

```
data/
  external/      # third-party data (kept out of git if large)
  raw/           # immutable raw data (DO NOT edit in place)
  processed/     # cleaned/feature-ready data
notebooks/       # exploration and experiments
reports/
  figures/       # exported charts used in README
src/             # reusable python code (importable)
tests/           # unit tests
```

## Quickstart (local)

Create a virtual environment and install dependencies:

```bash
python3 -m venv .venv
source .venv/bin/activate
python -m pip install -U pip
pip install -r requirements.txt
```

Run the example pipeline:

```bash
python -m src.models.train
```

## Suggested README sections for each project

- **Problem**
- **Data**
- **Approach**
- **Results**
- **How to reproduce**
- **Limitations & next steps**

