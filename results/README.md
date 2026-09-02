# Results

This directory contains only small, reviewable results. Raw CFD archives,
processed HDF5 datasets, checkpoints, logs, and large generated artifacts are
excluded by `.gitignore` and `.dockerignore`.

- `experiment_summary.csv` records the four controlled model configurations.
- `portability_summary.csv` records the three runtime smoke paths.
- `figures/` contains compact PNG visualizations from the fixed test report.

The reported metrics are velocity-field metrics with the cylinder obstacle
masked. See [`docs/PRETRAINING_REPORT.md`](../docs/PRETRAINING_REPORT.md) for
definitions and limitations.
