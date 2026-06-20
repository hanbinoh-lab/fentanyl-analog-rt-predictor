# Structure-based retention time prediction for fentanyl analogs

[![DOI](https://zenodo.org/badge/1275205749.svg)](https://doi.org/10.5281/zenodo.20774487)

Reproducible code, data, and trained models for structure-based LC-MS/MS retention time prediction of fentanyl analogs, using Mordred descriptors across five regression algorithms (Ridge, RF, XGBoost, GP-RBF, MLP). The models were developed using RT data acquired with serially coupled cyano and phenyl-hexyl columns.

## Overview

- **Task**: regression of RT (minutes) from structure.
- **Dataset**: 94 records for fentanyl analogs (curated from a published source; see `data/raw/README.md`).
- **Representation**: Mordred 2D+3D descriptors, reduced from 1,826 raw to 75 through a four-stage descriptor-processing pipeline.
- **Algorithms** (5): Ridge regression, random forest (RF), extreme gradient boosting (XGBoost), Gaussian process with an RBF kernel (GP-RBF), and a multilayer perceptron (MLP).
- **Validation**: repeated nested cross-validation, leave-one-out cross-validation, a y-scrambling permutation test, and k-nearest-neighbor applicability-domain analysis.

## Repository layout

```
.
├── configs/
│   ├── algorithms.yaml                 # run mapping: 5 algorithms for the Mordred representation
│   ├── representations.yaml            # Mordred feature-pipeline specification (provenance)
│   └── mordred_mechanism_groups.yaml   # RT-mechanism descriptor groups used in selection
├── data/
│   ├── raw/
│   │   ├── README.md                   # data provenance and source citation
│   │   ├── source_kim2025.csv          # transcribed source data (compound, m/z, RT)
│   │   └── base.csv                    # curated modeling table (compound, smiles, RT)
│   └── features/
│       └── mordred/
│           ├── mordred.csv             # 94 records x 75 descriptors, plus identifiers and RT
│           └── mordred_preprocessing/  # intermediate stage outputs
├── environments/
│   └── fa-rt.yml                       # conda environment specification
├── models/
│   ├── mordred/
│   │   └── 01_ridge/ ... 05_mlp/       # per-algorithm results + trained.joblib
│   └── aggregated/
│       ├── master_results.csv          # combined metrics across the 5 models
│       └── archive/                    # extended evaluation metrics and final hyperparameters
└── src/
    ├── data/                           # curated-dataset audit
    ├── features/                       # Mordred feature pipeline (4 stages) + shared lib
    ├── evaluation/                     # nested CV, LOOCV, y-scrambling, AD, orchestrator
    └── training/                       # final hyperparameter search + all-data model
```

## Data

The retention-time and mass-transition values were transcribed from the SCC-LC-MS/MS dataset reported by Kim et al. (2025) ([source article](https://doi.org/10.1016/j.aca.2024.343479)) and remain subject to the publisher's terms of use. The curated modeling table (`data/raw/base.csv`) adds canonical SMILES. Provenance, record count, and the full source citation are documented in `data/raw/README.md`. Mass-to-charge ratios were used only for structure-consistency checks and are not model inputs.

## Environment

The analysis uses a version-constrained conda environment.

```bash
conda env create -f environments/fa-rt.yml
conda activate fa-rt
```

## Reproduce

All commands are run from the repository root with the `fa-rt` environment active.

**1. Validate the curated dataset.**

```bash
python src/data/audit_curated_dataset.py
```

**2. (Optional) Regenerate Mordred features from `base.csv`.** The shipped `data/features/mordred/mordred.csv` already contains the final descriptors; run this only to rebuild it from structures.

```bash
python src/features/mordred/00_compute.py   # 1826 raw descriptors (3D conformers)
python src/features/mordred/01_filter.py    # drop NaN and constant columns
python src/features/mordred/02_select.py    # RT-mechanism descriptor selection
python src/features/mordred/03_reduce.py    # correlation clustering -> 75 descriptors
```

**3. Run the evaluation for all five models.**

```bash
python src/evaluation/run_evaluation.py --candidates all --stages all
```

Per-model outputs are written under `models/mordred/`, and the combined metric table to `models/aggregated/master_results.csv`.

Useful options:

- single model: `--candidates mordred:gp_rbf`
- a subset of stages: `--stages 1,2`
- `--resume` skips stages whose output already exists.

### Evaluation stages

| Stage | Procedure                              | Primary output                      |
| ----- | -------------------------------------- | ----------------------------------- |
| 1     | Repeated nested cross-validation       | `results_cv.csv`                  |
| 2     | Final all-data model + hyperparameters | `trained.joblib`, `config.json` |
| 3     | y-scrambling permutation test          | `results_yscrambling.csv`         |
| 4a    | Leave-one-out cross-validation         | `results_loocv.csv`               |
| 4b    | Applicability domain (kNN distance)    | `results_ad.csv`                  |

## Results

`models/aggregated/master_results.csv` holds the headline metrics (cross-validated RMSE, R^2, MAE, train-CV R^2 gap, LOOCV Q^2, y-scrambling p-value) for each of the five models. Per-model folders under `models/mordred/` hold the per-fold and per-compound records and the deployable `trained.joblib`.

## License

- All code and materials we created for this repository are released under the MIT License (see `LICENSE`).
- The retention-time and mass-transition values are not ours to license: they are reproduced from Kim et al. (2025) for research reproducibility. For any further use of those values, cite the source article, follow the publisher's terms, and direct permission requests to the original authors and publisher (see `data/raw/README.md`).
