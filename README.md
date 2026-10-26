# Multi-Kernel Gaussian Process Stacking for Boiling Point Prediction with Calibrated Uncertainty

Code for the paper of the same title. One script, `gp_stacking_bp.py`, runs every experiment and writes every table and figure in the paper.

**No data are included.** See [Data](#data).

## Overview

- **Features:** all RDKit 2D descriptors (`Descriptors.descList`, 217 in RDKit 2025.03.6); none are removed by hand. Zero-variance descriptors are removed and the rest standardised, both fitted on the training part of each split.
- **Target:** boiling point in K, Yeo-Johnson transformed (with standardisation) inside every GP fit.
- **Base models:** three exact Gaussian processes (GPyTorch, zero prior mean in the transformed target space, Gaussian likelihood):
  - rational quadratic (RQ)
  - Matérn ν = 5/2
  - RBF + linear
- **Stacking:** non-negative, sum-to-one weights (SLSQP), learned on 5-fold out-of-fold predictions. The predictive variance is the Gaussian-mixture variance (Eq. 9).
- **Evaluation:** 100 random 85/15 train/test splits (seeds 1–100). Metrics: RMSE, R², R²pred, CCC.
- **Matérn ablation:** stacking of RQ and RBF + linear only.
- **Baselines:** linear regression, random forest and gradient boosting (scikit-learn).
- **Uncertainty calibration:** cross-validation conformal prediction (CV-CP).
  - 10 outer folds on the training part, each with its own inner 5-fold stacking.
  - Normalised scores |y − μ| / σ, compared with the native ±1.96σ interval and with unnormalised scores |y − μ|.
  - conformal quantile: the ⌈(n+1)(1−α)⌉-th smallest score

## Installation

Tested on Linux with Python 3.11 and an NVIDIA GPU (driver supporting CUDA 12.1). The code also runs on CPU, but much more slowly.

```bash
# conda (recommended)
conda env create -f environment.yml
conda activate gp-stacking-bp

# or pip, in a Python 3.11 environment
pip install -r requirements.txt
```

`requirements.txt` and `environment.yml` pin the versions used for the paper:
- PyTorch 2.5.1 (CUDA 12.1 build), GPyTorch 1.14.3, linear_operator 0.6
- scikit-learn 1.8.0, RDKit 2025.03.6, NumPy 1.26.4, SciPy 1.17.0, pandas 2.3.3, matplotlib 3.10.8

## Data

The boiling-point data used in the paper come from the DIPPR 801 database. They are licensed and **cannot be redistributed**, so they are not part of this repository. Results, caches and derived tables are not included for the same reason.

To run the code, provide your own CSV file and pass it with `--data`:

| column | content |
|---|---|
| `SMILES` | molecule as a SMILES string |
| `BP` | normal boiling point in K |

Other columns are ignored.

The data are cleaned in this order:
1. Remove rows with a missing SMILES or BP.
2. Keep targets within Q1 − 1.5·IQR and Q3 + 1.5·IQR. This filter is applied to the whole file, before splitting.
3. Remove molecules that RDKit cannot parse.
4. Remove rows with any non-finite descriptor.

No descriptor is removed in these steps. Zero-variance removal (`VarianceThreshold(0)`) and standardisation are fitted later, on the training part of each split.

## Usage

All commands write to `--out` (default `0930_results/`). Finished jobs are skipped, so an interrupted run can be restarted with the same command. Each output folder records the input data and the settings in `run_config.json`, and a run with different data or settings on the same folder stops with an error; use a new `--out` whenever you change the data or the settings.

```bash
# show the job plan without training
python gp_stacking_bp.py --stage gpu_all --data BP.csv --seeds 1-100 --cp-seeds 1-100 --gpus 0,1,2,3 --dry-run

# GPU jobs: 100 main + 100 ablation splits and CV-CP for 100 seeds (10 outer folds each)
python gp_stacking_bp.py --stage gpu_all --data BP.csv --seeds 1-100 --cp-seeds 1-100 \
    --gpus 0,1,2,3 --tasks-per-gpu 2

# machine-learning baselines (CPU; can run at the same time as the GPU jobs)
python gp_stacking_bp.py --stage baseline --data BP.csv --seeds 1-100 --cpu-workers 4

# tables and figures (reads finished results only)
python gp_stacking_bp.py --stage tables  --data BP.csv
python gp_stacking_bp.py --stage figures --data BP.csv
```

Other stages:
- `main`, `ablation` and `cp` run single experiment types.
- `all` runs `gpu_all`, `baseline`, `tables` and `figures` one after another.

To stop a run gracefully, create the file `<out>/0930_STOP`. Running jobs finish and no new jobs start.

Useful options:

| option | meaning |
|---|---|
| `--tasks-per-gpu N` | jobs per GPU at the same time (default 2) |
| `--order interim\|lpt` | `interim` runs main → ablation → CP, so Tables 3–6 are available early; `lpt` runs the longest jobs first |
| `--figure-seed S` | split shown in Figures 2–3 (default 42) |
| `--no-deterministic` | do not call `torch.use_deterministic_algorithms(True)` |

**Run time.** On 4 × NVIDIA TITAN Xp, one main job takes about 7 min on its own GPU. The full GPU stage takes about 1.5–2 days. CV-CP for 100 seeds is most of this time.

## Output

```
0930_results/
├── seeds/     per-job results: 0930_seed042_main.npz, ..._main_meta.json, ..._main_metrics.csv,
│              0930_seed042_cp_fold03.npz, 0930_seed042_cp_test_predictions.csv, ...
├── tables/    every table as *_values.csv (numbers: mean, SD, SE, median, min, max)
│              and *_formatted.csv / *_formatted.md (mean ± SE, as in the paper)
├── figures/   Figures 2, 3, A.1, R2, R3 (PNG 300 dpi and PDF)
├── logs/      scheduler log, one log per job, environment and preset settings
├── cache/     RDKit descriptor matrix of the input file
├── 0930_manifest.csv           which file holds which table / figure
└── 0930_revision_results.md    summary of all tables
```

Each `*_main.npz` file stores the following, so that further analyses need no refitting:
- train/test indices
- out-of-fold and test predictions (mean and SD) of every kernel, in both the transformed space and Kelvin
- stacking weights and stacked predictions

## Reproducibility notes

- Each job runs in a fresh process and starts with fixed seeds (Python, NumPy, PyTorch, CUDA). Deterministic algorithms are requested.
- **GP training is stochastic.** For more than 800 training points, GPyTorch uses conjugate gradients and stochastic Lanczos quadrature with random probe vectors. Results therefore depend on the order of the GP fits inside a job, and this order is fixed in the code.
- With the pinned library versions and the same GPU model, repeated runs gave bit-identical results. Other GPUs or library builds can change the last digits.
- Tables report mean ± standard error over the 100 splits. The standard deviation is saved in the `*_values.csv` files.

## Citation

> Seonghyeok Park, Hyoju Kim, Keunhong Jeong, Seongil Jo, Jaeoh Kim. Multi-Kernel Gaussian Process Stacking for Boiling Point Prediction with Calibrated Uncertainty. [Journal, volume, pages, year — to be added]
