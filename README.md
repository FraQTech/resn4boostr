# BOOSTR RRC-ESN Reproducibility Repository

Physics-augmented reconstructive reservoir computing for low-latency candidate-event detection in Fermilab Booster signals.

This repository contains the single end-to-end Python suite used for the BOOSTR Partial Release experiments in the PRAB manuscript. The script performs:

- physics-aware feature engineering for BOOSTR / GMPS / beam channels,
- GPU/CPU multi-lag reconstructive ESN training and scoring,
- ensemble aggregation over temporal scales,
- validation-calibrated event extraction,
- baselines: PCA-Q, Isolation Forest, and LSTM reconstruction,
- ablations: no physics, no score exclusion, no ensemble, MAD threshold, single lag,
- proxy validation, synthetic injection stress tests, seed sensitivity, runtime accounting,
- figures, event tables, score traces, JSON summaries, and CSV artifacts.

The dataset is **not included** in this repository. Place `BOOSTR_PartialRelease.csv` in the repository root, or pass its path through `--data`.

## Repository contents

```text
boostr_rrcesn_suite_gpuopt.py     # single full experiment suite
requirements.txt                  # Python dependencies
requirements-cpu.txt              # CPU-only dependency set
scripts/run_full_cuda.bat         # Windows full paper run
scripts/run_full_cuda.sh          # Linux/macOS full paper run
scripts/run_smoke_cpu.bat         # Windows 200k-row smoke/full-suite run
scripts/run_smoke_cpu.sh          # Linux/macOS 200k-row smoke/full-suite run
scripts/run_threshold_strict.bat  # Windows strict-threshold sensitivity run
scripts/run_threshold_strict.sh   # Linux/macOS strict-threshold sensitivity run
data/README.md                    # dataset placement instructions
docs/REPRODUCIBILITY.md           # exact commands and expected headline metrics
```

## Setup

Python 3.10 or 3.11 is recommended.

### CPU-compatible environment

```bash
python -m venv .venv
# Windows: .venv\Scripts\activate
# Linux/macOS: source .venv/bin/activate
python -m pip install --upgrade pip
pip install -r requirements-cpu.txt
```

### CUDA environment

Install PyTorch for your CUDA version using the official PyTorch instructions, then install the remaining requirements:

```bash
python -m venv .venv
# Windows: .venv\Scripts\activate
# Linux/macOS: source .venv/bin/activate
python -m pip install --upgrade pip
pip install numpy pandas matplotlib tqdm scipy scikit-learn
# Then install torch with the CUDA wheel appropriate for your system.
```

Example for CUDA 12.1, when supported by your platform:

```bash
pip install torch --index-url https://download.pytorch.org/whl/cu121
```

## Data

Download the BOOSTR Partial Release CSV from the original BOOSTR release and put it at:

```text
BOOSTR_PartialRelease.csv
```

or pass the absolute path:

```bash
python boostr_rrcesn_suite_gpuopt.py --data /path/to/BOOSTR_PartialRelease.csv --time-col time
```

Large CSV files and run artifacts are intentionally ignored by `.gitignore`.

## Exact commands used for the paper-style runs

### Full CUDA run

```bash
python boostr_rrcesn_suite_gpuopt.py --data BOOSTR_PartialRelease.csv --time-col time --suite full --run-baselines --run-injection-eval --device cuda --chunk-size 1000000 --accum-batch 16384 --n-reservoir 1200 --n-ensemble 3 --temporal-scales 1 5 15 --ae-batch 256 --ae-hidden 256 --ae-max-train-windows 20000 --attribution-topk 10
```

### Strict-threshold sensitivity run

```bash
python boostr_rrcesn_suite_gpuopt.py --data BOOSTR_PartialRelease.csv --time-col time --suite main --device cuda --chunk-size 1000000 --accum-batch 16384 --n-reservoir 1200 --n-ensemble 3 --temporal-scales 1 5 15 --lags 1 3 5 7 --weights 0.5 0.3 0.15 0.05 --quantile 0.999 --target-events-per-hour 1 --smooth-s 1.0 --min-dwell-s 1.0 --merge-gap-s 0.5 --hysteresis 0.1 --attribution-topk 10 --experiment-name threshold_sensitivity_strict
```

### 200k-row smoke/full-suite CPU run

```bash
python boostr_rrcesn_suite_gpuopt.py --data BOOSTR_PartialRelease.csv --time-col time --suite full --run-baselines --run-injection-eval --max-rows 200000 --device auto
```

## Outputs

Each run writes a timestamped folder under `runs/`, for example:

```text
runs/boostr_rrcesn_suite_YYYYMMDD_HHMMSS/
```

Typical outputs include:

- `summary.csv`
- `runtime.csv`
- score and event CSV files
- proxy-validation CSV/JSON files
- ablation and baseline figures
- top-event attribution plots
- drift and split diagnostics

See `docs/REPRODUCIBILITY.md` for expected headline metrics and command notes.

## Important interpretation note

The detector is intended as an operator-facing diagnostic and candidate-event review layer. It is **not** an autonomous interlock or machine-protection system. The I:IB proxy is a weak alignment signal, not dense expert ground truth.


## Code organization

The original research script has been refactored into a small Python package while preserving the exact same command-line interface:

- `boostr_rrcesn_suite_gpuopt.py` — thin wrapper/entry point; keep using the same commands as before.
- `boostr_rrcesn/common.py` — constants, imports, dataclasses, utilities.
- `boostr_rrcesn/preprocessing.py` — BOOSTR physics-aware feature engineering and robust scaling.
- `boostr_rrcesn/events.py` — threshold calibration, hysteresis, event extraction, Jaccard overlap.
- `boostr_rrcesn/models.py` — GPU/CPU reconstructive multi-lag ESN and ensemble scoring.
- `boostr_rrcesn/baselines.py` — PCA-Q, Isolation Forest, and LSTM reconstruction baselines.
- `boostr_rrcesn/validation.py` — weak I:IB proxy validation and PR/AP metrics.
- `boostr_rrcesn/injection.py` — synthetic perturbation stress testing.
- `boostr_rrcesn/plotting.py` — all saved figures.
- `boostr_rrcesn/runtime.py` — method runners and artifact writing.
- `boostr_rrcesn/cli.py` — argument parser and full-suite orchestration.

This refactor is intentionally conservative: it changes code organization, not the experiment logic or the commands used in the paper.
