# Smartphone PCG-Based Atrial Fibrillation Screening

This repository provides a **reproducible analysis workflow** for a smartphone-recorded phonocardiogram (PCG) pipeline for atrial fibrillation (AF) screening, evaluated against a 12-lead ECG reference.

The repo is designed for **privacy-preserving sharing**: it contains code, aggregate results, and aggregate reporting figures, while the underlying clinical dataset remains **controlled-access**.

## Research question
Can a smartphone PCG pipeline provide useful AF vs sinus rhythm (SR) discrimination, and how does its **coverage** (ability to return an AF/SR classification vs OA/UI/Missing) compare with FDA-cleared comparator devices (Kardia single‑lead ECG and FibriCheck PPG)?

## What this repo includes
- Core pipeline modules (`src/`): preprocessing, clean-segment selection, RR feature extraction, RR‑feature RandomForest training with out‑of‑fold (OOF) evaluation, and diagnostic metric computation.
- Aggregate result tables (`results/`).
- Aggregate reporting figures (`figures/main/`).
- A notebook to reproduce aggregate reporting figures from aggregate tables (`notebooks/01_reproduce_aggregate_figures.ipynb`).
- Archived prior code under `legacy/previous_submission/` (sanitized).

## Data and figure privacy
- Raw clinical audio is **not** included.
- Individual-level labels and private annotation exports are **not** included.
- Individual-level signal-derived figures (e.g., waveform/RR examples) are **excluded** from the public repository.
- Aggregate figures and aggregate result tables are included.
- Full reruns require local access to the controlled-access clinical dataset.

## Repository structure
```
README.md
requirements.txt
.gitignore
LICENSE
src/                 # core pipeline modules
scripts/             # template scripts (use placeholders for controlled-access paths)
notebooks/           # reproducible aggregate reporting notebooks
figures/
  main/              # aggregate reporting figures (included)
  methodology/       # placeholders only (signal-derived figures excluded)
results/             # aggregate, non-sensitive outputs
docs/                # methods summary, figure captions, claims audit
legacy/previous_submission/  # archived previous code (sanitized)
```

## Workflow overview
This analysis reports:
1) **Coverage**: how often each method returns AF/SR vs OA/UI/Missing.
2) **Diagnostic performance**: sensitivity/specificity/PPV/NPV/accuracy computed **only among AF/SR-classified outputs**.

### Workflow table
| Step | Script / Module | Input (controlled-access unless noted) | Output | Purpose |
|---:|---|---|---|---|
| 1 | `src/preprocessing.py` | raw PCG WAV | standardized + preprocessed WAV | standardize audio (mono, resample, bandpass, normalization) |
| 2 | `src/clean_segments.py` | S1/S2 + quality annotations | clean segment intervals | identify analyzable PCG regions |
| 3 | `src/rr_features.py` | clean segment + S1 onsets | RR feature table | quantify rhythm irregularity |
| 4 | `src/train_rr_classifier.py` | RR features + ECG12 labels | OOF predictions + RF model | train internal AF/SR classifier |
| 5 | `src/evaluate_metrics.py` | predictions + labels | metrics/coverage tables | compute diagnostic performance and coverage |
| Reporting figures | `notebooks/01_reproduce_aggregate_figures.ipynb` | aggregate results table | aggregate figures | reproduce aggregate reporting figures |

## Installation
Recommended: Python 3.10+.

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

## Running locally (requires controlled-access dataset)
Template scripts use placeholders; edit paths for your local secure environment:

1) Preprocess raw audio
```bash
bash scripts/run_preprocessing.sh
```

2) Train RR-feature RandomForest and generate OOF predictions
```bash
bash scripts/run_rr_pipeline.sh
```

## Reproducing aggregate reporting figures (no controlled-access data required)
- Open and run: `notebooks/01_reproduce_aggregate_figures.ipynb`
- Inputs: aggregate tables under `results/` (e.g., `results/final_matched_metrics_table.csv`)
- Outputs: regenerated figures may be written locally (do not commit generated artifacts).

## Included outputs
- Aggregate metrics: `results/final_matched_metrics_table.csv`
- Aggregate figures: see `figures/main/`

## Limitations
- The RR-feature PCG pipeline is **annotation-dependent** for clean-segment selection.
- Coverage can be limited because RR features require sufficiently labeled clean segments.
- Reported performance is internal (OOF) rather than an external prospective validation.

## Citation / acknowledgement
This repository supports reporting for a smartphone PCG AF screening research project. Protected clinical data are excluded.
