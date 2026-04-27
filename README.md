# Smartphone PCG AF Screening (Vitogram)

This repository contains the **dissertation-ready code and aggregate results** for a smartphone phonocardiogram (PCG) pipeline to screen for atrial fibrillation (AF), evaluated against a 12-lead ECG reference.

**Important privacy note**
- **Protected data are not included**: raw audio, participant-level labels, and private annotation exports are excluded.
- The repo is designed so that figures and aggregate tables can be reviewed publicly, while **full reruns require local access** to the protected dataset.

## Research question
Can a smartphone-recorded PCG pipeline produce an AF vs sinus rhythm (SR) screening classification with useful discriminative performance, and how does its *coverage* compare to FDA-cleared comparator devices (Kardia single-lead ECG and FibriCheck PPG)?

## What this repo includes
- Clean, documented Python modules for:
  - audio preprocessing/standardization
  - clean-segment selection from S1/S2 + quality annotations
  - RR-like interval feature extraction
  - RandomForest AF/SR classifier training with out-of-fold (OOF) evaluation
  - diagnostic metrics and coverage reporting
- Final exported dissertation figures (aggregate / non-sensitive)
- Final aggregate result tables (non-sensitive)
- Archived prior code under `legacy/previous_submission/` (sanitized)

## Data and figure privacy

- Raw clinical audio is **not** included.
- Participant-level labels and private annotation exports are **not** included.
- Participant-specific waveform/RR example figures are **not** included in the public repository.
- Aggregate figures and aggregate result tables are included.
- Full reproduction requires local access to the protected dataset.

## What this repo does NOT include
- Raw smartphone recordings (`.wav`, `.m4a`, etc.)
- Private annotation exports (e.g., backend CSV with `segments` JSON)
- Participant-level clinical/device label tables
- Large intermediate artifacts or training runs

## Repository structure
```
README.md
requirements.txt
.gitignore
LICENSE
src/                 # dissertation-ready pipeline code (documented)
scripts/             # runnable templates (use placeholders for protected paths)
figures/
  main/              # main dissertation figures (aggregate)
  methodology/       # methodology figures (may be protected; keep local)
results/             # aggregate non-sensitive result tables
docs/                # methods + captions + claims audit
data/README.md       # where protected data should be placed locally
legacy/previous_submission/  # archived, sanitized previous code
```

## Workflow overview
The dissertation pipeline is a **two-part report**:
1) **Coverage**: how often each method returns an AF/SR classification vs OA/UI/Missing.
2) **Diagnostic performance**: sensitivity/specificity/accuracy/PPV/NPV computed **only among AF/SR-classified outputs**.

### Workflow table
| Step | Script / Module | Input (protected unless noted) | Output | Purpose |
|---:|---|---|---|---|
| 1 | `src/preprocessing.py` | raw PCG WAV | standardized + preprocessed WAV | standardize audio (mono, resample, bandpass, normalization) |
| 2 | `src/clean_segments.py` | S1/S2 + quality annotations | clean segment intervals | identify analyzable PCG regions |
| 3 | `src/rr_features.py` | clean segment + S1 onsets | RR feature table | quantify rhythm irregularity |
| 4 | `src/train_rr_classifier.py` | RR features + ECG12 labels | OOF predictions + RF model | train preliminary AF/SR classifier |
| 5 | `src/evaluate_metrics.py` | predictions + labels | metrics/coverage tables | compute diagnostic performance and coverage |
| Final figures | `notebooks/01_generate_final_figures.ipynb` | aggregate results table | public-safe figures | regenerate dissertation aggregate figures |

**Figure regeneration (public-safe):** aggregate dissertation figures (Figures 1–3) can be regenerated using `notebooks/01_generate_final_figures.ipynb` from `results/final_matched_metrics_table.csv`.

Participant-level signal-derived figures (e.g., RR/tachogram examples and waveform methodology panels) are **excluded** from this public repository and are included only in the submitted dissertation.


## Installation
Recommended: Python 3.10+.

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

## Running locally (requires protected data)
This repo provides **template scripts** with placeholders.

1) Preprocess raw audio
```bash
bash scripts/run_preprocessing.sh
```

2) Train RR-feature RandomForest and generate OOF predictions
```bash
bash scripts/run_rr_pipeline.sh
```

3) Recompute aggregate tables and regenerate figures
```bash
bash scripts/regenerate_final_figures.sh
```

## Main results (high-level)
- The RR-feature PCG classifier shows **promising AF/SR discrimination on the AF/SR-classified subset**, but **coverage is currently limited** because the RR-feature approach depends on annotated clean segments with sufficient S1/S2 cycles.

See `results/` and `figures/main/` for dissertation-ready summaries.

## Limitations (for honest reporting)
- The PCG RR-feature pipeline is **annotation-dependent** (clean segment selection uses S1/S2 labels).
- The evaluated AF/SR-classified subset can be small; ROC curves may look step-like.
- Results are internal (OOF) and not an external prospective validation.

## Reproducibility and troubleshooting
- No hard-coded personal paths are used in `src/`.
- If you see file-not-found errors, confirm your protected data paths in `scripts/*.sh`.

## Citation / acknowledgement
This repository is a dissertation artifact for the smartphone PCG AF screening project.
