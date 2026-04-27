# Methods summary

This document summarizes the implemented PCG AF/SR pipeline at a level suitable for a Methods section.

## Preprocessing (audio standardization)
- See: `src/preprocessing.py`
- Goal: standardize smartphone recordings to a consistent sample rate, channel configuration, and amplitude scale.
- Operations (high-level):
  - load WAV, convert to mono (first channel), peak-normalize
  - resample to 48 kHz and write standardized WAV
  - downsample to 11,025 Hz
  - Butterworth bandpass filter (25–400 Hz, order 4), zero-phase (`filtfilt`)
  - peak normalize and mean-center
  - write preprocessed WAV

## Annotation inputs
- Protected export (not in repo): backend CSV where a `segments` JSON field contains interval annotations.
- Key mapping CSV (not in repo): maps `fileId` to filename.

## Clean segment definition
- See: `src/clean_segments.py`
- Exclude intervals labeled `poor` or `extremely_poor`.
- Candidate clean intervals must satisfy thresholds:
  - minimum duration (default 10 s)
  - minimum number of valid cycles (default 10 cycles)
  - each S1→S1 interval must have exactly one S2 between them (cycle consistency)
  - a limited allowance for a small number of imperfect cycles (configurable)

Recordings without any qualifying clean interval are treated as **uninterpretable (UI)** for AF/SR classification.

## RR-like interval extraction and feature calculation
- See: `src/rr_features.py`
- Use S1 interval start times as S1 onsets.
- Restrict onsets to the selected clean interval.
- RR-like series: consecutive differences of S1 onsets.
- Physiologic bounds: RR filtered to [0.3, 1.8] s (defaults).
- Outlier trimming: percentile trimming (defaults 1st–99th) if enough RR intervals exist.
- Features:
  - number of intervals (`rr_n`)
  - mean RR
  - SDNN
  - RMSSD
  - coefficient of variation
  - Poincaré SD1, SD2, SD1/SD2

## Model training and evaluation
- See: `src/train_rr_classifier.py`
- Model: `RandomForestClassifier` with `class_weight='balanced'`.
- Evaluation: out-of-fold (OOF) probabilities from stratified K-fold CV (fold count limited by minority class size).
- Classification: p(AF) threshold `t_af` (configurable); outputs are mapped to AF/SR-classified for metric computation.

## Diagnostic metrics and coverage
- See: `src/evaluate_metrics.py`
- Cohort: reference ECG12 AF/SR subset.
- Coverage categories:
  - AF/SR-classified: method outputs AF or SR
  - OA: other arrhythmia / indeterminate
  - UI: uninterpretable
  - Missing: no result available
- Sensitivity, specificity, PPV, NPV, accuracy, F1 are computed **only among AF/SR-classified outputs**, while OA/UI/Missing are reported as coverage.
