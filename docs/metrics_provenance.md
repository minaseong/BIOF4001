# Metrics provenance (aggregate-only)

This document records where the values in `results/final_matched_metrics_table.csv` came from, with a focus on **confusion-matrix counts** (`tp/fp/tn/fn`).

## Privacy statement

- The public repository does **not** contain raw audio, annotation exports, or participant-level clinical labels.
- Where confusion counts required access to participant-level tables, counts were **recomputed locally** from a controlled-access file and only **aggregate totals** were copied into this repository.
- No participant IDs or row-level records were added to the public repo as part of this process.

## Files inspected

### Canonical run outputs (local / controlled-access project)

Located under:

- `/Users/soonheekim/Desktop/FYP/code_v04/vitogram_pcg/logs/rr_clean_10s_rf/`

Inspected files:

- `dissertation_figures/final_matched_metrics_table.csv` (aggregate; no `tp/fp/tn/fn`)
- `evaluation/summary_PCG.json` (aggregate; includes `tp/fp/tn/fn` for PCG with `n_ref=67`)
- `evaluation/summary_Kardia.json`, `evaluation/summary_Fibri_iOS.json`, `evaluation/summary_Fibri_Android.json` (aggregate; include `tp/fp/tn/fn` but evaluated on `n_ref=68`, i.e., *not* matched-to-PCG)
- `data/interim/master_table_with_pcg.csv` (**participant-level; controlled-access**) used only to recompute matched-cohort comparator confusion counts

### Public repository file

- `results/final_matched_metrics_table.csv`

## Source of the public metrics table

The base table (all metric columns **except** confusion counts) matches exactly:

- `/Users/soonheekim/Desktop/FYP/code_v04/vitogram_pcg/logs/rr_clean_10s_rf/dissertation_figures/final_matched_metrics_table.csv`

The public table adds the additional aggregate columns:

- `tp`, `fp`, `tn`, `fn`

## Matched cohort definition (Figure 2 denominator)

Matched-to-PCG cohort used throughout the public table:

- Reference cohort: `ecg12_4class ∈ {AF, SR}`
- PCG available: `pcg_attempt_count > 0`

This yields `n_ref = 67`.

One ECG12 AF/SR participant is excluded by this matched-to-PCG rule due to missing PCG:

- `participant_id = 4077` (ECG12 label SR)

## Confusion count recovery

Confusion-matrix orientation:

- Reference positive: ECG12 = AF
- Reference negative: ECG12 = SR
- Predicted positive: method = AF
- Predicted negative: method = SR
- Non-AF/SR outputs (`OA`, `UI`, missing) are excluded from `tp/fp/tn/fn`

### PCG

Counts were taken from an existing **aggregate** summary for the matched cohort:

- Source: `/Users/soonheekim/Desktop/FYP/code_v04/vitogram_pcg/logs/rr_clean_10s_rf/evaluation/summary_PCG.json`
- `n_ref = 67`
- Confusion counts: `tp=7, fp=2, tn=16, fn=1`

These values were cross-checked by recomputing from the controlled-access participant-level table below.

### Kardia, FibriCheck iOS, FibriCheck Android

The canonical run includes aggregate summaries for these methods on a cohort with `n_ref = 68`:

- `/Users/soonheekim/Desktop/FYP/code_v04/vitogram_pcg/logs/rr_clean_10s_rf/evaluation/summary_Kardia.json`
- `/Users/soonheekim/Desktop/FYP/code_v04/vitogram_pcg/logs/rr_clean_10s_rf/evaluation/summary_Fibri_iOS.json`
- `/Users/soonheekim/Desktop/FYP/code_v04/vitogram_pcg/logs/rr_clean_10s_rf/evaluation/summary_Fibri_Android.json`

Because the public reporting uses the **matched-to-PCG** denominator (`n_ref=67`), comparator confusion counts were recomputed from a controlled-access participant-level table:

- Source (participant-level; not copied to public repo):
  - `/Users/soonheekim/Desktop/FYP/code_v04/vitogram_pcg/data/interim/master_table_with_pcg.csv`

Columns used:

- Reference: `ecg12_4class`
- Matched-to-PCG availability: `pcg_attempt_count`
- Method outputs:
  - `kardia_4class`
  - `fibricheck_ios_4class`
  - `fibricheck_android_4class`

Filtering applied:

1. Keep `ecg12_4class ∈ {AF, SR}`
2. Keep `pcg_attempt_count > 0` (matched-to-PCG cohort)
3. For each method, keep only rows where method output ∈ {AF, SR} when computing `tp/fp/tn/fn`

Aggregate results copied to the public table:

- FibriCheck iOS: `tp=16, fp=1, tn=44, fn=0` (classified n=61)
- FibriCheck Android: `tp=10, fp=0, tn=28, fn=0` (classified n=38)
- Kardia: `tp=13, fp=1, tn=43, fn=0` (classified n=57)

Each method satisfies:

- `tp + fp + tn + fn = n_classified_AF_SR`

