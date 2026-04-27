#!/usr/bin/env bash
set -euo pipefail

# Regenerate public-safe aggregate dissertation figures.
#
# This script does NOT require protected raw data. It executes the clean,
# documented notebook that regenerates only aggregate figures from
# `results/final_matched_metrics_table.csv`.
#
# Requirements:
# - Python environment with: pandas, numpy, matplotlib, seaborn
# - Jupyter (for nbconvert execution)

NOTEBOOK="notebooks/01_generate_final_figures.ipynb"

if ! command -v jupyter >/dev/null 2>&1; then
  echo "ERROR: jupyter is not installed or not on PATH."
  echo "Install it (e.g., pip install jupyter) or run the notebook manually."
  exit 1
fi

mkdir -p notebooks/_executed

# Execute the notebook and write an executed copy under notebooks/_executed/.
# Figures will be saved under figures/main/regenerated/.
jupyter nbconvert   --to notebook   --execute "$NOTEBOOK"   --output-dir notebooks/_executed   --output 01_generate_final_figures.executed.ipynb

echo "Done. See: figures/main/regenerated/"
