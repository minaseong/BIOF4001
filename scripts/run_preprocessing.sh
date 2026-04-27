#!/usr/bin/env bash
set -euo pipefail

# Template: run preprocessing on protected raw PCG WAV files.

DATA_DIR="/path/to/protected/raw_pcg_wav"
OUT_BASE="/path/to/output/preprocessing"

# Example:
# python src/preprocessing.py --raw-dir "$DATA_DIR" --out-base "$OUT_BASE"

python src/preprocessing.py \
  --raw-dir "$DATA_DIR" \
  --out-base "$OUT_BASE"
