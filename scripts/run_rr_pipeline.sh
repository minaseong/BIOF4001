#!/usr/bin/env bash
set -euo pipefail

# Template: train RR-feature RandomForest AF/SR model and produce OOF predictions.

PREPROCESSED_DIR="/path/to/protected/AUSC_Standardized/05_preprocessed"
ANNOTATION_CSV="/path/to/protected/Annotation.csv"
ANNOTATION_KEY_CSV="/path/to/protected/FileId_ParticipantId_Key.csv"
LABELS_CSV="/path/to/protected/device_labels_4class.csv"
RUN_DIR="/path/to/output/run_rr"

mkdir -p "$RUN_DIR"

python src/train_rr_classifier.py   --preprocessed-dir "$PREPROCESSED_DIR"   --annotation-csv "$ANNOTATION_CSV"   --annotation-key-csv "$ANNOTATION_KEY_CSV"   --labels-csv "$LABELS_CSV"   --out-dir "$RUN_DIR"   --t-af 0.5
