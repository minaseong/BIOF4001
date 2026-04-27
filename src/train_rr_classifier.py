"""src/train_rr_classifier.py

Purpose
-------
Train the RR-feature RandomForest AF/SR classifier and produce out-of-fold (OOF)
predictions.

Why this file exists
--------------------
The dissertation's main PCG model is a feature-based classifier built on
RR-like intervals derived from S1 onset times in an annotated clean segment.
This module provides the end-to-end training logic:

- build a preprocessed-PCG manifest
- load annotation payloads (protected export)
- select an analyzable (clean) segment per participant
- extract RR features
- train a RandomForest model (class_weight balanced)
- generate out-of-fold predictions for honest internal evaluation

Inputs
------
Protected inputs (not included in this repo):
- preprocessed PCG WAV directory (e.g., `AUSC_Standardized/05_preprocessed/`)
- annotation export CSV with `segments` JSON
- annotation key CSV mapping fileId -> filename
- reference label CSV with ECG12 labels (at minimum: participant_id, ecg12_4class)

Outputs
-------
- `oof_predictions.csv`: participant-level OOF pAF and AF/SR label
- `full_predictions.csv`: participant-level full-model pAF and label
- `model.joblib`: fitted RandomForest model + feature schema
- `run_config.json`: run parameters for reproducibility

Workflow step
-------------
Step 4: Model training (internal CV).

Notes
-----
- This is *not* external validation. The OOF procedure reduces optimistic bias,
  but results may still be optimistic due to small sample size and conditioning
  on analyzable segments.
- OA/UI/Missing outputs are handled downstream in `evaluate_metrics.py`.
"""

from __future__ import annotations

import argparse
import json
import time
import wave
from dataclasses import asdict
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

try:
    from .clean_segments import CleanSegmentConfig, find_clean_segments
except ImportError:  # pragma: no cover
    from clean_segments import CleanSegmentConfig, find_clean_segments
try:
    from .rr_features import RRFeatureConfig, extract_rr_features_from_clean_segment, load_latest_annotation_payloads
except ImportError:  # pragma: no cover
    from rr_features import RRFeatureConfig, extract_rr_features_from_clean_segment, load_latest_annotation_payloads


def extract_participant_id(filename: str) -> str:
    """Extract participant_id from filenames like `4023_iData4023M.wav`.

    Parameters
    ----------
    filename
        WAV filename.

    Returns
    -------
    participant_id
        Digits before the first underscore; or "UNKNOWN" if not parseable.
    """

    import re

    m = re.match(r"^(\d+)_", str(filename))
    return m.group(1) if m else "UNKNOWN"


def wav_duration_seconds(path: Path) -> tuple[float, int]:
    """Return (duration_seconds, sample_rate) for a WAV file."""

    with wave.open(str(path), "rb") as wf:
        sr = int(wf.getframerate())
        n = int(wf.getnframes())
    dur = float(n) / float(sr) if sr else float("nan")
    return dur, sr


def build_pcg_manifest(preprocessed_dir: Path) -> pd.DataFrame:
    """Build a recording-level manifest from preprocessed WAV files."""

    rows: list[dict[str, Any]] = []
    for p in sorted(preprocessed_dir.glob("*.wav")):
        pid = extract_participant_id(p.name)
        dur, sr = wav_duration_seconds(p)
        rows.append({"participant_id": pid, "pcg_path": str(p), "filename": p.name, "duration_s": dur, "sample_rate": sr})
    return pd.DataFrame(rows)


def select_first_interpretable_attempt(
    manifest: pd.DataFrame,
    ann_map: dict[str, dict[str, Any]],
    clean_cfg: CleanSegmentConfig,
) -> pd.DataFrame:
    """Select the first attempt with ≥1 clean segment for each participant.

    Participants with no qualifying clean segment are retained with `selected=False`.
    """

    out_rows: list[dict[str, Any]] = []
    for pid, g in manifest.groupby("participant_id"):
        g = g.sort_values("filename")
        chosen = None
        for _, r in g.iterrows():
            payload = ann_map.get(str(r["filename"]))
            if not payload:
                continue
            segs = find_clean_segments(payload, duration_s=float(r["duration_s"]), cfg=clean_cfg)
            if segs:
                chosen = dict(r)
                chosen["selected"] = True
                chosen["clean_count"] = int(len(segs))
                chosen["clean_best_duration_s"] = float(max(float(s["duration_s"]) for s in segs))
                break
        if chosen is None:
            # keep provenance: first file if exists
            r0 = dict(g.iloc[0]) if len(g) else {"participant_id": pid}
            r0["selected"] = False
            r0["clean_count"] = 0
            r0["clean_best_duration_s"] = 0.0
            chosen = r0
        out_rows.append(chosen)
    return pd.DataFrame(out_rows)


def _binary_from_ecg12(x: Any) -> int | None:
    """Map reference labels to binary AF(1)/SR(0) or None."""

    u = str(x).strip().upper()
    if u == "AF":
        return 1
    if u == "SR":
        return 0
    return None


def train_oof_random_forest(
    X: np.ndarray,
    y: np.ndarray,
    *,
    seed: int,
    n_splits: int = 5,
) -> tuple[np.ndarray, Any, list[dict[str, Any]]]:
    """Train RandomForest with stratified CV and return OOF probabilities."""

    from sklearn.ensemble import RandomForestClassifier
    from sklearn.model_selection import StratifiedKFold

    y = np.asarray(y, dtype=int)
    n_pos = int(np.sum(y == 1))
    n_neg = int(np.sum(y == 0))
    min_class = min(n_pos, n_neg)
    if min_class < 2:
        clf = RandomForestClassifier(n_estimators=600, class_weight="balanced", random_state=int(seed), n_jobs=-1)
        clf.fit(X, y)
        return np.full((len(y),), np.nan, dtype=float), clf, [{"note": "insufficient minority class for CV", "n_pos": n_pos, "n_neg": n_neg}]

    n_splits = int(min(int(n_splits), int(min_class)))
    skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=int(seed))

    oof = np.full((len(y),), np.nan, dtype=float)
    folds: list[dict[str, Any]] = []
    for fold, (tr, te) in enumerate(skf.split(X, y), start=1):
        clf = RandomForestClassifier(n_estimators=600, class_weight="balanced", random_state=int(seed) + fold, n_jobs=-1)
        clf.fit(X[tr], y[tr])
        proba = clf.predict_proba(X[te])
        oof[te] = proba[:, 1].astype(float)
        folds.append({"fold": fold, "n_train": int(len(tr)), "n_val": int(len(te)), "pos_rate_val": float(np.mean(y[te]))})

    final = RandomForestClassifier(n_estimators=600, class_weight="balanced", random_state=int(seed), n_jobs=-1)
    final.fit(X, y)
    return oof, final, folds


def main() -> int:
    """Train the RR-feature RandomForest and write run artifacts.

    Returns
    -------
    exit_code
        0 on success.

    Notes
    -----
    This CLI expects protected local inputs (preprocessed WAVs, annotation export,
    and ECG12 AF/SR reference labels). The public repository does not include
    those inputs; see `data/README.md` and `scripts/run_rr_pipeline.sh`.
    """

    ap = argparse.ArgumentParser(description="Train RR-feature RandomForest AF/SR classifier (OOF CV).")
    ap.add_argument("--preprocessed-dir", required=True, help="Directory of preprocessed WAVs (protected).")
    ap.add_argument("--annotation-csv", required=True, help="Annotation export CSV (protected).")
    ap.add_argument("--annotation-key-csv", required=True, help="Annotation key CSV (protected).")
    ap.add_argument("--labels-csv", required=True, help="Reference label CSV with at least participant_id, ecg12_4class (protected).")
    ap.add_argument("--out-dir", required=True, help="Output directory for model + predictions.")

    ap.add_argument("--seed", type=int, default=1337)
    ap.add_argument("--t-af", type=float, default=0.5, help="AF threshold for AF/SR classification from pAF.")
    ap.add_argument("--min-clean-sec", type=float, default=10.0)
    ap.add_argument("--min-cycles", type=int, default=10)
    ap.add_argument("--min-rr", type=int, default=10)
    args = ap.parse_args()

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    clean_cfg = CleanSegmentConfig(min_duration_s=float(args.min_clean_sec), min_cycles=int(args.min_cycles))
    rr_cfg = RRFeatureConfig(min_rr=int(args.min_rr))

    # 1) manifest
    manifest = build_pcg_manifest(Path(args.preprocessed_dir))

    # 2) annotations
    ann_map = load_latest_annotation_payloads(Path(args.annotation_csv), Path(args.annotation_key_csv))

    # 3) attempt selection
    selected = select_first_interpretable_attempt(manifest, ann_map, clean_cfg)

    # 4) features
    feat_rows = []
    for _, r in selected.iterrows():
        pid = str(r["participant_id"])
        fn = str(r.get("filename") or "")
        payload = ann_map.get(fn)
        if bool(r.get("selected")) and payload:
            feats = extract_rr_features_from_clean_segment(payload, duration_s=float(r["duration_s"]), clean_cfg=clean_cfg, rr_cfg=rr_cfg)
        else:
            feats = {"rr_n": 0.0, "clean_start_s": float("nan"), "clean_end_s": float("nan")}
        feat_rows.append({"participant_id": pid, **{k: float(v) for k, v in feats.items()}})
    feat = pd.DataFrame(feat_rows)

    # 5) reference labels
    ref = pd.read_csv(Path(args.labels_csv))
    if "participant_id" not in ref.columns or "ecg12_4class" not in ref.columns:
        raise SystemExit("labels-csv must include columns: participant_id, ecg12_4class")
    ref["participant_id"] = ref["participant_id"].astype(str)
    ref["y"] = ref["ecg12_4class"].map(_binary_from_ecg12)

    merged = ref.merge(feat, on="participant_id", how="left")

    # Trainable cohort: ECG12 AF/SR + rr_n >= min_rr
    trainable = merged.dropna(subset=["y"]).copy()
    trainable["rr_n"] = pd.to_numeric(trainable["rr_n"], errors="coerce").fillna(0.0)
    trainable = trainable[trainable["rr_n"].astype(float) >= float(rr_cfg.min_rr)].copy()
    if trainable.empty:
        raise SystemExit("No trainable rows found (need ECG12 AF/SR and sufficient RR intervals).")

    feature_cols = [
        "rr_mean",
        "rr_sdnn",
        "rr_rmssd",
        "rr_cv",
        "rr_sd1",
        "rr_sd2",
        "rr_sd1_sd2",
        "rr_n",
    ]
    for c in feature_cols:
        if c not in trainable.columns:
            trainable[c] = np.nan

    X = trainable[feature_cols].astype(float).to_numpy()
    y = trainable["y"].astype(int).to_numpy()

    oof, clf, folds = train_oof_random_forest(X, y, seed=int(args.seed), n_splits=5)

    # save model
    import joblib

    joblib.dump({"model": clf, "feature_cols": feature_cols, "clean_cfg": asdict(clean_cfg), "rr_cfg": asdict(rr_cfg)}, out_dir / "model.joblib")

    def to_label(paf: float) -> str:
        if not np.isfinite(paf):
            return "UI"
        return "AF" if float(paf) >= float(args.t_af) else "SR"

    trainable = trainable.reset_index(drop=True)
    trainable["pcg_paf_oof"] = oof.astype(float)
    trainable["pcg_pred_oof"] = [to_label(p) for p in trainable["pcg_paf_oof"].tolist()]
    trainable[["participant_id", "ecg12_4class", "pcg_paf_oof", "pcg_pred_oof"]].to_csv(out_dir / "oof_predictions.csv", index=False)

    # full predictions for all participants with features
    all_ok = merged.copy()
    all_ok["rr_n"] = pd.to_numeric(all_ok["rr_n"], errors="coerce").fillna(0.0)
    all_ok = all_ok[all_ok["rr_n"].astype(float) >= float(rr_cfg.min_rr)].copy()
    if len(all_ok):
        X_all = all_ok[feature_cols].astype(float).to_numpy()
        paf_full = clf.predict_proba(X_all)[:, 1].astype(float)
        all_ok["pcg_paf_full"] = paf_full
        all_ok["pcg_pred_full"] = [to_label(p) for p in paf_full.tolist()]
        all_ok[["participant_id", "pcg_paf_full", "pcg_pred_full"]].to_csv(out_dir / "full_predictions.csv", index=False)
    else:
        pd.DataFrame(columns=["participant_id", "pcg_paf_full", "pcg_pred_full"]).to_csv(out_dir / "full_predictions.csv", index=False)

    (out_dir / "run_config.json").write_text(
        json.dumps(
            {
                "run_id": time.strftime("%Y%m%d-%H%M%S"),
                "seed": int(args.seed),
                "t_af": float(args.t_af),
                "clean_cfg": asdict(clean_cfg),
                "rr_cfg": asdict(rr_cfg),
                "feature_cols": feature_cols,
                "folds": folds,
                "n_trainable": int(len(trainable)),
                "trainable_label_counts": {"SR": int(np.sum(y == 0)), "AF": int(np.sum(y == 1))},
            },
            indent=2,
        )
        + "\n",
        encoding="utf-8",
    )

    print(f"Wrote: {out_dir / 'oof_predictions.csv'}")
    print(f"Wrote: {out_dir / 'full_predictions.csv'}")
    print(f"Wrote: {out_dir / 'model.joblib'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
