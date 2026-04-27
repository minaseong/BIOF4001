"""src/rr_features.py

Purpose
-------
Extract RR-like intervals from S1 onset times and compute RR-variability features.

Why this file exists
--------------------
Atrial fibrillation (AF) is characterized by irregular ventricular response.
When S1 onsets are reliably labeled in a clean segment, consecutive S1-to-S1
intervals can be used as an RR-like series for feature-based AF/SR screening.

Inputs
------
- Annotation payload containing S1/S2 intervals.
- A selected clean segment interval (start,end seconds).

Outputs
-------
- RR interval array (seconds) and derived features:
  mean RR, SDNN, RMSSD, coefficient of variation, Poincaré SD1/SD2, SD1/SD2 ratio,
  and number of RR intervals.

Workflow step
-------------
Step 3: RR extraction and feature generation.

Notes
-----
- RR intervals are filtered to physiologic bounds and trimmed for outliers.
- Features are used for the RandomForest AF/SR classifier.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

try:
    from .clean_segments import CleanSegmentConfig, onsets_in_interval, pick_longest_clean_interval
except ImportError:  # pragma: no cover
    from clean_segments import CleanSegmentConfig, onsets_in_interval, pick_longest_clean_interval


def _as_list(x: Any) -> list[dict[str, Any]]:
    return x if isinstance(x, list) else []


@dataclass(frozen=True)
class RRFeatureConfig:
    """Thresholds for RR extraction and trimming."""

    rr_min_s: float = 0.3
    rr_max_s: float = 1.8
    min_rr: int = 10
    trim_lo: float = 0.01
    trim_hi: float = 0.99


def s1_onsets_from_payload(payload: dict[str, Any]) -> np.ndarray:
    """Extract S1 onset times from payload (uses interval start times)."""

    s1 = payload.get("S1") or []
    onsets: list[float] = []
    for seg in _as_list(s1):
        try:
            onsets.append(float(seg.get("start")))
        except Exception:
            continue
    onsets = sorted(set([x for x in onsets if np.isfinite(x)]))
    return np.asarray(onsets, dtype=float)


def rr_from_onsets(onsets_s: np.ndarray) -> np.ndarray:
    """Compute RR-like intervals as consecutive differences of S1 onset times."""

    if onsets_s.size < 2:
        return np.asarray([], dtype=float)
    return np.diff(onsets_s).astype(float)


def poincare_sd1_sd2(rr: np.ndarray) -> tuple[float, float]:
    """Compute Poincaré SD1 and SD2 descriptors from an RR series."""

    if rr.size < 2:
        return float("nan"), float("nan")
    x1 = rr[:-1]
    x2 = rr[1:]
    diff = (x2 - x1) / np.sqrt(2.0)
    summ = (x2 + x1) / np.sqrt(2.0)
    sd1 = float(np.std(diff, ddof=1)) if diff.size > 1 else float("nan")
    sd2 = float(np.std(summ, ddof=1)) if summ.size > 1 else float("nan")
    return sd1, sd2


def filter_and_trim_rr(rr: np.ndarray, cfg: RRFeatureConfig) -> np.ndarray:
    """Physiologic RR filtering + percentile trimming."""

    rr = np.asarray(rr, dtype=float)
    rr = rr[np.isfinite(rr)]
    rr = rr[(rr >= float(cfg.rr_min_s)) & (rr <= float(cfg.rr_max_s))]
    if rr.size < int(cfg.min_rr):
        return rr.astype(float)

    lo = float(np.quantile(rr, float(cfg.trim_lo)))
    hi = float(np.quantile(rr, float(cfg.trim_hi)))
    rr = rr[(rr >= lo) & (rr <= hi)]
    return rr.astype(float)


def rr_features_from_rr(rr: np.ndarray) -> dict[str, float]:
    """Compute RR-derived features from a cleaned RR array."""

    rr = np.asarray(rr, dtype=float)
    if rr.size == 0:
        return {"rr_n": 0.0}

    mean_rr = float(np.mean(rr))
    sdnn = float(np.std(rr, ddof=1)) if rr.size > 1 else float("nan")
    rmssd = float(np.sqrt(np.mean(np.diff(rr) ** 2))) if rr.size > 2 else float("nan")
    cv = float(sdnn / mean_rr) if mean_rr else float("nan")

    sd1, sd2 = poincare_sd1_sd2(rr)
    sd1_sd2 = float(sd1 / sd2) if sd2 and np.isfinite(sd1) else float("nan")

    return {
        "rr_n": float(rr.size),
        "rr_mean": mean_rr,
        "rr_sdnn": sdnn,
        "rr_rmssd": rmssd,
        "rr_cv": cv,
        "rr_sd1": sd1,
        "rr_sd2": sd2,
        "rr_sd1_sd2": sd1_sd2,
    }


def extract_rr_features_from_clean_segment(
    payload: dict[str, Any],
    *,
    duration_s: float,
    clean_cfg: CleanSegmentConfig,
    rr_cfg: RRFeatureConfig,
) -> dict[str, float]:
    """Extract RR features from the longest clean segment.

    Returns
    -------
    features
        RR feature dict plus clean segment boundaries.

    Notes
    -----
    If no qualifying clean segment exists, returns rr_n=0 and NaN boundaries.
    """

    interval = pick_longest_clean_interval(payload, duration_s=float(duration_s), cfg=clean_cfg)
    if interval is None:
        return {"rr_n": 0.0, "clean_start_s": float("nan"), "clean_end_s": float("nan")}

    onsets = s1_onsets_from_payload(payload)
    onsets = onsets_in_interval(onsets, interval)
    rr = rr_from_onsets(onsets)
    rr = filter_and_trim_rr(rr, rr_cfg)

    feats = rr_features_from_rr(rr)
    feats["clean_start_s"] = float(interval[0])
    feats["clean_end_s"] = float(interval[1])
    return feats


def load_latest_annotation_payloads(annotation_csv: Path, key_csv: Path) -> dict[str, dict[str, Any]]:
    """Load latest annotation payload per fileId, keyed by filename.

    Parameters
    ----------
    annotation_csv
        CSV with a `segments` JSON column and timestamps.
    key_csv
        CSV mapping `fileId` to `filename`.

    Returns
    -------
    mapping
        Dict: filename -> annotation payload dict.

    Notes
    -----
    This function is provided for local reruns with protected annotation exports.
    The repository does not ship those CSVs.
    """

    ann = pd.read_csv(annotation_csv, encoding="utf-8-sig")
    key = pd.read_csv(key_csv, encoding="utf-8-sig")

    fid_to_fn = {str(r["id"]): str(r["filename"]) for _, r in key.iterrows() if pd.notna(r.get("id")) and pd.notna(r.get("filename"))}

    ann["_updatedAt"] = pd.to_datetime(ann.get("updatedAt"), errors="coerce")
    ann["_createdAt"] = pd.to_datetime(ann.get("createdAt"), errors="coerce")
    ann["_best_ts"] = ann["_updatedAt"].fillna(ann["_createdAt"])
    ann = ann.sort_values(["fileId", "_best_ts"]).drop_duplicates(subset=["fileId"], keep="last")

    out: dict[str, dict[str, Any]] = {}
    for _, r in ann.iterrows():
        fid = r.get("fileId")
        if pd.isna(fid):
            continue
        fn = fid_to_fn.get(str(fid))
        if not fn:
            continue
        raw = r.get("segments")
        if not isinstance(raw, str) or not raw.strip():
            continue
        try:
            payload = json.loads(raw)
        except Exception:
            continue
        if isinstance(payload, dict):
            out[str(fn)] = payload
    return out
