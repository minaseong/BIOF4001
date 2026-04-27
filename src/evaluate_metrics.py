"""src/evaluate_metrics.py

Purpose
-------
Compute matched-cohort diagnostic performance and coverage metrics.

Why this file exists
--------------------
For AF screening, it is essential to report *both*:

1) diagnostic performance when a method provides an AF/SR classification, and
2) coverage (how often a method returns AF/SR vs OA/UI/Missing).

This module standardizes definitions used in the dissertation:

- "AF/SR-classified" subset is used for sensitivity/specificity/PPV/NPV/accuracy/F1.
- OA/UI/Missing are excluded from those metrics but are retained for coverage.

Inputs
------
- A matched cohort table (protected labels not included in this repo).
  Required columns for evaluation:
  - `ecg12_4class` in {AF, SR} for the reference cohort
  - comparator outputs: e.g., `kardia_4class`, `fibricheck_ios_4class`, ...
  - PCG outputs: `pcg_pred_4class` and optionally `pcg_paf`

Outputs
-------
- A metrics table suitable for dissertation reporting.

Workflow step
-------------
Step 5: Diagnostic performance + coverage reporting.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Literal

import numpy as np
import pandas as pd


Bucket = Literal["classified", "OA", "UI", "missing"]


def bucket_output(x: Any) -> Bucket:
    """Bucket a 4-class output into AF/SR-classified vs OA/UI/Missing."""

    if pd.isna(x):
        return "missing"
    s = str(x).strip().upper()
    if s in {"AF", "SR"}:
        return "classified"
    if s == "OA":
        return "OA"
    if s == "UI":
        return "UI"
    return "missing"


@dataclass(frozen=True)
class BinaryMetrics:
    tp: int
    fp: int
    tn: int
    fn: int
    sensitivity: float
    specificity: float
    accuracy: float
    ppv: float
    npv: float
    f1: float


def compute_binary_metrics(y_true: np.ndarray, y_pred: np.ndarray) -> BinaryMetrics:
    """Compute binary classification metrics for AF(1) vs SR(0)."""

    y_true = np.asarray(y_true, dtype=int)
    y_pred = np.asarray(y_pred, dtype=int)
    tp = int(((y_true == 1) & (y_pred == 1)).sum())
    tn = int(((y_true == 0) & (y_pred == 0)).sum())
    fp = int(((y_true == 0) & (y_pred == 1)).sum())
    fn = int(((y_true == 1) & (y_pred == 0)).sum())

    def sd(a: float, b: float) -> float:
        return float(a) / float(b) if b else float("nan")

    sens = sd(tp, tp + fn)
    spec = sd(tn, tn + fp)
    acc = sd(tp + tn, tp + tn + fp + fn)
    ppv = sd(tp, tp + fp)
    npv = sd(tn, tn + fn)
    f1 = (2 * ppv * sens / (ppv + sens)) if (ppv + sens) else float("nan")

    return BinaryMetrics(tp, fp, tn, fn, sens, spec, acc, ppv, npv, f1)


def evaluate_method_vs_ecg12(
    df: pd.DataFrame,
    *,
    method: str,
    pred_col: str,
    prob_col: str | None = None,
) -> dict[str, Any]:
    """Evaluate one method against ECG12 reference on the AF/SR cohort."""

    ref = df[df["ecg12_4class"].isin(["AF", "SR"])].copy()
    n_ref = int(len(ref))

    buckets = ref[pred_col].map(bucket_output)
    n_class = int((buckets == "classified").sum())
    n_oa = int((buckets == "OA").sum())
    n_ui = int((buckets == "UI").sum())
    n_missing = int((buckets == "missing").sum())

    out: dict[str, Any] = {
        "method": method,
        "n_ref": n_ref,
        "n_classified_AF_SR": n_class,
        "n_OA": n_oa,
        "n_UI": n_ui,
        "n_missing": n_missing,
        "classified_rate": n_class / n_ref if n_ref else float("nan"),
        "OA_rate": n_oa / n_ref if n_ref else float("nan"),
        "UI_rate": n_ui / n_ref if n_ref else float("nan"),
        "missing_rate": n_missing / n_ref if n_ref else float("nan"),
    }

    # Metrics only on AF/SR-classified subset.
    sub = ref[buckets == "classified"].copy()
    if len(sub):
        yt = (sub["ecg12_4class"].astype(str).str.upper() == "AF").astype(int).to_numpy()
        yp = (sub[pred_col].astype(str).str.upper() == "AF").astype(int).to_numpy()
        m = compute_binary_metrics(yt, yp)
        out.update(
            {
                "sensitivity": m.sensitivity,
                "specificity": m.specificity,
                "accuracy": m.accuracy,
                "PPV": m.ppv,
                "NPV": m.npv,
                "F1": m.f1,
                "tp": m.tp,
                "fp": m.fp,
                "tn": m.tn,
                "fn": m.fn,
            }
        )

        auc = float("nan")
        if prob_col and prob_col in sub.columns:
            try:
                from sklearn.metrics import roc_auc_score

                p = pd.to_numeric(sub[prob_col], errors="coerce").to_numpy(dtype=float)
                ok = np.isfinite(p)
                if ok.sum() >= 3 and len(set(yt[ok].tolist())) == 2:
                    auc = float(roc_auc_score(yt[ok], p[ok]))
            except Exception:
                pass
        out["AUC"] = auc
    else:
        out.update({"sensitivity": float("nan"), "specificity": float("nan"), "accuracy": float("nan"), "PPV": float("nan"), "NPV": float("nan"), "F1": float("nan"), "AUC": float("nan")})

    return out
