"""src/evaluate_metrics.py

Purpose
-------
Compute matched-cohort diagnostic performance and coverage metrics.

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


def compute_roc_curve(
    y_true: np.ndarray,
    y_score: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, float]:
    """
    Compute ROC curve points and AUC for binary classification.

    Parameters
    ----------
    y_true
        Binary ground-truth labels encoded as 0/1 (0=negative, 1=positive).
        In this project: 0 = SR, 1 = AF.
    y_score
        Continuous scores/probabilities for the positive class (AF).

    Returns
    -------
    fpr, tpr, thresholds, auc
        False-positive rate array, true-positive rate array, thresholds array,
        and scalar AUC.

    Notes
    -----
    The public GitHub repository does not include participant-level probability
    scores. This helper is intended for *protected-local* runs where such
    probabilities are available.
    """

    from sklearn.metrics import auc as _auc
    from sklearn.metrics import roc_curve as _roc_curve

    y_true = np.asarray(y_true, dtype=int)
    y_score = np.asarray(y_score, dtype=float)
    fpr, tpr, thresholds = _roc_curve(y_true, y_score)
    return fpr, tpr, thresholds, float(_auc(fpr, tpr))


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

def confusion_matrix_from_counts(tp: int, fp: int, tn: int, fn: int) -> np.ndarray:
    """Construct a 2×2 confusion matrix from TP/FP/TN/FN counts.

    Parameters
    ----------
    tp, fp, tn, fn
        Standard binary confusion counts where the positive class is AF.

    Returns
    -------
    cm
        A 2×2 matrix with rows = ECG12 reference [SR, AF] and
        columns = predicted [SR, AF]::

            [[TN, FP],
             [FN, TP]]

    Notes
    -----
    This orientation matches the dissertation reporting convention:
    - Reference (rows): SR then AF
    - Prediction (cols): SR then AF
    """

    return np.asarray([[int(tn), int(fp)], [int(fn), int(tp)]], dtype=int)


def confusion_matrix_from_metrics_row(row: pd.Series | dict[str, Any]) -> np.ndarray:
    """Build a 2×2 confusion matrix from one metrics-table row.

    Parameters
    ----------
    row
        A row containing integer columns: `tp`, `fp`, `tn`, `fn`.

    Returns
    -------
    cm
        2×2 confusion matrix with rows = reference [SR, AF] and
        cols = predicted [SR, AF].

    Raises
    ------
    ValueError
        If required count columns are missing.
    """

    if isinstance(row, dict):
        row = pd.Series(row)

    required = ["tp", "fp", "tn", "fn"]
    missing = [c for c in required if c not in row.index]
    if missing:
        raise ValueError(
            "Cannot regenerate confusion matrix: metrics row is missing count columns "
            f"{missing}. Add aggregate tp/fp/tn/fn columns to the metrics table."
        )

    tp = int(row["tp"])
    fp = int(row["fp"])
    tn = int(row["tn"])
    fn = int(row["fn"])
    return confusion_matrix_from_counts(tp=tp, fp=fp, tn=tn, fn=fn)


def validate_metric_table(df: pd.DataFrame) -> pd.DataFrame:
    """Validate an aggregate metrics table.

    This is used by the public notebook to ensure the included aggregate results
    are self-consistent before plotting.

    Validation checks
    -----------------
    - Required columns exist.
    - For each method: n_classified_AF_SR + n_OA + n_UI + n_missing == n_ref.
    - Method names are present and non-empty.

    Parameters
    ----------
    df
        Metrics table loaded from CSV.

    Returns
    -------
    df
        The same dataframe (copied) after validation.

    Raises
    ------
    ValueError
        If the table is missing required columns or contains inconsistent counts.
    """

    required_cols = [
        "method",
        "n_ref",
        "n_classified_AF_SR",
        "n_OA",
        "n_UI",
        "n_missing",
        "classified_rate",
        "OA_rate",
        "UI_rate",
        "missing_rate",
        "sensitivity",
        "specificity",
        "accuracy",
        "PPV",
        "NPV",
    ]
    missing = [c for c in required_cols if c not in df.columns]
    if missing:
        raise ValueError(f"Metrics table is missing required columns: {missing}")

    out = df.copy()
    if out["method"].isna().any() or (out["method"].astype(str).str.strip() == "").any():
        raise ValueError("Metrics table contains empty method names.")

    # Count consistency
    for _, r in out.iterrows():
        total = int(r["n_classified_AF_SR"]) + int(r["n_OA"]) + int(r["n_UI"]) + int(r["n_missing"])
        if total != int(r["n_ref"]):
            raise ValueError(f"Count mismatch for {r['method']}: {total} != {int(r['n_ref'])}")

        # Optional confusion-count consistency:
        # If tp/fp/tn/fn are present and all non-null, validate their sum.
        has_counts = all(c in out.columns for c in ["tp", "fp", "tn", "fn"])
        if has_counts:
            vals = [r.get("tp"), r.get("fp"), r.get("tn"), r.get("fn")]
            if all(pd.notna(v) for v in vals):
                cm_total = int(float(r["tp"]) + float(r["fp"]) + float(r["tn"]) + float(r["fn"]))
                expected = int(r["n_classified_AF_SR"])
                if cm_total != expected:
                    raise ValueError(f"Confusion-count mismatch for {r['method']}: {cm_total} != {expected}")

    return out


def compute_method_metrics_table(
    df: pd.DataFrame,
    *,
    methods: list[dict[str, str]],
    reference_col: str = "ecg12_4class",
) -> pd.DataFrame:
    """Compute a metrics table for multiple methods against an ECG12 reference.

    This helper is intended for *local / protected-data* usage.
    Public GitHub does not include participant-level matched tables.

    Parameters
    ----------
    df
        Matched cohort table (participant-level) containing an ECG12 reference
        and method outputs.
    methods
        List of method descriptors with keys:
        - `method`: display name
        - `pred_col`: column containing 4-class outputs (AF/SR/OA/UI)
        - optional `prob_col`: column containing AF probability
    reference_col
        Column name for ECG12 reference labels.

    Returns
    -------
    table
        Aggregate metrics table (one row per method).

    Notes
    -----
    This function simply loops `evaluate_method_vs_ecg12` over the provided
    method configurations.
    """

    rows: list[dict[str, Any]] = []
    eval_df = df.rename(columns={reference_col: "ecg12_4class"}) if reference_col != "ecg12_4class" else df
    for m in methods:
        rows.append(
            evaluate_method_vs_ecg12(
                eval_df,
                method=m["method"],
                pred_col=m["pred_col"],
                prob_col=m.get("prob_col"),
            )
        )

    return pd.DataFrame(rows)
