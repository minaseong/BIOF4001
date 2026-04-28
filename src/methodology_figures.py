"""src/methodology_figures.py

Purpose
-------
Provide **optional** plotting helpers for methodology figures when working
with locally available (controlled-access) PCG audio and annotation exports.

This module keeps the methodology section's plotting logic
reusable and keeps the notebook cells short and readable.

Inputs
------
- Local paths to controlled-access WAV files (raw and/or preprocessed).
- Local paths to annotation exports (CSV) and a key CSV mapping file ids to
  participant ids (schema depends on the data export).

Outputs
-------
- Matplotlib Figure objects.

Workflow step
-------------
Optional reporting: methodology figures (local-only).

Notes
-----
- Functions in this file **do not save figures** by default.
- The notebook should call these functions only when local controlled-access
  paths are configured.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd


@dataclass(frozen=True)
class Waveform:
    """In-memory audio waveform container.

    Parameters
    ----------
    samples
        1-D float waveform (mono).
    fs
        Sample rate (Hz).
    """

    samples: np.ndarray
    fs: int


def _to_mono(samples: np.ndarray) -> np.ndarray:
    """Convert multi-channel waveform to mono by averaging channels.

    Parameters
    ----------
    samples
        WAV samples array (shape: [n] or [n, channels]).

    Returns
    -------
    mono
        1-D waveform.
    """

    x = np.asarray(samples)
    if x.ndim == 1:
        return x
    if x.ndim == 2 and x.shape[1] >= 1:
        return x.mean(axis=1)
    return x.reshape(-1)


def load_wav(path: str | Path) -> Waveform:
    """Load a WAV file as a normalized float mono waveform.

    Parameters
    ----------
    path
        Path to a WAV file.

    Returns
    -------
    waveform
        Samples normalized to approximately [-1, 1] and sample rate.

    Notes
    -----
    Uses `scipy.io.wavfile.read`. Only intended for **local** use where the
    controlled-access WAV files are available.
    """

    from scipy.io import wavfile  # local import to keep global import lightweight

    fs, samples = wavfile.read(str(path))
    samples = _to_mono(samples)

    # Convert to float32 with dtype-aware scaling.
    if np.issubdtype(samples.dtype, np.integer):
        info = np.iinfo(samples.dtype)
        x = samples.astype(np.float32) / float(max(abs(info.min), info.max))
    else:
        x = samples.astype(np.float32)

    # Robust centering for visualization (does not claim denoising improvement).
    x = x - float(np.mean(x))

    return Waveform(samples=x, fs=int(fs))


def slice_waveform(w: Waveform, *, start_s: float, duration_s: float) -> Waveform:
    """Slice waveform by time interval.

    Parameters
    ----------
    w
        Input waveform.
    start_s
        Start time in seconds.
    duration_s
        Duration in seconds.

    Returns
    -------
    sliced
        Waveform slice.
    """

    a = int(max(0.0, float(start_s)) * w.fs)
    b = int(max(0.0, float(start_s) + float(duration_s)) * w.fs)
    return Waveform(samples=w.samples[a:b].copy(), fs=w.fs)


def _infer_columns(df: pd.DataFrame) -> dict[str, str]:
    """Infer likely column names for annotation exports.

    This uses heuristics because annotation export schemas differ across tools.
    The goal is to make local regeneration convenient without hard-coding
    institution-specific formats into the public repo.

    Returns
    -------
    mapping
        Dict with keys: segment, start, end, file_id.

    Raises
    ------
    ValueError
        If required columns cannot be inferred.
    """

    cols = {c.lower(): c for c in df.columns}

    def pick(options: list[str]) -> str | None:
        for o in options:
            if o in cols:
                return cols[o]
        return None

    segment = pick(['segment', 'label', 'event', 'type'])
    start = pick(['start', 'start_s', 't_start', 'begin', 'onset', 'x0'])
    end = pick(['end', 'end_s', 't_end', 'stop', 'offset', 'x1'])
    file_id = pick(['fileid', 'file_id', 'file', 'recording', 'recording_id'])

    missing = [k for k, v in {'segment': segment, 'start': start, 'end': end, 'file_id': file_id}.items() if v is None]
    if missing:
        raise ValueError(f'Cannot infer required columns {missing}. Found columns: {list(df.columns)}')

    return {'segment': segment, 'start': start, 'end': end, 'file_id': file_id}


def _infer_key_columns(df: pd.DataFrame) -> dict[str, str]:
    """Infer likely key CSV columns mapping file id to participant id or filename.

    Returns
    -------
    mapping
        Dict with keys:
        - always: `file_id`
        - plus either: `participant_id` and/or `filename`
    """

    cols = {c.lower(): c for c in df.columns}

    def pick(options: list[str]) -> str | None:
        for o in options:
            if o in cols:
                return cols[o]
        return None

    participant_id = pick(['participantid', 'participant_id', 'pid', 'participant'])
    filename = pick(['filename', 'file_name', 'name', 'wav'])
    file_id = pick(['fileid', 'file_id', 'id', 'file', 'recording', 'recording_id'])

    if file_id is None:
        raise ValueError(f"Cannot infer key columns ['file_id']. Found columns: {list(df.columns)}")
    if participant_id is None and filename is None:
        raise ValueError(
            "Cannot infer key columns: need either a participant id column or a filename column "
            f"(columns={list(df.columns)})."
        )

    out = {'file_id': file_id}
    if participant_id is not None:
        out['participant_id'] = participant_id
    if filename is not None:
        out['filename'] = filename
    return out


def _read_csv_light(path: str | Path, *, wanted: set[str]) -> pd.DataFrame:
    """Read a CSV selecting only a small set of columns (best-effort).

    This is important for backend exports that may include very large columns
    (e.g., waveform arrays) that are not needed for plotting.
    """

    p = Path(path)
    # Read header first to decide `usecols`.
    header = pd.read_csv(p, nrows=0, encoding="utf-8-sig")
    cols = header.columns.tolist()
    usecols = [c for c in cols if c in wanted or c.lower() in {w.lower() for w in wanted}]
    if usecols:
        return pd.read_csv(p, encoding="utf-8-sig", usecols=usecols)
    return pd.read_csv(p, encoding="utf-8-sig")


def _parse_timestamp_series(s: pd.Series) -> pd.Series:
    """Parse timestamp strings into pandas datetimes (NaT on failure)."""

    return pd.to_datetime(s, errors="coerce", utc=False)


def load_annotation_payload_for_participant(
    annotation_csv: str | Path,
    key_csv: str | Path,
    *,
    participant_id: str | int,
    filename: str | None = None,
) -> dict[str, list[dict[str, float]]]:
    """Load annotation intervals for one participant and convert to a payload dict.

    Parameters
    ----------
    annotation_csv
        Path to annotation export CSV.
    key_csv
        Path to key CSV mapping file ids to participant ids.
    participant_id
        Participant identifier used in the key CSV.
    filename
        Optional recording filename (e.g., `4023_iData4023M.wav`). If provided
        and the key CSV contains a filename column, this is used to select the
        correct fileId deterministically.

    Returns
    -------
    payload
        Dict with keys: `S1`, `S2`, `poor`, `extremely_poor`. Each key maps to
        a list of intervals with `start` and `end` times in seconds.

    Notes
    -----
    This function uses column-name inference heuristics. If it fails, check the
    error message for the detected column names and adjust your local export or
    pass a pre-parsed payload directly to the plotting function.
    """

    # Read only lightweight columns if possible (backend exports can contain
    # huge waveform arrays not needed for plotting).
    ann = _read_csv_light(Path(annotation_csv), wanted={"id", "fileId", "userId", "segments", "updatedAt", "createdAt", "segment", "start", "end"})
    key = _read_csv_light(Path(key_csv), wanted={"id", "fileId", "filename", "participantId", "participant_id", "createdAt"})

    key_cols = _infer_key_columns(key)

    pid = str(participant_id)
    file_ids: list[str] = []

    if "participant_id" in key_cols:
        file_ids = (
            key[key[key_cols["participant_id"]].astype(str) == pid][key_cols["file_id"]]
            .astype(str)
            .unique()
            .tolist()
        )

    if not file_ids and "filename" in key_cols:
        fcol = key_cols["filename"]
        fidcol = key_cols["file_id"]
        if filename:
            match = key[key[fcol].astype(str) == str(filename)]
        else:
            # Fallback: match any filename starting with "<pid>_"
            match = key[key[fcol].astype(str).str.startswith(f"{pid}_", na=False)]
        file_ids = match[fidcol].astype(str).unique().tolist()

    if not file_ids:
        raise ValueError(f"No file ids found for participant_id={pid} (filename={filename!r}) in key CSV.")

    payload: dict[str, list[dict[str, float]]] = {"S1": [], "S2": [], "poor": [], "extremely_poor": []}

    # Support two common export schemas:
    # 1) "Flat" schema: one interval per row with columns like (segment, start, end, fileId)
    # 2) "Nested" schema: one row per recording with a JSON list column "segments"
    cols_lower = {c.lower(): c for c in ann.columns}
    has_segments_json = "segments" in cols_lower

    if has_segments_json:
        file_id_col = cols_lower.get("fileid") or cols_lower.get("file_id") or cols_lower.get("file")
        if file_id_col is None:
            raise ValueError(
                "Annotation CSV has a 'segments' column but no file id column was found "
                f"(columns={list(ann.columns)})."
            )
        segs_col = cols_lower["segments"]

        sub = ann[ann[file_id_col].astype(str).isin(file_ids)].copy()
        if sub.empty:
            raise ValueError(f'No annotation rows found for participant_id={pid} (file_ids={file_ids}).')

        # If multiple rows exist for the same fileId, keep the latest row.
        if "updatedAt" in sub.columns or "createdAt" in sub.columns:
            ts = _parse_timestamp_series(sub["updatedAt"]) if "updatedAt" in sub.columns else pd.Series([pd.NaT] * len(sub))
            if ts.isna().all() and "createdAt" in sub.columns:
                ts = _parse_timestamp_series(sub["createdAt"])
            sub["_ts"] = ts
            # Prefer latest overall; if multiple fileIds are present, latest per fileId is fine,
            # but for plotting we just take the newest row.
            sub = sub.sort_values(["_ts"], ascending=True)
            sub = sub.tail(1).copy()

        def _pick_key(d: dict[str, Any], options: list[str]) -> Any:
            for o in options:
                for k in (o, o.lower(), o.upper()):
                    if k in d:
                        return d[k]
            return None

        def _parse_segments_cell(cell: Any) -> list[dict[str, Any]]:
            if cell is None or (isinstance(cell, float) and not np.isfinite(cell)):
                return []
            if isinstance(cell, list):
                return [x for x in cell if isinstance(x, dict)]
            if isinstance(cell, dict):
                # Variant A: {"segments": [ ... ]}
                inner = cell.get("segments") if "segments" in cell else None
                if isinstance(inner, list):
                    return [x for x in inner if isinstance(x, dict)]

                # Variant B (observed in backend export): {"S1":[{start,end},...], "S2":[...], "poor":[...], ...}
                flattened: list[dict[str, Any]] = []
                for k, v in cell.items():
                    if isinstance(v, list) and k is not None:
                        for item in v:
                            if isinstance(item, dict):
                                flattened.append({"segment": str(k), **item})
                return flattened
            if isinstance(cell, str):
                s = cell.strip()
                if not s:
                    return []
                try:
                    parsed = json.loads(s)
                except Exception:
                    return []
                return _parse_segments_cell(parsed)
            return []

        for _, r in sub.iterrows():
            seg_list = _parse_segments_cell(r.get(segs_col))
            for seg in seg_list:
                label = _pick_key(seg, ["segment", "label", "type", "name"])
                start = _pick_key(seg, ["start", "start_s", "begin", "onset", "x0"])
                end = _pick_key(seg, ["end", "end_s", "stop", "offset", "x1"])
                if label is None or start is None or end is None:
                    continue
                try:
                    a = float(start)
                    b = float(end)
                except Exception:
                    continue
                if not np.isfinite(a) or not np.isfinite(b) or b <= a:
                    continue
                seg_u = str(label).strip().upper()
                if seg_u in {"S1", "S2"}:
                    payload[seg_u].append({"start": a, "end": b})
                elif seg_u in {"POOR", "EXTREMELY_POOR", "EXTREMELY POOR", "EXTREME", "BAD"}:
                    if "EXT" in seg_u:
                        payload["extremely_poor"].append({"start": a, "end": b})
                    else:
                        payload["poor"].append({"start": a, "end": b})
    else:
        ann_cols = _infer_columns(ann)
        sub = ann[ann[ann_cols['file_id']].astype(str).isin(file_ids)].copy()
        if sub.empty:
            raise ValueError(f'No annotation rows found for participant_id={pid} (file_ids={file_ids}).')

        for _, r in sub.iterrows():
            seg = str(r[ann_cols['segment']]).strip()
            try:
                a = float(r[ann_cols['start']])
                b = float(r[ann_cols['end']])
            except Exception:
                continue
            if not np.isfinite(a) or not np.isfinite(b) or b <= a:
                continue
            seg_u = seg.upper()
            if seg_u in {"S1", "S2"}:
                payload[seg_u].append({"start": a, "end": b})
            elif seg_u in {"POOR", "EXTREMELY_POOR", "EXTREMELY POOR", "EXTREME", "BAD"}:
                if "EXT" in seg_u:
                    payload["extremely_poor"].append({"start": a, "end": b})
                else:
                    payload["poor"].append({"start": a, "end": b})

    return payload


def plot_preprocessing_example(
    *,
    raw_wav: str | Path,
    preprocessed_wav: str | Path,
    start_s: float,
    duration_s: float,
    payload: dict[str, list[dict[str, float]]] | None = None,
):
    """Plot raw vs preprocessed waveform for a short window (Figure M1-style).

    Parameters
    ----------
    raw_wav
        Path to raw WAV.
    preprocessed_wav
        Path to preprocessed WAV.
    start_s, duration_s
        Time window.

    Returns
    -------
    fig
        Matplotlib figure.

    Notes
    -----
    This is a transparency figure. It does not claim that preprocessing improves
    performance; it simply shows the standardization applied.
    """

    import matplotlib.pyplot as plt

    t0 = float(start_s)
    t1 = float(start_s) + float(duration_s)

    w_raw = slice_waveform(load_wav(raw_wav), start_s=t0, duration_s=duration_s)
    w_pre = slice_waveform(load_wav(preprocessed_wav), start_s=t0, duration_s=duration_s)

    t_raw = np.arange(w_raw.samples.size, dtype=float) / float(w_raw.fs) + t0
    t_pre = np.arange(w_pre.samples.size, dtype=float) / float(w_pre.fs) + t0

    def _intervals(key: str) -> list[tuple[float, float]]:
        out: list[tuple[float, float]] = []
        if not payload:
            return out
        for seg in (payload.get(key) or []):
            try:
                a = float(seg.get("start"))
                b = float(seg.get("end"))
            except Exception:
                continue
            if np.isfinite(a) and np.isfinite(b) and b > a:
                out.append((a, b))
        out.sort()
        return out

    def _spans_in_window(ints: list[tuple[float, float]], a: float, b: float) -> list[tuple[float, float]]:
        out: list[tuple[float, float]] = []
        for x0, x1 in ints:
            y0, y1 = max(float(a), float(x0)), min(float(b), float(x1))
            if y1 > y0:
                out.append((y0, y1))
        return out

    def _draw_interval_spans(ax: Any, *, spans: list[tuple[float, float]], color: str, alpha: float, label: str) -> None:
        if not spans:
            return
        for a, b in spans:
            ax.axvspan(float(a), float(b), color=color, alpha=float(alpha), lw=0)
        ax.plot([], [], color=color, lw=6.0, alpha=float(alpha), label=label)

    s1_color = "#d62728"  # red
    s2_color = "#1f77b4"  # blue

    fig = plt.figure(figsize=(10.5, 6.2))
    gs = fig.add_gridspec(3, 1, hspace=0.35)

    ax1 = fig.add_subplot(gs[0, 0])
    ax1.plot(t_raw, w_raw.samples, lw=0.8, color="#1f77b4")
    ax1.set_title("A. Raw smartphone PCG (example window)", loc="left", fontsize=11)
    ax1.set_ylabel("Amplitude")
    ax1.set_xlim(float(t0), float(t1))
    ax1.grid(False)

    ax2 = fig.add_subplot(gs[1, 0], sharex=ax1)
    ax2.plot(t_pre, w_pre.samples, lw=0.8, color="#2ca02c")
    ax2.set_title("B. Preprocessed PCG (bandpass + normalization)", loc="left", fontsize=11)
    ax2.set_ylabel("Normalized amplitude")
    ax2.set_xlim(float(t0), float(t1))
    ax2.grid(False)

    ax3 = fig.add_subplot(gs[2, 0], sharex=ax1)
    ax3.plot(t_pre, w_pre.samples, lw=0.8, color="#2ca02c")
    s1_spans = _spans_in_window(_intervals("S1"), float(t0), float(t1))
    s2_spans = _spans_in_window(_intervals("S2"), float(t0), float(t1))
    _draw_interval_spans(ax3, spans=s1_spans, color=s1_color, alpha=0.14, label="S1")
    _draw_interval_spans(ax3, spans=s2_spans, color=s2_color, alpha=0.14, label="S2")
    ax3.set_title("C. Clean segment for RR extraction (S1/S2 markers)", loc="left", fontsize=11)
    ax3.set_xlabel("Time (s)")
    ax3.set_ylabel("Normalized amplitude")
    ax3.set_xlim(float(t0), float(t1))
    ax3.legend(loc="upper right", frameon=False, fontsize=9, ncols=2)
    ax3.grid(False)

    fig.tight_layout()
    return fig


def plot_s1s2_annotation_example(
    *,
    preprocessed_wav: str | Path,
    payload: dict[str, list[dict[str, float]]],
    start_s: float,
    duration_s: float,
):
    """Plot S1/S2 interval bands and poor-quality regions over waveform.

    Parameters
    ----------
    preprocessed_wav
        Path to preprocessed WAV.
    payload
        Annotation payload dict with keys: `S1`, `S2`, `poor`, `extremely_poor`.
        Values are lists of {'start','end'} intervals in seconds.
    start_s, duration_s
        Time window.

    Returns
    -------
    fig
        Matplotlib figure.

    Notes
    -----
    This uses interval **ranges** (bands) rather than onset markers, matching the
    GUI-style annotation concept of labeling an interval around S1/S2.
    """

    import matplotlib.pyplot as plt

    w = slice_waveform(load_wav(preprocessed_wav), start_s=start_s, duration_s=duration_s)
    t = np.arange(w.samples.size) / float(w.fs) + float(start_s)

    fig, ax = plt.subplots(1, 1, figsize=(10.5, 3.6))

    a = float(start_s)
    b = float(start_s) + float(duration_s)

    def _intervals(key: str) -> list[tuple[float, float]]:
        out: list[tuple[float, float]] = []
        for seg in (payload.get(key) or []):
            try:
                x0 = float(seg.get("start"))
                x1 = float(seg.get("end"))
            except Exception:
                continue
            if np.isfinite(x0) and np.isfinite(x1) and x1 > x0:
                out.append((x0, x1))
        out.sort()
        return out

    def _spans_in_window(ints: list[tuple[float, float]], x0: float, x1: float) -> list[tuple[float, float]]:
        out: list[tuple[float, float]] = []
        for a0, a1 in ints:
            y0, y1 = max(float(x0), float(a0)), min(float(x1), float(a1))
            if y1 > y0:
                out.append((y0, y1))
        return out

    def _draw_interval_spans(ax: Any, *, spans: list[tuple[float, float]], color: str, alpha: float, label: str) -> None:
        if not spans:
            return
        for x0, x1 in spans:
            ax.axvspan(float(x0), float(x1), color=color, alpha=float(alpha), lw=0)
        ax.plot([], [], color=color, lw=6.0, alpha=float(alpha), label=label)

    def _shrink_spans(
        spans: list[tuple[float, float]],
        *,
        shrink: float = 0.7,
        min_width_s: float = 0.02,
    ) -> list[tuple[float, float]]:
        out: list[tuple[float, float]] = []
        for x0, x1 in spans:
            x0 = float(x0)
            x1 = float(x1)
            if not (np.isfinite(x0) and np.isfinite(x1)) or x1 <= x0:
                continue
            mid = 0.5 * (x0 + x1)
            w = max(min_width_s, (x1 - x0) * float(shrink))
            out.append((mid - 0.5 * w, mid + 0.5 * w))
        return out

    # Poor / extremely poor shading behind everything else.
    poor_any = False
    for x0, x1 in _intervals("poor"):
        y0, y1 = max(x0, a), min(x1, b)
        if y1 > y0:
            poor_any = True
            ax.axvspan(y0, y1, color="0.86", alpha=0.18, lw=0, zorder=0)
    for x0, x1 in _intervals("extremely_poor"):
        y0, y1 = max(x0, a), min(x1, b)
        if y1 > y0:
            poor_any = True
            ax.axvspan(y0, y1, color="0.78", alpha=0.30, lw=0, zorder=0)

    # Waveform
    ax.plot(t, w.samples, lw=0.9, color="#2ca02c", zorder=2)

    # S1/S2 as GUI-like highlighted bands (slightly narrowed for readability).
    s1_spans = _shrink_spans(_spans_in_window(_intervals("S1"), a, b), shrink=0.7, min_width_s=0.02)
    s2_spans = _shrink_spans(_spans_in_window(_intervals("S2"), a, b), shrink=0.7, min_width_s=0.02)
    _draw_interval_spans(ax, spans=s1_spans, color="#e15759", alpha=0.16, label="S1")  # soft pink/red
    _draw_interval_spans(ax, spans=s2_spans, color="#1f77b4", alpha=0.16, label="S2")  # soft blue

    ax.set_title("S1/S2 annotation example", fontsize=11)
    ax.set_xlabel('Time (s)')
    ax.set_ylabel('Normalized amplitude')
    ax.grid(False)

    # Legend (compact, above the plot)
    if poor_any:
        ax.plot([], [], color="0.82", lw=6.0, alpha=0.22, label="Poor/extremely poor")
    ax.legend(
        loc="upper center",
        bbox_to_anchor=(0.5, 1.22),
        frameon=False,
        fontsize=8,
        ncols=3,
        columnspacing=1.3,
        handlelength=2.4,
    )

    fig.tight_layout()
    return fig


def infer_default_protected_paths(
    *,
    data_root: str | Path,
    participant_id: str | int,
    raw_subdir: str = "AUSC iPhone",
    preprocessed_subdir: str = "AUSC_Standardized/05_preprocessed",
    annotation_subdir: str = "Label/Annotation",
    annotation_csv_name: str = "Annotation 5 APR.csv",
    key_csv_name: str = "FileId_ParticipantId_Key.csv",
) -> dict[str, Path]:
    """
    Infer local (controlled-access) file paths for methodology plotting.

    Parameters
    ----------
    data_root
        Root directory that contains the controlled-access dataset folders
        (e.g., the directory that contains `AUSC iPhone/`, `AUSC_Standardized/`,
        and `Label/`).
    participant_id
        Participant identifier (used to search filenames).
    raw_subdir
        Relative path to the raw smartphone PCG folder under `data_root`.
    preprocessed_subdir
        Relative path to the standardized/preprocessed PCG folder under `data_root`.
    annotation_subdir
        Relative path to the annotation folder under `data_root`.
    annotation_csv_name
        Annotation CSV filename inside `annotation_subdir`.
    key_csv_name
        Key CSV filename inside `annotation_subdir`.

    Returns
    -------
    paths
        Dict with keys:
        - `raw_wav`
        - `preprocessed_wav`
        - `annotation_csv`
        - `annotation_key_csv`

    Notes
    -----
    This helper is best-effort and uses glob patterns so the public repository
    does not need to hard-code institution-specific absolute paths.
    """

    root = Path(data_root).expanduser().resolve()
    pid = str(participant_id)

    raw_dir = root / raw_subdir
    pre_dir = root / preprocessed_subdir
    ann_dir = root / annotation_subdir

    # Prefer a deterministic match when possible.
    raw_candidates = sorted(raw_dir.glob(f"*{pid}*.wav")) if raw_dir.exists() else []
    pre_candidates = sorted(pre_dir.glob(f"{pid}_*.wav")) if pre_dir.exists() else []

    if not raw_candidates:
        raise FileNotFoundError(f"No raw WAV found for participant_id={pid} under {raw_dir}")
    if not pre_candidates:
        raise FileNotFoundError(f"No preprocessed WAV found for participant_id={pid} under {pre_dir}")

    annotation_csv = ann_dir / annotation_csv_name
    key_csv = ann_dir / key_csv_name
    if not annotation_csv.exists():
        raise FileNotFoundError(f"Annotation CSV not found: {annotation_csv}")
    if not key_csv.exists():
        raise FileNotFoundError(f"Annotation key CSV not found: {key_csv}")

    return {
        "raw_wav": raw_candidates[0],
        "preprocessed_wav": pre_candidates[0],
        "annotation_csv": annotation_csv,
        "annotation_key_csv": key_csv,
    }
