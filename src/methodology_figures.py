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
    """Infer likely key CSV columns mapping file id to participant id.

    Returns
    -------
    mapping
        Dict with keys: participant_id, file_id.
    """

    cols = {c.lower(): c for c in df.columns}

    def pick(options: list[str]) -> str | None:
        for o in options:
            if o in cols:
                return cols[o]
        return None

    participant_id = pick(['participantid', 'participant_id', 'pid', 'participant'])
    file_id = pick(['fileid', 'file_id', 'file', 'recording', 'recording_id'])

    missing = [k for k, v in {'participant_id': participant_id, 'file_id': file_id}.items() if v is None]
    if missing:
        raise ValueError(f'Cannot infer key columns {missing}. Found columns: {list(df.columns)}')

    return {'participant_id': participant_id, 'file_id': file_id}


def load_annotation_payload_for_participant(
    annotation_csv: str | Path,
    key_csv: str | Path,
    *,
    participant_id: str | int,
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

    ann = pd.read_csv(annotation_csv)
    key = pd.read_csv(key_csv)

    key_cols = _infer_key_columns(key)

    pid = str(participant_id)
    file_ids = key[key[key_cols['participant_id']].astype(str) == pid][key_cols['file_id']].astype(str).unique().tolist()
    if not file_ids:
        raise ValueError(f'No file ids found for participant_id={pid} in key CSV.')

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
                inner = cell.get("segments") if "segments" in cell else None
                if isinstance(inner, list):
                    return [x for x in inner if isinstance(x, dict)]
                return []
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
):
    """Plot raw vs preprocessed waveform for a short window.

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

    w_raw = slice_waveform(load_wav(raw_wav), start_s=start_s, duration_s=duration_s)
    w_pre = slice_waveform(load_wav(preprocessed_wav), start_s=start_s, duration_s=duration_s)

    t_raw = np.arange(w_raw.samples.size) / float(w_raw.fs)
    t_pre = np.arange(w_pre.samples.size) / float(w_pre.fs)

    fig, axes = plt.subplots(2, 1, figsize=(10.5, 4.8), sharex=False)
    axes[0].plot(t_raw, w_raw.samples, lw=0.8)
    axes[0].set_title('Raw smartphone PCG (example window)')
    axes[0].set_xlabel('Time (s)')
    axes[0].set_ylabel('Amplitude')

    axes[1].plot(t_pre, w_pre.samples, lw=0.8)
    axes[1].set_title('Preprocessed PCG (example window)')
    axes[1].set_xlabel('Time (s)')
    axes[1].set_ylabel('Normalized amplitude')

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

    fig, ax = plt.subplots(1, 1, figsize=(10.5, 3.8))

    a = float(start_s)
    b = float(start_s) + float(duration_s)

    # Background: poor / extremely poor shading behind everything else.
    for key, alpha in [("poor", 0.22), ("extremely_poor", 0.35)]:
        for seg in payload.get(key) or []:
            try:
                s0 = float(seg.get('start'))
                s1 = float(seg.get('end'))
            except Exception:
                continue
            if s1 <= a or s0 >= b:
                continue
            ax.axvspan(max(a, s0), min(b, s1), color='#bdbdbd', alpha=alpha, zorder=0)

    # Waveform
    ax.plot(t, w.samples, lw=0.8, color='#1f77b4', zorder=2)

    # S1/S2 bands (drawn above grey background, below waveform markers)
    for seg in payload.get('S1') or []:
        try:
            s0 = float(seg.get('start'))
            s1 = float(seg.get('end'))
        except Exception:
            continue
        if s1 <= a or s0 >= b:
            continue
        ax.axvspan(max(a, s0), min(b, s1), color='#ff6b6b', alpha=0.25, zorder=1)

    for seg in payload.get('S2') or []:
        try:
            s0 = float(seg.get('start'))
            s1 = float(seg.get('end'))
        except Exception:
            continue
        if s1 <= a or s0 >= b:
            continue
        ax.axvspan(max(a, s0), min(b, s1), color='#4dabf7', alpha=0.25, zorder=1)

    ax.set_title('S1/S2 annotation and quality labeling (example window)')
    ax.set_xlabel('Time (s)')
    ax.set_ylabel('Normalized amplitude')

    # Compact legend (no clean-segment indicator by default)
    from matplotlib.patches import Patch

    handles = [
        Patch(facecolor='#bdbdbd', edgecolor='none', alpha=0.22, label='Poor/extremely poor'),
        Patch(facecolor='#ff6b6b', edgecolor='none', alpha=0.25, label='S1'),
        Patch(facecolor='#4dabf7', edgecolor='none', alpha=0.25, label='S2'),
    ]
    ax.legend(handles=handles, loc='upper center', bbox_to_anchor=(0.5, 1.18), ncol=3, frameon=False)

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
