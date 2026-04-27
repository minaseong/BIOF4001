"""src/clean_segments.py

Purpose
-------
Implements the "clean segment" selection logic using S1/S2 and quality annotations.

The RR-feature AF/SR model requires analyzable intervals where heart cycles are
consistently labeled (S1 and S2) and not interrupted by poor-quality regions.
Recordings without qualifying clean segments are treated as *uninterpretable (UI)*
for AF/SR classification, and are counted in coverage (not in sensitivity/specificity).

Inputs
------
- Annotation payload for a recording (JSON-like dict with keys such as `S1`, `S2`,
  `poor`, `extremely_poor`).
- Recording duration (seconds).

Outputs
-------
- List of clean segments with start/end/duration and cycle statistics.

Workflow step
-------------
Step 2: Clean-segment identification.

Notes
-----
- S1/S2 are stored as *intervals* in the backend; this code uses the `start`
  timestamps as onsets.
- The rules mirror the thresholds used in the reported analysis run.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np


def _as_list(x: Any) -> list[dict[str, Any]]:
    return x if isinstance(x, list) else []


def _segments_to_intervals(segments: list[dict[str, Any]]) -> list[tuple[float, float]]:
    """Parse a list of {start,end} dicts into sorted (start,end) intervals."""

    intervals: list[tuple[float, float]] = []
    for s in _as_list(segments):
        try:
            a = float(s.get("start"))
            b = float(s.get("end"))
        except Exception:
            continue
        if not np.isfinite(a) or not np.isfinite(b) or b <= a:
            continue
        intervals.append((a, b))
    intervals.sort()
    return intervals


def merge_intervals(intervals: list[tuple[float, float]]) -> list[tuple[float, float]]:
    """Merge overlapping intervals."""

    if not intervals:
        return []
    intervals = sorted(intervals)
    out = [intervals[0]]
    for a, b in intervals[1:]:
        la, lb = out[-1]
        if a <= lb:
            out[-1] = (la, max(lb, b))
        else:
            out.append((a, b))
    return out


def complement_intervals(total: tuple[float, float], bad: list[tuple[float, float]]) -> list[tuple[float, float]]:
    """Return intervals in `total` not covered by `bad`."""

    a0, b0 = float(total[0]), float(total[1])
    if b0 <= a0:
        return []

    bad = [(max(a0, a), min(b0, b)) for a, b in bad]
    bad = [(a, b) for a, b in bad if b > a]
    bad = merge_intervals(bad)
    if not bad:
        return [(a0, b0)]

    out: list[tuple[float, float]] = []
    cur = a0
    for a, b in bad:
        if a > cur:
            out.append((cur, a))
        cur = max(cur, b)
    if cur < b0:
        out.append((cur, b0))
    return out


def onsets_in_interval(onsets: np.ndarray, interval: tuple[float, float]) -> np.ndarray:
    """Filter onset times to those inside [start,end]."""

    a, b = float(interval[0]), float(interval[1])
    if onsets.size == 0:
        return onsets.astype(float)
    m = (onsets >= a) & (onsets <= b)
    return onsets[m].astype(float)


@dataclass(frozen=True)
class CleanSegmentConfig:
    """Thresholds for selecting a clean segment."""

    min_duration_s: float = 10.0
    min_cycles: int = 10
    rr_min_s: float = 0.3
    rr_max_s: float = 1.8
    max_bad_cycles: int = 2
    max_consecutive_bad_cycles: int = 2
    min_ok_fraction: float = 0.9
    merge_gap_s: float = 0.25
    onset_dedup_s: float = 0.05


def _count_s2_between(s2_onsets: np.ndarray, a: float, b: float) -> int:
    if s2_onsets.size == 0:
        return 0
    return int(np.sum((s2_onsets > a) & (s2_onsets < b)))


def _dedup_onsets(onsets: np.ndarray, *, tol_s: float) -> np.ndarray:
    """Deduplicate nearly-identical onsets (prevents double-label artifacts)."""

    if onsets.size == 0:
        return onsets.astype(float)
    xs = np.asarray(sorted([float(x) for x in onsets if np.isfinite(x)]), dtype=float)
    if xs.size == 0:
        return xs
    out = [float(xs[0])]
    for x in xs[1:]:
        if float(x) - float(out[-1]) > float(tol_s):
            out.append(float(x))
    return np.asarray(out, dtype=float)


def _merge_intervals_with_gap(intervals: list[tuple[float, float]], *, gap_s: float) -> list[tuple[float, float]]:
    if not intervals:
        return []
    intervals = sorted(intervals)
    out = [intervals[0]]
    for a, b in intervals[1:]:
        la, lb = out[-1]
        if float(a) <= float(lb) + float(gap_s):
            out[-1] = (float(la), float(max(lb, b)))
        else:
            out.append((float(a), float(b)))
    return out


def _segment_stats(
    s1_on: np.ndarray,
    s2_on: np.ndarray,
    *,
    start_s: float,
    end_s: float,
    cfg: CleanSegmentConfig,
) -> dict[str, float]:
    """Compute cycle statistics for a candidate segment."""

    s1w = onsets_in_interval(s1_on, (float(start_s), float(end_s)))
    if s1w.size < 2:
        return {
            "start_s": float(start_s),
            "end_s": float(end_s),
            "duration_s": float(end_s - start_s),
            "cycles_total": 0.0,
            "cycles_ok": 0.0,
            "cycles_bad": 0.0,
            "ok_fraction": float("nan"),
            "max_bad_streak": float("nan"),
        }

    ok: list[bool] = []
    bad_streak = 0
    max_bad_streak = 0
    for i in range(0, s1w.size - 1):
        a = float(s1w[i])
        b = float(s1w[i + 1])
        rr = b - a
        ok_rr = (rr >= float(cfg.rr_min_s)) and (rr <= float(cfg.rr_max_s))
        ok_s2 = _count_s2_between(s2_on, a, b) == 1
        is_ok = bool(ok_rr and ok_s2)
        ok.append(is_ok)
        if is_ok:
            bad_streak = 0
        else:
            bad_streak += 1
            max_bad_streak = max(max_bad_streak, bad_streak)

    cycles_total = int(len(ok))
    cycles_ok = int(sum(1 for v in ok if v))
    cycles_bad = int(cycles_total - cycles_ok)
    ok_fraction = float(cycles_ok / cycles_total) if cycles_total > 0 else float("nan")
    return {
        "start_s": float(start_s),
        "end_s": float(end_s),
        "duration_s": float(end_s - start_s),
        "cycles_total": float(cycles_total),
        "cycles_ok": float(cycles_ok),
        "cycles_bad": float(cycles_bad),
        "ok_fraction": float(ok_fraction),
        "max_bad_streak": float(max_bad_streak) if cycles_total > 0 else float("nan"),
    }


def _passes_clean_thresholds(stats: dict[str, float], *, cfg: CleanSegmentConfig) -> bool:
    """Return True if a candidate run satisfies clean-segment thresholds."""

    dur = float(stats.get("duration_s", float("nan")))
    cycles_ok = float(stats.get("cycles_ok", float("nan")))
    cycles_total = float(stats.get("cycles_total", float("nan")))
    cycles_bad = float(stats.get("cycles_bad", float("nan")))
    ok_fraction = float(stats.get("ok_fraction", float("nan")))
    max_bad_streak = float(stats.get("max_bad_streak", float("nan")))

    if not np.isfinite(dur) or dur < float(cfg.min_duration_s):
        return False
    if not np.isfinite(cycles_ok) or cycles_ok < float(cfg.min_cycles):
        return False
    if np.isfinite(cycles_bad) and cycles_bad > float(cfg.max_bad_cycles):
        return False
    if np.isfinite(max_bad_streak) and max_bad_streak > float(cfg.max_consecutive_bad_cycles):
        return False
    if np.isfinite(ok_fraction) and ok_fraction < float(cfg.min_ok_fraction):
        return False
    if np.isfinite(cycles_total) and cycles_total > 0 and np.isfinite(cycles_ok):
        if cycles_ok / cycles_total < float(cfg.min_ok_fraction):
            return False
    return True


def find_clean_segments(payload: dict[str, Any], *, duration_s: float, cfg: CleanSegmentConfig) -> list[dict[str, float]]:
    """Find clean segments in a recording.

    Parameters
    ----------
    payload
        Annotation dict with keys `S1`, `S2`, `poor`, `extremely_poor`.
    duration_s
        Recording duration (seconds).
    cfg
        Clean-segment thresholds.

    Returns
    -------
    segments
        List of segments with start/end/duration and cycle statistics.

    Notes
    -----
    This implements the clean-segment logic used for the reported analysis:
    - exclude poor/extremely_poor
    - require mostly-labeled cycles where each S1->S1 interval has exactly one S2
    - enforce duration and cycle thresholds
    """

    if not payload:
        return []
    dur = float(duration_s) if np.isfinite(duration_s) else float("nan")
    if not np.isfinite(dur) or dur <= 0:
        return []

    bad = _segments_to_intervals(payload.get("poor") or []) + _segments_to_intervals(payload.get("extremely_poor") or [])
    bad = merge_intervals(bad)
    good = complement_intervals((0.0, dur), bad)

    s1 = _segments_to_intervals(payload.get("S1") or [])
    s2 = _segments_to_intervals(payload.get("S2") or [])

    # Use start times as onsets.
    s1_on = _dedup_onsets(np.asarray([a for a, _b in s1], dtype=float), tol_s=float(cfg.onset_dedup_s))
    s2_on = _dedup_onsets(np.asarray([a for a, _b in s2], dtype=float), tol_s=float(cfg.onset_dedup_s))
    if s1_on.size < 2:
        return []

    candidate_intervals: list[tuple[float, float]] = []
    for ga, gb in good:
        ga = float(ga)
        gb = float(gb)
        if gb <= ga:
            continue

        idx = np.where((s1_on >= ga) & (s1_on <= gb))[0]
        if idx.size < 2:
            continue
        s1g = s1_on[idx]

        ok = []
        for i in range(0, s1g.size - 1):
            a = float(s1g[i])
            b = float(s1g[i + 1])
            rr = b - a
            ok_rr = (rr >= float(cfg.rr_min_s)) and (rr <= float(cfg.rr_max_s))
            ok_s2 = _count_s2_between(s2_on, a, b) == 1
            ok.append(bool(ok_rr and ok_s2))
        if not ok:
            continue

        # Find maximal windows allowing <= max_bad_cycles bad cycles.
        best_end_for_start: dict[int, int] = {}
        start = 0
        bad_positions: list[int] = []
        for end in range(0, len(ok)):
            if not ok[end]:
                bad_positions.append(end)
            while len(bad_positions) > int(cfg.max_bad_cycles):
                start = int(bad_positions.pop(0)) + 1
                bad_positions = [p for p in bad_positions if p >= start]
            prev = best_end_for_start.get(int(start))
            if prev is None or int(end) > int(prev):
                best_end_for_start[int(start)] = int(end)

        segments: list[tuple[int, int]] = [(s, e) for s, e in best_end_for_start.items() if e >= s]
        segments.sort(key=lambda t: (t[0], -t[1]))

        kept: list[tuple[int, int]] = []
        max_end = -1
        for s, e in segments:
            if e <= max_end:
                continue
            kept.append((s, e))
            max_end = max(max_end, e)

        for s, e in kept:
            a = float(s1g[s])
            b = float(s1g[e + 1])
            if b > a:
                candidate_intervals.append((a, b))

    merged = _merge_intervals_with_gap(candidate_intervals, gap_s=float(cfg.merge_gap_s))
    labeled = [_segment_stats(s1_on, s2_on, start_s=a, end_s=b, cfg=cfg) for a, b in merged]
    clean = [d for d in labeled if _passes_clean_thresholds(d, cfg=cfg)]
    return clean


def pick_longest_clean_interval(payload: dict[str, Any], *, duration_s: float, cfg: CleanSegmentConfig) -> tuple[float, float] | None:
    """Select the longest clean interval, if any."""

    segs = find_clean_segments(payload, duration_s=float(duration_s), cfg=cfg)
    if not segs:
        return None
    best = max(segs, key=lambda d: float(d["duration_s"]))
    return float(best["start_s"]), float(best["end_s"])
