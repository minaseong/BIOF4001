"""Helpers to discover protected-local inputs for figure regeneration.

These utilities keep notebooks portable across machines by resolving common
path layouts used in local (non-public) research folders.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True)
class LocalFigureSources:
    """Resolved local paths used by the notebook's protected-local mode."""

    run_dir: Path | None = None
    matched_table_csv: Path | None = None
    pcg_predictions_csv: Path | None = None
    labels_csv: Path | None = None
    raw_audio_root: Path | None = None
    preprocessed_audio_root: Path | None = None
    annotation_csv: Path | None = None
    annotation_key_csv: Path | None = None


def _first_existing(candidates: list[Path]) -> Path | None:
    for p in candidates:
        if p.exists():
            return p
    return None


def _candidate_roots(data_root: Path | None) -> list[Path]:
    if data_root is None:
        return []
    root = data_root.expanduser().resolve()
    roots = [root]
    if (root / "vitogram_pcg").exists():
        roots.append((root / "vitogram_pcg").resolve())
    if root.name == "vitogram_pcg":
        roots.append(root.parent.resolve())
    return roots


def discover_local_figure_sources(
    *,
    data_root: Path | None,
    run_dir_hint: Path | None = None,
    run_name: str = "rr_clean_10s_rf",
) -> LocalFigureSources:
    """Discover local file paths for dissertation-style figure regeneration.

    Parameters
    ----------
    data_root
        Root folder that may contain either:
        - a `vitogram_pcg/` project folder, or
        - the parent folder containing `vitogram_pcg/` and `Label/Annotation/`.
    run_dir_hint
        Optional explicit run directory path.
    run_name
        Run directory name under `logs/`.
    """

    roots = _candidate_roots(data_root)
    run_candidates: list[Path] = []
    if run_dir_hint is not None:
        run_candidates.append(run_dir_hint.expanduser().resolve())
    for r in roots:
        run_candidates.extend(
            [
                r / "logs" / run_name,
                r / "vitogram_pcg" / "logs" / run_name,
                r / run_name,
            ]
        )
    run_dir = _first_existing(run_candidates)

    matched_table_csv = _first_existing(
        [
            *([] if run_dir is None else [run_dir / "dissertation_figures" / "final_matched_metrics_table.csv"]),
            *([] if run_dir is None else [run_dir / "dissertation_figures" / "final_metrics_table.csv"]),
            *[r / "data" / "interim" / "master_table_with_pcg.csv" for r in roots],
            *[r / "vitogram_pcg" / "data" / "interim" / "master_table_with_pcg.csv" for r in roots],
        ]
    )

    pcg_predictions_csv = _first_existing(
        [
            *([] if run_dir is None else [run_dir / "af_rr" / "oof_predictions.csv"]),
            *([] if run_dir is None else [run_dir / "af_rr" / "full_predictions.csv"]),
        ]
    )

    labels_csv = _first_existing(
        [
            *[r / "Label" / "Annotation" / "device_labels_4class.csv" for r in roots],
            *[r.parent / "Label" / "Annotation" / "device_labels_4class.csv" for r in roots],
        ]
    )

    raw_audio_root = _first_existing(
        [
            *[r / "AUSC iPhone" for r in roots],
            *[r / "data" / "raw" / "AUSC iPhone" for r in roots],
        ]
    )
    preprocessed_audio_root = _first_existing(
        [
            *[r / "AUSC_Standardized" / "05_preprocessed" for r in roots],
            *[r / "data" / "processed" / "AUSC_Standardized" / "05_preprocessed" for r in roots],
        ]
    )
    annotation_csv = _first_existing(
        [
            *[r / "Label" / "Annotation" / "Annotation 5 APR.csv" for r in roots],
            *[r.parent / "Label" / "Annotation" / "Annotation 5 APR.csv" for r in roots],
        ]
    )
    annotation_key_csv = _first_existing(
        [
            *[r / "Label" / "Annotation" / "FileId_ParticipantId_Key.csv" for r in roots],
            *[r.parent / "Label" / "Annotation" / "FileId_ParticipantId_Key.csv" for r in roots],
        ]
    )

    return LocalFigureSources(
        run_dir=run_dir,
        matched_table_csv=matched_table_csv,
        pcg_predictions_csv=pcg_predictions_csv,
        labels_csv=labels_csv,
        raw_audio_root=raw_audio_root,
        preprocessed_audio_root=preprocessed_audio_root,
        annotation_csv=annotation_csv,
        annotation_key_csv=annotation_key_csv,
    )
