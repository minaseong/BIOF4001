"""src/preprocessing.py

Purpose
-------
Standardize and preprocess raw smartphone PCG WAV recordings.

Why this file exists
--------------------
Smartphone-recorded PCG audio can vary in sampling rate, channel count, and
amplitude scaling. For dissertation reproducibility, downstream feature
extraction expects a **consistent** signal format.

This module implements the same standardization/preprocessing logic used to
produce the dissertation's `AUSC_Standardized/05_preprocessed/*.wav` inputs.

Expected inputs (protected; not included in this repo)
------------------------------------------------------
- A directory of raw smartphone WAV recordings.
  Examples in the original project used filenames such as `iData4023M.wav`.

Expected outputs (protected)
----------------------------
Within `--out-base` the script creates:
- `01_original_backup/`   (copy of raw WAVs)
- `02_standardized/`      (mono, 48 kHz int16 WAV)
- `05_preprocessed/`      (mono, 11,025 Hz int16 WAV after bandpass + normalization)

Workflow step
-------------
Step 1: PCG standardization / preprocessing.

Notes
-----
- Preprocessing is used for **standardization**, not as evidence of denoising.
- No trimming/padding/silence-removal is applied; the full duration is retained.
- Output filenames are normalized to match the project convention:
  `<participant_id>_<original_stem>.wav`, e.g. `4023_iData4023M.wav`.
"""

from __future__ import annotations

import argparse
import re
import shutil
import wave
from dataclasses import dataclass
from fractions import Fraction
from pathlib import Path

import numpy as np
from scipy import signal


@dataclass(frozen=True)
class StandardizeConfig:
    """Configuration for standardizing raw PCG audio."""

    target_sr: int = 48_000


@dataclass(frozen=True)
class PreprocessConfig:
    """Configuration for preprocessing standardized PCG audio."""

    target_sr: int = 11_025
    lowcut_hz: float = 25.0
    highcut_hz: float = 400.0
    filter_order: int = 4


def extract_participant_id(filename: str) -> str:
    """Extract a participant id from raw filenames.

    Parameters
    ----------
    filename
        Raw filename (e.g., `iData4023M.wav`).

    Returns
    -------
    participant_id
        Digits parsed from the filename, or "UNKNOWN".

    Notes
    -----
    The original project used two main naming styles:
    - iPhone recordings: `iData<id>M.wav`
    - other styles starting with digits.
    """

    name = Path(filename).stem
    if name.startswith('iData'):
        m = re.match(r"iData(\d+)M", name)
    else:
        m = re.match(r"(\d+)", name)
    return m.group(1) if m else 'UNKNOWN'


def load_wav_mono_float(path: Path) -> tuple[np.ndarray, int]:
    """Load a WAV file as mono float32 in approximately [-1, 1].

    Parameters
    ----------
    path
        Input WAV path.

    Returns
    -------
    audio, sr
        Mono audio samples as float32 and sampling rate (Hz).

    Notes
    -----
    - Multi-channel audio is reduced to the first channel.
    - Supports 8/16/32-bit PCM. Other formats raise an error.
    - Peak normalization is applied to standardize amplitude scale.
    """

    with wave.open(str(path), 'rb') as wf:
        sr = int(wf.getframerate())
        n_channels = int(wf.getnchannels())
        sampwidth = int(wf.getsampwidth())
        n_frames = int(wf.getnframes())
        raw = wf.readframes(n_frames)

    if sampwidth == 1:
        audio_u8 = np.frombuffer(raw, dtype=np.uint8)
        audio = (audio_u8.astype(np.float32) - 128.0) / 128.0
    elif sampwidth == 2:
        audio_i16 = np.frombuffer(raw, dtype=np.int16)
        audio = audio_i16.astype(np.float32) / 32768.0
    elif sampwidth == 4:
        audio_i32 = np.frombuffer(raw, dtype=np.int32)
        audio = audio_i32.astype(np.float32) / 2147483648.0
    else:
        raise ValueError(f'Unsupported WAV sample width: {sampwidth} bytes')

    if n_channels > 1:
        audio = audio.reshape(-1, n_channels)[:, 0]

    if audio.size == 0:
        return audio.astype(np.float32), sr

    peak = float(np.max(np.abs(audio)))
    if peak > 1e-12:
        audio = audio / peak

    return audio.astype(np.float32), sr


def write_wav_int16_mono(path: Path, audio: np.ndarray, sr: int) -> None:
    """Write mono int16 WAV."""

    audio = np.asarray(audio, dtype=np.float32)
    if audio.size:
        audio = np.clip(audio, -1.0, 1.0)
        frames = (audio * 32767.0).astype(np.int16).tobytes()
    else:
        frames = np.zeros((0,), dtype=np.int16).tobytes()

    path.parent.mkdir(parents=True, exist_ok=True)
    with wave.open(str(path), 'wb') as wf:
        wf.setnchannels(1)
        wf.setsampwidth(2)
        wf.setframerate(int(sr))
        wf.writeframes(frames)


def resample_poly(audio: np.ndarray, orig_sr: int, target_sr: int) -> np.ndarray:
    """Polyphase resampling for high-quality rate conversion."""

    if orig_sr == target_sr or audio.size == 0:
        return audio.astype(np.float32)
    frac = Fraction(int(target_sr), int(orig_sr)).limit_denominator(10_000)
    up, down = frac.numerator, frac.denominator
    return signal.resample_poly(audio, up=up, down=down).astype(np.float32)


def downsample_fft(audio: np.ndarray, orig_sr: int, target_sr: int) -> np.ndarray:
    """FFT-based resampling used in the original project implementation."""

    if orig_sr == target_sr or audio.size == 0:
        return audio.astype(np.float32)
    decimation_factor = orig_sr / float(target_sr)
    num_samples = int(len(audio) / decimation_factor)
    return signal.resample(audio, num_samples).astype(np.float32)


def butter_bandpass(lowcut: float, highcut: float, fs: int, order: int) -> tuple[np.ndarray, np.ndarray]:
    """Design a Butterworth bandpass filter."""

    nyq = 0.5 * float(fs)
    low = float(lowcut) / nyq
    high = float(highcut) / nyq
    b, a = signal.butter(int(order), [low, high], btype='band')
    return b, a


def preprocess_heart_sound(audio: np.ndarray, sr: int, cfg: PreprocessConfig) -> np.ndarray:
    """Bandpass + peak normalization + mean-centering."""

    b, a = butter_bandpass(cfg.lowcut_hz, cfg.highcut_hz, sr, cfg.filter_order)
    filtered = signal.filtfilt(b, a, audio).astype(np.float32)
    normalized = filtered / (np.max(np.abs(filtered)) + 1e-10)
    processed = normalized - float(np.mean(normalized))
    return processed.astype(np.float32)


def standardize_one_wav(source_path: Path, out_path: Path, cfg: StandardizeConfig) -> None:
    """Standardize a raw WAV: mono + resample to 48 kHz, write int16."""

    audio, sr = load_wav_mono_float(source_path)
    audio_rs = resample_poly(audio, sr, cfg.target_sr)
    write_wav_int16_mono(out_path, audio_rs, cfg.target_sr)


def preprocess_one_wav(standardized_path: Path, out_path: Path, cfg: PreprocessConfig) -> None:
    """Preprocess standardized WAV: downsample + bandpass + normalize + mean-center."""

    audio_raw, sr_orig = load_wav_mono_float(standardized_path)
    audio_down = downsample_fft(audio_raw, sr_orig, cfg.target_sr)
    audio_proc = preprocess_heart_sound(audio_down, cfg.target_sr, cfg)
    write_wav_int16_mono(out_path, audio_proc, cfg.target_sr)


def main() -> int:
    """CLI entrypoint for preprocessing (template for protected local runs)."""

    ap = argparse.ArgumentParser(
        description='Standardize raw smartphone PCG WAVs and generate preprocessed WAVs.'
    )
    ap.add_argument('--raw-dir', required=True, help='Directory containing raw WAV files (protected).')
    ap.add_argument('--out-base', required=True, help='Base output directory (protected).')
    ap.add_argument('--dry-run', action='store_true', help='Print actions without writing files.')
    args = ap.parse_args()

    raw_dir = Path(args.raw_dir)
    out_base = Path(args.out_base)

    backup_dir = out_base / '01_original_backup'
    standardized_dir = out_base / '02_standardized'
    preprocessed_dir = out_base / '05_preprocessed'

    std_cfg = StandardizeConfig()
    pre_cfg = PreprocessConfig()

    raw_files = sorted([p for p in raw_dir.iterdir() if p.is_file() and p.suffix.lower() == '.wav'])
    if not raw_files:
        raise SystemExit(f'No .wav files found in: {raw_dir}')

    for src in raw_files:
        participant_id = extract_participant_id(src.name)
        out_name = f'{participant_id}_{src.stem}.wav'

        backup_path = backup_dir / src.name
        standardized_path = standardized_dir / out_name
        preprocessed_path = preprocessed_dir / out_name

        if args.dry_run:
            print(f'Would process: {src.name} -> {preprocessed_path.name}')
            continue

        backup_path.parent.mkdir(parents=True, exist_ok=True)
        if not backup_path.exists():
            shutil.copy2(src, backup_path)

        if not standardized_path.exists():
            standardize_one_wav(src, standardized_path, std_cfg)

        if not preprocessed_path.exists():
            preprocess_one_wav(standardized_path, preprocessed_path, pre_cfg)

    print('Done.')
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
