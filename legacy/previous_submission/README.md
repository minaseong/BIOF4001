# Multimodal Cardiac Data Processing Pipeline

A comprehensive Python-based signal processing pipeline for standardizing and analyzing multimodal cardiac recordings collected for BIOF4001. This pipeline integrates data from four different acquisition modalities and provides consistent processing, feature extraction, and visualization capabilities.

**Modalities Supported:**
- 2-Lead ECG (SQLite format from cardiac devices)
- 12-Lead ECG (DICOM format)
- iPhone Auscultation (WAV audio)
- PIN Sensor Auscultation (CSV amplitude data)

## Project Structure

```
Code_v01/
├── data/
│   ├── raw/
│   │   ├── 12_lead_ecg/         # DICOM files (.dcm)
│   │   ├── 2_lead_ecg/          # SQLite database with 2-lead ECG timeseries
│   │   ├── ausc_iphone/         # iPhone WAV audio recordings
│   │   └── ausc_pin/            # PIN sensor CSV recordings
│   └── processed/
│       └── {patient_id}/        # Output folder per patient
│           ├── 2_lead_ecg/      # Processed 2-lead signals and metrics
│           └── ...
├── notebooks/
│   ├── 00_test.ipynb            # Exploratory analysis and testing
│   └── 01_main_execution.ipynb  # Full pipeline execution
├── src/
│   ├── __init__.py              # Module exports
│   ├── loaders.py               # Data ingestion for all modalities
│   ├── processors.py            # Signal processing and feature extraction
│   ├── utils.py                 # Visualization and file utilities
│   └── __pycache__/
├── requirements.txt             # Python package dependencies
└── README.md
```

## Quick Start

### Installation
```bash
pip install -r requirements.txt
```

### Basic Usage
Import and use the core functions:

```python
from src import (
    load_2lead_ecg, load_dicom_12lead, load_iphone_wav, load_pin_csv,
    process_ecg, process_pcg,
    save_processed_ecg, plot_ecg_analysis, plot_pcg_result
)

# Load 2-lead ECG
ecg_df = load_2lead_ecg('data/raw/2_lead_ecg/4001-4006.sqlite3', patient_id=4001)

# Process and analyze
result = process_ecg(ecg_df['ecg_value'].values, fs=1024, lead_name="2-Lead")
plot_ecg_analysis(result['cleaned_signal'], fs=1024, r_peaks=result['r_peaks'])
```

## Usage & Workflows

### Workflow 1: Interactive Exploration (`00_test.ipynb`)
Use this notebook for:
- **Signal visualization**: Compare raw vs. cleaned signals side-by-side
- **Feature testing**: Experiment with individual processing functions
- **Quality checks**: Inspect data integrity and signal characteristics
- **Parameter tuning**: Test filter cutoffs, peak detection thresholds, etc.

### Workflow 2: Batch Processing (`01_main_execution.ipynb`)
Use this notebook for:
- **Full pipeline execution**: Process entire dataset systematically
- **Reproducible results**: Generate consistent outputs across all patients
- **Automated organization**: Save results to standardized directory structure
- **Summary metrics**: Generate JSON reports with ECG/PCG statistics

**Pipeline steps:**
1. Enumerate all data files from `data/raw/`. Download the data from the Google Drive link submitted along with the code.
2. Group recordings by patient ID
3. Load and process each modality independently
4. Extract standardized features (HR, HRV, envelope peaks, MFCCs)
5. Save processed signals (`.npz`), metrics (`.json`), and visualizations (`.png`)

## Data Formats & Processing Details


### 2-Lead ECG (SQLite)
- **Source**: Cardiac device database
- **Format**: SQLite with `ecgResults` table containing JSON-encoded waveform packets
- **Sampling rate**: 1024 Hz
- **Processing**:
  - Deserialize JSON packets into continuous signal
  - Generate uniform timestamp grid (1 sample per millisecond)
  - Returns: DataFrame with timestamp and ECG value columns

### 12-Lead ECG (DICOM)
- **Source**: Medical imaging DICOM files
- **Format**: Binary waveform data with channel metadata
- **Sampling rate**: Variable (extracted from file)
- **Processing**:
  - Extract multiplexed 16-bit integer data
  - Apply sensitivity correction per channel to convert to physical units (microvolts)
  - Returns: DataFrame with 12 leads + metadata (sampling rate, patient ID, manufacturer)

### iPhone Auscultation (WAV)
- **Source**: iPhone audio recordings of heart sounds
- **Format**: WAV files (variable sample rate)
- **Processing**:
  - Convert to mono and normalize to [-1, 1]
  - Generate timestamps if start time is provided
  - Returns: Signal array + timestamps + metadata

### PIN Sensor Auscultation (CSV)
- **Source**: Contact microphone (PIN sensor) amplitude data
- **Format**: CSV with Amplitude_Raw column
- **Sampling rate**: 4000 Hz (configurable)
- **Processing**:
  - Parse timestamp from filename (YYYY-MM-DD_HH-MM-SS format)
  - Convert to NumPy array and generate uniform timestamps
  - Returns: Signal array + timestamps + metadata

## Signal Processing Pipeline

### ECG Processing (`process_ecg`)
1. **Input validation**: Ensure 1D signal
2. **Cleaning**: Apply NeuroKit2 ECG filter (removes noise and baseline drift)
3. **Peak detection**: Identify R-peaks using NeuroKit2 algorithm
4. **Metrics extraction**: Calculate heart rate and HRV metrics (HR mean/std, SDNN, RMSSD)
5. **Output**: Cleaned signal, R-peak indices, computed metrics

### PCG Processing (`process_pcg`)
1. **Standardization**: Z-score normalization (handles different input scales)
2. **Bandpass filtering**: 20-400 Hz (standard cardiac auscultation range)
3. **Envelope extraction**: Hilbert transform to capture energy profile
4. **Envelope smoothing**: Lowpass filter at 20 Hz to merge internal sound components
5. **Peak detection**: Identify S1/S2 peaks with physiological constraints
6. **Feature extraction**: MFCC coefficients for audio characterization
7. **Output**: Cleaned signal, normalized envelope, peak indices, metrics

## Dependencies

| Library | Purpose | Version |
|---------|---------|---------|
| **NumPy** | Numerical arrays and operations | |
| **Pandas** | Data structures and timestamp handling | |
| **SciPy** | Signal processing (filters, peak detection, Hilbert transform) | |
| **Matplotlib** | Visualization and plot generation | |
| **Librosa** | Audio feature extraction (MFCC) | |
| **NeuroKit2** | ECG signal processing and R-peak detection | |
| **PyDICOM** | Reading DICOM 12-lead ECG files | |

## Notes & Limitations

- **Participant 4002**: Excluded from pipeline due to missing raw data files
- **Dataset size**: Full multimodal dataset is large; this repository contains representative participants
- **Output organization**: Results are saved to `data/processed/{patient_id}/{modality}/`
- **File formats**: 
  - Signals: NumPy `.npz` (compressed binary)
  - Metrics: JSON for easy parsing and analysis
  - Plots: PNG at 150 DPI for documentation