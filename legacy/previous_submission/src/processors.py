import numpy as np
import pandas as pd
import neurokit2 as nk

import scipy.signal
import librosa


def process_ecg(signal, fs, lead_name="signal"):
    """
    Processes a SINGLE channel of ECG.
    
    If you have 12-lead data, you should call this function 
    specifically on the lead you want to analyze (e.g., Lead II).
    
    Args:
        signal (array-like): 1D array of voltage values.
        fs (float): Sampling rate.
        lead_name (str): Label for the data (e.g., 'Lead II').
    
    Returns:
        dict: Standardized output with cleaned signal, peaks, and metrics.
    """
    signal = np.array(signal)
    
    # Validate input: Signal must be 1D array
    if signal.ndim > 1:
        raise ValueError(f"process_ecg expects a 1D array. Got shape {signal.shape}. "
                         "For 12-lead data, select a specific lead (e.g., signal[:, 1]) first.")

    # Step 1: Clean ECG signal using NeuroKit2 filter
    try:
        cleaned_signal = nk.ecg_clean(signal, sampling_rate=fs, method='neurokit')
    except Exception as e:
        print(f"[{lead_name}] Error cleaning: {e}")
        return None

    # Step 2: Detect R-peaks in cleaned signal
    try:
        peaks_dict = nk.ecg_findpeaks(cleaned_signal, sampling_rate=fs, method='neurokit')
        r_peaks = peaks_dict['ECG_R_Peaks']
    except Exception as e:
        print(f"[{lead_name}] Error finding peaks: {e}")
        return None

    # Step 3: Calculate ECG-derived metrics from detected peaks
    metrics = {}
    if len(r_peaks) > 1:
        # Calculate RR intervals and heart rate from peak positions
        rr_sec = np.diff(r_peaks) / fs
        rr_ms = rr_sec * 1000
        hr_bpm = 60 / rr_sec
        
        # Compile heart rate and heart rate variability metrics
        metrics = {
            'hr_mean': float(np.mean(hr_bpm)),
            'hr_std': float(np.std(hr_bpm)),
            'sdnn_ms': float(np.std(rr_ms)),  # Standard deviation of NN intervals
            'rmssd_ms': float(np.sqrt(np.mean(np.diff(rr_ms)**2))),  # Root mean square of successive differences
            'n_peaks': len(r_peaks)
        }
    else:
        # Insufficient peaks for full metrics calculation
        metrics = {'hr_mean': np.nan, 'n_peaks': len(r_peaks)}

    return {
        'cleaned_signal': cleaned_signal,
        'r_peaks': r_peaks,
        'metrics': metrics,
        'lead_name': lead_name
    }

def process_pcg(signal, fs):
    """
    Processes Heart Sound (PCG) data from ANY source (iPhone or PIN sensor).
    Robust to different amplitude scales (e.g. float [-1,1] vs int raw values).
    
    Pipeline:
    1. Input Standardization (Z-score) to handle scale differences.
    2. Bandpass Filter (20-400Hz) to isolate heart sounds.
    3. Envelope Extraction (Hilbert) to capture energy profile.
    4. Peak Detection (S1/S2 candidates).
    5. Feature Extraction (MFCCs, RMS).
    
    Args:
        signal (np.array): 1D signal array.
        fs (int/float): Sampling rate.
        
    Returns:
        dict: {
            'cleaned_signal': np.array,
            'envelope': np.array,
            'peaks': np.array,
            'metrics': dict
        }
    """
    # 1. Input Standardization
    # Convert to float
    # Step 1: Input standardization and validation
    signal = np.array(signal).astype(np.float32)
    fs = float(fs)
    
    # Remove DC offset and normalize to unit variance for consistent processing
    # This standardization handles different input scales (e.g., iPhone float [-1,1] vs PIN integer)
    if np.std(signal) > 0:
        signal = (signal - np.mean(signal)) / np.std(signal)
    else:
        # Return early if signal is flat (no variance)
        return {
            'cleaned_signal': signal, 
            'envelope': signal, 
            'peaks': np.array([]), 
            'metrics': {}
        }

    # Step 2: Apply bandpass filter to isolate heart sound frequencies (20-400 Hz)
    # Uses SOS (Second-Order Sections) representation for numerical stability
    sos = scipy.signal.butter(4, [20, 400], btype='bandpass', fs=fs, output='sos')
    cleaned_signal = scipy.signal.sosfiltfilt(sos, signal)
    
    # Step 3: Extract envelope using Hilbert transform to capture energy profile
    analytic_signal = scipy.signal.hilbert(cleaned_signal)
    amplitude_envelope = np.abs(analytic_signal)
    
    # Smooth envelope with lowpass filter at 20 Hz to merge internal components
    # This combines S1/S2 split sounds into single identifiable peaks
    sos_env = scipy.signal.butter(2, 20, btype='low', fs=fs, output='sos')
    smoothed_envelope = scipy.signal.sosfiltfilt(sos_env, amplitude_envelope)
    
    # Normalize envelope to range [0, 1] for consistent peak detection thresholds
    if np.max(smoothed_envelope) > 0:
        normalized_envelope = smoothed_envelope / np.max(smoothed_envelope)
    else:
        normalized_envelope = smoothed_envelope
        
    # Step 4: Detect peaks in normalized envelope
    # Height threshold 0.1: peak must be at least 10% of maximum amplitude
    # Distance constraint: minimum 0.2s between peaks (enforces physiological limit: heart rate < 300 bpm)
    peaks, _ = scipy.signal.find_peaks(normalized_envelope, height=0.1, distance=int(0.2 * fs))
    
    # Step 5: Extract audio features using Mel-Frequency Cepstral Coefficients (MFCCs)
    try:
        # Calculate 13 MFCC coefficients from cleaned signal to capture spectral characteristics
        mfccs = librosa.feature.mfcc(y=cleaned_signal, sr=fs, n_mfcc=13)
        mfcc_means = np.mean(mfccs, axis=1)
        mfcc_dict = {f"mfcc_{i+1}": float(m) for i, m in enumerate(mfcc_means)}
    except Exception as e:
        # Skip MFCC extraction if signal is too short or processing fails
        mfcc_dict = {}

    # Step 6: Compile metrics dictionary
    metrics = {
        'duration_sec': len(signal) / fs,
        'rms_energy': float(np.sqrt(np.mean(cleaned_signal**2))),  # Root mean square energy of standardized signal
        'num_peaks_detected': len(peaks),
        **mfcc_dict
    }

    return {
        'cleaned_signal': cleaned_signal,
        'envelope': normalized_envelope, # Return the NORMALIZED envelope for easy plotting
        'peaks': peaks,
        'metrics': metrics
    }