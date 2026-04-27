import os
import json
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

def save_processed_ecg(patient_id, raw_signal, timestamps, processed_result, output_dir):
    """
    Saves processed ECG data for a patient.
    Creates 'output_dir' if it does not exist.
    """
    # Create output directory structure if it does not already exist
    os.makedirs(output_dir, exist_ok=True)
    
    # Extract processing results from dictionary
    cleaned_signal = processed_result['cleaned_signal']
    r_peaks = processed_result['r_peaks']
    metrics = processed_result['metrics']
    
    # Step 1: Save signals and large arrays in compressed NumPy format
    npz_path = os.path.join(output_dir, f"{patient_id}_ecg.npz")
    np.savez_compressed(
        npz_path,
        raw_signal=raw_signal,
        cleaned_signal=cleaned_signal,
        timestamps=timestamps,
        r_peak_indices=r_peaks
    )
    
    # Step 2: Save metrics in JSON format for easy access and analysis
    json_path = os.path.join(output_dir, f"{patient_id}_metrics.json")
    
    output_meta = {
        'patient_id': int(patient_id),
        'signal_file': os.path.basename(npz_path),
        **metrics
    }
    
    with open(json_path, 'w') as f:
        json.dump(output_meta, f, indent=4)
        
    print(f"Saved Patient {patient_id}: Metrics -> JSON, Signals -> NPZ")


def plot_ecg_analysis(
    signal, 
    fs, 
    r_peaks=None, 
    cleaned_signal=None, 
    title="ECG Analysis", 
    save_path=None, 
    show=True,
    duration_sec=10
):
    """
    Visualizes ECG signal with R-peak annotations.
    Creates parent directory of 'save_path' if it does not exist.
    """
    
    # Step 1: Create time axis in seconds
    t = np.arange(len(signal)) / fs
    
    # Step 2: Optionally slice data to display only requested duration
    if duration_sec:
        limit_samples = int(duration_sec * fs)
        t = t[:limit_samples]
        signal = signal[:limit_samples]
        if cleaned_signal is not None:
            cleaned_signal = cleaned_signal[:limit_samples]
        if r_peaks is not None:
            r_peaks = r_peaks[r_peaks < limit_samples]

    # Step 3: Create figure with appropriate size
    plt.figure(figsize=(15, 5))
    
    # Plot raw signal (background reference)
    plt.plot(t, signal, color='gray', alpha=0.6, label='Raw Signal', linewidth=1)
    
    # Plot cleaned signal and R-peaks if available
    if cleaned_signal is not None:
        plt.plot(t, cleaned_signal, color='#2196F3', label='Cleaned Signal', linewidth=1.2)
        
        # Overlay detected R-peaks on cleaned signal
        if r_peaks is not None:
            peak_times = r_peaks / fs
            peak_vals = cleaned_signal[r_peaks]
            plt.scatter(peak_times, peak_vals, color='#FF5252', s=50, zorder=5, label='R-Peaks', marker='o')
    
    # Step 4: Configure plot appearance and labels
    plt.title(title, fontsize=14, fontweight='bold')
    plt.xlabel("Time (seconds)", fontsize=12)
    plt.ylabel("Amplitude", fontsize=12)
    plt.legend(loc='upper right')
    plt.grid(True, linestyle='--', alpha=0.4)
    plt.tight_layout()
    
    # Step 5: Save plot if output path is specified
    if save_path:
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        
        plt.savefig(save_path, dpi=150)
        print(f"✓ Plot saved to {save_path}")
        
    if show:
        plt.show()
    else:
        plt.close()


def plot_12lead_grid(signals_df, fs, title="12-Lead ECG", save_path=None, duration_sec=5):
    """
    Plots standard 12-lead layout (3 rows x 4 columns).
    Creates parent directory of 'save_path' if it does not exist.
    """
    # Define standard 12-lead ECG channel labels in display order
    leads = ['Lead I', 'Lead II', 'Lead III', 'Lead aVR', 'Lead aVL', 'Lead aVF', 
             'Lead V1', 'Lead V2', 'Lead V3', 'Lead V4', 'Lead V5', 'Lead V6']
    
    # Check which leads are available in the input data
    present_leads = [l for l in leads if l in signals_df.columns]
    if len(present_leads) < 12:
        print("Warning: Missing leads. Plotting available columns.")
    
    # Step 1: Create time axis and optionally slice to requested duration
    t = np.arange(len(signals_df)) / fs
    if duration_sec:
        limit = int(duration_sec * fs)
        t = t[:limit]
        plot_data = signals_df.iloc[:limit]
    else:
        plot_data = signals_df

    # Step 2: Create figure with 12-lead grid layout (3 rows x 4 columns)
    fig, axes = plt.subplots(3, 4, figsize=(20, 10), sharex=True, sharey=True)
    fig.suptitle(title, fontsize=16)

    # Step 3: Plot each lead in its corresponding subplot
    for i, lead in enumerate(leads):
        row = i // 4
        col = i % 4
        ax = axes[row, col]
        
        if lead in plot_data.columns:
            # Plot the ECG signal for this lead
            ax.plot(t, plot_data[lead], linewidth=0.8, color='black')
            ax.set_title(lead, fontsize=10, loc='left')
            ax.grid(True, alpha=0.3)
            
            # Remove redundant axis labels to reduce clutter
            if row < 2: ax.set_xticklabels([])
            if col > 0: ax.set_yticklabels([])
        else:
            # Indicate missing leads in plot
            ax.text(0.5, 0.5, "Missing", ha='center')

    plt.tight_layout(rect=[0, 0.03, 1, 0.95])
    
    # Step 4: Save plot if output path is specified
    if save_path:
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        
        plt.savefig(save_path, dpi=150)
        print(f"✓ Saved 12-lead plot to {save_path}")
    plt.show()

def plot_pcg_result(signal, fs, result, title="PCG Analysis", save_path=None, show=True, duration_sec=5, decimate=1):
    """
    Plots PCG processing results: cleaned signal + envelope + detected peaks.
    
    Args:
        signal (np.array): Raw input signal (not strictly needed for plot, but kept for consistency).
        fs (float): Sampling rate.
        result (dict): Output from processors.process_pcg().
        title (str): Plot title.
        save_path (str): Path to save figure.
        show (bool): Whether to display immediately.
        duration_sec (float): Duration to display (zoom window).
        decimate (int): Plot every Nth sample (for performance with high fs).
    """
    cleaned = result["cleaned_signal"]
    env = result["envelope"]
    peaks = result["peaks"]

    n = len(cleaned)
    limit = min(int(duration_sec * fs), n)

    # Decimation for plotting efficiency
    idx = np.arange(limit)
    if decimate > 1:
        idx = idx[::decimate]

    t = idx / fs

    # Filter peaks within window
    peaks_win = peaks[peaks < limit]

    plt.figure(figsize=(15, 5))
    
    # Cleaned Signal (Light Background)
    plt.plot(t, cleaned[idx], color="steelblue", alpha=0.4, linewidth=0.8, label="Cleaned Signal")
    
    # Envelope (Main Feature)
    plt.plot(t, env[idx], color="darkorange", linewidth=2.0, label="Envelope")

    # Detected Peaks
    if len(peaks_win) > 0:
        plt.scatter(peaks_win / fs, env[peaks_win], color="red", s=40, zorder=5, label="Detected Peaks", marker='o')

    plt.title(title, fontsize=14, fontweight='bold')
    plt.xlabel("Time (seconds)", fontsize=12)
    plt.ylabel("Amplitude (Normalized)", fontsize=12)
    plt.grid(True, alpha=0.3)
    plt.legend(loc="upper right")
    plt.xlim(0, t[-1] if len(t) else duration_sec)
    plt.tight_layout()

    if save_path:
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        plt.savefig(save_path, dpi=150)
        print(f"✓ PCG plot saved to {save_path}")

    if show:
        plt.show()
    else:
        plt.close()