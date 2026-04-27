"""
main.py
Batch processing pipeline for multi-modal cardiac data.
Processes 2-Lead ECG, 12-Lead ECG, iPhone PCG, and PIN PCG data.
Saves outputs per participant to data/processed/{patient_id}/
"""

import os
import glob
import re
import numpy as np
from tqdm import tqdm

# Import our custom modules
from src import (
    load_2lead_ecg,
    load_dicom_12lead,
    load_iphone_wav,
    load_pin_csv,
    process_ecg,
    process_pcg,
    save_processed_ecg,
    plot_ecg_analysis,
    plot_pcg_result,
    plot_12lead_grid
)

# --- CONFIGURATION ---
DATA_ROOT = "data/raw"
OUTPUT_ROOT = "data/processed"

# Define paths relative to data root
DIRS = {
    "ecg_2": os.path.join(DATA_ROOT, "2_lead_ecg"),
    "ecg_12": os.path.join(DATA_ROOT, "12_lead_ecg"),
    "pcg_iphone": os.path.join(DATA_ROOT, "ausc_iphone"),
    "pcg_pin": os.path.join(DATA_ROOT, "ausc_pin"),
}

# Participants to exclude
EXCLUDED_PIDS = [4002]  # Data lost

# --- HELPER: Manifest Builder ---
def build_patient_manifest():
    """
    Scans all raw directories and links files to Patient IDs (e.g., 4001).
    Returns a dict: { 4001: {'ecg_2': path, 'ecg_12': path, ...}, ... }
    """
    manifest = {}

    def add_to_manifest(pid, key, path):
        pid = int(pid)
        if pid not in manifest:
            manifest[pid] = {}
        manifest[pid][key] = path

    # 1. Scan 12-Lead DICOM (e.g., 4001.dcm)
    for f in glob.glob(os.path.join(DIRS["ecg_12"], "*.dcm")):
        match = re.search(r'(\d{4})', os.path.basename(f))
        if match:
            add_to_manifest(match.group(1), 'ecg_12', f)

    # 2. Scan iPhone WAV (e.g., iData4001M.wav)
    for f in glob.glob(os.path.join(DIRS["pcg_iphone"], "*.wav")):
        match = re.search(r'(\d{4})', os.path.basename(f))
        if match:
            add_to_manifest(match.group(1), 'pcg_iphone', f)

    # 3. Scan PIN CSV (e.g., 4001 audio_data_....csv)
    for f in glob.glob(os.path.join(DIRS["pcg_pin"], "*.csv")):
        match = re.search(r'^(\d{4})', os.path.basename(f))
        if match:
            add_to_manifest(match.group(1), 'pcg_pin', f)

    # 4. Scan 2-Lead SQLite (Grouped files, e.g., 4001-4006.sqlite3)
    for f in glob.glob(os.path.join(DIRS["ecg_2"], "*.sqlite3")):
        basename = os.path.basename(f)
        # Check range format "4001-4006"
        match_range = re.search(r'(\d+)-(\d+)', basename)
        if match_range:
            start, end = int(match_range.group(1)), int(match_range.group(2))
            for pid in range(start, end + 1):
                add_to_manifest(pid, 'ecg_2_db', f)
        else:
            # Single file format "4001.sqlite3"
            match_single = re.search(r'(\d+)', basename)
            if match_single:
                add_to_manifest(match_single.group(1), 'ecg_2_db', f)

    return manifest


# --- MAIN PIPELINE ---
def main():
    print("="*70)
    print("CARDIO MULTIMODAL PROCESSING PIPELINE")
    print("="*70)
    
    print("\n--- 1. Building Manifest ---")
    manifest = build_patient_manifest()
    sorted_pids = sorted(manifest.keys())
    
    # Filter excluded participants
    sorted_pids = [pid for pid in sorted_pids if pid not in EXCLUDED_PIDS]
    
    print(f"Found {len(sorted_pids)} patients: {sorted_pids}")
    if EXCLUDED_PIDS:
        print(f"Excluded: {EXCLUDED_PIDS} (Missing Data)")
    
    # Process each patient
    for pid in tqdm(sorted_pids, desc="Processing Patients"):
        files = manifest[pid]
        
        # Create Patient Folder: data/processed/4001/
        patient_out_dir = os.path.join(OUTPUT_ROOT, str(pid))
        os.makedirs(patient_out_dir, exist_ok=True)
        
        # Create subfolders for plots
        plots_dir = os.path.join(patient_out_dir, "plots")
        os.makedirs(plots_dir, exist_ok=True)

        print(f"\n{'='*70}")
        print(f"Processing Patient {pid}")
        print(f"{'='*70}")
        
        # --- A. 2-LEAD ECG ---
        if 'ecg_2_db' in files:
            print(f"  [1/4] 2-Lead ECG from {os.path.basename(files['ecg_2_db'])}")
            try:
                df_2lead = load_2lead_ecg(files['ecg_2_db'], pid)
                
                if df_2lead is not None and len(df_2lead) > 0:
                    fs = df_2lead.attrs['sampling_rate']
                    # Process
                    res_2lead = process_ecg(df_2lead['ecg_value'].values, fs=fs)
                    
                    # Save Data
                    save_processed_ecg(
                        patient_id=f"{pid}_2lead",
                        raw_signal=df_2lead['ecg_value'].values,
                        timestamps=df_2lead['timestamp'].values,
                        processed_result=res_2lead,
                        output_dir=patient_out_dir
                    )
                    
                    # Plot
                    plot_ecg_analysis(
                        signal=df_2lead['ecg_value'].values,
                        fs=fs,
                        cleaned_signal=res_2lead['cleaned_signal'],
                        r_peaks=res_2lead['r_peaks'],
                        title=f"Patient {pid} - 2-Lead ECG",
                        save_path=os.path.join(plots_dir, "2lead_rhythm.png"),
                        show=False
                    )
                    print(f"      ✓ Saved 2-Lead data and plot")
                else:
                    print(f"      ⊗ No data for Patient {pid} in 2-Lead DB")
            except Exception as e:
                print(f"      ✗ Error: {e}")

        # --- B. 12-LEAD ECG ---
        if 'ecg_12' in files:
            print(f"  [2/4] 12-Lead ECG from {os.path.basename(files['ecg_12'])}")
            try:
                res_dict = load_dicom_12lead(files['ecg_12'])
                
                if res_dict:
                    signals_df = res_dict['signals']
                    meta = res_dict['metadata']
                    fs = meta['sampling_rate']
                    
                    # Plot Grid Overview
                    plot_12lead_grid(
                        signals_df, fs, 
                        title=f"Patient {pid} - 12-Lead Overview",
                        save_path=os.path.join(plots_dir, "12lead_grid.png")
                    )
                    
                    # Process Lead II (Rhythm)
                    target_lead = 'Lead II'
                    if target_lead in signals_df.columns:
                        raw_ii = signals_df[target_lead].values
                        res_ii = process_ecg(raw_ii, fs=fs)
                        
                        # Save Data (Lead II rhythm strip)
                        t_vector = np.arange(len(raw_ii)) / fs
                        save_processed_ecg(
                            patient_id=f"{pid}_12lead_II",
                            raw_signal=raw_ii,
                            timestamps=t_vector, 
                            processed_result=res_ii,
                            output_dir=patient_out_dir
                        )
                        print(f"      ✓ Saved 12-Lead data and plots")
                    else:
                        print(f"      ⊗ Lead II not found in DICOM")
            except Exception as e:
                print(f"      ✗ Error: {e}")

        # --- C. iPhone PCG ---
        if 'pcg_iphone' in files:
            print(f"  [3/4] iPhone PCG from {os.path.basename(files['pcg_iphone'])}")
            try:
                iphone_data = load_iphone_wav(files['pcg_iphone'])
                
                if iphone_data:
                    fs = iphone_data['fs']
                    # Process
                    res_pcg = process_pcg(iphone_data['signal'], fs=fs)
                    
                    # Plot
                    plot_pcg_result(
                        signal=iphone_data['signal'],
                        fs=fs,
                        result=res_pcg,
                        title=f"Patient {pid} - iPhone PCG",
                        save_path=os.path.join(plots_dir, "pcg_iphone.png"),
                        show=False,
                        duration_sec=5,
                        decimate=max(1, int(fs // 2000))
                    )
                    
                    # Save Data
                    ts = iphone_data['timestamps']
                    if ts is None: 
                        ts = np.arange(len(iphone_data['signal'])) / fs
                    
                    save_processed_ecg(
                        patient_id=f"{pid}_pcg_iphone",
                        raw_signal=iphone_data['signal'],
                        timestamps=ts,
                        processed_result=res_pcg,
                        output_dir=patient_out_dir
                    )
                    print(f"      ✓ Saved iPhone PCG data and plot")
            except Exception as e:
                print(f"      ✗ Error: {e}")

        # --- D. PIN PCG ---
        if 'pcg_pin' in files:
            print(f"  [4/4] PIN PCG from {os.path.basename(files['pcg_pin'])}")
            try:
                pin_data = load_pin_csv(files['pcg_pin'], fs=4000)
                
                if pin_data:
                    fs = pin_data['fs']
                    # Process
                    res_pin = process_pcg(pin_data['signal'], fs=fs)
                    
                    # Plot
                    plot_pcg_result(
                        signal=pin_data['signal'],
                        fs=fs,
                        result=res_pin,
                        title=f"Patient {pid} - PIN Sensor",
                        save_path=os.path.join(plots_dir, "pcg_pin.png"),
                        show=False,
                        duration_sec=5
                    )
                    
                    # Save Data
                    ts = pin_data['timestamps']
                    if ts is None: 
                        ts = np.arange(len(pin_data['signal'])) / fs
                    
                    save_processed_ecg(
                        patient_id=f"{pid}_pcg_pin",
                        raw_signal=pin_data['signal'],
                        timestamps=ts,
                        processed_result=res_pin,
                        output_dir=patient_out_dir
                    )
                    print(f"      ✓ Saved PIN PCG data and plot")
            except Exception as e:
                print(f"      ✗ Error: {e}")

    print("\n" + "="*70)
    print("✅ PROCESSING COMPLETE")
    print("="*70)
    print(f"Output saved to: {OUTPUT_ROOT}/")


if __name__ == "__main__":
    main()