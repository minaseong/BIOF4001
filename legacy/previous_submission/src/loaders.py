import sqlite3
import pandas as pd
import json
import os
import numpy as np
import re
import pydicom  # For reading DICOM 12-lead ECG files
import scipy.io.wavfile as wav  # For reading audio files from iPhone auscultation


# Helper function to deserialize ECG values from JSON string format stored in the database
def _extract_ecg_values_from_json(json_string):
    return json.loads(json_string)

# Extract expected patient IDs from SQLite filename patterns (e.g., '4001-4006.sqlite3' or '4001.sqlite3')
def _get_patient_ids_from_filename(filepath):
    filename = os.path.basename(filepath)
    # Check for range pattern (e.g., 4001-4006)
    match_range = re.search(r'(\d+)-(\d+)\.sqlite3$', filename)
    if match_range:
        return list(range(int(match_range.group(1)), int(match_range.group(2)) + 1))
    # Check for single patient pattern (e.g., 4001)
    match_single = re.search(r'(\d+)\.sqlite3$', filename)
    if match_single:
        return [int(match_single.group(1))]
    return []

def load_2lead_ecg(sqlite_path, target_patient_id):
    target_patient_id = int(target_patient_id)
    expected_ids = _get_patient_ids_from_filename(sqlite_path)
    
    if target_patient_id not in expected_ids:
        print(f"Skipping: Patient {target_patient_id} not expected.")
        return None

    # Calculate which index this patient is in the file (0th, 1st, etc.)
    patient_idx = expected_ids.index(target_patient_id)

    try:
        conn = sqlite3.connect(sqlite_path)
        
        # Step 1: Query all unique parent record IDs from the database to establish the mapping
        unique_ids_query = "SELECT DISTINCT parentRecordID FROM ecgResults ORDER BY parentRecordID"
        unique_ids_df = pd.read_sql_query(unique_ids_query, conn)
        unique_ids = unique_ids_df['parentRecordID'].tolist()

        # Validate that database contains expected number of records
        if len(unique_ids) != len(expected_ids):
            print(f"⚠️ WARNING: Count Mismatch in {os.path.basename(sqlite_path)}!")
            print(f"   Expected {len(expected_ids)} patients ({expected_ids})")
            print(f"   Found {len(unique_ids)} distinct recordings in DB.")
            print(f"   Proceeding with index-based mapping assumption...")
        
        if patient_idx >= len(unique_ids):
            print(f"Error: Index {patient_idx} out of bounds for {len(unique_ids)} IDs in DB.")
            conn.close()
            return None
            
        target_parent_record_id = unique_ids[patient_idx]
        
        # Step 2: Fetch all ECG data packets for the target patient, ordered by receive time
        data_query = f"""
            SELECT arrECGdataJSON, receivedTime
            FROM ecgResults
            WHERE parentRecordID = {target_parent_record_id}
            ORDER BY receivedTime
        """
        df_raw = pd.read_sql_query(data_query, conn)
        conn.close()
        
        if df_raw.empty: return None

        # Step 3: Extract and process all ECG values from JSON-encoded packets
        SAMPLING_RATE = 1024
        TIME_PER_SAMPLE = 1.0 / SAMPLING_RATE
        
        # Deserialize ECG values from all JSON packets and flatten into single array
        all_ecg_values = []
        for json_str in df_raw['arrECGdataJSON']:
            all_ecg_values.extend(_extract_ecg_values_from_json(json_str))
            
        if not all_ecg_values: return None
        
        total_samples = len(all_ecg_values)
        
        # Step 4: Generate uniform timestamp grid
        # Set start time 1 second before the first recorded packet for alignment
        first_received_time = df_raw['receivedTime'].iloc[0]
        start_time = first_received_time - 1.0
        uniform_timestamps = start_time + np.arange(total_samples) * TIME_PER_SAMPLE
        
        # Step 5: Create DataFrame with timestamps and ECG values
        df = pd.DataFrame({
            'timestamp': pd.to_datetime(uniform_timestamps, unit='s'),
            'ecg_value': all_ecg_values
        })
        
        # Step 6: Attach metadata attributes
        df.attrs['sampling_rate'] = float(SAMPLING_RATE)
        df.attrs['patient_id'] = target_patient_id
        
        return df

    except Exception as e:
        print(f"Error loading SQLite {sqlite_path}: {e}")
        if 'conn' in locals(): conn.close()
        return None

# 12-lead ECG DICOM Loader
def load_dicom_12lead(filepath):
    """
    Loads raw 12-lead ECG waveform from a DICOM file.
    
    Args:
        filepath (str): Path to the .dcm file.
        
    Returns:
        dict: {
            'signals': pd.DataFrame,  # Columns = ['I', 'II', 'V1', etc.]
            'metadata': dict          # {fs, units, patient_id, etc.}
        }
        Returns None if loading fails.
    """
    try:
        ds = pydicom.dcmread(filepath)
        
        # Check if waveform data exists
        if not hasattr(ds, 'WaveformSequence'):
            print(f"Error: No WaveformSequence in {filepath}")
            return None

        # Extract waveform data from first sequence (12-lead ECG)
        waveform_seq = ds.WaveformSequence[0]
        
        # Step 1: Extract basic metadata from waveform
        fs = float(waveform_seq.SamplingFrequency)
        num_channels = waveform_seq.NumberOfWaveformChannels
        num_samples = waveform_seq.NumberOfWaveformSamples
        
        # Step 2: Extract channel definitions (names, sensitivity, units)
        channel_names = []
        sensitivities = []
        units = []
        
        for ch in waveform_seq.ChannelDefinitionSequence:
            # Extract channel label (e.g., "Lead I", "V1"); default to generic name if unavailable
            label = getattr(ch, 'ChannelLabel', f"Ch_{len(channel_names)}")
            channel_names.append(label)
            
            # Extract sensitivity value used to convert raw integer values to physical units
            # Default to 1.0 if not available
            sens = float(getattr(ch, 'ChannelSensitivity', 1.0))
            sensitivities.append(sens)
            
            # Extract measurement units (e.g., 'uV' or 'mV')
            unit_seq = getattr(ch, 'ChannelSensitivityUnitsSequence', None)
            if unit_seq:
                unit_str = unit_seq[0].CodeMeaning
            else:
                unit_str = "raw"
            units.append(unit_str)
            
        # Step 3: Extract raw signal data from DICOM waveform bytes
        # DICOM stores waveform data as multiplexed signed 16-bit integers: [ch1_t0, ch2_t0, ..., ch1_t1, ch2_t1, ...]
        raw_bytes = waveform_seq.WaveformData
        raw_ints = np.frombuffer(raw_bytes, dtype=np.int16)
        
        # Reshape raw data from flat array to (num_samples, num_channels) matrix
        raw_matrix = raw_ints.reshape(num_samples, num_channels)
        
        # Step 4: Apply sensitivity correction to convert raw integers to physical units
        # Multiply each channel by its sensitivity factor (typically converting to microvolts)
        physical_matrix = raw_matrix * np.array(sensitivities)
        
        # Step 5: Create DataFrame with processed signals and channel labels
        df_signals = pd.DataFrame(physical_matrix, columns=channel_names)
        
        # Step 6: Compile metadata dictionary with recording information
        metadata = {
            'sampling_rate': fs,
            'duration_sec': num_samples / fs,
            'units': units,  # List of measurement units per channel
            'patient_id': getattr(ds, 'PatientID', 'Unknown'),
            'study_date': getattr(ds, 'StudyDate', 'Unknown'),
            'manufacturer': getattr(ds, 'Manufacturer', 'Unknown')
        }
        
        return {
            'signals': df_signals,
            'metadata': metadata
        }

    except Exception as e:
        print(f"Error loading DICOM {filepath}: {e}")
        return None


# Auscultation (iPhone WAV) Loader
def load_iphone_wav(filepath, start_timestamp=None):
    """
    Loads .wav file and generates timestamps.
    
    Args:
        filepath (str): Path to .wav file.
        start_timestamp (pd.Timestamp or str): The absolute start time of the recording.
                                               Format: 'YYYY-MM-DD HH:MM:SS'
                                               Since the time is recorded as YYYY-MM-DD HH:MM during the data collection,
                                               input as SS=00.
        
    Returns:
        dict: {
            'signal': np.array,
            'timestamps': np.array (pd.Timestamp objects) or None,
            'fs': int,
            'metadata': dict
        }
    """
    try:
        fs, signal = wav.read(filepath)
        
        # Step 1: Convert to mono and normalize to float range [-1, 1]
        if len(signal.shape) > 1:
            signal = signal[:, 0]  # Select first channel if stereo
        
        # Normalize based on bit depth
        if signal.dtype == np.int16:
            signal = signal.astype(np.float32) / 32768.0
        elif signal.dtype == np.int32:
            signal = signal.astype(np.float32) / 2147483648.0
        elif signal.dtype == np.uint8:
            signal = (signal.astype(np.float32) - 128) / 128.0

        # Step 2: Generate timestamps if start time is provided
        timestamps = None
        if start_timestamp is not None:
            # Parse start time string to Timestamp object
            start_ts = pd.to_datetime(start_timestamp)
            
            # Create time offset array (in seconds) for each sample: [0, 1/fs, 2/fs, ...]
            n_samples = len(signal)
            time_offsets = np.arange(n_samples, dtype=np.float64) / fs
            
            # Convert offsets to timedelta and add to start time
            timestamps = start_ts + pd.to_timedelta(time_offsets, unit='s')

        # Step 3: Extract patient ID from filename and compile metadata
        filename = os.path.basename(filepath)
        pid_match = re.search(r'(\d+)', filename)
        patient_id = int(pid_match.group(1)) if pid_match else "Unknown"
        
        return {
            'signal': signal,
            'timestamps': timestamps,
            'fs': fs,
            'metadata': {
                'patient_id': patient_id,
                'duration_sec': len(signal) / fs,
                'total_samples': len(signal),
                'filename': filename,
                'start_time': start_timestamp
            }
        }

    except Exception as e:
        print(f"Error loading WAV {filepath}: {e}")
        return None

# Auscultation pin CSV Loader
def load_pin_csv(filepath, fs=4000):
    """
    Loads PIN sensor CSV.
    Extracts start time from filename (format: 'ID audio_data_YYYY-MM-DD_HH-MM-SS.csv')
    Constructs timestamps based on fs.
    """
    try:
        df = pd.read_csv(filepath)
        
        # Step 1: Extract signal from CSV (use standard column name or fallback to second column)
        if 'Amplitude_Raw' in df.columns:
            signal = df['Amplitude_Raw'].values.astype(np.float32)
        else:
            signal = df.iloc[:, 1].values.astype(np.float32)

        # Step 2: Extract timestamp from filename (format: YYYY-MM-DD_HH-MM-SS)
        filename = os.path.basename(filepath)
        match = re.search(r'(\d{4}-\d{2}-\d{2}_\d{2}-\d{2}-\d{2})', filename)
        
        start_ts = None
        timestamps = None
        
        if match:
            date_str = match.group(1)
            try:
                start_ts = pd.to_datetime(date_str, format='%Y-%m-%d_%H-%M-%S')
            except:
                print(f"Warning: Could not parse date from {date_str}")

        # Step 3: Generate uniform timestamps for all samples if start time was found
        if start_ts is not None:
            n_samples = len(signal)
            time_offsets = np.arange(n_samples, dtype=np.float64) / fs
            timestamps = start_ts + pd.to_timedelta(time_offsets, unit='s')
            
        return {
            'signal': signal,
            'timestamps': timestamps,
            'fs': fs,
            'metadata': {
                'start_time': start_ts,
                'filename': filename,
                'total_samples': len(signal)
            }
        }

    except Exception as e:
        print(f"Error loading PIN CSV {filepath}: {e}")
        return None