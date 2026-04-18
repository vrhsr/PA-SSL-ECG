"""
PA-SSL: Chapman-Shaoxing ECG Data Processor
Downloads and processes the Chapman-Shaoxing 12-lead ECG dataset.

Chapman-Shaoxing contains ~10,000+ 12-lead ECG recordings at 500 Hz
with rhythm and morphology labels. We process Lead II segments following
the same pipeline as PTB-XL and MIT-BIH.
"""

import wfdb
import numpy as np
import pandas as pd
from scipy.signal import resample
from src.data.signal_processing import bandpass_filter, z_score_normalize, detect_r_peaks
from tqdm import tqdm
import os

# ─── CONFIGURATION ───────────────────────────────────────────────────────────
SAMPLE_RATE = 100
WINDOW_SEC = 2.5
WINDOW_SAMPLES = int(SAMPLE_RATE * WINDOW_SEC)  # 250
FILTER_LOW = 0.5
FILTER_HIGH = 40.0
SOURCE_RATE = 500       # Chapman-Shaoxing native rate
LEAD_IDX = 1            # Lead II
DATA_DIR = 'data/chapman/WFDB_ChapmanShaoxing'
OUTPUT_FILE = 'data/chapman_processed.csv'

# Chapman rhythm labels → binary mapping
# Normal sinus rhythm, sinus bradycardia, sinus tachycardia → Normal (0)
# All others → Abnormal (1)
NORMAL_RHYTHMS = {
    'SB',    # Sinus Bradycardia (still sinus, normal conduction)
    'SR',    # Sinus Rhythm
    'ST',    # Sinus Tachycardia
}

# ─── SIGNAL PROCESSING ───────────────────────────────────────────────────────

def bandpass_filter(data, lowcut, highcut, fs, order=3):
    nyquist = 0.5 * fs
    low = lowcut / nyquist
    high = highcut / nyquist
    b, a = butter(order, [low, high], btype='band')
    return filtfilt(b, a, data)


# ─── R-PEAK DETECTION (with fallback) ────────────────────────────────────────



# ─── MAIN PROCESSING ─────────────────────────────────────────────────────────

def process_chapman(output_file=None, data_dir=None):
    """
    Process Chapman-Shaoxing dataset.
    
    NOTE: Chapman-Shaoxing must be downloaded manually from PhysioNet:
    https://physionet.org/content/chapman-shaoxing/1.0.0/
    
    Place the extracted data in data/chapman-shaoxing/
    """
    global DATA_DIR
    if output_file is None:
        output_file = OUTPUT_FILE
    if data_dir is not None:
        DATA_DIR = data_dir
        print(f"Using data_dir: {DATA_DIR}")
        
    # Ensure base data directory exists
    if not os.path.exists(DATA_DIR):
        print(f"ERROR: Chapman-Shaoxing data not found at {DATA_DIR}")
        print("Please download from: https://physionet.org/content/chapman-shaoxing/1.0.0/")
        print("Extract to: data/chapman-shaoxing/")
        print("\nAlternatively, run:")
        print(f"  wfdb.dl_database('chapman-shaoxing', '{DATA_DIR}')")
        return None
    
    # Try to download via wfdb if directory is empty, otherwise search recursively for actual .mat files
    dat_files = []
    actual_data_dir = DATA_DIR
    if os.path.exists(DATA_DIR):
        for root, dirs, files in os.walk(DATA_DIR):
            mat_files = [f for f in files if f.endswith('.mat')]
            if len(mat_files) > 0:
                actual_data_dir = root
                dat_files = mat_files
                break
                
    if len(dat_files) == 0:
        print("Attempting to download Chapman-Shaoxing via wfdb...")
        try:
            wfdb.dl_database('chapman-shaoxing', DATA_DIR)
            actual_data_dir = DATA_DIR
        except Exception as e:
            print(f"Download failed: {e}")
            print("Please download manually from PhysioNet.")
            return None
            
    # Redefine dat_files from actual_data_dir
    dat_files = [f for f in os.listdir(actual_data_dir) if f.endswith('.mat')]
    dat_files.sort()
    
    print(f"Found {len(dat_files)} recording files in {actual_data_dir}")
    
    # Load reference/labels CSV if available - check recursively as well
    label_file = os.path.join(actual_data_dir, 'Diagnostics.xlsx')
    label_df = None
    
    # Let's search recursively for any Excel/CSV reference file just in case
    if not os.path.exists(label_file):
        for root, dirs, files in os.walk(DATA_DIR):
            for f in files:
                if 'Diagnostics' in f or 'REFERENCE' in f:
                    if f.endswith('.xlsx') or f.endswith('.csv'):
                        label_file = os.path.join(root, f)
                        break

    if os.path.exists(label_file):
        if label_file.endswith('.xlsx'):
            label_df = pd.read_excel(label_file)
        else:
            label_df = pd.read_csv(label_file)
        print(f"Loaded {len(label_df)} label entries from {os.path.basename(label_file)}")
    
    # Recursively find all .mat records
    records_with_paths = []
    for root, dirs, files in os.walk(DATA_DIR):
        for f in files:
            if f.endswith('.mat'):
                rec_base = f.replace('.mat', '')
                if os.path.exists(os.path.join(root, rec_base + '.hea')):
                    records_with_paths.append((root, rec_base))
    
    records_with_paths = sorted(list(set(records_with_paths)))
    print(f"Found {len(records_with_paths)} Chapman-Shaoxing records across all directories. Processing...")
    
    all_beats, all_labels, all_patient_ids, all_record_ids, all_beat_idxs, all_rpeak_positions = [], [], [], [], [], []
    
    for rec_dir, rec in tqdm(records_with_paths, desc="Processing Chapman", mininterval=15.0):
        try:
            record_path = os.path.join(rec_dir, rec)
            signals, fields = wfdb.rdsamp(record_path)
            
            # Determine label from header comments or REFERENCE.csv
            lbl = -1
            if label_df is not None and rec in label_df.iloc[:, 0].values:
                row_data = label_df[label_df.iloc[:, 0] == rec]
                if len(row_data) > 0:
                    rhythm = str(row_data.iloc[0, 1]).strip()
                    lbl = 0 if rhythm in NORMAL_RHYTHMS else 1
            elif 'comments' in fields and fields.get('comments'):
                # Try to extract from WFDB header comments (SNOMED CT codes)
                # Chapman-Shaoxing uses SNOMED CT in the PhysioNet 2020 format: "Dx: 426783006, ..."
                # Normal SNOMED CT Codes: 426783006 (SR), 426285000 (Normal SR), 426177001 (SB), 427084000 (ST)
                normal_snomed = {'426783006', '426285000', '426177001', '427084000'}
                
                for comment in fields['comments']:
                    if comment.startswith('Dx:'):
                        dx_codes = set(code.strip() for code in comment.replace('Dx:', '').split(','))
                        if any(code in normal_snomed for code in dx_codes):
                            lbl = 0
                            break
                        else:
                            lbl = 1
                            break
                        
                if lbl == -1:
                    lbl = 1  # Assume abnormal if not explicitly normal
            else:
                continue  # Skip if no label info
            
            if lbl == -1:
                continue
            
            # Extract Lead II
            if signals.shape[1] <= LEAD_IDX:
                continue
            signal = signals[:, LEAD_IDX]
            
            # Filter
            signal = bandpass_filter(signal, FILTER_LOW, FILTER_HIGH, SOURCE_RATE)
            
            # R-peak detection
            qrs_inds = detect_r_peaks(signal, fs=SOURCE_RATE)
            
            # Segment beats
            for beat_idx, r_peak in enumerate(qrs_inds):
                window_raw = int(WINDOW_SEC * SOURCE_RATE)
                half_window = window_raw // 2
                
                start = r_peak - half_window
                end = r_peak + half_window
                
                if start < 0 or end > len(signal):
                    continue
                
                beat_raw = signal[start:end]
                beat_resampled = resample(beat_raw, WINDOW_SAMPLES)
                r_peak_resampled = int(half_window * WINDOW_SAMPLES / window_raw)
                beat_norm = z_score_normalize(beat_resampled)
                
                all_beats.append(beat_norm)
                all_labels.append(lbl)
                all_patient_ids.append(rec)  # Use record as patient ID
                all_record_ids.append(rec)
                all_beat_idxs.append(beat_idx)
                all_rpeak_positions.append(r_peak_resampled)
                
        except Exception as e:
            continue
    
    if len(all_beats) == 0:
        print("WARNING: No beats extracted. Check dataset structure.")
        return None
    
    df = pd.DataFrame(np.array(all_beats))
    df['label'] = all_labels
    df['patient_id'] = all_patient_ids
    df['record_id'] = all_record_ids
    df['beat_idx'] = all_beat_idxs
    df['r_peak_pos'] = all_rpeak_positions
    
    os.makedirs(os.path.dirname(output_file), exist_ok=True)
    df.to_csv(output_file, index=False)
    
    print(f"\n{'='*60}")
    print(f"Chapman-Shaoxing Processing Complete")
    print(f"  Total beats: {len(df):,}")
    print(f"  Unique records: {df['patient_id'].nunique()}")
    print(f"  Label distribution:")
    for lbl, count in df['label'].value_counts().sort_index().items():
        print(f"    Class {lbl}: {count:,} ({100*count/len(df):.1f}%)")
    print(f"  Saved to: {output_file}")
    print(f"{'='*60}")
    
    return df


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="Process Chapman-Shaoxing dataset")
    parser.add_argument('--output', type=str, default=OUTPUT_FILE,
                        help='Output CSV path')
    parser.add_argument('--data_dir', type=str, default=None,
                        help='Path to Chapman data directory (containing .mat/.hea files)')
    args = parser.parse_args()
    process_chapman(args.output, args.data_dir)

