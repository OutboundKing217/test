#!/usr/bin/env python3
"""
Python adaptation of UpperLimbAccelerometry.R

Processes bilateral wrist-worn accelerometry data (both 1Hz and 30Hz) 
to compute 26 upper limb sensor variables reflecting movement in daily life.
Outputs results for each subject+timepoint into a single CSV.
"""

import os
import glob
import numpy as np
import pandas as pd
from scipy.signal import butter, filtfilt, periodogram
from scipy.spatial.distance import pdist
import argparse

# Load functionality from data_io.py
from data_io import load_data, parse_filename

def get_affected_side(demo_df, subject_id, timepoint):
    """
    Flexibly searches the demographic dataframe to find the affected side for a subject.
    Defaults to 'Right' if not found.
    """
    if demo_df is None:
        return 'Right'
        
    # Locate subject column (fallback to first column if standard names aren't found)
    subj_col = next((c for c in ['subidname', 'subject_id', 'subjectid', 'id', 'subject', 'studyid'] if c in demo_df.columns), demo_df.columns[0])
    
    # Locate affected side column
    side_col = next((c for c in ['affectedside', 'affside', 'sideaffected', 'side', 'affected_side'] if c in demo_df.columns), None)
            
    if not side_col:
        print("Side col not found, defaulting to right")
        return 'Right'
        
    # Attempt to match subject_id (e.g. PMC12345_001)
    mask = demo_df[subj_col].astype(str).str.contains(str(subject_id), case=False, na=False)
    subset = demo_df[mask]
    
    # Fallback to match just the numeric part if the prefix is missing in demographic file
    if subset.empty and '_' in str(subject_id):
        clean_subj = str(subject_id).split('_')[-1]
        clean_subj_int = str(int(clean_subj)) if clean_subj.isdigit() else clean_subj
        
        demo_subj_str = demo_df[subj_col].astype(str).str.strip()
        
        # Attempt exact match on padded ('004') or unpadded ('4')
        mask = (demo_subj_str == clean_subj) | (demo_subj_str == clean_subj_int)
        
        # If still no match, fallback to word boundary regex to prevent '4' matching '14'
        if not mask.any():
            mask = demo_subj_str.str.contains(rf'(?<!\d){clean_subj_int}(?!\d)', regex=True, case=False, na=False)
            
        subset = demo_df[mask]
        
    if subset.empty:
        print(f"  -> Could not find subject {subject_id} in demographic file.")
        return 'Right'
        
    # Try to match timepoint if column exists
    tp_col = next((c for c in ['timepoint', 'time_point'] if c in demo_df.columns), None)
    if tp_col:
        tp_str = str(timepoint)
        tp_int = str(int(tp_str)) if tp_str.isdigit() else tp_str
        
        # Robust timepoint matching using lookarounds instead of word boundaries to handle "TimePoint0"
        tp_mask = subset[tp_col].astype(str).str.contains(rf'(?<!\d){tp_int}(?!\d)', regex=True, case=False, na=False)
        if tp_mask.any():
            subset = subset[tp_mask]
            
    # Parse actual value ('Left'/'L' vs everything else)
    side_val = str(subset[side_col].iloc[0]).strip().lower()
    
    if side_val.endswith('.0'):
        side_val = side_val[:-2]
    
    is_left = 'left' in side_val or side_val in ['l', 'lue', 'lle', '1']
    
    if side_val not in ['left', 'l', 'right', 'r', 'lue', 'rue', 'lle', 'rle', '0', '1', '2', 'nan', 'na']:
        print(f"  -> Warning: Unrecognized affected side value '{side_val}' for {subject_id}.")
        
    return 'Left' if is_left else 'Right'

def sample_entropy(U, m=2, r=0.2):
    """
    Computes Sample Entropy to match pracma::sample_entropy in R.
    """
    U = np.asarray(U)
    if len(U) <= m:
        return np.nan
        
    # R's sd() uses ddof=1
    r = r * np.std(U, ddof=1)
    
    def _phi(m_len):
        # Create overlapping windows of length m_len
        x = np.array([U[i:i+m_len] for i in range(len(U)-m_len+1)])
        if len(x) == 0:
            return 0
        # pdist calculates the Chebyshev distance between all distinct pairs.
        # Number of matching pairs is where distance <= r.
        d = pdist(x, 'chebyshev')
        return np.sum(d <= r)
        
    phi_m = _phi(m)
    phi_m1 = _phi(m + 1)
    
    if phi_m == 0 or phi_m1 == 0:
        return np.nan
    
    return -np.log(phi_m1 / phi_m)

def weighted_var(x, w):
    """
    Computes weighted variance matching R's unbiased weighted.var function.
    """
    sum_w = np.sum(w)
    sum_w2 = np.sum(w**2)
    if sum_w**2 == sum_w2:
        return 0
    mean_w = np.sum(x * w) / sum_w
    return (sum_w / (sum_w**2 - sum_w2)) * np.sum(w * (x - mean_w)**2)

def calculate_spectral_features(vm_signal, fs=30):
    """
    Calculates weighted mean frequency and standard deviation of frequency 
    matching R's periodogram spectrum(span=5) methodology.
    """
    freqs, Pxx = periodogram(vm_signal, fs=fs)
    # Emulate R's spectrum span=5 (modified Daniell smoother)
    Pxx_smooth = np.convolve(Pxx, np.ones(5)/5, mode='same')
    
    if np.sum(Pxx_smooth) == 0:
        return np.nan, np.nan
        
    density = 2 * Pxx_smooth
    mean_freq = np.average(freqs, weights=density)
    var_freq = weighted_var(freqs, density)
    
    return round(mean_freq, 3), round(np.sqrt(var_freq), 3)

def process_accelerometry(data_dir, output_file, demo_file=None):
    print(f"Scanning directory {data_dir} for 1Hz and 30Hz accelerometry files...")
    
    hz1_files = glob.glob(os.path.join(data_dir, "*1Hz.csv"))
    if not hz1_files:
        print("No *1Hz.csv files found. Please check your data directory.")
        return
        
    demo_df = None
    if demo_file and os.path.exists(demo_file):
        print(f"Loading demographic file from: {demo_file}")
        demo_df = pd.read_csv(demo_file)
        demo_df.columns = [str(c).strip().lower() for c in demo_df.columns]
        
    sessions = {}
    for f in hz1_files:
        meta = parse_filename(f)
        subj = meta.get('subject_id', 'Unknown')
        tp = meta.get('timepoint', -1)
        arm = meta.get('arm', 'Unknown')
        
        if subj == 'Unknown' or arm not in ['LUE', 'RUE']:
            continue
            
        key = (subj, tp)
        if key not in sessions:
            sessions[key] = {}
        sessions[key][arm] = f
    
    all_results = []
    
    for (sub_id, time_point), arms_dict in sessions.items():
        aff_side = get_affected_side(demo_df, sub_id, time_point)
        print(f"Processing Subject: {sub_id}, TimePoint: {time_point}")
        print(f"  -> Affected Side mapped to: {aff_side}")
        
        if 'LUE' not in arms_dict or 'RUE' not in arms_dict:
            print(f"  -> Missing LUE or RUE 1Hz files.")
            continue
            
        l_1hz_file = arms_dict['LUE']
        r_1hz_file = arms_dict['RUE']
        
        l_30hz_file = l_1hz_file.replace('1Hz.csv', '30Hz.csv')
        r_30hz_file = r_1hz_file.replace('1Hz.csv', '30Hz.csv')
        
        if not (os.path.exists(l_30hz_file) and os.path.exists(r_30hz_file)):
            print(f"  -> Missing corresponding 30Hz files.")
            continue
            
        # ==============================
        # 1 Hz Variables Processing
        # ==============================
        df_l_1, _ = load_data(l_1hz_file)
        df_r_1, _ = load_data(r_1hz_file)
        
        # Match lengths to shortest
        min_len_1 = min(len(df_l_1), len(df_r_1))
        x_l, y_l, z_l = df_l_1.iloc[:min_len_1, 0].values, df_l_1.iloc[:min_len_1, 1].values, df_l_1.iloc[:min_len_1, 2].values
        x_r, y_r, z_r = df_r_1.iloc[:min_len_1, 0].values, df_r_1.iloc[:min_len_1, 1].values, df_r_1.iloc[:min_len_1, 2].values
        
        lvm = np.sqrt(x_l**2 + y_l**2 + z_l**2)
        rvm = np.sqrt(x_r**2 + y_r**2 + z_r**2)
        
        threshold = 2
        l_count = (lvm >= threshold).astype(int)
        r_count = (rvm >= threshold).astype(int)
        
        recording_time = round(min_len_1 / 60.0, 0)
        
        no_mvt_min = np.sum((lvm == 0) & (rvm == 0)) / 60.0
        total_no_mvt_time = round(no_mvt_min / recording_time, 2) if recording_time else np.nan
        total_mvt_time = round(1 - total_no_mvt_time, 2) if recording_time else np.nan
        
        l_time_min = np.sum(l_count) / 60.0
        l_time = round(l_time_min / recording_time, 2) if recording_time else np.nan
        
        r_time_min = np.sum(r_count) / 60.0
        r_time = round(r_time_min / recording_time, 2) if recording_time else np.nan
        
        sim_time_sec = np.sum((lvm > threshold) & (rvm > threshold))
        simultaneous_time = round((sim_time_sec / 60.0) / recording_time, 2) if recording_time else np.nan
        
        l_only_sec = np.sum((lvm > threshold) & (rvm < threshold))
        l_only_time = round((l_only_sec / 60.0) / recording_time, 2) if recording_time else np.nan
        
        r_only_sec = np.sum((lvm < threshold) & (rvm > threshold))
        r_only_time = round((r_only_sec / 60.0) / recording_time, 2) if recording_time else np.nan
        
        l_active, r_active = lvm[l_count == 1], rvm[r_count == 1]
        
        l_mag = round(np.median(l_active), 3) if len(l_active) > 0 else np.nan
        r_mag = round(np.median(r_active), 3) if len(r_active) > 0 else np.nan
        l_mag_sd = round(np.std(l_active, ddof=1), 3) if len(l_active) > 1 else np.nan
        r_mag_sd = round(np.std(r_active, ddof=1), 3) if len(r_active) > 1 else np.nan
        
        bilateral_magnitude = (l_mag if not np.isnan(l_mag) else 0) + (r_mag if not np.isnan(r_mag) else 0)
        l_peak = round(np.max(l_active), 3) if len(l_active) > 0 else np.nan
        r_peak = round(np.max(r_active), 3) if len(r_active) > 0 else np.nan
        
        # Ratios calculations depending on affected side
        if aff_side.lower() == 'left':
            use_ratio = round(l_time / r_time, 3) if (r_time and not np.isnan(r_time) and r_time != 0) else np.nan
            var_ratio = round(l_mag_sd / r_mag_sd, 3) if (r_mag_sd and not np.isnan(r_mag_sd) and r_mag_sd != 0) else np.nan
            mag_ratio = round(l_mag / r_mag, 3) if (r_mag and not np.isnan(r_mag) and r_mag != 0) else np.nan
        else:
            use_ratio = round(r_time / l_time, 3) if (l_time and not np.isnan(l_time) and l_time != 0) else np.nan
            var_ratio = round(r_mag_sd / l_mag_sd, 3) if (l_mag_sd and not np.isnan(l_mag_sd) and l_mag_sd != 0) else np.nan
            mag_ratio = round(r_mag / l_mag, 3) if (l_mag and not np.isnan(l_mag) and l_mag != 0) else np.nan
            
        # Entropy calculation on the hour of max activity 
        entropy_target = lvm if aff_side.lower() == 'right' else rvm
        hour_frames = 60 * 60
        
        if len(entropy_target) > hour_frames:
            rm = pd.Series(entropy_target).rolling(window=hour_frames).mean().values
            end_idx = np.nanargmax(rm)
            start_idx = end_idx - hour_frames + 1
            
            l_entropy = round(sample_entropy(lvm[start_idx:end_idx+1], m=2, r=0.2), 3)
            r_entropy = round(sample_entropy(rvm[start_idx:end_idx+1], m=2, r=0.2), 3)
        else:
            l_entropy, r_entropy = np.nan, np.nan
            
        # ==============================
        # 30 Hz Variables Processing
        # ==============================
        df_l_30, _ = load_data(l_30hz_file)
        df_r_30, _ = load_data(r_30hz_file)
        
        min_len_30 = min(len(df_l_30), len(df_r_30))
        x_l_30, y_l_30, z_l_30 = df_l_30.iloc[:min_len_30, 0].values, df_l_30.iloc[:min_len_30, 1].values, df_l_30.iloc[:min_len_30, 2].values
        x_r_30, y_r_30, z_r_30 = df_r_30.iloc[:min_len_30, 0].values, df_r_30.iloc[:min_len_30, 1].values, df_r_30.iloc[:min_len_30, 2].values
        
        lvm_30 = np.sqrt(x_l_30**2 + y_l_30**2 + z_l_30**2)
        rvm_30 = np.sqrt(x_r_30**2 + y_r_30**2 + z_r_30**2)
        
        # Butterworth filter (0.2 Hz to 12 Hz, order 2)
        b, a = butter(2, [0.2, 12], btype='bandpass', fs=30)
        lvm_30_filt = filtfilt(b, a, lvm_30)
        rvm_30_filt = filtfilt(b, a, rvm_30)
        
        # Jerk
        tp = 1.0 / 30.0
        jerk_l = np.append(np.diff(lvm_30_filt) / tp, 0)
        jerk_r = np.append(np.diff(rvm_30_filt) / tp, 0)
        
        l_jerk_ave = round(np.mean(np.abs(jerk_l[jerk_l != 0])), 3) if np.any(jerk_l != 0) else np.nan
        r_jerk_ave = round(np.mean(np.abs(jerk_r[jerk_r != 0])), 3) if np.any(jerk_r != 0) else np.nan
        
        if pd.notna(l_jerk_ave) and pd.notna(r_jerk_ave) and (l_jerk_ave + r_jerk_ave) != 0:
            if aff_side.lower() == 'left':
                jerk_aym = round((l_jerk_ave - r_jerk_ave) / (l_jerk_ave + r_jerk_ave), 3)
            else:
                jerk_aym = round((r_jerk_ave - l_jerk_ave) / (l_jerk_ave + r_jerk_ave), 3)
        else:
            jerk_aym = np.nan
            
        # Frequency
        l_mean_freq, l_sd_freq = calculate_spectral_features(lvm_30_filt, fs=30)
        r_mean_freq, r_sd_freq = calculate_spectral_features(rvm_30_filt, fs=30)
        
        # Collate outputs (keys perfectly match variables in R output columns format)
        all_results.append({
            'StudyID': sub_id,
            'TimePoint': f"TimePoint{time_point}",
            'recording_time': recording_time,
            'NoMvt': round(no_mvt_min, 3),
            'total_no_mvt_time': total_no_mvt_time,
            'total_mvt_time': total_mvt_time,
            'LCount': round(l_time_min, 3),
            'l_time': l_time,
            'RCount': round(r_time_min, 3),
            'r_time': r_time,
            'SimTime': sim_time_sec,
            'simultaneous_time': simultaneous_time,
            'LVMAboveThreshold': l_only_sec,
            'l_only_time': l_only_time,
            'RVMAboveThreshold': r_only_sec,
            'r_only_time': r_only_time,
            'l_magnitude': l_mag,
            'r_magnitude': r_mag,
            'bilateral_magnitude': bilateral_magnitude,
            'l_magnitude_sd': l_mag_sd,
            'r_magnitude_sd': r_mag_sd,
            'l_peak_magnitude': l_peak,
            'r_peak_magnitude': r_peak,
            'use_ratio': use_ratio,
            'variation_ratio': var_ratio,
            'simple_magnitude_ratio': mag_ratio,
            'r_entropy': r_entropy,
            'l_entropy': l_entropy,
            'r_jerk_ave': r_jerk_ave,
            'l_jerk_ave': l_jerk_ave,
            'jerk_aym': jerk_aym,
            'l_mean_freq': l_mean_freq,
            'r_mean_freq': r_mean_freq,
            'l_sd_freq': l_sd_freq,
            'r_sd_freq': r_sd_freq
        })
        
    # Export to CSV
    if all_results:
        os.makedirs(os.path.dirname(output_file) or '.', exist_ok=True)
        df_out = pd.DataFrame(all_results)
        df_out.to_csv(output_file, index=False)
        print(f"\nSuccessfully processed {len(all_results)} files. Saved to {output_file}.")
    else:
        print("\nNo data processed successfully.")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Process bilateral upper limb accelerometry (1Hz and 30Hz)")
    parser.add_argument("--data_dir", type=str, required=True, help="Directory containing flattened 1Hz and 30Hz CSVs")
    parser.add_argument("--output_dir", type=str, default="../output/upper_limb_metrics", help="Directory to save output CSV")
    parser.add_argument("--output_filename", type=str, default="UpperLimbAccelerometry_Metrics.csv", help="Name of output CSV")
    parser.add_argument("--demo_file", type=str, default=None, help="Optional demographic CSV to determine affected side")
    
    args = parser.parse_args()
    
    os.makedirs(args.output_dir, exist_ok=True)
    full_output_path = os.path.join(args.output_dir, args.output_filename)
    
    process_accelerometry(args.data_dir, full_output_path, args.demo_file)
