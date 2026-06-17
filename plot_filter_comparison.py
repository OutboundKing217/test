"""
Plot Filter Comparisons

Generates 10-second visualization segments comparing raw dynamic magnitude 
to Butterworth, Chebyshev Type I, and Bessel-Thomson gravity-removal filters.
No smoothing or moving averages are applied (remains strict 30Hz).
"""

import os
import sys
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from scipy.signal import butter, cheby1, bessel, filtfilt

#matplotlib.rcParams["font.family"] = "Arial"
matplotlib.rcParams["pdf.fonttype"] = 42      # Ensures text is saved as real characters
matplotlib.rcParams["ps.fonttype"] = 42
matplotlib.rcParams["svg.fonttype"] = "none"  # Keep SVG text selectable

# Ensure we can import local modules
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from data_io import load_data

# =============================================================================
# CONFIGURATION
# =============================================================================

DATA_DIR = "../../data/fewStroke"  # Directory containing the 30Hz CSVs
OUTPUT_DIR = "../../output/filter_comparisons"

# Filter parameters
CUTOFF_HZ = 0.3
FILTER_ORDER = 4
CHEBY_RIPPLE_DB = 0.05

# Define the 10-second segments to plot. 
# You can add as many segments as you want here.
SEGMENTS_TO_PLOT = [
    {
        "filename": "PMC3905245_001_TimePoint0_LUE_30Hz.csv", 
        "start_time": 18000.0  # seconds 
    },
    {
        "filename": "PMC3905245_001_TimePoint0_LUE_30Hz.csv", 
        "start_time": 13500.0  # seconds 
    },
    {
        "filename": "PMC3905245_001_TimePoint0_LUE_30Hz.csv", 
        "start_time": 86400.0 # seconds
    },
    {
        "filename": "PMC8442937_001_TimePoint2_LUE_30Hz.csv", 
        "start_time": 7200.0 # seconds
    },
    {
        "filename": "PMC8442937_001_TimePoint2_LUE_30Hz.csv", 
        "start_time": 9000.0 # seconds
    },
    {
        "filename": "PMC8442937_001_TimePoint2_LUE_30Hz.csv", 
        "start_time": 10800.0 # seconds
    }
]

# =============================================================================

def get_dynamic_magnitude(df, b, a):
    """
    Isolates dynamic acceleration by removing gravity via a low-pass filter.
    Applies zero-phase forward and reverse filtering using filtfilt.
    """
    # Calculate the gravity vector
    grav_x = filtfilt(b, a, df[0].values)
    grav_y = filtfilt(b, a, df[1].values)
    grav_z = filtfilt(b, a, df[2].values)
    
    # Subtract gravity to isolate dynamic acceleration
    dyn_x = df[0].values - grav_x
    dyn_y = df[1].values - grav_y
    dyn_z = df[2].values - grav_z
    
    # Return the magnitude of the dynamic vector
    return np.sqrt(dyn_x**2 + dyn_y**2 + dyn_z**2)

def main():
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    
    # Group segments by file to avoid redundant loading and filtering
    files_to_process = {}
    for seg in SEGMENTS_TO_PLOT:
        fname = seg['filename']
        if fname not in files_to_process:
            files_to_process[fname] = []
        files_to_process[fname].append(seg['start_time'])
        
    for filename, start_times in files_to_process.items():
        filepath = os.path.join(DATA_DIR, filename)
        if not os.path.exists(filepath):
            print(f"File not found: {filepath}. Skipping.")
            continue
            
        print(f"Loading {filename}...")
        df, fs = load_data(filepath)
        
        # Calculate filter coefficients
        nyq = 0.5 * fs
        normal_cutoff = CUTOFF_HZ / nyq
        
        b_butt, a_butt = butter(FILTER_ORDER, normal_cutoff, btype='low')
        b_cheb, a_cheb = cheby1(FILTER_ORDER, CHEBY_RIPPLE_DB, normal_cutoff, btype='low')
        b_bess, a_bess = bessel(FILTER_ORDER, normal_cutoff, btype='low')
        
        print("Applying filters to the entire dataset")
        
        # 1. Raw Dynamic Magnitude (Zero-centered magnitude)
        raw_mag = np.sqrt(df[0]**2 + df[1]**2 + df[2]**2)
        df['dyn_mag_raw'] = raw_mag - raw_mag.mean()
        
        # 2. Butterworth Filter
        df['dyn_mag_butter'] = get_dynamic_magnitude(df, b_butt, a_butt)
        
        # 3. Chebyshev Type I Filter
        df['dyn_mag_cheby'] = get_dynamic_magnitude(df, b_cheb, a_cheb)
        
        # 4. Bessel-Thomson Filter
        df['dyn_mag_bessel'] = get_dynamic_magnitude(df, b_bess, a_bess)
        
        # Extract and plot the 10-second segments
        for start_time in start_times:
            end_time = start_time + 10.0
            print(f"  -> Plotting segment: {start_time}s to {end_time}s")
            
            mask = (df['time'] >= start_time) & (df['time'] < end_time)
            df_seg = df.loc[mask]
            
            if df_seg.empty:
                print(f"     No data found in window {start_time}s - {end_time}s.")
                continue
                
            fig, axes = plt.subplots(5, 1, figsize=(12, 12.5), sharex=True, sharey=True)
            time_sec = df_seg['time']
            
            axes[0].plot(time_sec, df_seg['dyn_mag_raw'], color='black', lw=1.5, alpha=0.8)
            axes[0].set_title('Raw Dynamic Magnitude')
            
            axes[1].plot(time_sec, df_seg['dyn_mag_butter'], color='blue', lw=1.5, alpha=0.8)
            axes[1].set_title(f'Butterworth Filter ({CUTOFF_HZ}Hz Lowpass)')
            
            axes[2].plot(time_sec, df_seg['dyn_mag_cheby'], color='green', lw=1.5, alpha=0.8)
            axes[2].set_title(f'Chebyshev Type I Filter ({CUTOFF_HZ}Hz, {CHEBY_RIPPLE_DB}dB Ripple)')
            
            axes[3].plot(time_sec, df_seg['dyn_mag_bessel'], color='red', lw=1.5, alpha=0.8)
            axes[3].set_title(f'Bessel-Thomson Filter ({CUTOFF_HZ}Hz)')
            
            axes[4].plot(time_sec, df_seg['dyn_mag_raw'], color='black', lw=1.5, alpha=0.3, label='Raw')
            axes[4].plot(time_sec, df_seg['dyn_mag_butter'], color='blue', lw=1.5, alpha=0.7, label='Butterworth')
            axes[4].plot(time_sec, df_seg['dyn_mag_cheby'], color='green', lw=1.5, alpha=0.7, label='Chebyshev')
            axes[4].plot(time_sec, df_seg['dyn_mag_bessel'], color='red', lw=1.5, alpha=0.7, label='Bessel')
            axes[4].set_title('Overlay of All Signals')
            axes[4].set_xlabel('Time (seconds)')
            axes[4].legend(loc='upper right', fontsize=8)

            for ax in axes:
                ax.set_ylabel('Amplitude (g)')
                ax.grid(True, alpha=0.3)
                
            plt.suptitle(f'30Hz Filter Comparison: {filename}\nTime: {start_time}s to {end_time}s', fontsize=14, fontweight='bold')
            plt.tight_layout()
            plt.savefig(os.path.join(OUTPUT_DIR, f"{filename.replace('.csv', '')}_{int(start_time)}s_filter_comp.pdf"), format='pdf', bbox_inches='tight')
            plt.close()

if __name__ == '__main__':
    main()
