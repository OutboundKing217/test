#!/usr/bin/env python3
"""
Plot Avalanche Segments

Generates 10-second visualization segments showing smoothed dynamic magnitude 
and highlights the detected avalanches (filled in under the curve) based on 
the pipeline established in main.py.
"""

import os
import sys
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

#matplotlib.rcParams["font.family"] = "Arial"
matplotlib.rcParams["pdf.fonttype"] = 42      # Ensures text is saved as real characters
matplotlib.rcParams["ps.fonttype"] = 42
matplotlib.rcParams["svg.fonttype"] = "none"  # Keep SVG text selectable

# Ensure we can import local modules
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from data_io import load_data
from signal_processing2 import butter_gravity_filter, detect_behavioral_events

# =============================================================================
# CONFIGURATION
# =============================================================================

DATA_DIR = "../../data/fewHealthy"  # Directory containing the 30Hz CSVs
OUTPUT_DIR = "../../output/avalanche_segments10"

# Define the 10-second segments to plot. 
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
]

# =============================================================================

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
        
        # Match pipeline from main.py
        print("Applying Butterworth gravity filter...")
        df = butter_gravity_filter(df, sampling_rate=fs)
        
        print("Applying dynamic magnitude smoothing...")
        #smoothing_samples = max(1, int((1/30) * fs))
        smoothing_samples = 10
        df['dynamic_mag_butter_smooth'] = df['dynamic_mag_butter'].rolling(
            window=smoothing_samples, center=True, min_periods=1
        ).mean()
        
        print("Calculating global threshold...")
        global_threshold = np.nanpercentile(df['dynamic_mag_butter_smooth'], 50)
        
        for start_time in start_times:
            end_time = start_time + 10.0
            print(f"  -> Plotting segment: {start_time}s to {end_time}s")
            
            mask = (df['time'] >= start_time) & (df['time'] < end_time)
            df_seg = df.loc[mask].copy()
            
            if df_seg.empty:
                print(f"     No data found in window {start_time}s - {end_time}s.")
                continue
                
            # Detect events natively via signal_processing2 module on this segment
            events_df = detect_behavioral_events(
                df_seg, 
                column='dynamic_mag_butter_smooth',
                median_threshold=global_threshold, 
                sampling_rate=fs,
                min_duration=0,
                min_auc=0
            )
            
            fig, ax = plt.subplots(figsize=(12, 5))
            time_sec = df_seg['time']
            dynamic = df_seg['dynamic_mag_butter_smooth']
            
            # Plot continuous line and threshold
            ax.plot(time_sec, dynamic, color='black', lw=1.5, label='Smoothed Dynamic Mag')
            ax.axhline(global_threshold, color='red', linestyle='--', lw=2, alpha=0.7, label=f'Global Threshold ({global_threshold:.4f}g)')
            
            # "Fill in" the avalanches
            added_avalanche_label = False
            for _, event in events_df.iterrows():
                e_start = event['start_time']
                e_end = event['end_time']
                
                e_mask = (time_sec >= e_start) & (time_sec <= e_end)
                
                label = 'Avalanche Area' if not added_avalanche_label else ""
                ax.fill_between(time_sec[e_mask], global_threshold, dynamic[e_mask], color='orange', alpha=0.6, label=label)
                added_avalanche_label = True

            ax.set_title(f'Avalanche Segments: {filename}\nTime: {start_time}s to {end_time}s', fontsize=14, fontweight='bold')
            ax.set_xlabel('Time (seconds)', fontsize=12)
            ax.set_ylabel('Amplitude (g)', fontsize=12)
            ax.set_xlim(start_time, end_time)
            
            # Give enough headroom for the legend
            ax.set_ylim(bottom=0, top=max(dynamic.max() * 1.3, global_threshold * 2))
            ax.legend(loc='upper right')
            ax.grid(True, alpha=0.3)
            
            plt.tight_layout()
            out_name = f"{filename.replace('.csv', '')}_{int(start_time)}s_avalanches.pdf"
            plt.savefig(os.path.join(OUTPUT_DIR, out_name), format='pdf', bbox_inches='tight')
            plt.close()

if __name__ == '__main__':
    main()