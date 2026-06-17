"""
Find Flatlines

Scans a 30Hz accelerometry file for periods where all three axes 
stay within a specific tolerance (±0.05) of a central value for at least 1 second.
"""

import os
import sys
import argparse
import numpy as np
import pandas as pd

# Ensure we can import local modules
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
try:
    from data_io import load_data
except ImportError:
    print("Warning: Could not import load_data from data_io. Using fallback pandas loader.")
    def load_data(filepath):
        df = pd.read_csv(filepath, skiprows=10, header=None)
        return df, 30.0

def find_flatlines(filepath, tolerance=0.05, window_size=30):
    print(f"Loading {os.path.basename(filepath)}...")
    df, fs = load_data(filepath)
    
    # Ensure we have the necessary columns (0, 1, 2 for X, Y, Z)
    if not all(col in df.columns for col in [0, 1, 2]):
        print("Error: DataFrame must contain columns 0, 1, and 2 for X, Y, and Z axes.")
        return

    # If the data is "within 0.05 of a number", the maximum distance 
    # between the highest and lowest points in the window is 2 * 0.05 = 0.10.
    max_range = tolerance * 2

    print("Scanning for flatlines...")
    # rolling().max() and rolling().min() align with the right edge of the window
    flat_x = (df[0].rolling(window_size).max() - df[0].rolling(window_size).min()) <= max_range
    flat_y = (df[1].rolling(window_size).max() - df[1].rolling(window_size).min()) <= max_range
    flat_z = (df[2].rolling(window_size).max() - df[2].rolling(window_size).min()) <= max_range

    # Combined condition: all 3 axes are "flat" simultaneously
    is_flat_end = flat_x & flat_y & flat_z

    # Find contiguous segments of True values
    # A transition from False to True is the start of a flat end-point
    # A transition from True to False is the end of a flat end-point
    is_flat_end_np = is_flat_end.fillna(False).astype(int).values
    diff = np.diff(np.insert(is_flat_end_np, 0, 0))
    
    starts = np.where(diff == 1)[0]
    ends = np.where(diff == -1)[0] - 1
    
    # If the array ends while still flat
    if len(starts) > len(ends):
        ends = np.append(ends, len(is_flat_end_np) - 1)

    segments = []
    # Use standard time generation if time column is missing
    time_vals = df['time'].values if 'time' in df.columns else np.arange(len(df)) / fs

    for s, e in zip(starts, ends):
        # The 30-sample window that triggered at index `s` started 29 samples prior
        actual_start_idx = max(0, s - window_size + 1)
        actual_end_idx = e
        
        start_time = time_vals[actual_start_idx]
        end_time = time_vals[actual_end_idx]
        duration = (actual_end_idx - actual_start_idx + 1) / fs
        
        segments.append((start_time, end_time, duration))

    num_segments = len(segments)
    print("-" * 50)
    print(f"Results for: {os.path.basename(filepath)}")
    print(f"Total flatline segments found: {num_segments}")
    print("-" * 50)

    if 0 < num_segments < 100:
        print("Locations (Start Time -> End Time [Duration]):")
        for idx, (st, et, dur) in enumerate(segments, 1):
            print(f"  {idx:2d}. {st:8.2f}s -> {et:8.2f}s  [{dur:.2f}s]")
    elif num_segments >= 100:
        print("More than 100 segments found. Suppressing location output.")

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Find periods where accelerometry data flatlines.")
    parser.add_argument("filepath", type=str, help="Path to the 30Hz CSV file")
    parser.add_argument("--tolerance", type=float, default=0.05, help="Tolerance for flatness (default: 0.05)")
    parser.add_argument("--window", type=int, default=30, help="Minimum duration in samples (default: 30)")
    args = parser.parse_args()

    if not os.path.exists(args.filepath):
        sys.exit(f"Error: File not found: {args.filepath}")

    find_flatlines(args.filepath, args.tolerance, args.window)