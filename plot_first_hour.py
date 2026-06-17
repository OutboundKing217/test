"""
Utility to plot the first hour of accelerometer data for a specified subject.
"""

import os
import glob
import argparse
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

#matplotlib.rcParams["font.family"] = "Arial"
matplotlib.rcParams["pdf.fonttype"] = 42      # Ensures text is saved as real characters
matplotlib.rcParams["ps.fonttype"] = 42
matplotlib.rcParams["svg.fonttype"] = "none"  # Keep SVG text selectable

from data_io import load_data, get_file_metadata
from signal_processing2 import butter_gravity_filter

def plot_first_hour(subject_id: str, data_dir: str, output_dir: str):
    """
    Finds all raw data files for a given subject, extracts the first hour,
    and plots the dynamic XYZ and magnitude data.
    """
    os.makedirs(output_dir, exist_ok=True)
    
    # Find raw files for this subject
    pattern = os.path.join(data_dir, f"*{subject_id}*_30Hz.csv")
    subject_files = glob.glob(pattern)
    
    if not subject_files:
        subject_files = glob.glob(os.path.join(data_dir, f"Patient_{subject_id}", "**", "*.gt3x"), recursive=True)
        if not subject_files:
            subject_files = glob.glob(os.path.join(data_dir, "**", f"*{subject_id}*.gt3x"), recursive=True)
            
    if not subject_files:
        print(f"No raw data files found for subject {subject_id} in {data_dir}")
        return
        
    print(f"Found {len(subject_files)} file(s) for subject {subject_id}.")
    
    plt.style.use("seaborn-v0_8-whitegrid")
    
    for filepath in subject_files:
        metadata = get_file_metadata(filepath)
        arm = metadata['arm']
        timepoint = metadata['timepoint']
        
        print(f"Processing Subject {subject_id} - Timepoint {timepoint} ({arm})...")
        
        # Load and filter data
        df, fs = load_data(filepath)
        if df.empty:
            print(f"  -> No data found for {filepath}")
            continue
        df = butter_gravity_filter(df, sampling_rate=fs)
        
        # Filter to the first hour (3600 seconds)
        df_hour = df[df['time'] <= 3600].copy()
        
        if df_hour.empty:
            print(f"  -> No data found in the first hour for {filepath}")
            continue
            
        # Create subplots for X, Y, Z, and Magnitude
        fig, axes = plt.subplots(4, 1, figsize=(15, 12), sharex=True)
        time_mins = df_hour['time'] / 60.0
        
        # Plot each axis
        axes[0].plot(time_mins, df_hour['dynamic_x_butter'], color='red', linewidth=0.5)
        axes[0].set_ylabel('X (g)', fontsize=12)
        axes[0].set_title(f"{subject_id} - Timepoint {timepoint} ({arm}) - First Hour", fontsize=14, fontweight='bold')
        
        axes[1].plot(time_mins, df_hour['dynamic_y_butter'], color='green', linewidth=0.5)
        axes[1].set_ylabel('Y (g)', fontsize=12)
        
        axes[2].plot(time_mins, df_hour['dynamic_z_butter'], color='blue', linewidth=0.5)
        axes[2].set_ylabel('Z (g)', fontsize=12)
        
        axes[3].plot(time_mins, df_hour['dynamic_mag_butter'], color='black', linewidth=0.5)
        axes[3].set_ylabel('Magnitude (g)', fontsize=12)
        axes[3].set_xlabel('Time (minutes)', fontsize=12)
        
        for ax in axes:
            ax.set_xlim([0, 60])
            
        plt.tight_layout()
        
        out_path = os.path.join(output_dir, f"{subject_id}_TP{timepoint}_{arm}_first_hour.png")
        plt.savefig(out_path, dpi=300, bbox_inches='tight')
        plt.close(fig)
        
        print(f"  -> Saved plot to {out_path}")

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Plot the first hour of accelerometer data for a subject.")
    parser.add_argument('--subject', type=str, required=True, help="Subject ID (e.g., '001' or 'PMC12345_001')")
    parser.add_argument('--data_dir', type=str, required=True, help="Path to the directory containing raw CSV files")
    parser.add_argument('--output_dir', type=str, default='../output/first_hour_plots', help="Directory to save the plots")
    
    args = parser.parse_args()
    
    plot_first_hour(args.subject, args.data_dir, args.output_dir)
