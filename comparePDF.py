#!/usr/bin/env python3
"""
Compare PDF Plotter

Runs scale-free analysis on two specific raw accelerometry files for a specified 
hour each, and plots an overlaid log-log PDF of the avalanche sizes with their 
respective power-law fits.
"""

import os
import argparse
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

from data_io import load_data, get_file_metadata
from signal_processing2 import butter_gravity_filter, detect_behavioral_events
from scale_free_math2 import analyze_scale_free_events

# Plot styling
matplotlib.rcParams["pdf.fonttype"] = 42
matplotlib.rcParams["ps.fonttype"] = 42
matplotlib.rcParams["svg.fonttype"] = "none"


def process_file_hour(filepath: str, hour: float):
    """
    Loads a file, isolates the specific hour, detects events, and calculates scale-free properties.
    """
    print(f"Processing {os.path.basename(filepath)} for Hour {hour}...")
    metadata = get_file_metadata(filepath)
    df, fs = load_data(filepath)
    if df.empty:
        return metadata, pd.DataFrame(), {}
    df = butter_gravity_filter(df, sampling_rate=fs)
    
    # Smooth dynamic magnitude (using same parameters as main.py)
    smoothing_samples = 6
    df['dynamic_mag_butter_smooth'] = df['dynamic_mag_butter'].rolling(
        window=smoothing_samples, center=True, min_periods=1
    ).mean()
    
    # Global threshold calculated over the entire file
    global_threshold = np.nanpercentile(df['dynamic_mag_butter_smooth'], 50)
    
    hour_start = hour * 3600
    hour_end = hour_start + 3600
    
    mask = (df['time'] >= hour_start) & (df['time'] < hour_end)
    df_hour = df.loc[mask].copy()
    
    if df_hour.empty:
        print(f"No data found in hour {hour} for {filepath}.")
        return metadata, pd.DataFrame(), {}
        
    events_df = detect_behavioral_events(
        df_hour,
        column='dynamic_mag_butter_smooth',
        median_threshold=global_threshold,
        sampling_rate=fs,
        min_duration=0,
        min_auc=0
    )
    
    sf_results = analyze_scale_free_events(events_df, event_size_column='auc', verbose=False)
    
    return metadata, events_df, sf_results


def plot_pdf(ax, events_df, sf_results, metadata, hour, color, marker):
    """
    Plots the data PDF and theoretical power-law fit on the given axis.
    """
    subject_id = metadata.get('subject_id', 'Unknown')
    arm = metadata.get('arm', 'Unknown')
    label_prefix = f"{subject_id} ({arm}) Hr {hour}"

    #if arm == "LUE":
    #    label_prefix = "Healthy"
    #elif arm == "RUE":
    #    label_prefix = "Stroke"
    
    if events_df.empty or 'auc' not in events_df.columns:
        print(f"No events to plot for {label_prefix}.")
        return
        
    auc = events_df['auc'].values
    auc = auc[auc > 0]
    
    if len(auc) < 2:
        print(f"Not enough events to plot PDF for {label_prefix}.")
        return
        
    # Calculate PDF with log-spaced bins
    min_val, max_val = np.min(auc), np.max(auc)
    dbrange = np.log10(max_val) - np.log10(min_val)
    bins = np.logspace(np.log10(min_val), np.log10(max_val), int(10 * dbrange) + 1)
    counts, _ = np.histogram(auc, bins=bins)
    bin_widths = np.diff(bins)
    bin_centers = bins[:-1] + bin_widths / 2
    
    pdf = counts / (bin_widths * len(auc))
    
    # Filter out empty bins for log-log plotting
    mask = pdf > 0
    x_plot = bin_centers[mask]
    y_plot = pdf[mask]
    
    # Plot Data PDF
    ax.loglog(x_plot, y_plot, marker, color=color, alpha=0.6, label=f'{label_prefix} Data', rasterized=True)
    
    # Plot Fit if successful
    if sf_results.get('success', False):
        mm = sf_results['lower_cutoff']
        MM = sf_results['upper_cutoff']
        tau = sf_results['tau']
        n_fit_samples = sf_results['n_samples']
        decades = sf_results['power_law_range']
        
        x_fit = np.logspace(np.log10(mm), np.log10(MM), 50)
        if tau == 1.0:
            y_fit = 1.0 / (np.log(MM) - np.log(mm)) / x_fit
        else:
            y_fit = ((tau - 1) / (mm**(1 - tau) - MM**(1 - tau))) * (x_fit ** -tau)
            
        scale_factor = n_fit_samples / len(auc)
        y_fit = y_fit * scale_factor
        
        ax.loglog(x_fit, y_fit, '-', color=color, linewidth=2, 
                  label=f'{label_prefix} Fit (τ={tau:.2f}, {decades:.2f} dec)')
    else:
        print(f"Scale-free fit failed for {label_prefix}.")


def main():
    parser = argparse.ArgumentParser(description="Compare PDF of two subjects for specific hours.")
    parser.add_argument('--file1', type=str, required=True, help="Path to first subject's CSV file")
    parser.add_argument('--hour1', type=float, required=True, help="Hour to analyze for first subject (e.g., 0)")
    parser.add_argument('--file2', type=str, required=True, help="Path to second subject's CSV file")
    parser.add_argument('--hour2', type=float, required=True, help="Hour to analyze for second subject (e.g., 0)")
    parser.add_argument('--output', type=str, default="compared_pdfs.pdf", help="Output plot filename")
    
    args = parser.parse_args()
    
    if not os.path.exists(args.file1):
        print(f"Error: {args.file1} does not exist.")
        return
    if not os.path.exists(args.file2):
        print(f"Error: {args.file2} does not exist.")
        return
        
    meta1, events1, sf1 = process_file_hour(args.file1, args.hour1)
    meta2, events2, sf2 = process_file_hour(args.file2, args.hour2)
    
    fig, ax = plt.subplots(figsize=(10, 8))
    
    plot_pdf(ax, events1, sf1, meta1, args.hour1, color='#7a5195', marker='o')
    plot_pdf(ax, events2, sf2, meta2, args.hour2, color='#ef5675', marker='s')
    
    ax.set_xlabel('Event Size (AUC)', fontsize=14)
    ax.set_ylabel('Probability Density', fontsize=14)
    ax.set_title('1-hour Stroke vs. Healthy PDF comparison', fontsize=16, fontweight='bold')
    ax.grid(True, alpha=0.3, which='both')
    ax.legend(fontsize=11)
    
    plt.tight_layout()
    plt.savefig(args.output, dpi=300, bbox_inches='tight')
    print(f"Successfully generated comparison plot: {args.output}")

if __name__ == "__main__":
    main()
