"""
Plotting Module

Generates the 25-hour timeseries and hourly metric plots for subjects.
Designed to be run concurrently using multiprocessing to speed up generation.
"""

import os
import glob
import pandas as pd
import numpy as np
import matplotlib
matplotlib.use('Agg')  # Required for thread-safe/headless plotting in multiprocessing
import matplotlib.pyplot as plt
from concurrent.futures import ProcessPoolExecutor, as_completed

#matplotlib.rcParams["font.family"] = "Arial"
matplotlib.rcParams["pdf.fonttype"] = 42      # Ensures text is saved as real characters
matplotlib.rcParams["ps.fonttype"] = 42
matplotlib.rcParams["svg.fonttype"] = "none"  # Keep SVG text selectable

# Import from our custom modules
from data_io import load_data, load_downsampled_data, get_file_metadata
from signal_processing2 import butter_gravity_filter

def plot_subject_timeline(subject_id: str, df_subject_results: pd.DataFrame, data_dir: str, output_dir: str, file_lookup: dict = None) -> str:
    """
    Generates a 25-hour timeseries plot for a single subject, separating
    by both arm and timepoint.
    
    Parameters:
    -----------
    subject_id : str
        The ID of the subject (e.g., 'PMC12345_67890').
    df_subject_results : pd.DataFrame
        The subset of the master results dataframe belonging to this subject.
    data_dir : str
        Path to the directory containing raw CSV files.
    output_dir : str
        Path to the directory where plots should be saved.
    file_lookup : dict
        Optional mapping of (subject_id, arm, timepoint) to filepath to bypass slow file searching.
        
    Returns:
    --------
    str
        A status message indicating success or failure.
    """
    try:
        subject_files = []
        if file_lookup:
            subject_files = [f for f, meta in file_lookup.items() if str(meta.get('subject_id', 'Unknown')) == str(subject_id)]
        else:
            # Find raw files for this subject (explicitly filtering for 30Hz to avoid 1Hz bugs)
            subject_files = glob.glob(os.path.join(data_dir, f"*{subject_id}*_30Hz.csv"))
            if not subject_files:
                # Fallback file searcher to catch nested .gt3x paths
                subject_files = glob.glob(os.path.join(data_dir, f"Patient_{subject_id}", "**", "*.gt3x"), recursive=True)
                if not subject_files:
                    subject_files = glob.glob(os.path.join(data_dir, f"patient_{subject_id}", "**", "*.gt3x"), recursive=True)
                if not subject_files:
                    subject_files = glob.glob(os.path.join(data_dir, "**", f"*{subject_id}*.gt3x"), recursive=True)
                
        if not subject_files:
            return f"No raw data files found for subject {subject_id}."
            
        # Plot styling
        plt.style.use("seaborn-v0_8-whitegrid")
        
        for filepath in subject_files:
            if file_lookup and filepath in file_lookup:
                metadata = file_lookup[filepath]
            else:
                metadata = get_file_metadata(filepath)
            arm = metadata.get('arm', 'Unknown')
            timepoint = metadata.get('timepoint', -1)  # Extract timepoint
            
            # Smooth dynamic magnitude (ensuring window is at least 1 sample)
            #smoothing_samples = max(1, int(0.2 * fs))
            #df['dynamic_mag_smooth'] = df['dynamic_mag_butter'].rolling(
            #    window=smoothing_samples, center=True, min_periods=1
            #).mean()
            
            # Filter results by BOTH arm and timepoint
            df_plot_results = df_subject_results[
                (df_subject_results['arm'] == arm) & 
                (df_subject_results['timepoint_number'] == timepoint)
            ].sort_values('hour_number')
            
            if df_plot_results.empty:
                continue

            # Load and prepare raw data for plotting
            df, fs = load_downsampled_data(filepath)
            if df.empty:
                continue
            df = butter_gravity_filter(df, sampling_rate=fs)
                
            global_threshold = df_plot_results['threshold'].iloc[0]
            
            # Create the figure
            fig, axes = plt.subplots(2, 1, figsize=(20, 10), height_ratios=[3, 1])
            
            # --- TOP PLOT: Timeseries ---
            time_hours = df['time'] / 3600
            
            # Linear interpolation smoothed over a 30-point rolling average
            dynamic_mag = df['dynamic_mag_butter'].rolling(window=30, min_periods=1, center=True).mean()
            
            ds = max(1, int(fs))
            axes[0].plot(time_hours[::ds], dynamic_mag[::ds], alpha=0.7, linewidth=0.8, color='blue', label='Dynamic Magnitude')
            axes[0].axhline(global_threshold, color='red', linestyle='--', linewidth=2, alpha=0.7, label=f'Threshold ({global_threshold:.4f}g)')
            
            axes[0].set_xlabel('Time (hours)', fontsize=12)
            axes[0].set_ylabel('Dynamic Magnitude (g)', fontsize=12)
            # Include Timepoint in the title
            axes[0].set_title(f'{subject_id} - Timepoint {timepoint} ({arm}): Dynamics', fontsize=14, fontweight='bold')
            axes[0].set_xlim([0, time_hours.max()])
            axes[0].legend(loc='upper right')
            
            # --- BOTTOM PLOT: Hourly Metrics ---
            ax2_twin = axes[1].twinx()
            
            df_success = df_plot_results[df_plot_results['success'] == True]
            df_failed = df_plot_results[df_plot_results['success'] == False]
            
            # Plot Goodness of Fit (Left Y-Axis)
            if not df_success.empty:
                hour_centers = df_success['hour_number'] + 0.5
                axes[1].plot(hour_centers, df_success['goodness_of_fit'], 'o-', color='green', linewidth=2, label='GOF')
                
                # Highlight Scale-Free points
                scale_free = df_success[df_success['is_scale_free'] == True]
                if not scale_free.empty:
                    axes[1].scatter(scale_free['hour_number'] + 0.5, scale_free['goodness_of_fit'], 
                                    color='green', s=100, edgecolors='darkgreen', zorder=5, label='Scale-Free')
                
                # Plot Power Law Range (Right Y-Axis)
                ax2_twin.plot(hour_centers, df_success['power_law_range'], 's-', color='orange', linewidth=2, label='Range (Decades)')
            
            # Mark failed/quiescent hours
            if not df_failed.empty:
                axes[1].scatter(df_failed['hour_number'] + 0.5, [0.75]*len(df_failed), 
                                color='red', marker='x', s=100, linewidths=2, label='Quiescent / Failed')

            axes[1].axhline(0.8, color='green', linestyle='--', alpha=0.5)
            axes[1].set_ylim([0.7, 1.0])
            axes[1].set_xlim([0, time_hours.max()])
            axes[1].set_xlabel('Time (hours)', fontsize=12)
            axes[1].set_ylabel('Goodness of Fit', color='green', fontsize=12)
            ax2_twin.set_ylabel('Power-Law Range', color='orange', fontsize=12)
            
            handles_1, labels_1 = axes[1].get_legend_handles_labels()
            if handles_1:
                axes[1].legend(loc='upper left')
                
            handles_2, labels_2 = ax2_twin.get_legend_handles_labels()
            if handles_2:
                ax2_twin.legend(loc='upper right')
            
            plt.tight_layout()
            
            # Save and close (include Timepoint in the filename)
            out_path = os.path.join(output_dir, f"{subject_id}_TP{timepoint}_{arm}_timeline.png")
            plt.savefig(out_path, dpi=100, bbox_inches='tight')
            plt.close(fig)
            
        return f"Successfully plotted {subject_id}."
    
    except Exception as e:
        return f"Error plotting {subject_id}: {str(e)}"

def generate_all_plots_concurrently(df_results: pd.DataFrame, data_dir: str, output_dir: str, max_workers: int = 4, file_lookup: dict = None):
    """
    Orchestrates the parallel generation of plots for all subjects.
    """
    plots_dir = os.path.join(output_dir, 'all_subject_plots')
    os.makedirs(plots_dir, exist_ok=True)
    
    unique_subjects = df_results['subject_id'].unique()
    total_subjects = len(unique_subjects)
    
    print(f"\nGenerating plots for {total_subjects} subjects concurrently...")
    
    with ProcessPoolExecutor(max_workers=max_workers) as executor:
        futures = []
        for subject in unique_subjects:
            df_subj = df_results[df_results['subject_id'] == subject]
            futures.append(executor.submit(plot_subject_timeline, subject, df_subj, data_dir, plots_dir, file_lookup))
            
        for i, future in enumerate(as_completed(futures), 1):
            print(f"[{i}/{total_subjects}] {future.result()}")


def plot_hourly_power_law(events_df: pd.DataFrame, hour_record: dict, output_dir: str):
    """
    Plots a log-log histogram of AUC vs Frequency (Probability Density) for a single hour.
    Overlays the theoretical power-law fit and shades the valid region.
    """
    from matplotlib.figure import Figure
    from matplotlib.backends.backend_pdf import FigureCanvasPdf
    
    # Create output directory for hourly plots
    hourly_dir = os.path.join(output_dir, 'hourly_pdfs')
    if not os.path.exists(hourly_dir):
        os.makedirs(hourly_dir, exist_ok=True)

    fig = Figure(figsize=(8, 6))
    canvas = FigureCanvasPdf(fig)
    ax = fig.add_subplot(111)

    status = "SUCCESS" if hour_record.get('success', False) else "FAILED"
    ax.set_title(f"{hour_record['subject_id']} ({hour_record['arm']}) TP{hour_record['timepoint_number']} - Hour {hour_record['hour_number']:.2f} [{status}]", fontweight='bold')
    ax.set_xlabel('Event Size (AUC)', fontsize=12)
    ax.set_ylabel('Probability Density (Frequency)', fontsize=12)
    ax.grid(True, alpha=0.3, which='both')
    
    filename = f"{hour_record['subject_id']}_TP{hour_record['timepoint_number']}_{hour_record['arm']}_Hour{hour_record['hour_number']:05.2f}.pdf"
    filepath = os.path.join(hourly_dir, filename)
    
    if events_df.empty or 'auc' not in events_df.columns:
        print(f"  -> Hourly Plot Hour {hour_record['hour_number']}: 0 events detected. Plotting empty canvas.")
        ax.text(0.5, 0.5, '0 Events Detected in this Hour', 
                horizontalalignment='center', verticalalignment='center', 
                transform=ax.transAxes, fontsize=14, color='red')
        fig.savefig(filepath, bbox_inches='tight')
        return
    
    auc = events_df['auc'].values
    auc = auc[auc > 0]
    
    if len(auc) < 2:
        print(f"  -> Hourly Plot Hour {hour_record['hour_number']}: Not enough events to plot PDF.")
        ax.text(0.5, 0.5, f'Insufficient events ({len(auc)}) to plot PDF', 
                horizontalalignment='center', verticalalignment='center', 
                transform=ax.transAxes, fontsize=14, color='red')
        fig.savefig(filepath, bbox_inches='tight')
        return
        
    # Create log-spaced bins for the histogram
    min_val, max_val = np.min(auc), np.max(auc)
    dbrange = np.log10(max_val) - np.log10(min_val)
        
    bins = np.logspace(np.log10(min_val), np.log10(max_val), int(10 * dbrange) + 1)
    counts, _ = np.histogram(auc, bins=bins)
    bin_widths = np.diff(bins)
    bin_centers = bins[:-1] + bin_widths / 2
    
    # Normalize to Probability Density
    pdf = counts / (bin_widths * len(auc))
    
    # Filter out empty bins for clean log-log plotting
    mask = pdf > 0
    x_plot = bin_centers[mask]
    y_plot = pdf[mask]
    
    # Plot raw data PDF
    ax.loglog(x_plot, y_plot, 'ko', alpha=0.6, label='Data PDF', rasterized=True)
    
    # If the math successfully found a fit, overlay the fit and shade the region
    if hour_record.get('success', False):
        mm = hour_record['lower_cutoff']
        MM = hour_record['upper_cutoff']
        tau = hour_record['tau']
        n_fit_samples = hour_record['n_samples']
        
        # Shade the fit region
        ax.axvspan(mm, MM, color='green', alpha=0.2, label=f'Fit Region ({hour_record["power_law_range"]:.2f} decades)')
        
        # Plot theoretical fit line
        x_fit = np.logspace(np.log10(mm), np.log10(MM), 50)
        if tau == 1.0:
            y_fit = 1.0 / (np.log(MM) - np.log(mm)) / x_fit
        else:
            y_fit = ((tau - 1) / (mm**(1 - tau) - MM**(1 - tau))) * (x_fit ** -tau)

        # Scale the fit down to match the global PDF normalization
        scale_factor = n_fit_samples / len(auc)
        y_fit = y_fit * scale_factor
            
        ax.loglog(x_fit, y_fit, 'r-', linewidth=2.5, label=f'Power-law Fit (tau={tau:.2f})')
    else:
        if len(auc) < 20:
            ax.plot([], [], ' ', label=f'Insufficient events for fit ({len(auc)} < 20)')
        else:
            ax.plot([], [], ' ', label='No valid power-law fit found')
        
    ax.legend(loc='lower left')
    
    fig.savefig(filepath, dpi=100)