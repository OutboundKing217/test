#!/usr/bin/env python3
"""
Longitudinal Analysis Plotter

This script reads a master scale-free results CSV and generates longitudinal
line plots for Power-Law Range, Percent Scale-Free Blocks, and Avalanche Count.
Each subject is represented by a unique color across timepoints.

Usage:
    python longitudinal.py --csv_path path/to/master_scale_free_results.csv --output_dir ../output/longitudinal
"""

import os
import argparse
import pandas as pd
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sns

matplotlib.rcParams["pdf.fonttype"] = 42
matplotlib.rcParams["ps.fonttype"] = 42
matplotlib.rcParams["svg.fonttype"] = "none"

def load_and_aggregate(csv_path, metadata_csv=None):
    print(f"Loading data from {csv_path}...")
    df = pd.read_csv(csv_path)
    
    # Ensure necessary columns exist
    req_cols = ['subject_id', 'timepoint_number', 'arm', 'is_scale_free', 'success', 'power_law_range', 'n_events']
    for col in req_cols:
        if col not in df.columns:
            raise ValueError(f"Missing required column: {col}")
            
    df['subject_id'] = df['subject_id'].astype(str)
    
    if metadata_csv:
        print(f"Loading metadata from {metadata_csv} to recompute timepoints as days...")
        meta_df = pd.read_csv(metadata_csv)
        meta_df['subject_id'] = meta_df['subject_id'].astype(str)
        meta_df['record date'] = pd.to_datetime(meta_df['record date'])
        meta_df = meta_df.dropna(subset=['record date']).sort_values(['subject_id', 'record date'])
        
        # Recreate the chronological numbering logic used in main.py
        meta_df['session_month'] = meta_df['record date'].dt.strftime('%Y-%m')
        meta_df['chronological_timepoint'] = meta_df.groupby('subject_id')['session_month'].rank(method='dense').astype(int)
        
        # Calculate days since first recording per subject
        first_dates = meta_df.groupby('subject_id')['record date'].transform('min')
        meta_df['days_since_first'] = (meta_df['record date'] - first_dates).dt.days.astype(int)
        
        mapping = meta_df.groupby(['subject_id', 'chronological_timepoint'])['days_since_first'].min().reset_index()
        df = df.merge(mapping, left_on=['subject_id', 'timepoint_number'], right_on=['subject_id', 'chronological_timepoint'], how='left')
        df['timepoint_number'] = df['days_since_first'].fillna(df['timepoint_number'])
        
    # Clean boolean columns safely
    df['is_scale_free'] = df['is_scale_free'].astype(bool)
    df['success'] = df['success'].astype(bool)
    
    # Aggregate per subject, timepoint, arm
    aggregated = []
    
    for (subj, tp, arm), group in df.groupby(['subject_id', 'timepoint_number', 'arm']):
        total_blocks = len(group)
        if total_blocks == 0:
            continue
            
        pct_sf = (group['is_scale_free'].sum() / total_blocks) * 100
        total_avalanches = group['n_events'].sum()
        
        # Calculate range metrics only for successful analyses
        success_group = group[group['success'] == True]
        
        if len(success_group) > 0:
            mean_rng = success_group['power_law_range'].mean()
            median_rng = success_group['power_law_range'].median()
        else:
            mean_rng = np.nan
            median_rng = np.nan
            
        aggregated.append({
            'subject_id': subj,
            'timepoint_number': tp,
            'arm': arm,
            'pct_scale_free': pct_sf,
            'mean_range': mean_rng,
            'median_range': median_rng,
            'total_avalanches': total_avalanches
        })
        
    df_agg = pd.DataFrame(aggregated)
    return df_agg

def plot_metric(df_agg, metric_col, ylabel, output_dir, filename_prefix, timepoints_as_days=False):
    """Plots a specific metric across timepoints as a line graph."""
    df_plot = df_agg.dropna(subset=[metric_col])
    
    if df_plot.empty:
        print(f"No valid data to plot for {metric_col}.")
        return

    arms = df_plot['arm'].unique()
    
    for arm in arms:
        arm_data = df_plot[df_plot['arm'] == arm].copy()
        arm_data = arm_data.sort_values('timepoint_number')
        
        timepoints = sorted(arm_data['timepoint_number'].unique())
        
        if len(timepoints) < 2:
            print(f"Not enough timepoints to plot longitudinal data for {arm} - {metric_col}.")
            continue
            
        plt.figure(figsize=(10, 6))
        
        # Generate dynamic color palette so each subject is distinct
        unique_subjects = arm_data['subject_id'].nunique()
        palette = sns.color_palette("husl", n_colors=unique_subjects)
        
        # Trajectory plot (one line per subject)
        sns.lineplot(
            data=arm_data,
            x='timepoint_number',
            y=metric_col,
            hue='subject_id',
            marker='o',
            linewidth=2,
            markersize=8,
            palette=palette,
            alpha=0.7
        )
            
        # Add an overall mean trend
        mean_data = arm_data.groupby('timepoint_number')[metric_col].mean().reset_index()
        
        if timepoints_as_days:
            min_day = int(mean_data['timepoint_number'].min())
            max_day = int(mean_data['timepoint_number'].max())
            full_days = pd.DataFrame({'timepoint_number': range(min_day, max_day + 1)})
            mean_data = pd.merge(full_days, mean_data, on='timepoint_number', how='left')
            mean_data[metric_col] = mean_data[metric_col].rolling(window=90, min_periods=1, center=True).mean()
            mean_data = mean_data.dropna(subset=[metric_col])
            
            plt.plot(
                mean_data['timepoint_number'], 
                mean_data[metric_col], 
                color='black', 
                linewidth=3, 
                linestyle='--',
                label='90-Day Mean Trend',
                zorder=10
            )
        else:
            plt.plot(
                mean_data['timepoint_number'], 
                mean_data[metric_col], 
                color='black', 
                linewidth=3, 
                linestyle='--',
                marker='D', 
                markersize=8, 
                label='Mean Trend',
                zorder=10
            )
        
        if timepoints_as_days:
            plt.xlabel('Timepoint (Days)', fontsize=12)
        else:
            plt.xlabel('Timepoint', fontsize=12)
            
        plt.ylabel(ylabel, fontsize=12)
        plt.title(f'Longitudinal Progression: {ylabel} ({arm})', fontsize=14, fontweight='bold')
        plt.grid(True, alpha=0.3)
        
        if not timepoints_as_days:
            plt.xticks(timepoints)
        
        # Place legend nicely outside the plot
        plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left', title='Subject ID')
        
        plt.tight_layout()
        out_name = f"{filename_prefix}_{arm}.pdf"
        plt.savefig(os.path.join(output_dir, out_name), dpi=300, bbox_inches='tight')
        plt.close()
        print(f"Saved {out_name}")

def main():
    parser = argparse.ArgumentParser(description="Generate Longitudinal Plots across timepoints")
    parser.add_argument('--csv_path', type=str, required=True, help="Path to master_scale_free_results.csv")
    parser.add_argument('--output_dir', type=str, default='../output/longitudinal_analysis', help="Directory to save the plots")
    parser.add_argument('--timepoints_as_days', action='store_true', help="Format x-axis as days and allow automatic tick placement")
    parser.add_argument('--metadata_csv', type=str, default=None, help="Optional: Path to gt3x metadata CSV to convert chronological timepoints to days")
    
    args = parser.parse_args()
    
    os.makedirs(args.output_dir, exist_ok=True)
    
    try:
        df_agg = load_and_aggregate(args.csv_path, args.metadata_csv)
    except Exception as e:
        print(f"Error processing data: {e}")
        return
        
    csv_out = os.path.join(args.output_dir, 'longitudinal_aggregated_metrics.csv')
    df_agg.to_csv(csv_out, index=False)
    print(f"Saved aggregated metrics to {csv_out}")
        
    metrics = [
        ('mean_range', 'Mean Power-Law Range (decades)', 'longitudinal_mean_range'),
        ('median_range', 'Median Power-Law Range (decades)', 'longitudinal_median_range'),
        ('pct_scale_free', 'Percent Scale-Free Blocks (%)', 'longitudinal_pct_scale_free'),
        ('total_avalanches', 'Total Avalanche Count', 'longitudinal_total_avalanches')
    ]
    
    timepoints_as_days = args.timepoints_as_days or (args.metadata_csv is not None)
    print(f"\nGenerating plots in {args.output_dir}...")
    for col, ylabel, prefix in metrics:
        plot_metric(df_agg, col, ylabel, args.output_dir, prefix, timepoints_as_days)
        
    print("\nLongitudinal analysis complete.")

if __name__ == '__main__':
    main()
