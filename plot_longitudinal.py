#!/usr/bin/env python3
"""
Longitudinal Analysis Plotter

This script reads a master scale-free results CSV and generates longitudinal
plots (e.g., across multiple timepoints) for Mean/Median Power-Law Range
and Percent Scale-Free Blocks.

Usage:
    python plot_longitudinal.py --csv_path path/to/master_scale_free_results.csv --output_dir ../output/longitudinal
"""

import os
import argparse
import pandas as pd
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.cm as cm
import matplotlib.colors as mcolors
import seaborn as sns

matplotlib.rcParams["pdf.fonttype"] = 42
matplotlib.rcParams["ps.fonttype"] = 42
matplotlib.rcParams["svg.fonttype"] = "none"

def load_and_aggregate(csv_path, combine_arms=False):
    print(f"Loading data from {csv_path}...")
    df = pd.read_csv(csv_path)
    
    # Ensure necessary columns
    req_cols = ['subject_id', 'timepoint_number', 'is_scale_free', 'success', 'power_law_range']
    for col in req_cols:
        if col not in df.columns:
            raise ValueError(f"Missing required column: {col}")
            
    # If arm is present, we'll keep it. If not, add a dummy
    if 'arm' not in df.columns:
        df['arm'] = 'All'
    elif combine_arms:
        df['arm'] = 'Combined'
        
    df['subject_id'] = df['subject_id'].astype(str)
    
    # Clean boolean columns
    df['is_scale_free'] = df['is_scale_free'].astype(bool)
    df['success'] = df['success'].astype(bool)
    
    # Aggregate per subject, timepoint, arm
    aggregated = []
    
    for (subj, tp, arm), group in df.groupby(['subject_id', 'timepoint_number', 'arm']):
        total_blocks = len(group)
        if total_blocks == 0:
            continue
            
        pct_sf = (group['is_scale_free'].sum() / total_blocks) * 100
        
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
            'total_blocks': total_blocks,
            'successful_blocks': len(success_group)
        })
        
    df_agg = pd.DataFrame(aggregated)
    return df_agg

def load_demographics(demo_path, color_col, timepoint=None):
    print(f"Loading demographic data from {demo_path}...")
    df = pd.read_csv(demo_path)
    
    id_col = 'SubIDName' if 'SubIDName' in df.columns else df.columns[0]
    df['subject_id'] = df[id_col].astype(str).str.strip()
    
    if timepoint is not None and 'TimePoint' in df.columns:
        df = df[pd.to_numeric(df['TimePoint'], errors='coerce') == timepoint]
        
    if color_col not in df.columns:
        print(f"Warning: Column '{color_col}' not found in demographic data.")
        return pd.DataFrame({'subject_id': []})
        
    df[color_col] = pd.to_numeric(df[color_col], errors='coerce')
    return df[['subject_id', color_col]].dropna(subset=[color_col]).drop_duplicates(subset=['subject_id'])

def plot_longitudinal_metric(df_agg, metric_col, ylabel, output_dir, filename_prefix, color_col=None):
    """
    Plots a specific metric across timepoints.
    Creates a combined plot (lines per subject) and a boxplot summary.
    """
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
            
        fig, axes = plt.subplots(1, 2, figsize=(16, 6))
        fig.suptitle(f'Longitudinal Progression: {ylabel} ({arm})', fontsize=16, fontweight='bold')
        
        use_cmap = False
        if color_col and color_col in arm_data.columns:
            valid_color_data = arm_data[color_col].dropna()
            if not valid_color_data.empty:
                use_cmap = True
                norm = mcolors.Normalize(vmin=valid_color_data.min(), vmax=valid_color_data.max())
                cmap = cm.viridis
                sm = cm.ScalarMappable(cmap=cmap, norm=norm)
                sm.set_array([])
                
        # 1. Trajectory plot (one line per subject) + Mean trend
        ax1 = axes[0]
        for subj in arm_data['subject_id'].unique():
            subj_data = arm_data[arm_data['subject_id'] == subj]
            
            line_color = 'gray'
            if use_cmap:
                val = subj_data[color_col].iloc[0]
                if pd.notna(val):
                    line_color = cmap(norm(val))
                    
            ax1.plot(
                subj_data['timepoint_number'], 
                subj_data[metric_col], 
                marker='o', 
                color=line_color, 
                alpha=0.6 if use_cmap else 0.4
            )
            
        # Add mean trend
        mean_data = arm_data.groupby('timepoint_number')[metric_col].mean().reset_index()
        ax1.plot(
            mean_data['timepoint_number'], 
            mean_data[metric_col], 
            color='red', 
            linewidth=3, 
            marker='D', 
            markersize=8, 
            label='Mean Trend'
        )
        
        ax1.set_xlabel('Timepoint', fontsize=12)
        ax1.set_ylabel(ylabel, fontsize=12)
        ax1.set_title('Individual Subject Trajectories', fontsize=14)
        ax1.grid(True, alpha=0.3)
        ax1.set_xticks(timepoints)
        ax1.legend()
        
        if use_cmap:
            cbar = fig.colorbar(sm, ax=ax1, fraction=0.05, pad=0.04)
            cbar.set_label(color_col, fontsize=12)
        
        # 2. Boxplot / Swarmplot summary
        ax2 = axes[1]
        sns.boxplot(
            data=arm_data,
            x='timepoint_number',
            y=metric_col,
            ax=ax2,
            color='lightblue',
            showfliers=False
        )
        sns.stripplot(
            data=arm_data,
            x='timepoint_number',
            y=metric_col,
            ax=ax2,
            color='black',
            alpha=0.6,
            size=4,
            jitter=True
        )
        ax2.set_xlabel('Timepoint', fontsize=12)
        ax2.set_ylabel(ylabel, fontsize=12)
        ax2.set_title('Distribution Summary', fontsize=14)
        ax2.grid(True, alpha=0.3)
        
        plt.tight_layout()
        out_name = f"{filename_prefix}_{arm}.pdf"
        plt.savefig(os.path.join(output_dir, out_name), dpi=300, bbox_inches='tight')
        plt.close(fig)
        print(f"Saved {out_name}")

def main():
    parser = argparse.ArgumentParser(description="Generate Longitudinal Plots across timepoints")
    parser.add_argument('--csv_path', type=str, required=True, help="Path to master_scale_free_results.csv")
    parser.add_argument('--output_dir', type=str, default='../output/longitudinal', help="Directory to save the plots")
    parser.add_argument('--demographic_csv', type=str, default=None, help="Optional: Path to DemographicClinicalData.csv for coloring")
    parser.add_argument('--color_col', type=str, default=None, help="Optional: Demographic column to color individual lines by")
    parser.add_argument('--color_timepoint', type=int, default=None, help="Optional: Specific TimePoint in demographic data to extract the color value from")
    parser.add_argument('--combine_arms', action='store_true', help="Combine left and right arm data into a single unit per subject")
    
    args = parser.parse_args()
    
    os.makedirs(args.output_dir, exist_ok=True)
    
    try:
        df_agg = load_and_aggregate(args.csv_path, combine_arms=args.combine_arms)
    except Exception as e:
        print(f"Error processing data: {e}")
        return
        
    if args.demographic_csv and args.color_col:
        df_demo = load_demographics(args.demographic_csv, args.color_col, args.color_timepoint)
        if not df_demo.empty:
            df_agg = pd.merge(df_agg, df_demo, on='subject_id', how='left')
        
    csv_out = os.path.join(args.output_dir, 'longitudinal_aggregated_metrics.csv')
    df_agg.to_csv(csv_out, index=False)
    print(f"Saved aggregated metrics to {csv_out}")
        
    metrics = [
        ('mean_range', 'Mean Power-Law Range (decades)', 'longitudinal_mean_range'),
        ('median_range', 'Median Power-Law Range (decades)', 'longitudinal_median_range'),
        ('pct_scale_free', 'Percent Scale-Free Blocks (%)', 'longitudinal_pct_scale_free')
    ]
    
    print(f"Generating plots in {args.output_dir}...")
    for col, ylabel, prefix in metrics:
        plot_longitudinal_metric(df_agg, col, ylabel, args.output_dir, prefix, args.color_col)
        
    print("Longitudinal analysis complete.")

if __name__ == '__main__':
    main()