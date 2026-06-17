#!/usr/bin/env python3
"""
Compare Scale-Free Analysis Results

This script compares two master_scale_free_results.csv files (e.g., from different
parameter settings or different cohorts).

Outputs:
1. Violin plots of Mean and Median Power-Law Ranges (per subject) with T-test.
2. Beeswarm plot of Percent Scale-Free Blocks (per subject) with Mann-Whitney U test.

Usage:
    python compare_scale_free_runs.py --file1 path/to/results1.csv --file2 path/to/results2.csv --label1 "Run A" --label2 "Run B"
"""

import os
import sys
import argparse
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')  # Non-interactive backend
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats

#matplotlib.rcParams["font.family"] = "Arial"
matplotlib.rcParams["pdf.fonttype"] = 42      # Ensures text is saved as real characters
matplotlib.rcParams["ps.fonttype"] = 42
matplotlib.rcParams["svg.fonttype"] = "none"  # Keep SVG text selectable

def load_and_aggregate(file_path, label):
    """Load results CSV and aggregate metrics by subject."""
    if not os.path.exists(file_path):
        print(f"Error: File not found: {file_path}")
        return None, None

    df = pd.read_csv(file_path)
    
    # Ensure subject_id is string
    if 'subject_id' in df.columns:
        df['subject_id'] = df['subject_id'].astype(str)
    
    # Filter for successful analyses for Range metrics
    if 'is_scale_free' in df.columns:
        df_success = df[(df['success'] == True) & (df['is_scale_free'] == True)].copy()
    else:
        df_success = df[df['success'] == True].copy()
    
    # Aggregate by subject
    # We calculate metrics per subject (averaging across arms/hours)
    subject_metrics = []
    
    unique_subjects = df['subject_id'].unique()
    
    for subj in unique_subjects:
        # All rows for this subject (for % scale free)
        subj_all = df[df['subject_id'] == subj]
        
        # Successful rows (for range stats)
        subj_success = df_success[df_success['subject_id'] == subj]
        
        if len(subj_all) == 0:
            continue
            
        # Calculate % Scale Free (blocks with GOF >= 0.8)
        # Assuming 'is_scale_free' column exists
        if 'is_scale_free' in subj_all.columns:
            pct_scale_free = (subj_all['is_scale_free'].sum() / len(subj_all)) * 100
        else:
            # Fallback if column missing
            pct_scale_free = np.nan
            
        # Calculate Range metrics
        if len(subj_success) > 0:
            mean_range = subj_success['power_law_range'].mean()
            median_range = subj_success['power_law_range'].median()
        else:
            mean_range = np.nan
            median_range = np.nan
            
        if 'timepoint_number' in subj_all.columns:
            tps = tuple(sorted(subj_all['timepoint_number'].dropna().unique()))
        else:
            tps = tuple()
            
        subject_metrics.append({
            'subject_id': subj,
            'dataset': label,
            'mean_range': mean_range,
            'median_range': median_range,
            'pct_scale_free': pct_scale_free,
            'timepoints': tps
        })
        
    return df, pd.DataFrame(subject_metrics)

def plot_violins(df_combined, output_dir):
    """Create violin plots for Mean and Median Range with T-test."""
    metrics = [
        ('mean_range', 'Mean Power-Law Range (decades)'),
        ('median_range', 'Median Power-Law Range (decades)')
    ]
    
    fig, axes = plt.subplots(1, 2, figsize=(14, 6))
    
    datasets = df_combined['dataset'].unique()
    if len(datasets) != 2:
        print("Error: Need exactly 2 datasets for comparison.")
        return

    d1_name, d2_name = datasets[0], datasets[1]
    custom_palette = {d1_name: '#7a5195', d2_name: '#ef5675'}
    
    for idx, (col, ylabel) in enumerate(metrics):
        ax = axes[idx]
        
        # Drop NaNs for plotting and stats
        data = df_combined.dropna(subset=[col])
        
        # Plot
        sns.violinplot(data=data, x='dataset', y=col, hue='dataset', legend=False, ax=ax, palette=custom_palette, inner="quartile")
        sns.stripplot(data=data, x='dataset', y=col, ax=ax, color='black', alpha=0.3, size=4)
        
        ax.set_ylabel(ylabel)
        ax.set_xlabel('')
        ax.set_title(ylabel)
        
        # T-test
        group1 = data[data['dataset'] == d1_name][col]
        group2 = data[data['dataset'] == d2_name][col]
        
        if len(group1) > 1 and len(group2) > 1:
            t_stat, p_val = stats.ttest_ind(group1, group2, nan_policy='omit')
            
            # Add stats to plot
            y_max = data[col].max()
            y_min = data[col].min()
            y_range = y_max - y_min
            
            # Draw bar
            h = y_range * 0.05
            y_line = y_max + h
            ax.plot([0, 0, 1, 1], [y_line, y_line+h, y_line+h, y_line], lw=1.5, c='k')
            ax.text(0.5, y_line+h, f'T-test p={p_val:.4f}', ha='center', va='bottom', color='k')
            ax.set_ylim(y_min - h, y_line + 4*h)

    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'comparison_violins_ranges.pdf'), dpi=300)
    plt.close()
    print(f"Saved violin plots to {os.path.join(output_dir, 'comparison_violins_ranges.pdf')}")

def plot_beeswarm(df_combined, output_dir):
    """Create beeswarm plot for Percent Scale-Free with Mann-Whitney U test."""
    col = 'pct_scale_free'
    ylabel = 'Percent Scale-Free Blocks (%)'
    
    plt.figure(figsize=(8, 6))
    
    datasets = df_combined['dataset'].unique()
    d1_name, d2_name = datasets[0], datasets[1]
    custom_palette = {d1_name: '#7a5195', d2_name: '#ef5675'}
    
    # Drop NaNs
    data = df_combined.dropna(subset=[col])
    
    # Plot
    sns.swarmplot(data=data, x='dataset', y=col, hue='dataset', legend=False, palette=custom_palette, size=6)
    # Add boxplot behind for context
    sns.boxplot(data=data, x='dataset', y=col, showfliers=False, 
                boxprops={'facecolor':'none', 'edgecolor':'gray'},
                whiskerprops={'color':'gray'},
                capprops={'color':'gray'})
    
    plt.ylabel(ylabel)
    plt.xlabel('')
    plt.title('Percent of Blocks Obeying Power Laws per Subject')
    
    # Mann-Whitney U test
    group1 = data[data['dataset'] == d1_name][col]
    group2 = data[data['dataset'] == d2_name][col]
    
    if len(group1) > 1 and len(group2) > 1:
        u_stat, p_val = stats.mannwhitneyu(group1, group2)
        
        # Add stats
        ax = plt.gca()
        y_max = data[col].max()
        y_min = data[col].min()
        y_range = y_max - y_min if y_max != y_min else 10
        
        h = y_range * 0.05
        y_line = y_max + h
        plt.plot([0, 0, 1, 1], [y_line, y_line+h, y_line+h, y_line], lw=1.5, c='k')
        plt.text(0.5, y_line+h, f'Mann-Whitney p={p_val:.4f}', ha='center', va='bottom', color='k')
        plt.ylim(y_min - h, y_line + 4*h)
        
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'comparison_beeswarm_percent.pdf'), dpi=300)
    plt.close()
    print(f"Saved beeswarm plot to {os.path.join(output_dir, 'comparison_beeswarm_percent.pdf')}")

def plot_paired_lines(df_combined, output_dir, min_tp_diff=1):
    """Create paired line plots for Mean and Median Range connecting the same subjects."""
    metrics = [
        ('mean_range', 'Mean Power-Law Range (decades)'),
        ('median_range', 'Median Power-Law Range (decades)')
    ]
    
    fig, axes = plt.subplots(1, 2, figsize=(14, 6))
    
    datasets = df_combined['dataset'].unique()
    if len(datasets) != 2:
        print("Error: Need exactly 2 datasets for paired comparison.")
        return

    d1_name, d2_name = datasets[0], datasets[1]
    custom_palette = {d1_name: '#7a5195', d2_name: '#ef5675'}
    
    # Identify and exclude subjects with overlapping timepoints
    df_d1 = df_combined[df_combined['dataset'] == d1_name]
    df_d2 = df_combined[df_combined['dataset'] == d2_name]
    
    if 'timepoints' in df_combined.columns:
        merged_tps = pd.merge(df_d1[['subject_id', 'timepoints']], df_d2[['subject_id', 'timepoints']], on='subject_id')
        excluded_subjects = []
        for _, row in merged_tps.iterrows():
            tps_x = row['timepoints_x']
            tps_y = row['timepoints_y']
            if tps_x and tps_y:
                min_diff = min(abs(x - y) for x in tps_x for y in tps_y)
                if min_diff < min_tp_diff:
                    excluded_subjects.append(row['subject_id'])
                
        if excluded_subjects:
            print(f"Excluding subjects from paired plots due to timepoint difference < {min_tp_diff}: {excluded_subjects}")
            df_plot = df_combined[~df_combined['subject_id'].isin(excluded_subjects)]
        else:
            df_plot = df_combined
    else:
        df_plot = df_combined
        
    for idx, (col, ylabel) in enumerate(metrics):
        ax = axes[idx]
        
        # Pivot data to pair by subject_id
        df_pivot = df_plot.pivot(index='subject_id', columns='dataset', values=col).dropna()
        
        if df_pivot.empty:
            ax.set_title(f"{ylabel}\n(No paired data)")
            continue

        # Plot lines connecting the same subject
        for i, row in df_pivot.iterrows():
            ax.plot([0, 1], [row[d1_name], row[d2_name]], color='gray', alpha=0.5)
            
        # Plot individual points grouped by dataset to maintain color coding
        paired_data = df_plot[df_plot['subject_id'].isin(df_pivot.index)]
        sns.stripplot(data=paired_data, x='dataset', y=col, ax=ax, palette=custom_palette, size=8, alpha=0.8, zorder=5)

        ax.set_ylabel(ylabel)
        ax.set_xlabel('')
        ax.set_title(ylabel)
        
        # Paired T-test
        group1 = df_pivot[d1_name]
        group2 = df_pivot[d2_name]
        
        if len(group1) > 1:
            t_stat, p_val = stats.ttest_rel(group1, group2)
            
            # Add stats to plot
            y_max = df_pivot.max().max()
            y_min = df_pivot.min().min()
            y_range = y_max - y_min if y_max != y_min else 1
            
            h = y_range * 0.05
            y_line = y_max + h
            ax.plot([0, 0, 1, 1], [y_line, y_line+h, y_line+h, y_line], lw=1.5, c='k')
            ax.text(0.5, y_line+h, f'Paired T-test p={p_val:.4f}', ha='center', va='bottom', color='k')
            ax.set_ylim(y_min - h, y_line + 4*h)

    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'comparison_paired_lines_ranges.pdf'), dpi=300)
    plt.close()
    print(f"Saved paired line plots to {os.path.join(output_dir, 'comparison_paired_lines_ranges.pdf')}")

def main():
    parser = argparse.ArgumentParser(description="Compare two scale-free analysis result files.")
    parser.add_argument('--file1', type=str, required=True, help='Path to first results CSV')
    parser.add_argument('--file2', type=str, required=True, help='Path to second results CSV')
    parser.add_argument('--label1', type=str, default='Dataset 1', help='Label for first dataset')
    parser.add_argument('--label2', type=str, default='Dataset 2', help='Label for second dataset')
    parser.add_argument('--output', type=str, default='comparison_output', help='Output directory')
    parser.add_argument('--min_tp_diff', type=int, default=1, help='Minimum timepoint difference required between datasets for paired plots (default: 1)')
    
    args = parser.parse_args()
    
    os.makedirs(args.output, exist_ok=True)
    
    print(f"Comparing:\n  1. {args.label1} ({args.file1})\n  2. {args.label2} ({args.file2})")
    
    # Load and aggregate
    _, df1_agg = load_and_aggregate(args.file1, args.label1)
    _, df2_agg = load_and_aggregate(args.file2, args.label2)
    
    if df1_agg is None or df2_agg is None:
        print("Failed to load data.")
        return
        
    # Combine for plotting
    df_combined = pd.concat([df1_agg, df2_agg], ignore_index=True)
    
    print(f"Loaded {len(df1_agg)} subjects from {args.label1}")
    print(f"Loaded {len(df2_agg)} subjects from {args.label2}")
    
    # 1. Violin Plots (Mean/Median Range) + T-test
    plot_violins(df_combined, args.output)
    
    # 2. Beeswarm Plot (% Scale Free) + Mann-Whitney
    plot_beeswarm(df_combined, args.output)
    
    # 3. Paired Line Graph (Mean/Median Range) connecting identical subjects
    plot_paired_lines(df_combined, args.output, args.min_tp_diff)
    
    print("\nComparison complete.")

if __name__ == "__main__":
    main()
