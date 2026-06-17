#!/usr/bin/env python3
"""
Compare Scale-Free Analysis Results (3 Files)

This script compares three master_scale_free_results.csv files.

Outputs:
1. Violin plots of Mean and Median Power-Law Ranges (per subject) with ANOVA.
2. Beeswarm plot of Percent Scale-Free Blocks (per subject) with Kruskal-Wallis test.
3. Paired line plots of Mean and Median Power-Law Ranges with Friedman test.

Usage:
    python compare_scale_free_3.py --file1 path/to/results1.csv --file2 path/to/results2.csv --file3 path/to/results3.csv
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
    subject_metrics = []
    unique_subjects = df['subject_id'].unique()
    
    for subj in unique_subjects:
        subj_all = df[df['subject_id'] == subj]
        subj_success = df_success[df_success['subject_id'] == subj]
        
        if len(subj_all) == 0:
            continue
            
        if 'is_scale_free' in subj_all.columns:
            pct_scale_free = (subj_all['is_scale_free'].sum() / len(subj_all)) * 100
        else:
            pct_scale_free = np.nan
            
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
            
        if 'timepoint_number' in subj_all.columns and not subj_all['timepoint_number'].dropna().empty:
            tp_val = subj_all['timepoint_number'].dropna().median()
        else:
            tp_val = np.nan
            
        subject_metrics.append({
            'subject_id': subj,
            'dataset': label,
            'mean_range': mean_range,
            'median_range': median_range,
            'pct_scale_free': pct_scale_free,
            'timepoints': tps,
            'timepoint_val': tp_val
        })
        
    return df, pd.DataFrame(subject_metrics)

def plot_violins(df_combined, output_dir):
    """Create violin plots for Mean and Median Range with ANOVA."""
    metrics = [
        ('mean_range', 'Mean Power-Law Range (decades)'),
        ('median_range', 'Median Power-Law Range (decades)')
    ]
    
    fig, axes = plt.subplots(1, 2, figsize=(16, 6))
    
    datasets = df_combined['dataset'].unique()
    d1_name, d2_name, d3_name = datasets[0], datasets[1], datasets[2]
    custom_palette = {d1_name: '#7C7C7C', d2_name: '#7a5195', d3_name: '#ef5675'}
    
    for idx, (col, ylabel) in enumerate(metrics):
        ax = axes[idx]
        data = df_combined.dropna(subset=[col])
        
        sns.violinplot(data=data, x='dataset', y=col, hue='dataset', legend=False, ax=ax, palette=custom_palette, inner="quartile")
        sns.stripplot(data=data, x='dataset', y=col, ax=ax, color='black', alpha=0.3, size=4, jitter=True)
        
        ax.set_ylabel(ylabel)
        ax.set_xlabel('')
        ax.set_title(ylabel)
        
        group1 = data[data['dataset'] == d1_name][col]
        group2 = data[data['dataset'] == d2_name][col]
        group3 = data[data['dataset'] == d3_name][col]
        
        if len(group1) > 1 and len(group2) > 1 and len(group3) > 1:
            f_stat, p_val = stats.f_oneway(group1, group2, group3)
            
            y_max = data[col].max()
            y_min = data[col].min()
            y_range = y_max - y_min
            
            h = y_range * 0.05
            y_line = y_max + h
            ax.plot([0, 0, 2, 2], [y_line, y_line+h, y_line+h, y_line], lw=1.5, c='k')
            ax.text(1.0, y_line+h, f'ANOVA p={p_val:.4f}', ha='center', va='bottom', color='k')
            ax.set_ylim(y_min - h, y_line + 4*h)

    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'comparison_violins_ranges_3.pdf'), dpi=300)
    plt.close()
    print(f"Saved violin plots to {os.path.join(output_dir, 'comparison_violins_ranges_3.pdf')}")

def plot_beeswarm(df_combined, output_dir):
    """Create beeswarm plot for Percent Scale-Free with Kruskal-Wallis test."""
    col = 'pct_scale_free'
    ylabel = 'Percent Scale-Free Blocks (%)'
    
    plt.figure(figsize=(10, 6))
    
    datasets = df_combined['dataset'].unique()
    d1_name, d2_name, d3_name = datasets[0], datasets[1], datasets[2]
    custom_palette = {d1_name: '#7C7C7C', d2_name: '#7a5195', d3_name: '#ef5675'}
    
    data = df_combined.dropna(subset=[col])
    
    sns.swarmplot(data=data, x='dataset', y=col, hue='dataset', legend=False, palette=custom_palette, size=6)
    sns.boxplot(data=data, x='dataset', y=col, showfliers=False, 
                boxprops={'facecolor':'none', 'edgecolor':'gray'},
                whiskerprops={'color':'gray'},
                capprops={'color':'gray'})
    
    plt.ylabel(ylabel)
    plt.xlabel('')
    plt.title('Percent of Blocks Obeying Power Laws per Subject')
    
    group1 = data[data['dataset'] == d1_name][col]
    group2 = data[data['dataset'] == d2_name][col]
    group3 = data[data['dataset'] == d3_name][col]
    
    if len(group1) > 1 and len(group2) > 1 and len(group3) > 1:
        h_stat, p_val = stats.kruskal(group1, group2, group3)
        
        y_max = data[col].max()
        y_min = data[col].min()
        y_range = y_max - y_min if y_max != y_min else 10
        
        h = y_range * 0.05
        y_line = y_max + h
        plt.plot([0, 0, 2, 2], [y_line, y_line+h, y_line+h, y_line], lw=1.5, c='k')
        plt.text(1.0, y_line+h, f'Kruskal-Wallis p={p_val:.4f}', ha='center', va='bottom', color='k')
        plt.ylim(y_min - h, y_line + 4*h)
        
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'comparison_beeswarm_percent_3.pdf'), dpi=300)
    plt.close()
    print(f"Saved beeswarm plot to {os.path.join(output_dir, 'comparison_beeswarm_percent_3.pdf')}")

def plot_paired_lines(df_combined, output_dir, use_timepoints_as_x=False):
    """Create paired line plots for Mean and Median Range connecting the same subjects."""
    metrics = [
        ('mean_range', 'Mean Power-Law Range (decades)'),
        ('median_range', 'Median Power-Law Range (decades)')
    ]
    
    fig, axes = plt.subplots(1, 2, figsize=(16, 6))
    
    datasets = df_combined['dataset'].unique()
    d1_name, d2_name, d3_name = datasets[0], datasets[1], datasets[2]
    custom_palette = {d1_name: '#7C7C7C', d2_name: '#7a5195', d3_name: '#ef5675'}
        
    for idx, (col, ylabel) in enumerate(metrics):
        ax = axes[idx]
        
        # Pivot data to pair by subject_id
        df_pivot = df_combined.pivot(index='subject_id', columns='dataset', values=col).dropna()
        
        if df_pivot.empty:
            ax.set_title(f"{ylabel}\n(No paired data)")
            continue

        if use_timepoints_as_x:
            df_tp_pivot = df_combined.pivot(index='subject_id', columns='dataset', values='timepoint_val')

        for i, row in df_pivot.iterrows():
            if use_timepoints_as_x:
                tp_row = df_tp_pivot.loc[i]
                if pd.isna(tp_row[d1_name]) or pd.isna(tp_row[d2_name]) or pd.isna(tp_row[d3_name]):
                    ax.plot([0, 1, 2], [row[d1_name], row[d2_name], row[d3_name]], color='gray', alpha=0.5)
                else:
                    x_vals = [tp_row[d1_name], tp_row[d2_name], tp_row[d3_name]]
                    ax.plot(x_vals, [row[d1_name], row[d2_name], row[d3_name]], color='gray', alpha=0.5)
            else:
                ax.plot([0, 1, 2], [row[d1_name], row[d2_name], row[d3_name]], color='gray', alpha=0.5)
            
        if use_timepoints_as_x:
            for ds_name in [d1_name, d2_name, d3_name]:
                ds_data = df_combined[(df_combined['dataset'] == ds_name) & (df_combined['subject_id'].isin(df_pivot.index))]
                ax.scatter(ds_data['timepoint_val'], ds_data[col], color=custom_palette[ds_name], s=64, alpha=0.8, zorder=5, label=ds_name)
            
            handles, labels = ax.get_legend_handles_labels()
            by_label = dict(zip(labels, handles))
            ax.legend(by_label.values(), by_label.keys())
            ax.set_xlabel('Timepoint (Days)')
        else:
            paired_data = df_combined[df_combined['subject_id'].isin(df_pivot.index)]
            sns.stripplot(data=paired_data, x='dataset', y=col, ax=ax, palette=custom_palette, size=8, alpha=0.8, zorder=5)
            ax.set_xlabel('')

        ax.set_ylabel(ylabel)
        ax.set_title(ylabel)
        
        group1 = df_pivot[d1_name]
        group2 = df_pivot[d2_name]
        group3 = df_pivot[d3_name]
        
        if len(group1) > 1:
            stat, p_val = stats.friedmanchisquare(group1, group2, group3)
            
            y_max = df_pivot.max().max()
            y_min = df_pivot.min().min()
            y_range = y_max - y_min if y_max != y_min else 1
            
            h = y_range * 0.05
            y_line = y_max + h
            ax.plot([0, 0, 2, 2], [y_line, y_line+h, y_line+h, y_line], lw=1.5, c='k')
            ax.text(1.0, y_line+h, f'Friedman p={p_val:.4f}', ha='center', va='bottom', color='k')
            ax.set_ylim(y_min - h, y_line + 4*h)

    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'comparison_paired_lines_ranges_3.pdf'), dpi=300)
    plt.close()
    print(f"Saved paired line plots to {os.path.join(output_dir, 'comparison_paired_lines_ranges_3.pdf')}")

def main():
    parser = argparse.ArgumentParser(description="Compare three scale-free analysis result files.")
    parser.add_argument('--file1', type=str, required=True, help='Path to first results CSV')
    parser.add_argument('--file2', type=str, required=True, help='Path to second results CSV')
    parser.add_argument('--file3', type=str, required=True, help='Path to third results CSV')
    parser.add_argument('--label1', type=str, default='Dataset 1', help='Label for first dataset')
    parser.add_argument('--label2', type=str, default='Dataset 2', help='Label for second dataset')
    parser.add_argument('--label3', type=str, default='Dataset 3', help='Label for third dataset')
    parser.add_argument('--output', type=str, default='comparison_output', help='Output directory')
    parser.add_argument('--timepoints_as_days', action='store_true', help='Use timepoint_number (days) as the x-axis for paired plots')
    
    args = parser.parse_args()
    
    os.makedirs(args.output, exist_ok=True)
    
    print(f"Comparing:\n  1. {args.label1} ({args.file1})\n  2. {args.label2} ({args.file2})\n  3. {args.label3} ({args.file3})")
    
    _, df1_agg = load_and_aggregate(args.file1, args.label1)
    _, df2_agg = load_and_aggregate(args.file2, args.label2)
    _, df3_agg = load_and_aggregate(args.file3, args.label3)
    
    if df1_agg is None or df2_agg is None or df3_agg is None:
        print("Failed to load data.")
        return
        
    df_combined = pd.concat([df1_agg, df2_agg, df3_agg], ignore_index=True)
    
    print(f"Loaded {len(df1_agg)} subjects from {args.label1}")
    print(f"Loaded {len(df2_agg)} subjects from {args.label2}")
    print(f"Loaded {len(df3_agg)} subjects from {args.label3}")
    
    # 1. Violin Plots (Mean/Median Range) + ANOVA
    plot_violins(df_combined, args.output)
    
    # 2. Beeswarm Plot (% Scale Free) + Kruskal-Wallis
    plot_beeswarm(df_combined, args.output)
    
    # 3. Paired Line Graph (Mean/Median Range) connecting identical subjects + Friedman
    plot_paired_lines(df_combined, args.output, use_timepoints_as_x=args.timepoints_as_days)
    
    print("\nComparison complete.")

if __name__ == "__main__":
    main()
