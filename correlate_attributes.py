import argparse
import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats

def load_and_aggregate_results(results_path, timepoint=None):
    """Load scale-free results and aggregate mean/median range per subject."""
    df = pd.read_csv(results_path)
    
    # Filter by timepoint_number if specified
    if timepoint is not None and 'timepoint_number' in df.columns:
        df = df[pd.to_numeric(df['timepoint_number'], errors='coerce') == timepoint]

    # Filter for successful fits if the column exists
    if 'success' in df.columns:
        df = df[df['success'] == True]
    
    # Drop rows without a power_law_range
    df = df.dropna(subset=['power_law_range'])
    
    # Aggregate mean and median per subject
    agg_df = df.groupby('subject_id').agg(
        mean_power_law_range=('power_law_range', 'mean'),
        median_power_law_range=('power_law_range', 'median')
    ).reset_index()
    
    # Ensure string type for merging
    agg_df['subject_id'] = agg_df['subject_id'].astype(str).str.strip()
    return agg_df

def load_attributes(attributes_path, timepoint=None):
    """Load patient attributes CSV."""
    df = pd.read_csv(attributes_path)
    
    # Filter by TimePoint if specified
    if timepoint is not None and 'TimePoint' in df.columns:
        df = df[pd.to_numeric(df['TimePoint'], errors='coerce') == timepoint]
        
    # Assume the ID column is 'SubIDName' based on the provided header, or fallback to the first column
    id_col = 'SubIDName' if 'SubIDName' in df.columns else df.columns[0]
    df['subject_id'] = df[id_col].astype(str).str.strip()
    return df

def plot_correlation(merged_df, col, metric, output_dir, timepoint=None):
    """Plot scatterplot with regression line and Pearson correlation."""
    # Drop rows with NaN in the column or metric
    plot_df = merged_df.dropna(subset=[col, metric]).copy()
    
    # Ensure the target column is numeric
    plot_df[col] = pd.to_numeric(plot_df[col], errors='coerce')
    plot_df = plot_df.dropna(subset=[col])
    
    if plot_df.empty or len(plot_df) < 3:
        print(f"Not enough valid numeric data to plot {metric} vs {col}")
        return
        
    plt.figure(figsize=(8, 6))
    sns.regplot(data=plot_df, x=col, y=metric, scatter_kws={'alpha': 0.7, 'edgecolor': 'k'})
    
    # Calculate Pearson correlation
    r, p = stats.pearsonr(plot_df[col], plot_df[metric])
    
    tp_str = f" (TP {timepoint})" if timepoint is not None else ""
    plt.title(f'{metric.replace("_", " ").title()} vs {col}{tp_str}\nPearson r = {r:.3f}, p = {p:.4f}')
    plt.xlabel(col)
    plt.ylabel(metric.replace('_', ' ').title())
    plt.grid(True, linestyle='--', alpha=0.6)
    
    plt.tight_layout()
    tp_suffix = f"_TP{timepoint}" if timepoint is not None else ""
    filename = f"{metric}_vs_{col}{tp_suffix}.png"
    plt.savefig(os.path.join(output_dir, filename), dpi=300, bbox_inches='tight')
    plt.close()
    print(f"Saved plot: {os.path.join(output_dir, filename)}")

def main():
    parser = argparse.ArgumentParser(description="Correlate Power Law Range with Patient Attributes")
    parser.add_argument('--results_csv', required=True, help="Path to master_scale_free_results.csv")
    parser.add_argument('--attributes_csv', required=True, help="Path to patient attributes CSV")
    parser.add_argument('--columns', required=True, help="Comma-separated list of attribute columns to plot (e.g., AgeVal,EduYrCt)")
    parser.add_argument('--output_dir', default="output_correlations", help="Directory to save plots")
    parser.add_argument('--timepoint', type=int, default=None, help="Optional: Filter data by a specific timepoint")
    
    args = parser.parse_args()
    
    os.makedirs(args.output_dir, exist_ok=True)
    
    results_df = load_and_aggregate_results(args.results_csv, args.timepoint)
    attr_df = load_attributes(args.attributes_csv, args.timepoint)
    
    merged_df = pd.merge(results_df, attr_df, on='subject_id', how='inner')
    
    print(f"Merged data has {len(merged_df)} subjects.")
    if len(merged_df) == 0:
        print("Warning: No matching subjects found between results and attributes.")
        return
        
    cols_to_plot = [c.strip() for c in args.columns.split(',')]
    
    for col in cols_to_plot:
        if col not in merged_df.columns:
            print(f"Column '{col}' not found in attributes CSV. Skipping.")
            continue
            
        plot_correlation(merged_df, col, 'mean_power_law_range', args.output_dir, args.timepoint)
        plot_correlation(merged_df, col, 'median_power_law_range', args.output_dir, args.timepoint)

if __name__ == '__main__':
    main()