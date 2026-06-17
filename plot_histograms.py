"""
Histogram Plotting Module

Reads the master scale-free results CSV and generates distribution histograms
of the Power-Law Range. It creates a separate plot for each timepoint, 
comparing the LUE and RUE distributions.
"""

import os
import argparse
import pandas as pd
import numpy as np
import matplotlib
import matplotlib.pyplot as plt
import seaborn as sns

#matplotlib.rcParams["font.family"] = "Arial"
matplotlib.rcParams["pdf.fonttype"] = 42      # Ensures text is saved as real characters
matplotlib.rcParams["ps.fonttype"] = 42
matplotlib.rcParams["svg.fonttype"] = "none"  # Keep SVG text selectable

def generate_range_histograms(csv_path: str, output_dir: str):
    """
    Generates and saves power-law range histograms from the results CSV.
    Calculates the mean range per subject to prevent subjects with more 
    active hours from artificially skewing the distribution.

    Parameters:
    -----------
    csv_path : str
        Path to the 'master_scale_free_results.csv' file.
    output_dir : str
        Directory to save the generated histogram PNGs.
    """
    # Load the results
    try:
        df = pd.read_csv(csv_path)
    except Exception as e:
        print(f"Error loading CSV: {e}")
        return

    # Ensure output directory exists
    os.makedirs(output_dir, exist_ok=True)

    # 1. Filter out failed/quiescent hours and missing data
    df_success = df[(df['success'] == True) & (df['power_law_range'].notna())].copy()

    if df_success.empty:
        print("No successful hours found in the dataset to plot.")
        return

    # 2. Calculate the mean power-law range per subject, per arm, per timepoint
    # This ensures each subject contributes equally to the distribution
    df_subject_means = df_success.groupby(['timepoint_number', 'arm', 'subject_id'])['power_law_range'].mean().reset_index()

    # 3. Generate a plot for each unique timepoint
    timepoints = sorted(df_subject_means['timepoint_number'].unique())

    # Set styling for publication-ready plots
    sns.set_style("whitegrid")
    
    for tp in timepoints:
        df_tp = df_subject_means[df_subject_means['timepoint_number'] == tp]
        
        # Create a figure with 1 row, 2 columns (LUE and RUE side-by-side)
        fig, axes = plt.subplots(1, 2, figsize=(16, 6), sharey=True, sharex=True)
        fig.suptitle(f'Mean Power-Law Range Distribution - Timepoint {tp}', fontsize=16, fontweight='bold', y=1.05)
        
        arms = ['LUE', 'RUE']
        colors = {'LUE': 'blue', 'RUE': 'orange'}
        
        for idx, arm in enumerate(arms):
            ax = axes[idx]
            df_arm = df_tp[df_tp['arm'] == arm]
            
            if not df_arm.empty:
                # Plot histogram with a Kernel Density Estimate (KDE) curve overlaid
                sns.histplot(
                    df_arm['power_law_range'], 
                    bins=15, 
                    kde=True, 
                    color=colors[arm], 
                    edgecolor='black', 
                    alpha=0.6,
                    ax=ax
                )
                
                # Calculate and plot mean/median lines
                mean_val = df_arm['power_law_range'].mean()
                median_val = df_arm['power_law_range'].median()
                
                ax.axvline(mean_val, color='green', linestyle='dashed', linewidth=2, label=f'Mean: {mean_val:.2f}')
                ax.axvline(median_val, color='red', linestyle='dotted', linewidth=2, label=f'Median: {median_val:.2f}')
                
                ax.set_title(f'{arm} (n={len(df_arm)} subjects)', fontsize=14)
                ax.set_xlabel('Mean Power-Law Range (Decades)', fontsize=12)
                ax.legend(loc='upper right')
            else:
                ax.set_title(f'{arm} (No Data)', fontsize=14)
                ax.text(0.5, 0.5, 'No successful analyses', horizontalalignment='center', verticalalignment='center', transform=ax.transAxes)

        axes[0].set_ylabel('Number of Subjects', fontsize=12)
        
        plt.tight_layout()
        
        # Save the plot
        out_path = os.path.join(output_dir, f'histogram_power_law_range_TP{tp}.pdf')
        plt.savefig(out_path, dpi=300, bbox_inches='tight', format='pdf')
        plt.close(fig)
        
        print(f"Saved histogram for Timepoint {tp} to {out_path}")

def main():
    parser = argparse.ArgumentParser(description="Generate Power-Law Range Histograms")
    parser.add_argument('--csv_path', type=str, required=True, help="Path to master_scale_free_results.csv")
    parser.add_argument('--output_dir', type=str, default='../output/histograms', help="Directory to save the plots")
    args = parser.parse_args()

    print("="*60)
    print("Generating Power-Law Range Histograms...")
    print("="*60)
    
    generate_range_histograms(args.csv_path, args.output_dir)
    
    print("Done!")

if __name__ == '__main__':
    main()