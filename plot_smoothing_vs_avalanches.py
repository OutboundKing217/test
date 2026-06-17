#!/usr/bin/env python3
"""
Plot Smoothing Samples vs Avalanche Counts

This script processes output folders named like "1-healthy", "1-stroke",
"30-healthy", "30-stroke" and generates a graph showing the total 
avalanche count per subject across different smoothing values.
"""

import os
import re
import argparse
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sns

#matplotlib.rcParams["font.family"] = "Arial"
matplotlib.rcParams["pdf.fonttype"] = 42      # Ensures text is saved as real characters
matplotlib.rcParams["ps.fonttype"] = 42
matplotlib.rcParams["svg.fonttype"] = "none"  # Keep SVG text selectable

def main():
    parser = argparse.ArgumentParser(description="Plot smoothing vs avalanche counts.")
    parser.add_argument("--input_dir", type=str, required=True, help="Directory containing the output folders (e.g., 1-healthy, 1-stroke, etc.)")
    parser.add_argument("--output_file", type=str, default="smoothing_vs_avalanches.pdf", help="Path to save the generated plot")
    args = parser.parse_args()

    if not os.path.exists(args.input_dir):
        print(f"Error: Input directory '{args.input_dir}' does not exist.")
        return

    data = []
    # Match pattern: optionally negative numbers and decimals followed by a dash and then healthy/stroke
    folder_pattern = re.compile(r'^(-?\d*\.?\d+)-(healthy|stroke)$', re.IGNORECASE)

    for folder_name in os.listdir(args.input_dir):
        folder_path = os.path.join(args.input_dir, folder_name)
        if not os.path.isdir(folder_path):
            continue

        match = folder_pattern.match(folder_name)
        if match:
            #denominator = int(match.group(1))
            
            # Convert the denominator back into actual window samples (assuming 30Hz)
            #smoothing_val = max(1, int((1 / denominator) * 30))
            smoothing_val = float(match.group(1))
            
            group = match.group(2).lower()

            csv_path = os.path.join(folder_path, 'master_scale_free_results.csv')
            if not os.path.exists(csv_path):
                print(f"Warning: Could not find 'master_scale_free_results.csv' in {folder_path}")
                continue

            try:
                df = pd.read_csv(csv_path)
                if 'subject_id' not in df.columns or 'n_events' not in df.columns:
                    print(f"Warning: Required columns missing in {csv_path}")
                    continue

                # Sum events across all hours and arms for each subject
                subject_totals = df.groupby('subject_id')['n_events'].sum().reset_index()

                for _, row in subject_totals.iterrows():
                    data.append({
                        'smoothing_samples': smoothing_val,
                        'group': group.capitalize(),
                        'subject_id': row['subject_id'],
                        'total_avalanches': row['n_events']
                    })

            except Exception as e:
                print(f"Error processing {csv_path}: {e}")

    if not data:
        print("No valid data found to plot. Check your input directory and folder names.")
        return

    df_plot = pd.DataFrame(data)

    # Plotting setup
    plt.figure(figsize=(12, 8))
    sns.set_style("whitegrid")

    # Colors for the groups
    palette = {'Healthy': 'blue', 'Stroke': 'red'}

    # Calculate a dynamic offset based on the x-axis step size to prevent overlap
    df_plot['smoothing_samples_offset'] = df_plot['smoothing_samples'].astype(float)
    x_vals = sorted(df_plot['smoothing_samples'].unique())
    if len(x_vals) > 1:
        offset = (x_vals[1] - x_vals[0]) * 0.1
    else:
        offset = 0.1
        
    df_plot.loc[df_plot['group'] == 'Healthy', 'smoothing_samples_offset'] -= offset
    df_plot.loc[df_plot['group'] == 'Stroke', 'smoothing_samples_offset'] += offset

    # Plot individual subjects as points
    sns.scatterplot(
        data=df_plot,
        x='smoothing_samples_offset',
        y='total_avalanches',
        hue='group',
        palette=palette,
        alpha=0.15,
        s=15
    )

    # Plot trend lines (mean) with lower opacity, no error bars
    sns.lineplot(
        data=df_plot,
        x='smoothing_samples',
        y='total_avalanches',
        hue='group',
        palette=palette,
        linewidth=2,
        alpha=0.4,
        errorbar=None,
        legend=False
    )

    # Plot error bars (95% CI) fully opaque, hiding the duplicate main line
    sns.lineplot(
        data=df_plot,
        x='smoothing_samples',
        y='total_avalanches',
        hue='group',
        palette=palette,
        errorbar=('ci', 95),
        err_style='bars',
        alpha=0.6,
        err_kws={'capsize': 5, 'capthick': 1, 'elinewidth': 1, 'ecolor': 'black'},
        linestyle='none',
        legend=False  # Hide legend here to avoid duplicates
    )

    plt.title('Effect of Percentile Threshold on Total Avalanche Count', fontsize=16, fontweight='bold')
    plt.xlabel('Threshold percentile', fontsize=14)
    plt.ylabel('Total Avalanche Count (per subject)', fontsize=14)
    
    # Ensure x-ticks show all unique smoothing values
    smoothing_values = sorted(df_plot['smoothing_samples'].unique())
    plt.xticks(smoothing_values)
    
    plt.legend(title='Cohort', title_fontsize='13', fontsize='12')
    plt.tight_layout()

    # Save the plot
    plt.savefig(args.output_file, dpi=300, bbox_inches='tight')
    print(f"Successfully generated plot: {args.output_file}")

if __name__ == '__main__':
    main()
