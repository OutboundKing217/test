#!/usr/bin/env python3
"""
Concordance Plotting Module

Reads the master scale-free results CSV and generates left-right (LUE vs RUE)
concordance plots for various scale-free metrics.
"""

import os
import argparse
import pandas as pd
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.stats import pearsonr, f
from scipy import odr

#matplotlib.rcParams["font.family"] = "Arial"
matplotlib.rcParams["pdf.fonttype"] = 42      # Ensures text is saved as real characters
matplotlib.rcParams["ps.fonttype"] = 42
matplotlib.rcParams["svg.fonttype"] = "none"  # Keep SVG text selectable

def plot_block_concordance(df: pd.DataFrame, output_dir: str, file_prefix: str = ""):
    """
    Generates and saves LUE vs RUE concordance plots for every individual time block.
    """
    subj_col = 'subject_id' if 'subject_id' in df.columns else 'subject'
    
    # Identify the matching keys for merging LUE and RUE by time block
    merge_cols = [subj_col]
    if 'timepoint_number' in df.columns:
        merge_cols.append('timepoint_number')
    if 'hour_number' in df.columns:
        merge_cols.append('hour_number')
    elif 'hour' in df.columns:
        merge_cols.append('hour')

    df_lue = df[df['arm'] == 'LUE'].copy()
    df_rue = df[df['arm'] == 'RUE'].copy()
    
    df_blocks = pd.merge(df_lue, df_rue, on=merge_cols, suffixes=('_lue', '_rue'))
    
    if len(df_blocks) == 0:
        print("No matching time blocks found between LUE and RUE.")
        return

    fig, axes = plt.subplots(2, 2, figsize=(16, 16))
    
    def plot_metric(ax, col_x, col_y, xlabel, ylabel, title_prefix, line_min, line_max):
        valid_data = df_blocks.dropna(subset=[col_x, col_y])
        m, b = None, None
        if len(valid_data) > 1:
            try:
                r, p = pearsonr(valid_data[col_x], valid_data[col_y])
                
                x_val = valid_data[col_x].values
                y_val = valid_data[col_y].values
                n = len(x_val)
                
                # Orthogonal Distance Regression (ODR)
                mydata = odr.Data(x_val, y_val)
                myodr = odr.ODR(mydata, odr.unilinear, beta0=[1.0, 0.0])
                myoutput = myodr.run()
                m, b = myoutput.beta
                
                # F-test for joint hypothesis: slope=1, intercept=0 (y=x)
                # Using orthogonal sum of squares
                rss0 = np.sum((y_val - x_val)**2) / 2.0
                rss1 = np.sum((y_val - (m * x_val + b))**2) / (1.0 + m**2)
                
                if rss1 > 0 and n > 2:
                    rss_diff = max(0.0, rss0 - rss1)
                    f_stat = ((rss_diff) / 2) / (rss1 / (n - 2))
                    p_yx = f.sf(f_stat, 2, n - 2)
                    yx_str = f"y=x p={p_yx:.4f}"
                else:
                    yx_str = "y=x p=N/A"
                    
                title = f"{title_prefix} Block Concordance\n(r={r:.3f}, p={p:.4f} | {yx_str})"
            except Exception:
                title = f"{title_prefix} Block Concordance\n(Variance error)"
        else:
            title = f"{title_prefix} Block Concordance\n(Not enough data)"
            
        # Using smaller points and lower alpha to handle dense overplotting
        ax.scatter(valid_data[col_x], valid_data[col_y], alpha=0.3, s=3)
        ax.plot([line_min, line_max], [line_min, line_max], 'r--', alpha=0.5, label='y=x')
        
        if m is not None and b is not None:
            x_fit = np.array([valid_data[col_x].min(), valid_data[col_x].max()])
            ax.plot(x_fit, m * x_fit + b, 'g-', alpha=0.7, linewidth=2, label=f'ODR Fit: y={m:.2f}x+{b:.2f}')
            
        ax.set_xlabel(xlabel, fontsize=12)
        ax.set_ylabel(ylabel, fontsize=12)
        ax.set_title(title, fontsize=14, fontweight='bold')
        ax.grid(True, alpha=0.3)
        ax.legend()
        
    plot_metric(axes[0, 0], 'goodness_of_fit_lue', 'goodness_of_fit_rue', 'LUE Goodness-of-Fit', 'RUE Goodness-of-Fit', 'GOF', 0.6, 1.0)
    plot_metric(axes[0, 1], 'power_law_range_lue', 'power_law_range_rue', 'LUE Power-Law Range', 'RUE Power-Law Range', 'Range', 0, 5)
    plot_metric(axes[1, 0], 'tau_lue', 'tau_rue', 'LUE Tau', 'RUE Tau', 'Tau', 1.0, 2.5)
    
    # New 2x2 Heatmap for Scale-Free Status Concordance
    ax_hm = axes[1, 1]
    if 'is_scale_free_lue' in df_blocks.columns and 'is_scale_free_rue' in df_blocks.columns:
        hm_data = df_blocks.dropna(subset=['is_scale_free_lue', 'is_scale_free_rue']).copy()
        hm_data['is_scale_free_lue'] = hm_data['is_scale_free_lue'].astype(bool)
        hm_data['is_scale_free_rue'] = hm_data['is_scale_free_rue'].astype(bool)
        
        ct = pd.crosstab(hm_data['is_scale_free_lue'], hm_data['is_scale_free_rue'])
        
        # Ensure it's a 2x2 matrix even if some boolean values are missing entirely
        ct = ct.reindex(index=[False, True], columns=[False, True], fill_value=0)
        
        ct.index = ['Not Scale-Free', 'Scale-Free']
        ct.columns = ['Not Scale-Free', 'Scale-Free']
        
        sns.heatmap(ct, annot=True, fmt='d', cmap='Blues', ax=ax_hm, cbar=False, annot_kws={"size": 16})
        ax_hm.set_xlabel('RUE Status', fontsize=12)
        ax_hm.set_ylabel('LUE Status', fontsize=12)
        
        total = ct.values.sum()
        agreement = (ct.iloc[0, 0] + ct.iloc[1, 1]) / total * 100 if total > 0 else 0
        ax_hm.set_title(f"Scale-Free Block Concordance\n(Agreement: {agreement:.1f}%)", fontsize=14, fontweight='bold')
    else:
        ax_hm.set_title("Scale-Free Block Concordance\n(Missing Data)", fontsize=14, fontweight='bold')
        ax_hm.axis('off')

    plt.tight_layout()
    out_path = os.path.join(output_dir, f'{file_prefix}lue_rue_block_concordance.pdf')
    plt.savefig(out_path, bbox_inches='tight', format='pdf', dpi=600)
    plt.close()
    
    csv_out = os.path.join(output_dir, f'{file_prefix}lue_rue_block_concordance.csv')
    df_blocks.to_csv(csv_out, index=False)
    print(f"Saved block-by-block concordance plots to {out_path}")
    print(f"Saved block-by-block concordance data to {csv_out}")

def plot_concordance(csv_path: str, output_dir: str, target_subject: str = None):
    """
    Generates and saves LUE vs RUE concordance plots from the results CSV.
    """
    try:
        df = pd.read_csv(csv_path)
    except Exception as e:
        print(f"Error loading CSV: {e}")
        return

    os.makedirs(output_dir, exist_ok=True)
    
    # Support 'subject_id' if 'subject' isn't available
    subj_col = 'subject_id' if 'subject_id' in df.columns else 'subject'
    if subj_col not in df.columns:
        print("Error: Could not find a valid subject column in the CSV.")
        return

    if target_subject is not None:
        df2 = df[df[subj_col].astype(str) == str(target_subject)].copy()
        if len(df2) == 0:
            print(f"No data found for subject: {target_subject}")
            matches = [str(s) for s in df[subj_col].unique() if str(target_subject) in str(s)]
            print(f"Did you mean one of the following: {matches}")
            return
        else:
            df = df2
            
    file_prefix = f"{target_subject}_" if target_subject else ""

    # Filter successful rows for continuous metrics
    df_success = df[df['success'] == True].copy()
    
    concordance_data = []
    unique_subjects = df[subj_col].unique()
    
    print(f"Aggregating data for {len(unique_subjects)} subjects...")
    for subject in unique_subjects:
        # Get all records for percentage calculation
        df_subj_all = df[df[subj_col] == subject]
        
        # Get successful records for mean values
        df_subj = df_success[df_success[subj_col] == subject]
        
        lue_all = df_subj_all[df_subj_all['arm'] == 'LUE']
        rue_all = df_subj_all[df_subj_all['arm'] == 'RUE']
        
        lue = df_subj[df_subj['arm'] == 'LUE']
        rue = df_subj[df_subj['arm'] == 'RUE']
        
        if len(lue_all) > 0 and len(rue_all) > 0:
            
            lue_gof = lue['goodness_of_fit'].mean() if len(lue) > 0 else np.nan
            rue_gof = rue['goodness_of_fit'].mean() if len(rue) > 0 else np.nan
            
            lue_range = lue['power_law_range'].mean() if len(lue) > 0 else np.nan
            rue_range = rue['power_law_range'].mean() if len(rue) > 0 else np.nan
            
            lue_tau = lue['tau'].mean() if len(lue) > 0 else np.nan
            rue_tau = rue['tau'].mean() if len(rue) > 0 else np.nan
            
            # Using all records to calculate % scale free blocks
            lue_sf_pct = 100 * lue_all['is_scale_free'].sum() / len(lue_all) if 'is_scale_free' in lue_all.columns else np.nan
            rue_sf_pct = 100 * rue_all['is_scale_free'].sum() / len(rue_all) if 'is_scale_free' in rue_all.columns else np.nan

            concordance_data.append({
                'subject': subject,
                'lue_gof_mean': lue_gof,
                'rue_gof_mean': rue_gof,
                'lue_range_mean': lue_range,
                'rue_range_mean': rue_range,
                'lue_tau_mean': lue_tau,
                'rue_tau_mean': rue_tau,
                'lue_scale_free_pct': lue_sf_pct,
                'rue_scale_free_pct': rue_sf_pct
            })
            
    if len(concordance_data) == 0:
        print("No subjects with both LUE and RUE data found.")
        return
        
    df_concordance = pd.DataFrame(concordance_data)
    
    # Plotting setup
    sns.set_style("whitegrid")
    fig, axes = plt.subplots(2, 2, figsize=(16, 16))
    
    def plot_metric(ax, col_x, col_y, xlabel, ylabel, title_prefix, line_min, line_max):
        valid_data = df_concordance.dropna(subset=[col_x, col_y])
        m, b = None, None
        if len(valid_data) > 1:
            try:
                r, p = pearsonr(valid_data[col_x], valid_data[col_y])
                
                x_val = valid_data[col_x].values
                y_val = valid_data[col_y].values
                n = len(x_val)
                
                # Orthogonal Distance Regression (ODR)
                mydata = odr.Data(x_val, y_val)
                myodr = odr.ODR(mydata, odr.unilinear, beta0=[1.0, 0.0])
                myoutput = myodr.run()
                m, b = myoutput.beta
                
                # F-test for joint hypothesis: slope=1, intercept=0 (y=x)
                # Using orthogonal sum of squares
                rss0 = np.sum((y_val - x_val)**2) / 2.0
                rss1 = np.sum((y_val - (m * x_val + b))**2) / (1.0 + m**2)
                
                if rss1 > 0 and n > 2:
                    rss_diff = max(0.0, rss0 - rss1)
                    f_stat = ((rss_diff) / 2) / (rss1 / (n - 2))
                    p_yx = f.sf(f_stat, 2, n - 2)
                    yx_str = f"y=x p={p_yx:.4f}"
                else:
                    yx_str = "y=x p=N/A"
                    
                title = f"{title_prefix} Concordance\n(r={r:.3f}, p={p:.4f} | {yx_str})"
            except Exception:
                title = f"{title_prefix} Concordance\n(Variance error)"
        else:
            title = f"{title_prefix} Concordance\n(Not enough data)"
            
        ax.scatter(valid_data[col_x], valid_data[col_y], alpha=0.6, s=50)
        ax.plot([line_min, line_max], [line_min, line_max], 'r--', alpha=0.5, label='y=x')
        
        if m is not None and b is not None:
            x_fit = np.array([valid_data[col_x].min(), valid_data[col_x].max()])
            ax.plot(x_fit, m * x_fit + b, 'g-', alpha=0.7, linewidth=2, label=f'ODR Fit: y={m:.2f}x+{b:.2f}')
            
        ax.set_xlabel(xlabel, fontsize=12)
        ax.set_ylabel(ylabel, fontsize=12)
        ax.set_title(title, fontsize=14, fontweight='bold')
        ax.grid(True, alpha=0.3)
        ax.legend()

    plot_metric(axes[0, 0], 'lue_gof_mean', 'rue_gof_mean', 'LUE Mean Goodness-of-Fit', 'RUE Mean Goodness-of-Fit', 'GOF', 0.6, 1.0)
    plot_metric(axes[0, 1], 'lue_range_mean', 'rue_range_mean', 'LUE Mean Power-Law Range (decades)', 'RUE Mean Power-Law Range (decades)', 'Range', 0, 5)
    plot_metric(axes[1, 0], 'lue_tau_mean', 'rue_tau_mean', 'LUE Mean Tau', 'RUE Mean Tau', 'Tau', 1.0, 2.5)
    plot_metric(axes[1, 1], 'lue_scale_free_pct', 'rue_scale_free_pct', 'LUE % Scale-Free Blocks', 'RUE % Scale-Free Blocks', 'Scale-Free %', 0, 100)
    
    plt.tight_layout()
    out_path = os.path.join(output_dir, f'{file_prefix}lue_rue_concordance.pdf')
    plt.savefig(out_path, bbox_inches='tight', format='pdf', dpi=300)
    plt.close()
    
    csv_out = os.path.join(output_dir, f'{file_prefix}lue_rue_concordance.csv')
    df_concordance.to_csv(csv_out, index=False)
    
    print(f"Saved concordance plots to {out_path}")
    print(f"Saved concordance data to {csv_out}")

    # Generate the block-by-block concordance
    plot_block_concordance(df, output_dir, file_prefix)

def main():
    parser = argparse.ArgumentParser(description="Generate LUE/RUE Concordance Plots")
    parser.add_argument('--csv_path', type=str, required=True, help="Path to results CSV (e.g., master_scale_free_results.csv)")
    parser.add_argument('--output_dir', type=str, default='../output/concordance', help="Output directory for the plot and aggregated CSV")
    parser.add_argument('--subject', type=str, default=None, help="Optional: Filter to a specific subject ID")

    "First p-value <0.05 means there exists a linear relationship."
    "Second p-value <0.05 means that y=x is NOT the true distrobution."
    
    args = parser.parse_args()
    plot_concordance(args.csv_path, args.output_dir, args.subject)
    
if __name__ == '__main__':
    main()
