#!/usr/bin/env python3
"""
Predict ARAT Scores from Accelerometry Metrics

This script merges the output of UpperLimbAccelerometry.py with demographic data
to evaluate how well upper limb metrics predict the AffARATTotal score. 
It uses non-parametric correlation (Spearman) and Random Forest Regression.
"""

import os
import argparse
from datetime import datetime
import pandas as pd
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import LeaveOneOut, cross_val_predict
from sklearn.impute import SimpleImputer
from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_error

# Plot styling
matplotlib.rcParams["pdf.fonttype"] = 42
matplotlib.rcParams["ps.fonttype"] = 42

def clean_id(val):
    """Extract the numeric part of the subject ID to ensure merging works."""
    val = str(val).strip()
    if '_' in val:
        return val.split('_')[-1]
    return val

def clean_tp(val):
    """Extract the integer timepoint to ensure merging works."""
    val = str(val).strip().lower().replace('timepoint', '')
    try:
        return int(float(val))
    except ValueError:
        return -1

def main():
    parser = argparse.ArgumentParser(description="Predict AffARATTotal from Accelerometry Metrics")
    parser.add_argument('--metrics_csv', type=str, required=True, help="Path to UpperLimbAccelerometry_Metrics.csv")
    parser.add_argument('--demo_csv', type=str, required=True, help="Path to DemographicClinicalData.csv")
    parser.add_argument('--scale_free_csv', type=str, default=None, help="Optional path to master_scale_free_results.csv to include median power law ranges")
    parser.add_argument('--output_dir', type=str, default=None, help="Directory to save analysis results (defaults to timestamped folder)")
    parser.add_argument('--timepoint', type=int, default=None, help="Optional timepoint to filter the data (e.g. 0, 1, 4)")
    
    args = parser.parse_args()
    
    if not args.output_dir:
        tp_str = f"_TP{args.timepoint}" if args.timepoint is not None else ""
        args.output_dir = os.path.join('../output/arat_prediction', datetime.now().strftime('%Y%m%d_%H%M%S') + tp_str)
    os.makedirs(args.output_dir, exist_ok=True)

    print("Loading data...")
    try:
        df_metrics = pd.read_csv(args.metrics_csv)
        df_demo = pd.read_csv(args.demo_csv)
    except Exception as e:
        print(f"Error loading files: {e}")
        return

    # Clean merge keys
    df_metrics['clean_id'] = df_metrics['StudyID'].apply(clean_id)
    df_metrics['clean_tp'] = df_metrics['TimePoint'].apply(clean_tp)
    
    # Process scale-free metrics if provided
    if args.scale_free_csv:
        try:
            print(f"Loading scale-free results from {args.scale_free_csv}...")
            df_sf = pd.read_csv(args.scale_free_csv)
            if 'success' in df_sf.columns:
                df_sf = df_sf[df_sf['success'] == True]
            
            df_sf['clean_id'] = df_sf['subject_id'].apply(clean_id)
            df_sf['clean_tp'] = df_sf['timepoint_number'].apply(lambda x: int(float(x)) if pd.notnull(x) else -1)
            
            # Calculate median power law range per subject/tp/arm
            sf_agg = df_sf.groupby(['clean_id', 'clean_tp', 'arm'])['power_law_range'].median().reset_index()
            sf_pivot = sf_agg.pivot(index=['clean_id', 'clean_tp'], columns='arm', values='power_law_range').reset_index()
            
            col_mapping = {}
            if 'LUE' in sf_pivot.columns: col_mapping['LUE'] = 'l_median_power_law_range'
            if 'RUE' in sf_pivot.columns: col_mapping['RUE'] = 'r_median_power_law_range'
            sf_pivot = sf_pivot.rename(columns=col_mapping)
            
            # Calculate total median across both arms combined
            sf_total = df_sf.groupby(['clean_id', 'clean_tp'])['power_law_range'].median().reset_index()
            sf_total = sf_total.rename(columns={'power_law_range': 'total_median_power_law_range'})
            
            # Count scale-free blocks per subject/tp/arm
            sf_count_agg = df_sf.groupby(['clean_id', 'clean_tp', 'arm'])['is_scale_free'].sum().reset_index()
            sf_count_pivot = sf_count_agg.pivot(index=['clean_id', 'clean_tp'], columns='arm', values='is_scale_free').reset_index()
            
            col_mapping_counts = {}
            if 'LUE' in sf_count_pivot.columns: col_mapping_counts['LUE'] = 'l_scale_free_blocks'
            if 'RUE' in sf_count_pivot.columns: col_mapping_counts['RUE'] = 'r_scale_free_blocks'
            sf_count_pivot = sf_count_pivot.rename(columns=col_mapping_counts)
            
            # Calculate total scale-free blocks across both arms combined
            sf_count_total = df_sf.groupby(['clean_id', 'clean_tp'])['is_scale_free'].sum().reset_index()
            sf_count_total = sf_count_total.rename(columns={'is_scale_free': 'total_scale_free_blocks'})
            
            # Combine all features
            sf_features_df = pd.merge(sf_pivot, sf_total, on=['clean_id', 'clean_tp'], how='outer')
            sf_features_df = pd.merge(sf_features_df, sf_count_pivot, on=['clean_id', 'clean_tp'], how='outer')
            sf_features_df = pd.merge(sf_features_df, sf_count_total, on=['clean_id', 'clean_tp'], how='outer')
            
            df_metrics = pd.merge(df_metrics, sf_features_df, on=['clean_id', 'clean_tp'], how='left')
            print("Successfully merged scale-free features (power-law ranges and block counts).")
        except Exception as e:
            print(f"Error processing scale_free_csv: {e}")

    # Locate ID column in demographics
    subj_col = next((c for c in ['SubIDName', 'subject_id', 'subjectid'] if c in df_demo.columns), df_demo.columns[0])
    df_demo['clean_id'] = df_demo[subj_col].apply(clean_id)
    
    tp_col = next((c for c in ['TimePoint', 'timepoint'] if c in df_demo.columns), None)
    if tp_col:
        df_demo['clean_tp'] = df_demo[tp_col].apply(clean_tp)
    else:
        print("TimePoint column not found in demographics, assuming all match.")
        df_demo['clean_tp'] = df_metrics['clean_tp'].iloc[0] if not df_metrics.empty else 0

    # Merge datasets
    df_merged = pd.merge(df_metrics, df_demo, on=['clean_id', 'clean_tp'], how='inner')
    
    print(f"Merged datasets. Found {len(df_merged)} matched sessions.")
    
    if args.timepoint is not None:
        df_merged = df_merged[df_merged['clean_tp'] == args.timepoint]
        print(f"Filtered to TimePoint {args.timepoint}. Remaining sessions: {len(df_merged)}")
        
    if 'AffARATTotal' not in df_merged.columns:
        print("Error: 'AffARATTotal' column not found in demographics data.")
        return
        
    # Filter to subjects who actually have an ARAT score
    df_merged['AffARATTotal'] = pd.to_numeric(df_merged['AffARATTotal'], errors='coerce')
    df_valid = df_merged.dropna(subset=['AffARATTotal']).copy()
    
    if len(df_valid) < 10:
        print(f"Insufficient data: Only {len(df_valid)} subjects have an AffARATTotal score.")
        return
        
    print(f"Proceeding with {len(df_valid)} sessions containing valid AffARATTotal scores.")

    # Extract features (exclude identifiers and demographic columns)
    exclude_cols = ['StudyID', 'TimePoint', 'clean_id', 'clean_tp'] + list(df_demo.columns)
    feature_cols = [c for c in df_metrics.columns if c not in exclude_cols]
    
    X_raw = df_valid[feature_cols]
    y = df_valid['AffARATTotal']
    
    # Impute missing feature values with the median (non-parametric approach)
    imputer = SimpleImputer(strategy='median')
    X = pd.DataFrame(imputer.fit_transform(X_raw), columns=feature_cols, index=X_raw.index)
    
    # ==========================================
    # 1. Correlation Analysis
    # ==========================================
    print("\nCalculating correlations...")
    correlations = []
    for col in feature_cols:
        # Spearman (Non-parametric - better for ARAT bounds)
        rho, p_spearman = stats.spearmanr(X[col], y)
        # Pearson (Parametric - for comparison)
        r, p_pearson = stats.pearsonr(X[col], y)
        
        correlations.append({
            'Feature': col,
            'Spearman_Rho': rho,
            'Spearman_P': p_spearman,
            'Pearson_R': r,
            'Pearson_P': p_pearson
        })
        
    df_corr = pd.DataFrame(correlations).sort_values(by='Spearman_Rho', key=abs, ascending=False).reset_index(drop=True)
    df_corr.to_csv(os.path.join(args.output_dir, 'feature_correlations.csv'), index=False)
    
    print("\nTop 15 Correlated Features (Spearman):")
    print(df_corr[['Feature', 'Spearman_Rho', 'Spearman_P']].head(15).to_string(index=False))
    
    # Plot Top 4 correlations
    top_features = df_corr['Feature'].head(4).tolist()
    fig, axes = plt.subplots(2, 2, figsize=(12, 10))
    axes = axes.flatten()
    
    for i, feature in enumerate(top_features):
        sns.regplot(x=X[feature], y=y, ax=axes[i], scatter_kws={'alpha': 0.6}, line_kws={'color': 'red'})
        rho = df_corr.loc[df_corr['Feature'] == feature, 'Spearman_Rho'].values[0]
        p = df_corr.loc[df_corr['Feature'] == feature, 'Spearman_P'].values[0]
        axes[i].set_title(f"{feature}\nSpearman ρ = {rho:.3f} (p = {p:.3f})")
        axes[i].set_ylabel("Affected ARAT Total")
        axes[i].grid(True, linestyle='--', alpha=0.5)
        
    plt.tight_layout()
    plt.savefig(os.path.join(args.output_dir, 'top_4_correlations.pdf'))
    plt.close()

    # ==========================================
    # 2. Predictive Modeling (Random Forest)
    # ==========================================
    print("\nTraining Random Forest Regressor (Non-parametric ML model)...")
    
    # Random Forest is naturally robust to non-linearities and outliers
    rf = RandomForestRegressor(n_estimators=100, random_state=42, max_depth=5)
    
    # Evaluate using Leave-One-Out Cross-Validation (LOOCV), better for very small datasets
    cv = LeaveOneOut()
    y_pred_cv = cross_val_predict(rf, X, y, cv=cv)
    
    # Calculate Metrics
    r2 = r2_score(y, y_pred_cv)
    rmse = np.sqrt(mean_squared_error(y, y_pred_cv))
    mae = mean_absolute_error(y, y_pred_cv)
    
    print("Cross-Validation Results:")
    print(f"  R-squared (R2): {r2:.3f}")
    print(f"  RMSE:           {rmse:.3f} points")
    print(f"  MAE:            {mae:.3f} points")
    
    # Train on full dataset to extract Feature Importances
    rf.fit(X, y)
    importances = rf.feature_importances_
    
    df_imp = pd.DataFrame({
        'Feature': feature_cols,
        'Importance': importances
    }).sort_values(by='Importance', ascending=False).reset_index(drop=True)
    
    df_imp.to_csv(os.path.join(args.output_dir, 'rf_feature_importances.csv'), index=False)
    
    # ==========================================
    # 3. Plots for Modeling
    # ==========================================
    # Feature Importance Plot
    plt.figure(figsize=(10, 8))
    sns.barplot(data=df_imp.head(15), x='Importance', y='Feature', palette='viridis')
    plt.title('Top 15 Random Forest Feature Importances')
    plt.xlabel('Importance (Reduction in Impurity)')
    plt.grid(True, axis='x', linestyle='--', alpha=0.5)
    plt.tight_layout()
    plt.savefig(os.path.join(args.output_dir, 'rf_feature_importances.pdf'))
    plt.close()
    
    # Predicted vs Actual Plot
    plt.figure(figsize=(8, 8))
    plt.scatter(y_pred_cv, y, alpha=0.7, color='blue', edgecolor='k')
    
    # Ideal line
    min_val = min(y.min(), y_pred_cv.min())
    max_val = max(y.max(), y_pred_cv.max())
    plt.plot([min_val, max_val], [min_val, max_val], 'r--', lw=2, label='Perfect Prediction')
    
    plt.title(f'Random Forest: Predicted vs Actual ARAT\nCross-Validated R² = {r2:.3f}, MAE = {mae:.2f}')
    plt.xlabel('Predicted AffARATTotal')
    plt.ylabel('Actual AffARATTotal')
    plt.legend()
    plt.grid(True, linestyle='--', alpha=0.5)
    plt.tight_layout()
    plt.savefig(os.path.join(args.output_dir, 'rf_predicted_vs_actual.pdf'))
    plt.close()
    
    # Check if power law ranges were good predictors
    sf_features = [c for c in [
        'l_median_power_law_range', 'r_median_power_law_range', 'total_median_power_law_range',
        'l_scale_free_blocks', 'r_scale_free_blocks', 'total_scale_free_blocks'
    ] if c in df_corr['Feature'].values]
    sf_performance = []
    if sf_features:
        print("\n--- Scale-Free Feature Predictiveness ---")
        for sf_feat in sf_features:
            rank = df_corr.index[df_corr['Feature'] == sf_feat].tolist()[0] + 1
            total_feats = len(df_corr)
            rho = df_corr.loc[df_corr['Feature'] == sf_feat, 'Spearman_Rho'].values[0]
            imp_rank = df_imp.index[df_imp['Feature'] == sf_feat].tolist()[0] + 1
            sf_performance.append(f"{sf_feat}: Correlation Rank {rank}/{total_feats} (Rho={rho:.3f}), Importance Rank {imp_rank}/{total_feats}")
            print(sf_performance[-1])
            
        is_better = any(df_corr.index[df_corr['Feature'] == sf_feat].tolist()[0] < 5 for sf_feat in sf_features)
        if is_better:
            print("Conclusion: Yes, scale-free metrics appear to be very strong predictors, ranking in the top 5 features!")
        else:
            print("Conclusion: Scale-free metrics have some predictive value, but traditional kinematic metrics outrank them for predicting ARAT.")

    # Save summary report
    with open(os.path.join(args.output_dir, 'prediction_summary.txt'), 'w') as f:
        f.write("ARAT Prediction Analysis Summary\n")
        f.write("================================\n\n")
        if args.timepoint is not None:
            f.write(f"Filtered to TimePoint: {args.timepoint}\n")
        f.write(f"Matched subjects/sessions: {len(df_valid)}\n\n")
        f.write("Random Forest (Non-Parametric) Cross-Validation Results:\n")
        f.write(f"  R-squared: {r2:.3f}\n")
        f.write(f"  RMSE:      {rmse:.3f}\n")
        f.write(f"  MAE:       {mae:.3f}\n\n")
        f.write("Top 5 Correlated Features (Spearman Rho):\n")
        for _, row in df_corr.head(5).iterrows():
            f.write(f"  {row['Feature']}: {row['Spearman_Rho']:.3f} (p={row['Spearman_P']:.4f})\n")
            
        if sf_performance:
            f.write("\nScale-Free Feature Performance:\n")
            for perf in sf_performance:
                f.write(f"  {perf}\n")

    print(f"\nAnalysis complete. Results and plots saved to {args.output_dir}")

if __name__ == '__main__':
    main()