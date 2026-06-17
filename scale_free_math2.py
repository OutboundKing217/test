"""
Scale-Free Mathematics Module (v2)

Contains the core statistical wrappers and Maximum Likelihood Estimation 
logic for finding truncated power-law fits (Jones et al. 2023).

Optimizations in v2:
- Added required empty state definitions for safe exits.
- Swapped inefficient linear grid searches with bounded scalar optimization.
- Swapped O(N) array scans with O(log N) searchsorted logic.
- Processed surrogates in memory-safe chunks to prevent parallel OOM crashes.
- Completely decoupled plotting UI logic from pure mathematical functions.
"""
import warnings
import numpy as np
import pandas as pd
from typing import Dict, Union
from scipy.interpolate import interp1d
from scipy.optimize import minimize_scalar

# suppress divide by 0
warnings.filterwarnings('ignore', category=RuntimeWarning)


def _empty_results() -> Dict[str, Union[float, int, np.ndarray, None]]:
    """
    Provides a standardized empty result dictionary if the data fails preconditions.
    """
    return {
        'tau': np.nan,
        'power_law_range': np.nan,
        'lower_cutoff': np.nan,
        'upper_cutoff': np.nan,
        'n_samples': 0,
        'goodness_of_fit': np.nan,
        'uncertainty': 0,
        'is_scale_free': False,
        'event_sizes_used': np.array([]),
        'surrogate_bounds': None,
        'bin_centers': None,
        'success': False
    }


def analyze_scale_free_events(
    events_df: pd.DataFrame,
    event_size_column: str = 'auc',
    plotflag: bool = False,  # Deprecated in v2, kept for signature compatibility
    verbose: bool = True
) -> Dict[str, Union[float, int, np.ndarray, None]]:
    """
    Analyze scale-free properties of behavioral events using PLfitPLrange
    
    Parameters:
    -----------
    events_df : pandas DataFrame
        DataFrame containing behavioral events.
    event_size_column : str, optional
        Name of column containing event sizes. Default: 'auc'
    plotflag : bool, optional
        Deprecated. Plotting is decoupled in v2. Kept to avoid breaking existing code.
    verbose : bool, optional
        Whether to print progress information. Default: True
        
    Returns:
    --------
    results : dict
        Dictionary containing tau, range, bounds, fitness, and successful flag.
    """
    
    # Validate inputs
    if events_df.empty:
        if verbose:
            print("Warning: Empty events DataFrame provided")
        return _empty_results()
    
    if event_size_column not in events_df.columns:
        raise ValueError(f"Column '{event_size_column}' not found in events_df. "
                        f"Available columns: {list(events_df.columns)}")
    
    # Extract event sizes
    event_sizes = events_df[event_size_column].values
    
    # Filter out invalid values (NaN, inf, zero, negative)
    valid_mask = np.isfinite(event_sizes) & (event_sizes > 0)
    event_sizes_clean = event_sizes[valid_mask]
    
    # Check if not enough events (Need >20)
    if len(event_sizes_clean) < 20:
        if verbose:
            print(f"Warning: Too few valid events ({len(event_sizes_clean)}). "
                  f"Need at least 20 events for analysis.")
        return _empty_results()
    
    if verbose:
        print(f"Analyzing {len(event_sizes_clean)} events for scale-free properties...")
        print(f"  Event size range: {np.min(event_sizes_clean):.6f} - {np.max(event_sizes_clean):.6f}")
    
    # Call PLfitPLrange (Returns result dictionary directly in v2)
    try:
        results = PLfitPLrange(event_sizes_clean)
    except Exception as e:
        if verbose:
            print(f"Error during analysis: {e}")
        return _empty_results()
    
    if verbose and results['success']:
        print(f"\nScale-free analysis results:")
        print(f"  Power-law exponent (tau): {results['tau']:.4f}")
        print(f"  Power-law range: {results['power_law_range']:.4f} decades")
        print(f"  Fit range: {results['lower_cutoff']:.6f} - {results['upper_cutoff']:.6f}")
        print(f"  Events in fit range: {results['n_samples']}")
        gof_status = "(GOOD)" if results['goodness_of_fit'] >= 0.8 else "(POOR)"
        print(f"  Goodness of fit: {results['goodness_of_fit']:.4f} {gof_status}")
    
    return results


def PLfitPLrange(evsiz: np.ndarray) -> Dict:
    """
    Performs event size distribution analysis to find best truncated power-law fit.
    (Jones et al eLife 2023)
    """
    
    # Convert to numpy array, flatten, and sort
    evsiz = np.sort(np.asarray(evsiz).flatten())
    
    GOFcrit = 0.8
    nav = len(evsiz)
    
    if nav < 20:
        return _empty_results()
    
    dbr = np.log10(np.max(evsiz)) - np.log10(np.min(evsiz))
    if dbr <= 0:
        return _empty_results()
        
    # Exclude outliers (points adjacent to gaps in the dB range >10% of the total range)
    log_evsiz = np.log10(evsiz)
    diff_log = np.diff(log_evsiz)
    outliers = np.where(np.abs(diff_log) / dbr > 0.1)[0]
    
    if len(outliers) > 0:
        firstgoodpoint = int(np.max([0] + list(outliers[outliers < nav / 10]))) + 1
        lastgoodpoint = int(np.min([nav] + list(outliers[outliers > 9 * nav / 10])))
    else:
        firstgoodpoint = 0
        lastgoodpoint = nav
    
    evsiz = evsiz[firstgoodpoint:lastgoodpoint]
    
    if len(evsiz) < 20:
        return _empty_results()
    
    dbr = np.log10(np.max(evsiz)) - np.log10(np.min(evsiz))
    if dbr <= 0:
        return _empty_results()
        
    # Logarithmically spaced bin vector for cutoffs
    binv = np.logspace(np.log10(np.min(evsiz)), np.log10(np.max(evsiz)), int(10 * dbr))
    ns = len(binv)
    nrep = 500
    
    best_results = _empty_results()
    best_results['event_sizes_used'] = evsiz
    bestrho = 0
    
    # Loop through different lower cutoffs
    for mmm in range(ns - 1):
        mm = binv[mmm]
        MM = binv[-1] 
        z = evsiz[(evsiz >= mm) & (evsiz <= MM)]
        nz = len(z)
        
        if nz <= 20:
            continue
            
        # Reject fit if there is a gap of >1 decade in the data that would be fitted
        if np.max(np.diff(np.log10(z))) > 1.0:
            continue

        rho = np.log10(MM / mm)
        
        # Cancel the fit if the fit length is less than one decade
        if rho < 1.0:
            continue

        if rho > bestrho:
            # Find max likelihood power-law exponent utilizing bounded scalar optimization
            def nll(aa):
                # Use np.isclose to catch optimizer values that are functionally 1.0
                if np.isclose(aa, 1.0, atol=1e-5):
                    prs = 1.0 / (np.log(MM) - np.log(mm)) / z
                else:
                    # Calculate denominator safely
                    denom = (mm**(1 - aa) - MM**(1 - aa))
                    # Prevent zero division if floating point precision fails
                    if denom == 0:
                        prs = 1.0 / (np.log(MM) - np.log(mm)) / z
                    else:
                        prs = ((aa - 1) / denom) * (z ** -aa)
                
                # Keep bounds valid to prevent log(0)
                prs = np.maximum(prs, 1e-12)
                return -np.mean(np.log(prs))
                
            res = minimize_scalar(nll, bounds=(0.7, 2.01), method='bounded')
            aa = res.x
            
            # Resampled real data CDF: 10 points per decade logarithmically spaced
            z_sorted = np.sort(z)
            cdf_real = np.arange(1, nz + 1) / nz
            cprob, rlz = resample_logspace(cdf_real, np.log10(z_sorted), 10)
            ncomp = len(cprob)
            rz = 10**rlz
            
            # Surrogates bounds logic (Memory-safe Chunking + O(log N) searchsorted)
            q_min = np.full(ncomp, np.inf)
            q_max = np.full(ncomp, -np.inf)
            batch_size = 50
            
            for b in range(0, nrep, batch_size):
                curr_batch = min(batch_size, nrep - b)
                u = np.random.rand(curr_batch, nz)
                if aa == 1.0:
                    sz_batch = np.exp(u * (np.log(MM) - np.log(mm)) + np.log(mm))
                else:
                    sz_batch = ((u * (MM**(1 - aa) - mm**(1 - aa)) + mm**(1 - aa)) ** (1 / (1 - aa)))
                sz_batch.sort(axis=1)
                
                # Rapid empirical CDF tracking
                for row in sz_batch:
                    cdf_vals = np.searchsorted(row, rz, side='right') / nz
                    q_min = np.minimum(q_min, cdf_vals)
                    q_max = np.maximum(q_max, cdf_vals)
            
            fracgood = np.mean((cprob > q_min) & (cprob < q_max))
            
            if fracgood > GOFcrit:
                bestrho = rho
                best_results.update({
                    'tau': aa,
                    'power_law_range': rho,
                    'lower_cutoff': mm,
                    'upper_cutoff': MM,
                    'n_samples': nz,
                    'goodness_of_fit': fracgood,
                    'is_scale_free': True,
                    'success': True
                })

    return best_results


def resample_logspace(y_values, x_values_log, points_per_decade):
    """ Resample data at logarithmically spaced points """
    log_min, log_max = np.min(x_values_log), np.max(x_values_log)
    x_resampled_log = np.linspace(log_min, log_max, int(np.ceil((log_max - log_min) * points_per_decade)) + 1)
    interp_func = interp1d(x_values_log, y_values, kind='linear', bounds_error=False, fill_value=(y_values[0], y_values[-1]))
    return interp_func(x_resampled_log), x_resampled_log
