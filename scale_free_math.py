
"""
Scale-Free Mathematics Module

Contains the core statistical wrappers and Maximum Likelihood Estimation 
logic for finding truncated power-law fits (Jones et al. 2023).
"""
import warnings
import numpy as np
import pandas as pd
from typing import Dict, Union, Optional
from scipy.interpolate import interp1d

# suppress divide by 0
warnings.filterwarnings('ignore', category=RuntimeWarning)



def analyze_scale_free_events(
    events_df: pd.DataFrame,
    event_size_column: str = 'auc',
    plotflag: bool = False,
    verbose: bool = True
) -> Dict[str, Union[float, int, np.ndarray, None]]:
    """
    Analyze scale-free properties of behavioral events using PLfitPLrange
    
    This wrapper function extracts event sizes from your events DataFrame and
    performs scale-free analysis following Jones et al. (2023) methodology.
    
    Parameters:
    -----------
    events_df : pandas DataFrame
        DataFrame containing behavioral events, typically from detect_behavioral_events()
        Must contain a column with event sizes (default: 'auc')
    event_size_column : str, optional
        Name of column containing event sizes. Default: 'auc'
    plotflag : bool, optional
        Whether to generate plots (slower). Default: False
    verbose : bool, optional
        Whether to print progress information. Default: True
    
    Returns:
    --------
    results : dict
        Dictionary containing:
        - 'tau' : float
            Best fit power-law exponent (lower = more scale-free)
        - 'power_law_range' : float
            Power-law range in decades (how many orders of magnitude fit power-law)
        - 'lower_cutoff' : float
            Small-size cutoff for the power-law fit
        - 'upper_cutoff' : float
            Large-size cutoff for the power-law fit
        - 'n_samples' : int
            Number of events in the best fit range
        - 'goodness_of_fit' : float
            Fraction of CDF within confidence bounds (0-1, higher = better fit)
            Values > 0.8 indicate good power-law fit
        - 'is_scale_free' : bool
            True if goodness_of_fit >= 0.8 (indicating scale-free behavior)
        - 'event_sizes_used' : ndarray
            Array of event sizes that were analyzed
        - 'surrogate_bounds' : ndarray or None
            Confidence intervals for plotting (if fit was successful)
        - 'bin_centers' : ndarray or None
            Bin centers for plotting (if fit was successful)
        - 'success' : bool
            True if analysis completed successfully
    
    Examples:
    --------
    >>> from explore_pvalue_parameters import detect_behavioral_events, butter_gravity_filter
    >>> import pandas as pd
    >>> 
    >>> # Load and preprocess data (your existing workflow)
    >>> df = pd.read_csv('data.csv', skiprows=10, header=None)
    >>> df['time'] = df.index / 30  # 30 Hz sampling rate
    >>> df = butter_gravity_filter(df, sampling_rate=30)
    >>> 
    >>> # Detect events (your existing workflow)
    >>> events = detect_behavioral_events(df, median_threshold=0.05)
    >>> 
    >>> # Analyze scale-free properties (NEW - using wrapper)
    >>> results = analyze_scale_free_events(events, event_size_column='auc')
    >>> 
    >>> # Check results
    >>> print(f"Power-law exponent (tau): {results['tau']:.3f}")
    >>> print(f"Power-law range: {results['power_law_range']:.2f} decades")
    >>> print(f"Is scale-free: {results['is_scale_free']}")
    >>> print(f"Goodness of fit: {results['goodness_of_fit']:.3f}")
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
    
    # Call PLfitPLrange
    try:
        tau, bestrho, bestmm, bestMM, nsamp, bestfracgood, bestuncert, surbound, binvs = \
            PLfitPLrange(event_sizes_clean, plotflag=plotflag)
    except Exception as e:
        if verbose:
            print(f"Error during analysis: {e}")
        return _empty_results()
    
    # Check if analysis was successful (tau > 0 indicates a fit was found)
    success = tau > 0 and bestrho > 0
    
    if verbose and success:
        print(f"\nScale-free analysis results:")
        print(f"  Power-law exponent (tau): {tau:.4f}")
        print(f"  Power-law range: {bestrho:.4f} decades")
        print(f"  Fit range: {bestmm:.6f} - {bestMM:.6f}")
        print(f"  Events in fit range: {nsamp}")
        print(f"  Goodness of fit: {bestfracgood:.4f} {'(GOOD)' if bestfracgood >= 0.8 else '(POOR)'}")
    
    # Package results
    results = {
        'tau': tau,
        'power_law_range': bestrho,
        'lower_cutoff': bestmm,
        'upper_cutoff': bestMM,
        'n_samples': nsamp,
        'goodness_of_fit': bestfracgood,
        'uncertainty': bestuncert,
        'is_scale_free': bestfracgood >= 0.8,
        'event_sizes_used': event_sizes_clean,
        'surrogate_bounds': surbound,
        'bin_centers': binvs,
        'success': success
    }
    
    return results


def PLfitPLrange(evsiz, plotflag=False):
    """
    This function performs event size distribution analysis as described in
    Jones et al eLife 2023
    
    Parameters:
    -----------
    evsiz : array-like
        1D array containing the sizes of n events
    plotflag : bool, optional
        Set to True to create plots of results (slow)
        Set to False to skip plots (fast)
        Default: False
    
    Returns:
    --------
    tau : float
        Best fit power-law exponent
    bestrho : float
        Power-law range for the best fit (in decades)
    bestmm : float
        Small-size cut off for the power-law
    bestMM : float
        Large-size cut off for the power-law
    nsamp : int
        Number of samples that fall within the best fit range
    bestfracgood : float
        Goodness-of-fit measure for the best fit
    bestuncert : float
        Uncertainty measure (not explicitly calculated in original, set to 0)
    surbound : ndarray or None
        Confidence intervals for plotting PDFs (shape: (2, n_bins) or None)
    binvs : ndarray or None
        List of bin centers for plotting PDFs
    
    Notes:
    ------
    The algorithm has two main goals:
    1. Obtain a maximum likelihood truncated power-law fit to the event size distribution
    2. Obtain the power-law range (number of decades over which the power-law is a good fit)
    
    Maximum likelihood fit for powerlaw with upper and lower cutoffs.
    Range of power law fit is defined by:
    - Criterion 1: at least 80% of the range must be within 5-95% confidence
      interval of 500 surrogate data sets drawn from best fit distribution
    - Criterion 2: the 500 surrogate CDFs must not be too broadly scattered
    """
    
    # Convert to numpy array and ensure 1D
    evsiz = np.asarray(evsiz).flatten()
    
    # Sort data
    evsiz = np.sort(evsiz)
    
    # Goodness-of-fit criterion
    GOFcrit = 0.8  # (set lower to allow worse fits, higher to require better fits)
    
    nav = len(evsiz)  # number of events in data set
    
    if nav < 20:
        # Not enough data points for meaningful analysis
        return (0, 0, 0, 0, 0, 0, 0, None, None)
    
    # Prep some variables for plotting probability density function
    if plotflag:
        dbrange1 = np.log10(np.max(evsiz)) - np.log10(np.min(evsiz))
        binv = np.logspace(np.log10(np.min(evsiz)), np.log10(np.max(evsiz)), 
                          int(10 * dbrange1))
        plotbins_all = binv[:-1] + np.diff(binv) / 2
        # Use np.histogram instead of histc
        n, _ = np.histogram(evsiz, bins=binv)
        pdf_all = n / np.diff(binv) / nav
    
    # Note evsize is SORTED

    # Exclude outliers (points adjacent to gaps in the dB range >3% of the total range)
    # dbr is the range of the data in log-10 space
    dbr = np.log10(np.max(evsiz)) - np.log10(np.min(evsiz))
    # log_evsiz is an array with log-10 of each data point
    log_evsiz = np.log10(evsiz)
    # diff_log is an array with each element equal to the difference from the previous
    # ie this is percent change because this is in log-10 space
    diff_log = np.diff(log_evsiz)
    # outliers is where the difference in event size divided by the range of the data is >0.03
    # this means having a larger range of data leads to larger percent change allowed per data point?
    # Is this correct?????
    outliers = np.where(np.abs(diff_log) / dbr > 0.03)[0]
    
    # if there are outliers:
    #   Set first good point to the largest outlier in first quartile, plus 1
    #   Set last good point to the smallest outlier in thrid quartile
    # this essentially filters out up to 50% of the data if there is are large-ish breaks in the data
    if len(outliers) > 0:
        firstgoodpoint = int(np.max([0] + list(outliers[outliers < nav / 4]))) + 1
        lastgoodpoint = int(np.min([nav] + list(outliers[outliers > 3 * nav / 4])))
    else:
        firstgoodpoint = 0
        lastgoodpoint = nav
    
    evsiz = evsiz[firstgoodpoint:lastgoodpoint]
    
    # check if not enough data after outlier removal
    if len(evsiz) < 20:
        # Not enough data after outlier removal
        return (0, 0, 0, 0, 0, 0, 0, None, None)
    
    # List of sizes to try for min and max sizes of power law truncation (also used for plotting)

    # calculate new range
    dbr = np.log10(np.max(evsiz)) - np.log10(np.min(evsiz))

    # create 10 points for every order of magnitude in the range, spaced evenly in log-10 sapce
    # this is for lower cutoffs
    binv = np.logspace(np.log10(np.min(evsiz)), np.log10(np.max(evsiz)), int(10 * dbr))
    ns = len(binv)
    
    if plotflag:
        plotbins = binv[:-1] + np.diff(binv) / 2
    
    # List of power law exponents to try fitting
    # exlist is a list of power law exponents to try fitting. 
    # is is every 0.02 between 0.7 to 2.01
    # correct? Why not try many more values outside this range (or calculate it)?
    exlist = np.arange(0.7, 2.01, 0.02)
    nex = len(exlist)
    
    # Number of surrogate datasets to use
    nrep = 500
    
    # Initialize best values
    bestK = 0
    bestMM = 0
    bestmm = 0
    tau = 0
    bestrho = 0
    nsamp = 0
    bestfracgood = 0
    bestuncert = 0
    
    # Loop through different lower cutoffs
    for mmm in range(ns - 1):
        # Truncate the data set using the specified lower cutoff
        mm = binv[mmm]
        MM = binv[-1] 

        # z is all data between mm and MM
        z = evsiz[(evsiz >= mm) & (evsiz <= MM)]
        nz = len(z)
        
        # Compute range of truncated data set
        rho = np.log10(MM / mm)
        
        # if range of set is bigger than best range so far:
        if rho > bestrho and nz > 20:

            # Find max likelihood power-law exponent
            L = np.zeros(nex)
            for q in range(nex):
                aa = exlist[q]  # the qth candidate exponent
                # Probabilities: normalized truncated power-law PDF
                # Avoid division by zero for edge cases
                if aa == 1.0:
                    # Special case: uniform distribution in log space
                    prs = 1.0 / (np.log(MM) - np.log(mm)) / z
                else:
                    prs = ((aa - 1) / (mm**(1 - aa) - MM**(1 - aa))) * (z ** -aa)
                L[q] = np.mean(np.log(prs))  # log likelihoods
            
            ind = np.argmax(L)  # find max likelihood exponent
            aa = exlist[ind]
            
            # Make sample-size-matched surrogate data sets randomly drawn from the best fit truncated power law
            # Generate uniform random numbers and transform to power-law distribution
            u = np.random.rand(nrep, nz)
            if aa == 1.0:
                # Special case: uniform in log space
                sz = np.exp(u * (np.log(MM) - np.log(mm)) + np.log(mm))
            else:
                sz = ((u * (MM**(1 - aa) - mm**(1 - aa)) + mm**(1 - aa)) ** (1 / (1 - aa)))
            sz = np.sort(sz, axis=1)  # Sort each row
            
            # Resampled real data CDF: 10 points per decade logarithmically spaced
            z_sorted = np.sort(z)
            cdf_real = np.arange(1, nz + 1) / nz
            cprob, rlz = resample_logspace(cdf_real, np.log10(z_sorted), 10)
            ncomp = len(cprob)
            rz = 10**rlz
            
            # Find range of CDF variation across surrogate data sets
            q = np.zeros((2, ncomp))
            for s in range(ncomp):
                # Find indices in surrogate data closest to rz[s]
                surind = np.argmin(np.abs(sz - rz[s]), axis=1)
                # Get CDF values at these indices
                sur_cdf = (surind + 1) / nz  # +1 because MATLAB is 1-indexed
                q[0, s] = np.min(sur_cdf)
                q[1, s] = np.max(sur_cdf)
            
            # Fraction of range (in decades) of real CDF that falls within bounds of surrogate CDFs
            fracgood = np.mean((cprob > q[0, :]) & (cprob < q[1, :]))
            
            if plotflag:
                # Show CDFs
                try:
                    import matplotlib.pyplot as plt
                    plt.figure(50)
                    plt.subplot(211)
                    plt.semilogx(rz, q[0, :], 'm', label='5% surrogates')  # 5% for surrogates
                    plt.semilogx(rz, q[1, :], 'm', label='95% surrogates')  # 95% for surrogates
                    plt.semilogx(z_sorted, np.arange(1, nz + 1) / nz, 'b', label='data CDF')
                    plt.semilogx(rz, cprob, 'r', label='resampled data CDF')
                    plt.legend()
                    plt.grid(True)
                except ImportError:
                    pass
            
            if fracgood > GOFcrit:
                bestMM = MM
                bestmm = mm
                tau = aa
                bestrho = rho
                nsamp = nz
                bestfracgood = fracgood
                # print(f'rho={rho:.4f}, aa={aa:.4f}, fg={fracgood:.4f}, nsamp={nsamp}')
                
                # Show PDF and CDF plots
                if plotflag:
                    try:
                        import matplotlib.pyplot as plt
                        plt.figure(100)
                        
                        plt.subplot(221)
                        plt.loglog(plotbins_all, pdf_all, '.k', label='All data PDF')
                        n, _ = np.histogram(evsiz, bins=binv)
                        plt.loglog(plotbins, n / np.diff(binv) / nz, label='Filtered PDF')
                        n2, _ = np.histogram(z, bins=binv)
                        plt.loglog(plotbins, n2 / np.diff(binv) / nz, 'r', label='Fit range PDF')
                        fit_bins = plotbins[mmm:ns-1]
                        if aa == 1.0:
                            fit_pdf = 1.0 / (np.log(MM) - np.log(mm)) / fit_bins
                        else:
                            fit_pdf = ((aa - 1) / (mm**(1 - aa) - MM**(1 - aa))) * (fit_bins ** -aa)
                        plt.loglog(fit_bins, fit_pdf, 'r', label='Fitted power-law')
                        plt.xlabel('Event Size')
                        plt.ylabel('Probability Density')
                        plt.legend()
                        plt.grid(True)
                        
                        # Real data CDF
                        datCDF = np.arange(1, nz + 1) / nz
                        
                        # Theory fit powerlaw CDF
                        z_sorted = np.sort(z)
                        if aa == 1.0:
                            refCDF = (np.log(z_sorted) - np.log(mm)) / (np.log(MM) - np.log(mm))
                        else:
                            refCDF = ((z_sorted**(1 - aa) - mm**(1 - aa)) / 
                                     (MM**(1 - aa) - mm**(1 - aa)))
                        
                        plt.subplot(222)
                        plt.semilogx(z_sorted, datCDF, label='Data CDF')
                        plt.semilogx(z_sorted, refCDF, 'r', label='Fitted CDF')
                        plt.xlabel('Event Size')
                        plt.ylabel('Cumulative Probability')
                        plt.legend()
                        plt.grid(True)
                        plt.tight_layout()
                    except ImportError:
                        pass
    
    # Calculate 5-95% of pdf probabilities for surrogate data
    # This is used to provide visual estimate of uncertainty in pdf plots of powerlaws
    if bestmm + bestMM != 0 and 10 * np.log10(bestMM / bestmm) >= 3:
        # Generate surrogate data from best fit
        u = np.random.rand(nrep, nsamp)
        if tau == 1.0:
            sz = np.exp(u * (np.log(bestMM) - np.log(bestmm)) + np.log(bestmm))
        else:
            sz = ((u * (bestMM**(1 - tau) - bestmm**(1 - tau)) + bestmm**(1 - tau)) ** 
                  (1 / (1 - tau)))
        
        # New bin vector for best fit range
        binvs = np.logspace(np.log10(bestmm), np.log10(bestMM), 
                           int(10 * np.log10(bestMM / bestmm)))
        # Histogram each surrogate dataset
        probden = np.zeros((nrep, len(binvs) - 1))
        for i in range(nrep):
            cnts, _ = np.histogram(sz[i, :], bins=binvs)
            probden[i, :] = cnts / np.diff(binvs) / nsamp
        
        # Calculate 1st and 99th percentiles (equivalent to 0.01 and 0.99 quantiles)
        surbound = np.percentile(probden, [1, 99], axis=0)
    else:
        surbound = None
        binvs = None
    
    return (tau, bestrho, bestmm, bestMM, nsamp, bestfracgood, bestuncert, 
            surbound, binvs)

def resample_logspace(y_values, x_values_log, points_per_decade):
    """
    Resample data at logarithmically spaced points (equivalent to MATLAB resample function)
    
    Parameters:
    -----------
    y_values : array-like
        Y-values to resample (e.g., CDF values)
    x_values_log : array-like
        X-values in log space (e.g., log10 of data)
    points_per_decade : int
        Number of points per decade in log space
        
    Returns:
    --------
    y_resampled : ndarray
        Resampled y-values
    x_resampled_log : ndarray
        Resampled x-values in log space
    """
    x_values_log = np.asarray(x_values_log)
    y_values = np.asarray(y_values)
    
    # Create logarithmically spaced points
    log_min = np.min(x_values_log)
    log_max = np.max(x_values_log)
    n_decades = log_max - log_min
    n_points = int(np.ceil(n_decades * points_per_decade)) + 1
    x_resampled_log = np.linspace(log_min, log_max, n_points)
    
    # Interpolate y-values at these points
    # Use linear interpolation, with constant extrapolation for edge cases
    interp_func = interp1d(x_values_log, y_values, kind='linear', 
                          bounds_error=False, fill_value=(y_values[0], y_values[-1]))
    y_resampled = interp_func(x_resampled_log)
    
    return y_resampled, x_resampled_log
