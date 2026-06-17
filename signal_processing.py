"""
Signal Processing Module

Handles filtering raw accelerometry data to remove gravity and grouping
continuous data into discrete behavioral events.
"""

import numpy as np
import pandas as pd
from scipy.signal import butter, filtfilt
from typing import Optional

def butter_gravity_filter(df: pd.DataFrame, cutoff: float = 0.3, order: int = 4, sampling_rate: float = 30) -> pd.DataFrame:
    """
    Applies a low-pass Butterworth filter to isolate and remove gravity 
    from the raw accelerometer signal.
    
    Parameters:
    -----------
    df : pd.DataFrame
        Dataframe with raw accelerometer columns 0 (X), 1 (Y), 2 (Z).
    cutoff : float, optional
        Cutoff frequency in Hz. Frequencies below this are considered gravity. Default is 0.3 Hz.
    order : int, optional
        Filter order (higher = sharper cutoff). Default is 4.
    sampling_rate : float, optional
        Sampling rate of the data in Hz. Default is 30.
    
    Returns:
    --------
    pd.DataFrame
        Original dataframe appended with dynamic (gravity-removed) columns:
        - 'dynamic_x_butter', 'dynamic_y_butter', 'dynamic_z_butter'
        - 'dynamic_mag_butter' (overall magnitude of movement)
    """
    # Calculate Nyquist frequency
    nyq = 0.5 * sampling_rate
    normal_cutoff = cutoff / nyq
    
    # Design Butterworth filter
    b, a = butter(order, normal_cutoff, btype='low')
    
    # Apply filter to each axis
    gravity_x = filtfilt(b, a, df[0])
    gravity_y = filtfilt(b, a, df[1])
    gravity_z = filtfilt(b, a, df[2])
    
    # Subtract gravity to get dynamic acceleration
    df['dynamic_x_butter'] = df[0] - gravity_x
    df['dynamic_y_butter'] = df[1] - gravity_y
    df['dynamic_z_butter'] = df[2] - gravity_z
    
    # Calculate vector magnitude
    df['dynamic_mag_butter'] = np.sqrt(
        df['dynamic_x_butter']**2 + 
        df['dynamic_y_butter']**2 + 
        df['dynamic_z_butter']**2
    )
    
    return df

def detect_behavioral_events(
    df: pd.DataFrame, 
    median_threshold: Optional[float] = None, 
    use_butter: bool = True, 
    sampling_rate: float = 30,
    min_duration: float = 0,    # NEW: Minimum duration in seconds
    min_auc: float = 1e-3         # NEW: Minimum AUC to prevent log-space inflation
) -> pd.DataFrame:
    """
    Detects behavioral events by finding continuous segments where 
    acceleration magnitude exceeds a calculated threshold.
    
    Parameters:
    -----------
    df : pd.DataFrame
        Dataframe with dynamic acceleration and time columns.
    median_threshold : float, optional
        Threshold value. If None, calculates the 50th percentile of the current window.
    use_butter : bool, optional
        Use Butterworth filtered data. Default is True.
    sampling_rate : float, optional
        Sampling rate in Hz. Default is 30.
    min_duration : float, optional
        Minimum event duration in seconds. Filters out microscopic noise. Default: 0.1s.
    min_auc : float, optional
        Minimum Area Under Curve. Prevents lower-cutoff from anchoring too low,
        which causes power-law ranges > 7 decades. Default: 1e-3.
        
    Returns:
    --------
    pd.DataFrame
        Dataframe containing isolated event metrics (start, end, duration, AUC).
    """
    # Select which dynamic magnitude column to use
    if use_butter:
        if 'dynamic_mag_butter_smooth' in df.columns:
            dynamic = df['dynamic_mag_butter_smooth'].values
        else:
            dynamic = df['dynamic_mag_butter'].values
    else:
        dynamic = df['dynamic_mag'].values
        
    time = df['time'].values
    
    # Calculate threshold if not provided
    if median_threshold is None:
        median_threshold = np.median(dynamic)
    
    # Boolean array of where signal is above threshold
    above_threshold = dynamic > median_threshold
    
    # Find transitions (1 = start of event, -1 = end of event)
    threshold_diff = np.diff(above_threshold.astype(int))
    
    event_starts_idx = np.where(threshold_diff == 1)[0] + 1
    event_ends_idx = np.where(threshold_diff == -1)[0] + 1
    
    # Handle edge cases (signal starts or ends already above threshold)
    if above_threshold[0]:
        event_starts_idx = np.concatenate([[0], event_starts_idx])
    if above_threshold[-1]:
        event_ends_idx = np.concatenate([event_ends_idx, [len(dynamic) - 1]])
    
    events = []
    event_counter = 1
    
    # Pair starts and ends
    for start_idx, end_idx in zip(event_starts_idx, event_ends_idx):
        duration = time[end_idx] - time[start_idx]
        
        # Safeguard 1: Discard events that are physically too short (e.g., noise spikes)
        if duration < min_duration:
            continue
            
        event_segment = dynamic[start_idx:end_idx+1]
        event_time = time[start_idx:end_idx+1]
        
        # Calculate AUC (Area under the curve)
        auc = np.trapezoid(event_segment, event_time)
        
        # Safeguard 2: Discard microscopic AUC values to prevent log-math explosions
        if auc < min_auc:
            continue
            
        events.append({
            'event_num': event_counter,
            'start_time': time[start_idx],
            'end_time': time[end_idx],
            'duration': duration,
            'peak_amplitude': np.max(event_segment),
            'mean_amplitude': np.mean(event_segment),
            'auc': auc,
            'median_threshold': median_threshold
        })
        event_counter += 1
    
    return pd.DataFrame(events)