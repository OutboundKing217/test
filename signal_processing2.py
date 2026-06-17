"""
Signal Processing Module (V2)

Handles filtering raw accelerometry data to remove gravity and grouping
continuous data into discrete behavioral events. Uses chunk-based filtering
to preserve flatlines and prevent filter artifacts.
"""

import numpy as np
import pandas as pd
from scipy.signal import butter, filtfilt
from typing import Optional

def detect_flatlines(signal, flat_threshold=0.001, min_duration_samples=8):
    """
    Detect flat line regions using an expanding window approach.
    When a rolling window of min_duration_samples has a range (max - min)
    <= flat_threshold, the window is expanded until the range exceeds the threshold.

    Returns a boolean mask: True = flat line, False = active signal
    """
    signal_np = np.asarray(signal)
    n = len(signal_np)
    flat_mask = np.zeros(n, dtype=bool)
    
    if min_duration_samples < 2:
        min_duration_samples = 2
        
    if n < min_duration_samples:
        return flat_mask

    # Pre-filter: consecutive differences MUST be <= flat_threshold
    # If consecutive diff > threshold, that step can NEVER be inside a flat region
    diffs = np.abs(np.diff(signal_np))
    valid_steps = (diffs <= flat_threshold).astype(int)
    
    # Find contiguous regions where consecutive steps are within threshold
    step_diff = np.diff(np.insert(valid_steps, 0, 0))
    starts = np.where(step_diff == 1)[0]
    ends = np.where(step_diff == -1)[0]
    
    if len(starts) > len(ends):
        ends = np.append(ends, len(valid_steps))
        
    for st, en in zip(starts, ends):
        # The candidate chunk involves points from `st` to `en` inclusive.
        chunk_len_samples = (en - st) + 1
        if chunk_len_samples < min_duration_samples:
            continue
            
        i = st
        while i <= en - min_duration_samples + 1:
            window = signal_np[i:i + min_duration_samples]
            w_min = np.min(window)
            w_max = np.max(window)
            
            if w_max - w_min <= flat_threshold:
                # Found a valid starting window, expand it greedily
                end_idx = i + min_duration_samples - 1
                j = end_idx + 1
                
                while j <= en:
                    val = signal_np[j]
                    if val < w_min:
                        w_min = val
                    elif val > w_max:
                        w_max = val
                        
                    if w_max - w_min <= flat_threshold:
                        end_idx = j
                        j += 1
                    else:
                        break
                        
                # Mark the expanded region as flat
                flat_mask[i:end_idx + 1] = True
                
                # Advance `i` to the next possible start that isn't fully enclosed
                i = max(i + 1, j - min_duration_samples + 1)
            else:
                i += 1

    return flat_mask


def get_active_chunks(flat_mask):
    """
    Extract contiguous active (non-flat) segments from mask.

    Returns list of (start, end) index tuples.
    """
    active = ~flat_mask
    chunks = []
    in_chunk = False

    for i, a in enumerate(active):
        if a and not in_chunk:
            start = i
            in_chunk = True
        elif not a and in_chunk:
            chunks.append((start, i))
            in_chunk = False
    if in_chunk:
        chunks.append((start, len(active)))

    return chunks


def filter_chunk(chunk, b, a, padlen=None):
    """
    Filter a single active chunk with mirror padding.
    Returns NaN array if chunk is too short to filter stably.
    """
    if padlen is None:
        padlen = 3 * (max(len(b), len(a)) - 1)

    if len(chunk) < padlen * 2:
        return np.full(len(chunk), np.nan)

    return filtfilt(b, a, chunk, padtype='odd', padlen=padlen)


def butter_gravity_filter(
    df: pd.DataFrame, 
    cutoff: float = 0.3, 
    order: int = 4, 
    sampling_rate: float = 30,
    flat_threshold: float = 0.001,
    min_flat_duration_sec: float = 0.5
) -> pd.DataFrame:
    """
    Applies a low-pass Butterworth filter to isolate and remove gravity 
    from the raw accelerometer signal.
    Processes active chunks separately to avoid filtering artifacts across flatlines.
    Too short active chunks are replaced with NaN.
    Flat regions are assigned the raw dynamic magnitude.
    """
    # Calculate raw dynamic magnitude
    raw_mag = np.sqrt(df[0]**2 + df[1]**2 + df[2]**2)
    raw_dynamic_mag = raw_mag - raw_mag.mean()
    
    # 1. Detect flatlines based on raw_dynamic_mag
    signal_np = raw_dynamic_mag.values
    min_flat_samples = max(2, int(min_flat_duration_sec * sampling_rate))
    
    flat_mask = detect_flatlines(
        signal_np,
        flat_threshold=flat_threshold,
        min_duration_samples=min_flat_samples
    )
    
    # Calculate Nyquist frequency
    nyq = 0.5 * sampling_rate
    normal_cutoff = cutoff / nyq
    
    # Design Butterworth filter
    b, a = butter(order, normal_cutoff, btype='low')
    padlen = 3 * (max(len(b), len(a)) - 1)
    
    # Initialize output columns
    dyn_x = np.full(len(df), np.nan)
    dyn_y = np.full(len(df), np.nan)
    dyn_z = np.full(len(df), np.nan)
    dyn_mag = np.full(len(df), np.nan)
    
    # Get active chunks and process each
    chunks = get_active_chunks(flat_mask)
    
    for start, end in chunks:
        chunk_x = df[0].values[start:end]
        chunk_y = df[1].values[start:end]
        chunk_z = df[2].values[start:end]
        
        grav_x = filter_chunk(chunk_x, b, a, padlen=padlen)
        grav_y = filter_chunk(chunk_y, b, a, padlen=padlen)
        grav_z = filter_chunk(chunk_z, b, a, padlen=padlen)
        
        # Calculate dynamic acceleration
        dyn_x[start:end] = chunk_x - grav_x
        dyn_y[start:end] = chunk_y - grav_y
        dyn_z[start:end] = chunk_z - grav_z
        
        # Calculate vector magnitude for this chunk
        dyn_mag[start:end] = np.sqrt(dyn_x[start:end]**2 + dyn_y[start:end]**2 + dyn_z[start:end]**2)

    # Stitch the flat regions back using the raw dynamic magnitude
    flat_indices = np.where(flat_mask)[0]
    dyn_mag[flat_indices] = raw_dynamic_mag.values[flat_indices]
    
    # Assign back to DataFrame
    df['dynamic_x_butter'] = dyn_x
    df['dynamic_y_butter'] = dyn_y
    df['dynamic_z_butter'] = dyn_z
    df['dynamic_mag_butter'] = dyn_mag
    
    return df

def detect_behavioral_events(
    df: pd.DataFrame, 
    column: str = 'dynamic_mag_butter_smooth',
    median_threshold: Optional[float] = None,
    sampling_rate: float = 30,
    min_duration: float = 0,
    min_auc: float = 0         # Minimum AUC to prevent log-space inflation
) -> pd.DataFrame:
    """
    Detects behavioral events by finding continuous segments where 
    acceleration magnitude exceeds a calculated threshold.
    
    Parameters:
    -----------
    df : pd.DataFrame
        Dataframe with dynamic acceleration and time columns.
    column : string
        coloum to use in the dataframe for dynamic magnitude
    median_threshold : float, optional
        Threshold value. If None, calculates the 50th percentile of the current window.
    sampling_rate : float, optional
        Sampling rate in Hz. Default is 30.
    min_duration : float, optional
        Minimum event duration in seconds. Filters out microscopic noise. Default: 0.1s.
    min_auc : float, optional
        Minimum Area Under Curve. Prevents lower-cutoff from anchoring too low,
        which causes power-law ranges > 7 decades. Default: 0.
        
    Returns:
    --------
    pd.DataFrame
        Dataframe containing isolated event metrics (start, end, duration, AUC).
    """
    # Select which dynamic magnitude column to use
    dynamic = df[column].values
        
    time = df['time'].values
    
    # Calculate threshold if not provided
    if median_threshold is None:
        # Use nanmedian to safely ignore 'too short' filtered chunks
        median_threshold = np.nanmedian(dynamic)
    
    # Boolean array of where signal is above threshold (NaNs evaluate to False, so all avalanches are cut off there)
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
