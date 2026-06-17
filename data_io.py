"""
Data Input/Output Module

Handles reading raw accelerometry CSV files and parsing metadata 
(Subject ID, Arm, Timepoint, Sampling Rate) directly from the filenames.
"""

import os
import re
import zipfile
import pandas as pd
from typing import Tuple, Dict, Union

def get_file_metadata(filepath: str) -> Dict[str, Union[str, int]]:
    """
    Unified metadata extractor handling both .gt3x internal parameters and .csv filename parsing.
    """
    if filepath.lower().endswith('.gt3x'):
        gt3x_meta = parse_gt3x_metadata(filepath)
        arm = gt3x_meta.get('Side', 'Unknown')
        limb_location = gt3x_meta.get('Limb', 'Unknown')
        if arm == 'Right':
            arm = 'RUE' if limb_location != 'Ankle' else 'RLE'
        elif arm == 'Left':
            arm = 'LUE' if limb_location != 'Ankle' else 'LLE'
            
        subject_id = gt3x_meta.get('Subject Name', 'Unknown')
        if subject_id.lower().startswith('patient_'):
            subject_id = subject_id[8:]
            
        return {
            'subject_id': subject_id,
            'timepoint': int(gt3x_meta.get('Timepoint', -1)),
            'arm': arm,
            'limb': gt3x_meta.get('Limb', 'Unknown'),
            'sampling_rate': int(float(gt3x_meta.get('Sample Rate', 30)))
        }
    else:
        meta = parse_filename(filepath)
        meta['limb'] = 'Unknown'
        return meta

def parse_filename(filepath: str) -> Dict[str, Union[str, int]]:
    """
    Extracts subject metadata and sampling rate from the standard naming convention.
    
    Expected filename format example: 
    'PMC12345_67890_TimePoint2_LUE_30Hz.csv'
    
    Parameters:
    -----------
    filepath : str
        The full path or filename of the CSV file.
        
    Returns:
    --------
    dict
        Dictionary containing:
        - 'subject_id' (str): e.g., 'PMC12345_67890'
        - 'timepoint' (int): e.g., 0, 1, 2
        - 'arm' (str): 'LUE' or 'RUE'
        - 'sampling_rate' (int): e.g., 30
    """
    filename = os.path.basename(filepath)
    if filename.endswith(".gt3x"):
        print(f"Warning: Attempted to parse .gt3x filename '{filename}'. Use get_file_metadata instead.")
        return {
            'subject_id': 'Unknown',
            'timepoint': -1,
            'arm': 'Unknown',
            'sampling_rate': 30
        }
    
    # Regex to capture: Prefix_ID, TimePoint, Arm, SamplingRate
    # Adjust this regex if your file naming convention differs slightly!
    pattern = r'(PMC\d+)_(\d+)_TimePoint(\d+)_(LUE|RUE)_(\d+)Hz'
    match = re.search(pattern, filename)
    
    if match:
        prefix = match.group(1)
        subj_num = match.group(2)
        
        return {
            'subject_id': f"{prefix}_{subj_num}",
            'timepoint': int(match.group(3)),
            'arm': match.group(4),
            'sampling_rate': int(match.group(5))
        }
    else:
        # Fallback defaults if the filename doesn't match the strict pattern
        print(f"Warning: Could not fully parse filename '{filename}'. Using fallback values.")
        return {
            'subject_id': 'Unknown',
            'timepoint': -1,
            'arm': 'Unknown',
            'sampling_rate': 30
        }
    
def parse_gt3x_metadata(filepath: str) -> Dict[str, str]:
    """
    Extracts metadata from a .gt3x file's info.txt
    """
    metadata = {}
    with zipfile.ZipFile(filepath) as archive:
        with archive.open("info.txt") as metadata_file:
            content = metadata_file.read().decode("utf-8")
            for line in content.strip().split('\n'):
                if ':' in line:
                    key, value = line.split(':', 1)
                    metadata[key.strip()] = value.strip()
    return metadata

def load_data(
    filepath: str, 
    expected_limb: str = None, 
    expected_side: str = None, 
    expected_subject: str = None
) -> Tuple[pd.DataFrame, int]:
    """
    Loads raw 3-axis accelerometer data from a CSV or .gt3x file into a Pandas DataFrame.
    
    Skips the first 10 rows (header/metadata) for CSV files and ensures the time column 
    is properly calculated based on the sampling rate. Supports filtering for .gt3x files.
    
    Parameters:
    -----------
    filepath : str
        Path to the CSV or .gt3x file.
    expected_limb : str, optional
        Filter for .gt3x file metadata 'Limb' (e.g. 'Ankle')
    expected_side : str, optional
        Filter for .gt3x file metadata 'Side' (e.g. 'Left')
    expected_subject : str, optional
        Filter for .gt3x file metadata 'Subject Name' (e.g. 'patient_1005')
    
    Returns:
    --------
    Tuple[pd.DataFrame, int]
        - df: DataFrame containing columns [0, 1, 2, 'time']
        - sampling_rate: The sampling rate in Hz extracted from the filename or metadata
    """
    if filepath.lower().endswith('.gt3x'):        
        metadata = parse_gt3x_metadata(filepath)
        
        if expected_limb and metadata.get('Limb') != expected_limb:
            return pd.DataFrame(), 0
        if expected_side and metadata.get('Side') != expected_side:
            return pd.DataFrame(), 0
        if expected_subject and metadata.get('Subject Name') != expected_subject:
            return pd.DataFrame(), 0
            
        sampling_rate = int(float(metadata.get('Sample Rate', 30)))

        from pygt3x.reader import FileReader

        with FileReader(filepath) as reader:
            df = reader.to_pandas()
            
        df = df.reset_index()
        df = df.rename(columns={'X': 0, 'Y': 1, 'Z': 2})
        df[[0, 1, 2]] = df[[0, 1, 2]].astype('float32')
        df = df.dropna(subset=[0, 1, 2]).reset_index(drop=True)
        df['time'] = df.index / sampling_rate
        
        return df, sampling_rate


    # CSV parsing

    # Parse sampling rate from filename
    metadata = get_file_metadata(filepath)
    sampling_rate = metadata['sampling_rate']
    
    # Load the data, skipping the first 11 rows of metadata
    df = pd.read_csv(filepath, skiprows=11, header=None, low_memory=False, dtype={0: 'float32', 1: 'float32', 2: 'float32'})
    
    # Ensure numeric data for accelerometer axes (forces errors to NaN)
    df[[0, 1, 2]] = df[[0, 1, 2]].apply(pd.to_numeric, errors='coerce', downcast='float')
        
    # Drop rows with NaN values in the XYZ columns and reset index
    df = df.dropna(subset=[0, 1, 2]).reset_index(drop=True)
    
    # Create the time column (index / sampling_rate = seconds)
    df['time'] = df.index / sampling_rate
    
    return df, sampling_rate

def load_downsampled_data(filepath: str, step: int = 30) -> Tuple[pd.DataFrame, float]:
    """
    Loads raw accelerometer data from a CSV file into a Pandas DataFrame,
    but only reads every `step`-th row. This vastly speeds up parsing and
    reduces memory usage when you want a lower effective sampling rate.
    
    Parameters:
    -----------
    filepath : str
        Path to the CSV file.
    step : int
        The step size for rows to read (e.g., 30 reads every 30th data row).
        
    Returns:
    --------
    Tuple[pd.DataFrame, float]
        - df: DataFrame containing downsampled columns [0, 1, 2, 'time']
        - effective_sr: The new effective sampling rate in Hz (e.g., 1.0)
    """
    if filepath.lower().endswith('.gt3x'):
        df, original_sr = load_data(filepath)
        if df.empty:
            return df, 0.0
        df = df.iloc[::step].reset_index(drop=True)
        effective_sr = original_sr / step
        df['time'] = df.index / effective_sr
        return df, effective_sr

    metadata = get_file_metadata(filepath)
    original_sampling_rate = metadata['sampling_rate']
    
    # A chunk_size that is a multiple of step ensures perfect interval alignment across chunks
    chunk_size = step * 10000
    chunks = []
    
    # Skip the 11 rows of metadata and load using the native C engine in chunks
    for chunk in pd.read_csv(
        filepath, 
        skiprows=11, 
        header=None, 
        low_memory=False, 
        chunksize=chunk_size,
        dtype={0: 'float32', 1: 'float32', 2: 'float32'}
    ):
        chunks.append(chunk.iloc[::step])
        
    df = pd.concat(chunks, ignore_index=True)
    
    df[[0, 1, 2]] = df[[0, 1, 2]].apply(pd.to_numeric, errors='coerce', downcast='float')
        
    df = df.dropna(subset=[0, 1, 2]).reset_index(drop=True)
    
    # Create the time column based on the new effective sampling rate
    effective_sr = original_sampling_rate / step
    df['time'] = df.index / effective_sr
    
    return df, effective_sr