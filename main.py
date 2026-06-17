"""
Main Orchestrator for Scale-Free Analysis

This script utilizes multiprocessing to rapidly analyze raw accelerometry data,
extracting scale-free properties and generating timeline visualizations.
"""

import os
import glob
import time
import argparse
import pandas as pd
import numpy as np
import re
from concurrent.futures import ProcessPoolExecutor, as_completed
from datetime import datetime

import cProfile

# Import from our custom modules
from data_io import load_data, parse_filename, parse_gt3x_metadata, get_file_metadata
from signal_processing2 import butter_gravity_filter, detect_behavioral_events
from scale_free_math2 import analyze_scale_free_events
from plotting import generate_all_plots_concurrently, plot_hourly_power_law

def process_single_file(filepath: str, output_dir: str, plot_hourly: bool = False, window_duration_sec: int = 3600, window_step_sec: int = 3600, preloaded_meta: dict = None) -> list:
    """
    Worker function executed in parallel. Processes a single raw CSV file completely.
    
    Returns:
    --------
    list of dicts
        A list containing the results for each hour analyzed in this file.
    """
    try:
        if preloaded_meta is not None:
            metadata = preloaded_meta
        else:
            metadata = get_file_metadata(filepath)
        
        # Load and filter
        df, fs = load_data(filepath)
        if df.empty:
            return []
        df = butter_gravity_filter(df, sampling_rate=fs)
        
        # Add a heavily smoothed column specifically for critical thresholding
        smoothing_samples = 6
        #smoothing_samples = max(1, int((1/5) * fs))
        df['dynamic_mag_butter_smooth'] = df['dynamic_mag_butter'].rolling(
            window=smoothing_samples, center=True, min_periods=1
        ).mean()
        
        # Calculate global threshold using Median + k * MAD
        signal = df['dynamic_mag_butter_smooth']
        signal_median = np.nanmedian(signal)
        #signal_mad = np.nanmedian(np.abs(signal - signal_median))
        
        #k_multiplier = 0.5  # Adjust this depending on your desired sensitivity
        global_threshold = signal_median #+ (k_multiplier * signal_mad)
        
        total_duration = df['time'].max()
        n_hours = int(np.ceil(total_duration / window_duration_sec))
        
        file_results = []

        current_start = 0
        
        # Iterate through hours
        while current_start < total_duration:
            hour_start = current_start
            hour_end = min(current_start + window_duration_sec, total_duration)
            
            # Slice the hour
            mask = (df['time'] >= hour_start) & (df['time'] < hour_end)
            df_hour = df.loc[mask].copy()
            
            if df_hour.empty:
                continue
                
            # Detect events (includes our new min_duration and min_auc safeguards)
            events_df = detect_behavioral_events(
                df_hour, 
                median_threshold=global_threshold, 
                #use_butter=True, 
                sampling_rate=fs,
                min_duration=0,
                min_auc=0
            )
            
            n_events = len(events_df)
            
            # Dictionary template mapping perfectly to your requested schema
            hour_record = {
                'subject_id': metadata['subject_id'],
                'arm': metadata['arm'],
                'timepoint_number': metadata['timepoint'],
                # Store start time in hours. plotting.py adds 0.5, placing the point perfectly in the middle.
                'hour_number': hour_start / 3600.0,
                'n_events': n_events,
                'tau': np.nan,
                'power_law_range': np.nan,
                'lower_cutoff': np.nan,
                'upper_cutoff': np.nan,
                'n_samples': 0,
                'goodness_of_fit': np.nan,
                'is_scale_free': False,
                'success': False,
                'threshold': global_threshold
            }
            
            if n_events >= 20:
                # Perform the math
                sf_results = analyze_scale_free_events(events_df, event_size_column='auc', plotflag=False, verbose=False)
                
                # Update record if successful
                hour_record.update({
                    'success': sf_results.get('success', False),
                    'tau': sf_results.get('tau', np.nan),
                    'power_law_range': sf_results.get('power_law_range', np.nan),
                    'lower_cutoff': sf_results.get('lower_cutoff', np.nan),
                    'upper_cutoff': sf_results.get('upper_cutoff', np.nan),
                    'n_samples': sf_results.get('n_samples', 0),
                    'goodness_of_fit': sf_results.get('goodness_of_fit', np.nan),
                    'is_scale_free': sf_results.get('is_scale_free', False)
                })

            # If in targeted mode, generate the log-log hourly plots!
            if plot_hourly:
                plot_hourly_power_law(events_df, hour_record, output_dir)

            # Step forward by the defined interval
            current_start += window_step_sec

            file_results.append(hour_record)
            
        return file_results
        
    except Exception as e:
        print(f"Failed processing {filepath}: {str(e)}")
        return []

def process_single_file_profiled(filepath, output_dir, plot_hourly, window_duration_sec, window_step_sec):
    """
    Wraps the main processing function in a cProfile instance so we can see inside the multiprocessing workers.
    """
    profiler = cProfile.Profile()
    profiler.enable()
    
    # Run your actual math function
    result = process_single_file(filepath, output_dir, plot_hourly, window_duration_sec, window_step_sec)
    
    profiler.disable()
    
    # Save a unique profile file for this specific file and worker process
    filename = os.path.basename(filepath).replace('.csv', '')
    pid = os.getpid()
    prof_name = f'worker_pid{pid}_{filename}.prof'
    
    profiler.dump_stats(prof_name)
    
    return result

def parse_subject_args(subject_arg):
    if not subject_arg:
        return None
    
    subjects = []
    parts = subject_arg.split(',')
    for part in parts:
        part = part.strip()
        if '-' in part and not part.startswith('-'):
            try:
                start, end = part.split('-', 1)
                start = start.strip()
                end = end.strip()
                
                match_start = re.match(r'^(.*?)(\d+)$', start)
                match_end = re.match(r'^(.*?)(\d+)$', end)
                
                if match_start and match_end and match_start.group(1) == match_end.group(1):
                    prefix = match_start.group(1)
                    s_num = match_start.group(2)
                    e_num = match_end.group(2)
                    width = len(s_num)
                    for i in range(int(s_num), int(e_num) + 1):
                        subjects.append(f"{prefix}{i:0{width}d}")
                else:
                    subjects.append(part)
            except ValueError:
                subjects.append(part)
        else:
            subjects.append(part)
    return subjects

def main():
    from math import floor
    parser = argparse.ArgumentParser(description="Multi-threaded Scale-Free Analysis Pipeline")
    parser.add_argument('--data_dir', type=str, default="", help="Path to raw CSV files")
    parser.add_argument('--workers', type=int, default=os.cpu_count() - 2, help="Number of CPU cores to use")
    parser.add_argument('--subject', type=str, default=None, help="Target specific subjects (e.g., PMC8442937_001, PMC8442937_002 or 001-005)")
    parser.add_argument('--timepoint', type=int, default=None, help="Target a specific timepoint (e.g., 0)")
    parser.add_argument('--step_size', type=int, default=60, help="Sliding window step size in minutes. Default: 60 (sequential 1hr buckets).")
    parser.add_argument('--plot_hourly', type=float, default=0.0, help="Probability (0.0 to 1.0) to generate hourly log-log PDF plots for a subject")
    parser.add_argument('--output_dir', type=str, default=None, help="Directory to save the output files. Defaults to a timestamped folder in ../../output")
    parser.add_argument('--limb', type=str, default=None, help="Target a specific limb or arm (e.g., LUE, RUE)")
    parser.add_argument('--use_metadata_csv', action='store_true', help="Use pre-generated metadata CSV for gt3x files instead of directory traversal")
    parser.add_argument('--metadata_csv', type=str, default='/media/bs007r/als_gupta_2023/gt3x_metadata.csv', help="Path to the metadata CSV file")
    parser.add_argument('--timepoints_as_days', action='store_true', help="Calculate timepoint as number of days since the first recording for each subject")
    args = parser.parse_args()

    # Create output directory
    if args.output_dir:
        output_dir = args.output_dir
    else:
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        output_dir = os.path.join('../..', 'output', timestamp)
    os.makedirs(output_dir, exist_ok=True)
    
    preloaded_metadata_dict = {}
    
    if args.use_metadata_csv:
        print(f"Loading metadata from {args.metadata_csv}...")
        meta_df = pd.read_csv(args.metadata_csv)
        
        # Calculate timepoints for all files based on chronological record date per subject
        meta_df['record date'] = pd.to_datetime(meta_df['record date'])
        meta_df = meta_df.dropna(subset=['record date']).sort_values(['subject_id', 'record date'])
        
        if args.timepoints_as_days:
            # Calculate days since first recording per subject
            first_dates = meta_df.groupby('subject_id')['record date'].transform('min')
            meta_df['timepoint'] = (meta_df['record date'] - first_dates).dt.days.astype(int)
        else:
            # Use dense rank on Year-Month so LUE and RUE files recorded simultaneously get the SAME timepoint
            meta_df['session_month'] = meta_df['record date'].dt.strftime('%Y-%m')
            meta_df['timepoint'] = meta_df.groupby('subject_id')['session_month'].rank(method='dense').astype(int)
        
        # Filter by subject
        if args.subject is not None:
            target_subjects = parse_subject_args(args.subject)
            mask = pd.Series(False, index=meta_df.index)
            for subj in target_subjects:
                mask |= meta_df['subject_id'].astype(str).str.contains(subj, case=False)
            meta_df = meta_df[mask]
            
        # Filter by limb
        if args.limb is not None:
            meta_df = meta_df[meta_df['unit'].astype(str) == args.limb]
            
        # Filter by timepoint
        if args.timepoint is not None:
            filtered_dfs = []
            for subj, group in meta_df.groupby('subject_id'):
                unique_tps = sorted(group['timepoint'].unique())
                if not unique_tps:
                    continue
                tp_idx = args.timepoint - 1 if args.timepoint > 0 else args.timepoint
                try:
                    target_tp = unique_tps[tp_idx]
                    filtered_dfs.append(group[group['timepoint'] == target_tp])
                except IndexError:
                    pass
            if filtered_dfs:
                meta_df = pd.concat(filtered_dfs)
            else:
                meta_df = pd.DataFrame(columns=meta_df.columns)
                
        file_names = meta_df['file path'].tolist()
        
        # Prepare the quick-lookup dictionary for process_single_file
        for _, row in meta_df.iterrows():
            preloaded_metadata_dict[row['file path']] = {
                'subject_id': str(row['subject_id']),
                'timepoint': int(row['timepoint']),
                'arm': str(row['unit']),
                'limb': 'Unknown',
                'sampling_rate': 30
            }
            
        if not file_names:
            print("No files matched your filtering parameters in the CSV.")
            return
    else:
        if not args.data_dir:
            print("Error: --data_dir is required when not using --use_metadata_csv")
            return
            
        file_names = glob.glob(os.path.join(args.data_dir, '*_30Hz.csv'))
        if file_names:        
            # Do filtering by command-line arguments for CSV files
            target_subjects = parse_subject_args(args.subject) if args.subject else None
            
            # First pass: collect metadata and filter by subject/limb
            csv_metadata = []
            for f in file_names:
                meta = parse_filename(f)
                if target_subjects is not None and not any(subj.lower() in meta['subject_id'].lower() for subj in target_subjects):
                    continue
                if args.limb is not None and args.limb not in [meta.get('arm'), meta.get('limb')]:
                    continue
                csv_metadata.append({'file': f, 'meta': meta})
            
            # Second pass: handle timepoint logic dynamically per subject
            from collections import defaultdict
            subj_groups = defaultdict(list)
            for item in csv_metadata:
                subj_groups[item['meta']['subject_id']].append(item)
            
            filtered_files = []
            for subj, items in subj_groups.items():
                unique_tps = sorted(list(set(item['meta']['timepoint'] for item in items)))
                if not unique_tps:
                    continue
                    
                if args.timepoint is not None:
                    tp_idx = args.timepoint - 1 if args.timepoint > 0 else args.timepoint
                    try:
                        target_tp = unique_tps[tp_idx]
                        for item in items:
                            if item['meta']['timepoint'] == target_tp:
                                item['meta']['timepoint'] = tp_idx + 1 if tp_idx >= 0 else len(unique_tps) + tp_idx + 1
                                filtered_files.append(item['file'])
                                preloaded_metadata_dict[item['file']] = item['meta']
                    except IndexError:
                        pass
                else:
                    for item in items:
                        idx = unique_tps.index(item['meta']['timepoint'])
                        item['meta']['timepoint'] = idx + 1
                        filtered_files.append(item['file'])
                        preloaded_metadata_dict[item['file']] = item['meta']
                        
            file_names = filtered_files
            
            if not file_names:
                print("No files matched your filtering parameters.")
                return
        else: # no CSV files found
            print(f"No valid CSV files found in {args.data_dir}, assuming .gt3x")
            folder_names = glob.glob(os.path.join(args.data_dir, 'Patient_*'))
            if not folder_names:
                print(f"No valid folders found in {args.data_dir}")
                return
            
            if args.subject is not None:
                print(f"Filtering for Subjects: {args.subject}")
                target_subjects = parse_subject_args(args.subject)
                folder_names = [f for f in folder_names if any(subj.lower() in os.path.basename(f).lower() for subj in target_subjects)]
                
            monthToNum = {
                "Jan": "01", "Feb": "02", "Mar": "03", "Apr": "04",
                "May": "05", "Jun": "06", "Jul": "07", "Aug": "08",
                "Sep": "09", "Oct": "10", "Nov": "11", "Dec": "12"
            }
            
            def get_sort_key(path):
                basename = os.path.basename(path)
                if len(basename) >= 7 and basename[:3] in monthToNum:
                    return basename[3:7] + monthToNum[basename[:3]]
                return basename
                
            for folder_name in folder_names:
                this_time = [d for d in glob.glob(os.path.join(folder_name, '*')) if os.path.isdir(d)] # e.g. Apr2018
                this_time.sort(key=get_sort_key)
                
                if not this_time:
                    continue
                    
                target_folders = []
                if args.timepoint is not None:
                    tp_idx = args.timepoint - 1 if args.timepoint > 0 else args.timepoint
                    if tp_idx >= len(this_time) or tp_idx < -len(this_time):
                        print(f"Timepoint {args.timepoint} out of range for {os.path.basename(folder_name)}")
                    else:
                        tp_num = tp_idx + 1 if tp_idx >= 0 else len(this_time) + tp_idx + 1
                        target_folders.append((this_time[tp_idx], tp_num))
                else:
                    for idx, folder in enumerate(this_time):
                        target_folders.append((folder, idx + 1))
                        
                for target_folder, tp_num in target_folders:
                    gt3x_files = glob.glob(os.path.join(target_folder, '*.gt3x'))
                    for f in gt3x_files:
                        meta = get_file_metadata(f)
                        meta['timepoint'] = tp_num
                        
                        if args.limb is not None and args.limb not in [meta.get('arm'), meta.get('limb')]:
                            continue
                            
                        file_names.append(f)
                        preloaded_metadata_dict[f] = meta
                
            if not file_names:
                print("No files matched your filtering parameters.")
                return
    

    print("="*80)
    print(f"Starting Multi-threaded Analysis on {len(file_names)} files using {args.workers} workers...")
    print(f"Using a 1-hour window stepping every {args.step_size} minutes.")
    print("="*80)
    
    start_time = time.time()
    all_results = []

    # Convert minutes to seconds for the function
    step_size_sec = args.step_size * 60
    
    # 1. MATHEMATICAL ANALYSIS PHASE (Concurrent)
    subject_plot_decisions = {}

    with ProcessPoolExecutor(max_workers=args.workers) as executor:
        future_to_file = {}
        for filepath in file_names:
            metadata = preloaded_metadata_dict.get(filepath)
            if not metadata:
                metadata = get_file_metadata(filepath)
            subj_id = metadata['subject_id']
                
            if subj_id not in subject_plot_decisions:
                subject_plot_decisions[subj_id] = np.random.rand() < args.plot_hourly
            
            future_to_file[executor.submit(
                process_single_file, 
                filepath, 
                output_dir, 
                subject_plot_decisions[subj_id],
                3600,            # window_duration_sec
                step_size_sec,   # window_step_sec
                metadata
            )] = filepath
        
        # Gather results as they complete
        for j, future in enumerate(as_completed(future_to_file), 1):
            filepath = future_to_file[future]
            try:
                file_hourly_data = future.result()
                all_results.extend(file_hourly_data)
                print(f"[{j}/{len(file_names)}] Completed math for {os.path.basename(filepath)}")
            except Exception as e:
                print(f"[{j}/{len(file_names)}] Exception in {os.path.basename(filepath)}: {e}")


    # 2. DATA AGGREGATION PHASE
    if not all_results:
        print("Error: No results were generated.")
        return
        
    # Enforce exact column order as requested
    columns_order = [
        'subject_id', 'arm', 'timepoint_number', 'hour_number', 'n_events', 
        'tau', 'power_law_range', 'lower_cutoff', 'upper_cutoff', 'n_samples', 
        'goodness_of_fit', 'is_scale_free', 'success', 'threshold'
    ]
    
    df_final = pd.DataFrame(all_results)[columns_order]
    
    # Save the master CSV
    csv_out_path = os.path.join(output_dir, 'master_scale_free_results.csv')
    df_final.to_csv(csv_out_path, index=False)
    print(f"\nMath phase complete! Results saved to {csv_out_path}")
    
    # Create a lookup dictionary so plotting.py doesn't have to re-traverse the directory
    file_lookup = {}
    for f in file_names:
        meta = preloaded_metadata_dict.get(f)
        if not meta:
            meta = get_file_metadata(f)
        file_lookup[f] = meta

    # 3. PLOTTING PHASE (Concurrent)
    actual_data_dir = args.data_dir if args.data_dir else os.path.dirname(args.metadata_csv)
    generate_all_plots_concurrently(df_final, actual_data_dir, output_dir, max_workers=max(1, floor(args.workers/2)), file_lookup=file_lookup)
    
    total_time = time.time() - start_time
    print("="*80)
    print(f"FULL PIPELINE COMPLETE in {total_time/60:.2f} minutes.")
    print("="*80)

if __name__ == '__main__':
    #profiler = cProfile.Profile()
    #profiler.enable()
    main()
    #profiler.disable()
    #profiler.dump_stats('pipeline.prof')
    #print("Profiling data saved to pipeline.prof")