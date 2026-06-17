#!/usr/bin/env python3
"""
Extracts metadata from all .gt3x files in a directory to a CSV file.
"""

import os
import glob
import zipfile
import csv
import datetime
import argparse
import re

def parse_gt3x_metadata(filepath):
    """Parses info.txt from a .gt3x zip archive."""
    metadata = {}
    try:
        with zipfile.ZipFile(filepath) as archive:
            with archive.open("info.txt") as metadata_file:
                content = metadata_file.read().decode("utf-8")
                for line in content.strip().split('\n'):
                    if ':' in line:
                        key, value = line.split(':', 1)
                        metadata[key.strip()] = value.strip()
    except Exception as e:
        print(f"Error reading {filepath}: {e}")
    return metadata

def get_unit(side, limb):
    """Determines the unit (LUE, RUE, LLE, RLE) from side and limb."""
    if side == 'Right':
        return 'RUE' if limb != 'Ankle' else 'RLE'
    elif side == 'Left':
        return 'LUE' if limb != 'Ankle' else 'LLE'
    return 'Unknown'

def ticks_to_datetime(ticks_str):
    """Converts .NET ticks to a readable date/time string."""
    try:
        ticks = int(ticks_str)
        if ticks == 0:
            return ""
        # 1 tick = 100 nanoseconds = 0.1 microseconds
        # .NET epoch is Jan 1, 0001
        dt = datetime.datetime(1, 1, 1) + datetime.timedelta(microseconds=ticks / 10)
        return dt.strftime('%Y-%m-%d %H:%M:%S')
    except (ValueError, TypeError, OverflowError):
        return ""

def main():
    parser = argparse.ArgumentParser(description="Extract gt3x metadata to CSV")
    parser.add_argument('--input_dir', type=str, required=True, help="Directory containing .gt3x files")
    parser.add_argument('--output_csv', type=str, default="gt3x_metadata.csv", help="Output CSV file path")
    args = parser.parse_args()

    # Search for all .gt3x files recursively
    gt3x_files = glob.glob(os.path.join(args.input_dir, '**', '*.gt3x'), recursive=True)
    
    total_files = len(gt3x_files)
    if not gt3x_files:
        print(f"No .gt3x files found in {args.input_dir}. Exiting.")
        return

    print(f"Found {total_files} .gt3x files. Processing...")
    
    results = []
    for i, filepath in enumerate(gt3x_files, 1):
        # Add a simple progress indicator
        print(f"  [{i}/{total_files}] Processing: {os.path.basename(filepath)}", end='\r')
        meta = parse_gt3x_metadata(filepath)
        
        # Normalize subject ID to be consistent with other project scripts
        subject_id = meta.get('Subject Name', 'Unknown')
        if subject_id.lower().startswith('patient_'):
            subject_id = subject_id[8:]
        elif subject_id == 'Unknown':
            # Fallback: Extract from folder name (e.g., Patient_1428)
            match = re.search(r'Patient_([a-zA-Z0-9_]+)', filepath, re.IGNORECASE)
            if match:
                subject_id = match.group(1)
                
        record_date = ticks_to_datetime(meta.get('Start Date', '0'))
        if not record_date:
            # Fallback 1: Date in filename like (2019-12-18)
            date_match = re.search(r'\((\d{4}-\d{2}-\d{2})\)', filepath)
            if date_match:
                record_date = date_match.group(1) + " 00:00:00"
            else:
                # Fallback 2: MonthYear folder like Dec2019
                folder_name = os.path.basename(os.path.dirname(filepath))
                monthToNum = {
                    "Jan": "01", "Feb": "02", "Mar": "03", "Apr": "04",
                    "May": "05", "Jun": "06", "Jul": "07", "Aug": "08",
                    "Sep": "09", "Oct": "10", "Nov": "11", "Dec": "12"
                }
                if len(folder_name) >= 7 and folder_name[:3] in monthToNum:
                    year = folder_name[3:7]
                    month = monthToNum[folder_name[:3]]
                    record_date = f"{year}-{month}-01 00:00:00"

        results.append({
            'subject_id': subject_id,
            'record date': record_date,
            'unit': get_unit(meta.get('Side', 'Unknown'), meta.get('Limb', 'Unknown')),
            'file path': os.path.abspath(filepath)
        })

    print() # Newline after progress indicator is done.

    # Write to CSV
    if results:
        with open(args.output_csv, 'w', newline='') as f:
            writer = csv.DictWriter(f, fieldnames=['subject_id', 'record date', 'unit', 'file path'])
            writer.writeheader()
            writer.writerows(results)
        print(f"Metadata extraction complete. Saved to {args.output_csv}")
    else:
        print("No files processed. Output CSV not created.")

if __name__ == "__main__":
    main()
