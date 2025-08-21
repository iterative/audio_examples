#!/usr/bin/env python3
"""
Parquet waveform processor that reads waveform data from a parquet file (local or S3),
calculates statistics for specified rows, and measures throughput.
"""

import argparse
import csv
import time
import os
from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import Tuple, Optional, Union
import logging
from pathlib import Path

import numpy as np
import pandas as pd
import pyarrow.parquet as pq
import s3fs
from tqdm import tqdm

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


def read_parquet_file(filepath: str, columns: list = None) -> Optional[pd.DataFrame]:
    """
    Read parquet file from local or S3
    
    Args:
        filepath: Path to parquet file (local path or s3://bucket/path/file.parquet)
        columns: List of columns to read (None for all)
        
    Returns:
        DataFrame or None if failed
    """
    try:
        if filepath.startswith('s3://'):
            # Read from S3
            fs = s3fs.S3FileSystem()
            df = pd.read_parquet(filepath, columns=columns, filesystem=fs)
        else:
            # Read from local filesystem
            df = pd.read_parquet(filepath, columns=columns)
        
        return df
        
    except Exception as e:
        logger.error(f"Failed to read parquet file {filepath}: {e}")
        return None


def get_file_size(filepath: str) -> int:
    """Get file size in bytes"""
    try:
        if filepath.startswith('s3://'):
            fs = s3fs.S3FileSystem()
            info = fs.info(filepath)
            return info.get('size', 0)
        else:
            return os.path.getsize(filepath)
    except:
        return 0


def process_waveform_row(
    row: pd.Series,
    waveform_column: str,
    identifier: str
) -> Optional[Tuple[str, float, int]]:
    """
    Process a single row containing waveform data.
    
    Returns:
        Tuple of (identifier, mean, length) or None if failed
    """
    try:
        waveform = row[waveform_column]
        
        # Handle different waveform formats
        if isinstance(waveform, (list, np.ndarray)):
            waveform_array = np.array(waveform)
        elif isinstance(waveform, bytes):
            # Assume it's serialized numpy array
            waveform_array = np.frombuffer(waveform, dtype=np.float32)
        else:
            logger.warning(f"Unexpected waveform type for {identifier}: {type(waveform)}")
            return None
        
        # Calculate statistics
        mean_value = float(np.mean(waveform_array))
        length = len(waveform_array)
        
        return (identifier, mean_value, length)
        
    except Exception as e:
        logger.error(f"Error processing row {identifier}: {e}")
        return None


def main():
    parser = argparse.ArgumentParser(
        description='Process waveform data from a parquet file and calculate statistics'
    )
    parser.add_argument('input_csv', help='CSV file with list of identifiers')
    parser.add_argument('parquet_file', help='Path to parquet file (local or s3://bucket/path/file.parquet)')
    parser.add_argument('waveform_column', help='Column name containing waveform data')
    parser.add_argument('output_file', help='Output file for results')
    parser.add_argument('--id-column', help='Column name in parquet for matching with CSV identifiers')
    parser.add_argument('--workers', type=int, default=4,
                        help='Number of parallel workers (default: 4)')
    parser.add_argument('--csv-column', type=int, default=0,
                        help='CSV column index containing identifiers (default: 0)')
    parser.add_argument('--limit', type=int, default=None,
                        help='Maximum number of rows to process (default: all)')
    
    args = parser.parse_args()
    
    # Read input CSV for identifiers
    identifiers = []
    with open(args.input_csv, 'r') as f:
        reader = csv.reader(f)
        for row in reader:
            if len(row) > args.csv_column:
                identifier = row[args.csv_column].strip()
                identifiers.append(identifier)
                if args.limit and len(identifiers) >= args.limit:
                    break
    
    if not identifiers:
        logger.error("No identifiers found in CSV")
        return
    
    logger.info(f"Processing {len(identifiers)} rows from parquet file")
    logger.info(f"Parquet file: {args.parquet_file}")
    logger.info(f"Waveform column: {args.waveform_column}")
    if args.id_column:
        logger.info(f"ID column in parquet: {args.id_column}")
    logger.info(f"Using {args.workers} parallel workers")
    
    # Read the parquet file
    start_time = time.time()
    
    # Determine columns to read
    columns_to_read = [args.waveform_column]
    if args.id_column:
        columns_to_read.append(args.id_column)
    
    logger.info("Reading parquet file...")
    df = read_parquet_file(args.parquet_file, columns=columns_to_read)
    
    if df is None or df.empty:
        logger.error("Failed to read parquet file or file is empty")
        return
    
    if args.waveform_column not in df.columns:
        logger.error(f"Column '{args.waveform_column}' not found. Available columns: {df.columns.tolist()}")
        return
    
    logger.info(f"Parquet file loaded with {len(df)} rows")
    
    # Get file size for throughput calculation
    file_size = get_file_size(args.parquet_file)
    file_size_mb = file_size / (1024 * 1024)
    
    # Filter dataframe if id_column is specified
    if args.id_column:
        if args.id_column not in df.columns:
            logger.error(f"ID column '{args.id_column}' not found. Available columns: {df.columns.tolist()}")
            return
        
        # Convert identifiers to appropriate type for matching
        df_filtered = df[df[args.id_column].astype(str).isin(identifiers)]
        logger.info(f"Filtered to {len(df_filtered)} matching rows")
    else:
        # Use row indices if no id_column specified
        # Map identifiers to integer indices if they look like numbers
        try:
            indices = [int(i) for i in identifiers if i.isdigit()]
            df_filtered = df.iloc[indices] if indices else df.iloc[:len(identifiers)]
        except:
            df_filtered = df.iloc[:len(identifiers)]
        logger.info(f"Using first {len(df_filtered)} rows")
    
    # Process rows in parallel
    results = []
    total_bytes = file_size_mb * (len(df_filtered) / len(df)) if len(df) > 0 else 0
    
    with ThreadPoolExecutor(max_workers=args.workers) as executor:
        # Submit all tasks
        futures = {}
        for idx, (df_idx, row) in enumerate(df_filtered.iterrows()):
            if args.id_column and args.id_column in row:
                identifier = str(row[args.id_column])
            else:
                identifier = identifiers[idx] if idx < len(identifiers) else str(df_idx)
            
            future = executor.submit(process_waveform_row, row, args.waveform_column, identifier)
            futures[future] = identifier
        
        # Process results as they complete
        with tqdm(total=len(futures), desc="Processing rows") as pbar:
            for future in as_completed(futures):
                result = future.result()
                if result:
                    results.append(result)
                pbar.update(1)
    
    elapsed_time = time.time() - start_time
    
    # Write results to output file
    with open(args.output_file, 'w') as f:
        f.write("identifier,mean,length\n")
        for identifier, mean_val, length in results:
            f.write(f"{identifier},{mean_val:.6f},{length}\n")
    
    # Calculate and display throughput statistics
    successful_rows = len(results)
    failed_rows = len(df_filtered) - successful_rows
    throughput_mbps = total_bytes / elapsed_time if elapsed_time > 0 else 0
    rows_per_second = successful_rows / elapsed_time if elapsed_time > 0 else 0
    
    logger.info("\n" + "="*50)
    logger.info("Processing Complete!")
    logger.info(f"Total rows processed: {successful_rows}/{len(df_filtered)}")
    logger.info(f"Failed rows: {failed_rows}")
    logger.info(f"Total time: {elapsed_time:.2f} seconds")
    logger.info(f"Data processed: {total_bytes:.2f} MB")
    logger.info(f"Throughput: {throughput_mbps:.2f} MB/s")
    logger.info(f"Rows per second: {rows_per_second:.2f}")
    logger.info(f"Results written to: {args.output_file}")


if __name__ == "__main__":
    main()