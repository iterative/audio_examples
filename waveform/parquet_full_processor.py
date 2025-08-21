#!/usr/bin/env python3
"""
Optimized parquet processor that reads entire parquet file and processes all waveforms
for maximum throughput.
"""

import argparse
import time
import os
from concurrent.futures import ThreadPoolExecutor, ProcessPoolExecutor, as_completed
from typing import Tuple, Optional, List
import logging
from multiprocessing import cpu_count

import numpy as np
import pandas as pd
import pyarrow.parquet as pq
import pyarrow as pa
import s3fs
from tqdm import tqdm

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


def process_waveform_batch(
    data: List[Tuple[any, any]],
    waveform_column: str,
    id_column: Optional[str] = None
) -> List[Tuple[str, float, int]]:
    """
    Process a batch of waveforms in a single worker.
    
    Args:
        data: List of (index/id, row_data) tuples
        waveform_column: Name of waveform column
        id_column: Name of ID column (optional)
        
    Returns:
        List of (identifier, mean, length) tuples
    """
    results = []
    
    for idx, row_data in data:
        try:
            if isinstance(row_data, dict):
                waveform = row_data[waveform_column]
                identifier = str(row_data.get(id_column, idx)) if id_column else str(idx)
            else:
                waveform = row_data
                identifier = str(idx)
            
            # Handle different waveform formats
            if isinstance(waveform, (list, np.ndarray)):
                waveform_array = np.array(waveform, dtype=np.float32)
            elif isinstance(waveform, bytes):
                waveform_array = np.frombuffer(waveform, dtype=np.float32)
            else:
                continue
            
            # Calculate statistics
            mean_value = float(np.mean(waveform_array))
            length = len(waveform_array)
            
            results.append((identifier, mean_value, length))
            
        except Exception as e:
            logger.debug(f"Error processing row {idx}: {e}")
            continue
    
    return results


def read_parquet_optimized(filepath: str, columns: List[str] = None) -> Optional[pa.Table]:
    """
    Read parquet file using PyArrow for better performance.
    
    Args:
        filepath: Path to parquet file (local or s3://)
        columns: List of columns to read
        
    Returns:
        PyArrow Table or None if failed
    """
    try:
        if filepath.startswith('s3://'):
            fs = s3fs.S3FileSystem()
            with fs.open(filepath, 'rb') as f:
                table = pq.read_table(f, columns=columns)
        else:
            table = pq.read_table(filepath, columns=columns)
        
        return table
        
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


def main():
    parser = argparse.ArgumentParser(
        description='Optimized processor for entire parquet file waveform data'
    )
    parser.add_argument('parquet_file', help='Path to parquet file (local or s3://bucket/path/file.parquet)')
    parser.add_argument('waveform_column', help='Column name containing waveform data')
    parser.add_argument('output_file', help='Output file for results')
    parser.add_argument('--id-column', help='Column name for row identifiers (optional)')
    parser.add_argument('--workers', type=int, default=None,
                        help='Number of parallel workers (default: CPU count)')
    parser.add_argument('--batch-size', type=int, default=1000,
                        help='Batch size for parallel processing (default: 1000)')
    parser.add_argument('--limit', type=int, default=None,
                        help='Maximum number of rows to process (default: all)')
    parser.add_argument('--use-processes', action='store_true',
                        help='Use process pool instead of thread pool (may be faster for CPU-intensive work)')
    
    args = parser.parse_args()
    
    if args.workers is None:
        args.workers = cpu_count()
    
    logger.info(f"Processing parquet file: {args.parquet_file}")
    logger.info(f"Waveform column: {args.waveform_column}")
    if args.id_column:
        logger.info(f"ID column: {args.id_column}")
    logger.info(f"Using {args.workers} workers with batch size {args.batch_size}")
    if args.use_processes:
        logger.info("Using process pool for parallel processing")
    
    # Read the parquet file
    start_time = time.time()
    
    # Determine columns to read
    columns_to_read = [args.waveform_column]
    if args.id_column:
        columns_to_read.append(args.id_column)
    
    logger.info("Reading parquet file...")
    read_start = time.time()
    
    # Use PyArrow for better performance
    table = read_parquet_optimized(args.parquet_file, columns_to_read)
    
    if table is None:
        logger.error("Failed to read parquet file")
        return
    
    read_time = time.time() - read_start
    logger.info(f"Parquet file loaded in {read_time:.2f}s with {table.num_rows} rows")
    
    # Get file size for throughput calculation
    file_size = get_file_size(args.parquet_file)
    file_size_mb = file_size / (1024 * 1024)
    
    # Convert to pandas for easier processing (or process directly from Arrow)
    df = table.to_pandas()
    
    if args.waveform_column not in df.columns:
        logger.error(f"Column '{args.waveform_column}' not found. Available columns: {df.columns.tolist()}")
        return
    
    # Apply limit if specified
    if args.limit and args.limit < len(df):
        df = df.iloc[:args.limit]
        logger.info(f"Limited to {args.limit} rows")
    
    # Prepare data for batch processing
    total_rows = len(df)
    batches = []
    
    for i in range(0, total_rows, args.batch_size):
        batch_df = df.iloc[i:i + args.batch_size]
        batch_data = []
        
        for idx, row in batch_df.iterrows():
            row_dict = row.to_dict()
            batch_data.append((idx, row_dict))
        
        batches.append(batch_data)
    
    logger.info(f"Created {len(batches)} batches for processing")
    
    # Process batches in parallel
    results = []
    process_start = time.time()
    
    # Choose between thread pool and process pool
    PoolExecutor = ProcessPoolExecutor if args.use_processes else ThreadPoolExecutor
    
    with PoolExecutor(max_workers=args.workers) as executor:
        # Submit all batch processing tasks
        futures = {
            executor.submit(process_waveform_batch, batch, args.waveform_column, args.id_column): i
            for i, batch in enumerate(batches)
        }
        
        # Collect results
        with tqdm(total=len(futures), desc="Processing batches") as pbar:
            for future in as_completed(futures):
                batch_results = future.result()
                if batch_results:
                    results.extend(batch_results)
                pbar.update(1)
    
    process_time = time.time() - process_start
    total_time = time.time() - start_time
    
    # Write results to output file
    logger.info("Writing results...")
    with open(args.output_file, 'w') as f:
        f.write("identifier,mean,length\n")
        for identifier, mean_val, length in results:
            f.write(f"{identifier},{mean_val:.6f},{length}\n")
    
    # Calculate and display throughput statistics
    successful_rows = len(results)
    failed_rows = total_rows - successful_rows
    throughput_mbps = file_size_mb / total_time if total_time > 0 else 0
    rows_per_second = successful_rows / process_time if process_time > 0 else 0
    
    logger.info("\n" + "="*50)
    logger.info("Processing Complete!")
    logger.info(f"Total rows processed: {successful_rows}/{total_rows}")
    logger.info(f"Failed rows: {failed_rows}")
    logger.info(f"File size: {file_size_mb:.2f} MB")
    logger.info(f"Read time: {read_time:.2f}s")
    logger.info(f"Process time: {process_time:.2f}s")
    logger.info(f"Total time: {total_time:.2f}s")
    logger.info(f"Throughput: {throughput_mbps:.2f} MB/s")
    logger.info(f"Rows per second: {rows_per_second:.2f}")
    logger.info(f"Results written to: {args.output_file}")


if __name__ == "__main__":
    main()