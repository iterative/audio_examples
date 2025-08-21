#!/usr/bin/env python3
"""
Numpy array processor that reads .npy files directly from S3,
calculates statistics, and measures throughput.
"""

import argparse
import csv
import time
import os
import io
from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import Tuple, Optional, List
import logging
from pathlib import Path
import threading
import psutil

import numpy as np
import boto3
from botocore.exceptions import NoCredentialsError, ClientError
from google.cloud import storage as gcs
from tqdm import tqdm

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


class CPUMonitor:
    """Monitor CPU utilization during processing"""
    
    def __init__(self, interval: float = 1.0, lightweight: bool = False):
        """
        Initialize CPU monitor
        
        Args:
            interval: Sampling interval in seconds
            lightweight: If True, only monitor total CPU (not per-core)
        """
        self.interval = interval
        self.lightweight = lightweight
        self.cpu_samples = []
        self.memory_samples = []
        self.monitoring = False
        self.monitor_thread = None
        self.process = psutil.Process()
        
    def _monitor_loop(self):
        """Background thread that samples CPU usage"""
        if self.lightweight:
            # Lightweight monitoring - only total CPU
            while self.monitoring:
                try:
                    cpu_percent = psutil.cpu_percent(interval=self.interval)
                    
                    self.cpu_samples.append({
                        'timestamp': time.time(),
                        'total': cpu_percent,
                        'per_core': [],
                        'process': 0,
                        'cores_used': 0
                    })
                    
                except Exception as e:
                    logger.debug(f"Error in CPU monitoring: {e}")
                    time.sleep(self.interval)
        else:
            # Full monitoring with reduced frequency for expensive operations
            sample_counter = 0
            while self.monitoring:
                try:
                    # Only do detailed sampling every 5th iteration to reduce overhead
                    if sample_counter % 5 == 0:
                        # Get per-core CPU utilization (more expensive)
                        cpu_per_core = psutil.cpu_percent(interval=self.interval, percpu=True)
                        cpu_percent = sum(cpu_per_core) / len(cpu_per_core) if cpu_per_core else 0
                        cores_used = sum(1 for c in cpu_per_core if c > 50)
                    else:
                        # Just get total CPU (less expensive)
                        cpu_percent = psutil.cpu_percent(interval=self.interval)
                        cpu_per_core = []
                        cores_used = 0
                    
                    # Process CPU only every 3rd sample to reduce overhead
                    if sample_counter % 3 == 0:
                        process_cpu = self.process.cpu_percent()
                    else:
                        process_cpu = 0
                    
                    # Memory only every 2nd sample
                    if sample_counter % 2 == 0:
                        memory = psutil.virtual_memory()
                        mem_data = {
                            'timestamp': time.time(),
                            'percent': memory.percent,
                            'used_gb': memory.used / (1024**3),
                            'available_gb': memory.available / (1024**3)
                        }
                        self.memory_samples.append(mem_data)
                    
                    self.cpu_samples.append({
                        'timestamp': time.time(),
                        'total': cpu_percent,
                        'per_core': cpu_per_core,
                        'process': process_cpu,
                        'cores_used': cores_used
                    })
                    
                    sample_counter += 1
                    
                except Exception as e:
                    logger.debug(f"Error in CPU monitoring: {e}")
                    time.sleep(self.interval)
                
    def start(self):
        """Start monitoring CPU usage"""
        if not self.monitoring:
            self.monitoring = True
            self.monitor_thread = threading.Thread(target=self._monitor_loop, daemon=True)
            self.monitor_thread.start()
            # Initial reading to prime the CPU percent
            psutil.cpu_percent(interval=None)
            
    def stop(self):
        """Stop monitoring and return statistics"""
        self.monitoring = False
        if self.monitor_thread:
            self.monitor_thread.join(timeout=2)
            
        if not self.cpu_samples:
            return None
            
        # Calculate statistics
        cpu_totals = [s['total'] for s in self.cpu_samples]
        process_cpus = [s['process'] for s in self.cpu_samples if s['process'] > 0]
        cores_used = [s['cores_used'] for s in self.cpu_samples]
        memory_percents = [s['percent'] for s in self.memory_samples]
        
        stats = {
            'cpu_avg': np.mean(cpu_totals) if cpu_totals else 0,
            'cpu_max': np.max(cpu_totals) if cpu_totals else 0,
            'cpu_min': np.min(cpu_totals) if cpu_totals else 0,
            'process_cpu_avg': np.mean(process_cpus) if process_cpus else 0,
            'process_cpu_max': np.max(process_cpus) if process_cpus else 0,
            'avg_cores_used': np.mean(cores_used) if cores_used else 0,
            'max_cores_used': np.max(cores_used) if cores_used else 0,
            'memory_avg': np.mean(memory_percents) if memory_percents else 0,
            'memory_max': np.max(memory_percents) if memory_percents else 0,
            'num_samples': len(self.cpu_samples),
            'total_cores': psutil.cpu_count()
        }
        
        return stats
    
    def save_to_csv(self, filepath: str):
        """Save detailed CPU monitoring data to CSV file"""
        if not self.cpu_samples:
            return
            
        with open(filepath, 'w', newline='') as f:
            writer = csv.writer(f)
            # Write header
            writer.writerow(['timestamp', 'cpu_total', 'process_cpu', 'cores_active', 
                           'memory_percent', 'memory_used_gb'] + 
                          [f'core_{i}' for i in range(psutil.cpu_count())])
            
            # Write data
            start_time = self.cpu_samples[0]['timestamp'] if self.cpu_samples else 0
            for i, cpu_sample in enumerate(self.cpu_samples):
                mem_sample = self.memory_samples[i] if i < len(self.memory_samples) else {}
                row = [
                    cpu_sample['timestamp'] - start_time,  # Relative time
                    cpu_sample['total'],
                    cpu_sample['process'],
                    cpu_sample['cores_used'],
                    mem_sample.get('percent', 0),
                    mem_sample.get('used_gb', 0)
                ] + cpu_sample['per_core']
                writer.writerow(row)


class CloudStorage:
    """Abstract interface for cloud storage operations"""
    
    def __init__(self, bucket_name: str, provider: str = 's3'):
        self.bucket_name = bucket_name
        self.provider = provider
        
        if provider == 's3':
            self.client = boto3.client('s3')
        elif provider == 'gcs':
            self.client = gcs.Client()
            self.bucket = self.client.bucket(bucket_name)
        else:
            raise ValueError(f"Unsupported provider: {provider}")
    
    def read_numpy_array(self, key: str, array_key: str = None) -> Optional[Tuple[np.ndarray, int]]:
        """
        Read numpy array directly from cloud storage without downloading to disk.
        Supports both .npy and .npz files.
        
        Args:
            key: S3/GCS key to the file
            array_key: For npz files, the key of the array to extract (optional)
            
        Returns:
            Tuple of (array, file_size) or None if failed
        """
        try:
            if self.provider == 's3':
                response = self.client.get_object(Bucket=self.bucket_name, Key=key)
                file_size = response['ContentLength']
                data = response['Body'].read()
            elif self.provider == 'gcs':
                blob = self.bucket.blob(key)
                data = blob.download_as_bytes()
                file_size = len(data)
            else:
                return None
            
            # Load numpy data
            buffer = io.BytesIO(data)
            
            # Check if it's npz or npy based on file extension
            if key.endswith('.npz'):
                # Load npz file
                npz_data = np.load(buffer, allow_pickle=False)
                
                # If array_key is specified, use it
                if array_key:
                    if array_key in npz_data:
                        array = npz_data[array_key]
                    else:
                        logger.error(f"Key '{array_key}' not found in npz file. Available keys: {list(npz_data.keys())}")
                        return None
                else:
                    # If no key specified, try common keys or use the first array
                    if 'data' in npz_data:
                        array = npz_data['data']
                    elif 'arr_0' in npz_data:
                        array = npz_data['arr_0']
                    elif len(npz_data.keys()) == 1:
                        # If only one array, use it
                        array = npz_data[list(npz_data.keys())[0]]
                    else:
                        logger.error(f"Multiple arrays in npz file. Please specify array_key. Available keys: {list(npz_data.keys())}")
                        return None
            else:
                # Load npy file
                array = np.load(buffer, allow_pickle=False)
            
            return array, file_size
            
        except Exception as e:
            logger.error(f"Failed to read {key}: {e}")
            return None


def process_numpy_file(
    storage: CloudStorage,
    filename: str,
    s3_dir: str,
    channel: int,
    suffix: str,
    array_key: str = None
) -> Optional[Tuple[str, float, int, float, float]]:
    """
    Read and process a single numpy file from S3.
    
    Returns:
        Tuple of (filename, mean, length, file_size_mb, process_time) or None if failed
    """
    start_time = time.time()
    
    # Construct the S3 key
    # Extract just the base filename (no path, no extension)
    base_filename = os.path.basename(filename)
    # Remove any file extension from filename if present
    base_filename = base_filename.rsplit('.', 1)[0] if '.' in base_filename else base_filename
    # Also remove any existing channel suffix if present (e.g., _ch0)
    if '_ch' in base_filename:
        base_filename = base_filename.rsplit('_ch', 1)[0]
    
    s3_key = f"{s3_dir}/{base_filename}_ch{channel}.{suffix}"
    
    try:
        # Read numpy array directly from S3
        result = storage.read_numpy_array(s3_key, array_key)
        # print(f"reading {s3_key}, {array_key}")
        if result is None:
            return None
        
        array, file_size = result
        file_size_mb = file_size / (1024 * 1024)
        
        # Calculate statistics
        mean_value = float(np.mean(array))
        length = len(array) if array.ndim == 1 else array.shape[0]
        
        process_time = time.time() - start_time
        
        return (filename, mean_value, length, file_size_mb, process_time)
        
    except Exception as e:
        logger.error(f"Error processing {filename}: {e}")
        return None


def main():
    parser = argparse.ArgumentParser(
        description='Process numpy files from S3 and calculate statistics'
    )
    parser.add_argument('input_csv', help='CSV file with list of filenames (without path)')
    parser.add_argument('s3_dir', help='S3 directory path (e.g., s3://bucket/path/to/dir or just path/to/dir)')
    parser.add_argument('channel', type=int, help='Channel number for filename pattern')
    parser.add_argument('output_file', help='Output file for results')
    parser.add_argument('--suffix', default='npy',
                        help='File suffix/extension (default: npy, can be npz)')
    parser.add_argument('--array-key', help='For npz files, the key of the array to extract (optional)')
    parser.add_argument('--bucket', help='Bucket name (if not included in s3_dir)')
    parser.add_argument('--provider', choices=['s3', 'gcs'], default='s3',
                        help='Cloud storage provider (default: s3)')
    parser.add_argument('--workers', type=int, default=4,
                        help='Number of parallel workers (default: 4)')
    parser.add_argument('--column', type=int, default=0,
                        help='CSV column index containing filenames (default: 0)')
    parser.add_argument('--limit', type=int, default=None,
                        help='Maximum number of files to process (default: all)')
    parser.add_argument('--cpu-log', help='Optional CSV file to save CPU monitoring data')
    parser.add_argument('--monitor-cpu', action='store_true', 
                        help='Enable CPU monitoring (may impact performance)')
    parser.add_argument('--monitor-interval', type=float, default=2.0,
                        help='CPU monitoring interval in seconds (default: 2.0)')
    parser.add_argument('--lightweight-monitor', action='store_true',
                        help='Use lightweight CPU monitoring (less detailed, better performance)')
    
    args = parser.parse_args()
    
    # Parse S3 directory and bucket
    s3_dir = args.s3_dir
    bucket_name = args.bucket
    
    # Handle full S3 URL format
    if s3_dir.startswith('s3://'):
        s3_dir = s3_dir[5:]
        args.provider = 's3'
        if '/' in s3_dir:
            parts = s3_dir.split('/', 1)
            bucket_name = parts[0]
            s3_dir = parts[1]
        else:
            bucket_name = s3_dir
            s3_dir = ''
    elif s3_dir.startswith('gs://'):
        s3_dir = s3_dir[5:]
        args.provider = 'gcs'
        if '/' in s3_dir:
            parts = s3_dir.split('/', 1)
            bucket_name = parts[0]
            s3_dir = parts[1]
        else:
            bucket_name = s3_dir
            s3_dir = ''
    
    # Remove trailing slash from directory
    s3_dir = s3_dir.rstrip('/')
    
    if not bucket_name:
        logger.error("Bucket name must be provided either in s3_dir or via --bucket")
        return
    
    # Read input CSV
    filenames = []
    with open(args.input_csv, 'r') as f:
        reader = csv.reader(f)
        for row in reader:
            if len(row) > args.column:
                filename = row[args.column].strip()
                filenames.append(filename)
                if args.limit and len(filenames) >= args.limit:
                    break
    
    if not filenames:
        logger.error("No filenames found in CSV")
        return
    
    logger.info(f"Processing {len(filenames)} files from {bucket_name}/{s3_dir}")
    logger.info(f"Using pattern: {s3_dir}/<filename>_ch{args.channel}.{args.suffix}")
    if args.suffix == 'npz' and args.array_key:
        logger.info(f"Extracting array key: {args.array_key}")
    logger.info(f"Using {args.workers} parallel workers")
    
    # Initialize storage client
    storage = CloudStorage(bucket_name, args.provider)
    
    # Initialize CPU monitor if requested
    cpu_monitor = None
    cpu_stats = None
    if args.monitor_cpu or args.cpu_log:
        cpu_monitor = CPUMonitor(interval=args.monitor_interval, lightweight=args.lightweight_monitor)
        mode = "lightweight" if args.lightweight_monitor else "full"
        logger.info(f"CPU monitoring enabled ({mode} mode, interval: {args.monitor_interval}s)")
    
    # Process files in parallel
    results = []
    total_bytes = 0
    total_time = 0
    start_time = time.time()
    
    # Start CPU monitoring if enabled
    if cpu_monitor:
        cpu_monitor.start()
    
    with ThreadPoolExecutor(max_workers=args.workers) as executor:
        # Submit all tasks
        futures = {
            executor.submit(process_numpy_file, storage, fname, s3_dir, args.channel, args.suffix, args.array_key): fname
            for fname in filenames
        }
        
        # Process results as they complete
        with tqdm(total=len(filenames), desc="Processing files") as pbar:
            for future in as_completed(futures):
                result = future.result()
                if result:
                    filename, mean_val, length, size_mb, proc_time = result
                    results.append((filename, mean_val, length))
                    total_bytes += size_mb * 1024 * 1024
                    total_time += proc_time
                pbar.update(1)
    
    elapsed_time = time.time() - start_time
    
    # Stop CPU monitoring and get statistics if enabled
    if cpu_monitor:
        cpu_stats = cpu_monitor.stop()
    
    # Write results to output file
    with open(args.output_file, 'w') as f:
        f.write("filename,mean,length\n")
        for filename, mean_val, length in results:
            f.write(f"{filename},{mean_val:.6f},{length}\n")
    
    # Calculate and display throughput statistics
    successful = len(results)
    failed = len(filenames) - successful
    throughput_mbps = (total_bytes / (1024 * 1024)) / elapsed_time if elapsed_time > 0 else 0
    files_per_second = successful / elapsed_time if elapsed_time > 0 else 0
    
    logger.info("\n" + "="*50)
    logger.info("Processing Complete!")
    logger.info(f"Total files processed: {successful}/{len(filenames)}")
    logger.info(f"Failed files: {failed}")
    logger.info(f"Total time: {elapsed_time:.2f} seconds")
    logger.info(f"Total data processed: {total_bytes / (1024 * 1024):.2f} MB")
    logger.info(f"Throughput: {throughput_mbps:.2f} MB/s")
    logger.info(f"Files per second: {files_per_second:.2f}")
    
    # Display CPU statistics
    if cpu_stats:
        logger.info("\n" + "-"*50)
        logger.info("CPU Utilization Statistics:")
        logger.info(f"System CPU - Avg: {cpu_stats['cpu_avg']:.1f}%, Max: {cpu_stats['cpu_max']:.1f}%, Min: {cpu_stats['cpu_min']:.1f}%")
        logger.info(f"Process CPU - Avg: {cpu_stats['process_cpu_avg']:.1f}%, Max: {cpu_stats['process_cpu_max']:.1f}%")
        logger.info(f"Cores utilized - Avg: {cpu_stats['avg_cores_used']:.1f}/{cpu_stats['total_cores']}, Max: {cpu_stats['max_cores_used']}/{cpu_stats['total_cores']}")
        logger.info(f"Memory usage - Avg: {cpu_stats['memory_avg']:.1f}%, Max: {cpu_stats['memory_max']:.1f}%")
        logger.info(f"CPU efficiency: {(cpu_stats['process_cpu_avg'] / (args.workers * 100) * 100):.1f}% of theoretical max")
        
        # Save detailed CPU log if requested
        if args.cpu_log:
            cpu_monitor.save_to_csv(args.cpu_log)
            logger.info(f"CPU monitoring data saved to: {args.cpu_log}")
    
    logger.info(f"\nResults written to: {args.output_file}")


if __name__ == "__main__":
    main()