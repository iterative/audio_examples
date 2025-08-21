#!/usr/bin/env python3
"""
Audio file processor that downloads files from S3/GCS bucket, 
calculates channel statistics, and measures throughput.
"""

import argparse
import csv
import time
import os
import tempfile
from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import Tuple, Optional, List
import logging
from pathlib import Path
import warnings
import threading
import psutil

# Disable librosa's parallel processing to avoid resource leaks
os.environ['LIBROSA_CACHE_COMPRESS'] = '0'
os.environ['LIBROSA_CACHE_LEVEL'] = '0'

import numpy as np
import librosa
import boto3
from botocore.exceptions import NoCredentialsError, ClientError
from google.cloud import storage as gcs
from tqdm import tqdm

# Suppress multiprocessing resource tracker warnings
warnings.filterwarnings("ignore", category=UserWarning, module="multiprocessing.resource_tracker")

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
    
    def download_file(self, key: str, local_path: str) -> bool:
        """Download file from cloud storage"""
        try:
            if self.provider == 's3':
                self.client.download_file(self.bucket_name, key, local_path)
            elif self.provider == 'gcs':
                blob = self.bucket.blob(key)
                blob.download_to_filename(local_path)
            return True
        except Exception as e:
            logger.error(f"Failed to download {key}: {e}")
            return False


def process_audio_file(
    storage: CloudStorage,
    filename: str,
    channel: int,
    temp_dir: str
) -> Optional[Tuple[str, float, int, float]]:
    """
    Download and process a single audio file.
    
    Returns:
        Tuple of (filename, mean, length, file_size_mb) or None if failed
    """
    start_time = time.time()
    temp_path = os.path.join(temp_dir, os.path.basename(filename))
    
    try:
        # Download file
        if not storage.download_file(filename, temp_path):
            return None
        
        # Get file size for throughput calculation
        file_size_mb = os.path.getsize(temp_path) / (1024 * 1024)
        
        # Load audio file with librosa (disable internal multiprocessing to avoid resource leaks)
        audio_data, sample_rate = librosa.load(temp_path, sr=None, mono=False)
        
        # Handle mono vs multi-channel
        if audio_data.ndim == 1:
            # Mono file
            if channel > 0:
                logger.warning(f"{filename}: Requested channel {channel} but file is mono")
                return None
            channel_data = audio_data
        else:
            # Multi-channel file
            if channel >= audio_data.shape[0]:
                logger.warning(f"{filename}: Channel {channel} not available (has {audio_data.shape[0]} channels)")
                return None
            channel_data = audio_data[channel]
        
        # Calculate statistics
        mean_value = float(np.mean(channel_data))
        length = len(channel_data)
        
        # Clean up temp file
        os.remove(temp_path)
        
        process_time = time.time() - start_time
        
        return (filename, mean_value, length, file_size_mb, process_time)
        
    except Exception as e:
        logger.error(f"Error processing {filename}: {e}")
        if os.path.exists(temp_path):
            os.remove(temp_path)
        return None


def main():
    parser = argparse.ArgumentParser(
        description='Process audio files from cloud storage and calculate channel statistics'
    )
    parser.add_argument('input_csv', help='CSV file with list of audio filenames')
    parser.add_argument('bucket', help='Bucket name where files are located (can include s3:// or gs:// prefix)')
    parser.add_argument('channel', type=int, help='Channel number to process (0-indexed)')
    parser.add_argument('output_file', help='Output file for results')
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
    
    # Parse bucket name - remove protocol prefix and trailing slash
    bucket_name = args.bucket
    if bucket_name.startswith('s3://'):
        bucket_name = bucket_name[5:]
        args.provider = 's3'
    elif bucket_name.startswith('gs://'):
        bucket_name = bucket_name[5:]
        args.provider = 'gcs'
    
    # Remove trailing slash if present
    bucket_name = bucket_name.rstrip('/')
    
    # Extract bucket name and prefix if path included
    if '/' in bucket_name:
        parts = bucket_name.split('/', 1)
        bucket_name = parts[0]
        prefix = parts[1] + '/' if len(parts) > 1 else ''
    else:
        prefix = ''
    
    # Read input CSV
    filenames = []
    with open(args.input_csv, 'r') as f:
        reader = csv.reader(f)
        for row in reader:
            if len(row) > args.column:
                filename = row[args.column].strip()
                # Prepend prefix if bucket had a path component
                if prefix and not filename.startswith(prefix):
                    filename = prefix + filename
                filenames.append(filename)
                if args.limit and len(filenames) >= args.limit:
                    break
    
    if not filenames:
        logger.error("No filenames found in CSV")
        return
    
    logger.info(f"Processing {len(filenames)} files from {bucket_name}")
    logger.info(f"Using {args.workers} parallel workers")
    if prefix:
        logger.info(f"Using prefix: {prefix}")
    
    # Initialize storage client
    storage = CloudStorage(bucket_name, args.provider)
    
    # Initialize CPU monitor if requested
    cpu_monitor = None
    cpu_stats = None
    if args.monitor_cpu or args.cpu_log:
        cpu_monitor = CPUMonitor(interval=args.monitor_interval, lightweight=args.lightweight_monitor)
        mode = "lightweight" if args.lightweight_monitor else "full"
        logger.info(f"CPU monitoring enabled ({mode} mode, interval: {args.monitor_interval}s)")
    
    # Create temporary directory for downloads
    with tempfile.TemporaryDirectory() as temp_dir:
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
                executor.submit(process_audio_file, storage, fname, args.channel, temp_dir): fname
                for fname in filenames
            }
            
            # Process results as they complete
            with tqdm(total=len(filenames), desc="Processing files") as pbar:
                for future in as_completed(futures):
                    try:
                        result = future.result()
                        if result:
                            filename, mean_val, length, size_mb, proc_time = result
                            results.append((filename, mean_val, length))
                            total_bytes += size_mb * 1024 * 1024
                            total_time += proc_time
                    except Exception as e:
                        logger.error(f"Failed to process future: {e}")
                    finally:
                        pbar.update(1)
            
            # Ensure all futures are complete and cleaned up
            executor.shutdown(wait=True)
        
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