import argparse
import numpy as np
import pyarrow.parquet as pq
import pyarrow.compute as pc


def get_waveform(parquet_file: str, filename: str, channel: int) -> np.ndarray:
    """
    Query waveform from parquet file using PyArrow for memory efficiency.
    
    Args:
        parquet_file: Path to parquet file
        filename: Audio filename to search for
        channel: Channel ID (0-based)
        
    Returns:
        Numpy array of waveform data
    """
    table = pq.read_table(parquet_file)

    # Create filter mask
    filename_mask = pc.equal(table['waveform.filename'], filename)
    channel_mask = pc.equal(table['waveform.channel'], channel)
    combined_mask = pc.and_(filename_mask, channel_mask)
    
    # Filter table
    filtered_table = table.filter(combined_mask)
    
    if filtered_table.num_rows == 0:
        raise ValueError(f"No waveform found for file '{filename}' channel {channel}")
    
    if filtered_table.num_rows > 1:
        print(f"Warning: {filtered_table.num_rows} matches found, using first one")
    
    # Get waveform bytes and convert to numpy
    waveform_bytes = filtered_table.column('waveform.waveform')[0].as_py()
    waveform = np.frombuffer(waveform_bytes, dtype='float32')
    
    return waveform


def main():
    parser = argparse.ArgumentParser(description='Query waveform from parquet file')
    parser.add_argument('parquet_file', help='Path to parquet file')
    parser.add_argument('filename', help='Audio filename (e.g., "example.wav")')
    parser.add_argument('channel', type=int, help='Channel ID (0-based)')
    parser.add_argument('--info', action='store_true', help='Show audio info')
    
    args = parser.parse_args()
    
    try:
        if args.info:
            table = pq.read_table(args.parquet_file)
            import pyarrow.compute as pc
            
            filename_mask = pc.equal(table['waveform.filename'], args.filename)
            channel_mask = pc.equal(table['waveform.channel'], args.channel)
            combined_mask = pc.and_(filename_mask, channel_mask)
            filtered_table = table.filter(combined_mask)
            
            if filtered_table.num_rows == 0:
                raise ValueError(f"No waveform found for file '{args.filename}' channel {args.channel}")
            
            # Extract info
            waveform_bytes = filtered_table.column('waveform.waveform')[0].as_py()
            waveform = np.frombuffer(waveform_bytes, dtype='float32')
            
            print(f"Waveform shape: {waveform.shape}")
            print(f"Min value: {waveform.min():.6f}")
            print(f"Max value: {waveform.max():.6f}")
            
            print("\nAudio Info:")
            print(f"  Sample rate: {filtered_table.column('waveform.info.sample_rate')[0].as_py()} Hz")
            print(f"  Channels: {filtered_table.column('waveform.info.channels')[0].as_py()}")
            print(f"  Duration: {filtered_table.column('waveform.info.duration')[0].as_py():.2f} seconds")
            print(f"  Format: {filtered_table.column('waveform.info.format')[0].as_py()}")
            print(f"  Channel name: {filtered_table.column('waveform.channel_name')[0].as_py()}")
        else:
            waveform = get_waveform(args.parquet_file, args.filename, args.channel)
            print(f"Waveform shape: {waveform.shape}")
            print(f"Min value: {waveform.min():.6f}")
            print(f"Max value: {waveform.max():.6f}")
            
    except Exception as e:
        print(f"Error: {e}")
        return 1
    
    return 0


if __name__ == '__main__':
    exit(main())
