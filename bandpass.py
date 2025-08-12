from io import BytesIO

from pydub import AudioSegment
from scipy.signal import butter, sosfiltfilt
import numpy as np

import datachain as dc
from datachain import File

local = False
bucket = "data-flac/datachain-usw2-main-dev" if local else "s3://datachain-usw2-main-dev"
input_dir = f"balanced_train_segments/audio"
input_path = f"{bucket}/test/{input_dir}"
output_path = f"{bucket}/test/balanced_train_segments/filtered_audio/"

low_cutoff = 500  # Lower cutoff frequency (Hz)
high_cutoff = 2000 # Upper cutoff frequency (Hz)
filter_order = 4   # Order of the Butterworth filter


def bandpass_mp3(file: File) -> File:
    """
    Applies a Butterworth bandpass filter to an MP3 audio file and saves the result.

    Args:
        input_mp3_path (str): Path to the input MP3 file.
        output_mp3_path (str): Path to save the filtered MP3 file.
        lowcut (float): Lower cutoff frequency in Hz.
        highcut (float): Upper cutoff frequency in Hz.
        order (int): Order of the Butterworth filter (higher order for steeper roll-off).
    """
    try:
        # 1. Load the MP3 audio
        data = file.get_local_path()
        audio = AudioSegment.from_file(data, "mp3")
    except Exception as e:
        # print(f"Error loading MP3 file: {e}")
        raise e

    # Convert AudioSegment to a NumPy array for processing
    samples = np.array(audio.get_array_of_samples())

    # Get sample rate and channels
    fs = audio.frame_rate
    channels = audio.channels

    # Design the Butterworth bandpass filter
    # Normalize cutoff frequencies by the Nyquist frequency (half the sample rate)
    nyq = 0.5 * fs  # Nyquist frequency
    low = low_cutoff / nyq
    high = high_cutoff / nyq
    sos = butter(filter_order, [low, high], btype='band', output='sos') # Output in second-order sections for stability

    # Apply the filter to each channel if stereo
    if channels == 2:
        # Separate channels, filter each individually
        left_channel = samples[::2]
        right_channel = samples[1::2]
        filtered_left = sosfiltfilt(sos, left_channel)
        filtered_right = sosfiltfilt(sos, right_channel)
        # Interleave filtered channels back into a single array
        filtered_samples = np.empty_like(samples)
        filtered_samples[::2] = filtered_left
        filtered_samples[1::2] = filtered_right
    else: # Mono
        filtered_samples = sosfiltfilt(sos, samples)

    # Convert the filtered NumPy array back to AudioSegment
    filtered_audio = AudioSegment(
        filtered_samples.tobytes(),
        frame_rate=fs,
        sample_width=audio.sample_width,
        channels=channels
    )

    # Export the filtered audio
    try:
        mp3_name = f"{file.get_file_stem()}.mp3"
        buffer = BytesIO()
        filtered_audio.export(buffer, format="mp3")
        buffer.seek(0)
        bytes = buffer.getvalue()

        return File.upload(bytes, f"{output_path}/{mp3_name}")
    except Exception as e:
        # print(f"Error exporting MP3 file: {e}")
        raise e

if __name__ == "__main__":
    print(f"from {input_path} to {output_path}")
    (
        dc
        .read_storage(input_path)
        .filter(dc.C("file.path").glob("*.mp3"))
        #.settings(parallel=8)
        .limit(10)
        .map(mp3_file=bandpass_mp3)
        .save("mp3")
        # .to_storage(output_path, "mp3_file", "filename")
    )

