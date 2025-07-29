import io

import librosa
import numpy as np
from tempfile import TemporaryDirectory
from pathlib import Path

from pydub import AudioSegment
import datachain as dc
from datachain.lib.audio import audio_to_bytes
from datachain import AudioFile, File


LOCAL = True
BUCKET = "data-flac/datachain-usw2-main-dev" if LOCAL else "s3://datachain-usw2-main-dev"
INPUT_DIR = f"balanced_train_segments/audio"
INPUT_PATH = f"{BUCKET}/test/{INPUT_DIR}"
OUTPUT_PATH = f"{BUCKET}/test/{INPUT_DIR}/spectrograms"
OUTPUT = "spectrogram"


def create_spectrogram_data(file: AudioFile) -> File:
    """
    Creates and returns the spectrogram data from an audio file.

    Args:
        audio_filepath (str): Path to the audio file.
        sr (int): Sample rate of the audio. Defaults to 22050.
        n_fft (int): Length of the FFT window. Defaults to 2048.
        hop_length (int): Number of samples between successive frames. Defaults to 512.

    Returns:
        numpy.ndarray: The computed spectrogram (magnitude or power) in decibels.
    """
    audio = AudioSegment.from_file(file.get_local_path(), format="mp3")
    audio_bytes = audio_to_bytes(file, format="mp3")

    with TemporaryDirectory() as temp_dir:
        temp_file_path = Path(temp_dir, "audio.mp3")  # or .mp3, .ogg etc.
        temp_file_path.write_bytes(audio_bytes)
        y, sr = librosa.load(temp_file_path)
    # Load the audio file
    # y, sr = librosa.load(audio_filepath)

    # Compute the spectrogram using Short-Time Fourier Transform (STFT)
    spectrogram = np.abs(librosa.stft(y))

    # Convert amplitude spectrogram to decibels for better interpretation
    spectrogram_db = librosa.amplitude_to_db(spectrogram, ref=np.max)

    # Convert to bytes in numpy format
    buffer = io.BytesIO()
    np.save(buffer, spectrogram_db)
    npy_bytes = buffer.getvalue()

    # Upload data and return it as File structure
    save_path = f"{OUTPUT_PATH}/{file.get_file_stem()}.npy"
    spectrum_file = File.upload(npy_bytes, save_path)
    return spectrum_file


result = (
    dc
    .read_storage(INPUT_PATH, type="audio")
    .filter(dc.C("file.path").glob("*.mp3"))
    # .limit(10)
    .map(spec=create_spectrogram_data)
    .save(OUTPUT)
)

if not LOCAL:
    dc.read_storage(BUCKET, update=True).exec()
else:
   result.show()
