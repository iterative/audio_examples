import io

import librosa
import numpy as np

import datachain as dc
from datachain.lib.audio import audio_to_bytes
from datachain import AudioFile, File
import matplotlib.pyplot as plt

LOCAL = True
BUCKET = "data-flac/datachain-usw2-main-dev" if LOCAL else "s3://datachain-usw2-main-dev"
INPUT_DIR = f"balanced_train_segments/audio"
INPUT_PATH = f"{BUCKET}/test/{INPUT_DIR}"
OUTPUT_PATH = f"{BUCKET}/test/{INPUT_DIR}/spectrograms"
OUTPUT = "spectrogram"


def create_spectrogram(file: AudioFile):
    audio_bytes = audio_to_bytes(file, format="mp3")
    with io.BytesIO(audio_bytes) as f:
        y, sr = librosa.load(f)

    # Compute the spectrogram using Short-Time Fourier Transform (STFT)
    spectrogram = np.abs(librosa.stft(y))

    # Convert amplitude spectrogram to decibels for better interpretation
    spectrogram_db = librosa.amplitude_to_db(spectrogram, ref=np.max)
    return spectrogram_db, sr


def create_spectrogram_image(file: AudioFile) -> File:
    spectrogram_db, sr = create_spectrogram(file)
    f = plt.figure(figsize=(10, 6))
    librosa.display.specshow(spectrogram_db, x_axis='time', y_axis='mel', sr=sr, cmap='viridis')
    plt.colorbar(format='%+2.0f dB')
    plt.title('Mel Spectrogram')
    plt.tight_layout()

    buffer = io.BytesIO()
    plt.savefig(buffer, dpi=300, bbox_inches='tight')
    plt.close()
    buffer.seek(0)
    img_bytes = buffer.read()

    # Upload data and return it as File structure
    save_path = f"{OUTPUT_PATH}/{file.get_file_stem()}.jpg"
    spectrum_file = File.upload(img_bytes, save_path)
    return spectrum_file


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

    spectrogram_db, sr = create_spectrogram(file)
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
    .settings(parallel=8)
    .limit(10)
    .map(spec=create_spectrogram_data)
    .map(spec_img=create_spectrogram_image)
    .save(OUTPUT)
)
