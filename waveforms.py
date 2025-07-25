# Requirements: datachain[audio], librosa, numpy

import io
from typing import Optional, Iterator

import numpy as np
import librosa
from pydantic import BaseModel

import datachain as dc
from datachain import AudioFile

LOCAL = True
STORAGE = "data-flac/" if LOCAL else "s3://datachain-usw2-main-dev/sony_av_data"
OUTPUT = "waveforms"
SAMPLE_RATE = None  # None to keep original sample rate


class AudioWaveform(BaseModel):
    file_name: str
    file_path: str
    channel: int
    channel_name: str
    waveform: bytes  # Binary storage for efficient Parquet
    sample_rate: int
    duration: float
    num_samples: int
    dtype: str = "float32"
    
    @property
    def waveform_np(self) -> np.ndarray:
        """Get waveform as numpy array"""
        return np.frombuffer(self.waveform, dtype=self.dtype)


def get_channel_name(num_channels: int, channel_idx: int) -> str:
    """Map channel index to meaningful name based on common audio formats"""
    if num_channels == 1:
        return "Mono"
    elif num_channels == 2:
        return ["Left", "Right"][channel_idx]
    elif num_channels == 4:
        return ["W", "X", "Y", "Z"][channel_idx]  # First-order Ambisonics
    elif num_channels == 6:
        return ["FL", "FR", "FC", "LFE", "BL", "BR"][channel_idx]  # 5.1 surround
    elif num_channels == 8:
        return ["FL", "FR", "FC", "LFE", "BL", "BR", "SL", "SR"][channel_idx]  # 7.1 surround
    else:
        return f"Ch{channel_idx + 1}"


def extract_waveforms(file: AudioFile) -> Iterator[AudioWaveform]:
    """
    Extract waveforms from audio file, one per channel
    
    Args:
        file: AudioFile from DataChain
        
    Yields:
        AudioWaveform objects for each channel
    """
    # Load audio data
    data = io.BytesIO(file.read())
    audio, sr = librosa.load(data, sr=SAMPLE_RATE, mono=False)
    
    # Ensure audio is 2D (channels x samples)
    if audio.ndim == 1:
        audio = audio.reshape(1, -1)
    
    num_channels = audio.shape[0]
    duration = audio.shape[1] / sr
    file_name = file.path.split("/")[-1]
    
    for ch_idx in range(num_channels):
        channel_data = audio[ch_idx].astype(np.float32)
        yield AudioWaveform(
            file_name=file_name,
            file_path=file.path,
            channel=ch_idx,
            channel_name=get_channel_name(num_channels, ch_idx),
            waveform=channel_data.tobytes(),
            sample_rate=sr,
            duration=duration,
            num_samples=len(channel_data)
        )


chain = (
    dc
    .read_storage(STORAGE, type="audio")
    .filter(dc.C("file.path").glob("*.wav") | dc.C("file.path").glob("*.flac") | dc.C("file.path").glob("*.mp3"))
    .gen(waveform=extract_waveforms)
    .save(OUTPUT)
)

if LOCAL:
    dc.read_dataset(OUTPUT).show()
