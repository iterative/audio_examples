# Requirements: datachain[audio], librosa, numpy

import io
import os
from typing import Iterator, ClassVar

import numpy as np
import librosa
from pydantic import BaseModel

import datachain as dc
from datachain import AudioFile, Audio, func

LOCAL = True
STORAGE = "data-flac-full/datachain-usw2-main-dev/balanced_train_segments/audio/" \
                if LOCAL else "s3://datachain-usw2-main-dev/balanced_train_segments"
LIMIT = 300
OUTPUT = "waveforms"
SAMPLE_RATE = None  # None to keep original sample rate


class Waveform(BaseModel):
    file: AudioFile
    filename: str  # Just the filename for easy lookup
    info: Audio
    channel: int
    channel_name: str
    waveform: bytes  # Binary storage for efficient Parquet

    DTYPE: ClassVar[str] = "float32"  # Constant data type for all waveforms

    @property
    def waveform_np(self) -> np.ndarray:
        return np.frombuffer(self.waveform, dtype=self.DTYPE)


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


def extract_waveforms(file: AudioFile) -> Iterator[Waveform]:
    """
    Extract waveforms from audio file, yielding one Waveform per channel.
    File and audio metadata are duplicated for each channel to enable independent
    processing.
    """
    data = io.BytesIO(file.read())
    audio, sr = librosa.load(data, sr=SAMPLE_RATE, mono=False)
    audio_info = file.get_info()

    # Ensure audio is 2D (channels x samples)
    if audio.ndim == 1:
        audio = audio.reshape(1, -1)

    num_channels = audio.shape[0]
    for ch_idx in range(num_channels):
        channel_data = audio[ch_idx].astype(Waveform.DTYPE)
        yield Waveform(
            file=file,
            filename=os.path.basename(file.path),
            info=audio_info,
            channel=ch_idx,
            channel_name=get_channel_name(num_channels, ch_idx),
            waveform=channel_data.tobytes()
        )


chain = (
    dc
    .read_storage(STORAGE, type="audio")
    .filter(dc.C("file.path").glob("*.wav") | dc.C("file.path").glob("*.flac"))
)

if LIMIT:
    chain = chain.limit(LIMIT)

chain = (
    chain
    .gen(waveform=extract_waveforms)
    .save(OUTPUT)
)

# chain.to_parquet("waveform.pq")

if LOCAL:
    dc.read_dataset(OUTPUT).show()
