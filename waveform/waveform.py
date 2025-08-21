# /// script
# dependencies = [
#	"numpy",
#	"librosa",
#   "datachain[video,audio]",
# ]
# ///


import io
import os
from typing import Iterator, ClassVar

import numpy as np
import librosa
from pydantic import BaseModel

import datachain as dc
from datachain import AudioFile, Audio

# Run locally (not in cluster) until binary batching limit issue is fixed.
LOCAL = False
STORAGE = "data-flac/datachain-usw2-main-dev/balanced_train_segments/audio/" \
                if LOCAL else "s3://datachain-usw2-main-dev/balanced_train_segments"
OUTPUT = "waveforms"
OUTPUT_PQ = "s3://datachain-usw2-main-dev/test/waveforms-flac.pq"

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
            channel_name=Audio.get_channel_name(num_channels, ch_idx),
            waveform=channel_data.tobytes()
        )

chain = (
    dc
    .read_storage(STORAGE, type="audio", update=True)
    .filter(dc.C("file.path").glob("*.flac"))
    # .limit(100)
    .settings(batch_rows=300)
    .gen(waveform=extract_waveforms)
    .save(OUTPUT)
)

chain.to_parquet(OUTPUT_PQ, chunk_size=500)
