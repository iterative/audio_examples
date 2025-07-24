# Requirements: datachain[audio], librosa

import io
from collections.abc import Iterator
from typing import Optional

import numpy as np
import librosa
from pydantic import BaseModel

import datachain as dc
from datachain import AudioFile, AudioFragment

LOCAL = False
STORAGE = "data15/" if LOCAL else "s3://datachain-usw2-main-dev/sony_av_data"
WINDOW = 5
OVERLAP = 1
SAMPLE_RATE = 16000
OUTPUT = "segments"


class Segment(BaseModel):
    fragment: AudioFragment
    id: int
    channel: str
    rms: float
    rms_mean: float
    rms_std: float
    rms_min: float
    rms_max: float
    azimuth: Optional[float] = None
    elevation: Optional[float] = None


def get_channel_name(channel_type: str, channel_idx: int) -> str:
    """Map channel index to meaningful name"""
    names = {
        'mono': ['Mono'],
        'stereo': ['Left', 'Right'],
        'foa': ['W', 'X', 'Y', 'Z']
    }
    return names.get(channel_type, [f'Ch{i}' for i in range(10)])[channel_idx]


def segment_audio(file: AudioFile) -> Iterator[Segment]:
    """
    Generate audio segments from a file

    Args:
        file: AudioFile from DataChain

    Yields:
        Clip objects for each channel in each segment
    """
    hop_duration = WINDOW - OVERLAP
    window_samples = int(WINDOW * SAMPLE_RATE)
    hop_samples = int(hop_duration * SAMPLE_RATE)

    # Load audio - DataChain AudioFile doesn't have direct load method
    # We'll use librosa with the file path
    data = io.BytesIO(file.read())
    audio, sr = librosa.load(data, sr=None, mono=False)
    if audio.ndim == 1:
        audio = audio.reshape(1, -1)

    # Resample if needed
    if sr != SAMPLE_RATE:
        audio = librosa.resample(audio, orig_sr=sr, target_sr=SAMPLE_RATE)
        sr = SAMPLE_RATE

    # Get file info
    num_channels = audio.shape[0]
    duration = audio.shape[1] / sr
    channel_type = {1: "mono", 2: "stereo", 4: "foa"}.get(num_channels,
                                                          f"multi_{num_channels}ch")
    # Segment the audio
    segment_index = 0
    segment_counter = 0

    for start_sample in range(0, audio.shape[1] - window_samples + 1, hop_samples):
        segment_counter += 1
        end_sample = start_sample + window_samples
        start_time = start_sample / sr
        end_time = end_sample / sr

        # Extract segment
        segment_audio = audio[:, start_sample:end_sample]

        # Calculate per-channel RMS
        channel_rms = np.sqrt(np.mean(segment_audio ** 2, axis=1))
        mean_rms = np.mean(channel_rms)
        std_rms = np.std(channel_rms)
        min_rms = np.min(channel_rms)
        max_rms = np.max(channel_rms)

        # Calculate spatial info for FOA
        azimuth = elevation = None
        if channel_type == "foa" and num_channels >= 4:
            w, x, y, z = segment_audio[:4]
            intensity = np.sqrt(x ** 2 + y ** 2 + z ** 2).mean()
            if intensity > 0:
                azimuth = np.arctan2(y.mean(), x.mean()) * 180 / np.pi
                elevation = np.arctan2(z.mean(), np.sqrt(
                    x.mean() ** 2 + y.mean() ** 2)) * 180 / np.pi

        for ch_idx, ch_rms in enumerate(channel_rms):
            yield Segment(
                fragment=AudioFragment(audio=file, start=start_time, end=end_time),
                id=segment_index,
                channel=get_channel_name(channel_type, ch_idx),
                rms=ch_rms,
                rms_mean=mean_rms,
                rms_std=std_rms,
                rms_min=min_rms,
                rms_max=max_rms,
                azimuth=azimuth,
                elevation=elevation
            )

        segment_index += 1


if OVERLAP >= WINDOW:
    exit(1)

chain = (
    dc
    .read_storage(STORAGE, type="audio")
    .filter(dc.C("file.path").glob("*.wav"))
    .gen(segm=segment_audio)
    .order_by("segm.fragment.audio.path", "segm.id")
    .save(OUTPUT)
)

if LOCAL:
    dc.read_dataset(OUTPUT).show()
