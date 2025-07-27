# Requirements: datachain[audio], librosa, numpy

import io
import os
from pathlib import PurePosixPath
from typing import Iterator, ClassVar

import numpy as np
import librosa
from pydantic import BaseModel

import datachain as dc
from datachain import AudioFile, Audio, File

LOCAL = False
STORAGE = "data-flac-full/datachain-usw2-main-dev/balanced_train_segments/audio/" \
                if LOCAL else "s3://datachain-usw2-main-dev/balanced_train_segments"
LIMIT = 10
OUTPUT_BUCKET = "" if LOCAL else "s3://datachain-usw2-main-dev"
OUTPUT_DIR = f"{STORAGE}/waveforms"
OUTPUT_DATASET = "waveform-files"
SAMPLE_RATE = None  # None to keep original sample rate


def relocate_path(
    path,
    base_dir,
    output_dir,
    suffix: str = "",
    extension: str = "",
) -> str:
    """
    Return a new file path in `output_dir`, preserving the relative path from `base_dir`.
    Optionally change the file extension and append a suffix to the filename.
    """
    path = str(path)
    base_dir = str(base_dir)
    output_dir = str(output_dir)

    # Preserve URI scheme like "s3://"
    if "://" in output_dir:
        scheme, out_rest = output_dir.split("://", 1)
        output_prefix = f"{scheme}://"
        output_base = PurePosixPath(out_rest)
    else:
        output_prefix = ""
        output_base = PurePosixPath(output_dir)

    if base_dir not in path:
        raise ValueError(f"base_dir '{base_dir}' not found in path '{path}'")

    relative_str = path.split(base_dir, 1)[1].lstrip("/")
    relative_path = PurePosixPath(relative_str)

    stem = relative_path.stem
    parent = relative_path.parent

    if suffix:
        stem += suffix

    if extension:
        filename = f"{stem}.{extension}"
    else:
        filename = f"{stem}{relative_path.suffix}"

    final_path = output_prefix + str(output_base / parent / filename)
    return final_path


class Waveform(BaseModel):
    file: AudioFile
    filename: str  # Just the filename for easy lookup
    info: Audio
    channel: int
    channel_name: str
    waveform_file: File

    DTYPE: ClassVar[str] = "float32"  # Constant data type for all waveforms


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
        channel_name = get_channel_name(num_channels, ch_idx)
        
        # Save as numpy binary array
        buffer = io.BytesIO()
        np.save(buffer, channel_data)
        buffer.seek(0)

        output = f"{OUTPUT_BUCKET}//{OUTPUT_DIR}" if OUTPUT_BUCKET else OUTPUT_DIR
        uri = file.get_uri()
        output_filename = relocate_path(uri, STORAGE, output,
                                        f"_ch{ch_idx}", "npy")

        wave_file = File.upload(buffer.read(), output_filename)
        print(f"{file.source}, {file.path} --> {wave_file.source}, {wave_file.path}")

        yield Waveform(
            file=file,
            filename=os.path.basename(file.path),
            channel=ch_idx,
            channel_name=channel_name,
            info=audio_info,
            waveform_file=wave_file
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
    .save(OUTPUT_DATASET)
)

if LOCAL:
    dc.read_dataset(OUTPUT_DATASET).show()
