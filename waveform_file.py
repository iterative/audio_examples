# /// script
# dependencies = [
#   "numpy",
#	"librosa",
#   "datachain[video,audio]",
# ]
# ///

import io
import os
from pathlib import PurePosixPath
from typing import Iterator, ClassVar

import numpy as np
import librosa
from pydantic import BaseModel

import datachain as dc
from datachain import AudioFile, Audio, File

LOCAL = True
INPUT = "data-flac-full/datachain-usw2-main-dev/balanced_train_segments/" \
                if LOCAL else "s3://datachain-usw2-main-dev/balanced_train_segments/"
LIMIT = 10
INPUT_BASE_DIR = "balanced_train_segments"
OUTPUT_BASE_DIR = "data-flac-full-waveform" if LOCAL \
    else "s3://datachain-usw2-main-dev/balanced_train_segments-waveform/"

OUTPUT_DATASET = "waveform-files"
SAMPLE_RATE = None  # None to keep original sample rate


def rebase_path(
    src_path,
    old_base,
    new_base,
    suffix: str = "",
    extension: str = "",
) -> str:
    """
    Return a new file path in `new_base`, preserving the relative path from `old_base`.
    Optionally change the file extension and append a suffix to the filename.
    """
    src_path = str(src_path)
    old_base = str(old_base)
    new_base = str(new_base)

    # Preserve URI scheme like "s3://"
    if "://" in new_base:
        scheme, out_rest = new_base.split("://", 1)
        output_prefix = f"{scheme}://"
        output_base = PurePosixPath(out_rest)
    else:
        output_prefix = ""
        output_base = PurePosixPath(new_base)

    if old_base not in src_path:
        raise ValueError(f"old_base '{old_base}' not found in src_path '{src_path}'")

    relative_str = src_path.split(old_base, 1)[1].lstrip("/")
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

        uri = file.get_uri()
        output_filename = rebase_path(uri, INPUT_BASE_DIR, OUTPUT_BASE_DIR,
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
    .read_storage(INPUT, type="audio")
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
