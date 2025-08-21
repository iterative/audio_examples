# /// script
# dependencies = [
#	"pydub",
#   "datachain[audio]",
# ]
# ///

from io import BytesIO
from typing import Iterator

from pydantic import BaseModel
from pydub import AudioSegment

import datachain as dc
from datachain import AudioFile, AudioFragment

RMS_THRESHOLD = 0.006
EPS = 1e-9
FORMAT = "wav"

INPUT_DATASET = "segments"
OUTPUT_DATASET = "segments_trimmed"

LOCAL = False

BUCKET = "s3://datachain-usw2-main-dev/"
INPUT = "data-sony_av_data/" if LOCAL else f"{BUCKET}/sony_av_data"
OUTPUT = f"data-sony-out_{RMS_THRESHOLD:.4f}" if LOCAL \
    else f"{BUCKET}/temp/my_sony_av_data_trimmed_{RMS_THRESHOLD:.4f}"

class TrimmedResult(BaseModel):
    audio: AudioFile
    orig_duration: float
    new_duration: float
    orig_intervals: list[list[float]]
    new_intervals: list[list[float]]


def agg_segments(
        fragment: list[AudioFragment],
        channel: list[int]
) -> Iterator[TrimmedResult]:
    file = fragment[0].audio

    intervals = get_intervals(channel, fragment)
    merged_intervals = merge_intervals(intervals)

    buf = trim_audio(file, merged_intervals)
    upload_to = file.rebase(INPUT, OUTPUT, extension=FORMAT)
    res_file = AudioFile.upload(buf.read(), upload_to)

    orig_duration = max(e for _, e in intervals) - min(s for s, _ in intervals)
    new_duration = sum([e - s for s, e in merged_intervals])

    yield TrimmedResult(
        audio=AudioFile(**res_file.model_dump()),
        orig_duration=orig_duration,
        new_duration=new_duration,
        orig_intervals=intervals,
        new_intervals=merged_intervals,
    )


def get_intervals(channel, fragment):
    intervals = []
    for ch, fragm in zip(channel, fragment):
        if ch == "Ch1":
            intervals.append((fragm.start, fragm.end))
    intervals.sort(key=lambda x: x[0])
    return intervals


def trim_audio(file: AudioFile, merged):
    buffer = BytesIO(file.read())
    audio = AudioSegment.from_file(buffer, FORMAT)
    out = AudioSegment.silent(duration=0, frame_rate=audio.frame_rate)
    dur_ms = len(audio)
    for s, e in merged:
        s_ms = int(round(s * 1000))
        e_ms = int(round(e * 1000))
        s_ms = max(0, min(s_ms, dur_ms))
        e_ms = max(0, min(e_ms, dur_ms))
        if e_ms > s_ms:
            out += audio[s_ms:e_ms]
    buf = BytesIO()
    out.export(buf, format=FORMAT)
    buf.seek(0)
    return buf


def merge_intervals(intervals):
    merged = []
    cur_start, cur_end = intervals[0]
    for s, e in intervals[1:]:
        if s <= cur_end + EPS:  # overlaps or touches
            cur_end = max(cur_end, e)
        else:
            merged.append((cur_start, cur_end))
            cur_start, cur_end = s, e
    merged.append((cur_start, cur_end))
    return merged


chain = (
    dc
    .read_dataset(INPUT_DATASET)
    .filter(dc.C("segm.rms_mean") > RMS_THRESHOLD)
    # .limit(50)
    .settings(parallel=True)
    .agg(
        trimmed=agg_segments,
        partition_by="segm.fragment.audio",
        params=["segm.fragment", "segm.channel"]
    )
    .save(OUTPUT_DATASET)
)

if LOCAL:
    chain.show()
