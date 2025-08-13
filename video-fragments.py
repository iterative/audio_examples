# /// script
# dependencies = [
#   "datachain[video,audio]",
# ]
# ///

from typing import Iterator

import datachain as dc
from datachain import VideoFile, VideoFragment

local = False
bucket = "data-video" if local else "s3://datachain-usw2-main-dev"
input_path = f"{bucket}/balanced_train_segments/video"
output_path = f"{bucket}/temp/video-fragments"
fragments_dataset = "video-fragments"
segment_duration = 7


class VideoClip(VideoFile):
    orig: VideoFragment


def video_fragments(file: VideoFile) -> Iterator[VideoClip]:
    for fragment in file.get_fragments(segment_duration):
        clip = fragment.save(output_path)
        yield VideoClip(**clip.model_dump(), orig=fragment)


chain = (
    dc
    .read_storage(input_path, type="video")
    .filter(dc.C("file.path").glob("*.mp4"))
    .sample(10)
    .settings(parallel=5)

    .gen(clip=video_fragments)

    .order_by("clip.path", "clip.orig.start")
    .save(fragments_dataset)
)

if local:
    chain.show()
