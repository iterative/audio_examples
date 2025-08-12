from typing import Iterator

import datachain as dc
from datachain import VideoFile, ImageFile, VideoFragment
from datachain.model.ultralytics import YoloBBoxes, YoloSegments, YoloPoses

from pydantic import BaseModel
from ultralytics import YOLO

local = False
bucket = "data-video" if local else "s3://datachain-usw2-main-dev"
input_path = f"{bucket}/balanced_train_segments/video"
output_path = f"{bucket}/temp/video-fragments"
detection_dataset = "yolo-frames-detector"
segment_duration = 5


def video_fragnebts(file: VideoFile) -> Iterator[VideoFragment]:
    yield from file.get_fragments(segment_duration)


chain = (
    dc
    .read_storage(input_path, type="video")
    .filter(dc.C("file.path").glob("*.mp4"))
    .limit(2)
    .settings(parallel=5)

    .gen(frame=extract_frames)

    # Initialize models: once per processing thread
    .setup(
        yolo=lambda: YOLO("yolo11n.pt"),
        yolo_segm=lambda: YOLO("yolo11n-seg.pt"),
        yolo_pose=lambda: YOLO("yolo11n-pose.pt")
    )

    # Apply yolo detector to frames
    # .map(bbox=process_bbox)
    .map(yolo=video_segments)
    .order_by("frame.path", "frame.num")
    .save(detection_dataset)
)

if local:
    chain.show()
