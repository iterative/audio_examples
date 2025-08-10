from typing import Iterator

import datachain as dc
from datachain import VideoFile, ImageFile
from datachain.model.ultralytics import YoloBBoxes, YoloSegments, YoloPoses

from pydantic import BaseModel
from ultralytics import YOLO

local = True
bucket = "data-video/" if local else "s3://datachain-usw2-main-dev/balanced_train_segments/video"
output_path = f"{bucket}/temp/frames/"
detection_dataset = "yolo-detector"


class YoloDataModel(BaseModel):
    bbox: YoloBBoxes
    segm: YoloSegments
    poses: YoloPoses


class VideoFrameImage(ImageFile):
    num: int
    video_ref: VideoFile


def extract_frames(file: VideoFile) -> Iterator[VideoFrameImage]:
    info = file.get_info()

    # one frame per sec
    frames = file.get_frames(step=int(info.fps))

    for num, frame in enumerate(frames):
        image = frame.save(output_path, format="jpg")
        yield VideoFrameImage(**image.model_dump(), num=num, video_ref=file)


def process_yolo(yolo:YOLO, yolo_segm: YOLO, yolo_pose: YOLO, frame: ImageFile) -> YoloDataModel:
    img = frame.read()
    return YoloDataModel(
        bbox=YoloBBoxes.from_results(yolo(img, verbose=False)),
        segm=YoloSegments.from_results(yolo_segm(img, verbose=False)),
        poses=YoloPoses.from_results(yolo_pose(img, verbose=False))
    )

chain = (
    dc
    .read_storage(bucket, type="video")
    .filter(dc.C("file.path").glob("*.mp4"))
    .limit(2)
    .settings(parallel=4)

    .gen(frame=extract_frames)

    # Apply yolo detector to frames
  	.setup(
        yolo=lambda: YOLO("yolo11n.pt"),
        yolo_segm=lambda: YOLO("yolo11n-seg.pt"),
        yolo_pose=lambda: YOLO("yolo11n-pose.pt")
    )
    .map(yolo=process_yolo)
    .order_by("frame.path", "frame.num")
    .save(detection_dataset)
)

if local:
    chain.show()
