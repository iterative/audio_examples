from typing import Iterator

import datachain as dc
from datachain import VideoFile, ImageFile, func
from datachain.model.ultralytics import YoloBBoxes, YoloSegments, YoloPoses

from pydantic import BaseModel
from ultralytics import YOLO

local = False
bucket = "data-video" if local else "s3://datachain-usw2-main-dev"
input_path = f"{bucket}/balanced_train_segments/video"
output_path = f"{bucket}/temp/video-detector-frames"
detection_dataset = "frames-detector"
detection_humans = "human-videos"
target_fps = 1

class YoloDataModel(BaseModel):
    bbox: YoloBBoxes
    segm: YoloSegments
    poses: YoloPoses


class VideoFrameImage(ImageFile):
    num: int
    orig: VideoFile



def extract_frames(file: VideoFile) -> Iterator[VideoFrameImage]:
    info = file.get_info()

    # one frame per sec
    step = int(info.fps / target_fps) if target_fps else 1
    frames = file.get_frames(step=step)

    for num, frame in enumerate(frames):
        image = frame.save(output_path, format="jpg")
        yield VideoFrameImage(**image.model_dump(), num=num, orig=file)


def process_all(yolo: YOLO, yolo_segm: YOLO, yolo_pose: YOLO, frame: ImageFile) -> YoloDataModel:
    img = frame.read()
    return YoloDataModel(
        bbox=YoloBBoxes.from_results(yolo(img, verbose=False)),
        segm=YoloSegments.from_results(yolo_segm(img, verbose=False)),
        poses=YoloPoses.from_results(yolo_pose(img, verbose=False))
    )


def process_bbox(yolo: YOLO, frame: ImageFile) -> YoloBBoxes:
    return YoloBBoxes.from_results(yolo(frame.read(), verbose=False))


# Ultralitics YOLO download is not stable
def yolo_with_retry(model: str) -> YOLO:
    max_retries = 7
    delay_seconds = 2

    for attempt in range(1, max_retries + 1):
        try:
            return YOLO(model)
        except Exception as e:
            if attempt < max_retries:
                time.sleep(delay_seconds)
            else:
                raise e

chain = (
    dc
    .read_storage(input_path, type="video")
    .filter(dc.C("file.path").glob("*.mp4"))
    .sample(7)
    .settings(parallel=5)

    .gen(frame=extract_frames)

    # Initialize models: once per processing thread
    .setup(
        yolo=lambda: yolo_with_retry("yolo11n.pt"),
        yolo_segm=lambda: yolo_with_retry("yolo11n-seg.pt"),
        yolo_pose=lambda: yolo_with_retry("yolo11n-pose.pt")
    )

    # Apply yolo detector to frames
    .map(bbox=process_bbox)
    # .map(yolo=process_all)
    .order_by("frame.path", "frame.num")
    .save(detection_dataset)
)

if local:
    chain.show()
