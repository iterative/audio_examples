import tempfile

import numpy as np
import datachain as dc
import cv2
from datachain import VideoFile, File

local = False
bucket = "data-video/" if local else "s3://datachain-usw2-main-dev"
output_path = f"{bucket}/temp/noisy-video"
output_dataset = "noisy-video"


def add_gaussian_noise_to_video(file: VideoFile, mean, stddev) -> VideoFile:
    """
    Reads a video, adds noise to each frame, and saves the modified video.
    Args:
        input_video_path (str): Path to the input video file.
        output_video_path (str): Path to save the output video file.
        noise_type (str): Type of noise to add ("gaussian", "salt_and_pepper", or "random").
        **kwargs: Additional arguments for noise generation (e.g., mean, stddev for Gaussian noise).
    """

    input_video_path = file.get_local_path()
    cap = cv2.VideoCapture(input_video_path)
    if not cap.isOpened():
        raise FileNotFoundError(f"Error: Could not process video file {file.get_uri()}")
    # Get video properties (width, height, frames per second)
    frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = int(cap.get(cv2.CAP_PROP_FPS))

    # Define the codec and create a VideoWriter object
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')  # Codec for MP4 files

    with tempfile.NamedTemporaryFile(suffix=".mp4") as tmp:
        out = cv2.VideoWriter(tmp.name, fourcc, fps, (frame_width, frame_height))
        while True:
            ret, frame = cap.read()  # Read each frame
            if not ret:
                break  # Break if no more frames to read
            # Add noise to the current frame based on the specified noise_type
            gauss_noise = np.random.normal(mean, stddev, frame.shape).astype(np.uint8)
            noisy_frame = cv2.add(frame, gauss_noise)
            out.write(noisy_frame)  # Write the noisy frame to the output video
        cap.release()  # Release the VideoCapture object
        out.release()  # Release the VideoWriter object
        cv2.destroyAllWindows()  # Destroy all OpenCV windows

        upload_to = file.rebase(bucket, output_path)
        return File.upload(open(tmp.name, "rb").read(), upload_to)

chain = (
    dc
    .read_storage(bucket, type="video")
    .filter(dc.C("file.path").glob("*.mp4"))
    .limit(7)
    .settings(parallel=4)
    .map(segm=lambda file: add_gaussian_noise_to_video(file, 0, 25),
         output=File)
    .save(output_dataset)
)

chain.show()
