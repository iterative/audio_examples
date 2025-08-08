import tempfile

import numpy as np
import datachain as dc
import cv2
from datachain import VideoFile, File

local = False
bucket = "data-video/" if local else "s3://datachain-usw2-main-dev/balanced_train_segments/video"
output_path = f"{bucket}/temp/noisy-video"
output_dataset = "noisy-video"


def new_dimensions(width, height, new_width=None, new_height=None, percent=None):
    """
    Calculates new dimensions for resizing based on various parameters.
    
    Args:
        width (int): Original width of the frame/video.
        height (int): Original height of the frame/video.
        new_width (int, optional): The desired new width.
        new_height (int, optional): The desired new height.
        percent (float, optional): The percentage to scale (e.g., 50 for 50%).
    
    Returns:
        tuple: (width, height) dimensions after resizing.
    """
    if percent is not None:
        width = int(width * percent / 100)
        height = int(height * percent / 100)
    elif new_width is not None and new_height is not None:
        width = new_width
        height = new_height
    elif new_width is not None:
        # Calculate height to maintain aspect ratio
        ratio = new_width / float(width)
        width = new_width
        height = int(height * ratio)
    elif new_height is not None:
        # Calculate width to maintain aspect ratio
        ratio = new_height / float(height)
        width = int(width * ratio)
        height = new_height
    else:
        # No resizing specified
        width = width
        height = height

    return width, height


def add_gaussian_noise_to_video_and_normalize(file: VideoFile, mean, stddev) -> VideoFile:
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

    output_dim = new_dimensions(frame_width, frame_height, percent=50)

    # Define the codec and create a VideoWriter object with the calculated dimensions
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')  # Codec for MP4 files

    with tempfile.NamedTemporaryFile(suffix=".mp4") as tmp:
        out = cv2.VideoWriter(tmp.name, fourcc, fps, output_dim)
        while True:
            ret, frame = cap.read()  # Read each frame

            if not ret:
                break  # Break if no more frames to read

            if output_dim != (frame.shape[1], frame.shape[0]):
                frame = cv2.resize(frame, output_dim, interpolation=cv2.INTER_AREA)

            # Add noise to the current frame based on the specified noise_type
            gauss_noise = np.random.normal(mean, stddev, frame.shape).astype(np.uint8)
            noisy_frame = cv2.add(frame, gauss_noise)
            noisy_frame = cv2.normalize(noisy_frame, None, 0, 1.0, cv2.NORM_MINMAX)
            out.write(noisy_frame)  # Write the noisy frame to the output video
        cap.release()  # Release the VideoCapture object
        out.release()  # Release the VideoWriter object
        cv2.destroyAllWindows()  # Destroy all OpenCV windows

        upload_to = file.rebase(bucket, output_path)
        tmp.seek(0)
        return File.upload(tmp.read(), upload_to)

chain = (
    dc
    .read_storage(bucket, type="video")
    .filter(dc.C("file.path").glob("*.mp4"))
    .limit(7)
    .settings(parallel=4)
    .map(segm=lambda file: add_gaussian_noise_to_video_and_normalize(file, 0, 25),
         output=File)
    .save(output_dataset)
)

chain.show()
