import datachain as dc
from datachain.func.array import contains

class_names = ["person", "handbag", "car", "truck"]
input_dataset = "frames-detector"
stats_dataset = "detector-stats"

chain = dc.read_dataset(input_dataset)

total_frames = chain.count()
total_videos = chain.distinct("frame.orig").count()

dc.read_values(
    class_name = class_names,
    frame_coverage = [
        chain.filter(contains("bbox.name", name)).count()*1.0/total_frames
        for name in class_names
    ],
    video_coverage = [
        chain.filter(contains("bbox.name", name)).distinct("frame.orig").count()*1.0/total_videos
        for name in class_names
    ],
).save(stats_dataset)
