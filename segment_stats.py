import numpy as np
import matplotlib.pyplot as plt

import datachain as dc
from datachain import File
from pydantic import BaseModel

LOCAL = False
BUCKET = "data15" if LOCAL else "s3://datachain-usw2-main-dev"
HIST_OUTPUT = "segment_hist" if LOCAL else f"{BUCKET}/test/segment_hist"

INPUT_DATASET = "segments"
OUTPUT_STATS = INPUT_DATASET + f'_stats'
OUTPUT_HIST = INPUT_DATASET + f'_hist'


class RMSStats(BaseModel):
    channel: str
    count: int
    mean: float
    std: float
    min: float
    max: float
    ptl_25: float
    ptl_50: float
    ptl_75: float


def generate_rms_histogram(vals, channel_name, histogram_path):
    plt.figure(figsize=(10, 6))
    plt.hist(vals, bins=50, edgecolor='black', alpha=0.7)
    plt.xlabel('RMS Value')
    plt.ylabel('Frequency')
    plt.title(f'Distribution of RMS Values for Channel {channel_name}')
    plt.grid(True, alpha=0.3)

    plt.savefig(histogram_path, dpi=300, bbox_inches='tight')
    plt.close()


chain = dc.read_dataset(INPUT_DATASET)

stats = []
hist_files = []
for channel in chain.distinct("segm.channel").to_values("segm.channel"):
    chain_w = chain.filter(dc.C("segm.channel") == channel)

    vals = chain_w.to_values("segm.rms")
    stats.append(
        RMSStats(
            channel=channel,
            count=len(vals),
            mean=np.mean(vals),
            std=np.std(vals),
            min=np.min(vals),
            max=np.max(vals),
            ptl_25=np.percentile(vals, 25),
            ptl_50=np.percentile(vals, 50),
            ptl_75=np.percentile(vals, 75)
        )
    )

    histogram_path = f"rms_histogram_{channel}.png"
    generate_rms_histogram(vals, channel, histogram_path)
    hist_files.append(File(source=HIST_OUTPUT, path=histogram_path))


dc.read_values(stats=stats).save(OUTPUT_STATS)
if LOCAL:
    dc.read_dataset(OUTPUT_STATS).show()

dc.read_values(file=hist_files).save(OUTPUT_HIST).to_storage(HIST_OUTPUT)
