# /// script
# dependencies = ["datachain"]
# ///

import datachain as dc
from datachain.func.array import contains

target_class = "person"
input_dataset = "frames-detector"
output_dataset = "detector-human-videos"

chain_humans = (
    dc.read_dataset(input_dataset)
    .filter(contains("bbox.name", target_class))

    # Select only signals that are required
    .mutate(file=dc.C("frame.orig"))
    .select("file")

    # Remove file duplicats
    .distinct("file")
    .save(output_dataset)
)
