# /// script
# dependencies = ["datachain"]
# ///

import datachain as dc

LOCAL = True
BUCKET = "data-flac/datachain-usw2-main-dev" if LOCAL else "s3://datachain-usw2-main-dev"
INPUT_DIR = f"balanced_train_segments/audio"
INPUT_PATH = f"{BUCKET}/{INPUT_DIR}"
OUTPUT_PATH = f"{BUCKET}/test/{INPUT_DIR}"

(
    dc
    .read_storage(INPUT_PATH, type="audio")
    .filter(dc.C("file.path").glob("*.flac"))
    .map(res=lambda file: file.save(OUTPUT_PATH, format="mp3"))
    .exec()
)

# Update bucket for visibility in Studio UI
if not LOCAL:
    dc.read_storage(BUCKET, update=True).exec()
