# Requirements: datachain[audio], pydub

import tempfile

from pydub import AudioSegment

import datachain as dc

from datachain import AudioFile, File
from datachain.lib.audio import audio_to_bytes

LOCAL = True
BUCKET = "data-flac/datachain-usw2-main-dev" if LOCAL else "s3://datachain-usw2-main-dev"
INPUT_DIR = f"balanced_train_segments/audio"
INPUT_PATH = f"{BUCKET}/{INPUT_DIR}"
OUTPUT_PATH = f"{BUCKET}/test/{INPUT_DIR}"


def convert_flac_to_mp3(file: AudioFile):
    audio = AudioSegment.from_file(file.get_local_path(), format="flac")

    with tempfile.NamedTemporaryFile(suffix=".mp3", delete=True) as tmpfile:
        audio.export(tmpfile.name, format="mp3")
        audio_bytes = audio_to_bytes(file, format="mp3")

    output_file = f"{OUTPUT_PATH}/{file.get_file_stem()}.mp3"
    return AudioFile.upload(audio_bytes, output_file)

(
    dc
    .read_storage(INPUT_PATH, type="audio")
    .filter(dc.C("file.path").glob("*.flac"))
    .map(res=convert_flac_to_mp3)
    .exec()
)

# Update bucket for visibility in Studio UI
if not LOCAL:
    dc.read_storage(BUCKET, update=True).exec()
