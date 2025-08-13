import datachain as dc


chain = (
    dc
    .read_storage("s3://datachain-usw2-main-dev/")
    .filter(dc.C("file.path").glob("*.flac"))
    .limit(3000)
    .save(name='dataset-flac')
)

for (name, size) in chain.to_list("file.path", "file.size"):
    print(name, size)

chain.to_storage("./data-flac/")
