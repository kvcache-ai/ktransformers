from modelscope.hub.snapshot_download import snapshot_download

snapshot_download(
    model_id="Qwen/Qwen3-VL-30B-A3B-Instruct",
    local_dir="/mnt/data2/models/Qwen3-VL-30B-A3B-Instruct",
    allow_file_pattern=None
)
