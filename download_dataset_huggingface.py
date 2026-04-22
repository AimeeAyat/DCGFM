from huggingface_hub import snapshot_download

snapshot_download(
    repo_id="haitengzhao/molecule_property_instruction",
    repo_type="dataset",
    local_dir="OFA/cache_data/dataset/molecule_property_instruction"
)
