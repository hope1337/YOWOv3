from huggingface_hub import HfApi
api = HfApi()

api.upload_folder(
    folder_path="/home/manh/Datasets/ava",
    repo_id="manh6054/AVAv2.2",
    repo_type="dataset",
)

