import argparse
from huggingface_hub import snapshot_download
from pathlib import Path
import sys

parser = argparse.ArgumentParser()
## Download and install medgemma once by decommenting following line
# snapshot_download(repo_id=repo_id, local_dir=data_dir, max_workers=4, token=token)

def download_model(repo_id, local_dir, token):
    local_dir = Path(local_dir) / "data" / "models" / repo_id.split('/')[-1]
    print(f"Downloading model to {local_dir}")
    # sys.exit()
    snapshot_download(
        repo_id=repo_id,
        local_dir=str(local_dir.resolve()),
        max_workers=4,
        token=token,
        local_dir_use_symlinks=False
    )

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Download a Hugging Face repo snapshot.")
    parser.add_argument('--repo_id', required=True, help='The model or dataset repo id.')
    parser.add_argument('--local_dir', required=True, help='Local directory to store the snapshot.')
    parser.add_argument('--token', required=True, help='Your Hugging Face access token.')
    args = parser.parse_args()
    print(vars(args))
    download_model(args.repo_id, args.local_dir, args.token)