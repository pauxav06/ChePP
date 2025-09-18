from huggingface_hub import list_repo_files, hf_hub_download, dataset_info
import os
import os
import json
import time
import requests
import zstandard as zstd
from huggingface_hub import hf_hub_download

import os
import json
import time
import zstandard as zstd
from huggingface_hub import hf_hub_download

RETRY_LIMIT = 5
RETRY_DELAY = 5
TARGET_DIR = "./binpacks"

os.makedirs(TARGET_DIR, exist_ok=True)

def extract_binpack_zst(file_path):
    output_path = file_path.replace(".binpack.zst", "")
    if os.path.exists(output_path):
        print(f"Skipping extraction, already exists: {output_path}")
        return
    try:
        with open(file_path, "rb") as f_in, open(output_path, "wb") as f_out:
            dctx = zstd.ZstdDecompressor()
            dctx.copy_stream(f_in, f_out)
        print(f"Extracted {file_path} -> {output_path}")
    except Exception as e:
        print(f"Failed to extract {file_path}: {e}")

def download_with_retry(repo_id, filename):
    for attempt in range(1, RETRY_LIMIT + 1):
        try:
            file_path = hf_hub_download(
                repo_id=repo_id,
                filename=filename,
                local_dir=TARGET_DIR,
                repo_type="dataset"
            )
            print(f"Downloaded {filename} from {repo_id}")
            return file_path
        except Exception as e:
            print(f"Attempt {attempt}/{RETRY_LIMIT} failed for {filename}: {e}")
            if attempt < RETRY_LIMIT:
                print(f"Retrying in {RETRY_DELAY} seconds...")
                time.sleep(RETRY_DELAY)
            else:
                print(f"Failed to download {filename} from {repo_id}")
                return None
    return None


def main():
    with open("config.json", "r") as f:
        repos = json.load(f)

    for repo in repos:
        repo_id = repo["repo"]
        for filename in repo["files"]:
            print(f"Processing {filename} from {repo_id}")
            file_path = download_with_retry(repo_id, filename)
            if file_path:
                extract_binpack_zst(file_path)

if __name__ == "__main__":
    main()
