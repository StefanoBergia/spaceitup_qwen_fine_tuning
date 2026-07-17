"""Download the ShareRobot trajectory dataset and Qwen3.5-2B weights.

Run on the login node (GPU nodes share the same NFS, so downloads are visible there):

    uv run scripts/download.py                 # dataset + model
    uv run scripts/download.py --dataset-only
    uv run scripts/download.py --model-only

Dataset  -> data/sharerobot/trajectory/   (only the trajectory subset, ~6,870 images)
Model    -> default Hugging Face cache (~/.cache/huggingface)

Both downloads are idempotent: re-running resumes/skips already-downloaded files.
"""

import argparse
import os
from pathlib import Path

# Faster parallel downloads via hf_transfer (installed as an extra of huggingface_hub)
os.environ.setdefault("HF_HUB_ENABLE_HF_TRANSFER", "1")

from huggingface_hub import snapshot_download

REPO_ROOT = Path(__file__).resolve().parent.parent
DATASET_REPO = "BAAI/ShareRobot"
DATASET_DIR = REPO_ROOT / "data" / "sharerobot"
MODEL_REPO = "Qwen/Qwen3.5-2B"


def download_dataset() -> None:
    print(f"Downloading {DATASET_REPO} (trajectory subset) -> {DATASET_DIR}")
    snapshot_download(
        repo_id=DATASET_REPO,
        repo_type="dataset",
        allow_patterns=["trajectory/**"],
        local_dir=DATASET_DIR,
    )
    ann = DATASET_DIR / "trajectory" / "trajectory.json"
    n_images = sum(1 for _ in (DATASET_DIR / "trajectory" / "images").rglob("*.png"))
    print(f"Done. Annotations: {ann} (exists={ann.exists()}), images: {n_images}")


def download_model() -> None:
    print(f"Downloading {MODEL_REPO} -> HF cache")
    path = snapshot_download(repo_id=MODEL_REPO)
    print(f"Done. Cached at: {path}")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    group = parser.add_mutually_exclusive_group()
    group.add_argument("--dataset-only", action="store_true", help="download only the dataset")
    group.add_argument("--model-only", action="store_true", help="download only the model")
    args = parser.parse_args()

    if not args.model_only:
        download_dataset()
    if not args.dataset_only:
        download_model()


if __name__ == "__main__":
    main()
