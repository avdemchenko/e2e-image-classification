"""
Data processing utilities for CIFAR-10 dataset.

This module contains functions for downloading, preprocessing, and creating
datasets for the CIFAR-10 classification task.
"""

import pickle
import shutil
import tarfile
from pathlib import Path
from typing import List, Tuple

import requests
import numpy as np
from PIL import Image
from torch.utils.data import Dataset


# ---------------------------------------------------------------------
# Constants & URLs
# ---------------------------------------------------------------------
CIFAR10_URL = "https://www.cs.toronto.edu/~kriz/cifar-10-python.tar.gz"
CIFAR10_LABELS = [
    "plane",
    "car",
    "bird",
    "cat",
    "deer",
    "dog",
    "frog",
    "horse",
    "ship",
    "truck",
]


# ---------------------------------------------------------------------
# Data Download & Preprocessing Functions
# ---------------------------------------------------------------------

def download_and_unpack_cifar10(raw_data_dir: Path, extracted_dir: Path) -> None:
    """Download and extract CIFAR-10 if not already present."""
    raw_data_dir.mkdir(parents=True, exist_ok=True)
    tar_path = raw_data_dir / "cifar-10-python.tar.gz"

    if extracted_dir.exists():
        print("[Data] CIFAR-10 already extracted.")
        return

    if not tar_path.exists():
        print("[Data] Downloading CIFAR-10 dataset…")
        with requests.get(CIFAR10_URL, stream=True, timeout=60) as r:
            r.raise_for_status()
            with open(tar_path, "wb") as f_out:
                for chunk in r.iter_content(chunk_size=8192):
                    if chunk:
                        f_out.write(chunk)
        print("[Data] Download complete.")

    print("[Data] Extracting CIFAR-10…")
    with tarfile.open(tar_path, "r:gz") as tar:
        tar.extractall(path=raw_data_dir)
    print("[Data] Extraction finished.")


def _save_image(np_img: np.ndarray, label: int, split: str, idx: int, 
                train_subdir: Path, test_subdir: Path) -> None:
    """Helper to save a single image to PNG format under the correct directory."""
    # reshape to (3,32,32) then transpose to (32,32,3)
    img_arr = np_img.reshape(3, 32, 32).transpose(1, 2, 0)
    img = Image.fromarray(img_arr)
    class_dir = (train_subdir if split == "train" else test_subdir) / f"{label}_{CIFAR10_LABELS[label]}"
    class_dir.mkdir(parents=True, exist_ok=True)
    img.save(class_dir / f"{split}_{idx}.png")


def create_image_file_structure(extracted_dir: Path, processed_dir: Path, 
                               train_subdir: Path, test_subdir: Path) -> None:
    """Convert CIFAR-10 batch files to an image folder structure understandable by our custom Dataset."""
    if train_subdir.exists() and test_subdir.exists():
        print("[Data] Processed image folders already exist.")
        return

    if not extracted_dir.exists():
        raise RuntimeError("Raw CIFAR-10 data not found. Please run download procedure first.")

    print("[Data] Creating PNG files from raw batches…")
    # Remove any previous partially-processed data
    if processed_dir.exists():
        shutil.rmtree(processed_dir)

    # Process training batches 1-5
    train_idx = 0
    for batch_id in range(1, 6):
        batch_path = extracted_dir / f"data_batch_{batch_id}"
        with open(batch_path, "rb") as f:
            batch = pickle.load(f, encoding="bytes")
        data = batch[b"data"]
        labels = batch[b"labels"]
        for img_arr, label in zip(data, labels):
            _save_image(img_arr, label, "train", train_idx, train_subdir, test_subdir)
            train_idx += 1

    # Process test batch
    test_idx = 0
    test_path = extracted_dir / "test_batch"
    with open(test_path, "rb") as f:
        batch = pickle.load(f, encoding="bytes")
    data = batch[b"data"]
    labels = batch[b"labels"]
    for img_arr, label in zip(data, labels):
        _save_image(img_arr, label, "test", test_idx, train_subdir, test_subdir)
        test_idx += 1

    print("[Data] Image files created successfully.")


# ---------------------------------------------------------------------
# Dataset Class
# ---------------------------------------------------------------------

class CustomImageDataset(Dataset):
    """Custom dataset that recursively reads PNG images and returns tensors."""

    def __init__(self, root_dir: Path, transform=None):
        self.root_dir = Path(root_dir)
        self.transform = transform

        if not self.root_dir.exists():
            raise RuntimeError(f"Dataset directory {self.root_dir} not found. Have you run preprocessing?")

        # Recursively gather image paths and labels
        self.samples: List[Tuple[Path, int]] = []
        for class_path in sorted(self.root_dir.iterdir()):
            if not class_path.is_dir():
                continue
            label = int(class_path.name.split("_")[0])
            for img_path in class_path.rglob("*.png"):
                self.samples.append((img_path, label))

        if len(self.samples) == 0:
            raise RuntimeError(f"No PNG files found under {self.root_dir}.")

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        img_path, label = self.samples[idx]
        image = Image.open(img_path).convert("RGB")
        if self.transform:
            image = self.transform(image)
        return image, label
