import random
import tarfile
import pickle
import shutil
from pathlib import Path
from typing import Tuple, List, Dict

import requests
import numpy as np
from PIL import Image

import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim.lr_scheduler import CosineAnnealingWarmRestarts
from torch.utils.data import Dataset, DataLoader, random_split
from torchvision import transforms
from torch.utils.tensorboard import SummaryWriter
from tqdm.auto import tqdm

from torchmetrics.classification import (
    MulticlassAccuracy,
    MulticlassPrecision,
    MulticlassRecall,
    MulticlassF1Score,
)

import argparse  # kept for legacy but not used when running with Hydra
from omegaconf import DictConfig, OmegaConf
import hydra
from hydra.utils import to_absolute_path

# ---------------------------------------------------------------------
# Constants & URLs
# ---------------------------------------------------------------------
CIFAR10_URL = "https://www.cs.toronto.edu/~kriz/cifar-10-python.tar.gz"
RAW_DATA_DIR = Path("./data_raw")
EXTRACTED_DIR = RAW_DATA_DIR / "cifar-10-batches-py"
PROCESSED_DIR = Path("./my_cifar_data")
TRAIN_SUBDIR = PROCESSED_DIR / "train"
TEST_SUBDIR = PROCESSED_DIR / "test"
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
# Utility Functions
# ---------------------------------------------------------------------

def set_seed(seed: int) -> None:
    """Set random seed for reproducibility."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def download_and_unpack_cifar10() -> None:
    """Download and extract CIFAR-10 if not already present."""
    RAW_DATA_DIR.mkdir(parents=True, exist_ok=True)
    tar_path = RAW_DATA_DIR / "cifar-10-python.tar.gz"

    if EXTRACTED_DIR.exists():
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
        tar.extractall(path=RAW_DATA_DIR)
    print("[Data] Extraction finished.")


def _save_image(np_img: np.ndarray, label: int, split: str, idx: int) -> None:
    """Helper to save a single image to PNG format under the correct directory."""
    # reshape to (3,32,32) then transpose to (32,32,3)
    img_arr = np_img.reshape(3, 32, 32).transpose(1, 2, 0)
    img = Image.fromarray(img_arr)
    class_dir = (TRAIN_SUBDIR if split == "train" else TEST_SUBDIR) / f"{label}_{CIFAR10_LABELS[label]}"
    class_dir.mkdir(parents=True, exist_ok=True)
    img.save(class_dir / f"{split}_{idx}.png")


def create_image_file_structure() -> None:
    """Convert CIFAR-10 batch files to an image folder structure understandable by our custom Dataset."""
    if TRAIN_SUBDIR.exists() and TEST_SUBDIR.exists():
        print("[Data] Processed image folders already exist.")
        return

    if not EXTRACTED_DIR.exists():
        raise RuntimeError("Raw CIFAR-10 data not found. Please run download procedure first.")

    print("[Data] Creating PNG files from raw batches…")
    # Remove any previous partially-processed data
    if PROCESSED_DIR.exists():
        shutil.rmtree(PROCESSED_DIR)

    # Process training batches 1-5
    train_idx = 0
    for batch_id in range(1, 6):
        batch_path = EXTRACTED_DIR / f"data_batch_{batch_id}"
        with open(batch_path, "rb") as f:
            batch = pickle.load(f, encoding="bytes")
        data = batch[b"data"]
        labels = batch[b"labels"]
        for img_arr, label in zip(data, labels):
            _save_image(img_arr, label, "train", train_idx)
            train_idx += 1

    # Process test batch
    test_idx = 0
    test_path = EXTRACTED_DIR / "test_batch"
    with open(test_path, "rb") as f:
        batch = pickle.load(f, encoding="bytes")
    data = batch[b"data"]
    labels = batch[b"labels"]
    for img_arr, label in zip(data, labels):
        _save_image(img_arr, label, "test", test_idx)
        test_idx += 1

    print("[Data] Image files created successfully.")


# ---------------------------------------------------------------------
# Dataset
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


# ---------------------------------------------------------------------
# Improved Model with BatchNorm and Deeper Architecture
# ---------------------------------------------------------------------

class ImprovedNet(nn.Module):
    def __init__(self, num_classes: int = 10, dropout: float = 0.5):
        super().__init__()

        self.features = nn.Sequential(
            # Block 1: 3x32x32 -> 64x32x32 -> 64x16x16
            nn.Conv2d(3, 64, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.Conv2d(64, 64, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2, 2),
            nn.Dropout2d(0.1),

            # Block 2: 64x16x16 -> 128x16x16 -> 128x8x8
            nn.Conv2d(64, 128, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            nn.Conv2d(128, 128, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2, 2),
            nn.Dropout2d(0.2),

            # Block 3: 128x8x8 -> 256x8x8 -> 256x4x4
            nn.Conv2d(128, 256, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),
            nn.Conv2d(256, 256, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2, 2),
            nn.Dropout2d(0.3),

            # Block 4: 256x4x4 -> 512x4x4 -> 512x2x2
            nn.Conv2d(256, 512, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(512),
            nn.ReLU(inplace=True),
            nn.Conv2d(512, 512, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(512),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2, 2),
        )

        self.classifier = nn.Sequential(
            nn.AdaptiveAvgPool2d((1, 1)),  # Global Average Pooling
            nn.Flatten(),
            nn.Dropout(dropout),
            nn.Linear(512, 256),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout * 0.6),
            nn.Linear(256, 128),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout * 0.3),
            nn.Linear(128, num_classes),
        )

    def forward(self, x):
        x = self.features(x)
        return self.classifier(x)


# ---------------------------------------------------------------------
# Training & Evaluation Utils
# ---------------------------------------------------------------------

def get_latest_checkpoint(ckpt_dir: Path) -> Path:
    """Return path to the latest checkpoint in directory or None."""
    ckpts = sorted(ckpt_dir.glob("checkpoint_epoch_*.pth"))
    return ckpts[-1] if ckpts else None


def save_checkpoint(state: Dict, ckpt_dir: Path, epoch: int) -> Path:
    ckpt_dir.mkdir(parents=True, exist_ok=True)
    ckpt_path = ckpt_dir / f"checkpoint_epoch_{epoch}.pth"
    torch.save(state, ckpt_path)
    return ckpt_path


def train_one_epoch(model, loader, criterion, optimizer, device, epoch_idx: int = None):
    """Train for a single epoch with tqdm progress bar."""
    model.train()
    running_loss = 0.0
    desc = f"Epoch {epoch_idx+1} [Train]" if epoch_idx is not None else "Train"
    loop = tqdm(loader, desc=desc, leave=False)
    for inputs, targets in loop:
        inputs, targets = inputs.to(device), targets.to(device)
        optimizer.zero_grad()
        outputs = model(inputs)
        loss = criterion(outputs, targets)
        loss.backward()

        # Gradient clipping for stability
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)

        optimizer.step()

        running_loss += loss.item() * inputs.size(0)
        loop.set_postfix(loss=loss.item())

    epoch_loss = running_loss / len(loader.dataset)
    return epoch_loss


def validate(model, loader, criterion, device, metrics, epoch_idx: int = None):
    """Validate model with tqdm progress bar."""
    model.eval()
    running_loss = 0.0
    for m in metrics.values():
        m.reset()

    desc = f"Epoch {epoch_idx+1} [Val]" if epoch_idx is not None else "Val"
    loop = tqdm(loader, desc=desc, leave=False)

    with torch.no_grad():
        for inputs, targets in loop:
            inputs, targets = inputs.to(device), targets.to(device)
            outputs = model(inputs)
            loss = criterion(outputs, targets)
            running_loss += loss.item() * inputs.size(0)
            preds = torch.argmax(outputs, dim=1)
            metrics["accuracy"].update(preds, targets)
            metrics["precision"].update(preds, targets)
            metrics["recall"].update(preds, targets)
            metrics["f1"].update(preds, targets)

    val_loss = running_loss / len(loader.dataset)
    results = {k: float(m.compute()) for k, m in metrics.items()}
    return val_loss, results


# ---------------------------------------------------------------------
# Argument Parsing
# ---------------------------------------------------------------------

def get_args():
    parser = argparse.ArgumentParser(description="CIFAR-10 Training Pipeline")
    parser.add_argument("--lr", type=float, default=0.001, help="Learning rate")
    parser.add_argument("--epochs", type=int, default=100, help="Number of epochs")
    parser.add_argument("--batch-size", type=int, default=128, help="Batch size")
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    parser.add_argument(
        "--checkpoint-dir",
        type=str,
        default="./checkpoints_improved",
        help="Directory for saving checkpoints",
    )
    parser.add_argument(
        "--log-dir",
        type=str,
        default="./runs_improved",
        help="TensorBoard log directory",
    )
    return parser.parse_args()


# ---------------------------------------------------------------------
# Main Function
# ---------------------------------------------------------------------

@hydra.main(version_base=None, config_path="../conf", config_name="config")
def main(cfg: DictConfig):
    # Optional: print resolved config
    print(OmegaConf.to_yaml(cfg))

    set_seed(cfg.seed)

    # Ensure all paths are absolute (Hydra changes the working dir)
    global RAW_DATA_DIR, EXTRACTED_DIR, PROCESSED_DIR, TRAIN_SUBDIR, TEST_SUBDIR
    RAW_DATA_DIR = Path(to_absolute_path(cfg.raw_data_dir))
    EXTRACTED_DIR = RAW_DATA_DIR / "cifar-10-batches-py"
    PROCESSED_DIR = Path(to_absolute_path(cfg.processed_dir))
    TRAIN_SUBDIR = PROCESSED_DIR / "train"
    TEST_SUBDIR = PROCESSED_DIR / "test"

    writer = SummaryWriter(log_dir=to_absolute_path(cfg.log_dir))

    # Step 1: Ensure data is present
    download_and_unpack_cifar10()
    create_image_file_structure()

    # Step 2: Enhanced Transforms with more augmentations
    mean = (0.4914, 0.4822, 0.4465)
    std = (0.2023, 0.1994, 0.2010)

    # Enhanced training augmentations - ИСПРАВЛЕНО: RandomErasing после ToTensor
    train_transform = transforms.Compose([
        transforms.RandomCrop(32, padding=4, padding_mode='reflect'),
        transforms.RandomHorizontalFlip(p=0.5),
        transforms.RandomRotation(15),
        transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.1),
        transforms.ToTensor(),
        transforms.Normalize(mean, std),
        transforms.RandomErasing(p=0.1, scale=(0.02, 0.33), ratio=(0.3, 3.3)),
    ])

    test_transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(mean, std)
    ])

    # Step 3: Datasets & Dataloaders with larger batch size
    full_train_dataset = CustomImageDataset(TRAIN_SUBDIR, transform=train_transform)
    val_size = int(0.15 * len(full_train_dataset))  # Reduced validation size
    train_size = len(full_train_dataset) - val_size
    train_dataset, val_dataset = random_split(
        full_train_dataset,
        [train_size, val_size],
        generator=torch.Generator().manual_seed(cfg.seed),
    )
    test_dataset = CustomImageDataset(TEST_SUBDIR, transform=test_transform)

    # Increased batch size for better gradient estimates
    batch_size = getattr(cfg, 'batch_size', 128)
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True,
                             num_workers=4, pin_memory=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False,
                           num_workers=4, pin_memory=True)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False,
                            num_workers=4, pin_memory=True)

    # Step 4: Improved Model, Optimizer, Loss, Scheduler
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = ImprovedNet(dropout=0.5).to(device)

    # Count parameters
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"[Model] Total parameters: {total_params:,}")
    print(f"[Model] Trainable parameters: {trainable_params:,}")

    # Label smoothing for better generalization
    criterion = nn.CrossEntropyLoss(label_smoothing=0.1)

    # AdamW with weight decay for better regularization
    optimizer = optim.AdamW(model.parameters(), lr=cfg.lr, weight_decay=0.01)

    # Cosine Annealing with Warm Restarts
    scheduler = CosineAnnealingWarmRestarts(
        optimizer, T_0=10, T_mult=2, eta_min=1e-6
    )

    # Metrics
    metrics = {
        "accuracy": MulticlassAccuracy(num_classes=10).to(device),
        "precision": MulticlassPrecision(num_classes=10, average="macro").to(device),
        "recall": MulticlassRecall(num_classes=10, average="macro").to(device),
        "f1": MulticlassF1Score(num_classes=10, average="macro").to(device),
    }

    # Step 5: Checkpoint Handling
    ckpt_dir = Path(to_absolute_path(cfg.checkpoint_dir))
    start_epoch = 0
    best_val_acc = 0.0
    history = {
        "train_loss": [],
        "val_loss": [],
        "val_accuracy": [],
        "val_precision": [],
        "val_recall": [],
        "val_f1": [],
        "lr": [],
    }

    latest_ckpt = get_latest_checkpoint(ckpt_dir)
    if latest_ckpt:
        print(f"[Checkpoint] Loading {latest_ckpt}…")
        checkpoint = torch.load(latest_ckpt, map_location=device)
        model.load_state_dict(checkpoint["model_state_dict"])
        optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
        scheduler.load_state_dict(checkpoint["scheduler_state_dict"])
        start_epoch = checkpoint["epoch"] + 1
        history = checkpoint["history"]
        best_val_acc = checkpoint.get("best_val_acc", 0.0)
        print(f"[Checkpoint] Resumed from epoch {start_epoch}. Best val acc: {best_val_acc:.4f}")

    # Step 6: Training Loop
    patience_counter = 0
    max_patience = 20  # Early stopping patience

    for epoch in range(start_epoch, cfg.epochs):
        print(f"\nEpoch {epoch+1}/{cfg.epochs}")
        train_loss = train_one_epoch(model, train_loader, criterion, optimizer, device, epoch)
        val_loss, val_metrics = validate(model, val_loader, criterion, device, metrics, epoch)
        scheduler.step()  # For CosineAnnealingWarmRestarts, step after each epoch
        current_lr = optimizer.param_groups[0]["lr"]

        # Update history
        history["train_loss"].append(train_loss)
        history["val_loss"].append(val_loss)
        history["val_accuracy"].append(val_metrics["accuracy"])
        history["val_precision"].append(val_metrics["precision"])
        history["val_recall"].append(val_metrics["recall"])
        history["val_f1"].append(val_metrics["f1"])
        history["lr"].append(current_lr)

        # TensorBoard logging
        writer.add_scalar("Loss/train", train_loss, epoch)
        writer.add_scalar("Loss/val", val_loss, epoch)
        writer.add_scalar("Accuracy/val", val_metrics["accuracy"], epoch)
        writer.add_scalar("F1/val", val_metrics["f1"], epoch)
        writer.add_scalar("LR", current_lr, epoch)

        print(
            f"Train Loss: {train_loss:.4f} | Val Loss: {val_loss:.4f} | "
            f"Val Acc: {val_metrics['accuracy']:.4f} | Val F1: {val_metrics['f1']:.4f} | LR: {current_lr:.6f}"
        )

        # Early stopping and best model saving
        is_best = val_metrics["accuracy"] > best_val_acc
        if is_best:
            best_val_acc = val_metrics["accuracy"]
            patience_counter = 0
            print(f"[Best Model] New best validation accuracy: {best_val_acc:.4f}")
        else:
            patience_counter += 1

        # Save checkpoint
        ckpt = {
            "epoch": epoch,
            "model_state_dict": model.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
            "scheduler_state_dict": scheduler.state_dict(),
            "history": history,
            "best_val_acc": best_val_acc,
        }

        save_path = save_checkpoint(ckpt, ckpt_dir, epoch)
        if is_best:
            # Save best model separately
            best_path = ckpt_dir / "best_model.pth"
            torch.save(ckpt, best_path)
            print(f"[Best Model] Saved to {best_path}")

        print(f"[Checkpoint] Saved to {save_path}")

        # Early stopping
        if patience_counter >= max_patience:
            print(f"[Early Stopping] No improvement for {max_patience} epochs. Stopping training.")
            break

    # Step 7: Final Test Evaluation with best model
    print("\n[Testing] Loading best model for final evaluation…")
    best_model_path = ckpt_dir / "best_model.pth"
    if best_model_path.exists():
        checkpoint = torch.load(best_model_path, map_location=device)
        model.load_state_dict(checkpoint["model_state_dict"])
        print(f"[Testing] Loaded best model from epoch {checkpoint['epoch']+1}")

    model.eval()
    test_acc_metric = MulticlassAccuracy(num_classes=10).to(device)
    test_f1_metric = MulticlassF1Score(num_classes=10, average="macro").to(device)

    with torch.no_grad():
        for inputs, targets in tqdm(test_loader, desc="Testing"):
            inputs, targets = inputs.to(device), targets.to(device)
            outputs = model(inputs)
            preds = torch.argmax(outputs, dim=1)
            test_acc_metric.update(preds, targets)
            test_f1_metric.update(preds, targets)

    test_accuracy = float(test_acc_metric.compute())
    test_f1 = float(test_f1_metric.compute())

    print(f"[Testing] Final Test Accuracy: {test_accuracy:.4f}")
    print(f"[Testing] Final Test F1 Score: {test_f1:.4f}")

    writer.add_scalar("Accuracy/test", test_accuracy, cfg.epochs)
    writer.add_scalar("F1/test", test_f1, cfg.epochs)
    writer.close()

    print(f"\n[Summary] Best Validation Accuracy: {best_val_acc:.4f}")
    print(f"[Summary] Final Test Accuracy: {test_accuracy:.4f}")


if __name__ == "__main__":
    # For standalone running without Hydra
    args = get_args()
    cfg = OmegaConf.create({
        "lr": args.lr,
        "epochs": args.epochs,
        "batch_size": args.batch_size,
        "seed": args.seed,
        "raw_data_dir": "./data_raw",
        "processed_dir": "./my_cifar_data",
        "checkpoint_dir": args.checkpoint_dir,
        "log_dir": args.log_dir,
    })
    main(cfg)
