"""
Main training script for CIFAR-10 classification.

This script orchestrates the training pipeline using modularized components
for data processing, model definition, training utilities, and more.
"""

from pathlib import Path

import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim.lr_scheduler import CosineAnnealingWarmRestarts
from torch.utils.data import DataLoader, random_split
from torchvision import transforms
from torch.utils.tensorboard import SummaryWriter

from torchmetrics.classification import (
    MulticlassAccuracy,
    MulticlassPrecision,
    MulticlassRecall,
    MulticlassF1Score,
)

import click
from omegaconf import DictConfig, OmegaConf
import hydra
from hydra.utils import to_absolute_path

# Import our custom modules
from data import download_and_unpack_cifar10, create_image_file_structure, CustomImageDataset
from model import ImprovedNet
from training import train_one_epoch, validate, evaluate_test_set
from utils import set_seed, get_latest_checkpoint, save_checkpoint, load_checkpoint, create_checkpoint_state


# ---------------------------------------------------------------------
# Click Commands
# ---------------------------------------------------------------------

@click.command()
@click.option('--lr', default=0.001, type=float, help='Learning rate')
@click.option('--epochs', default=100, type=int, help='Number of epochs')
@click.option('--batch-size', default=128, type=int, help='Batch size')
@click.option('--seed', default=42, type=int, help='Random seed')
@click.option('--checkpoint-dir', default='./checkpoints_improved', type=str, help='Directory for saving checkpoints')
@click.option('--log-dir', default='./runs_improved', type=str, help='TensorBoard log directory')
def train_standalone(lr, epochs, batch_size, seed, checkpoint_dir, log_dir):
    """CIFAR-10 Training Pipeline"""
    cfg = OmegaConf.create({
        "lr": lr,
        "epochs": epochs,
        "batch_size": batch_size,
        "seed": seed,
        "raw_data_dir": "./data_raw",
        "processed_dir": "./my_cifar_data",
        "checkpoint_dir": checkpoint_dir,
        "log_dir": log_dir,
    })
    main(cfg)


# ---------------------------------------------------------------------
# Main Function
# ---------------------------------------------------------------------

@hydra.main(version_base=None, config_path=".", config_name="config")
def main(cfg: DictConfig):
    """Main training function that orchestrates the entire pipeline."""
    # Optional: print resolved config
    print(OmegaConf.to_yaml(cfg))

    set_seed(cfg.seed)

    # Ensure all paths are absolute (Hydra changes the working dir)
    raw_data_dir = Path(to_absolute_path(cfg.raw_data_dir))
    extracted_dir = raw_data_dir / "cifar-10-batches-py"
    processed_dir = Path(to_absolute_path(cfg.processed_dir))
    train_subdir = processed_dir / "train"
    test_subdir = processed_dir / "test"

    writer = SummaryWriter(log_dir=to_absolute_path(cfg.log_dir))

    # Step 1: Ensure data is present
    download_and_unpack_cifar10(raw_data_dir, extracted_dir)
    create_image_file_structure(extracted_dir, processed_dir, train_subdir, test_subdir)

    # Step 2: Enhanced Transforms with more augmentations
    mean = (0.4914, 0.4822, 0.4465)
    std = (0.2023, 0.1994, 0.2010)

    # Enhanced training augmentations
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

    # Step 3: Datasets & Dataloaders
    full_train_dataset = CustomImageDataset(train_subdir, transform=train_transform)
    val_size = int(0.15 * len(full_train_dataset))  # Reduced validation size
    train_size = len(full_train_dataset) - val_size
    train_dataset, val_dataset = random_split(
        full_train_dataset,
        [train_size, val_size],
        generator=torch.Generator().manual_seed(cfg.seed),
    )
    test_dataset = CustomImageDataset(test_subdir, transform=test_transform)

    # Increased batch size for better gradient estimates
    batch_size = getattr(cfg, 'batch_size', 128)
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True,
                             num_workers=4, pin_memory=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False,
                           num_workers=4, pin_memory=True)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False,
                            num_workers=4, pin_memory=True)

    # Step 4: Model, Optimizer, Loss, Scheduler
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = ImprovedNet(dropout=0.5).to(device)

    # Count parameters
    total_params, trainable_params = model.count_parameters()
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
        start_epoch, history, best_val_acc = load_checkpoint(
            latest_ckpt, model, optimizer, scheduler, device
        )

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
        ckpt = create_checkpoint_state(epoch, model, optimizer, scheduler, history, best_val_acc)
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

    # Create test metrics
    test_metrics = {
        "accuracy": MulticlassAccuracy(num_classes=10).to(device),
        "f1": MulticlassF1Score(num_classes=10, average="macro").to(device),
    }
    
    test_results = evaluate_test_set(model, test_loader, device, test_metrics)
    test_accuracy = test_results["accuracy"]
    test_f1 = test_results["f1"]

    print(f"[Testing] Final Test Accuracy: {test_accuracy:.4f}")
    print(f"[Testing] Final Test F1 Score: {test_f1:.4f}")

    writer.add_scalar("Accuracy/test", test_accuracy, cfg.epochs)
    writer.add_scalar("F1/test", test_f1, cfg.epochs)
    writer.close()

    print(f"\n[Summary] Best Validation Accuracy: {best_val_acc:.4f}")
    print(f"[Summary] Final Test Accuracy: {test_accuracy:.4f}")


if __name__ == "__main__":
    # For standalone running without Hydra
    train_standalone()