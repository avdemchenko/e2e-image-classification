"""
Utility functions for training pipeline.

This module contains helper functions for reproducibility, checkpointing,
and other common utilities used throughout the training process.
"""

import random
from pathlib import Path
from typing import Dict

import numpy as np
import torch


def set_seed(seed: int) -> None:
    """
    Set random seed for reproducibility across all random number generators.
    
    Args:
        seed: Random seed value to set
    """
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def get_latest_checkpoint(ckpt_dir: Path) -> Path:
    """
    Return path to the latest checkpoint in directory or None.
    
    Args:
        ckpt_dir: Directory containing checkpoint files
        
    Returns:
        Path to the most recent checkpoint file, or None if no checkpoints found
    """
    ckpts = sorted(ckpt_dir.glob("checkpoint_epoch_*.pth"))
    return ckpts[-1] if ckpts else None


def save_checkpoint(state: Dict, ckpt_dir: Path, epoch: int) -> Path:
    """
    Save model checkpoint to disk.
    
    Args:
        state: Dictionary containing model state and training information
        ckpt_dir: Directory to save checkpoint in
        epoch: Current epoch number
        
    Returns:
        Path where the checkpoint was saved
    """
    ckpt_dir.mkdir(parents=True, exist_ok=True)
    ckpt_path = ckpt_dir / f"checkpoint_epoch_{epoch}.pth"
    torch.save(state, ckpt_path)
    return ckpt_path


def load_checkpoint(ckpt_path: Path, model, optimizer=None, scheduler=None, device='cpu'):
    """
    Load model checkpoint from disk.
    
    Args:
        ckpt_path: Path to checkpoint file
        model: PyTorch model to load state into
        optimizer: Optional optimizer to load state into
        scheduler: Optional scheduler to load state into
        device: Device to map tensors to
        
    Returns:
        tuple: (start_epoch, history, best_val_acc)
    """
    print(f"[Checkpoint] Loading {ckpt_path}…")
    checkpoint = torch.load(ckpt_path, map_location=device)
    
    model.load_state_dict(checkpoint["model_state_dict"])
    
    if optimizer is not None and "optimizer_state_dict" in checkpoint:
        optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
    
    if scheduler is not None and "scheduler_state_dict" in checkpoint:
        scheduler.load_state_dict(checkpoint["scheduler_state_dict"])
    
    start_epoch = checkpoint.get("epoch", 0) + 1
    history = checkpoint.get("history", {
        "train_loss": [],
        "val_loss": [],
        "val_accuracy": [],
        "val_precision": [],
        "val_recall": [],
        "val_f1": [],
        "lr": [],
    })
    best_val_acc = checkpoint.get("best_val_acc", 0.0)
    
    print(f"[Checkpoint] Resumed from epoch {start_epoch}. Best val acc: {best_val_acc:.4f}")
    
    return start_epoch, history, best_val_acc


def create_checkpoint_state(epoch: int, model, optimizer, scheduler, history: Dict, best_val_acc: float) -> Dict:
    """
    Create a checkpoint state dictionary.
    
    Args:
        epoch: Current epoch number
        model: PyTorch model
        optimizer: Optimizer
        scheduler: Learning rate scheduler
        history: Training history dictionary
        best_val_acc: Best validation accuracy so far
        
    Returns:
        Dictionary containing all checkpoint information
    """
    return {
        "epoch": epoch,
        "model_state_dict": model.state_dict(),
        "optimizer_state_dict": optimizer.state_dict(),
        "scheduler_state_dict": scheduler.state_dict(),
        "history": history,
        "best_val_acc": best_val_acc,
    }
