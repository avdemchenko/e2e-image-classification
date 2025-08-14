"""
Training and validation utilities for neural network training.

This module contains functions for training loops, validation, and
evaluation metrics handling.
"""

import torch
from tqdm.auto import tqdm


def train_one_epoch(model, loader, criterion, optimizer, device, epoch_idx: int = None):
    """
    Train model for a single epoch with tqdm progress bar.
    
    Args:
        model: PyTorch model to train
        loader: Training data loader
        criterion: Loss function
        optimizer: Optimizer for updating model parameters
        device: Device to run training on (cuda/cpu)
        epoch_idx: Current epoch index for display purposes
        
    Returns:
        float: Average training loss for the epoch
    """
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
    """
    Validate model with tqdm progress bar.
    
    Args:
        model: PyTorch model to validate
        loader: Validation data loader
        criterion: Loss function
        device: Device to run validation on (cuda/cpu)
        metrics: Dictionary of torchmetrics metrics to compute
        epoch_idx: Current epoch index for display purposes
        
    Returns:
        tuple: (validation_loss, metrics_dict)
            - validation_loss (float): Average validation loss
            - metrics_dict (dict): Dictionary of computed metric values
    """
    model.eval()
    running_loss = 0.0
    
    # Reset all metrics
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
            
            # Update metrics
            preds = torch.argmax(outputs, dim=1)
            for metric in metrics.values():
                metric.update(preds, targets)

    val_loss = running_loss / len(loader.dataset)
    results = {k: float(m.compute()) for k, m in metrics.items()}
    return val_loss, results


def evaluate_test_set(model, test_loader, device, metrics):
    """
    Evaluate model on test set.
    
    Args:
        model: PyTorch model to evaluate
        test_loader: Test data loader
        device: Device to run evaluation on (cuda/cpu)
        metrics: Dictionary of torchmetrics metrics to compute
        
    Returns:
        dict: Dictionary of computed test metrics
    """
    model.eval()
    
    # Reset all metrics
    for m in metrics.values():
        m.reset()

    with torch.no_grad():
        for inputs, targets in tqdm(test_loader, desc="Testing"):
            inputs, targets = inputs.to(device), targets.to(device)
            outputs = model(inputs)
            preds = torch.argmax(outputs, dim=1)
            
            # Update metrics
            for metric in metrics.values():
                metric.update(preds, targets)

    results = {k: float(m.compute()) for k, m in metrics.items()}
    return results
