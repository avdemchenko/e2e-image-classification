import argparse
import os
from pathlib import Path

import torch
import matplotlib.pyplot as plt


def parse_args():
    parser = argparse.ArgumentParser(description="Visualize training history from checkpoint")
    parser.add_argument(
        "--checkpoint-file",
        type=str,
        required=True,
        help="Path to checkpoint file containing training history",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default="./plots",
        help="Directory to save generated plots",
    )
    return parser.parse_args()


def plot_history(history: dict, output_dir: Path):
    output_dir.mkdir(parents=True, exist_ok=True)

    epochs = range(1, len(history["train_loss"]) + 1)

    # Loss & Accuracy plot
    fig, ax1 = plt.subplots(figsize=(10, 6))
    ax1.plot(epochs, history["train_loss"], label="Train Loss", color="tab:blue")
    ax1.plot(epochs, history["val_loss"], label="Val Loss", color="tab:orange")
    ax1.set_xlabel("Epoch")
    ax1.set_ylabel("Loss")
    ax1.legend(loc="upper left")

    ax2 = ax1.twinx()
    ax2.plot(epochs, history["val_accuracy"], label="Val Accuracy", color="tab:green")
    ax2.set_ylabel("Accuracy")
    ax2.legend(loc="upper right")

    plt.title("Training & Validation Loss / Accuracy")
    plt.tight_layout()
    fig_path = output_dir / "loss_accuracy_plot.png"
    plt.savefig(fig_path)
    print(f"Saved plot to {fig_path}")
    plt.close()

    # Optional metrics dashboard
    if all(k in history for k in ["val_precision", "val_recall", "val_f1"]):
        plt.figure(figsize=(10, 6))
        plt.plot(epochs, history["val_precision"], label="Precision")
        plt.plot(epochs, history["val_recall"], label="Recall")
        plt.plot(epochs, history["val_f1"], label="F1-Score")
        plt.xlabel("Epoch")
        plt.ylabel("Score")
        plt.title("Validation Precision / Recall / F1")
        plt.legend()
        plt.grid(True)
        plt.tight_layout()
        fig_path = output_dir / "metrics_dashboard.png"
        plt.savefig(fig_path)
        print(f"Saved plot to {fig_path}")
        plt.close()

    # Learning rate schedule
    if "lr" in history:
        plt.figure(figsize=(8, 4))
        plt.plot(epochs, history["lr"], marker="o")
        plt.xlabel("Epoch")
        plt.ylabel("Learning Rate")
        plt.title("Learning Rate Schedule")
        plt.grid(True)
        plt.tight_layout()
        fig_path = output_dir / "lr_schedule.png"
        plt.savefig(fig_path)
        print(f"Saved plot to {fig_path}")
        plt.close()


def main():
    args = parse_args()
    ckpt_path = Path(args.checkpoint_file)
    if not ckpt_path.exists():
        raise FileNotFoundError(f"Checkpoint file {ckpt_path} not found.")

    checkpoint = torch.load(ckpt_path, map_location="cpu")
    history = checkpoint.get("history")
    if history is None:
        raise KeyError("History dictionary not found in checkpoint.")

    plot_history(history, Path(args.output_dir))


if __name__ == "__main__":
    main() 