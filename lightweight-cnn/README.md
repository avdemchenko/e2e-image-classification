# CIFAR-10 Training Pipeline

A complete PyTorch implementation for training a CNN on the CIFAR-10 dataset with comprehensive monitoring, modern architecture, and production-ready features.

## 🎯 Performance Highlights

**Current Results:**
- **Final Test Accuracy: 90.77%** 
- **Final Test F1-Score: 90.73%**
- **Training Efficiency: 89 epochs**
- **Significant improvement over baseline lightweight CNN**

## 🚀 Key Features

- **🏗️ Modular Architecture**: Clean separation of concerns with dedicated modules
- **🖱️ Click CLI Interface**: Professional command-line interface with intuitive options
- **⚙️ Hydra Configuration**: Advanced config management with override capabilities
- **📊 Advanced Data Pipeline**: Automatic CIFAR-10 download and preprocessing
- **🧠 Enhanced CNN Architecture**: Deep CNN with BatchNorm, dropout, and modern techniques
- **📈 Comprehensive Monitoring**: TensorBoard integration with real-time metrics
- **💾 Production Ready**: Robust checkpointing and resumable training
- **🔄 Reproducible Results**: Full seed control and deterministic training
- **📦 Easy Deployment**: Docker support with GPU acceleration

## 🏗️ Project Structure

```
lightweight-cnn/
├── 📄 config.yaml          # Main configuration file (moved to root)
├── 🐍 train.py             # Main training orchestrator
├── 📊 data.py              # Data processing and dataset utilities
├── 🧠 model.py             # Neural network architecture definitions
├── 🏃 training.py          # Training and validation loops
├── 🔧 utils.py             # Utility functions (checkpoints, seeding, etc.)
├── 📈 visualize.py         # Training history visualization
├── 📋 requirements.txt     # Python dependencies
├── 🐳 Dockerfile          # Container build recipe
├── 📚 README.md           # This documentation
├── 📂 checkpoints/        # Model checkpoints (created during training)
├── 📊 runs/              # TensorBoard logs (created during training)
├── 📈 plots/             # Generated visualization plots
├── 🗂️ data_raw/          # Raw CIFAR-10 data (downloaded automatically)
└── 🖼️ my_cifar_data/     # Processed PNG images organized by class
    ├── train/
    │   ├── 0_plane/
    │   ├── 1_car/
    │   └── ...
    └── test/
        ├── 0_plane/
        ├── 1_car/
        └── ...
```

## 🛠️ Installation

### Option 1: pip (Recommended)
```bash
git clone <repository-url>
cd lightweight-cnn
pip install -r requirements.txt
```

### Option 2: Conda Environment
```bash
conda env create -f environment.yml
conda activate lightweight-cnn
```

### Option 3: Docker
```bash
docker build -t lightweight-cnn .
docker run --rm -it -v $(pwd):/workspace lightweight-cnn
```

## 🎯 Quick Start

### Basic Training with Click CLI
```bash
# Train with default configuration
python train.py

# View all available options
python train.py --help

# Train with custom parameters
python train.py --lr 0.0005 --batch-size 64 --epochs 50

# Monitor training in real-time
tensorboard --logdir=runs
```

### Hydra Configuration Overrides
```bash
# Override config parameters directly
python train.py lr=0.0005 batch_size=64 epochs=50

# Change output directories
python train.py checkpoint_dir=./my_checkpoints log_dir=./my_logs
```

## ⚙️ Configuration

The training uses Hydra for configuration management. The main `config.yaml` is now located in the project root:

```yaml
# Training parameters
lr: 0.001
epochs: 100              # Adjust based on your needs
batch_size: 128          # Larger batches for stable gradients
seed: 42

# Paths (relative to project root; Hydra converts to absolute)
raw_data_dir: ./data_raw
processed_dir: ./my_cifar_data
checkpoint_dir: ./checkpoints
log_dir: ./runs

# Scheduler settings
scheduler:
  factor: 0.1
  patience: 2
```

### Click CLI Options
```bash
Usage: train.py [OPTIONS]

  CIFAR-10 Training Pipeline

Options:
  --lr FLOAT             Learning rate
  --epochs INTEGER       Number of epochs
  --batch-size INTEGER   Batch size
  --seed INTEGER         Random seed
  --checkpoint-dir TEXT  Directory for saving checkpoints
  --log-dir TEXT         TensorBoard log directory
  --help                 Show this message and exit.
```

## 🏛️ Modular Architecture

### Core Modules

#### 📊 `data.py` - Data Processing
- **CIFAR-10 Download & Extraction**: Automatic dataset management
- **Image Preprocessing**: Convert binary data to organized PNG structure
- **Custom Dataset Class**: Efficient PyTorch dataset implementation
- **Data Augmentation**: Training and validation transforms

#### 🧠 `model.py` - Neural Network Architecture
```python
class ImprovedNet(nn.Module):
    """Enhanced CNN with modern techniques"""
    - 4 Convolutional blocks with increasing channels (64→128→256→512)
    - Batch Normalization for training stability
    - Dropout layers for regularization
    - Global Average Pooling to reduce parameters
    - Progressive dimension reduction in classifier
```

#### 🏃 `training.py` - Training Utilities
- **Training Loop**: Single epoch training with progress bars
- **Validation**: Comprehensive metrics evaluation
- **Test Evaluation**: Final model assessment
- **Gradient Clipping**: Training stability improvements

#### 🔧 `utils.py` - Utility Functions
- **Reproducibility**: Seed management across all RNG sources
- **Checkpoint Management**: Save/load model states
- **Path Utilities**: Checkpoint discovery and organization

#### 🎯 `train.py` - Main Orchestrator
- **Configuration Management**: Hydra and Click integration
- **Pipeline Coordination**: Orchestrates all training components
- **Monitoring Setup**: TensorBoard and metrics initialization
- **Training Loop**: Main training and evaluation logic

### Enhanced CNN Architecture
```python
# Modern architecture with advanced techniques
ImprovedNet Features:
├── 🔄 4 Convolutional Blocks
│   ├── Conv2d → BatchNorm2d → ReLU → Conv2d → BatchNorm2d → ReLU
│   ├── MaxPool2d + Dropout2d (progressive rates: 0.1→0.2→0.3)
│   └── Channel progression: 3→64→128→256→512
├── 🌐 Global Average Pooling (reduces parameters significantly)
└── 🎯 Progressive Classifier
    ├── 512 → 256 → 128 → 10 classes
    └── Dropout rates: 0.5 → 0.3 → 0.15
```

### Advanced Training Features
- **🏷️ Label Smoothing**: Better generalization (0.1 smoothing)
- **⚖️ AdamW Optimizer**: Weight decay for regularization
- **📈 Cosine Annealing**: Warm restarts for better convergence
- **⏰ Early Stopping**: Automatic training termination (patience: 20)
- **🎯 Gradient Clipping**: Stability with max norm 1.0

## 📈 Results Analysis

### Performance Metrics

| Metric | Value | Notes |
|--------|-------|-------|
| **Test Accuracy** | 90.77% | Excellent performance for lightweight CNN |
| **Test F1-Score** | 90.73% | Balanced performance across all classes |
| **Training Epochs** | 89 | Efficient convergence with early stopping |
| **Model Parameters** | ~2.1M | Lightweight yet powerful architecture |

### Training Characteristics

The model demonstrates:
- **🎯 Smooth Convergence**: Stable training without oscillations
- **🚫 No Overfitting**: Train/validation curves remain close
- **⚡ Efficient Learning**: Reaches high accuracy quickly
- **💪 Robust Performance**: Consistent results across runs

## 🔧 Advanced Usage

### Resume Training from Checkpoint
```bash
# Training automatically resumes from latest checkpoint if available
python train.py

# Checkpoints are saved every epoch with complete training state
```

### Custom Training Parameters
```bash
# Fine-tune learning rate and batch size
python train.py --lr 0.0005 --batch-size 256

# Extended training with custom paths
python train.py --epochs 200 --checkpoint-dir ./long_training

# Quick testing run
python train.py --epochs 5 --batch-size 32
```

### Hyperparameter Tuning
```bash
# Grid search example with Click CLI
for lr in 0.001 0.0005 0.002; do
    for bs in 64 128 256; do
        python train.py --lr $lr --batch-size $bs
    done
done
```

## 📊 Monitoring & Visualization

### TensorBoard Integration
```bash
# Start TensorBoard
tensorboard --logdir=runs --port=6006

# View metrics in browser at http://localhost:6006
```

### Available Metrics
- **📉 Loss Curves**: Training and validation loss tracking
- **🎯 Accuracy**: Validation accuracy over epochs
- **📊 F1-Score**: Macro F1-score for balanced evaluation
- **📈 Learning Rate**: LR schedule visualization
- **⏱️ Training Progress**: Real-time epoch monitoring

### Post-Training Visualization
```bash
# Generate comprehensive plots with Click interface
python visualize.py --checkpoint-file ./checkpoints/best_model.pth --output-dir ./plots

# Available visualizations:
# - Loss and accuracy curves
# - Metrics dashboard (Precision, Recall, F1)
# - Learning rate schedule
```

## 🐳 Docker Support

### Build and Run
```bash
# Build image
docker build -t lightweight-cnn .

# Run training with Click CLI
docker run --rm -it -v $(pwd):/workspace lightweight-cnn python train.py

# Run with custom parameters
docker run --rm -it -v $(pwd):/workspace lightweight-cnn python train.py --lr 0.0005
```

### GPU Support
```bash
# Run with GPU acceleration
docker run --gpus all --rm -it -v $(pwd):/workspace lightweight-cnn python train.py
```

## 🔬 Implementation Highlights

### Data Pipeline (`data.py`)
- **🔄 Automatic Download**: Handles CIFAR-10 dataset acquisition
- **📁 Smart Organization**: Creates class-based directory structure
- **🖼️ Efficient Loading**: On-demand image loading with caching
- **🎨 Advanced Augmentation**: Training-specific transformations

### Model Architecture (`model.py`)
- **🏗️ Modern Design**: BatchNorm + Dropout + Global pooling
- **📊 Parameter Efficiency**: ~2M parameters with high performance
- **🔧 Configurable**: Adjustable dropout rates and class numbers
- **📈 Performance Tracking**: Built-in parameter counting

### Training System (`training.py`)
- **⚡ Optimized Loops**: Efficient training and validation cycles
- **📊 Rich Metrics**: Comprehensive evaluation with torchmetrics
- **🎯 Progress Tracking**: Real-time progress bars with tqdm
- **🛡️ Stability Features**: Gradient clipping and error handling

### Utility Functions (`utils.py`)
- **🔄 Reproducibility**: Complete seed management system
- **💾 Smart Checkpointing**: Automatic state preservation
- **📂 Path Management**: Intelligent checkpoint discovery
- **🔧 Helper Functions**: Common training utilities

## 🚀 Future Enhancements

### Planned Improvements
- **🏗️ Architecture**: ResNet blocks, attention mechanisms
- **📊 Augmentation**: MixUp, CutMix, AutoAugment
- **⚙️ Training**: Learning rate warmup, advanced schedulers
- **🎯 Optimization**: Different optimizers, gradient accumulation
- **🔬 Regularization**: Advanced techniques, knowledge distillation

### Expected Performance Gains
With additional improvements:
- **92-94% accuracy** with ResNet-like architecture
- **95%+ accuracy** with modern techniques and larger models
- **Faster convergence** with advanced training techniques

## 🛠️ Development

### Code Quality
- **🧹 Modular Design**: Clean separation of concerns
- **📝 Type Hints**: Full type annotation support
- **📚 Documentation**: Comprehensive docstrings
- **🧪 Testable**: Isolated components for easy testing

### Contributing
- **🔧 Easy Extension**: Add new modules or modify existing ones
- **📊 New Models**: Simple integration of different architectures
- **🎨 Custom Augmentations**: Easy to add new data transformations
- **📈 Metrics**: Simple addition of new evaluation metrics

## 📚 References

- **CIFAR-10 Dataset**: [Learning Multiple Layers of Features from Tiny Images](https://www.cs.toronto.edu/~kriz/cifar.html)
- **PyTorch Documentation**: [Official PyTorch Tutorials](https://pytorch.org/tutorials/)
- **Hydra Configuration**: [Hydra Documentation](https://hydra.cc/)
- **Click CLI**: [Click Documentation](https://click.palletsprojects.com/)
- **TensorBoard**: [TensorBoard Guide](https://pytorch.org/docs/stable/tensorboard.html)

---

🎉 **Ready to train? Run `python train.py --help` to get started!**