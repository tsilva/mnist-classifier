> [!WARNING]
> ## Archived
> This project is archived and no longer maintained. It has been superseded by newer ML experiment repositories.

<div align="center">
  <img src="logo.png" alt="mnist-classifier" width="512"/>

  [![License](https://img.shields.io/badge/License-MIT-blue.svg)](LICENSE)
  [![Python](https://img.shields.io/badge/Python-3.8+-3776ab.svg)](https://python.org)
  [![PyTorch](https://img.shields.io/badge/PyTorch-2.0+-ee4c2c.svg)](https://pytorch.org)
  [![W&B](https://img.shields.io/badge/Weights_&_Biases-Enabled-ffcc33.svg)](https://wandb.ai)

  **🧠 Train and evaluate CNN classifiers on MNIST-like datasets with experiment tracking, hyperparameter sweeps, and beautiful visualizations 📊**

  [Features](#features) · [Quick Start](#quick-start) · [Models](#models) · [Benchmarks](#benchmarks)
</div>

## Features

- **Multiple CNN Architectures** - LeNet5, LeNet5Improved, and AdvancedCNN with up to 99.59% accuracy
- **Five MNIST-like Datasets** - MNIST, FashionMNIST, EMNIST, KMNIST, and QMNIST
- **Experiment Tracking** - Full Weights & Biases integration for metrics, artifacts, and visualizations
- **Hyperparameter Sweeps** - Automated tuning with wandb sweeps
- **Training Utilities** - Learning rate scheduling, early stopping, custom weight initialization
- **Rich Visualizations** - Confusion matrices and misclassified image analysis

## Quick Start

```bash
# Clone the repository
git clone https://github.com/tsilva/mnist-classifier.git
cd mnist-classifier

# Create and activate conda environment
conda env create -f environment.yml
conda activate mnist-classifier

# Train a model
python main.py train --hyperparams_path configs/train/LeNet5.yml --n_epochs 50 --dataset MNIST
```

## Requirements

| Requirement | Specification |
|-------------|---------------|
| **Python** | 3.8+ |
| **GPU** | NVIDIA with CUDA support (optional but recommended) |
| **CUDA** | 11.8+ (if using GPU) |
| **RAM** | 8GB+ recommended |

## Installation

1. **Install Miniconda** from the [official website](https://docs.conda.io/en/latest/miniconda.html)

2. **Create the environment:**
   ```bash
   conda env create -f environment.yml
   conda activate mnist-classifier
   ```

3. **Verify CUDA availability** (optional):
   ```bash
   python -c "import torch; print(f'CUDA available: {torch.cuda.is_available()}')"
   ```

   If CUDA is not detected:
   ```bash
   pip uninstall torch torchvision
   pip install torch torchvision torchaudio --extra-index-url https://download.pytorch.org/whl/cu118
   ```

## Usage

### Training

```bash
python main.py train --hyperparams_path configs/train/LeNet5.yml --n_epochs 50 --dataset MNIST
```

### Evaluation

```bash
# Local model
python main.py eval --model_path outputs/best_model.pth --dataset FashionMNIST

# From W&B run
python main.py eval --model_path https://wandb.ai/username/project/runs/run_id --dataset EMNIST-letters
```

### Hyperparameter Sweeps

```bash
# Create sweep
python main.py sweep --dataset KMNIST

# Start agent
wandb agent username/project/sweep_id
```

## Datasets

| Dataset | Description | Classes |
|---------|-------------|---------|
| **MNIST** | Handwritten digits | 10 |
| **FashionMNIST** | Fashion product images | 10 |
| **EMNIST** | Extended MNIST with letters and digits | 47+ |
| **KMNIST** | Kuzushiji (Japanese characters) | 10 |
| **QMNIST** | Higher quality MNIST alternative | 10 |

## Models

### LeNet5Original

Classic architecture from the original 1998 paper:
- 2 convolutional layers + 3 fully connected layers
- Average pooling, Tanh activation

### LeNet5Improved

Modernized LeNet with better performance:
- 3 convolutional layers + 2 fully connected layers
- Batch normalization, dropout, max pooling, ReLU

### AdvancedCNN

Deep architecture for maximum accuracy:
- 7 convolutional layers with varying kernel sizes
- 1 fully connected layer (reduced overfitting)
- Batch normalization, dropout, ReLU

## Benchmarks

| Model | MNIST | FashionMNIST | QMNIST | KMNIST | EMNIST-digits |
|-------|-------|--------------|--------|--------|---------------|
| LeNet5 | 97.05% | - | - | - | - |
| LeNet5Improved | 99.55% | - | - | - | - |
| AdvancedCNN | 99.58% | - | - | - | - |
| **Ensemble** | **99.59%** | - | - | - | - |

## Configuration

Hyperparameters are specified in YAML files under `configs/train/`:

```yaml
data_loader:
  dataset: "mnist"
  batch_size: 64

model:
  id: "LeNet5"
  params:
    conv1_filters: 6
    conv2_filters: 16
    conv3_filters: 120
    fc1_neurons: 84
    fc2_neurons: 10
    weight_init: "he"

optimizer:
  id: "Adam"
  params:
    lr: 0.001

loss_function:
  id: "CrossEntropyLoss"

lr_scheduler:
  id: "StepLR"
  params:
    step_size: 10
    gamma: 0.1
```

## Experiment Tracking

All experiments are logged to [Weights & Biases](https://wandb.ai):

- Training/validation loss and accuracy
- Precision, recall, and F1 score
- Confusion matrices
- Misclassified images
- Best model checkpoints

## Project Structure

```
mnist-classifier/
├── main.py              # CLI entry point (train/eval/sweep)
├── configs/
│   ├── train/           # Training hyperparameters
│   └── sweep/           # Sweep configurations
├── libs/
│   ├── models.py        # CNN architectures
│   ├── datasets.py      # Dataset loaders
│   ├── data_loaders.py  # DataLoader utilities
│   ├── optimizers.py    # Optimizer builders
│   ├── lr_schedulers.py # LR scheduler builders
│   ├── loss_functions.py
│   ├── early_stopping.py
│   └── wandb_utils.py   # W&B integration
├── tools/               # Utility scripts
└── outputs/             # Saved models and artifacts
```

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.
