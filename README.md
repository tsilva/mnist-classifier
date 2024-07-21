# MNIST-like Classifier

This project implements a Convolutional Neural Network (CNN) classifier for MNIST-like datasets using PyTorch. It includes features for training, evaluation, and hyperparameter tuning using Weights & Biases (wandb) for experiment tracking.

## Table of Contents

- [Features](#features)
- [Datasets](#datasets)
- [Installation](#installation)
- [Usage](#usage)
- [Models](#models)
- [Benchmarks](#benchmarks)
- [Configuration](#configuration)
- [Logging and Visualization](#logging-and-visualization)
- [License](#license)

## Features

- Implementation of multiple CNN architectures (LeNet5, SimpleCNN, AdvancedCNN)
- Support for various MNIST-like datasets (MNIST, FashionMNIST, EMNIST, KMNIST, QMNIST)
- Training with customizable hyperparameters
- Evaluation on test set
- Hyperparameter tuning using wandb sweeps
- Logging of training progress, metrics, and artifacts to wandb
- Visualization of confusion matrices and misclassified images
- Support for custom weight initialization
- Learning rate scheduling

## Datasets

The project supports the following datasets:

- **MNIST**: Handwritten digit recognition dataset
- **FashionMNIST**: Fashion product recognition dataset
- **EMNIST**: Extended MNIST dataset with letters and digits
- **KMNIST**: Kuzushiji-MNIST dataset (Japanese characters)
- **QMNIST**: QMNIST dataset (MNIST alternative with better quality)

To use a specific dataset, specify it in the configuration file or as a command-line argument.

## Installation

1. Clone this repository:

```
git clone https://github.com/tsilva/mnist-classifier.git
cd mnist-classifier
```

2. Install Miniconda:
   - Visit the [Miniconda website](https://docs.conda.io/en/latest/miniconda.html) and download the appropriate installer for your operating system.
   - Follow the installation instructions for your platform.

3. Create a new Conda environment:

```
conda env create -f environment.yml
```

3. Activate the new environment:

```
conda activate mnist-classifier
```

4. Ensure that CUDA is available:

```
python -c "import torch; print(torch.cuda.is_available()); print(torch.cuda.current_device()); print(torch.cuda.get_device_name(0))"
```

If not available try the following (for CUDA 11.8):

```
conda activate mnist-classifier
pip uninstall torch torchvision
pip install torch torchvision torchaudio --extra-index-url https://download.pytorch.org/whl/cu118
```

## Usage

The script can be run in three modes: train, eval, and sweep.

### Training

To train a model:

```
python main.py train --hyperparams_path configs/hyperparams/LeNet5.yml --n_epochs 50 --dataset MNIST
```

### Evaluation

To evaluate a trained model:

```
python main.py eval --model_path outputs/best_model.pth --dataset FashionMNIST
```

Or reference the URL of the Weights & Biases run:

```
python main.py eval --model_path https://wandb.ai/username/project/runs/run_id --dataset EMNIST-letters
```

### Hyperparameter Tuning

To create hyperparameter sweep:

```
python main.py sweep --dataset KMNIST
```

Then start the sweep agent:

```
wandb agent username/project/sweep_id
```

## Models

The project includes three CNN architectures:

1. **LeNet5**: Attempts to replicate the original LeNet-5 architecture with 2 convolutional layers and 3 fully connected layers.
2. **LeNet5Improved**: An improved version of LeNet-5 with 3 convolutional layers, 2 fully connected layers, batch normalization, dropout, max pooling, and ReLU.
3. **AdvancedCNN**: A more advanced CNN architecture with 7 convolutional layers (more features), with different kernel sizes and strides (different feature scales), 1 fully connected layer (less classification overfitting), batch normalization, dropout, and ReLU.

## Benchmarks

The following table shows the test set accuracy achieved for each model on each dataset:

| Model                              | MNIST   | FashionMNIST | QMNIST | KMNIST | EMNIST-digits |
|------------------------------------|---------|--------------| ------ | ------ | ------------- |
| LeNet5                             | 97.05%  | N/A          | N/A    | N/A    | N/A           |
| LeNet5Improved                     | 99.55%  | N/A          | N/A    | N/A    | N/A           |
| Advanced CNN                       | 99.58%  | N/A          | N/A    | N/A    | N/A           |
| Weighted Averaging Ensemble        | 99.59%  | N/A          | N/A    | N/A    | N/A           |

## Configuration

Hyperparameters and model configurations are specified in YAML files in the `configs/hyperparams/` directory. You can create custom configuration files to experiment with different settings.

Example configuration (LeNet5.yml):

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

## Logging and Visualization

The project uses Weights & Biases (wandb) for experiment tracking and visualization. During training and evaluation, the following metrics and artifacts are logged:

- Training and validation loss
- Training and validation accuracy
- Precision, recall, and F1 score
- Confusion matrix
- Misclassified images
- Best model weights

You can view the results and compare experiments in the wandb dashboard.

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.