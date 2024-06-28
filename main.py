import argparse
import atexit
import logging
import multiprocessing
import os
import random
import sys
import time

import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
from sklearn.metrics import confusion_matrix, f1_score, precision_score, recall_score
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torchvision
import torchvision.transforms as transforms
from torch.utils.data import DataLoader, random_split
from tqdm import tqdm
import wandb
import yaml

PROJECT_NAME = 'mnist-classifier'

# Define the default configuration for the model
DEFAULT_CONFIG = {
    "logging" : {
        "image_interval": 5,
    },
    "data_loader": {
        "batch_size": 64
    },
    "optimizer" : {
        "id": "Adam"
    },
    "loss_function": {
        "id": "CrossEntropyLoss"
    }
}

# Retrieve the device to run the models on
# (use HW acceleration if available, otherwise use CPU)
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Setup logger
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(message)s')
logger = logging.getLogger(__name__)

# Make sure the user is aware if HW acceleration is not available
logger.info(f"Using device: {DEVICE}")
if str(DEVICE) == "cpu":
    logging.warning("CUDA is not available. Running on CPU. Press Enter to continue...")
    input()

# Make sure W&B is terminated even if the script crashes
def cleanup():
    if wandb.run: wandb.finish()
atexit.register(cleanup)

"""
Set a seed in the random number generators for reproducibility.
"""
def set_seed(seed):
    random.seed(seed) # Set the seed for the random number generator
    np.random.seed(seed) # Set the seed for NumPy
    torch.manual_seed(seed) # Set the seed for PyTorch
    torch.cuda.manual_seed(seed) # Set the seed for CUDA
    torch.cuda.manual_seed_all(seed) # Set the seed for all CUDA devices
    torch.backends.cudnn.deterministic = True # Ensure deterministic results (WARNING: can slow down training!)
    torch.backends.cudnn.benchmark = False # Disable cuDNN benchmarking (WARNING: can slow down training!)

"""
Initialize weights using the specified initialization mode.
"""
def init_model_weights(model, mode):
    # He initialization
    if mode == 'he':
        for m in model.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            elif isinstance(m, nn.Linear):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)

"""
Load a model from the specified path.
If the path is the URL for a W&B run, download associated model artifact.
"""
def load_model(model_path):
    if model_path.startswith('https://wandb.ai/'):
        url_tokens = model_path.split('/')
        entity_id = url_tokens[3]
        project_name = url_tokens[4]
        run_id = url_tokens[6]
        model_file_name = f"best_model_{run_id}.pth"
        artifact_path = f"{entity_id}/{project_name}/{model_file_name}:latest"
        logging.info(f"Downloading model artifact: {artifact_path}")
        artifact = wandb.use_artifact(artifact_path, type='model')
        artifact_dir = artifact.download()
        model_path = f"{artifact_dir}/{model_file_name}"
    
    if model_path.endswith('.pth'): model = torch.load(model_path)
    elif model_path.endswith('.onnx'): model = torch.onnx.load(model_path)
    else: raise ValueError(f"Unsupported model file format: {model_path}")
    return model
    
"""
Returns the default configuration merged with the specified hyperparameters.
"""
def load_config(hyperparams_path=None):
    # Load hyperparameters from the specified file (if provided)
    hyperparams = {}
    if hyperparams_path:
        with open(hyperparams_path, 'r') as file:
            hyperparams = yaml.safe_load(file)

    # Merge the hyperparameters with the default configuration
    return {
        **DEFAULT_CONFIG,
        **hyperparams
    }

"""
Parse the sweep configuration from W&B to a dictionary.
Mapping dependent parameters from the flattened structure of the sweep config to the nested structure of our config.
"""
def parse_wandb_sweep_config(sweep_config):
    config = {}
    for key, value in sweep_config.items():
        if key == "method": continue

        if isinstance(value, dict) and "id" in value:
            _id = value["id"]
            _id_l = _id.lower()
            params = {k.replace(f"{_id_l}_", ""): v for k, v in value.items() if k.startswith(_id_l)}
            value = {"id": _id, "params": params}
        
        config[key] = value
    return config

"""
LeNet-5 model:
- 2 convolutional layers
- 3 fully connected layers
- Average pooling (legacy reasons from the original paper, max pooling would be better)
- Tanh activation functions (legacy reasons from the original paper, ReLU would be better)
"""
class LeNet5(nn.Module):
    def __init__(
        self, 
        conv1_filters=6, 
        conv2_filters=16, 
        conv3_filters=120, 
        fc1_neurons=84, 
        fc2_neurons=10, 
        weight_init=None
    ):
        super(LeNet5, self).__init__()

        # Store the hyperparameters
        self.conv1_filters = conv1_filters
        self.conv2_filters = conv2_filters
        self.conv3_filters = conv3_filters
        self.fc1_neurons = fc1_neurons
        self.fc2_neurons = fc2_neurons

        # Encodes the input into higher-dimensional representations
        self.encoder = nn.Sequential(
            # Input shape: (batch_size, 1, 28, 28)
            # Kernel size: 5x5, Stride: 1, Padding: 2
            # Output shape: (batch_size, conv1_filters, 28, 28)
            nn.Conv2d(1, conv1_filters, kernel_size=5, stride=1, padding=2),
            nn.Tanh(), 
            
            # Input shape: (batch_size, conv1_filters, 28, 28)
            # Kernel size: 2x2, Stride: 2
            # Output shape: (batch_size, conv1_filters, 14, 14)
            nn.AvgPool2d(kernel_size=2, stride=2),

            # Input shape: (batch_size, conv1_filters, 14, 14)
            # Kernel size: 5x5, Stride: 1, Padding: 0
            # Output shape: (batch_size, conv2_filters, 10, 10)
            nn.Conv2d(conv1_filters, conv2_filters, kernel_size=5, stride=1),
            nn.Tanh(),

            # Input shape: (batch_size, conv2_filters, 10, 10)
            # Kernel size: 2x2, Stride: 2
            # Output shape: (batch_size, conv2_filters, 5, 5)
            nn.AvgPool2d(kernel_size=2, stride=2),
            
            # Input shape: (batch_size, conv2_filters, 5, 5)
            # Kernel size: 5x5, Stride: 1, Padding: 0
            # Output shape: (batch_size, conv3_filters, 1, 1)
            nn.Conv2d(conv2_filters, conv3_filters, kernel_size=5, stride=1),
            nn.Tanh()
        )

        # Classifies the encoded input into output classes
        self.classifier = nn.Sequential(
            # Input shape: (batch_size, conv3_filters)
            # Output shape: (batch_size, fc1_neurons)
            nn.Linear(conv3_filters, fc1_neurons),

            # Tanh activation
            nn.Tanh(),

            # Input shape: (batch_size, fc1_neurons)
            # Output shape: (batch_size, fc2_neurons)
            nn.Linear(fc1_neurons, fc2_neurons)
        )

        # Conditional custom weight initialization
        init_model_weights(self, weight_init)

    def forward(self, x):
        # Pass through the encoder
        # Input shape: (batch_size, 1, 28, 28)
        # Output shape: (batch_size, conv3_filters, 1, 1)
        x = self.encoder(x)
        
        # Flatten the output for the classifier
        # Input shape: (batch_size, conv3_filters, 1, 1)
        # Output shape: (batch_size, conv3_filters)
        # x = x.view(-1, self.conv3_filters)
        # x = torch.flatten(x, 1)
        x = x.view(x.size(0), -1)
        
        # Pass through the classifier
        # Input shape: (batch_size, conv3_filters)
        # Output shape: (batch_size, fc2_neurons)
        x = self.classifier(x)
        
        # Return output with shape (batch_size, fc2_neurons)
        return x

"""
Simple CNN model with:
- 2 convolutional layers
- 2 fully connected layers
- ReLU activation functions
- Max pooling
"""
class SimpleCNN(nn.Module):
    def __init__(self, conv1_filters=32, conv2_filters=64, fc1_neurons=1000, fc2_neurons=10, weight_init=None):
        super(SimpleCNN, self).__init__()
        
        self.encoder = nn.Sequential(
            # First convolutional layer
            # Input shape: (batch_size, 1, 28, 28)
            # Kernel size: 3x3, Stride: 1, Padding: 1
            # Output shape: (batch_size, conv1_filters, 28, 28)
            nn.Conv2d(1, conv1_filters, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2), # Shape: (batch_size, conv1_filters, 14, 14)

            # Second convolutional layer
            # Input shape: (batch_size, conv1_filters, 14, 14)
            # Kernel size: 3x3, Stride: 1, Padding: 0 (default)
            # Output shape: (batch_size, conv2_filters, 12, 12)
            nn.Conv2d(conv1_filters, conv2_filters, kernel_size=3),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2) # Shape: (batch_size, conv2_filters, 6, 6)
        )

        self.classifier = nn.Sequential(
            # First fully connected layer
            # Input shape: (batch_size, conv2_filters * 6 * 6)
            # Output shape: (batch_size, fc1_neurons)
            nn.Linear(conv2_filters * 6 * 6, fc1_neurons),
            nn.ReLU(),

            # Second fully connected layer
            # Input shape: (batch_size, fc1_neurons)
            # Output shape: (batch_size, fc2_neurons)
            nn.Linear(fc1_neurons, fc2_neurons)
        )

        init_model_weights(self, weight_init)

    def forward(self, x):
        # Encode input using convolutional layers
        x = self.encoder(x) 
        
        # Flatten the output for the fully connected layers
        x = x.view(x.size(0), -1) # Shape: (batch_size, conv2_filters * 6 * 6)
        
        # Classify encoded input using fully connected layers
        x = self.classifier(x)
        
        # Return the output
        return x # Shape: (batch_size, fc2_neurons)

"""
Advanced CNN model with:
- 4 convolutional layers
- 4 fully connected layers
- Batch normalization
- ReLU activation functions
- Max pooling
- Dropout
"""
class AdvancedCNN(nn.Module):
    def __init__(self, weight_init=None):
        super(AdvancedCNN, self).__init__()
        
        # Encode input into higher-dimensional representations
        self.encoder = nn.Sequential(
            # First convolutional layer (feature extraction)
            # Input shape: (batch_size, 1, 28, 28)
            # Kernel size: 5x5, Stride: 1, Padding: 2
            # Output shape: (batch_size, 32, 28, 28)
            nn.Conv2d(in_channels=1, out_channels=32, kernel_size=5, stride=1, padding=2),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),

            # Second convolutional layer (feature extraction)
            # Input shape: (batch_size, 32, 28, 28)
            # Kernel size: 5x5, Stride: 1, Padding: 2
            # Output shape: (batch_size, 32, 28, 28)
            nn.Conv2d(in_channels=32, out_channels=32, kernel_size=5, stride=1, padding=2, bias=False),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),

            # Max pooling layer (downsampling)
            # Input shape: (batch_size, 32, 28, 28)
            # Kernel size: 2x2, Stride: 2
            # Output shape: (batch_size, 32, 14, 14)
            nn.MaxPool2d(kernel_size=2, stride=2),

            # Dropout layer (regularization)
            # (randomly zeroes some of the elements of the input tensor 
            # with probability in order to prevent overfitting)
            nn.Dropout(0.5),

            # Third convolutional layer (feature extraction)
            # Input shape: (batch_size, 32, 14, 14)
            # Kernel size: 3x3, Stride: 1, Padding: 1
            # Output shape: (batch_size, 64, 14, 14)
            nn.Conv2d(in_channels=32, out_channels=64, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),

            # Fourth convolutional layer (feature extraction)
            # Input shape: (batch_size, 64, 14, 14)
            # Kernel size: 3x3, Stride: 1, Padding: 1
            # Output shape: (batch_size, 64, 14, 14)
            nn.Conv2d(in_channels=64, out_channels=64, kernel_size=3, stride=1, padding=1, bias=False),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),

            # Max pooling layer (downsampling)
            # Input shape: (batch_size, 64, 14, 14)
            # Kernel size: 2x2, Stride: 2
            # Output shape: (batch_size, 64, 7, 7)
            nn.MaxPool2d(kernel_size=2, stride=2),

            # Dropout layer (regularization)
            nn.Dropout(0.5)
        )
        
        # Calculate the flattened size after the last pooling layer
        # (needed to determine the input size for the first fully connected layer;
        # this approach allows us to more easily change the encoder architecture
        # without having to manually calculate the flattened size based on the changes)
        dummy_input = torch.randn(1, 1, 28, 28)
        dummy_output = self.encoder(dummy_input)
        dummy_output = torch.flatten(dummy_output, 1)
        flattened_size = dummy_output.numel()

        # Classification layers
        self.classifier = nn.Sequential(
            # First fully connected layer (classification)
            # Input shape: (batch_size, flattened_size)
            # Output shape: (batch_size, 256)
            nn.Linear(flattened_size, 256, bias=False),
            nn.BatchNorm1d(256),
            nn.ReLU(inplace=True),

            # Dropout layer (regularization)
            nn.Dropout(0.5),
            
            # Second fully connected layer (classification)
            # Input shape: (batch_size, 256)
            # Output shape: (batch_size, 128)
            nn.Linear(256, 128, bias=False),
            nn.BatchNorm1d(128),
            nn.ReLU(inplace=True),

            # Dropout layer (regularization)
            nn.Dropout(0.5),
            
            # Third fully connected layer (classification)
            # Input shape: (batch_size, 128)
            # Output shape: (batch_size, 84)
            nn.Linear(128, 84, bias=False),
            nn.BatchNorm1d(84),
            nn.ReLU(inplace=True),

            # Dropout layer (regularization)
            nn.Dropout(0.5),

            # Output layer (final classification)
            nn.Linear(84, 10)
        )

        # Conditional custom weight initialization
        init_model_weights(self, weight_init)

    def forward(self, x):
        # Encode input into higher-dimensional representations
        x = self.encoder(x)

        # Flatten the input into 1D tensor that can be fed into the classifier
        x = torch.flatten(x, 1)

        # Classify encoded input into output classes
        x = self.classifier(x)

        # Convert the output to log probabilities
        #x = F.log_softmax(x, dim=1)

        # Return the classification probabilities
        return x

"""
Factory method to build a model based on the specified configuration.
"""
def build_model(model_config, model_state=None):
    def _create_torchvision_resnet18(**kwargs):
        model = torchvision.models.resnet18(**kwargs)
        model.conv1 = nn.Conv2d(1, 64, kernel_size=7, stride=2, padding=3, bias=False)
        model.fc = nn.Linear(512, 10)
        return model

    model_id = model_config['id']
    model_params = model_config.get('params', {})
    model = {
        "SimpleCNN": SimpleCNN,
        "LeNet5": LeNet5,
        "AdvancedCNN": AdvancedCNN,
        "TorchvisionPretrainedResNet18" : lambda **kwargs: _create_torchvision_resnet18(**kwargs)
    }[model_id](**model_params)
    if model_state: model.load_state_dict(model_state)
    model = model.to(DEVICE)
    return model

"""
Factory method to build an optimizer based on the specified configuration.
"""
def build_optimizer(model, optimizer_config):
    optimizer_id = optimizer_config['id']
    optimizer_params = optimizer_config.get('params', {})
    optimizer = {
        "Adam": optim.Adam,
        "SGD": optim.SGD
    }[optimizer_id](model.parameters(), **optimizer_params)
    return optimizer

"""
Factory method to build a learning rate scheduler based on the specified configuration.
"""
def build_lr_scheduler(optimizer, scheduler_config):
    if not scheduler_config: return None
    scheduler_id = scheduler_config['id']
    scheduler_params = scheduler_config.get('params', {})
    scheduler = {
        "StepLR": optim.lr_scheduler.StepLR,
        "ReduceLROnPlateau": optim.lr_scheduler.ReduceLROnPlateau
    }[scheduler_id](optimizer, **scheduler_params)
    return scheduler

"""
Factory method to build a loss function based on the specified configuration.
"""
def build_loss_function(loss_function_config):
    loss_function_id = loss_function_config['id']
    loss_function_params = loss_function_config.get('params', {})
    loss_function = {
        "CrossEntropyLoss": nn.CrossEntropyLoss
    }[loss_function_id](**loss_function_params)
    return loss_function

"""
Create data loaders for the training, validation, and test sets.
"""
def create_data_loaders(dataset="MNIST", batch_size=64, validation_split=0.2):
    # Define transformations for training data with augmentation
    train_transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.5,), (0.5,))
    ])

    # Define transformations for validation and test data (no augmentation)
    test_transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.5,), (0.5,))
    ])

    # Load the specified dataset
    if dataset == "MNIST":
        full_train_dataset = torchvision.datasets.MNIST(root='./data', train=True, download=True, transform=train_transform)
        test_dataset = torchvision.datasets.MNIST(root='./data', train=False, download=True, transform=test_transform)
    elif dataset == "FashionMNIST":
        full_train_dataset = torchvision.datasets.FashionMNIST(root='./data', train=True, download=True, transform=train_transform)
        test_dataset = torchvision.datasets.FashionMNIST(root='./data', train=False, download=True, transform=test_transform)
    elif dataset.startswith("EMNIST"):
        split = dataset.split('-')[1] if '-' in dataset else 'balanced'
        full_train_dataset = torchvision.datasets.EMNIST(root='./data', split=split, train=True, download=True, transform=train_transform)
        test_dataset = torchvision.datasets.EMNIST(root='./data', split=split, train=False, download=True, transform=test_transform)
    elif dataset == "KMNIST":
        full_train_dataset = torchvision.datasets.KMNIST(root='./data', train=True, download=True, transform=train_transform)
        test_dataset = torchvision.datasets.KMNIST(root='./data', train=False, download=True, transform=test_transform)
    elif dataset == "QMNIST":
        full_train_dataset = torchvision.datasets.QMNIST(root='./data', train=True, download=True, transform=train_transform)
        test_dataset = torchvision.datasets.QMNIST(root='./data', train=False, download=True, transform=test_transform)
    else:
        raise ValueError(f"Unsupported dataset: {dataset}")

    # Split the full training dataset into training and validation sets
    train_size = int((1 - validation_split) * len(full_train_dataset))
    validation_size = len(full_train_dataset) - train_size
    train_dataset, validation_dataset = random_split(full_train_dataset, [train_size, validation_size])

    # Setup the data loaders
    num_workers = multiprocessing.cpu_count()
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=False, num_workers=num_workers, pin_memory=True)
    validation_loader = DataLoader(validation_dataset, batch_size=batch_size, shuffle=False, num_workers=num_workers, pin_memory=True)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False, num_workers=num_workers, pin_memory=True)
    data_loaders = {'train': train_loader, 'validation': validation_loader, 'test': test_loader}

    return data_loaders

def _train(config, data_loaders, n_epochs):
    # Retrieve hyperparameters from the config
    model_config = config['model']
    optimizer_config = config["optimizer"]
    loss_function_config = config["loss_function"]
    lr_scheduler_config = config.get("lr_scheduler")

    # Unpack data loaders
    train_loader = data_loaders['train']
    
    # Build model, optimizer, learning rate scheduler, and loss function
    model = build_model(model_config)
    optimizer = build_optimizer(model, optimizer_config)
    lr_scheduler = build_lr_scheduler(optimizer, lr_scheduler_config)
    loss_function = build_loss_function(loss_function_config)

    # Log the model architecture
    wandb.watch(model)

    # Train for X epochs
    best_validation_accuracy = 0.0
    best_model_state = None
    best_epoch = None
    for epoch in tqdm(range(1, n_epochs + 1), desc="Training model"):
        # Set the model in training mode
        model.train()

        running_loss = 0.0
        correct = 0
        total = 0
        all_labels = []
        all_predictions = []
        for images, labels in train_loader:
            images, labels = images.to(DEVICE), labels.to(DEVICE)
            
            # Perform forward pass
            predictions = model(images)

            # Compute loss
            loss = loss_function(predictions, labels)

            # Perform backpropagation
            optimizer.zero_grad() # Zero out the gradients
            loss.backward() # Compute gradients
            optimizer.step() # Update weights

            # TODO: comment this
            running_loss += loss.item() * images.size(0)
            _, predicted = torch.max(predictions.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()

            # Collect labels and predictions for confusion matrix
            all_labels.extend(labels.cpu().numpy())
            all_predictions.extend(predicted.cpu().numpy())

        # Calculate train loss and accuracy
        train_loss = running_loss / len(train_loader.dataset)
        train_accuracy = 100 * correct / total

        # Evaluate the model on the test set
        eval_metrics = _evaluate(model, data_loaders, "validation")
        validation_accuracy = eval_metrics["validation_accuracy"]
        validation_loss = eval_metrics["validation_loss"]
        
        # Update the learning rate based on current validation loss
        if lr_scheduler: lr_scheduler.step(validation_loss)

        # Create metrics
        metrics = {
            "epoch": epoch,
            "train_loss": train_loss,
            "train_accuracy": train_accuracy,
            "best_validation_accuracy": best_validation_accuracy,
            **eval_metrics
        }
        if lr_scheduler: metrics["learning_rate"] = lr_scheduler.get_last_lr()[0]

        # Log metrics to W&B
        wandb.log(metrics, step=epoch)

        # If the model is the best so far, save it to disk
        if validation_accuracy > best_validation_accuracy:
            best_validation_accuracy = validation_accuracy
            best_epoch = epoch
            best_model_state = model.state_dict()
            logging.debug(f'Saved best model with accuracy: {best_validation_accuracy:.2f}%')

    # Return training results
    return {
        "best_validation_accuracy": best_validation_accuracy,
        "best_model_state": best_model_state,
        "best_epoch": best_epoch,
        "last_epoch": epoch
    }

"""
Train the model for the specified number of epochs.
Logs the training progress and results to W&B.
"""
def train(config, data_loaders, n_epochs, model_output_dir): 
    # Perform training within the context of a W&B run
    model_config = config['model']
    model_id = model_config['id']
    date_s = time.strftime('%Y%m%dT%H%M%S')
    run_id = f"train__{date_s}__{model_id}"
    with wandb.init(project=PROJECT_NAME, id=run_id, config=config) as run:
        # Perform training
        train_results = _train(config, data_loaders, n_epochs)
        best_model_state = train_results["best_model_state"]

        # Save the best model to disk
        best_model = build_model(model_config, model_state=best_model_state)
        if not os.path.exists(model_output_dir): os.makedirs(model_output_dir)
        best_model_path = f"{model_output_dir}/best_model_{run_id}.pth"
        torch.save(best_model, best_model_path)
        
        # Upload best model to W&B
        logging.info(f"Uploading model to W&B: {best_model_path}")
        wandb.save(best_model_path)
        artifact_name = best_model_path.split("/")[-1]
        artifact = wandb.Artifact(artifact_name, type='model')
        artifact.add_file(best_model_path)
        run.log_artifact(artifact)

        # Evaluate the best model on the test set
        eval_results = _evaluate(
            best_model, data_loaders, "test", 
            log_confusion_matrix=True, 
            log_misclassifications=True
        )
        wandb.log(eval_results)

        # Return training result
        return {
            **train_results,
            "best_model_path": best_model_path,
            **eval_results
        }

"""
Evaluates the model on the specified test set.
Logs the evaluation results to W&B.
"""
def evaluate(model, data_loaders, loader_type):
    date_s = time.strftime('%Y%m%dT%H%M%S')
    run_id = f"evaluate__{date_s}"
    with wandb.init(project=PROJECT_NAME, id=run_id): 
        metrics = _evaluate(model, data_loaders, loader_type)
        wandb.log(metrics)
        return metrics

def _evaluate(
    model, 
    data_loaders, 
    loader_type, 
    log_confusion_matrix=False, 
    log_misclassifications=False
):
    # Load the model if a path was provided
    if isinstance(model, str): model = load_model(model)
    
    # Ensure the model is on the correct device
    model = model.to(DEVICE)

    # Set the model in evaluation mode
    model.eval()

    # Define the loss function
    criterion = nn.CrossEntropyLoss()

    correct = total = running_loss = 0
    all_data = []
    misclassifications = []

    # Evaluate the model
    loader = data_loaders[loader_type]
    with torch.no_grad():  # Disable gradient tracking (no backpropagation needed for evaluation)
        for images, labels in loader:
            # Move images and labels to the device for inference
            images, labels = images.to(DEVICE), labels.to(DEVICE)

            # Perform forward pass
            predictions = model(images)

            # Compute loss
            loss = criterion(predictions, labels)
            running_loss += loss.item() * images.size(0)

            # Get the index of the max log-probability (the predicted class)
            _, predicted = torch.max(predictions, 1)

            # Update total and correct counts
            total += labels.size(0)
            correct += (predicted == labels).sum().item()

            # Move data back to CPU and collect results
            cpu_images, cpu_labels, cpu_predicted = images.cpu(), labels.cpu(), predicted.cpu()
            all_data.extend(zip(cpu_images, cpu_labels, cpu_predicted))
            
            # Collect misclassifications
            misclassifications.extend([
                (img, lbl, pred) for img, lbl, pred in zip(cpu_images, cpu_labels, cpu_predicted) 
                if lbl != pred
            ])

    # Extract all labels and predictions
    all_labels, all_predictions = zip(*[(lbl.item(), pred.item()) for _, lbl, pred in all_data])

    # Calculate metrics:
    # - Accuracy: the percentage of correctly classified samples
    # - Average loss: the average loss over all samples
    # - Precision: the weighted average of the precision score
    # - Recall: the weighted average of the recall score
    # - F1: the weighted average of the F1 score
    accuracy = 100 * correct / total
    average_loss = running_loss / total
    precision = precision_score(all_labels, all_predictions, average='weighted')
    recall = recall_score(all_labels, all_predictions, average='weighted')
    f1 = f1_score(all_labels, all_predictions, average='weighted')

    # Initialize evaluation metrics to be returned to the caller
    metrics = {
        f"{loader_type}_accuracy": accuracy,
        f"{loader_type}_loss": average_loss,
        f"{loader_type}_precision": precision,
        f"{loader_type}_recall": recall,
        f"{loader_type}_f1": f1
    }

    # Log confusion matrix
    if log_confusion_matrix:
        confusion_matrix_data = confusion_matrix(all_labels, all_predictions)
        plt.figure(figsize=(10, 10))
        sns.heatmap(confusion_matrix_data, annot=True, fmt='d', cmap='Blues')
        plt.xlabel('Predicted')
        plt.ylabel('True')
        confusion_matrix_image = wandb.Image(plt)
        plt.close()
        metrics[f"{loader_type}_confusion_matrix"] = confusion_matrix_image

    # Log misclassifications
    if log_misclassifications:
        misclassified_images = []
        for img, true_label, pred_label in misclassifications[:25]:  # Limit to 25 images
            plt.figure(figsize=(2, 2))
            plt.imshow(img.squeeze(), cmap='gray')
            plt.title(f'True: {true_label}, Pred: {pred_label}')
            plt.axis('off')
            misclassified_images.append(wandb.Image(plt))
            plt.close()
        metrics[f"{loader_type}_misclassified_images"] = misclassified_images

    # Return metrics
    return metrics

"""
Perform a hyperparameter sweep using W&B.
"""
def sweep(seeds=[42]):        
    data_loader_config = config['data_loader']

    # Perform training within the context of the sweep
    with wandb.init():
        # Merge selected sweep params with config
        sweep_config = parse_wandb_sweep_config(wandb.config)
        default_config = load_config()
        config = {**default_config, **sweep_config}

        # Perform training with multiple seeds to average 
        # out the randomness and get more robust results
        n_epochs = wandb.config.n_epochs
        best_validation_accuracies = []
        for seed in seeds:
            set_seed(seed)
            data_loaders = create_data_loaders(**data_loader_config)
            best_validation_accuracy, _ = _train(config, data_loaders, n_epochs)
            best_validation_accuracies.append(best_validation_accuracy)
        best_validation_accuracy = np.mean(best_validation_accuracies)
        
        # Set the score as the best validation accuracy
        # (we could make the score a function of multiple metrics)
        score = best_validation_accuracy

        # Log the score to be maximized by the sweep
        wandb.log({"score": score})

"""
Main function to parse command line arguments and run the script.
Runs by default when script is not being executed by a W&B sweep agent.
"""
def main():
    # Parse command line arguments
    parser = argparse.ArgumentParser(description='Train, evaluate, or tune a CNN on MNIST dataset.')
    parser.add_argument('mode', choices=['train', 'eval', 'sweep'], help='Mode to run the script in')
    parser.add_argument("--dataset", type=str, default="MNIST", help='Dataset to use for training and evaluation')
    parser.add_argument("--seed", type=int, default=42, help="Random seeds to use for training")
    parser.add_argument("--n_epochs", type=int, default=50, help='Number of epochs to train the model for')
    parser.add_argument("--hyperparams_path", type=str, default="configs/hyperparams/LeNet5.yml", help='Path to the hyperparameters file')
    parser.add_argument("--model_path", type=str, default="outputs/best_model.pth", help='Path to the model file for evaluation')
    parser.add_argument("--model_output_dir", type=str, default="outputs", help='Directory to save the model file')
    args = parser.parse_args()

    # Set the random seed for reproducibility
    set_seed(args.seed)

    # Create data loaders
    config = load_config(args.hyperparams_path)
    data_loader_config = config['data_loader']
    create_data_loaders_kwargs = {
        **data_loader_config, 
        "dataset": data_loader_config.get("dataset", args.dataset) # Use command-line dataset if none was specified in the config file
    }
    data_loaders = create_data_loaders(**create_data_loaders_kwargs)

    # Train the model
    if args.mode == 'train': train(config, data_loaders, args.n_epochs, args.model_output_dir)
    elif args.mode == 'eval': evaluate(args.model_path, data_loaders, "test")

if __name__ == '__main__':
    # In case this is being run by a wandb 
    # sweep agent then call the sweep code
    if "--sweep=1" in sys.argv: 
        sweep()
    # Otherwise assume normal run
    else: 
        main()
