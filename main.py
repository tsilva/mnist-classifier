# Configure maximum number of threads for NumExpr 
# before importing libs that require it
import os
import multiprocessing
NUM_CORES = multiprocessing.cpu_count()
if not "NUMEXPR_MAX_THREADS" in os.environ: os.environ["NUMEXPR_MAX_THREADS"] = str(NUM_CORES)

import argparse
import atexit
import logging
import json
import random
import sys
import time

import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
from sklearn.metrics import confusion_matrix, f1_score, precision_score, recall_score
import torch
import torch.nn as nn
import torch.optim as optim
import torchvision.transforms as transforms
from torch.utils.data import Subset, DataLoader, random_split
from tqdm import tqdm
import wandb
import yaml
import torch.cuda.amp as amp
from datasets import load_dataset, get_dataset_metrics, build_albumentations_pipeline, DatasetTransformWrapper

PROJECT_NAME = 'mnist-classifier'

# Define the default configuration for the model
DEFAULT_CONFIG = {
    "seed": 42,
    "use_mixed_precision": False,
    "logging": {
        "interval": 1
    },
    "data_loader": {
        "batch_size": 64,
        "validation_split": 0.0
    },
    "optimizer": {
        "id": "Adam"
    },
    "loss_function": {
        "id": "CrossEntropyLoss"
    },
    "early_stopping": {
        "id": "EarlyStopping",
        "params": {
            "patience": 20,
            "min_delta": 0.0001,
            "verbose": False
        }
    },
    "data_augmentation": {
        "pipeline": [
            {
                "name": "Rotate",
                "params": {"limit": 30}
            },
            {
                "name": "RandomResizedCrop",
                "params": {"height": 28, "width": 28, "scale": [0.8, 1.0]}
            },
            {
                "name": "GridDistortion",
                "params": {"num_steps": 5, "distort_limit": 0.3, "p": 0.5}
            },
            {
                "name": "CoarseDropout",
                "params": {"max_holes": 1, "max_height": 10, "max_width": 10, "p": 0.5}
            }
        ]
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

class EnsembleModel(nn.Module):

    def __init__(self, models):
        super(EnsembleModel, self).__init__()
        self.models = nn.ModuleList(models)

    def forward(self, x):
        outputs = [model(x) for model in self.models]
        return torch.stack(outputs).mean(dim=0)
    
class EarlyStopping:

    def __init__(self, patience=5, min_delta=0, verbose=False):
        self.patience = patience
        self.min_delta = min_delta
        self.verbose = verbose
        self.counter = 0
        self.best_score = None
        self.early_stop = False
        self.val_loss_min = np.Inf

    def __call__(self, val_loss):
        score = -val_loss

        if self.best_score is None:
            self.best_score = score
        elif score < self.best_score + self.min_delta:
            self.counter += 1
            if self.verbose:
                logging.debug(f'EarlyStopping counter: {self.counter} out of {self.patience}')
            if self.counter >= self.patience:
                self.early_stop = True
        else:
            self.best_score = score
            self.counter = 0

        return self.early_stop

class OptionalWandbContext:

    def __init__(self, use_wandb, *args, **kwargs):
        self.use_wandb = use_wandb
        self.args = args
        self.kwargs = kwargs
        self.run = None

    def __enter__(self):
        if self.use_wandb:
            self.run = wandb.init(*self.args, **self.kwargs)
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        if self.use_wandb and self.run:
            self.run.finish()

    def log(self, *args, **kwargs):
        if self.use_wandb:
            wandb.log(*args, **kwargs)

    def watch(self, *args, **kwargs):
        if self.use_wandb:
            wandb.watch(*args, **kwargs)

    def save(self, *args, **kwargs):
        if self.use_wandb:
            wandb.save(*args, **kwargs)

    def Artifact(self, *args, **kwargs):
        if self.use_wandb:
            return wandb.Artifact(*args, **kwargs)
        return None

    def log_artifact(self, *args, **kwargs):
        if self.use_wandb and self.run:
            self.run.log_artifact(*args, **kwargs)
    
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
        model_file_name = f"best_model_{run_id}.jit"
        artifact_path = f"{entity_id}/{project_name}/{model_file_name}:latest"
        logging.info(f"Downloading model artifact: {artifact_path}")
        artifact = wandb.use_artifact(artifact_path, type='model')
        artifact_dir = artifact.download()
        model_path = f"{artifact_dir}/{model_file_name}"
    
    if model_path.endswith('.jit'): model = torch.jit.load(model_path)
    elif model_path.endswith('.onnx'): model = torch.onnx.load(model_path)
    else: raise ValueError(f"Unsupported model file format: {model_path}")
    return model

def load_ensemble(model_path):
    if os.path.isdir(model_path):
        models = []
        for filename in os.listdir(model_path):
            if filename.endswith('jit') or filename.endswith('.onnx'):
                full_path = os.path.join(model_path, filename)
                models.append(load_model(full_path))
    else:
        models = [load_model(model_path)]
    
    return EnsembleModel(models)

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
    config = {**DEFAULT_CONFIG, **hyperparams}

    return config

def create_subset_loader(original_loader, percentage):
    dataset = original_loader.dataset
    num_samples = len(dataset)
    num_subset_samples = int(num_samples * percentage)
    
    # Create a list of random indices
    subset_indices = random.sample(range(num_samples), num_subset_samples)
    
    # Create a Subset of the original dataset
    subset = Subset(dataset, subset_indices)
    
    # Create a new DataLoader with the subset
    subset_loader = DataLoader(subset, batch_size=original_loader.batch_size, 
                               shuffle=True, num_workers=original_loader.num_workers)
    
    return subset_loader

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
LeNet-5 original model:

- 2 convolutional layers
- 3 fully connected layers
- Average pooling (legacy reasons from the original paper, max pooling would be better)
- Tanh activation functions (legacy reasons from the original paper, ReLU would be better)
"""
class LeNet5Original(nn.Module):
    def __init__(
        self, 
        conv1_filters=6, 
        conv2_filters=16, 
        conv3_filters=120, 
        fc1_neurons=84, 
        fc2_neurons=10
    ):
        super(LeNet5Original, self).__init__()

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
LeNet-5 model with improvements (based on Kaggle entry `https://www.kaggle.com/code/cdeotte/25-million-images-0-99757-mnist`):

- 6 convolutional layers
- 2 fully connected layers
- Batch normalization
- Dropout
- ReLU activation functions
"""
class LeNet5Improved(nn.Module):
    def __init__(self):
        super(LeNet5Improved, self).__init__()
        
        # Encodes the input into higher-dimensional representations
        self.encoder = nn.Sequential(
            # First convolutional layer
            # Input shape: (batch_size, 1, 28, 28)
            # Kernel size: 3x3, Stride: 1, Padding: 1
            # Output shape: (batch_size, 32, 28, 28)
            nn.Conv2d(in_channels=1, out_channels=32, kernel_size=3, padding=1),
            nn.ReLU(),

            # Second convolutional layer
            # Input shape: (batch_size, 32, 28, 28)
            # Kernel size: 3x3, Stride: 1, Padding: 1
            # Output shape: (batch_size, 32, 28, 28)
            nn.Conv2d(in_channels=32, out_channels=32, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.BatchNorm2d(32),
            
            # Third convolutional layer
            # Input shape: (batch_size, 32, 28, 28)
            # Kernel size: 3x3, Stride: 2, Padding: 1
            # Output shape: (batch_size, 64, 14, 14) 
            nn.Conv2d(in_channels=32, out_channels=64, kernel_size=3, padding=1, stride=2),
            nn.ReLU(),

            # Fourth convolutional layer
            # Input shape: (batch_size, 64, 14, 14)
            # Kernel size: 3x3, Stride: 1, Padding: 1
            # Output shape: (batch_size, 64, 14, 14)
            nn.Conv2d(in_channels=64, out_channels=64, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.BatchNorm2d(64),

            # Fifth convolutional layer
            # Input shape: (batch_size, 64, 14, 14)
            # Kernel size: 3x3, Stride: 2, Padding: 1
            # Output shape: (batch_size, 128, 7, 7)
            nn.Conv2d(in_channels=64, out_channels=128, kernel_size=3, padding=1, stride=2),
            nn.ReLU(),

            # Sixth convolutional layer
            # Input shape: (batch_size, 128, 7, 7)
            # Kernel size: 3x3, Stride: 1, Padding: 1
            # Output shape: (batch_size, 128, 7, 7)
            nn.Conv2d(in_channels=128, out_channels=128, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.BatchNorm2d(128)
        )
        
        # Classifies the encoded input into output classes
        self.classifier = nn.Sequential(
            # First fully connected layer
            # Input shape: (batch_size, 128*7*7)
            # Output shape: (batch_size, 256)
            nn.Linear(128*7*7, 256),
            nn.ReLU(),
            
            # Dropout layer
            nn.Dropout(p=0.5),

            # Second fully connected layer
            nn.Linear(256, 10)
        )
    
    def forward(self, x):
        # Pass through the encoder
        # Input shape: (batch_size, 1, 28, 28)
        # Output shape: (batch_size, 128, 7, 7)
        x = self.encoder(x)

        # Flatten the output for the classifier
        x = x.view(x.size(0), -1)

        # Pass through the classifier
        # Input shape: (batch_size, 128*7*7)
        # Output shape: (batch_size, 10)
        x = self.classifier(x)

        # Return output with shape (batch_size, 10)
        return x

"""
Advanced CNN model:

- 7 convolutional layers
- Different kernel sizes and strides
- 1 fully connected layer
- Batch normalization
- Dropout
- ReLU activation functions
"""
class AdvancedCNN(nn.Module):
    def __init__(self):
        super(AdvancedCNN, self).__init__()
            
        # Encodes the input into higher-dimensional representations
        self.encoder = nn.Sequential(
            # First convolutional layer
            # Input shape: (batch_size, 1, 28, 28)
            # Kernel size: 3x3, Stride: 1, Padding: 0
            # Output shape: (batch_size, 32, 26, 26)
            nn.Conv2d(1, 32, kernel_size=3),
            nn.ReLU(),
            nn.BatchNorm2d(32),

            # Second convolutional layer
            # Input shape: (batch_size, 32, 26, 26)
            # Kernel size: 3x3, Stride: 1, Padding: 0
            # Output shape: (batch_size, 32, 24, 24)
            nn.Conv2d(32, 32, kernel_size=3),
            nn.ReLU(),
            nn.BatchNorm2d(32),

            # Third convolutional layer
            # Input shape: (batch_size, 32, 24, 24)
            # Kernel size: 5x5, Stride: 2, Padding: 2
            # Output shape: (batch_size, 32, 12, 12)
            nn.Conv2d(32, 32, kernel_size=5, stride=2, padding=2),
            nn.ReLU(),
            nn.BatchNorm2d(32),
            nn.Dropout(0.4),
            
            # Fourth convolutional layer
            # Input shape: (batch_size, 32, 12, 12)
            # Kernel size: 3x3, Stride: 1, Padding: 0
            # Output shape: (batch_size, 64, 10, 10)
            nn.Conv2d(32, 64, kernel_size=3),
            nn.ReLU(),
            nn.BatchNorm2d(64),

            # Fifth convolutional layer
            # Input shape: (batch_size, 64, 10, 10)
            # Kernel size: 3x3, Stride: 1, Padding: 0
            # Output shape: (batch_size, 64, 8, 8)
            nn.Conv2d(64, 64, kernel_size=3),
            nn.ReLU(),
            nn.BatchNorm2d(64),

            # Sixth convolutional layer
            # Input shape: (batch_size, 64, 8, 8)
            # Kernel size: 5x5, Stride: 2, Padding: 2
            # Output shape: (batch_size, 64, 4, 4)
            nn.Conv2d(64, 64, kernel_size=5, stride=2, padding=2),
            nn.ReLU(),
            nn.BatchNorm2d(64),
            nn.Dropout(0.4),
            
            # Seventh convolutional layer
            # Input shape: (batch_size, 64, 4, 4)
            # Kernel size: 4x4, Stride: 1, Padding: 0
            # Output shape: (batch_size, 128, 1, 1)
            nn.Conv2d(64, 128, kernel_size=4),
            nn.ReLU(),
            nn.BatchNorm2d(128),
            nn.Dropout(0.4)
        )

        # Classifies the encoded input into output classes
        self.decoder = nn.Sequential(
            nn.Linear(128, 10)
        )

    def forward(self, x):
        # Pass through the encoder
        # Input shape: (batch_size, 1, 28, 28)
        # Output shape: (batch_size, 128, 1, 1)
        x = self.encoder(x)

        # Flatten the output for the classifier
        x = x.view(x.size(0), -1)
        
        # Pass through the classifier
        # Input shape: (batch_size, 128)
        # Output shape: (batch_size, 10)
        x = self.decoder(x)

        # Return output with shape (batch_size, 10)
        return x
        
"""
Factory method to build a model based on the specified configuration.
"""
def build_model(model_config, model_state=None):
    model_id = model_config['id']
    model_params = model_config.get('params', {})
    model_constructor = {
        "LeNet5Original": LeNet5Original,
        "LeNet5Improved": LeNet5Improved,
        "AdvancedCNN": AdvancedCNN
    }[model_id]
    model = model_constructor(**model_params)
    if model_state: model.load_state_dict(model_state)
    else: init_model_weights(model, 'he') # TODO: softcode
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
        "AdamW": optim.AdamW,
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
        "ReduceLROnPlateau": optim.lr_scheduler.ReduceLROnPlateau,
        "CosineAnnealingLR": optim.lr_scheduler.CosineAnnealingLR,
        "OneCycleLR": optim.lr_scheduler.OneCycleLR
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

def build_early_stopping(early_stopping_config):
    early_stopping_id = early_stopping_config['id']
    early_stopping_params = early_stopping_config.get('params', {})
    early_stopping = {
        "EarlyStopping": EarlyStopping
    }[early_stopping_id](**early_stopping_params)
    return early_stopping

def _data_loader_worker_init_fn(worker_id):
    worker_info = torch.utils.data.get_worker_info()
    base_seed = worker_info.seed
    worker_seed = (base_seed + worker_id) % 2**32
    set_seed(worker_seed)

def create_data_loaders(config, dataset_name="mnist", batch_size=64, validation_split=0.0):
    # Retrieve config params
    seed = config['seed']
    data_augmentation_config = config.get('data_augmentation', {})
    data_augmentation_pipeline_config = data_augmentation_config.get('pipeline', [])

    # Load dataste and calculate its metrics
    dataset = load_dataset(dataset_name)
    dataset_metrics = get_dataset_metrics(dataset)
    logging.info("Dataset metrics:" + json.dumps(dataset_metrics, indent=2))

    # Unpack the dataset into training, validation, and test sets
    train_dataset = dataset['train']
    test_dataset = dataset['test']
    validation_dataset = test_dataset

    # In case a validation split was specified then replace 
    # the validation set with a subset of the training set
    if validation_split > 0:
        train_size = int((1 - validation_split) * len(train_dataset))
        validation_size = len(train_dataset) - train_size
        train_dataset, validation_dataset = random_split(train_dataset, [train_size, validation_size])

    # Apply the default transformation to the datasets
    # (normalize the inputs using the overall mean and standard deviation)
    dataset_mean = dataset_metrics['mean']
    dataset_std = dataset_metrics['std']
    default_transform = transforms.Compose([
        transforms.Normalize((dataset_mean,), (dataset_std,))
    ])

    # Build the data augmentation pipeline
    augment_transform = build_albumentations_pipeline(data_augmentation_pipeline_config, dataset_mean, dataset_std)
       
    # Wrap datasets with their respective transform operations
    # (this will make transformations work regardless of wheter
    # the dataset is a Dataset or a Subset)
    train_dataset = DatasetTransformWrapper(train_dataset, transform=augment_transform)
    validation_dataset = DatasetTransformWrapper(validation_dataset, transform=default_transform)
    test_dataset = DatasetTransformWrapper(test_dataset, transform=default_transform)

    # Create the data loaders
    num_workers = NUM_CORES
    prefetch_factor = 2
    train_loader = DataLoader(
        train_dataset, 
        batch_size=batch_size, 
        shuffle=True, 
        num_workers=num_workers,
        persistent_workers=True,
        worker_init_fn=_data_loader_worker_init_fn, 
        generator=torch.Generator().manual_seed(seed), 
        prefetch_factor=prefetch_factor,
        pin_memory=True
    )
    validation_loader = DataLoader(
        validation_dataset, 
        batch_size=batch_size, 
        shuffle=False, 
        num_workers=num_workers, 
        persistent_workers=True,
        worker_init_fn=_data_loader_worker_init_fn, 
        generator=torch.Generator().manual_seed(seed), 
        prefetch_factor=prefetch_factor,
        pin_memory=True
    )
    test_loader = DataLoader(
        test_dataset, 
        batch_size=batch_size, 
        shuffle=False, 
        num_workers=num_workers, 
        persistent_workers=True,
        worker_init_fn=_data_loader_worker_init_fn, 
        generator=torch.Generator().manual_seed(seed), 
        prefetch_factor=prefetch_factor,
        pin_memory=True
    )

    # Return the data loaders
    return {
        'train': train_loader, 
        'validation': validation_loader, 
        'test': test_loader
    }


def _train(config, data_loaders, n_epochs, best_model_path, train_data_percentage=None, wandb_run=None):
    # Retrieve hyperparameters from the config
    use_mixed_precision = config["use_mixed_precision"]
    model_config = config['model']
    optimizer_config = config["optimizer"]
    loss_function_config = config["loss_function"]
    lr_scheduler_config = config.get("lr_scheduler")
    early_stopping_config = config.get("early_stopping", {
        "id" : "EarlyStopping",
        "params" : {
            "patience": 10, 
            "min_delta": 0.001, 
            "verbose": True
        }
    })
    logging_config = config.get("logging", {})
    logging_interval = logging_config["interval"]

    # Unpack data loaders
    train_loader = data_loaders['train']

    # If a subset of the training data should be used, create a new loader
    # (eg: this is used for creating bagging ensembles where each model 
    # is trained on a different subset of the training data)
    if train_data_percentage: train_loader = create_subset_loader(train_loader, train_data_percentage)
    
    # Build model, optimizer, learning rate scheduler, and loss function
    model = build_model(model_config)
    optimizer = build_optimizer(model, optimizer_config)
    lr_scheduler = build_lr_scheduler(optimizer, lr_scheduler_config)
    loss_function = build_loss_function(loss_function_config)
    early_stopping = build_early_stopping(early_stopping_config) 
    gradient_scaler = amp.GradScaler() if use_mixed_precision else None

    # Log the model architecture
    if wandb_run: wandb_run.watch(model)

    # Train for X epochs
    best_validation_accuracy = 0.0
    best_model = None
    best_epoch = None
    for epoch in tqdm(range(1, n_epochs + 1), desc="Training model"):
        # Set the model in training mode
        model.train()

        running_loss = 0.0
        correct = 0
        total = 0

        num_batch_loads = 0
        batch_load_start = time.time()
        total_batch_load_time = 0
        for images, labels in train_loader:
            images, labels = images.to(DEVICE, non_blocking=True), labels.to(DEVICE, non_blocking=True)
            num_batch_loads += 1
            total_batch_load_time += time.time() - batch_load_start

            if use_mixed_precision:
                with amp.autocast():
                    predictions = model(images)
                    loss = loss_function(predictions, labels)
                optimizer.zero_grad()
                gradient_scaler.scale(loss).backward()
                gradient_scaler.step(optimizer)
                gradient_scaler.update()
            else:
                # Perform forward pass
                predictions = model(images)

                # Compute loss
                loss = loss_function(predictions, labels)

                # Perform backpropagation
                optimizer.zero_grad() # Zero out the gradients
                loss.backward() # Compute gradients
                optimizer.step() # Update weights

            # Accumulate metrics on GPU
            running_loss += loss
            _, predicted = torch.max(predictions.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum()

            # Reset the batch load timer
            batch_load_start = time.time()
        
        # Calculate average batch load time
        average_batch_load_time = total_batch_load_time / num_batch_loads

        # Calculate metrics on CPU only once per epoch
        train_loss = (running_loss / len(train_loader)).item()
        train_accuracy = (100 * correct / total).item()

        # Evaluate performance on the validation set
        validation_metrics = _evaluate(
            config, model, data_loaders, "validation", 
            log_confusion_matrix=epoch % logging_interval == 0,
            log_misclassifications=epoch % logging_interval == 0, 
        )
        validation_accuracy = validation_metrics["validation/accuracy"]
        validation_loss = validation_metrics["validation/loss"]

        # If the model is the best so far, save it to disk
        if validation_accuracy > best_validation_accuracy:
            best_validation_accuracy = validation_accuracy
            best_epoch = epoch
            best_model = torch.jit.script(model)
            best_model.save(best_model_path)
            logging.debug(f'Saved best model with accuracy: {best_validation_accuracy:.2f}%')

        # Create metrics
        learning_rate = lr_scheduler.get_last_lr()[0] if lr_scheduler else None
        metrics = {
            "train/loss": train_loss,
            "train/accuracy": train_accuracy,
            "train/batches/total_load_time": total_batch_load_time,
            "train/batches/average_load_time": average_batch_load_time,
            "validation/best_accuracy": best_validation_accuracy,
            **validation_metrics
        }
        if learning_rate: metrics["train/learning_rate"] = learning_rate

        # Log metrics to W&B
        if wandb_run: wandb_run.log(metrics, step=epoch)

        # Periodic logging to console
        if epoch % logging_interval == 0:
            logging.info(json.dumps({
                "epoch": epoch,
                "train/loss" : train_loss,
                "train/accuracy" : train_accuracy,
                "train/learning_rate" : learning_rate,
                "validation/loss" : validation_loss,
                "validation/accuracy" : validation_accuracy,
                "validation/best_accuracy" : best_validation_accuracy
            }, indent=4))

        # Update the learning rate based on current validation loss
        if lr_scheduler: 
            lr_scheduler.step(validation_loss)

        # Check if we should stop early
        if early_stopping and early_stopping(validation_loss):
            logging.info(f"Early stopping triggered at epoch {epoch}")
            break
        
    # Return training results
    return {
        "train/last_epoch": epoch,
        "validation/best_accuracy": best_validation_accuracy,
        "validation/best_epoch": best_epoch,
        "validation/best_model": best_model
    }

"""
Train the model for the specified number of epochs.
Logs the training progress and results to W&B.
"""
def train(config, data_loaders, n_epochs, model_output_dir, n_models=1, wandb_enabled=False): 
    # Perform training within the context of a W&B run
    model_config = config['model']
    model_id = model_config['id']

    # Create the model output dir in case it doesn't exist yet
    if not os.path.exists(model_output_dir): os.makedirs(model_output_dir)

    # Train the number of specified models
    # (eg: >1 for training ensembles)
    results = {}
    for model_idx in range(n_models):
        if n_models > 1: model_run_id = f"{model_id}__{time.strftime('%Y%m%d')}__{model_idx}"
        else: model_run_id = f"{model_id}__{time.strftime('%Y%m%dT%H%M%S')}"
        wandb_run_id = f"train__{model_run_id}"
        with OptionalWandbContext(wandb_enabled, project=PROJECT_NAME, id=wandb_run_id, config=config) as run:                
            # Perform training
            best_model_path = f"{model_output_dir}/{model_run_id}.jit"
            train_results = _train(config, data_loaders, n_epochs, best_model_path, wandb_run=run)
            best_model = train_results["validation/best_model"]
            
            # Upload best model to W&B
            logging.info(f"Uploading model to W&B: {best_model_path}")
            run.save(best_model_path)
            artifact_name = best_model_path.split("/")[-1]
            artifact = wandb.Artifact(artifact_name, type='model')
            artifact.add_file(best_model_path)
            run.log_artifact(artifact)

            # Evaluate the best model on the test set
            test_metrics = _evaluate(
                config, best_model, data_loaders, "test", 
                log_confusion_matrix=True, 
                log_misclassifications=True
            )
            run.log(test_metrics)     

            # Set the training results
            results[model_idx] = {
                **train_results,
                "best_model_path": best_model_path
            }
    
    # Return training result
    return results       

"""
Evaluates the model on the specified test set.
Logs the evaluation results to W&B.
"""
def evaluate(config, model, data_loaders, loader_type, wandb_enabled=False):
    date_s = time.strftime('%Y%m%dT%H%M%S')
    run_id = f"evaluate__{date_s}"
    with OptionalWandbContext(wandb_enabled, project=PROJECT_NAME, id=run_id) as run: 
        metrics = _evaluate(config, model, data_loaders, loader_type)
        if wandb_enabled: run.log(metrics)
        else: logging.info(json.dumps(metrics, indent=4))
        return metrics

def _evaluate(
    config,
    model, 
    data_loaders, 
    loader_type, 
    log_confusion_matrix=False, 
    log_misclassifications=False
):
    use_mixed_precision = config["use_mixed_precision"]

    # Load the model if a path was provided
    if isinstance(model, str): model = load_ensemble(model)
    
    # Ensure the model is on the correct device
    model = model.to(DEVICE)

    # Set the model in evaluation mode
    model.eval()

    # Define the loss function
    loss_function = nn.CrossEntropyLoss()

    correct = total = running_loss = 0
    all_labels = []
    all_predictions = []
    misclassifications = []

    # Evaluate the model
    loader = data_loaders[loader_type]
    with torch.no_grad():  # Disable gradient tracking (no backpropagation needed for evaluation)
        num_batch_loads = 0
        batch_load_start = time.time()
        total_batch_load_time = 0
        for images, labels in loader:
            images, labels = images.to(DEVICE, non_blocking=True), labels.to(DEVICE, non_blocking=True)
            num_batch_loads += 1
            total_batch_load_time += time.time() - batch_load_start

            if use_mixed_precision:
                with amp.autocast():
                    predictions = model(images)
                    loss = loss_function(predictions, labels)
            else:
                predictions = model(images)
                loss = loss_function(predictions, labels)

            # Compute loss
            running_loss += loss.item() * images.size(0)

            # Get the index of the max log-probability (the predicted class)
            _, predicted = torch.max(predictions, 1)

            # Update total and correct counts
            total += labels.size(0)
            correct += (predicted == labels).sum().item()

            # Collect labels and predictions
            all_labels.append(labels)
            all_predictions.append(predicted)

            # Collect misclassifications
            misclassifications.extend([
                (img, lbl, pred) for img, lbl, pred in zip(images, labels, predicted) 
                if lbl != pred
            ])

            # Reset the batch load timer
            batch_load_start = time.time()

    # Concatenate all labels and predictions
    all_labels = torch.cat(all_labels).cpu()
    all_predictions = torch.cat(all_predictions).cpu()

    # Calculate average batch load time
    average_batch_load_time = total_batch_load_time / num_batch_loads

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
        f"{loader_type}/batches/total_load_time": total_batch_load_time,
        f"{loader_type}/batches/average_load_time": average_batch_load_time,
        f"{loader_type}/accuracy": accuracy,
        f"{loader_type}/loss": average_loss,
        f"{loader_type}/precision": precision,
        f"{loader_type}/recall": recall,
        f"{loader_type}/f1": f1
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
        metrics[f"{loader_type}/confusion_matrix"] = confusion_matrix_image

    # Log misclassifications
    if log_misclassifications:
        misclassified_images = []
        for img, true_label, pred_label in misclassifications[:25]:  # Limit to 25 images
            plt.figure(figsize=(2, 2))
            plt.imshow(img.squeeze().cpu(), cmap='gray')
            plt.title(f'True: {true_label.item()}, Pred: {pred_label.item()}')
            plt.axis('off')
            misclassified_images.append(wandb.Image(plt))
            plt.close()
        metrics[f"{loader_type}/misclassified"] = misclassified_images

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
            data_loaders = create_data_loaders({**config, "seed" : seed}, **data_loader_config)
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
    parser.add_argument("--dataset", type=str, default="mnist", help='Dataset to use for training and evaluation')
    parser.add_argument("--seed", type=int, default=42, help="Random seeds to use for training")
    parser.add_argument("--n_epochs", type=int, default=200, help='Number of epochs to train the model for')
    parser.add_argument("--n_models", type=int, default=1, help='How many models to train (eg: train an ensemble)')
    parser.add_argument("--hyperparams_path", type=str, default="configs/hyperparams/AdvancedCNN.yml", help='Path to the hyperparameters file')
    parser.add_argument("--model_path", type=str, help='Path to the model file for evaluation')
    parser.add_argument("--model_output_dir", type=str, default="outputs", help='Directory to save the model file')
    parser.add_argument("--wandb_enabled", type=bool, default=False, help='Whether to enable W&B logging')
    args = parser.parse_args()

    # Set the random seed for reproducibility
    set_seed(args.seed)

    # Create data loaders
    config = load_config(args.hyperparams_path)
    data_loader_config = config['data_loader']
    create_data_loaders_kwargs = {
        **data_loader_config, 
        "dataset_name": data_loader_config.get("dataset", args.dataset) # Use command-line dataset if none was specified in the config file
    }
    data_loaders = create_data_loaders(config, **create_data_loaders_kwargs)

    # Train the model
    if args.mode == 'train': train(config, data_loaders, args.n_epochs, args.model_output_dir, n_models=args.n_models)
    elif args.mode == 'eval': evaluate(config, args.model_path, data_loaders, "test")

if __name__ == '__main__':
    # In case this is being run by a wandb 
    # sweep agent then call the sweep code
    if "--sweep=1" in sys.argv: 
        sweep()
    # Otherwise assume normal run
    else: 
        main()
