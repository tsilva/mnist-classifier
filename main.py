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

from deepmerge import Merger
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
from sklearn.metrics import confusion_matrix, f1_score, precision_score, recall_score
import torch
import torch.nn as nn
import torch.optim as optim
import torchvision.transforms as transforms
from torch.utils.data import DataLoader, random_split
from tqdm import tqdm
import wandb
import yaml
import torch.cuda.amp as amp
from datasets import load_dataset, get_dataset_metrics, build_albumentations_pipeline, DatasetTransformWrapper, create_bootstrap_dataset
from models import build_model, load_model

PROJECT_NAME = 'mnist-classifier'

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

class CustomJSONEncoder(json.JSONEncoder):
    def default(self, obj):
        try:
            return json.JSONEncoder.default(self, obj)
        except (TypeError, OverflowError):
            return None  # Or some other placeh

class CustomDataLoader(DataLoader):
    def __init__(self, *args, device=None, **kwargs):
        super().__init__(*args, **kwargs)

        assert device is not None, "Device must be specified"
        self.device = device

    def __iter__(self):
        self.iter_data = super().__iter__()
        self.sample_times = []
        self.transfer_times = []
        self.load_times = []
        self.processing_time = []
        self.last_end_time = None
        return self

    def __next__(self):
        if self.last_end_time is not None:
            processing_time = time.time() - self.last_end_time
            self.processing_time.append(processing_time)

        # Measure time to load data
        start_time = time.time()
        batch = next(self.iter_data)
        sample_time = time.time() - start_time
        self.sample_times.append(sample_time)

        # Measure time to transfer data to GPU
        start_time = time.time()
        batch = self._transfer_to_device(batch)
        transfer_time = time.time() - start_time
        self.transfer_times.append(transfer_time)
        
        load_time = sample_time + transfer_time
        self.load_times.append(load_time)

        self.last_end_time = time.time()

        return batch

    def _transfer_to_device(self, batch):
        if isinstance(batch, dict):
            return {k: v.to(self.device, non_blocking=True) for k, v in batch.items()}
        elif isinstance(batch, list) or isinstance(batch, tuple):
            return [self._transfer_to_device(v) for v in batch]
        else:
            return batch.to(self.device, non_blocking=True)

    def get_stats(self):
        load_time = np.sum(self.load_times)
        batches_per_second = 1.0 / load_time
        stats = {
            "sample_time": round(np.sum(self.sample_times), 2),
            "transfer_time": round(np.sum(self.transfer_times), 2),
            "load_time": round(load_time, 2),
            "processing_time": round(np.sum(self.processing_time), 2),
            "batches_per_second": round(batches_per_second, 4)
        }
        return stats
        
class BasicPatienceEarlyStopping:
    """
    Early stopping detector to stop training when the model stops improving.
    """

    def __init__(self, patience=5, min_delta=0):
        self.patience = patience
        self.min_delta = min_delta
        self.counter = 0
        self.best_score = None
        self.early_stop_triggered = False

    def __call__(self, score):
        assert self.early_stop_triggered == False, "Early stopping is already triggered"
        best_score_threshold = self.best_score + self.min_delta if self.best_score is not None else None
        if self.best_score is None or score >= best_score_threshold: self.best_score, self.counter = score, 0
        else: self.early_stop_triggered, self.counter = self.counter >= self.patience, self.counter + 1
        return self.early_stop_triggered, self.best_score, self.counter, self.patience

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
    
def set_seed(seed):
    """
    Set a seed in the random number generators for reproducibility.
    """

    random.seed(seed) # Set the seed for the random number generator
    np.random.seed(seed) # Set the seed for NumPy
    torch.manual_seed(seed) # Set the seed for PyTorch
    torch.cuda.manual_seed(seed) # Set the seed for CUDA
    torch.cuda.manual_seed_all(seed) # Set the seed for all CUDA devices
    torch.backends.cudnn.deterministic = True # Ensure deterministic results (WARNING: can slow down training!)
    torch.backends.cudnn.benchmark = False # Disable cuDNN benchmarking (WARNING: can slow down training!)

def load_config(config_path, extra_config={}):
    """
    Creates the configuration data structure by merging configuration data from multiple sources.
    The configuration data holds all parameters that will have an impact on the training process
    (eg: model output dir is not part of the config because it doesn't affect the training process).
    """

    config_root = "configs/train/" # TODO: move to constant
    assert config_path.startswith(config_root), f"config path must start with {config_root}"
    assert not config_path.endswith("_default.yml"), "config path must not end with _default.yml"

    config_id = config_path.replace(config_root, "").replace(".yml", "")
    config_id = config_id.replace("/", "__")

    def _collect_default_configs(path):
        """
        Collect all _default.yml files from the specified path up to the configs/train/ directory.
        """
        default_configs = [{"id": config_id}]
        while path and os.path.dirname(path) != path:
            default_config_path = os.path.join(path, "_default.yml")
            if os.path.exists(default_config_path):
                with open(default_config_path, "r", encoding="utf-8") as file:
                    default_configs.append(yaml.safe_load(file))
            if path.endswith(config_root):
                break
            path = os.path.dirname(path)
        return default_configs

    # Define the merger object with custom strategies using a lambda function
    merger = Merger(
        [
            (list, lambda config, path, base, nxt: nxt), 
            (dict, "merge") # Merge dictionaries
        ],  
        # List merge strategy using a lambda
        ["override"], # Default strategies for other types
        ["override"]  # Default conflict resolution strategy
    )

    # Merge all default configurations
    config_dir = os.path.dirname(config_path)
    default_configs = list(reversed(_collect_default_configs(config_dir)))
    
    merged_default_config = {}
    for default_config in default_configs:
        merged_default_config = merger.merge(merged_default_config, default_config)

    # Load the specified configuration
    with open(config_path, "r", encoding="utf-8") as file:
        file_config = yaml.safe_load(file)

    # Merge the configurations
    config = merger.merge(merged_default_config, file_config)
    config = merger.merge(config, extra_config)
    
    return config


def parse_wandb_sweep_config(sweep_config):
    """
    Parse the sweep configuration from W&B to a dictionary.
    Mapping dependent parameters from the flattened structure of the sweep config to the nested structure of our config.
    """

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


def build_optimizer(model, optimizer_config):
    """
    Factory method to build an optimizer based on the specified configuration.
    """

    optimizer_id = optimizer_config['id']
    optimizer_params = optimizer_config.get('params', {})
    optimizer = {
        "Adam": optim.Adam,
        "AdamW": optim.AdamW,
        "SGD": optim.SGD
    }[optimizer_id](model.parameters(), **optimizer_params)
    return optimizer


def build_lr_scheduler(optimizer, scheduler_config):
    """
    Factory method to build a learning rate scheduler based on the specified configuration.
    """

    if not scheduler_config: return None
    scheduler_id = scheduler_config['id']
    scheduler_params = scheduler_config.get('params', {})
    scheduler = {
        "StepLR": optim.lr_scheduler.StepLR,
        "CyclicLR": optim.lr_scheduler.CyclicLR,
        "ReduceLROnPlateau": optim.lr_scheduler.ReduceLROnPlateau,
        "CosineAnnealingLR": optim.lr_scheduler.CosineAnnealingLR,
        "OneCycleLR": optim.lr_scheduler.OneCycleLR
    }[scheduler_id](optimizer, **scheduler_params)
    return scheduler


def build_loss_function(loss_function_config):
    """
    Factory method to build a loss function based on the specified configuration.
    """

    loss_function_id = loss_function_config['id']
    loss_function_params = loss_function_config.get('params', {})
    loss_function = {
        "CrossEntropyLoss": nn.CrossEntropyLoss
    }[loss_function_id](**loss_function_params)
    return loss_function

def build_early_stopping(early_stopping_config):
    if not early_stopping_config: return None
    early_stopping_id = early_stopping_config['id']
    early_stopping_params = early_stopping_config.get('params', {})
    early_stopping = {
        "BasicPatienceEarlyStopping": BasicPatienceEarlyStopping
    }[early_stopping_id](**early_stopping_params)
    return early_stopping

def _data_loader_worker_init_fn(worker_id):
    worker_info = torch.utils.data.get_worker_info()
    base_seed = worker_info.seed
    worker_seed = (base_seed + worker_id) % 2**32
    set_seed(worker_seed)

def create_data_loaders(config, dataset_id="mnist", batch_size=64, train_bootstrap_percentage=None, validation_split=None):
    # Retrieve config params
    seed = config['seed']
    data_augmentation_config = config.get('data_augmentation', {})
    data_augmentation_pipeline_config = data_augmentation_config.get('pipeline', [])

    # Load dataste and calculate its metrics
    dataset = load_dataset(dataset_id)
    dataset_metrics = get_dataset_metrics(dataset)
    logging.info("Dataset metrics:" + json.dumps(dataset_metrics, indent=2))

    # Unpack the dataset into training, validation, and test sets
    train_dataset = dataset['train']
    test_dataset = dataset['test']
    validation_dataset = test_dataset

    # In case a bootstrap percentage was specified then create 
    # an alternative bootstap dataset (new dataset is created 
    # through resampling with repetition of original dataset,
    # useful for creating bagging ensemble models)
    if train_bootstrap_percentage:
        train_dataset = create_bootstrap_dataset(train_dataset, train_bootstrap_percentage)
    
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
    train_loader = CustomDataLoader(
        train_dataset, 
        batch_size=batch_size, 
        shuffle=True, 
        num_workers=num_workers,
        persistent_workers=True,
        worker_init_fn=_data_loader_worker_init_fn, 
        generator=torch.Generator().manual_seed(seed), 
        prefetch_factor=prefetch_factor,
        pin_memory=True,
        device=DEVICE
    )
    validation_loader = CustomDataLoader(
        validation_dataset, 
        batch_size=batch_size, 
        shuffle=False, 
        num_workers=num_workers, 
        persistent_workers=True,
        worker_init_fn=_data_loader_worker_init_fn, 
        generator=torch.Generator().manual_seed(seed), 
        prefetch_factor=prefetch_factor,
        pin_memory=True,
        device=DEVICE
    )
    test_loader = CustomDataLoader(
        test_dataset, 
        batch_size=batch_size, 
        shuffle=False, 
        num_workers=num_workers, 
        persistent_workers=True,
        worker_init_fn=_data_loader_worker_init_fn, 
        generator=torch.Generator().manual_seed(seed), 
        prefetch_factor=prefetch_factor,
        pin_memory=True,
        device=DEVICE
    )

    # Return the data loaders
    return {
        'train': train_loader, 
        'validation': validation_loader, 
        'test': test_loader
    }


def _train(config, data_loaders, n_epochs, best_model_path, wandb_run=None):
    # Retrieve hyperparameters from the config
    use_mixed_precision = config["use_mixed_precision"]
    model_config = config['model']
    optimizer_config = config["optimizer"]
    loss_function_config = config["loss_function"]
    lr_scheduler_config = config.get("lr_scheduler")
    early_stopping_config = config.get("early_stopping")
    logging_config = config.get("logging", {})
    logging_interval = logging_config["interval"]

    # Unpack data loaders
    train_loader = data_loaders['train']

    # Build model
    model = build_model(model_config)

    # init_model_weights(model, 'he') # TODO: not seeing much benefit from 'he' init so disabling for now
    model = model.to(DEVICE)

    # Build optimizer, loss function, and learning rate scheduler
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
        epoch_start = time.time()
        
        # Set the model in training mode
        model.train()

        running_loss = 0.0
        correct = 0
        total = 0
        save_best_model_meta_path = None

        for images, labels in train_loader:
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

        train_loader_metrics = train_loader.get_stats()
        train_loader_metrics = {f"train/loader/{k}": v for k, v in train_loader_metrics.items()}
        
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
        is_best_model = validation_accuracy > best_validation_accuracy
        if is_best_model:
            best_validation_accuracy = validation_accuracy
            best_epoch = epoch
            best_model = torch.jit.script(model)
            best_model.save(best_model_path)
            logging.debug(f'Saved best model with accuracy: {best_validation_accuracy:.2f}%')

        # Update the learning rate based on current validation loss
        if lr_scheduler: 
            lr_scheduler.step(validation_loss)

        # In case early stopping is enabled then 
        # check if we should stop training
        score = -validation_loss
        early_stop_triggered, early_stop_best_score, early_stop_counter, early_stop_patience = early_stopping(score) if early_stopping else (False, None, None, None)
        early_stopping_metrics = {
            "early_stopping/best_score": early_stop_best_score, 
            "early_stopping/counter": early_stop_counter, 
            "early_stopping/patience": early_stop_patience,
            "early_stopping/patience_percentage": round(early_stop_counter / early_stop_patience, 2) if early_stop_patience else 0
        } if early_stopping else {}

        # Create metrics
        epoch_time = time.time() - epoch_start
        learning_rate = lr_scheduler.get_last_lr()[0] if lr_scheduler else None
        metrics = {
            "train/loss": train_loss,
            "train/accuracy": train_accuracy,
            "train/epoch_time": epoch_time,
            "validation/best_accuracy": best_validation_accuracy, # TODO: redundant?
            "validation/best_epoch": best_epoch, # TODO: redundant?
            **train_loader_metrics,
            **validation_metrics,
            **early_stopping_metrics
        }
        if learning_rate: metrics["train/learning_rate"] = learning_rate

        # In case this is the best model then save the metrics
        # to a separate file (eg: restore pre-processing pipeline; review metrics, etc.)
        if is_best_model:
            best_model_meta_path = best_model_path.replace(".jit", ".json")
            with open(best_model_meta_path, "w", encoding="utf-8") as file:
                file.write(json.dumps({
                    "config": config,
                    "metrics": metrics
                }, cls=CustomJSONEncoder, indent=4))

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
                "validation/best_epoch" : best_epoch,
                "validation/best_accuracy" : best_validation_accuracy
            }, indent=4))

        # In case early stopping was triggered then stop training
        if early_stop_triggered:
            logging.info(f"Early stopping triggered at epoch {epoch}")
            break
        

    # Return training results
    return {
        "train/last_epoch": epoch,
        "validation/best_accuracy": best_validation_accuracy,
        "validation/best_epoch": best_epoch,
        "validation/best_model": best_model
    }


def train(config, model_output_dir, n_models=1, wandb_enabled=False): 
    """
    Train the model for the specified number of epochs.
    Logs the training progress and results to W&B.
    """

    # Perform training within the context of a W&B run
    config_id = config["id"]
    seed = config["seed"]
    n_epochs = config["n_epochs"]
    dataset_config = config["dataset"]
    dataset_id = dataset_config["id"]
    train_bootstrap_percentage = dataset_config["train"]["bootstrap_percentage"]

    # Set the random seed for reproducibility
    set_seed(seed)

    # Create the model output dir in case it doesn't exist yet
    if not os.path.exists(model_output_dir): os.makedirs(model_output_dir)

    # Create the data loaders
    data_loader_config = config['data_loader']
    create_data_loaders_kwargs = {
        **data_loader_config, 
        "dataset_id": data_loader_config.get("dataset", dataset_id),
        "train_bootstrap_percentage": train_bootstrap_percentage
    }
    data_loaders = create_data_loaders(config, **create_data_loaders_kwargs)

    # Train the number of specified models
    # (eg: >1 for training ensembles)
    results = {}
    datetime_s = time.strftime('%Y%m%dT%H%M%S')
    for model_idx in range(n_models):
        if n_models > 1: model_run_id = f"{config_id}__{datetime_s}__{model_idx}"
        else: model_run_id = f"{config_id}__{datetime_s}"
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

def evaluate(config, model, loader_type, wandb_enabled=False):
    """
    Evaluates the model on the specified test set.
    Logs the evaluation results to W&B.
    """

    seed = config["seed"]
    dataset_id = config["dataset"]["id"]
    data_loader_config = config["data_loader"]

    # Set the random seed for reproducibility
    set_seed(seed)

    # Create the data loaders
    create_data_loaders_kwargs = {
        **data_loader_config, 
        "dataset_id": data_loader_config.get("dataset", dataset_id)
    }
    data_loaders = create_data_loaders(config, **create_data_loaders_kwargs)

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
    if isinstance(model, str): model = load_model(model, DEVICE)
 
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
        for images, labels in loader:
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

    loader_metrics = loader.get_stats()
    loader_metrics = {f"{loader_type}/loader/{k}": v for k, v in loader_metrics.items()}
    
    # Concatenate all labels and predictions
    all_labels = torch.cat(all_labels).cpu()
    all_predictions = torch.cat(all_predictions).cpu()

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
        **loader_metrics,
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

def sweep(seeds=[42]):        
    """
    Perform a hyperparameter sweep using W&B.
    """

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


def main():
    """
    Main function to parse command line arguments and run the script.
    Runs by default when script is not being executed by a W&B sweep agent.
    """

    # Parse command line arguments
    parser = argparse.ArgumentParser(description='Train, evaluate, or tune a CNN on MNIST dataset.')
    parser.add_argument('mode', choices=['train', 'eval', 'sweep'], help='Mode to run the script in')
    parser.add_argument("--dataset", type=str, default="mnist", help='Dataset to use for training and evaluation')
    parser.add_argument("--seed", type=int, default=42, help="Random seeds to use for training")
    parser.add_argument("--n_epochs", type=int, default=200, help='Number of epochs to train the model for')
    parser.add_argument("--n_models", type=int, default=1, help='How many models to train (eg: train an ensemble)')
    parser.add_argument("--train_bootstrap_percentage", type=float, default=0, help='Percentage of the dataset size for each bootstrap sample (0 < percentage <= 1)')
    parser.add_argument("--config_path", type=str, default="configs/train/MLP.yml", help='Path to the training config file') # TODO: also used in eval
    parser.add_argument("--model_path", type=str, help='Path to the model file for evaluation')
    parser.add_argument("--model_output_dir", type=str, default="outputs", help='Directory to save the model file')
    parser.add_argument("--wandb_enabled", type=bool, default=False, help='Whether to enable W&B logging')
    args = parser.parse_args()

    # Load the configuration
    config = load_config(args.config_path, {
        "seed": args.seed,
        "n_epochs": args.n_epochs,
        "dataset": {
            "id": args.dataset,
            "train": {
                "bootstrap_percentage" : args.train_bootstrap_percentage
            }
        }
    })
    
    # Train the model
    if args.mode == 'train': train(config, args.model_output_dir, n_models=args.n_models, wandb_enabled=args.wandb_enabled)
    elif args.mode == 'eval': evaluate(config, args.model_path, "test", wandb_enabled=args.wandb_enabled)

if __name__ == '__main__':
    # In case this is being run by a wandb 
    # sweep agent then call the sweep code
    if "--sweep=1" in sys.argv: 
        sweep()
    # Otherwise assume normal run
    else: 
        main()
