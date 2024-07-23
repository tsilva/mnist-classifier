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
import sys
import time

import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix, f1_score, precision_score, recall_score
import torch
import torch.nn as nn
from tqdm import tqdm
import wandb
import yaml
import torch.cuda.amp as amp

from libs.misc import CustomJSONEncoder, set_global_seed, get_device, config_merger
from libs.models import build_model, load_model
from libs.data_loaders import create_data_loaders
from libs.loss_functions import build_loss_function
from libs.optimizers import build_optimizer
from libs.lr_schedulers import build_lr_scheduler
from libs.early_stopping import build_early_stopping
from libs.wandb_utils import parse_wandb_sweep_config, OptionalWandbContext

# Setup logger
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(message)s')
logger = logging.getLogger(__name__)

PROJECT_NAME = 'mnist-classifier' # TODO: remove this
CONFIG_ROOT = "configs/train/" # TODO: softcode this
OUTPUTS_DIR = "outputs" # TODO: softcode this

# Retrieve the device to run the models on
# (use HW acceleration if available, otherwise use CPU)
DEVICE = get_device()

# Make sure W&B is terminated even if the script crashes
def cleanup():
    if wandb.run: wandb.finish()
atexit.register(cleanup)

# TODO: should this be called load_train_config?
def load_config(config_path, extra_config={}):
    """
    Creates the configuration data structure by merging configuration data from multiple sources.
    The configuration data holds all parameters that will have an impact on the training process
    (eg: model output dir is not part of the config because it doesn't affect the training process).
    """

    assert config_path.startswith(CONFIG_ROOT), f"config path must start with {CONFIG_ROOT}"
    assert not config_path.endswith("_default.yml"), "config path must not end with _default.yml"

    config_id = config_path.replace(CONFIG_ROOT, "").replace(".yml", "")
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
            if path.endswith(CONFIG_ROOT):
                break
            path = os.path.dirname(path)
        return default_configs

    # Merge all default configurations
    config_dir = os.path.dirname(config_path)
    default_configs = list(reversed(_collect_default_configs(config_dir)))
    
    merged_default_config = {}
    for default_config in default_configs:
        merged_default_config = config_merger.merge(merged_default_config, default_config)

    # Load the specified configuration
    with open(config_path, "r", encoding="utf-8") as file:
        file_config = yaml.safe_load(file)

    # Merge the configurations
    config = config_merger.merge(merged_default_config, file_config)
    config = config_merger.merge(config, extra_config)
    
    return config

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
        validation_loader = data_loaders["validation"]
        validation_metrics = _evaluate(
            model, validation_loader, "validation",
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
            **validation_metrics
        }
        if learning_rate: metrics["train/learning_rate"] = learning_rate # TODO: set learning rate even if scheduler is not present

        # In case early stopping is enabled then 
        # check if we should stop training
        early_stop_triggered, early_stop_best_score, early_stop_counter, early_stop_patience = early_stopping(metrics) if early_stopping else (False, None, None, None)
        early_stopping_metrics = {
            "early_stopping/best_score": early_stop_best_score, 
            "early_stopping/counter": early_stop_counter, 
            "early_stopping/patience": early_stop_patience,
            "early_stopping/patience_percentage": round(early_stop_counter / early_stop_patience, 2) if early_stop_patience else 0
        } if early_stopping else {}
        metrics = {**metrics, **early_stopping_metrics}

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
            _metrics = {k: v for k, v in metrics.items() if not k in ["validation/confusion_matrix", "validation/misclassified"]}
            logging.info(json.dumps(_metrics, indent=4))

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


def train(config, n_models=1, wandb_enabled=False): 
    """
    Train the model for the specified number of epochs.
    Logs the training progress and results to W&B.
    """

    # Perform training within the context of a W&B run
    config_id = config["id"]
    n_epochs = config["n_epochs"]
    dataset_id = config["dataset"]["id"]

    # Create the model output dir in case it doesn't exist yet
    if not os.path.exists(OUTPUTS_DIR): os.makedirs(OUTPUTS_DIR)

    # Create the data loaders
    data_loader_config = config['data_loader']
    data_loaders = create_data_loaders(dataset_id, DEVICE, **data_loader_config)

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
            best_model_path = f"{OUTPUTS_DIR}/{model_run_id}.jit"
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
            test_loader = data_loaders["test"]
            test_metrics = _evaluate(
                best_model, test_loader, "test", 
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

def evaluate(model_path, wandb_enabled=False):
    """
    Evaluates the model on the specified test set.
    Logs the evaluation results to W&B.
    """

    # Load the model
    model, model_meta = load_model(model_path, DEVICE)
    dataset_id = model_meta["config"]["dataset"]["id"]

    data_loaders = create_data_loaders(dataset_id, DEVICE)
    test_loader = data_loaders["test"]

    date_s = time.strftime('%Y%m%dT%H%M%S')
    run_id = f"evaluate__{date_s}"
    with OptionalWandbContext(wandb_enabled, project=PROJECT_NAME, id=run_id) as run: 
        metrics = _evaluate(model, test_loader, "test")
        if wandb_enabled: run.log(metrics)
        else: logging.info(json.dumps(metrics, indent=4))
        return metrics

def _evaluate(
    model, 
    loader, 
    loader_type, 
    use_mixed_precision=False,
    log_confusion_matrix=False, 
    log_misclassifications=False
):
    # Set the model in evaluation mode
    model.eval()

    # Define the loss function
    loss_function = nn.CrossEntropyLoss()

    correct = total = running_loss = 0
    all_labels = []
    all_predictions = []
    misclassifications = []

    # Evaluate the model
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

def sweep():        
    """
    Perform a hyperparameter sweep using W&B.
    """

    # Perform training within the context of the sweep
    with wandb.init():
        # Merge the sweep config with the 
        # training run config referenced within it
        sweep_config = parse_wandb_sweep_config(wandb.config)
        base_config_path = sweep_config["config"]
        base_config_path = f"{CONFIG_ROOT}/{base_config_path}"
        base_config = load_config(base_config_path)
        config = config_merger.merge(base_config, sweep_config)

        # Load config params
        config_id = config["id"]
        seed = config["seed"]
        n_epochs = config["n_epochs"]
        dataset_id = config["dataset"]["id"]
        data_loader_config = config["data_loader"]
        
        # Set the global seed for reproducibility
        # (sweep config should specify multiple 
        # seeds to average out randomness)
        set_global_seed(seed)

        # Run the training session
        datetime_s = time.strftime('%Y%m%dT%H%M%S')
        model_run_id = f"sweep__{config_id}__{datetime_s}__{seed}"
        best_model_path = f"{OUTPUTS_DIR}/{model_run_id}.jit"
        data_loaders = create_data_loaders(dataset_id, DEVICE, **data_loader_config)
        results = _train(config, data_loaders, n_epochs, best_model_path, wandb_run=wandb.run)
        
        # Set the score as the best validation accuracy
        # (we could make the score a function of multiple metrics)
        best_validation_accuracy = results["validation/best_accuracy"]
        score = best_validation_accuracy

        # Log the score to be maximized by the sweep
        wandb.log({"score": score})


def main():
    """
    Main function to parse command line arguments and run the script.
    Runs by default when script is not being executed by a W&B sweep agent.
    """

    # TODO: reorder these, re-evaluate which ones are mandatory
    # Parse command line arguments
    parser = argparse.ArgumentParser(description='Train, evaluate, or tune a CNN on MNIST dataset.')
    parser.add_argument('mode', choices=['train', 'eval', 'sweep'], help='Mode to run the script in')
    parser.add_argument("--config_path", type=str, default="configs/train/mnist/MLP.yml", help='Path to the training config file') # TODO: also used in eval
    parser.add_argument("--dataset", type=str, default="mnist", help='Dataset to use for training and evaluation')
    parser.add_argument("--seed", type=int, default=42, help="Random seeds to use for training")
    parser.add_argument("--n_epochs", type=int, default=200, help='Number of epochs to train the model for')
    parser.add_argument("--n_models", type=int, default=1, help='How many models to train (eg: train an ensemble)')
    parser.add_argument("--train_bootstrap_percentage", type=float, default=0, help='Percentage of the dataset size for each bootstrap sample (0 < percentage <= 1)')
    parser.add_argument("--model_path", type=str, help='Path to the model file for evaluation')
    parser.add_argument("--wandb_enabled", type=bool, default=False, help='Whether to enable W&B logging')
    args = parser.parse_args()

    # Set the random seed for reproducibility
    set_global_seed(args.seed)

    # Train the model
    if args.mode == 'train': 
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
        train(config, n_models=args.n_models, wandb_enabled=args.wandb_enabled)
    elif args.mode == 'eval': 
        evaluate(args.model_path, wandb_enabled=args.wandb_enabled)

if __name__ == '__main__':
    # In case this is being run by a wandb 
    # sweep agent then call the sweep code
    if "--sweep=1" in sys.argv: 
        sweep()
    # Otherwise assume normal run
    else: 
        main()
