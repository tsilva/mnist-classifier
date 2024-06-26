import os
import time
import random
import argparse
import torch
import multiprocessing
import logging
import json
import torch.nn as nn
import torch.optim as optim
import torchvision
import torchvision.transforms as transforms
from torch.utils.data import DataLoader, random_split
import numpy as np
import optuna
import wandb
import atexit
from sklearn.metrics import confusion_matrix
import seaborn as sns
import matplotlib.pyplot as plt
import torch.nn.functional as F


PROJECT_NAME = 'mnist-classifier'
OPTUNA_STORAGE_URI = "mysql://root@localhost/optuna"
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Setup logger
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(message)s')
logger = logging.getLogger(__name__)

# Indicate the device being used
logger.info(f"Using device: {DEVICE}")

# Ensure wandb is terminated properly
# even if the script is interrupted
def cleanup_wandb():
    if wandb.run: wandb.finish()
atexit.register(cleanup_wandb)

def set_seed(seed):
    random.seed(seed)  
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

class LeNet5(nn.Module):
    def __init__(self):
        super(LeNet5, self).__init__()

        # First convolutional layer
        # Input shape: (batch_size, 1, 28, 28)
        # Kernel size: 5x5, Stride: 1, Padding: 2
        # Output shape: (batch_size, 6, 28, 28)
        self.conv1 = nn.Conv2d(1, 6, kernel_size=5, stride=1, padding=2)

        # Average pooling layer
        # Input shape: (batch_size, 6, 28, 28)
        # Kernel size: 2x2, Stride: 2
        # Output shape: (batch_size, 6, 14, 14)
        self.pool1 = nn.AvgPool2d(kernel_size=2, stride=2)

        # Second convolutional layer
        # Input shape: (batch_size, 6, 14, 14)
        # Kernel size: 5x5, Stride: 1
        # Output shape: (batch_size, 16, 10, 10)
        self.conv2 = nn.Conv2d(6, 16, kernel_size=5, stride=1)

        # Average pooling layer
        # Input shape: (batch_size, 16, 10, 10)
        # Kernel size: 2x2, Stride: 2
        # Output shape: (batch_size, 16, 5, 5)
        self.pool2 = nn.AvgPool2d(kernel_size=2, stride=2)

        # Fully connected layers
        # Input shape: (batch_size, 400)
        # Output shape: (batch_size, 120)
        self.fc1 = nn.Linear(16*5*5, 120)

        # Input shape: (batch_size, 120)
        # Output shape: (batch_size, 84)
        self.fc2 = nn.Linear(120, 84)

        # Input shape: (batch_size, 84)
        # Output shape: (batch_size, 10)
        self.fc3 = nn.Linear(84, 10)

    def forward(self, x):
        # First convolutional layer
        # Input shape: (batch_size, 1, 28, 28)
        x = self.conv1(x) # Shape: (batch_size, 6, 28, 28)
        x = F.relu(x) # ReLU activation
        x = self.pool1(x) # Shape: (batch_size, 6, 14, 14)

        # Second convolutional layer
        # Input shape: (batch_size, 6, 14, 14)
        x = self.conv2(x) # Shape: (batch_size, 16, 10, 10)
        x = F.relu(x) # ReLU activation
        x = self.pool2(x) # Shape: (batch_size, 16, 5, 5)
        
        # Flatten the output for the fully connected layers
        # Input shape: (batch_size, 16, 5, 5)
        x = x.view(-1, 16*5*5) # Shape: (batch_size, 400)
        
        # First fully connected layer
        # Input shape: (batch_size, 400)
        x = self.fc1(x) # Shape: (batch_size, 120)
        x = F.relu(x) # ReLU activation
        
        # Second fully connected layer
        # Input shape: (batch_size, 120)
        x = self.fc2(x) # Shape: (batch_size, 84)
        x = F.relu(x) # ReLU activation
        
        # Output layer
        # Input shape: (batch_size, 84)
        x = self.fc3(x) # Shape: (batch_size, 10)
        
        # Return the output
        return x # Shape: (batch_size, 10)
    
class VGG11(nn.Module):
    def __init__(self, num_classes=10):
        super(VGG11, self).__init__()
        self.features = nn.Sequential(
            nn.Conv2d(1, 64, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),
            nn.Conv2d(64, 128, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),
            nn.Conv2d(128, 256, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(256, 256, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),
            nn.Conv2d(256, 512, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(512, 512, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),
            nn.Conv2d(512, 512, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(512, 512, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),
        )
        self.classifier = nn.Sequential(
            nn.Linear(512 * 1 * 1, 4096),
            nn.ReLU(inplace=True),
            nn.Dropout(),
            nn.Linear(4096, 4096),
            nn.ReLU(inplace=True),
            nn.Dropout(),
            nn.Linear(4096, num_classes),
        )

    def forward(self, x):
        x = self.features(x)
        x = x.view(x.size(0), -1)
        x = self.classifier(x)
        return x

class AlexNet(nn.Module):
    def __init__(self, num_classes=10):
        super(AlexNet, self).__init__()
        self.features = nn.Sequential(
            nn.Conv2d(1, 64, kernel_size=11, stride=4, padding=2),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=3, stride=2),
            nn.Conv2d(64, 192, kernel_size=5, padding=2),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=3, stride=2),
            nn.Conv2d(192, 384, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(384, 256, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(256, 256, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=3, stride=2),
        )
        self.classifier = nn.Sequential(
            nn.Dropout(),
            nn.Linear(256 * 1 * 1, 4096),  # Adjusted for the 28x28 input size
            nn.ReLU(inplace=True),
            nn.Dropout(),
            nn.Linear(4096, 4096),
            nn.ReLU(inplace=True),
            nn.Linear(4096, num_classes),
        )

    def forward(self, x):
        x = self.features(x)
        x = x.view(x.size(0), -1)
        x = self.classifier(x)
        return x

class CNN(nn.Module):
    def __init__(self, conv1_filters=32, conv2_filters=64, fc1_neurons=1000, fc2_neurons=10):
        super(CNN, self).__init__()
        
        # First convolutional layer
        # Input shape: (batch_size, 1, 28, 28)
        # Kernel size: 3x3, Stride: 1, Padding: 1
        # Output shape: (batch_size, conv1_filters, 28, 28)
        self.conv1 = nn.Conv2d(1, conv1_filters, kernel_size=3, padding=1)
        
        # Second convolutional layer
        # Input shape: (batch_size, conv1_filters, 14, 14)
        # Kernel size: 3x3, Stride: 1, Padding: 0 (default)
        # Output shape: (batch_size, conv2_filters, 12, 12)
        self.conv2 = nn.Conv2d(conv1_filters, conv2_filters, kernel_size=3)
        
        # Fully connected layers
        self.fc1 = nn.Linear(conv2_filters * 6 * 6, fc1_neurons)
        self.fc2 = nn.Linear(fc1_neurons, fc2_neurons)

    def forward(self, x):
        # First convolutional layer
        # Input shape: (batch_size, 1, 28, 28)
        x = self.conv1(x) # Shape: (batch_size, conv1_filters, 28, 28)
        x = F.relu(x) # ReLU activation
        x = F.max_pool2d(x, kernel_size=2, stride=2) # Shape: (batch_size, conv1_filters, 14, 14)

        # Second convolutional layer       
        x = self.conv2(x) # Shape: (batch_size, conv2_filters, 12, 12)
        x = F.relu(x) # ReLU activation
        x = F.max_pool2d(x, kernel_size=2, stride=2) # Shape: (batch_size, conv2_filters, 6, 6)
        
        # Flatten the output for the fully connected layers
        x = x.view(x.size(0), -1) # Shape: (batch_size, conv2_filters * 6 * 6)
        
        # First fully connected layer
        x = self.fc1(x) # Shape: (batch_size, fc1_neurons)
        x = F.relu(x) # ReLU activation

        # Output layer
        # Input shape: (batch_size, fc1_neurons)
        x = self.fc2(x) # Shape: (batch_size, fc2_neurons)
        
        # Return the output
        return x # Shape: (batch_size, fc2_neurons)

def build_model(model_config):
    model_id = model_config['id']
    model_params = model_config.get('params', {})
    model = {
        "CNN": CNN,
        "LeNet5": LeNet5,
        "VGG11": VGG11,
        "AlexNet": AlexNet
    }[model_id](**model_params)
    model = model.to(DEVICE)
    return model

def build_optimizer(model, optimizer_config):
    optimizer_id = optimizer_config['id']
    optimizer_params = optimizer_config.get('params', {})
    optimizer = {
        "Adam": optim.Adam,
        "SGD": optim.SGD
    }[optimizer_id](model.parameters(), **optimizer_params)
    return optimizer

def build_learning_rate_scheduler(optimizer, scheduler_config):
    scheduler_id = scheduler_config['id']
    scheduler_params = scheduler_config.get('params', {})
    scheduler = {
        "StepLR": optim.lr_scheduler.StepLR
    }[scheduler_id](optimizer, **scheduler_params)
    return scheduler

def build_loss_function(loss_function_config):
    loss_function_id = loss_function_config['id']
    loss_function_params = loss_function_config.get('params', {})
    loss_function = {
        "CrossEntropyLoss": nn.CrossEntropyLoss
    }[loss_function_id](**loss_function_params)
    return loss_function

def train(config, seed, data_loaders, num_epochs):
    # Retrieve hyperparameters from the config
    model_output_dir = config['model_output_dir']
    image_logging_interval = config['image_logging_interval']
    model_config = config['model']
    model_id = model_config['id']
    optimizer_config = config["optimizer"]
    learning_rate_scheduler_config = config["learning_rate_scheduler"]
    loss_function_config = config["loss_function"]

    # Unpack data loaders
    train_loader = data_loaders['train']
    validation_loader = data_loaders['validation']

    # Set the random seed for training reproducibility
    set_seed(seed)
    
    # Build model, optimizer, learning rate scheduler, and loss function
    model = build_model(model_config)
    optimizer = build_optimizer(model, optimizer_config)
    learning_rate_scheduler = build_learning_rate_scheduler(optimizer, learning_rate_scheduler_config)
    loss_function = build_loss_function(loss_function_config)

    # Initialize W&B
    date_s = time.strftime('%Y%m%dT%H%M%S')
    run_id = f"{date_s}-{model_id}"
    run = wandb.init(project=PROJECT_NAME, id=run_id, config=config) # Start a new W&B run
    wandb.watch(model) # Log the model architecture

    # Ensure model file path directory exists
    os.makedirs(model_output_dir, exist_ok=True)
    model_output_path = f"{model_output_dir}/best_model_{run.id}.pth"

    # Train for X epochs
    best_validation_accuracy = 0.0
    for epoch in range(num_epochs):
        logging.info(f"Epoch {epoch + 1}/{num_epochs}")

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

        # Let the scheduler update the 
        # learning rate now that the epoch is done
        learning_rate_scheduler.step()

        # Calculate train loss and accuracy
        train_loss = running_loss / len(train_loader.dataset)
        train_accuracy = 100 * correct / total

        # Evaluate the model on the test set
        validation_accuracy, validation_loss, validation_labels, validation_predictions, _ = evaluate(model, validation_loader)

        # Log metrics to console
        metrics = {
            "epoch": epoch,
            "train_loss": train_loss,
            "train_accuracy": train_accuracy,
            "validation_loss": validation_loss,
            "validation_accuracy": validation_accuracy,
            "best_validation_accuracy": best_validation_accuracy,
            "learning_rate": learning_rate_scheduler.get_last_lr()[0]
        }
        logging.info(json.dumps(metrics, indent=4))

        # Log metrics to W&B (add confusion matrix every X epochs)
        wandb_metrics = {**metrics}
        if epoch % image_logging_interval == 0: 
            cm = confusion_matrix(validation_labels, validation_predictions)
            plt.figure(figsize=(10, 10))
            sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=np.arange(10), yticklabels=np.arange(10))
            plt.xlabel('Predicted')
            plt.ylabel('True')
            confusion_matrix_image = wandb.Image(plt)
            wandb_metrics["confusion_matrix"] = confusion_matrix_image
            plt.close()
        wandb.log(wandb_metrics)

        # If the model is the best so far, save it to disk
        if validation_accuracy > best_validation_accuracy:
            best_validation_accuracy = validation_accuracy
            torch.save(model, model_output_path)
            logging.info(f'Saved best model with accuracy: {best_validation_accuracy:.2f}%')

    # Upload best model to W&B
    logging.info(f"Uploading model to W&B: {model_output_path}")
    wandb.save(model_output_path)
    artifact_name = model_output_path.split("/")[-1]
    artifact = wandb.Artifact(artifact_name, type='model')
    artifact.add_file(model_output_path)
    run.log_artifact(artifact)

    # Mark the run as finished
    run.finish()

    # Return the best validation accuracy
    return best_validation_accuracy


def evaluate(model, test_loader):
    # Load the model if it's a path
    if isinstance(model, str):
        model_path = model
        if model_path.startswith('https://wandb.ai/'):
            wandb.init(project=PROJECT_NAME)
            try:
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
                model = torch.load(model_path).to(DEVICE)
            finally:
                wandb.finish()
        else:
            model = torch.load(model_path).to(DEVICE)
        
    # Set the model in evaluation mode
    model.eval()

    # Define the loss function
    # TODO: use data from w&b
    criterion = nn.CrossEntropyLoss()

    correct = total = running_loss = 0
    all_data = []
    misclassifications = []

    # Evaluate the model
    with torch.no_grad():  # Disable gradient tracking (no backpropagation needed for evaluation)
        for images, labels in test_loader:
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

    # Calculate accuracy
    accuracy = 100 * correct / total

    # Calculate the average loss
    average_loss = running_loss / total

    # Extract all labels and predictions
    all_labels, all_predictions = zip(*[(lbl.item(), pred.item()) for _, lbl, pred in all_data])

    return accuracy, average_loss, all_labels, all_predictions, misclassifications


def tune(config, study_name, seeds, n_trials, trial_prune_interval=None):
    # Define the objective function for the Optuna study
    _config = config
    def _objective(trial):
        config = {
            **_config,
            "learning_rate" : trial.suggest_loguniform('lr', 1e-5, 1e-1),
            "conv1_filters" : trial.suggest_int('conv1_filters', 16, 64),
            "conv2_filters" : trial.suggest_int('conv2_filters', 32, 128),
            "fc1_neurons" : trial.suggest_int('fc1_neurons', 500, 2000),
            "fc2_neurons" : trial.suggest_int('fc2_neurons', 10, 100)
        }

        # Train with multiple seeds to average out the randomness
        accuracies = []
        for seed in seeds:
            accuracy = train(config, seed, trial=trial, trial_prune_interval=trial_prune_interval)
            accuracies.append(accuracy)
        return np.mean(accuracies)
    
    # Run the optuna study
    study_name = f"study={study_name}"
    seed = seeds[0]
    sampler = optuna.samplers.TPESampler(seed=seed)
    pruner = optuna.pruners.HyperbandPruner()
    study = optuna.create_study(
        direction="maximize",
        study_name=study_name,
        storage=OPTUNA_STORAGE_URI,
        sampler=sampler,
        pruner=pruner,
        load_if_exists=True
    )
    study.optimize(_objective, n_trials=n_trials)

    # Return the study result
    study_results = {
        "num_trials": len(study.trials),
        "num_pruned_trials": len([t for t in study.trials if t.state == optuna.trial.TrialState.PRUNED]),
        "best_value": study.best_trial.value,
        "best_params": study.best_trial.params,
    }
    logger.info(json.dumps(study_results, indent=4))
    return study_results


def _main():
    # Parse command line arguments
    parser = argparse.ArgumentParser(description='Train, evaluate, or tune a CNN on MNIST dataset.')
    parser.add_argument('mode', choices=['train', 'eval', 'tune'], help='Mode to run the script in')
    parser.add_argument("--validation_split", type=float, default=0.2, help="Validation split ratio")
    parser.add_argument('--model_path', type=str, default="outputs/best_model.pth", help='Path to the model file for evaluation')
    parser.add_argument('--n_epochs', type=int, default=10, help='Number of epochs to train the model for')
    parser.add_argument("--seeds", type=int, nargs="+", default=[123], help="Random seeds to use for training")
    parser.add_argument("--n_trials", type=int, default=100, help="Number of trials for tuning")
    parser.add_argument("--trial_prune_interval", type=int, default=1, help="Interval for pruning trials in Optuna")
    parser.add_argument("--trial_epoch_ratio", type=float, default=0.8, help="Ratio of epochs to run per trial in Optuna")
    args = parser.parse_args()

    # Set the random seed for reproducibility (we are going to perform data 
    # splitting and shuffling and want to do so deterministically, however 
    # we will set the seed again in the training set because during tuning 
    # we want to train with different seeds on the same dataset splits/shuffles)
    seed = args.seeds[0]
    set_seed(seed)

    # Define the configuration
    config = {
        "model_output_dir": "outputs", # Path to save the best model to
        'data_loader_batch_size': 64, # Batch size for the data loader
        'image_logging_interval': 5, # Log confusion matrix every 5 epochs
        "model" : {
            "id" : "CNN"
        },
        "optimizer" : {
            "id" : "Adam", # "Adam", "SGD"
            "params" : {
                "lr" : 0.001 # Learning rate
            }
        },
        "learning_rate_scheduler" : {
            "id" : "StepLR", # "StepLR"
            "params" : {
                "step_size" : 7, # Step size for the learning rate scheduler
                "gamma" : 0.1 # Multiplicative factor of learning rate decay
            }
        },
        "loss_function" : {
            "id" : "CrossEntropyLoss" # "CrossEntropyLoss"
        }
    }

    # Define transformations
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.5,), (0.5,))
    ])

    # Load datasets
    full_train_dataset = torchvision.datasets.MNIST(root='./data', train=True, download=True, transform=transform)
    test_dataset = torchvision.datasets.MNIST(root='./data', train=False, download=True, transform=transform)

    # Split the full training dataset into training and validation sets
    train_size = int((1 - args.validation_split) * len(full_train_dataset))
    validation_size = len(full_train_dataset) - train_size
    train_dataset, validation_dataset = random_split(full_train_dataset, [train_size, validation_size])

    # Setup the data loaders
    batch_size = config['data_loader_batch_size']
    num_workers = multiprocessing.cpu_count()
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=num_workers, pin_memory=True)
    validation_loader = DataLoader(validation_dataset, batch_size=batch_size, shuffle=False, num_workers=num_workers, pin_memory=True)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False, num_workers=num_workers, pin_memory=True)
    data_loaders = {'train': train_loader, 'validation': validation_loader, 'test': test_loader}

    def _eval(model_path, test_loader):
        accuracy, average_loss, all_labels, all_predictions, _ = evaluate(model_path, test_loader)
        cm = confusion_matrix(all_labels, all_predictions)
        logging.info(f"Test Set - Accuracy: {accuracy:.2f}%")
        logging.info(f"Test Set - Loss: {average_loss:.4f}")
        logging.info(f"Test Set - Confusion Matrix:\n{cm}")

    # Train the model
    if args.mode == 'train':
        seed = args.seeds[0]
        train(config, seed, data_loaders, num_epochs=args.n_epochs)
        _eval(args.model_path, test_loader)
    # Evaluate a model
    elif args.mode == 'eval':
        _eval(args.model_path, test_loader)
    # Perform hyperparameter tuning
    elif args.mode == 'tune':
        n_epochs = int(args.n_timesteps * args.trial_epoch_ratio)
        tune(config, args.study_name, args.seeds, args.n_trials, n_epochs, algo=args.algo, trial_prune_interval=args.trial_prune_interval)

def main():
    try: _main()
    finally: cleanup_wandb()
    
if __name__ == '__main__':
    main()