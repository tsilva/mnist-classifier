import os
import random
import argparse
import torch
import logging
import json
import torch.nn as nn
import torch.optim as optim
import torchvision
import torchvision.transforms as transforms
from torch.utils.data import DataLoader
import numpy as np
import optuna
import wandb
from sklearn.metrics import confusion_matrix
import seaborn as sns
import matplotlib.pyplot as plt

PROJECT_NAME = 'mnist-classifier'
OPTUNA_STORAGE_URI = "mysql://root@localhost/optuna"
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Setup logger
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(message)s')
logger = logging.getLogger(__name__)

# Indicate the device being used
logger.info(f"Using device: {DEVICE}")

class CNN(nn.Module):
    def __init__(self, conv1_filters=32, conv2_filters=64, fc1_neurons=1000, fc2_neurons=10):
        super(CNN, self).__init__()
        
        # First convolutional layer
        # Input: 1 channel, Output: conv1_filters channels
        # Kernel size: 3x3, Stride: 1, Padding: 1
        # Input shape: (batch_size, 1, 28, 28)
        self.layer1 = nn.Sequential(
            nn.Conv2d(1, conv1_filters, kernel_size=3, padding=1), # Shape: (batch_size, conv1_filters, 28, 28)
            nn.ReLU(), # Shape: (batch_size, conv1_filters, 28, 28)
            nn.MaxPool2d(kernel_size=2, stride=2) # Shape: (batch_size, conv1_filters, 14, 14)
        )
        
        # Second convolutional layer
        # Input: conv1_filters channels, Output: conv2_filters channels
        # Kernel size: 3x3, Stride: 1, Padding: 0 (default)
        # Input shape: (batch_size, conv1_filters, 14, 14)
        self.layer2 = nn.Sequential(
            nn.Conv2d(conv1_filters, conv2_filters, kernel_size=3), # Shape: (batch_size, conv2_filters, 12, 12)
            nn.ReLU(), # Shape: (batch_size, conv2_filters, 12, 12)
            nn.MaxPool2d(2) # Shape: (batch_size, conv2_filters, 6, 6)
        )
        
        # Fully connected layers
        # Input: conv2_filters * 6 * 6, Output: fc1_neurons
        self.fc1 = nn.Linear(conv2_filters * 6 * 6, fc1_neurons) # Shape: (batch_size, fc1_neurons)
        self.fc2 = nn.Linear(fc1_neurons, fc2_neurons) # Shape: (batch_size, fc2_neurons)

    def forward(self, x):
        # Input shape: (batch_size, 1, 28, 28)
        x = self.layer1(x) # Shape: (batch_size, conv1_filters, 14, 14)
        x = self.layer2(x) # Shape: (batch_size, conv2_filters, 6, 6)
        x = x.view(x.size(0), -1) # Shape: (batch_size, conv2_filters * 6 * 6)
        x = self.fc1(x) # Shape: (batch_size, fc1_neurons)
        x = self.fc2(x) # Shape: (batch_size, fc2_neurons)
        return x # Shape: (batch_size, fc2_neurons)


def train(config, seed, data_loaders, num_epochs=100):
    # Retrieve hyperparameters from the config
    model_output_path = config['model_output_path']
    confusion_matrix_logging_interval = config['confusion_matrix_logging_interval']
    conv1_filters = config['conv1_filters']
    conv2_filters = config['conv2_filters']
    fc1_neurons = config['fc1_neurons']
    fc2_neurons = config['fc2_neurons']
    learning_rate = config['learning_rate']

    # Unpack data loaders
    train_loader = data_loaders['train']
    test_loader = data_loaders['test']
    
    # Ensure model file path directory exists
    if model_output_path:
        model_file_dir = os.path.dirname(model_output_path)
        if not os.path.exists(model_file_dir):
            os.makedirs(model_file_dir)

    # Set the random seed for reproducibility
    np.random.seed(seed)
    random.seed(seed)
    
    # Create the model and move it to the device (eg: GPU)
    model = CNN(
        conv1_filters=conv1_filters, 
        conv2_filters=conv2_filters, 
        fc1_neurons=fc1_neurons, 
        fc2_neurons=fc2_neurons
    ).to(DEVICE)

    # Log model architecture
    wandb.watch(model)

    # Initialize the optimizer and learning rate scheduler
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)

    # Initialize the learning rate scheduler
    scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=7, gamma=0.1) # TODO: softcode

    # Define the loss function
    criterion = nn.CrossEntropyLoss()

    # Train for X epochs
    best_accuracy = 0.0
    for epoch in range(num_epochs):
        # Set the model in training mode
        model.train()

        running_loss = 0.0
        correct = 0
        total = 0
        all_labels = []
        all_preds = []
        for images, labels in train_loader:
            images, labels = images.to(DEVICE), labels.to(DEVICE)
            
            # Perform forward pass
            predictions = model(images)

            # Compute loss
            loss = criterion(predictions, labels)

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
            all_preds.extend(predicted.cpu().numpy())

        # Let the scheduler update the 
        # learning rate now that the epoch is done
        scheduler.step()

        # Calculate train loss and accuracy
        train_loss = running_loss / len(train_loader.dataset)
        train_accuracy = 100 * correct / total

        validation_accuracy, validation_loss, val_labels, val_preds = evaluate(model, test_loader)
        
        # Log metrics to W&B
        metrics = {
            "epoch": epoch + 1,
            "train_loss": train_loss,
            "train_accuracy": train_accuracy,
            "validation_loss": validation_loss,
            "validation_accuracy": validation_accuracy,
            "best_accuracy": best_accuracy,
            "learning_rate": scheduler.get_last_lr()[0]
        }
        wandb.log(metrics)
        
        # Log metrics to console
        logging.info(json.dumps(metrics, indent=4))

        # Log confusion matrix periodically
        if (epoch + 1) % confusion_matrix_logging_interval == 0:
            cm = confusion_matrix(val_labels, val_preds)
            plt.figure(figsize=(10, 10))
            sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=np.arange(10), yticklabels=np.arange(10))
            plt.xlabel('Predicted')
            plt.ylabel('True')
            wandb.log({"confusion_matrix": wandb.Image(plt)})
            plt.close()

        # If the model is the best so far, save it
        if validation_accuracy > best_accuracy:
            best_accuracy = validation_accuracy
            torch.save(model, model_output_path)
            logging.info(f'Saved best model with accuracy: {best_accuracy:.2f}%')

    # Return the best accuracy
    return best_accuracy


def evaluate(model, test_loader):
    # Load the model if it's a path
    if isinstance(model, str):
        model = torch.load(model).to(DEVICE)

    # Set the model in evaluation mode
    model.eval()

    correct = 0
    total = 0
    running_loss = 0.0
    all_images = []
    all_labels = []
    all_preds = []
    misclassified_images = []
    misclassified_labels = []
    misclassified_preds = []
    
    # Define the loss function
    criterion = nn.CrossEntropyLoss()

    # Evaluate the model
    with torch.no_grad():
        for images, labels in test_loader:
            images, labels = images.to(DEVICE), labels.to(DEVICE)
            outputs = model(images)
            loss = criterion(outputs, labels)
            running_loss += loss.item() * images.size(0)
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()

            # Collect images and predictions for logging
            all_images.extend(images.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())
            all_preds.extend(predicted.cpu().numpy())

            # Identify misclassified images
            for i in range(len(labels)):
                if labels[i] != predicted[i]:
                    misclassified_images.append(images[i].cpu().numpy())
                    misclassified_labels.append(labels[i].cpu().numpy())
                    misclassified_preds.append(predicted[i].cpu().numpy())

    accuracy = 100 * correct / total
    avg_loss = running_loss / len(test_loader.dataset)

    # Log misclassified images, labels, and predictions to W&B
    wandb.log({
        "misclassified_images": [wandb.Image(img, caption=f"True: {true}, Pred: {pred}") for img, true, pred in zip(misclassified_images, misclassified_labels, misclassified_preds)]
    })

    return accuracy, avg_loss, all_labels, all_preds


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


def main():
    # Parse command line arguments
    parser = argparse.ArgumentParser(description='Train, evaluate, or tune a CNN on MNIST dataset.')
    parser.add_argument('mode', choices=['train', 'eval', 'tune'], help='Mode to run the script in')
    parser.add_argument('--model_path', type=str, help='Path to the model file for evaluation')
    parser.add_argument('--n_epochs', type=int, default=20, help='Number of epochs to train the model for')
    parser.add_argument("--seeds", type=int, nargs="+", default=[123], help="Random seeds to use for training")
    parser.add_argument("--n_trials", type=int, default=100, help="Number of trials for tuning")
    parser.add_argument("--trial_prune_interval", type=int, default=1, help="Interval for pruning trials in Optuna")
    parser.add_argument("--trial_epoch_ratio", type=float, default=0.8, help="Ratio of epochs to run per trial in Optuna")
    args = parser.parse_args()

    # Define the configuration
    config = {
        'model_output_path' : "outputs/best_model.pth", # Path to save the best model
        'data_loader_batch_size': 64, # Batch size for the data loader
        'data_loader_num_workers': 4, # Number of workers for the data loader
        'confusion_matrix_logging_interval': 5, # Log confusion matrix every 5 epochs   
        "conv1_filters": 32, # Number of filters in the first convolutional layer
        "conv2_filters": 64, # Number of filters in the second convolutional layer
        "fc1_neurons": 1000, # Number of neurons in the first fully connected layer
        "fc2_neurons": 10, # Number of neurons in the second fully connected layer
        "learning_rate": 0.001 # Learning rate for the optimizer
    }

    # Initialize W&B
    wandb.init(project=PROJECT_NAME)

    # Define transformations
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.5,), (0.5,))
    ])

    # Load datasets
    train_dataset = torchvision.datasets.MNIST(root='./data', train=True, download=True, transform=transform)
    test_dataset = torchvision.datasets.MNIST(root='./data', train=False, download=True, transform=transform)

    # Define data loaders
    batch_size = config['data_loader_batch_size']
    num_workers = config['data_loader_num_workers']
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=num_workers, pin_memory=True)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False, num_workers=num_workers, pin_memory=True)
    data_loaders = {'train': train_loader, 'test': test_loader}
    
    # Train the model
    if args.mode == 'train':
        seed = args.seeds[0]
        train(config, seed, data_loaders, num_epochs=args.n_epochs)
    # Evaluate a model
    elif args.mode == 'eval':
        assert args.model_path is not None, 'Please provide the model path for evaluation.'
        assert os.path.exists(args.model_path), 'Model path does not exist.'
        evaluate(args.model_path, test_loader)
    # Perform hyperparameter tuning
    elif args.mode == 'tune':
        n_epochs = int(args.n_timesteps * args.trial_epoch_ratio)
        tune(config, args.study_name, args.seeds, args.n_trials, n_epochs, algo=args.algo, trial_prune_interval=args.trial_prune_interval)

if __name__ == '__main__':
    main()