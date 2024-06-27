import sys
import yaml
from tqdm import tqdm
import time
import random
import argparse
import torch
import multiprocessing
import logging
import torch.nn as nn
import torch.optim as optim
import torchvision
import torchvision.transforms as transforms
from torch.utils.data import DataLoader, random_split
import numpy as np
import wandb
from sklearn.metrics import confusion_matrix
import seaborn as sns
import matplotlib.pyplot as plt
import torch.nn.functional as F

PROJECT_NAME = 'mnist-classifier'
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Setup logger
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(message)s')
logger = logging.getLogger(__name__)

# Indicate the device being used
logger.info(f"Using device: {DEVICE}")

def load_config():
    with open("hyperparams.yml", 'r') as file:
        hyperparams = yaml.safe_load(file)

    return {
        "logging" : {
            "image_interval": 5,
        },
        **hyperparams
    }

def set_seed(seed):
    random.seed(seed)  
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

class LeNet5(nn.Module):
    def __init__(self, conv1_filters=6, conv2_filters=16, fc1_neurons=120, fc2_neurons=84, fc3_neurons=10):
        super(LeNet5, self).__init__()

        # Store the hyperparameters
        self.conv1_filters = conv1_filters
        self.conv2_filters = conv2_filters
        self.fc1_neurons = fc1_neurons
        self.fc2_neurons = fc2_neurons
        self.fc3_neurons = fc3_neurons

        # First convolutional layer
        # Input shape: (batch_size, 1, 28, 28)
        # Kernel size: 5x5, Stride: 1, Padding: 2
        # Output shape: (batch_size, conv1_filters, 28, 28)
        self.conv1 = nn.Conv2d(1, conv1_filters, kernel_size=5, stride=1, padding=2)

        # Average pooling layer
        # Input shape: (batch_size, conv1_filters, 28, 28)
        # Kernel size: 2x2, Stride: 2
        # Output shape: (batch_size, conv1_filters, 14, 14)
        self.pool1 = nn.AvgPool2d(kernel_size=2, stride=2)

        # Second convolutional layer
        # Input shape: (batch_size, conv1_filters, 14, 14)
        # Kernel size: 5x5, Stride: 1
        # Output shape: (batch_size, conv2_filters, 10, 10)
        self.conv2 = nn.Conv2d(conv1_filters, conv2_filters, kernel_size=5, stride=1)

        # Average pooling layer
        # Input shape: (batch_size, conv2_filters, 10, 10)
        # Kernel size: 2x2, Stride: 2
        # Output shape: (batch_size, conv2_filters, 5, 5)
        self.pool2 = nn.AvgPool2d(kernel_size=2, stride=2)

        # Fully connected layers
        # Input shape: (batch_size, conv2_filters * 5 * 5)
        # Output shape: (batch_size, fc1_neurons)
        self.fc1 = nn.Linear(conv2_filters * 5 * 5, fc1_neurons)

        # Input shape: (batch_size, fc1_neurons)
        # Output shape: (batch_size, fc2_neurons)
        self.fc2 = nn.Linear(fc1_neurons, fc2_neurons)

        # Input shape: (batch_size, fc2_neurons)
        # Output shape: (batch_size, fc3_neurons)
        self.fc3 = nn.Linear(fc2_neurons, fc3_neurons)

    def forward(self, x):
        # First convolutional layer
        # Input shape: (batch_size, 1, 28, 28)
        x = self.conv1(x) # Shape: (batch_size, conv1_filters, 28, 28)
        x = F.relu(x) # ReLU activation
        x = self.pool1(x) # Shape: (batch_size, conv1_filters, 14, 14)

        # Second convolutional layer
        # Input shape: (batch_size, conv1_filters, 14, 14)
        x = self.conv2(x) # Shape: (batch_size, conv2_filters, 10, 10)
        x = F.relu(x) # ReLU activation
        x = self.pool2(x) # Shape: (batch_size, conv2_filters, 5, 5)
        
        # Flatten the output for the fully connected layers
        # Input shape: (batch_size, conv2_filters, 5, 5)
        x = x.view(-1, self.conv2_filters * 5 * 5) # Shape: (batch_size, conv2_filters * 5 * 5)
        
        # First fully connected layer
        # Input shape: (batch_size, conv2_filters * 5 * 5)
        x = self.fc1(x) # Shape: (batch_size, fc1_neurons)
        x = F.relu(x) # ReLU activation
        
        # Second fully connected layer
        # Input shape: (batch_size, fc1_neurons)
        x = self.fc2(x) # Shape: (batch_size, fc2_neurons)
        x = F.relu(x) # ReLU activation
        
        # Output layer
        # Input shape: (batch_size, fc2_neurons)
        x = self.fc3(x) # Shape: (batch_size, fc3_neurons)
        
        # Return the output
        return x # Shape: (batch_size, fc3_neurons)

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

def build_model(model_config, model_state=None):
    model_id = model_config['id']
    model_params = model_config.get('params', {})
    model = {
        "CNN": CNN,
        "LeNet5": LeNet5,
        "VGG11": VGG11,
        "AlexNet": AlexNet
    }[model_id](**model_params)
    if model_state: model.load_state_dict(model_state)
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

def build_lr_scheduler(optimizer, scheduler_config):
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

def create_data_loaders(batch_size=64, validation_split=0.2):
    # Define transformations
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.5,), (0.5,))
    ])

    # Load datasets
    full_train_dataset = torchvision.datasets.MNIST(root='./data', train=True, download=True, transform=transform)
    test_dataset = torchvision.datasets.MNIST(root='./data', train=False, download=True, transform=transform)

    # Split the full training dataset into training and validation sets
    train_size = int((1 - validation_split) * len(full_train_dataset))
    validation_size = len(full_train_dataset) - train_size
    train_dataset, validation_dataset = random_split(full_train_dataset, [train_size, validation_size])

    # Setup the data loaders
    num_workers = multiprocessing.cpu_count()
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=num_workers, pin_memory=True)
    validation_loader = DataLoader(validation_dataset, batch_size=batch_size, shuffle=False, num_workers=num_workers, pin_memory=True)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False, num_workers=num_workers, pin_memory=True)
    data_loaders = {'train': train_loader, 'validation': validation_loader, 'test': test_loader}

    return data_loaders

def _train(config, data_loaders, n_epochs):
    # Retrieve logging configuration
    logging_config = config['logging']
    image_logging_interval = logging_config['image_interval']

    # Retrieve hyperparameters from the config
    model_config = config['model']
    optimizer_config = config["optimizer"]
    lr_scheduler_config = config["lr_scheduler"]
    loss_function_config = config["loss_function"]

    # Unpack data loaders
    train_loader = data_loaders['train']
    validation_loader = data_loaders['validation']
    
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
    for epoch in tqdm(range(n_epochs), desc="Training model"):
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
        lr_scheduler.step()

        # Calculate train loss and accuracy
        train_loss = running_loss / len(train_loader.dataset)
        train_accuracy = 100 * correct / total

        # Evaluate the model on the test set
        validation_accuracy, validation_loss, validation_labels, validation_predictions, _ = evaluate(model, validation_loader)

        # Create metrics
        metrics = {
            "epoch": epoch,
            "train_loss": train_loss,
            "train_accuracy": train_accuracy,
            "validation_loss": validation_loss,
            "validation_accuracy": validation_accuracy,
            "best_validation_accuracy": best_validation_accuracy,
            "learning_rate": lr_scheduler.get_last_lr()[0]
        }

        # Add images to metrics every X epochs
        if epoch > 0 and epoch % image_logging_interval == 0: 
            cm = confusion_matrix(validation_labels, validation_predictions)
            plt.figure(figsize=(10, 10))
            sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=np.arange(10), yticklabels=np.arange(10))
            plt.xlabel('Predicted')
            plt.ylabel('True')
            confusion_matrix_image = wandb.Image(plt)
            metrics["confusion_matrix"] = confusion_matrix_image
            plt.close()

        # Log metrics to W&B
        wandb.log(metrics, step=epoch)

        # If the model is the best so far, save it to disk
        if validation_accuracy > best_validation_accuracy:
            best_validation_accuracy = validation_accuracy
            best_model_state = model.state_dict()
            logging.debug(f'Saved best model with accuracy: {best_validation_accuracy:.2f}%')

    # Return the best validation accuracy
    return best_validation_accuracy, best_model_state

def train(config, data_loaders, n_epochs, model_output_dir): 
    # Perform training within the context of a W&B run
    model_config = config['model']
    model_id = model_config['id']
    date_s = time.strftime('%Y%m%dT%H%M%S')
    run_id = f"train__{date_s}__{model_id}"
    with wandb.init(project=PROJECT_NAME, id=run_id, config=config) as run:
        # Perform training
        best_model_accuracy, best_model_state = _train(config, data_loaders, n_epochs)

        # Save the best model to disk
        best_model = build_model(model_config, model_state=best_model_state)
        best_model_path = f"{model_output_dir}/best_model_{run_id}.pth"
        torch.save(best_model, best_model_path)
        
        # Upload best model to W&B
        logging.info(f"Uploading model to W&B: {best_model_path}")
        wandb.save(best_model_path)
        artifact_name = best_model_path.split("/")[-1]
        artifact = wandb.Artifact(artifact_name, type='model')
        artifact.add_file(best_model_path)
        run.log_artifact(artifact)

        # Return training result
        return best_model_accuracy, best_model_path

def evaluate(model, test_loader):
    def _load_model(model_path):
        if model_path.startswith('https://wandb.ai/'):
            with wandb.init(project=PROJECT_NAME):
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
    
    # Load the model if a path was provided
    if isinstance(model, str): model = _load_model(model)
    
    # Ensure the model is on the correct device
    model = model.to(DEVICE)

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

def sweep(seeds=[42]):        
    def _parse_sweep_config(sweep_config):
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

    # Perform training within the context of the sweep
    with wandb.init():
        # Merge selected sweep params with config
        sweep_config = _parse_sweep_config(wandb.config)
        default_config = load_config()
        config = {**default_config, **sweep_config}

        # Perform training with multiple seeds to average 
        # out the randomness and get more robust results
        n_epochs = wandb.config.n_epochs
        best_validation_accuracies = []
        for seed in seeds:
            set_seed(seed)
            batch_size = config['data_loader']['batch_size']
            data_loaders = create_data_loaders(batch_size=batch_size)
            best_validation_accuracy, _ = _train(config, data_loaders, n_epochs)
            best_validation_accuracies.append(best_validation_accuracy)
        best_validation_accuracy = np.mean(best_validation_accuracies)
        
        # Set the score as the best validation accuracy
        # (we could make the score a function of multiple metrics)
        score = best_validation_accuracy

        # Log the score to be maximized by the sweep
        wandb.log({"score": score})

def main():
    # Parse command line arguments
    parser = argparse.ArgumentParser(description='Train, evaluate, or tune a CNN on MNIST dataset.')
    parser.add_argument('mode', choices=['train', 'eval', 'sweep'], help='Mode to run the script in')
    parser.add_argument("--seed", type=int, default=42, help="Random seeds to use for training")
    parser.add_argument('--n_epochs', type=int, default=50, help='Number of epochs to train the model for')
    parser.add_argument('--model_path', type=str, default="outputs/best_model.pth", help='Path to the model file for evaluation')
    parser.add_argument('--model_output_dir', type=str, default="outputs", help='Directory to save the model file')
    args = parser.parse_args()

    # Set the random seed for reproducibility
    set_seed(args.seed)

    # Create data loaders
    config = load_config()
    batch_size = config['data_loader']['batch_size']
    data_loaders = create_data_loaders(batch_size=batch_size)

    # Train the model
    if args.mode == 'train':
        _, best_model_path = train(config, data_loaders, args.n_epochs, args.model_output_dir)
        test_loader = data_loaders['test']
        accuracy, average_loss, all_labels, all_predictions, _ = evaluate(best_model_path, test_loader)
        cm = confusion_matrix(all_labels, all_predictions)
        logging.info(f"Test Set - Accuracy: {accuracy:.2f}%")
        logging.info(f"Test Set - Loss: {average_loss:.4f}")
        logging.info(f"Test Set - Confusion Matrix:\n{cm}")
    # Evaluate a model
    elif args.mode == 'eval':
        test_loader = data_loaders['test']
        accuracy, average_loss, all_labels, all_predictions, _ = evaluate(args.model_path, test_loader)
        cm = confusion_matrix(all_labels, all_predictions)
        logging.info(f"Test Set - Accuracy: {accuracy:.2f}%")
        logging.info(f"Test Set - Loss: {average_loss:.4f}")
        logging.info(f"Test Set - Confusion Matrix:\n{cm}")

if __name__ == '__main__':
    # In case this is being run by a wandb 
    # sweep agent then call the sweep code
    if "--sweep=1" in sys.argv: 
        sweep()
    # Otherwise assume normal run
    else: 
        main()
