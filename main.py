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
from sklearn.metrics import confusion_matrix, precision_score, recall_score, f1_score
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

# Make sure the user is aware if HW acceleration is not available
if str(DEVICE) == "cpu":
    logging.warning("CUDA is not available. Running on CPU. Press Enter to continue...")
    input()

def load_config(hyperparams_path=None):
    hyperparams = {}

    if hyperparams_path:
        with open(hyperparams_path, 'r') as file:
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
    def __init__(self, conv1_filters=6, conv2_filters=16, conv3_filters=120, fc1_neurons=84, fc2_neurons=10):
        super(LeNet5, self).__init__()

        # Store the hyperparameters
        self.conv1_filters = conv1_filters
        self.conv2_filters = conv2_filters
        self.conv3_filters = conv3_filters
        self.fc1_neurons = fc1_neurons
        self.fc2_neurons = fc2_neurons

        # Encoder block
        self.encoder = nn.Sequential(
            nn.Conv2d(1, conv1_filters, kernel_size=5, stride=1, padding=2),  # Input shape: (batch_size, 1, 28, 28)
            nn.Tanh(),  # Tanh activation
            nn.AvgPool2d(kernel_size=2, stride=2),  # Output shape: (batch_size, conv1_filters, 14, 14)

            nn.Conv2d(conv1_filters, conv2_filters, kernel_size=5, stride=1),  # Input shape: (batch_size, conv1_filters, 14, 14)
            nn.Tanh(),  # Tanh activation
            nn.AvgPool2d(kernel_size=2, stride=2)  # Output shape: (batch_size, conv2_filters, 5, 5)
        )

        # Third Convolutional Layer to match LeNet-5
        self.conv3 = nn.Conv2d(conv2_filters, conv3_filters, kernel_size=5, stride=1)  # Input shape: (batch_size, conv2_filters, 5, 5)
        
        # Classifier block
        self.classifier = nn.Sequential(
            nn.Tanh(),  # Tanh activation after conv3

            nn.Linear(conv3_filters, fc1_neurons),  # Input shape: (batch_size, conv3_filters)
            nn.Tanh(),  # Tanh activation

            nn.Linear(fc1_neurons, fc2_neurons)  # Input shape: (batch_size, fc1_neurons)
        )

    def forward(self, x):
        # Pass through the encoder
        x = self.encoder(x)  # Shape: (batch_size, conv2_filters, 5, 5)
        
        # Pass through the third convolutional layer
        x = self.conv3(x)  # Shape: (batch_size, conv3_filters, 1, 1)
        
        # Flatten the output for the classifier
        x = x.view(-1, self.conv3_filters)  # Shape: (batch_size, conv3_filters)
        
        # Pass through the classifier
        x = self.classifier(x)  # Shape: (batch_size, fc2_neurons)
        
        # Return the output
        return x  # Shape: (batch_size, fc2_neurons)

class SimpleCNN(nn.Module):
    def __init__(self, conv1_filters=32, conv2_filters=64, fc1_neurons=1000, fc2_neurons=10):
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
            nn.Linear(conv2_filters * 6 * 6, fc1_neurons),
            nn.ReLU(),

            # Output layer
            nn.Linear(fc1_neurons, fc2_neurons)
        )

    def forward(self, x):
        # Encode input using convolutional layers
        x = self.encoder(x) 
        
        # Flatten the output for the fully connected layers
        x = x.view(x.size(0), -1) # Shape: (batch_size, conv2_filters * 6 * 6)
        
        # Classify encoded input using fully connected layers
        x = self.classifier(x)
        
        # Return the output
        return x # Shape: (batch_size, fc2_neurons)

class AdvancedCNN(nn.Module):
    def __init__(self):
        super(AdvancedCNN, self).__init__()
        
        # Layer 1
        self.conv1 = nn.Conv2d(in_channels=1, out_channels=32, kernel_size=5, stride=1, padding=2)
        self.conv1_bn = nn.BatchNorm2d(32)
        
        # Layer 2
        self.conv2 = nn.Conv2d(in_channels=32, out_channels=32, kernel_size=5, stride=1, padding=2, bias=False)
        self.conv2_bn = nn.BatchNorm2d(32)
        
        # Layer 3
        self.conv3 = nn.Conv2d(in_channels=32, out_channels=64, kernel_size=3, stride=1, padding=1)
        self.conv3_bn = nn.BatchNorm2d(64)
        
        # Layer 4
        self.conv4 = nn.Conv2d(in_channels=64, out_channels=64, kernel_size=3, stride=1, padding=1, bias=False)
        self.conv4_bn = nn.BatchNorm2d(64)
        
        # Dropout layer
        self.dropout = nn.Dropout(0.5)

        # Calculate the flattened size after the last pooling layer
        dummy_input = torch.randn(1, 1, 28, 28)
        dummy_output = self._forward_features(dummy_input)
        flattened_size = dummy_output.numel()

        # Fully connected layers
        self.fc1 = nn.Linear(flattened_size, 256, bias=False)
        self.fc1_bn = nn.BatchNorm1d(256)
        
        self.fc2 = nn.Linear(256, 128, bias=False)
        self.fc2_bn = nn.BatchNorm1d(128)
        
        self.fc3 = nn.Linear(128, 84, bias=False)
        self.fc3_bn = nn.BatchNorm1d(84)
        
        self.fc4 = nn.Linear(84, 10)
        
    def _forward_features(self, x):
        x = F.relu(self.conv1_bn(self.conv1(x)))
        x = self.conv2_bn(self.conv2(x))
        x = F.relu(x)
        x = F.max_pool2d(x, kernel_size=2, stride=2)
        x = self.dropout(x)
        
        x = F.relu(self.conv3_bn(self.conv3(x)))
        x = self.conv4_bn(self.conv4(x))
        x = F.relu(x)
        x = F.max_pool2d(x, kernel_size=2, stride=2)
        x = self.dropout(x)
        
        x = torch.flatten(x, 1)
        return x

    def forward(self, x):
        x = self._forward_features(x)
        x = F.relu(self.fc1_bn(self.fc1(x)))
        x = self.dropout(x)  # Dropout after first FC layer
        x = F.relu(self.fc2_bn(self.fc2(x)))
        x = self.dropout(x)  # Dropout after second FC layer
        x = F.relu(self.fc3_bn(self.fc3(x)))
        x = self.dropout(x)  # Dropout after third FC layer
        x = self.fc4(x)
        return F.log_softmax(x, dim=1)

def build_model(model_config, model_state=None):
    model_id = model_config['id']
    model_params = model_config.get('params', {})
    model = {
        "SimpleCNN": SimpleCNN,
        "LeNet5": LeNet5,
        "AdvancedCNN": AdvancedCNN
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
    if not scheduler_config: return None
    scheduler_id = scheduler_config['id']
    scheduler_params = scheduler_config.get('params', {})
    scheduler = {
        "StepLR": optim.lr_scheduler.StepLR,
        "ReduceLROnPlateau": optim.lr_scheduler.ReduceLROnPlateau
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
    # Define transformations for training data with augmentation
    train_transform = transforms.Compose([
        transforms.RandomRotation(15),
        transforms.RandomAffine(0, translate=(0.1, 0.1), scale=(0.9, 1.1), shear=10),
        transforms.ColorJitter(brightness=0.2, contrast=0.2),
        transforms.ToTensor(),
        transforms.Normalize((0.1307,), (0.3081,))
    ])

    # Define transformations for validation and test data (no augmentation)
    test_transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.5,), (0.5,))
    ])

    # Load datasets
    full_train_dataset = torchvision.datasets.MNIST(root='./data', train=True, download=True, transform=test_transform)
    test_dataset = torchvision.datasets.MNIST(root='./data', train=False, download=True, transform=test_transform)

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

        # Calculate train loss and accuracy
        train_loss = running_loss / len(train_loader.dataset)
        train_accuracy = 100 * correct / total

        # Evaluate the model on the test set
        validation_accuracy, validation_loss, _, _, _, _, _, _  = _evaluate(model, data_loaders, "validation", training_epoch=epoch)
        
        # Update the learning rate based on current validation loss
        if lr_scheduler: lr_scheduler.step(validation_loss)

        # Create metrics
        metrics = {
            "epoch": epoch,
            "train_loss": train_loss,
            "train_accuracy": train_accuracy,
            "best_validation_accuracy": best_validation_accuracy
        }
        if lr_scheduler: metrics["learning_rate"] = lr_scheduler.get_last_lr()[0]

        # Log metrics to W&B
        wandb.log(metrics, step=epoch)

        # If the model is the best so far, save it to disk
        if validation_accuracy > best_validation_accuracy:
            best_validation_accuracy = validation_accuracy
            best_model_state = model.state_dict()
            logging.debug(f'Saved best model with accuracy: {best_validation_accuracy:.2f}%')

    # Return the best validation accuracy
    last_epoch = epoch
    return best_validation_accuracy, best_model_state, last_epoch

def train(config, data_loaders, n_epochs, model_output_dir): 
    # Perform training within the context of a W&B run
    model_config = config['model']
    model_id = model_config['id']
    date_s = time.strftime('%Y%m%dT%H%M%S')
    run_id = f"train__{date_s}__{model_id}"
    with wandb.init(project=PROJECT_NAME, id=run_id, config=config) as run:
        # Perform training
        best_model_accuracy, best_model_state, last_epoch = _train(config, data_loaders, n_epochs)

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

        # Evaluate the best model on the test set
        _evaluate(best_model, data_loaders, "test", training_epoch=last_epoch)
        
        # Return training result
        return best_model_accuracy, best_model_path

def evaluate(model, data_loaders, loader_type):
    date_s = time.strftime('%Y%m%dT%H%M%S')
    run_id = f"evaluate__{date_s}"
    with wandb.init(project=PROJECT_NAME, id=run_id): 
        return _evaluate(model, data_loaders, loader_type)

def _evaluate(model, data_loaders, loader_type, training_epoch=None):
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

    # In case evaluation is happening within a training 
    # epoch then log the evaluation metrics to the W&B run
    if training_epoch:
        # Log confusion matrix
        cm = confusion_matrix(all_labels, all_predictions)
        plt.figure(figsize=(10, 10))
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')
        plt.xlabel('Predicted')
        plt.ylabel('True')
        confusion_matrix_image = wandb.Image(plt)
        plt.close()

        # Log misclassifications
        misclassified_images = []
        for img, true_label, pred_label in misclassifications[:25]:  # Limit to 25 images
            plt.figure(figsize=(2, 2))
            plt.imshow(img.squeeze(), cmap='gray')
            plt.title(f'True: {true_label}, Pred: {pred_label}')
            plt.axis('off')
            misclassified_images.append(wandb.Image(plt))
            plt.close()
        
        # Log metrics to W&B
        wandb.log({
            f"{loader_type}_accuracy": accuracy,
            f"{loader_type}_loss": average_loss,
            f"{loader_type}_precision": precision,
            f"{loader_type}_recall": recall,
            f"{loader_type}_f1": f1,
            f"{loader_type}_confusion_matrix": confusion_matrix_image,
            f"{loader_type}_misclassified_images": misclassified_images,
        }, step=training_epoch)

    return accuracy, average_loss, precision, recall, f1, all_labels, all_predictions, misclassifications

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
    parser.add_argument('--n_epochs', type=int, default=50, help='Number of epochs to train the model for') # TODO: move to hyperparams
    parser.add_argument('--hyperparams_path', type=str, default="configs/hyperparams/LeNet5.yml", help='Path to the hyperparameters file')
    parser.add_argument('--model_path', type=str, default="outputs/best_model.pth", help='Path to the model file for evaluation')
    parser.add_argument('--model_output_dir', type=str, default="outputs", help='Directory to save the model file')
    args = parser.parse_args()

    # Set the random seed for reproducibility
    set_seed(args.seed)

    # Create data loaders
    config = load_config(args.hyperparams_path)
    batch_size = config['data_loader']['batch_size']
    data_loaders = create_data_loaders(batch_size=batch_size)

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
