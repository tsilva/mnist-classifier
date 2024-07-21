import os
import logging
import torch
import torch.nn as nn
import wandb
from glob import glob


class LeNet5Original(nn.Module):
    """
    LeNet-5 original model:

    - 2 convolutional layers
    - 3 fully connected layers
    - Average pooling (legacy reasons from the original paper, max pooling would be better)
    - Tanh activation functions (legacy reasons from the original paper, ReLU would be better)
    """

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


class LeNet5Improved(nn.Module):
    """
    LeNet-5 model with improvements (based on Kaggle entry `https://www.kaggle.com/code/cdeotte/25-million-images-0-99757-mnist`):

    - 6 convolutional layers
    - 2 fully connected layers
    - Batch normalization
    - Dropout
    - ReLU activation functions
    """

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

# 8 filters, 1 layer, 1 decoder layer: 98.51%
# 8 filters, 2 layer, 1 decoder layer: 98.98%
# 8 filters, 3 layer, 1 decoder layer: 98.60%

# 8 filters, 1 layer, 2 decoder layers: 99.01%
# 8 filters, 1 layer, 2 decoder layers: 99.26%
# 8 filters, 1 layer, 2 decoder layers: 98.60%

# 16 filters 1 layer: 99.02%
# 16 filters 2 layer: 99.37%
# 16 filters 3 layer: 99.05%

# 32 filters 1 layer: 98.97%
# 32 filters 2 layer: 99.42%
# 32 filters 3 layer: 99.18%

# 64 filters 1 layer: 99.35%
# 64 filters 2 layer: 99.36%
# 64 filters 3 layer: 


# TODO: keep increasing filters until max reached
# TODO: check if its easier to train more layers with batch norm

class MinimalCNN(nn.Module):
    def __init__(self, n_encoder_layers=3, n_decoder_layers=2, out_channels=64):
        super(MinimalCNN, self).__init__()
        self.n_encoder_layers = n_encoder_layers
        self.n_decoder_layers = n_decoder_layers
        
        encoder_layers = []
        in_channels = 1

        for _ in range(n_encoder_layers):
            encoder_layers.append(nn.Conv2d(in_channels, out_channels, kernel_size=3))
            encoder_layers.append(nn.ReLU())
            encoder_layers.append(nn.MaxPool2d(kernel_size=2, stride=2))
            #encoder_layers.append(nn.BatchNorm2d(out_channels))
            in_channels = out_channels
            out_channels *= 2

        self.encoder = nn.Sequential(*encoder_layers)

        # Calculate the size of the output from the final MaxPool2d layer
        final_conv_size = self._compute_final_conv_size()

        decoder_layers = []
        in_features = final_conv_size * final_conv_size * in_channels

        for i in range(n_decoder_layers - 1):
            out_features = max(in_features // 2, 10)  # Ensure the number of neurons decreases and does not fall below 10
            decoder_layers.append(nn.Linear(in_features, out_features))
            decoder_layers.append(nn.ReLU())
            in_features = out_features

        # Final layer for classification
        decoder_layers.append(nn.Linear(in_features, 10))

        self.decoder = nn.Sequential(*decoder_layers)

    def _compute_final_conv_size(self):
        size = 28
        for _ in range(self.n_encoder_layers):
            size = (size - 2) // 2  # Each Conv2d reduces size by 2, each MaxPool2d reduces size by half
        return size

    def forward(self, x):
        # Pass through the encoder
        x = self.encoder(x)
        # Flatten the output for the classifier
        x = x.view(x.size(0), -1)
        # Pass through the classifier
        x = self.decoder(x)
        # Return output with shape (batch_size, 10)
        return x

class AdvancedCNN(nn.Module):
    """
    Advanced CNN model:

    - 7 convolutional layers
    - Different kernel sizes and strides
    - 1 fully connected layer
    - Batch normalization
    - Dropout
    - ReLU activation functions
    """

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
        

def build_model(model_config):
    """
    Factory method to build a model based on the specified configuration.
    """

    model_id = model_config['id']
    model_params = model_config.get('params', {})
    model_constructor = {
        "MinimalCNN": MinimalCNN, # TODO: remove this
        "LeNet5Original": LeNet5Original,
        "LeNet5Improved": LeNet5Improved,
        "AdvancedCNN": AdvancedCNN
    }[model_id]
    model = model_constructor(**model_params)
    return model


def init_model_weights(model, mode):
    """
    Initialize weights using the specified initialization mode.
    """

    # He initialization
    if mode == 'he':
        for m in model.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_in', nonlinearity='relu')
            elif isinstance(m, nn.Linear):
                nn.init.kaiming_normal_(m.weight, mode='fan_in', nonlinearity='relu')
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)

class EnsembleModel(torch.nn.Module):
    def __init__(self, models):
        super().__init__()
        self.models = torch.nn.ModuleList(models)
    
    def forward(self, x):
        return torch.mean(torch.stack([model(x) for model in self.models]), dim=0)

def load_model(model_path, device):
    def _load_model(path):
        assert os.path.exists(path) and path.endswith('.jit'), f"Invalid model file: {path}"
        return torch.jit.load(path, map_location=device).to(device).eval()

    if model_path.startswith('https://wandb.ai/'):
        entity_id, project_name, run_id = model_path.split('/')[3:6]
        artifact_path = f"{entity_id}/{project_name}/best_model_{run_id}.jit:latest"
        logging.info(f"Downloading model artifact: {artifact_path}")
        model_path = f"{wandb.use_artifact(artifact_path, type='model').download()}/best_model_{run_id}.jit"

    model_files = [model_path] if not os.path.isdir(model_path) else glob(os.path.join(model_path, "*.jit"))
    models = [_load_model(model_file) for model_file in model_files]
    return models[0] if len(models) == 1 else EnsembleModel(models).to(device)
