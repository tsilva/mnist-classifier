import logging
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import torchvision
import torchvision.transforms as transforms
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix, classification_report
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
from sklearn.ensemble import RandomForestClassifier
import seaborn as sns
import os

# Setup logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

output_dir = "outputs/analytics"
os.makedirs(output_dir, exist_ok=True)

def load_mnist():
    logging.info("Loading MNIST dataset...")
    transform = transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.1307,), (0.3081,))])
    train_dataset = torchvision.datasets.MNIST(root='./data', train=True, download=True, transform=transform)
    test_dataset = torchvision.datasets.MNIST(root='./data', train=False, download=True, transform=transform)
    
    X_train = train_dataset.data.numpy()
    y_train = train_dataset.targets.numpy()
    X_test = test_dataset.data.numpy()
    y_test = test_dataset.targets.numpy()
    
    X = np.concatenate((X_train, X_test))
    y = np.concatenate((y_train, y_test))
    
    return X.reshape(-1, 28*28), y

def preprocess_data(X, y):
    logging.info("Preprocessing data...")
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    X_train = X_train / 255.0
    X_test = X_test / 255.0
    return X_train, X_test, y_train, y_test

def visualize_class_distribution(y):
    logging.info("Visualizing class distribution...")
    class_dist = np.bincount(y)
    plt.figure()
    plt.bar(range(10), class_dist)
    plt.title('Class Distribution in MNIST')
    plt.xlabel('Digit')
    plt.ylabel('Count')
    plt.savefig(os.path.join(output_dir, 'class_distribution.png'))
    plt.close()

def visualize_sample_digits(X, y):
    logging.info("Visualizing sample digits...")
    fig, axes = plt.subplots(2, 5, figsize=(12, 5))
    for i, ax in enumerate(axes.flat):
        sample_idx = np.where(y == i)[0][0]
        ax.imshow(X[sample_idx].reshape(28, 28), cmap='gray')
        ax.set_title(f'Digit: {i}')
        ax.axis('off')
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'sample_digits.png'))
    plt.close()

def visualize_pixel_intensity(X):
    logging.info("Visualizing average pixel intensity...")
    avg_intensity = np.mean(X, axis=0).reshape(28, 28)
    plt.figure()
    plt.imshow(avg_intensity, cmap='gray')
    plt.title('Average Pixel Intensity Across All Images')
    plt.colorbar()
    plt.savefig(os.path.join(output_dir, 'pixel_intensity.png'))
    plt.close()

def perform_pca(X, y):
    logging.info("Performing PCA...")
    pca = PCA(n_components=2)
    X_pca = pca.fit_transform(X)
    plt.figure()
    scatter = plt.scatter(X_pca[:, 0], X_pca[:, 1], c=y, cmap='tab10', alpha=0.5)
    plt.colorbar(scatter)
    plt.title('PCA of MNIST Dataset')
    plt.xlabel('First Principal Component')
    plt.ylabel('Second Principal Component')
    plt.savefig(os.path.join(output_dir, 'pca.png'))
    plt.close()
    logging.info(f"Explained variance ratio: {pca.explained_variance_ratio_}")

def perform_tsne(X, y):
    logging.info("Performing t-SNE...")
    tsne = TSNE(n_components=2, random_state=42)
    X_tsne = tsne.fit_transform(X[:5000])  # Using a subset for speed
    plt.figure()
    scatter = plt.scatter(X_tsne[:, 0], X_tsne[:, 1], c=y[:5000], cmap='tab10', alpha=0.5)
    plt.colorbar(scatter)
    plt.title('t-SNE Visualization of MNIST Dataset')
    plt.xlabel('t-SNE feature 1')
    plt.ylabel('t-SNE feature 2')
    plt.savefig(os.path.join(output_dir, 't-sne.png'))
    plt.close()

def compute_confusion_matrix(X_train, X_test, y_train, y_test):
    logging.info("Computing confusion matrix...")
    rf = RandomForestClassifier(n_estimators=100, random_state=42)
    rf.fit(X_train, y_train)
    y_pred = rf.predict(X_test)
    cm = confusion_matrix(y_test, y_pred)
    plt.figure(figsize=(10, 8))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')
    plt.title('Confusion Matrix')
    plt.xlabel('Predicted Label')
    plt.ylabel('True Label')
    plt.savefig(os.path.join(output_dir, 'confusion_matrix.png'))
    plt.close()
    report = classification_report(y_test, y_pred)
    with open(os.path.join(output_dir, 'classification_report.txt'), 'w') as f:
        f.write(report)

def main():
    logging.info("Starting MNIST analysis...")
    X, y = load_mnist()
    X_train, X_test, y_train, y_test = preprocess_data(X, y)
    
    logging.info("MNIST Dataset Analytics:")
    logging.info(f"Total samples: {len(X)}")
    logging.info(f"Training samples: {len(X_train)}")
    logging.info(f"Testing samples: {len(X_test)}")
    logging.info(f"Image shape: {X[0].shape}")
    logging.info(f"Number of classes: {len(np.unique(y))}")

    visualize_class_distribution(y)
    visualize_sample_digits(X, y)
    visualize_pixel_intensity(X)
    perform_pca(X, y)
    perform_tsne(X, y)
    compute_confusion_matrix(X_train, X_test, y_train, y_test)

    logging.info(f"Analysis complete. Output saved to {output_dir} directory.")

if __name__ == "__main__":
    main()