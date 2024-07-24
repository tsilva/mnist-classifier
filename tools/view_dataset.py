import torch
import torchvision
import multiprocessing
import matplotlib.pyplot as plt
import numpy as np
import random
import matplotlib.animation as animation
import argparse

from libs.datasets import load_dataset, get_dataset_metrics, get_dataset_names

def load_and_prepare_dataset(dataset_name, num_samples):
    # Load the dataset
    dataset = load_dataset(dataset_name)
    trainset = dataset["train"]
    dataset_metrics = get_dataset_metrics(dataset)
    n_classes = dataset_metrics['n_classes']
    
    # Create a DataLoader for efficient data loading
    num_workers = multiprocessing.cpu_count()
    trainloader = torch.utils.data.DataLoader(
        trainset, 
        batch_size=1, 
        shuffle=False, 
        num_workers=num_workers
    )
    
    # Initialize a dictionary to store images for each category
    category_images = {i: [] for i in range(n_classes)}
    
    # Populate the dictionary with images from each category
    for data in trainloader:
        images, labels = data
        label = labels.item()
        category_images[label].append(images[0])
        
        # Stop if we have collected enough samples for each category
        if all(len(category_images[i]) >= num_samples for i in range(n_classes)):
            break
    
    return category_images, n_classes

def create_image_grid(category_images, num_samples, n_classes):
    # Randomly select images for each category
    grid_images = [
        random.choice(category_images[k])
        for _ in range(num_samples)
        for k in range(n_classes)
    ]
    
    # Create a grid of images
    grid = torchvision.utils.make_grid(grid_images, nrow=n_classes, padding=0)
    return grid

def update_display(frame, ax, category_images, num_samples, n_classes):
    # Create a new image grid
    grid = create_image_grid(category_images, num_samples, n_classes)
    
    # Clear the previous image and display the new one
    ax.clear()
    ax.imshow(np.transpose(grid.numpy(), (1, 2, 0)), cmap='gray')
    ax.axis('off')

def visualize_dataset(category_images, num_samples, n_classes, interval):
    # Set up the plot
    fig, ax = plt.subplots()
    
    # Create the animation
    _ = animation.FuncAnimation(
        fig,
        update_display,
        fargs=(ax, category_images, num_samples, n_classes),
        interval=interval
    )
    
    # Display the animation
    plt.show()

def main(args):
    # Load and prepare the dataset
    category_images, n_classes = load_and_prepare_dataset(args.dataset, args.num_samples)
    
    # Visualize the dataset
    visualize_dataset(category_images, args.num_samples, n_classes, args.interval)

if __name__ == "__main__":
    dataset_names = get_dataset_names()
    parser = argparse.ArgumentParser(description='Display a grid of images from a dataset.')
    parser.add_argument('--dataset', type=str, default='mnist', choices=dataset_names, help='Dataset name')
    parser.add_argument('--num_samples', type=int, default=10, help='Number of samples per category')
    parser.add_argument('--interval', type=int, default=500, help='Interval in milliseconds for animation')
    args = parser.parse_args()
    main(args)