# Configure maximum number of threads for NumExpr 
# before importing libs that require it
import os
import multiprocessing
NUM_CORES = multiprocessing.cpu_count()
if not "NUMEXPR_MAX_THREADS" in os.environ: os.environ["NUMEXPR_MAX_THREADS"] = str(NUM_CORES)

import argparse
from matplotlib import pyplot as plt
from libs.datasets import load_dataset, get_dataset_metrics, get_dataset_names, DatasetTransformWrapper, build_albumentations_pipeline

# Define the default configuration for the model
PIPELINE_CONFIG = [
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

def fetch_samples(dataset):
    samples = []
    for i in range(10):
        sample = next(x for x in dataset if x[1] == i)
        samples.append(sample)
    return samples


def show_image(img, ax):
    img = img.cpu().numpy().transpose(1, 2, 0).squeeze()
    ax.imshow(img, cmap='gray')
    ax.axis('off')

def main(args):
    # Load the dataset
    dataset = load_dataset(args.dataset)
    dataset_metrics = get_dataset_metrics(dataset)
    mean = dataset_metrics['mean']
    std = dataset_metrics['std']

    dataset = load_dataset(args.dataset)['train']
    augment_transform = build_albumentations_pipeline(PIPELINE_CONFIG, mean, std)
    dataset = DatasetTransformWrapper(dataset, transform=augment_transform)

    # Display images
    _, axes = plt.subplots(1, 10, figsize=(15, 2))
    plt.ion()

    try:
        while True:
            samples = fetch_samples(dataset)
            for i, (img, label) in enumerate(samples):
                ax = axes[i]
                ax.clear()
                show_image(img, ax)
                ax.set_title(f'{label}')
            
            plt.draw()
            plt.pause(0.5)
    except KeyboardInterrupt:
        print("Stopped by user")
        plt.ioff()
        plt.show()


if __name__ == "__main__":
    dataset_names = get_dataset_names()
    parser = argparse.ArgumentParser(description='Display a grid of images from a dataset.')
    parser.add_argument('--dataset', type=str, default='mnist', choices=dataset_names, help='Dataset name')
    args = parser.parse_args()
    main(args)