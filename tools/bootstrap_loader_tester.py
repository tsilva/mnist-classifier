import argparse
import numpy as np
from torchvision import datasets, transforms
import logging
from datasets import create_bootstrap_dataset
from multiprocessing import Pool, cpu_count

# Setup logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

def calculate_frequencies(dataset):
    frequencies = np.zeros(10)
    for _, target in dataset:
        frequencies[target] += 1
    return (frequencies / frequencies.sum()) * 100

def log_frequencies(frequencies, prefix=""):
    for digit, freq in enumerate(frequencies):
        logging.info(f"{prefix}Digit {digit}: {freq:.2f}%")

def calculate_statistics(frequencies):
    return {
        'mean': np.mean(frequencies),
        'median': np.median(frequencies),
        'max': np.max(frequencies),
        'min': np.min(frequencies),
        'variance': np.var(frequencies),
        'std': np.std(frequencies)
    }

def log_statistics(statistics, prefix=""):
    for stat, value in statistics.items():
        logging.info(f"{prefix}{stat.capitalize()}: {value:.2f}")

def bootstrap_sample_analysis(train_dataset, bootstrap_percentage):
    # Create a bootstrap dataset
    bootstrap_dataset = create_bootstrap_dataset(train_dataset, bootstrap_percentage)
    bootstrap_frequencies = calculate_frequencies(bootstrap_dataset)
    sample_stats = calculate_statistics(bootstrap_frequencies)
    return bootstrap_frequencies, sample_stats

def main(args):
    num_bootstraps = args.num_bootstraps
    bootstrap_percentage = args.bootstrap_percentage

    # Load MNIST dataset
    transform = transforms.Compose([transforms.ToTensor()])
    train_dataset = datasets.MNIST(root='./data', train=True, download=True, transform=transform)

    # Calculate and log original frequencies
    original_frequencies = calculate_frequencies(train_dataset)
    logging.info("Original dataset frequencies (percentage):")
    log_frequencies(original_frequencies)

    # Create N bootstrap samples and calculate frequencies using multiprocessing
    with Pool(cpu_count()) as pool:
        bootstrap_results = pool.starmap(bootstrap_sample_analysis, [(train_dataset, bootstrap_percentage)] * num_bootstraps)

    # Collect bootstrap statistics and frequencies
    bootstrap_statistics = {stat: [] for stat in ['mean', 'median', 'max', 'min', 'variance', 'std']}
    for i, (bootstrap_frequencies, sample_stats) in enumerate(bootstrap_results):
        logging.info(f"\nBootstrap sample {i+1} frequencies (percentage):")
        log_frequencies(bootstrap_frequencies)
        for stat, value in sample_stats.items():
            bootstrap_statistics[stat].append(value)

    # Calculate and log average bootstrap statistics
    average_statistics = {key: np.mean(values) for key, values in bootstrap_statistics.items()}
    logging.info("\nAverage statistics across all bootstrap samples:")
    log_statistics(average_statistics)

    # Calculate and log original data statistics
    original_statistics = calculate_statistics(original_frequencies)
    logging.info("\nOriginal data statistics:")
    log_statistics(original_statistics)

    # Compare bootstrap and original statistics with percentage delta
    logging.info("\nComparison of average bootstrap statistics with original data statistics:")
    for stat in average_statistics.keys():
        bootstrap_value = average_statistics[stat]
        original_value = original_statistics[stat]
        delta_percentage = ((bootstrap_value - original_value) / original_value) * 100
        logging.info(f"{stat.capitalize()} - Bootstrap: {bootstrap_value:.2f}, Original: {original_value:.2f}, Delta: {delta_percentage:.2f}%")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Bootstrap Sampling for MNIST Dataset")
    parser.add_argument('--num_bootstraps', type=int, default=20, help='Number of bootstrap samples')
    parser.add_argument('--bootstrap_percentage', type=float, default=1.0, help='Percentage of the dataset size for each bootstrap sample (0 < percentage <= 1)')
    
    args = parser.parse_args()
    main(args)
