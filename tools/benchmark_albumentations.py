"""
Benchmarking different image augmentation libraries.
"""

import time
import numpy as np
import albumentations as A
from albumentations.pytorch import ToTensorV2
from torchvision import transforms
from PIL import Image

# Function to load images
def load_images(num_images, size=(128, 128)):
    images = []
    for _ in range(num_images):
        # Generate a random image for benchmarking
        image = np.random.randint(0, 256, size + (3,), dtype=np.uint8)
        images.append(image)
    return images

# Albumentations pipeline matching Torchvision
albumentations_pipeline = A.Compose([
    A.Rotate(limit=10),
    A.RandomResizedCrop(height=28, width=28, scale=(0.8, 1.0)),
    A.HorizontalFlip(),
    A.VerticalFlip(),
    A.Affine(rotate=(-15, 15), translate_percent=(0.1, 0.1)),
    A.Normalize(mean=(0.5, 0.5, 0.5), std=(0.5, 0.5, 0.5)),
    A.OneOf([
        A.CoarseDropout(max_holes=1, max_height=8, max_width=8, min_holes=1, min_height=4, min_width=4, fill_value=0),
        A.CoarseDropout(max_holes=1, max_height=8, max_width=8, min_holes=1, min_height=4, min_width=4, fill_value=255)
    ], p=0.5),
    ToTensorV2()
])

# Torchvision pipeline
torchvision_pipeline = transforms.Compose([
    transforms.RandomRotation(degrees=10),
    transforms.RandomResizedCrop(size=28, scale=(0.8, 1.0)),
    transforms.RandomHorizontalFlip(),
    transforms.RandomVerticalFlip(),
    transforms.RandomAffine(degrees=15, translate=(0.1, 0.1)),
    transforms.ToTensor(),
    transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
    transforms.RandomErasing(p=0.5, scale=(0.02, 0.33), ratio=(0.3, 3.3), value='random')
])

# Function to benchmark Albumentations
def benchmark_albumentations(images):
    start_time = time.time()
    for image in images:
        albumentations_pipeline(image=image)
    return time.time() - start_time

# Function to benchmark Torchvision
def benchmark_torchvision(images):
    start_time = time.time()
    for image in images:
        pil_image = Image.fromarray(image)
        torchvision_pipeline(pil_image)
    return time.time() - start_time

# Main function to run the benchmarks
def main(num_images=100):
    images = load_images(num_images)

    albumentations_time = benchmark_albumentations(images)
    torchvision_time = benchmark_torchvision(images)

    print(f"Albumentations time for {num_images} images: {albumentations_time:.4f} seconds")
    print(f"Torchvision time for {num_images} images: {torchvision_time:.4f} seconds")

if __name__ == "__main__":
    main()
