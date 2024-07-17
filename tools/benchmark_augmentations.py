import time
import logging
from multiprocessing import Pool, cpu_count

import numpy as np
import albumentations as A
from albumentations.pytorch import ToTensorV2
import torchvision.transforms as T

# Global variables for pipelines
albumentations_pipeline = None
torchvision_pipeline = None

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def load_image(size=(128, 128)):
    return np.random.randint(0, 256, size + (3,), dtype=np.uint8)

def process_image_albumentations(_):
    image = load_image()
    albumentations_pipeline(image=image)

def process_image_torchvision(_):
    image = load_image()
    torchvision_pipeline(image)

def main():
    global albumentations_pipeline, torchvision_pipeline
    
    # Configuration
    num_images = 1000

    logger.info(f"Running benchmarks for {num_images:,} images...")

    # Define Albumentations pipeline
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

    # Define torchvision pipeline
    torchvision_pipeline = T.Compose([
        T.ToPILImage(),
        T.RandomRotation(10),
        T.RandomResizedCrop(size=(28, 28), scale=(0.8, 1.0)),
        T.RandomHorizontalFlip(),
        T.RandomVerticalFlip(),
        T.RandomAffine(degrees=(-15, 15), translate=(0.1, 0.1)),
        T.ToTensor(),
        T.Normalize(mean=(0.5, 0.5, 0.5), std=(0.5, 0.5, 0.5)),
        T.RandomErasing(p=0.5, scale=(0.02, 0.08), ratio=(0.5, 2), value='random')
    ])

    # Run Albumentations benchmark
    start_time = time.time()
    with Pool(processes=cpu_count()) as pool:
        pool.map(process_image_albumentations, range(num_images))
    albumentations_time = time.time() - start_time
    logger.info(f"Albumentations time: {albumentations_time:.4f} seconds")

    # Run torchvision benchmark
    start_time = time.time()
    with Pool(processes=cpu_count()) as pool:
        pool.map(process_image_torchvision, range(num_images))
    torchvision_time = time.time() - start_time
    logger.info(f"Torchvision time: {torchvision_time:.4f} seconds")

    # Log comparison results
    logger.info("Comparison Results:")
    logger.info(f"Speed difference: {abs(albumentations_time - torchvision_time):.4f} seconds")
    faster = "Albumentations" if albumentations_time < torchvision_time else "Torchvision"
    ratio = torchvision_time / albumentations_time if albumentations_time < torchvision_time else albumentations_time / torchvision_time
    logger.info(f"{faster} is {ratio:.2f}x faster")

if __name__ == "__main__":
    main()