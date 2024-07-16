import time
import numpy as np
import albumentations as A
from albumentations.pytorch import ToTensorV2
from multiprocessing import Pool, cpu_count

def load_image(size=(128, 128)):
    return np.random.randint(0, 256, size + (3,), dtype=np.uint8)

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

def process_image(_):
    image = load_image()
    albumentations_pipeline(image=image)

def main(num_images=100000000):
    start_time = time.time()
    
    with Pool(processes=cpu_count()) as pool:
        pool.map(process_image, range(num_images))
    
    end_time = time.time()
    print(f"Multiprocessing Albumentations time for {num_images} images: {end_time - start_time:.4f} seconds")

if __name__ == "__main__":
    main()