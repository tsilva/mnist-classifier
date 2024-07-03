import torch
import numpy as np
import albumentations as A
from albumentations.pytorch import ToTensorV2
from torchvision import transforms
from matplotlib import pyplot as plt
from torchvision import datasets

class AlbumentationsToTorchvision:
    def __init__(self, albumentations_transform):
        self.albumentations_transform = albumentations_transform

    def __call__(self, img):
        img = np.array(img)
        transformed = self.albumentations_transform(image=img)
        return transformed['image']

def fetch_samples(mnist):
    samples = []
    for i in range(10):
        samples.append(next(x for x in mnist if x[1] == i))
    return samples

def show_image(img, ax):
    img = img.cpu().numpy().transpose(1, 2, 0).squeeze()
    ax.imshow(img, cmap='gray')
    ax.axis('off')

# Load the dataset without any transformations and calculate its mean and standard deviation
plain_mnist = datasets.MNIST(root='./data', train=True, download=True, transform=transforms.ToTensor())
loader = torch.utils.data.DataLoader(plain_mnist)
data = next(iter(loader))[0]
mean = data.mean().item()
std = data.std().item()
print(f'Mean: {mean}, Std: {std}')

# Improved Albumentations pipeline
albumentations_transform = AlbumentationsToTorchvision(
    A.Compose([
        A.Rotate(limit=30),
        A.RandomResizedCrop(height=28, width=28, scale=(0.8, 1.0)),
        A.GridDistortion(num_steps=5, distort_limit=0.3, p=0.5),
        A.CoarseDropout(max_holes=1, max_height=10, max_width=10, p=0.5),
        A.Normalize(mean=(mean,), std=(std,)),
        ToTensorV2()
    ])
)

# Load the MNIST dataset with Albumentations transformations
mnist = datasets.MNIST(root='./data', train=True, download=True, transform=albumentations_transform)

# Display a batch of transformed images every X seconds
fig, axes = plt.subplots(1, 10, figsize=(15, 2))
plt.ion()
try:
    # Infinite loop to continuously load and display augmented images
    samples = fetch_samples(mnist)
    while True:
        # Plot each digit
        for i, (img, label) in enumerate(samples):
            ax = axes[i]
            ax.clear()
            show_image(img, ax)
            ax.set_title(f'{label}')

        # Draw the plot
        plt.draw()
         
        # Wait X seconds
        plt.pause(0.5)

        # Fetch another batch of digits
        samples = fetch_samples(mnist)
except KeyboardInterrupt:
    print("Stopped by user")
    plt.ioff()
    plt.show()
