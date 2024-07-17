import os
import pandas as pd
import numpy as np
import zipfile
import torch
from torch.utils.data import Dataset, DataLoader
from kaggle.api.kaggle_api_extended import KaggleApi
import zipfile
import torchvision
import torchvision.transforms as transforms
import albumentations as A
from albumentations.pytorch import ToTensorV2
import numpy as np
import torch
import torchvision.transforms as transforms

class DatasetTransformWrapper(Dataset):

    def __init__(self, dataset, transform=None):
        self.dataset = dataset
        self.transform = transform

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, idx):
        img, label = self.dataset[idx]
        if self.transform: img = self.transform(img)
        return img, label
    
class KaggleDigitRecognizer(Dataset):
    base_folder = 'digit-recognizer'
    kaggle_competition = 'digit-recognizer'
    train_file = 'train.csv'
    test_file = 'test.csv'

    def __init__(self, root, train=True, transform=None, download=False):
        self.root = os.path.expanduser(root)
        self.transform = transform
        self.train = train  # training set or test set

        if download:
            self.download()

        if not self._check_exists():
            raise RuntimeError('Dataset not found. You can use download=True to download it')

        train_file = os.path.join(self.root, self.base_folder, self.train_file)
        test_file = os.path.join(self.root, self.base_folder, self.test_file)

        train_df = pd.read_csv(train_file)
        test_df = pd.read_csv(test_file)

        self.train_data = train_df.iloc[:, 1:].values.reshape(-1, 28, 28).astype(np.float32) / 255.0
        self.train_targets = train_df.iloc[:, 0].values
        self.test_data = test_df.values.reshape(-1, 28, 28).astype(np.float32) / 255.0
        self.test_targets = np.zeros(len(self.test_data))  # Dummy targets for test set

        if self.train:
            self.data = self.train_data
            self.targets = self.train_targets
        else:
            self.data = self.test_data
            self.targets = self.test_targets

    def __len__(self):
        return len(self.data)

    def __getitem__(self, index):
        img, target = self.data[index], self.targets[index]

        if self.transform is not None:
            img = self.transform(img)

        return img, target

    def _check_exists(self):
        return os.path.exists(os.path.join(self.root, self.base_folder, self.train_file)) and \
               os.path.exists(os.path.join(self.root, self.base_folder, self.test_file))

    def download(self):
        if self._check_exists():
            print("Files already downloaded and verified")
            return

        os.makedirs(os.path.join(self.root, self.base_folder), exist_ok=True)

        # Initialize Kaggle API
        try:
            api = KaggleApi()
            api.authenticate()
        except Exception as e:
            print(f"Error authenticating Kaggle API: {e}")
            print("Please ensure you have set up your Kaggle API credentials correctly.")
            raise

        # Download the dataset
        try:
            print(f"Downloading {self.kaggle_competition} dataset...")
            api.competition_download_file(self.kaggle_competition, self.train_file, path=os.path.join(self.root, self.base_folder))
            api.competition_download_file(self.kaggle_competition, self.test_file, path=os.path.join(self.root, self.base_folder))
            print("Dataset downloaded successfully.")

            # Extract zip files
            for file in [self.train_file, self.test_file]:
                zip_path = os.path.join(self.root, self.base_folder, file + '.zip')
                if os.path.exists(zip_path):
                    with zipfile.ZipFile(zip_path, 'r') as zip_ref:
                        zip_ref.extractall(os.path.join(self.root, self.base_folder))
                    os.remove(zip_path)  # Remove the zip file after extraction
            print("Files extracted successfully.")

        except Exception as e:
            print(f"Error downloading or extracting dataset: {e}")
            raise RuntimeError("Failed to download or extract the dataset. Please check your internet connection and try again.")

        if not self._check_exists():
            raise RuntimeError("Downloaded files not found. The download might have failed.")

    def get_num_samples(self):
        return {
            'train': len(self.train_data),
            'test': len(self.test_data)
        }

DATASETS = {
    "mnist": torchvision.datasets.MNIST,
    "kmnist": torchvision.datasets.KMNIST,
    "qmnist": torchvision.datasets.QMNIST,
    "fashionmnist": torchvision.datasets.FashionMNIST,
    "kaggledigitrecognizer": KaggleDigitRecognizer
}

def get_dataset_names():
    return list(DATASETS.keys())

def load_dataset(dataset_name, dataset_root='./data'):
    dataset_class = DATASETS[dataset_name]
    train_dataset = dataset_class(root=dataset_root, train=True, download=True, transform=transforms.ToTensor())
    test_dataset = dataset_class(root=dataset_root, train=False, download=True, transform=transforms.ToTensor())
    dataset = {
        "dataset_class" : dataset_class, 
        "train": train_dataset, 
        "test": test_dataset
    }
    return dataset

def get_dataset_metrics(dataset):
    train = dataset["train"]
    test = dataset["test"]

    loader = DataLoader(train)
    data = next(iter(loader))[0]
    train_min = data.min().item()
    train_max = data.max().item()
    train_mean = data.mean().item()
    train_std = data.std().item()
    train_median = data.median().item()
    train_variance = data.var().item()
    train_n_samples = len(train)
    
    loader = DataLoader(test)
    data = next(iter(loader))[0]
    test_min = data.min().item()
    test_max = data.max().item()
    test_mean = data.mean().item()
    test_std = data.std().item()
    test_median = data.median().item()
    test_variance = data.var().item()
    test_n_samples = len(test)

    data = torch.utils.data.ConcatDataset([train, test])
    data = next(iter(loader))[0]
    _min = data.min().item()
    _max = data.max().item()
    mean = data.mean().item()
    std = data.std().item()
    median = data.median().item()
    variance = data.var().item()

    n_classes = len(set(train.targets.tolist()))
    n_samples = train_n_samples + test_n_samples

    metrics = {
        'n_classes':n_classes,
        'n_samples': n_samples,
        'min': _min,
        'max': _max,
        'mean': mean,
        'std': std,
        'median' : median,
        'variance' : variance,
        'train' : {
            'n_samples': train_n_samples,
            'min': train_min,
            'max': train_max,
            'mean': train_mean,
            'std': train_std,
            'median' : train_median,
            'variance' : train_variance
        },
        'test' : {
            'n_samples': n_samples,
            'min': test_min,
            'max': test_max,
            'mean': test_mean,
            'std': test_std,
            'median' : test_median,
            'variance' : test_variance
        }
    }
    
    return metrics

class AlbumentationsToTorchvisionWrapper:

    def __init__(self, albumentations_transform):
        self.albumentations_transform = albumentations_transform

    def __call__(self, image):
        assert isinstance(image, torch.Tensor), f"Invalid image type: {type(image)}"
        image = image.squeeze()
        image = np.array(image)
        assert image.shape == (28, 28), f"Invalid image shape: {image.shape}"
        transformed = self.albumentations_transform(image=image)
        transformed_image = transformed['image']
        return transformed_image
    
def build_albumentations_pipeline(pipeline_config, mean, std):
    a_pipeline = []
    for transform in pipeline_config:
        name = transform['name']
        params = transform['params']
        a_pipeline.append(getattr(A, name)(**params))
    a_pipeline.append(ToTensorV2())
    transform = transforms.Compose([
        AlbumentationsToTorchvisionWrapper(A.Compose(a_pipeline)), 
        transforms.Normalize((mean,), (std,))
    ])
    return transform