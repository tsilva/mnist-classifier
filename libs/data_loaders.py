import logging
import json
import time
import torch
import multiprocessing
import numpy as np
import torchvision.transforms as transforms
from torch.utils.data import DataLoader, random_split

from libs.misc import get_global_seed, set_global_seed
from libs.datasets import load_dataset, get_dataset_metrics, build_albumentations_pipeline, DatasetTransformWrapper, create_bootstrap_dataset

class CustomDataLoader(DataLoader):
    def __init__(self, *args, device=None, **kwargs):
        super().__init__(*args, **kwargs)

        assert device is not None, "Device must be specified"
        self.device = device

    def __iter__(self):
        self.iter_data = super().__iter__()
        self.sample_times = []
        self.transfer_times = []
        self.load_times = []
        self.processing_time = []
        self.last_end_time = None
        return self

    def __next__(self):
        if self.last_end_time is not None:
            processing_time = time.time() - self.last_end_time
            self.processing_time.append(processing_time)

        # Measure time to load data
        start_time = time.time()
        batch = next(self.iter_data)
        sample_time = time.time() - start_time
        self.sample_times.append(sample_time)

        # Measure time to transfer data to GPU
        start_time = time.time()
        batch = self._transfer_to_device(batch)
        transfer_time = time.time() - start_time
        self.transfer_times.append(transfer_time)
        
        load_time = sample_time + transfer_time
        self.load_times.append(load_time)

        self.last_end_time = time.time()

        return batch

    def _transfer_to_device(self, batch):
        if isinstance(batch, dict):
            return {k: v.to(self.device, non_blocking=True) for k, v in batch.items()}
        elif isinstance(batch, list) or isinstance(batch, tuple):
            return [self._transfer_to_device(v) for v in batch]
        else:
            return batch.to(self.device, non_blocking=True)

    def get_stats(self):
        load_time = np.sum(self.load_times)
        batches_per_second = 1.0 / load_time
        stats = {
            "sample_time": round(np.sum(self.sample_times), 2),
            "transfer_time": round(np.sum(self.transfer_times), 2),
            "load_time": round(load_time, 2),
            "processing_time": round(np.sum(self.processing_time), 2),
            "batches_per_second": round(batches_per_second, 4)
        }
        return stats
        
def _data_loader_worker_init_fn(worker_id):
    worker_info = torch.utils.data.get_worker_info()
    base_seed = worker_info.seed
    worker_seed = (base_seed + worker_id) % 2**32
    set_global_seed(worker_seed)

def create_data_loaders(
    dataset_id, 
    device, 
    batch_size=64, 
    train_augmentation=None, 
    train_bootstrap_percentage=None, 
    validation_split=0
):
    # Load dataset and calculate its metrics
    dataset = load_dataset(dataset_id)
    dataset_metrics = get_dataset_metrics(dataset)
    logging.info("Dataset metrics:" + json.dumps(dataset_metrics, indent=2))

    # Unpack the dataset into training, validation, and test sets
    train_dataset = dataset['train']
    test_dataset = dataset['test']
    validation_dataset = test_dataset

    # In case a bootstrap percentage was specified then create 
    # an alternative bootstap dataset (new dataset is created 
    # through resampling with repetition of original dataset,
    # useful for creating bagging ensemble models)
    if train_bootstrap_percentage:
        train_dataset = create_bootstrap_dataset(train_dataset, train_bootstrap_percentage)
    
    # In case a validation split was specified then replace 
    # the validation set with a subset of the training set
    if validation_split > 0:
        train_size = int((1 - validation_split) * len(train_dataset))
        validation_size = len(train_dataset) - train_size
        train_dataset, validation_dataset = random_split(train_dataset, [train_size, validation_size])

    # Apply the default transformation to the datasets
    # (normalize the inputs using the overall mean and standard deviation)
    dataset_mean = dataset_metrics['mean']
    dataset_std = dataset_metrics['std']
    default_transform = transforms.Compose([
        transforms.Normalize((dataset_mean,), (dataset_std,))
    ])

    # Build the data augmentation pipeline
    augment_transform = build_albumentations_pipeline(train_augmentation, dataset_mean, dataset_std) if train_augmentation else default_transform
       
    # Wrap datasets with their respective transform operations
    # (this will make transformations work regardless of wheter
    # the dataset is a Dataset or a Subset)
    train_dataset = DatasetTransformWrapper(train_dataset, transform=augment_transform)
    validation_dataset = DatasetTransformWrapper(validation_dataset, transform=default_transform)
    test_dataset = DatasetTransformWrapper(test_dataset, transform=default_transform)

    # Create the data loaders
    seed = get_global_seed()
    num_workers = multiprocessing.cpu_count()
    prefetch_factor = 2
    train_loader = CustomDataLoader(
        train_dataset, 
        batch_size=batch_size, 
        shuffle=True, 
        num_workers=num_workers,
        persistent_workers=True,
        worker_init_fn=_data_loader_worker_init_fn, 
        generator=torch.Generator().manual_seed(seed),
        prefetch_factor=prefetch_factor,
        pin_memory=True,
        device=device
    )
    validation_loader = CustomDataLoader(
        validation_dataset, 
        batch_size=batch_size, 
        shuffle=False, 
        num_workers=num_workers, 
        persistent_workers=True,
        worker_init_fn=_data_loader_worker_init_fn, 
        generator=torch.Generator().manual_seed(seed), 
        prefetch_factor=prefetch_factor,
        pin_memory=True,
        device=device
    )
    test_loader = CustomDataLoader(
        test_dataset, 
        batch_size=batch_size, 
        shuffle=False, 
        num_workers=num_workers, 
        persistent_workers=True,
        worker_init_fn=_data_loader_worker_init_fn, 
        generator=torch.Generator().manual_seed(seed), 
        prefetch_factor=prefetch_factor,
        pin_memory=True,
        device=device
    )

    # Return the data loaders
    return {
        'train': train_loader, 
        'validation': validation_loader, 
        'test': test_loader
    }

