
import json
import random
import numpy as np
import torch
import logging

class CustomJSONEncoder(json.JSONEncoder):
    def default(self, obj):
        try: return json.JSONEncoder.default(self, obj)
        except (TypeError, OverflowError): return None

def set_seed(seed):
    """
    Set a seed in the random number generators for reproducibility.
    """

    random.seed(seed) # Set the seed for the random number generator
    np.random.seed(seed) # Set the seed for NumPy
    torch.manual_seed(seed) # Set the seed for PyTorch
    torch.cuda.manual_seed(seed) # Set the seed for CUDA
    torch.cuda.manual_seed_all(seed) # Set the seed for all CUDA devices
    torch.backends.cudnn.deterministic = True # Ensure deterministic results (WARNING: can slow down training!)
    torch.backends.cudnn.benchmark = False # Disable cuDNN benchmarking (WARNING: can slow down training!)

def get_device():
    # Retrieve the device to run the models on
    # (use HW acceleration if available, otherwise use CPU)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Make sure the user is aware if HW acceleration is not available
    logging.info(f"Using device: {device}")
    if str(device) == "cpu":
        logging.warning("CUDA is not available. Running on CPU. Press Enter to continue...")
        input()

    return device