# mnist-classifier

## Installation

1. Clone this repository:

```
git clone https://github.com/tsilva/mnist-classifier.git
cd mnist-classifier
```

2. Install Miniconda:
   - Visit the [Miniconda website](https://docs.conda.io/en/latest/miniconda.html) and download the appropriate installer for your operating system.
   - Follow the installation instructions for your platform.

3. Create a new Conda environment:

```
conda env create -f environment.yml
```

3. Activate the new environment:

```
conda activate mnist-classifier
```

4. Ensure that CUDA is available:

```
python -c "import torch; print(torch.cuda.is_available()); print(torch.cuda.current_device()); print(torch.cuda.get_device_name(0))"
```

If not available try the following (for CUDA 11.8):

```
conda activate mnist-classifier
pip uninstall torch torchvision
pip install torch torchvision torchaudio --extra-index-url https://download.pytorch.org/whl/cu118
```