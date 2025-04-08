"""Dataset modules for training and evaluating Variational Autoencoders.

This module provides PyTorch Lightning DataModule implementations for various
datasets commonly used in VAE research and applications. Each DataModule
encapsulates all the necessary logic for loading, preprocessing, and splitting
datasets into training, validation, and test sets.

Available Datasets:
-----------------
- MNISTDataModule: Implementation for the classic MNIST handwritten digits dataset,
  consisting of 28x28 grayscale images of handwritten digits (0-9).

- FashionMNISTDataModule: Implementation for the Fashion-MNIST dataset, a more
  challenging drop-in replacement for MNIST, containing 28x28 grayscale images of
  fashion items from 10 categories.

Usage:
-----
Each DataModule follows the PyTorch Lightning pattern and can be used as follows:

```python
from datasets import MNISTDataModule

# Initialize the data module with desired configurations
datamodule = MNISTDataModule(
    data_dir='./data',
    batch_size=64,
    num_workers=4
)

# Set up data for use with a PyTorch Lightning Trainer
trainer = pl.Trainer(...)
trainer.fit(model, datamodule=datamodule)
trainer.test(datamodule=datamodule)
```

Common Features:
--------------
All DataModules provide:
- Automatic download capabilities
- Standardized preprocessing
- Train/validation/test splits
- DataLoader configuration with batch size and worker control
- Transforms for data augmentation where applicable

See individual class docstrings for detailed documentation on each dataset implementation.
"""

from .fashionmnist import FashionMNISTDataModule
from .mnist import MNISTDataModule

__all__ = [
    "MNISTDataModule",
    "FashionMNISTDataModule",
]
