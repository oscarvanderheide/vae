from torch.utils.data import random_split
from torchvision import datasets, transforms

from datasets.abstract_datamodule import AbstractDataModule


class MNISTDataModule(AbstractDataModule):
    """MNIST DataModule for training and evaluating VAE models.

    This module handles downloading, preprocessing, and creating dataloaders
    for the MNIST handwritten digits dataset. It splits the training data into
    train and validation sets based on the provided split ratio.

    Args:
        batch_size (int, optional): Number of samples per batch. Default: 128
        split_val (float, optional): Fraction of training data to use for training
            (remainder used for validation). Value between 0 and 1. Default: 0.8
        num_workers (int, optional): Number of subprocesses to use for data loading.
            Default: 7
    """

    def __init__(
        self,
        batch_size=128,
        split_val=0.8,
        num_workers=7,
    ):
        super().__init__(batch_size, num_workers)
        self.split_val = split_val
        self.transform = transforms.Compose(
            [
                transforms.ToTensor(),
                # transforms.Normalize((0.1307,), (0.3081,)),
            ]
        )

    def setup(self, stage=None):
        """Prepare datasets for training, validation, and testing.

        This method is called by Lightning before training/validation/testing.
        It downloads the MNIST dataset if needed and splits it into train/val/test sets.

        Args:
            stage (str, optional): Stage of training ('fit', 'validate', 'test', or None).
                If None, sets up all stages. Default: None
        """
        # Load the full training dataset
        mnist_full = datasets.MNIST(
            root="./data", train=True, transform=self.transform, download=True
        )
        # Split the dataset into training and validation sets
        train_size = int(self.split_val * len(mnist_full))
        val_size = len(mnist_full) - train_size
        self.train_dataset, self.val_dataset = random_split(
            mnist_full, [train_size, val_size]
        )
        # Load the test dataset
        self.test_dataset = datasets.MNIST(
            root="./data", train=False, transform=self.transform, download=True
        )
