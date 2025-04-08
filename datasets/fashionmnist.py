from torch.utils.data import Subset, random_split
from torchvision import datasets, transforms

from datasets.abstract_datamodule import AbstractDataModule

# 10 labels in FashionMNIST:
# https://github.com/zalandoresearch/fashion-mnist
# 0: T-shirt/top, 1: Trouser, 2: Pullover, 3: Dress, 4: Coat, 5: Sandal, 6: Shirt, 7: Sneaker, 8: Bag, 9: Ankle boot


class FashionMNISTDataModule(AbstractDataModule):
    """FashionMNIST DataModule for training and evaluating VAE models.

    This module handles downloading, preprocessing, and creating dataloaders for the
    Fashion-MNIST dataset. It splits the training data into train and validation sets
    based on the provided split ratio. Optionally, it can filter the dataset to include
    only specific clothing categories.

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
                # transforms.Normalize((0.5,), (0.5,)),  # Optional normalization
            ]
        )

    def setup(self, stage=None):
        """Prepare datasets for training, validation, and testing.

        This method is called by Lightning before training/validation/testing.
        It downloads the Fashion-MNIST dataset if needed and splits it into
        train/val/test sets. Includes commented code for filtering specific categories.

        Args:
            stage (str, optional): Stage of training ('fit', 'validate', 'test', or None).
                If None, sets up all stages. Default: None
        """
        # Load the full FashionMNIST training dataset
        fashion_mnist_full = datasets.FashionMNIST(
            root="./data", train=True, transform=self.transform, download=True
        )
        # # Filter dataset to include only samples with labels 8 (Bag) and 9 (Ankle boot)
        # indices = [
        #     # i for i, (_, label) in enumerate(fashion_mnist_full) if label == 8 # include only samples with label 8 (Bag)
        #     # i for i, (_, label) in enumerate(fashion_mnist_full) if label in [8, 9]
        #     i
        #     for i, (_, label) in enumerate(fashion_mnist_full)
        #     if label in [0, 9]
        # ]
        # filtered_dataset = Subset(fashion_mnist_full, indices)
        filtered_dataset = fashion_mnist_full

        # Split the filtered dataset into training and validation sets
        train_size = int(self.split_val * len(filtered_dataset))
        val_size = len(filtered_dataset) - train_size
        self.train_dataset, self.val_dataset = random_split(
            filtered_dataset, [train_size, val_size]
        )

        # Load the test dataset and filter it as needed
        fashion_mnist_test = datasets.FashionMNIST(
            root="./data", train=False, transform=self.transform, download=True
        )
        test_indices = [
            # i for i, (_, label) in enumerate(fashion_mnist_test) if label == 8 # label 8 (bags) only
            # i for i, (_, label) in enumerate(fashion_mnist_test) if label in [8, 9]
            i
            for i, (_, label) in enumerate(fashion_mnist_test)
            if label in [0, 9]
        ]
        self.test_dataset = Subset(fashion_mnist_test, test_indices)
