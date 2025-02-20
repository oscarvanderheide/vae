from torchvision import datasets, transforms
from torch.utils.data import DataLoader, random_split
import pytorch_lightning as pl


class MNISTDataModule(pl.LightningDataModule):
    def __init__(
        self,
        batch_size=128,
        split_val=0.8,
        num_workers=7,
    ):
        super().__init__()
        self.batch_size = batch_size
        self.split_val = split_val
        self.transform = transforms.Compose(
            [
                transforms.ToTensor(),
                # transforms.Normalize((0.1307,), (0.3081,)),
            ]
        )
        self.num_workers = num_workers

    def setup(self, stage=None):
        # Load the full training dataset
        mnist_full = datasets.MNIST(
            root="./data", train=True, transform=self.transform, download=True
        )
        # Split the dataset into training and validation sets
        train_size = int(self.split_val * len(mnist_full))
        val_size = len(mnist_full) - train_size
        self.mnist_train, self.mnist_val = random_split(
            mnist_full, [train_size, val_size]
        )
        # Load the test dataset
        self.mnist_test = datasets.MNIST(
            root="./data", train=False, transform=self.transform, download=True
        )

    def train_dataloader(self):
        return DataLoader(
            self.mnist_train,
            batch_size=self.batch_size,
            shuffle=True,
            num_workers=self.num_workers,
            persistent_workers=True
        )

    def val_dataloader(self):
        return DataLoader(
            self.mnist_val,
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=self.num_workers,
            persistent_workers=True
        )

    def test_dataloader(self):
        return DataLoader(
            self.mnist_test,
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=self.num_workers,
            persistent_workers=True
        )
