from torchvision import datasets, transforms
from torch.utils.data import DataLoader, random_split, Subset
import pytorch_lightning as pl

# 10 labels in FashionMNIST:
# https://github.com/zalandoresearch/fashion-mnist 
# 0: T-shirt/top, 1: Trouser, 2: Pullover, 3: Dress, 4: Coat, 5: Sandal, 6: Shirt, 7: Sneaker, 8: Bag, 9: Ankle boot

class FashionMNISTDataModule(pl.LightningDataModule):
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
                # transforms.Normalize((0.5,), (0.5,)),  # Optional normalization
            ]
        )
        self.num_workers = num_workers

    def setup(self, stage=None):
        # Load the full FashionMNIST training dataset
        fashion_mnist_full = datasets.FashionMNIST(
            root="./data", train=True, transform=self.transform, download=True
        )
        # Filter dataset to include only samples with labels 8 (Bag) and 9 (Ankle boot)
        indices = [
            # i for i, (_, label) in enumerate(fashion_mnist_full) if label == 8 # include only samples with label 8 (Bag)
            # i for i, (_, label) in enumerate(fashion_mnist_full) if label in [8, 9]
            i for i, (_, label) in enumerate(fashion_mnist_full) if label in [0, 9]
        ]
        filtered_dataset = Subset(fashion_mnist_full, indices)

        # Split the filtered dataset into training and validation sets
        train_size = int(self.split_val * len(filtered_dataset))
        val_size = len(filtered_dataset) - train_size
        self.mnist_train, self.mnist_val = random_split(
            filtered_dataset, [train_size, val_size]
        )

        # Load the test dataset and filter it for label 8
        fashion_mnist_test = datasets.FashionMNIST(
            root="./data", train=False, transform=self.transform, download=True
        )
        test_indices = [
            # i for i, (_, label) in enumerate(fashion_mnist_test) if label == 8 # label 8 (bags) only
            # i for i, (_, label) in enumerate(fashion_mnist_test) if label in [8, 9]
            i for i, (_, label) in enumerate(fashion_mnist_test) if label in [0, 9]
        ]
        self.mnist_test = Subset(fashion_mnist_test, test_indices)

    def train_dataloader(self):
        return DataLoader(
            self.mnist_train,
            batch_size=self.batch_size,
            shuffle=True,
            num_workers=self.num_workers,
            persistent_workers=True,
        )

    def val_dataloader(self):
        return DataLoader(
            self.mnist_val,
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=self.num_workers,
            persistent_workers=True,
        )

    def test_dataloader(self):
        return DataLoader(
            self.mnist_test,
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=self.num_workers,
            persistent_workers=True,
        )
