import os
import glob
import torch
import numpy as np
import h5py
from torchvision import transforms
from torch.utils.data import Dataset, DataLoader, random_split
import pytorch_lightning as pl


class BraTSDataset(Dataset):
    """BraTS dataset: four contrast images (T1, T1c, T2, FLAIR) per slice"""

    def __init__(self, data_dir, transform=None):
        self.data_dir = data_dir
        self.transform = transform
        # self.file_paths = sorted(
        #     glob.glob(os.path.join(data_dir, "volume_*_slice_*.h5"))
        # )
        self.file_paths = sorted(
            glob.glob(os.path.join(data_dir, "volume_*_slice_80.h5"))
        )

    def __len__(self):
        return len(self.file_paths)

    def __getitem__(self, idx):
        img_path = self.file_paths[idx]
        with h5py.File(img_path, "r") as f:
            img = f["image"][()]
        # Move channel dimension to first axis
        img = np.moveaxis(img, -1, 0)

        # Normalize per channel
        img = (img - img.min(axis=(1, 2), keepdims=True)) / (
            img.max(axis=(1, 2), keepdims=True)
            - img.min(axis=(1, 2), keepdims=True)
            + 1e-8
        )

        if self.transform:
            img = self.transform(img)

        label = -1  # won't be used
        return torch.tensor(img, dtype=torch.float32), label


class BraTSDataModule(pl.LightningDataModule):
    def __init__(
        self,
        data_dir,
        batch_size=128,
        split_val=0.8,
        num_workers=7,
    ):
        super().__init__()
        self.data_dir = data_dir
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
        brats_full = BraTSDataset(self.data_dir)
        # Split the dataset into training and validation sets
        print(len(brats_full))
        train_size = int(self.split_val * len(brats_full))
        val_size = len(brats_full) - train_size
        self.brats_train, self.brats_val = random_split(
            brats_full, [train_size, val_size]
        )

    def train_dataloader(self):
        return DataLoader(
            self.brats_train,
            batch_size=self.batch_size,
            shuffle=True,
            num_workers=self.num_workers,
            persistent_workers=True,
        )

    def val_dataloader(self):
        return DataLoader(
            self.brats_val,
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=self.num_workers,
            persistent_workers=True,
        )

    def test_dataloader(self):
        return DataLoader(
            self.brats_test,
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=self.num_workers,
            persistent_workers=True,
        )
