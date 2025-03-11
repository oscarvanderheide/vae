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
            glob.glob(os.path.join(data_dir, "volume_*_slice_80.h5")))[0:47]

    def __len__(self):
        return len(self.file_paths)

    def __getitem__(self, idx):
        img_path = self.file_paths[idx]
        with h5py.File(img_path, "r") as f:
            img = f["image"][()]
        # Move channel dimension to first axis
        img = np.moveaxis(img, -1, 0)

        # Normalize per channel using z-score normalization with 5th percentile
        for c in range(img.shape[0]):
            channel = img[c,:,:]

            # Get non-zero values
            non_zero_indices = np.where(channel > 0)
            non_zero_values = channel[non_zero_indices]

            if len(non_zero_values) > 0:
                # Get 5th percentile of non-zero values
                p5 = np.percentile(non_zero_values, 0.05)
                
                # Get values above 5th percentile
                valid_indices = np.where(channel > p5)
                valid_values = channel[valid_indices]

                # Compute mean and std of values above 5th percentile
                mean = np.mean(valid_values)
                std = np.std(valid_values)

                # Set values below 5th percentile to 0
                channel[channel <= p5] = 0

                # Apply normalization to values above 5th percentile
                channel[valid_indices] = (valid_values - mean) / (std + 1e-8)
                


        # Select only first channel (T1W) while keeping channel dim
        img = img[0:1, :, :]

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
