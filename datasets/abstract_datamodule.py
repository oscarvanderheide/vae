import pytorch_lightning as pl
from torch.utils.data import DataLoader


class AbstractDataModule(pl.LightningDataModule):
    """Abstract base class for all dataset modules used in the VAE framework.

    This class implements common functionality shared across all dataset modules,
    particularly the dataloader creation methods, to avoid code duplication. Specific
    dataset implementations need only implement the `setup` method and properly
    set the training, validation and test datasets.

    Args:
        batch_size (int, optional): Number of samples per batch. Default: 128
        num_workers (int, optional): Number of subprocesses to use for data loading.
            Default: 7
    """

    def __init__(
        self,
        batch_size=128,
        num_workers=7,
    ):
        super().__init__()
        self.batch_size = batch_size
        self.num_workers = num_workers

    def _create_dataloader(self, dataset, shuffle=False):
        """Helper method to create a dataloader with standard parameters.

        Args:
            dataset: PyTorch dataset to create a dataloader for
            shuffle (bool, optional): Whether to shuffle the data. Default: False

        Returns:
            DataLoader: Configured PyTorch DataLoader
        """
        return DataLoader(
            dataset,
            batch_size=self.batch_size,
            shuffle=shuffle,
            num_workers=self.num_workers,
            persistent_workers=True if self.num_workers > 0 else False,
        )

    def train_dataloader(self):
        """Creates the training dataloader with shuffling enabled.

        Returns:
            DataLoader: DataLoader for the training dataset
        """
        return self._create_dataloader(self.train_dataset, shuffle=True)

    def val_dataloader(self):
        """Creates the validation dataloader.

        Returns:
            DataLoader: DataLoader for the validation dataset
        """
        return self._create_dataloader(self.val_dataset)

    def test_dataloader(self):
        """Creates the test dataloader.

        Returns:
            DataLoader: DataLoader for the test dataset
        """
        return self._create_dataloader(self.test_dataset)
