from abc import ABC, abstractmethod

import torch
import torch.nn as nn


class AbstractBackbone(nn.Module, ABC):
    """
    Abstract base class for VAE backbones.

    Ensures that all backbone implementations provide methods for feature extraction
    and sample generation with consistent signatures.
    """

    @abstractmethod
    def extract_features(self, x: torch.Tensor) -> tuple[torch.Tensor, list | None]:
        """
        Extracts features from the input tensor.

        Args:
            x (torch.Tensor): The input tensor (e.g., batch of images).

        Returns:
            tuple[torch.Tensor, list | None]:
                - features (torch.Tensor): The extracted feature tensor (usually flattened).
                - auxiliary_info (list | None): Optional information needed by the generator
                  (e.g., feature maps for skip connections). Return None if not applicable.
        """
        pass

    @abstractmethod
    def generate_sample(
        self, features: torch.Tensor, auxiliary_info: list | None
    ) -> torch.Tensor:
        """
        Generates a sample from features and optional auxiliary information.

        Args:
            features (torch.Tensor): The feature tensor (output of the encoder's mapping).
            auxiliary_info (list | None): Optional information from the feature extractor
              (e.g., skip connections).

        Returns:
            torch.Tensor: The generated sample, matching input dimensions.
        """
        pass
