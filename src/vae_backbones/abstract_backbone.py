from abc import ABC, abstractmethod

import torch
import torch.nn as nn


class AbstractBackbone(nn.Module, ABC):
    """Abstract base class for VAE backbones (feature extractors/sample generators).

    This class defines the required interface for any network architecture intended
    to serve as the backbone for a Variational Autoencoder (VAE) within this framework.
    It ensures that concrete backbone implementations (like CNNs, MLPs) provide
    standardized methods for encoding inputs to features (`extract_features`) and
    decoding features back to samples (`generate_sample`).

    Inheriting from `nn.Module` makes concrete backbones standard PyTorch modules,
    allowing them to contain learnable parameters and be easily integrated.
    Inheriting from `ABC` (Abstract Base Class) allows defining abstract methods
    that *must* be implemented by subclasses.
    """

    @abstractmethod
    def extract_features(self, x: torch.Tensor) -> tuple[torch.Tensor, list | None]:
        """Extract features (encode) from an input tensor.

        This method defines the "encoding" part of the backbone, mapping the input
        sample (e.g., an MRI image) to a feature representation.
        This feature representation is typically a flattened vector suitable for
        connecting to the VAE's latent space mapping layers.

        Args:
            x (torch.Tensor): The input tensor, typically a batch of samples with shape
              (batch_size, channels, *spatial_dims), e.g., (N, C, H, W) or (N, C, D, H, W).

        Returns:
            tuple[torch.Tensor, list | None]: A tuple containing:
                - features (torch.Tensor): The extracted feature tensor, expected to be
                  2D (batch_size, feature_dimension).
                - auxiliary_info (list | None): Optional extra information generated
                  during encoding that might be needed for decoding. For architectures
                  like U-Nets, this often includes intermediate feature maps (from
                  different spatial resolutions) used for skip connections during
                  decoding. Return `None` if no auxiliary information is produced.
        """
        pass

    @abstractmethod
    def generate_sample(
        self, features: torch.Tensor, auxiliary_info: list | None
    ) -> torch.Tensor:
        """Generate (decode) a sample from a feature tensor and optional auxiliary info.

        This method defines the "decoding" part of the backbone, mapping a feature
        vector (derived from the VAE's latent space) back towards the original
        sample space (e.g., reconstructing an MRI image).

        Args:
            features (torch.Tensor): The feature tensor, typically 2D
              (batch_size, feature_dimension), provided by the VAE's latent space
              to feature mapping layer.
            auxiliary_info (list | None): Optional information passed from the
              `extract_features` method. This is crucial for architectures using
              skip connections (like U-Nets), where intermediate feature maps
              from the encoder are concatenated or added during the decoding process
              to preserve finer details. If `extract_features` returned `None`, this
              will also be `None`.

        Returns:
            torch.Tensor: The generated/reconstructed sample tensor, which should have
              the same shape as the original input samples `x` passed to
              `extract_features` (e.g., (N, C, H, W) or (N, C, D, H, W)).
        """
        pass
