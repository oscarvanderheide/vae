import math
from dataclasses import dataclass

import torch
import torch.nn as nn

from .abstract_backbone import AbstractBackbone


@dataclass
class MLPParams:
    """Holds parameters needed to instantiate MLP-based feature extractor
    and sample generator modules that can form the backbone of a VAE."""

    hidden_dims: tuple = (512, 256, 128, 64)
    activation: nn.Module = nn.SiLU()
    final_activation: nn.Module = nn.Sigmoid()


class MLPBackbone(AbstractBackbone):
    def __init__(self, sample_shape: list, params: MLPParams):
        """Assemble a MLP-based feature extractor and sample generator that can form
        the backbone of a variational auto-encoder.

        Args:
            sample_shape (list): Shape of the input samples (excluding batch dimension).
            params (MLPParams): Parameters specific to the MLP-based networks.

        Returns:
            MLPBackbone: An instance of the MLP backbone module.
        """
        super().__init__()
        self.params = params
        self.sample_shape = sample_shape
        self.numel_sample = math.prod(sample_shape)
        self._build_feature_extractor()
        self._build_sample_generator()

    def _build_feature_extractor(self):
        """Builds the MLP feature extractor layers."""
        p = self.params
        extractor_modules = []

        # Build the MLP network:
        # Flatten the input, then apply a series of linear layers with activations
        for i, hidden_dim in enumerate(p.hidden_dims):
            if i == 0:
                extractor_modules.append(nn.Flatten())
                extractor_modules.append(nn.Linear(self.numel_sample, hidden_dim))
            else:
                extractor_modules.append(p.activation)
                extractor_modules.append(nn.Linear(p.hidden_dims[i - 1], hidden_dim))

        self.feature_extractor = nn.Sequential(*extractor_modules)

    def _build_sample_generator(self):
        """Builds the MLP sample generator layers."""
        p = self.params
        generator_modules = []

        # Build the MLP network:
        # Apply a series of linear layers with activations, then unflatten
        for i in range(len(p.hidden_dims)):
            if i < (len(p.hidden_dims) - 1):
                generator_modules.append(
                    nn.Linear(p.hidden_dims[-(i + 1)], p.hidden_dims[-(i + 2)])
                )
                generator_modules.append(p.activation)
            else:
                # Final layer maps to the flattened sample size
                generator_modules.append(nn.Linear(p.hidden_dims[0], self.numel_sample))
                generator_modules.append(p.final_activation)
                generator_modules.append(nn.Unflatten(1, self.sample_shape))

        self.sample_generator = nn.Sequential(*generator_modules)

    def extract_features(self, x: torch.Tensor) -> tuple[torch.Tensor, None]:
        """Forward pass through the feature extractor. Returns features and None for auxiliary info."""
        features = self.feature_extractor(x)
        # Return None for auxiliary_info to match the expected signature
        return features, None

    def generate_sample(
        self, features: torch.Tensor, auxiliary_info: list | None
    ) -> torch.Tensor:
        """Forward pass through the sample generator. Ignores auxiliary_info."""
        # auxiliary_info is ignored in the MLP case
        return self.sample_generator(features)
