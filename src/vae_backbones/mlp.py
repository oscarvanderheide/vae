import math
from dataclasses import dataclass

import torch
import torch.nn as nn

from .abstract_backbone import AbstractBackbone


@dataclass
class MLPParams:
    """Configuration parameters for building a Multi-Layer Perceptron (MLP) backbone.

    This dataclass holds parameters for defining a simple MLP architecture suitable
    for VAEs, particularly when dealing with non-spatial or flattened input data.
    It defines the hidden layer dimensions and activation functions.

    Attributes:
        hidden_dims (tuple): A tuple of integers specifying the number of neurons
            in each hidden layer of the MLP, for both the extractor and generator.
            The sequence defines the extractor's layer sizes; the generator uses
            the reverse sequence. Default: (512, 256, 128, 64).
        activation (nn.Module): The PyTorch activation function class instance to use
            between hidden layers (e.g., `nn.ReLU()`, `nn.SiLU()`). Default: `nn.SiLU()`.
        final_activation (nn.Module): The activation function applied to the final output
            of the sample generator (decoder) to scale the reconstructed sample to the
            desired range (e.g., `nn.Sigmoid()` for [0, 1], `nn.Tanh()` for [-1, 1],
            or `nn.Identity()` for no scaling). Default: `nn.Sigmoid()`.
    """

    hidden_dims: tuple = (512, 256, 128, 64)
    activation: nn.Module = nn.SiLU()
    final_activation: nn.Module = nn.Sigmoid()


class MLPBackbone(AbstractBackbone):
    def __init__(self, sample_shape: tuple, params: MLPParams):
        """Initializes and builds the MLP Backbone.

        Constructs the feature extractor (encoder) and sample generator (decoder)
        networks based on a simple MLP architecture. Input samples are flattened
        before being processed by the extractor.

        Args:
            sample_shape (tuple): The shape of a single input sample, excluding the batch
                dimension (e.g., (C, H, W) or (Features,)). Used to calculate the
                flattened input size.
            params (MLPParams): An instance of `MLPParams` containing the configuration
                for the hidden layers and activations.
        """
        super().__init__()
        self.params = params
        self.sample_shape = sample_shape
        # Calculate the total number of elements in a single sample after flattening.
        self.numel_sample = math.prod(sample_shape)

        # Build the network components
        self._build_feature_extractor()  # Encoder part
        self._build_sample_generator()  # Decoder part

    def _build_feature_extractor(self):
        """Builds the MLP feature extractor (encoder) network."""
        p = self.params
        extractor_modules = []

        # Start with flattening the input sample (e.g., image) into a 1D vector
        extractor_modules.append(nn.Flatten())

        # Determine the size of the first layer's input (flattened sample size)
        in_features = self.numel_sample

        # Sequentially add Linear layers and Activation functions
        for i, hidden_dim in enumerate(p.hidden_dims):
            # Add activation function *before* the linear layer (except for the very first layer)
            if i > 0:
                extractor_modules.append(p.activation)

            # Add the linear layer
            extractor_modules.append(nn.Linear(in_features, hidden_dim))

            # Update in_features for the next layer
            in_features = hidden_dim

        # Combine all layers into a sequential module
        self.feature_extractor = nn.Sequential(*extractor_modules)

    def _build_sample_generator(self):
        """Builds the MLP sample generator (decoder) network."""
        p = self.params
        generator_modules = []

        # Input features for the generator start from the last hidden dim of the extractor
        in_features = p.hidden_dims[-1]

        # Sequentially add Linear layers and Activation functions in reverse order of hidden_dims
        for i in range(len(p.hidden_dims) - 1):
            # Add activation function before the linear layer
            generator_modules.append(p.activation)

            # Output features for this layer are the previous hidden dim size
            out_features = p.hidden_dims[-(i + 2)]
            generator_modules.append(nn.Linear(in_features, out_features))

            # Update in_features for the next layer
            in_features = out_features

        # --- Final Layer ---
        # Add activation before the final linear layer
        generator_modules.append(p.activation)
        # Final linear layer maps back to the flattened original sample size
        generator_modules.append(nn.Linear(in_features, self.numel_sample))
        # Apply the final scaling activation (e.g., Sigmoid)
        generator_modules.append(p.final_activation)
        # Unflatten the output vector back to the original sample shape
        generator_modules.append(
            nn.Unflatten(dim=1, unflattened_size=self.sample_shape)
        )

        # Combine all layers into a sequential module
        self.sample_generator = nn.Sequential(*generator_modules)

    def extract_features(self, x: torch.Tensor) -> tuple[torch.Tensor, None]:
        """Performs the forward pass through the MLP feature extractor (encoder).

        Args:
            x (torch.Tensor): Input tensor (batch of samples).

        Returns:
            tuple[torch.Tensor, None]:
                - features (torch.Tensor): Flattened feature tensor (batch_size, feature_dim).
                - None: MLP backbone does not produce auxiliary info for skip connections.
        """
        features = self.feature_extractor(x)
        # MLP doesn't use skip connections, so auxiliary_info is None
        return features, None

    def generate_sample(
        self, features: torch.Tensor, auxiliary_info: list | None
    ) -> torch.Tensor:
        """Performs the forward pass through the MLP sample generator (decoder).

        Args:
            features (torch.Tensor): Input feature tensor (batch_size, feature_dim).
            auxiliary_info (list | None): Ignored in the MLP backbone. Kept for
              interface consistency with AbstractBackbone.

        Returns:
            torch.Tensor: Reconstructed sample tensor.
        """
        # auxiliary_info is ignored as MLP doesn't use skip connections.
        return self.sample_generator(features)
