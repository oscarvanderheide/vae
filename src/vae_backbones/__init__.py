"""Neural network backbone architectures for Variational Autoencoders.

This module provides different backbone architectures that can be used within the VAE
framework to encode inputs into feature vectors and decode feature vectors back into
reconstructed samples. The backbone is responsible for the actual feature extraction
and sample generation, while the VAE handles the probabilistic mapping to and from
latent space.

Available Backbones:
------------------
- ConvBackbone: Convolutional neural network backbone suitable for structured data
  like images or 3D volumes. Supports various CNN architectures with configurable
  parameters like number of layers, channels, kernel sizes, etc. Implements skip
  connections (U-Net style) when specified.

- MLPBackbone: Multi-layer perceptron (fully connected) backbone suitable for
  non-structured data or as a simpler alternative for smaller images.

Each backbone implementation comes with a corresponding parameters class:
- ConvParams: Configuration parameters for convolutional backbones
- MLPParams: Configuration parameters for MLP backbones

Usage:
-----
The recommended way to create a backbone is through the 'assemble_backbone' function,
which automatically selects the appropriate backbone class based on the provided
parameters:

```python
from src.vae_backbones import assemble_backbone, ConvParams

# Configure a CNN backbone with specific parameters
backbone_params = ConvParams(
    hidden_dims=[32, 64, 128],  # Channel dimensions for CNN layers
    use_skip_connections=True    # Enable U-Net style skip connections
)

# Create the backbone for 3-channel 64x64 images
backbone = assemble_backbone(
    sample_shape=(3, 64, 64),  # (channels, height, width)
    params=backbone_params
)
```

Alternatively, backbone classes can be instantiated directly:

```python
from src.vae_backbones import ConvBackbone, ConvParams

backbone_params = ConvParams(...)
backbone = ConvBackbone(sample_shape=(3, 64, 64), params=backbone_params)
```

Custom Backbones:
---------------
To implement a custom backbone, inherit from the AbstractBackbone class and implement
the required methods:
- extract_features: Map input samples to feature vectors
- generate_sample: Map feature vectors back to reconstructed samples

Then, extend the assemble_backbone function to recognize your custom parameters.

See individual class docstrings for detailed documentation on each backbone implementation.
"""

from .conv import ConvBackbone, ConvParams
from .mlp import MLPBackbone, MLPParams

__all__ = [
    "MLPParams",
    "ConvParams",
    "MLPBackbone",
    "ConvBackbone",
    "assemble_backbone",
]


def assemble_backbone(sample_shape: list, params: object) -> object:
    """Creates the appropriate backbone based on the provided parameters.

    This factory function examines the type of parameters provided and instantiates
    the corresponding backbone architecture. It also performs validation on the
    sample shape to ensure it meets the requirements of the backbone implementations.

    Args:
        sample_shape (list or tuple): Shape of the input samples (excluding batch dimension).
            Expected format: (channels, *spatial_dims), e.g., (C, H, W) for images or
            (C, D, H, W) for 3D volumes.
        params (object): Configuration parameters for the backbone. Must be an instance of
            either MLPParams or ConvParams, which determines which backbone type is created.

    Returns:
        AbstractBackbone: An initialized backbone instance ready to be used within a VAE.

    Raises:
        TypeError: If sample_shape is not a list or tuple, or if params is not a recognized
            parameter type.
        ValueError: If sample_shape has fewer than 2 dimensions (must have at least channels
            and one spatial dimension) or more than 4 dimensions (channels + 3 spatial dims).
    """
    # Validate input
    if not isinstance(sample_shape, (list, tuple)):
        raise TypeError("sample_shape must be a list or tuple.")

    if len(sample_shape) < 2:
        raise ValueError(
            "sample_shape must have at least two dimensions (channels and spatial)."
        )
    if len(sample_shape) > 4:
        raise ValueError(
            "sample_shape can have at most four dimensions (channels, height, width, depth)."
        )

    # Dispatch to the appropriate assembly backbone module
    if isinstance(params, MLPParams):
        return MLPBackbone(sample_shape, params)
    elif isinstance(params, ConvParams):
        return ConvBackbone(sample_shape, params)
    else:
        raise TypeError("params must be an instance of MLPParams or ConvParams.")
