from .conv import ConvBackbone, ConvParams
from .mlp import MLPBackbone, MLPParams

__all__ = [
    "MLPParams",
    "ConvParams",
    "MLPBackbone",
    "ConvBackbone",
    "assemble_backbone",
]


# Dispatcher function
def assemble_backbone(sample_shape: list, params: object) -> object:
    """Dispatch to the correct backbone assembly function based on the type of params.
    Args:
        sample_shape (list): Shape of the input samples (excluding batch dimension).
        params: Parameters specific to the backbone networks.
    Returns:
        Backbone: An instance of the backbone module (MLP or Conv).
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
