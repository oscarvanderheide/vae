from .mlp import MLPParams, _assemble_mlp_backbone
from .conv import ConvParams, _assemble_conv_backbone

__all__ = [
    "MLPParams",
    "ConvParams",
    "assemble_backbone",
]


# Dispatcher function
def assemble_backbone(sample_shape, params):
    """Dispatch to the correct _assembler function based on params type."""
    if isinstance(params, MLPParams):
        return _assemble_mlp_backbone(sample_shape, params)
    elif isinstance(params, ConvParams):
        return _assemble_conv_backbone(sample_shape, params)
    else:
        raise TypeError(f"Unknown parameter type: {type(params)}")
