"""Utilities for Hydra configuration."""

import torch.nn.functional as F
from omegaconf import OmegaConf


def get_loss_function(name: str) -> callable:
    """Get loss function by name with optional parameters.

    Supported loss types:
    - mse: Mean squared error loss
    - bce: Binary cross entropy loss (without logits)
    - bce_logits: Binary cross entropy with logits loss
    - l1: L1 loss (mean absolute error)
    - l1_l2: Combination of L1 and L2 losses (with equal weight)
    """
    loss_functions = {
        "mse": F.mse_loss,
        "bce": F.binary_cross_entropy,
        "bce_logits": F.binary_cross_entropy_with_logits,
        "l1": F.l1_loss,
        "l2": F.mse_loss,
        "l1_l2": lambda x, y: F.l1_loss(x, y) + F.mse_loss(x, y),
    }

    # Return the requested loss function
    if name in loss_functions:
        return loss_functions[name]
    else:
        raise ValueError(f"Unknown loss function: {name}")


def set_custom_resolvers() -> None:
    """Register custom resolvers for Hydra configuration."""
    # Register resolver that maps loss function names to actual functions
    OmegaConf.register_new_resolver("loss_func", get_loss_function)

    # Also register a custom resolver for converting lists in yaml files to tuples
    OmegaConf.register_new_resolver("as_tuple", lambda *args: tuple(args))

    return None
