"""Variational Autoencoder (VAE) model implementations.

This module provides a collection of Variational Autoencoder implementations that follow
a common interface defined by the AbstractVAE base class. These implementations allow for
easy experimentation with different VAE variants while maintaining a consistent API.

Available Models:
---------------
- AbstractVAE: Base class that defines the common interface and implements the core VAE
  architecture. This is an abstract class that cannot be instantiated directly.

- StandardVAE: Implementation of the original VAE as described in "Auto-Encoding
  Variational Bayes" (Kingma & Welling, 2014), with the option to adjust the weight
  of the KL divergence term.

Usage:
-----
To use any of the VAE models, import them directly from this module:

```python
from src.vae_models import StandardVAE

# Initialize a standard VAE model
model = StandardVAE(
    input_shape=(3, 64, 64),  # (channels, height, width) for RGB images
    latent_dim=32,            # Dimension of latent space
    kl_weight=1.0,            # Weight for the KL divergence term
)
```

To implement a custom VAE variant, inherit from AbstractVAE and implement
the required loss_function method:

```python
from src.vae_models import AbstractVAE

class MyCustomVAE(AbstractVAE):
    def __init__(self, input_shape, latent_dim=20, **kwargs):
        super().__init__(input_shape, latent_dim, **kwargs)

    def loss_function(self, recon_x, x, z_mean, z_logvar):
        # Implement your custom loss function here
        pass
```

See individual class docstrings for detailed documentation on each model.
"""

from .abstract_vae import AbstractVAE
from .standard import StandardVAE

__all__ = [
    "AbstractVAE",
    "StandardVAE",
]
