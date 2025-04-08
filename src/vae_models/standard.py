import torch
import torch.nn.functional as F

from src.vae_backbones import ConvParams
from src.vae_models import AbstractVAE


class StandardVAE(AbstractVAE):
    """Standard Variational Autoencoder (VAE) implementation.

    This class implements the original VAE as described in "Auto-Encoding Variational Bayes"
    (Kingma & Welling, 2014). The loss function is the standard ELBO (Evidence Lower Bound)
    which consists of two components:
    1. Reconstruction loss: Measures how well the decoder reconstructs the input data
    2. KL divergence: Regularization term that encourages the learned latent distribution
       to be close to a standard normal distribution N(0,1)

    The loss function is defined as:
        L(θ,φ;x) = E[log p(x|z)] - D_KL(q(z|x) || p(z))
    where:
        - E[log p(x|z)] is the expected log-likelihood (reconstruction term)
        - D_KL(q(z|x) || p(z)) is the KL divergence between the encoder distribution q(z|x)
          and the prior distribution p(z)
        - θ are the decoder parameters
        - φ are the encoder parameters

    The StandardVAE adds a hyperparameter `kl_weight` to control the trade-off between
    reconstruction quality and latent space regularization. When kl_weight=1.0, it's the
    original VAE formulation; higher values emphasize regularization while lower values
    prioritize reconstruction quality.

    Args:
        input_shape (tuple): Shape of the input samples (excluding batch dimension).
            For images, this is typically (C, H, W) or (C, D, H, W) for 3D data.
        latent_dim (int, optional): Dimension of the latent space. Controls the capacity
            and compression level of the model. Default: 20
        backbone_params (ConvParams, optional): Parameters for the backbone architecture.
            Default: ConvParams() (uses default CNN architecture)
        recon_loss_function (function, optional): Function to compute reconstruction loss.
            Default: F.mse_loss (Mean Squared Error loss suitable for continuous data)
        learning_rate (float, optional): Learning rate for the optimizer. Default: 1e-3
        kl_weight (float, optional): Weight applied to the KL divergence term in the loss
            function. Higher values enforce more regularization on the latent space.
            Default: 1.0 (standard VAE formulation)
    """

    def __init__(
        self,
        input_shape,
        latent_dim: int = 20,
        backbone_params=ConvParams(),
        recon_loss_function=F.mse_loss,  # for non-binary input (use F.binary_cross_entropy for binary input)
        learning_rate: float = 1e-3,
        kl_weight: float = 1.0,
    ):
        # Store the KL weight (β in β-VAE terminology) that determines the importance
        # of the KL divergence term relative to the reconstruction term
        self.kl_weight = kl_weight

        # Initialize the parent class with all other parameters
        # The parent class (AbstractVAE) handles setting up the encoder, decoder,
        # and all the network architecture components
        super().__init__(
            input_shape,
            latent_dim,
            backbone_params,
            recon_loss_function,
            learning_rate,
        )

    def loss_function(self, x_recon, x, z_mean, z_logvar):
        """Computes the standard VAE loss function (ELBO) with optional KL weighting.

        This implementation follows the standard Evidence Lower Bound (ELBO) objective:
        L(θ,φ;x) = E[log p(x|z)] - β * D_KL(q(z|x) || p(z))

        The KL divergence between the approximate posterior q(z|x) = N(μ(x), σ²(x))
        and the prior p(z) = N(0, 1) has a closed-form expression:
        D_KL = 0.5 * sum(1 + log(σ²) - μ² - σ²)

        Args:
            x_recon (torch.Tensor): Reconstructed input from the decoder
            x (torch.Tensor): Original input to the encoder
            z_mean (torch.Tensor): Mean vectors of the latent distributions
            z_logvar (torch.Tensor): Log variance vectors of the latent distributions

        Returns:
            torch.Tensor: The total loss value (reconstruction loss + weighted KL divergence)
        """
        # Calculate reconstruction loss with mean reduction
        # This measures how well the model reconstructs the input data
        recon_loss = self.recon_loss_function(x_recon, x, reduction="mean")

        # Calculate KL divergence using the analytical formula for two Gaussians
        # This closed-form expression computes the KL divergence between:
        # q(z|x) = N(μ, σ²) and p(z) = N(0, 1)
        kl_div = -0.5 * torch.mean(1 + z_logvar - z_mean.pow(2) - z_logvar.exp())

        # Combine the two terms with KL weighting to get the total loss
        # When kl_weight=1.0, this is the standard VAE objective
        total_loss = recon_loss + self.kl_weight * kl_div

        return total_loss
