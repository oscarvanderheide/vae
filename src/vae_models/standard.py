from src.vae_models import AbstractVAE
import torch
import torch.nn.functional as F
from src.vae_backbones import ConvParams


class StandardVAE(AbstractVAE):
    """Standard VAE with loss being the sum of reconstruction loss and KL-divergence.

    The only additional hyperparameter compared to the AbstractVAE is the KL weight,
    which determines the importance of the KL divergence in the loss function.
    """

    def __init__(
        self,
        input_shape,
        latent_dim: int = 20,
        backbone_params=ConvParams(),
        recon_loss_function= F.mse_loss, # for non-binary input # F.binary_cross_entropy, (NOTE: BCE for binary input)
        learning_rate: float = 1e-3,
        kl_weight: float = 1.0,
    ):
        # KL weight is a hyperparameter that determines the importance of the KL divergence
        self.kl_weight = kl_weight
        # All other parameters are passed to the AbstractVAE parent class
        super().__init__(
            input_shape,
            latent_dim,
            backbone_params,
            recon_loss_function,
            learning_rate,
        )

    def loss_function(self, x_recon, x, z_mean, z_logvar):
        """Computes the loss (i.e. recon loss + KL divergence for a standard VAE).

        Uses mean reduction for both the reconstruction loss and the KL divergence.
        """
        # This loss functions default to mean reduction I think but for clarity I specify it anyways
        # recon_loss = F.binary_cross_entropy(x_recon, x, reduction="mean")
        recon_loss = self.recon_loss_function(x_recon, x, reduction="mean")
        # Since the recon loss is calculated using mean, we should probably do the same
        # for the KL divergence loss to prevent them from becoming highly imbalanced.
        kl_div = -0.5 * torch.mean(1 + z_logvar - z_mean.pow(2) - z_logvar.exp())

        return recon_loss + self.kl_weight * kl_div
