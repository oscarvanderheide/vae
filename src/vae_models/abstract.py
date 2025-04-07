from abc import ABC, abstractmethod

import pytorch_lightning as pl
import torch
import torch.nn as nn

from src.vae_backbones import assemble_backbone


class AbstractVAE(pl.LightningModule, ABC):
    """
    Abstract class that provides a starting point for different variations
    of the Variational Autoencoder (VAE).

    Terminology:
    - Sample: An input sample (e.g. image, audio clip)
    - Feature vector: A vector of features extracted from the sample. In case of
    images, this could be the flattened output of a CNN encoder. Note that the
    feature vector is not the same as the latent space vector.
    - Backbone: The network architecture that is used to map samples to feature vectors
    and vice versa. Could be CNN, MLP or Transformer based. The backbone is composed
    of two parts: a feature extractor (stored as `sample_to_feature_vec`)
    and a sample generator (stored as `feature_vec_to_sample`).
    - Encoder: The part of the VAE that maps samples to the latent space by first
    extracting features and then mapping these features to a mean and (log)variance
    in latent space using linear layers.
    - Decoder: The part of the VAE that maps latent space vectors to samples. The latent
    vector (obtained from sampling from a distribution in latent space) is first mapped
    to a feature vector using a linear layer, and then the feature vector is mapped to
    a sample using the sample generator.

    This abstract class in principle only requires the definition of the loss
    function to be implemented in a subclass. The loss function should be a function
    of the reconstruction loss and the KL divergence. The forward pass is also
    implemented in the abstract class, but the backbone must be assembled
    elsewhere and passed to the VAE constructor.

    Args:
        input_shape (tuple): Shape of the input samples (excluding batch dimension).
        latent_dim (int): Dimension of the latent space.
        backbone_params: Parameters specific to the backbone networks.
        recon_loss_function: Function that computes the reconstruction loss.
        learning_rate (float): Learning rate for the optimizer.
    """

    def __init__(
        self,
        input_shape: tuple,
        latent_dim: int,
        backbone_params,
        recon_loss_function,
        learning_rate: float,
    ):
        super().__init__()
        self.save_hyperparameters(ignore=["recon_loss_function"])

        # Assemble the backbone of the VAE
        backbone = assemble_backbone(input_shape, backbone_params)
        self.sample_to_feature_vec = backbone.extract_features
        self.feature_vec_to_sample = backbone.generate_sample

        # Pass dummy input through encoder to get feature dimension
        dummy_input_sample = torch.randn((1,) + input_shape)
        dummy_feature_vec, _ = self.sample_to_feature_vec(dummy_input_sample)
        # Check that the feature vector has two dimensions only (batch_size, feature_dim)
        assert dummy_feature_vec.ndim == 2
        feature_dim = dummy_feature_vec.shape[1]

        # Assemble linear layers that map feature vector to mean and (log) variance
        # of the latent space
        self.feature_vec_to_z_mean = nn.Linear(feature_dim, latent_dim)
        self.feature_vec_to_z_logvar = nn.Linear(feature_dim, latent_dim)

        # Assemble linear that maps latent space vector to feature vector
        self.latent_vec_to_feature_vec = nn.Linear(latent_dim, feature_dim)

        self.recon_loss_function = recon_loss_function
        self.learning_rate = learning_rate

    @abstractmethod
    def loss_function(
        self,
        recon_x: torch.Tensor,
        x: torch.Tensor,
        z_mean: torch.Tensor,
        z_logvar: torch.Tensor,
    ) -> float:
        """Computes the loss (i.e. recon loss + KL divergence for a standard VAE)."""
        pass

    def forward(
        self, x: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Forward pass through the VAE."""
        z_mean, z_logvar, auxiliary_info = self.encoder(x)
        z = self.sample_latent_vec(z_mean, z_logvar)
        recon_x = self.decoder(z, auxiliary_info)
        return recon_x, z_mean, z_logvar

    def encoder(
        self, x: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor, list | None]:
        """Map input sample to latent space distribution parameters and auxiliary info."""
        features, auxiliary_info = self.sample_to_feature_vec(x)
        z_mean = self.feature_vec_to_z_mean(features)
        z_logvar = self.feature_vec_to_z_logvar(features)
        return z_mean, z_logvar, auxiliary_info

    def sample_latent_vec(
        self, z_mean: torch.Tensor, z_logvar: torch.Tensor
    ) -> torch.Tensor:
        """Sample from the latent space using the reparametrization trick."""
        std = torch.exp(0.5 * z_logvar)
        eps = torch.randn_like(std)
        z = z_mean + eps * std
        return z

    def decoder(self, z: torch.Tensor, auxiliary_info: list | None) -> torch.Tensor:
        """Map latent space vector to a sample in input space using auxiliary info."""
        features = self.latent_vec_to_feature_vec(z)
        recon_x = self.feature_vec_to_sample(features, auxiliary_info)
        return recon_x

    def training_step(self, batch: torch.Tensor, batch_idx: int) -> float:
        """Perform a single training step."""
        x, _ = batch
        recon_x, z_mean, z_logvar = self.forward(x)
        loss = self.loss_function(recon_x, x, z_mean, z_logvar)
        self.log("train_loss", loss, prog_bar=True)

        return loss

    def validation_step(self, batch: torch.Tensor, batch_idx: int) -> float:
        """Perform a single validation step."""
        x, _ = batch
        recon_x, z_mean, z_logvar = self.forward(x)
        loss = self.loss_function(recon_x, x, z_mean, z_logvar)
        self.log("val_loss", loss, prog_bar=True)
        return loss

    def configure_optimizers(self):
        """Configure the optimizer."""
        optimizer = torch.optim.Adam(self.parameters(), lr=self.learning_rate)
        scheduler = {
            "scheduler": torch.optim.lr_scheduler.ReduceLROnPlateau(
                optimizer, mode="min", patience=5, factor=0.5
            ),
            "monitor": "val_loss",  # Reduce LR when val_loss stops improving
            "interval": "epoch",  # Check after each epoch
            "frequency": 1,
        }
        return {"optimizer": optimizer, "lr_scheduler": scheduler}
