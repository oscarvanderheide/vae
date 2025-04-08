from abc import ABC, abstractmethod

import pytorch_lightning as pl
import torch
import torch.nn as nn

from src.vae_backbones import assemble_backbone
from src.vae_backbones.abstract_backbone import AbstractBackbone


class AbstractVAE(pl.LightningModule, ABC):
    """Abstract base class that provides a starting point for different variations
    of the Variational Autoencoder (VAE).

    This class defines the foundational structure and components required for any VAE
    implementation within this framework. It implements the standard VAE architecture
    consisting of an encoder that maps inputs to a latent distribution, a sampling
    mechanism, and a decoder that reconstructs inputs from the latent samples.

    The class leverages PyTorch Lightning for training functionality and requires
    only the loss function to be implemented in subclasses, which allows for quickly
    creating different VAE variants (standard VAE, beta-VAE, etc.).

    Terminology:
        - Sample: An input sample (e.g., image) that the VAE
          will learn to encode and reconstruct.
        - Feature vector: A different representation of the input sample produced by
          the feature extraction part of the backbone. This is a deterministic intermediate
          representation between the input and the latent space.
        - Backbone: The network architecture used to map samples to feature vectors
          and vice versa. Could be CNN, MLP or Transformer based. The backbone is composed
          of two parts: a feature extractor and a sample generator.
        - Encoder: The part of the VAE that maps samples to distributions in latent space.
          It consists of the feature extractor and linear layers that map features to
          distribution parameters (mean and log-variance).
        - Decoder: The part of the VAE that maps latent space vectors to reconstructed samples.
          It maps a latent vector to a feature vector using a linear layer, and then maps
          the feature vector to a sample using the sample generator.
        - Latent space: The lower-dimensional space where each input is encoded as a
          probability distribution (typically Gaussian). This is where the VAE performs
          its generative modeling.

    Args:
        input_shape (tuple): Shape of the input samples (excluding batch dimension).
            For images, this is typically (C, H, W) or (C, D, H, W) for 3D data.
        latent_dim (int): Dimension of the latent space. Controls the capacity and
            compression level of the model. Typical values range from 2-256 depending
            on the complexity of the data distribution.
        backbone_params: Parameters specific to the backbone networks. These define
            the architecture used for feature extraction and sample generation.
        recon_loss_function: Function that computes the reconstruction loss between
            the original input and the reconstructed output. Common choices include
            MSE for continuous data or BCE for binary data.
        learning_rate (float): Learning rate for the optimizer. Controls the step size
            for parameter updates during training.
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

        # Assemble the backbone of the VAE containing the feature extractor and
        # sample generator components that are accessed as
        # self.backbone.extract_features and self.backbone.generate_sample
        # respectively.
        self.backbone: AbstractBackbone = assemble_backbone(
            input_shape, backbone_params
        )

        # Pass dummy input through encoder to determine the feature dimension
        # This allows the model to automatically adapt to different input shapes
        dummy_input_sample = torch.randn((1,) + input_shape)
        dummy_feature_vec, _ = self.backbone.extract_features(dummy_input_sample)
        # Ensure the feature vector has exactly two dimensions (batch_size, feature_dim)
        assert dummy_feature_vec.ndim == 2
        feature_dim = dummy_feature_vec.shape[1]

        # Create the mapping networks from feature space to latent space distribution parameters
        self.feature_vec_to_z_mean = nn.Linear(feature_dim, latent_dim)
        self.feature_vec_to_z_logvar = nn.Linear(feature_dim, latent_dim)

        # Create the mapping network from latent space back to feature space
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
    ) -> dict:
        """Computes the VAE loss function consisting of reconstruction error and KL divergence.

        This method must be implemented by subclasses to define the specific VAE variant's
        loss function. Typically, the loss is a weighted sum of:
        1. Reconstruction error: Measures how well the decoder reconstructs the input
        2. KL divergence: Measures how much the learned latent distribution deviates
           from a predefined prior distribution (often a standard normal distribution)

        Different VAE variants use different weighting strategies between these two components.

        Args:
            recon_x (torch.Tensor): Reconstructed input tensor. Should match the shape
                of the original input x.
            x (torch.Tensor): Original input tensor.
            z_mean (torch.Tensor): Mean of the latent distribution produced by the encoder.
                Shape: (batch_size, latent_dim)
            z_logvar (torch.Tensor): Log variance of the latent distribution produced
                by the encoder. Shape: (batch_size, latent_dim)

        Returns:
            dict: A dictionary containing at minimum the 'loss' key with the total loss value,
                and optionally other metrics to log during training.
        """
        pass

    def forward(
        self, x: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Performs the complete forward pass through the VAE.

        This method implements the standard VAE pipeline:
        1. Encode the input to distribution parameters in latent space
        2. Sample a latent vector from that distribution using the reparameterization trick
        3. Decode the latent vector to reconstruct the input

        Args:
            x (torch.Tensor): Input tensor with shape (batch_size, *input_shape)

        Returns:
            tuple[torch.Tensor, torch.Tensor, torch.Tensor]: A tuple containing:
                - recon_x (torch.Tensor): Reconstructed input with same shape as x
                - z_mean (torch.Tensor): Mean vectors of the latent distributions
                - z_logvar (torch.Tensor): Log variance vectors of the latent distributions
        """
        z_mean, z_logvar, auxiliary_info = self.encoder(x)
        z = self.sample_latent_vec(z_mean, z_logvar)
        recon_x = self.decoder(z, auxiliary_info)
        return recon_x, z_mean, z_logvar

    def encoder(
        self, x: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor, list | None]:
        """Maps input samples to latent space distribution parameters.

        The encoder pipeline consists of:
        1. Feature extraction using the backbone network
        2. Mapping the extracted features to the mean vector of the latent distribution
        3. Mapping the extracted features to the log variance vector of the latent distribution

        Args:
            x (torch.Tensor): Input tensor with shape (batch_size, *input_shape)

        Returns:
            tuple[torch.Tensor, torch.Tensor, list | None]: A tuple containing:
                - z_mean (torch.Tensor): Mean vectors of shape (batch_size, latent_dim)
                - z_logvar (torch.Tensor): Log variance vectors of shape (batch_size, latent_dim)
                - auxiliary_info (list | None): Auxiliary information from the backbone,
                  typically intermediate feature maps for skip connections
        """
        # Extract features from input using the backbone's feature extractor
        features, auxiliary_info = self.backbone.extract_features(x)

        # Map features to latent distribution parameters
        z_mean = self.feature_vec_to_z_mean(features)
        z_logvar = self.feature_vec_to_z_logvar(features)

        return z_mean, z_logvar, auxiliary_info

    def sample_latent_vec(
        self, z_mean: torch.Tensor, z_logvar: torch.Tensor
    ) -> torch.Tensor:
        """Samples latent vectors from the encoded distribution using the reparameterization trick.

        The reparameterization trick enables backpropagation through the sampling process
        by expressing the random sampling as a deterministic function of the distribution
        parameters and an auxiliary noise variable:
        z = μ + σ ⊙ ε where:
        - μ is the mean vector
        - σ is the standard deviation (calculated as exp(0.5 * log_variance))
        - ε is a random noise vector sampled from a standard normal distribution
        - ⊙ represents element-wise multiplication

        Args:
            z_mean (torch.Tensor): Mean vectors of shape (batch_size, latent_dim)
            z_logvar (torch.Tensor): Log variance vectors of shape (batch_size, latent_dim)

        Returns:
            torch.Tensor: Sampled latent vectors of shape (batch_size, latent_dim)
        """
        # Convert log variance to standard deviation
        std = torch.exp(0.5 * z_logvar)

        # Sample noise from standard normal distribution
        eps = torch.randn_like(std)

        # Apply reparameterization trick
        z = z_mean + eps * std

        return z

    def decoder(self, z: torch.Tensor, auxiliary_info: list | None) -> torch.Tensor:
        """Reconstructs input samples from latent vectors and optional auxiliary information.

        The decoder pipeline consists of:
        1. Mapping the latent vector to a feature vector
        2. Generating a reconstructed sample from the feature vector using the backbone

        Args:
            z (torch.Tensor): Latent vectors of shape (batch_size, latent_dim)
            auxiliary_info (list | None): Auxiliary information from the encoder,
                typically intermediate feature maps for skip connections in U-Net-like
                architectures, or None if not using skip connections

        Returns:
            torch.Tensor: Reconstructed input samples with shape (batch_size, *input_shape)
        """
        # Map latent vector to feature vector
        features = self.latent_vec_to_feature_vec(z)

        # Generate reconstructed sample from feature vector
        recon_x = self.backbone.generate_sample(features, auxiliary_info)

        return recon_x

    def training_step(self, batch: torch.Tensor, batch_idx: int) -> torch.Tensor:
        """Performs a single training step as defined in PyTorch Lightning.

        This method:
        1. Extracts the input from the batch
        2. Performs a forward pass through the VAE
        3. Calculates the loss
        4. Logs metrics for monitoring training progress

        Args:
            batch (torch.Tensor): A batch of data, typically (inputs, targets) or just inputs
            batch_idx (int): Index of the current batch

        Returns:
            torch.Tensor: The computed loss value for this training step
        """
        x, _ = batch  # Extract input (assuming batch contains inputs and targets)
        recon_x, z_mean, z_logvar = self.forward(x)
        loss = self.loss_function(recon_x, x, z_mean, z_logvar)
        self.log("train_loss", loss, prog_bar=True)

        return loss

    def validation_step(self, batch: torch.Tensor, batch_idx: int) -> torch.Tensor:
        """Performs a single validation step as defined in PyTorch Lightning.

        Similar to training_step but used for validation data. Results are used
        to monitor overfitting and potentially adjust the learning rate.

        Args:
            batch (torch.Tensor): A batch of validation data
            batch_idx (int): Index of the current batch

        Returns:
            torch.Tensor: The computed loss value for this validation step
        """
        x, _ = batch
        recon_x, z_mean, z_logvar = self.forward(x)
        loss = self.loss_function(recon_x, x, z_mean, z_logvar)
        self.log("val_loss", loss, prog_bar=True)
        return loss

    def configure_optimizers(self):
        """Configures the optimizer and learning rate scheduler for training.

        This method is called by PyTorch Lightning to set up the optimization process.
        It creates:
        1. An Adam optimizer with the specified learning rate
        2. A ReduceLROnPlateau scheduler that reduces the learning rate when validation
           loss plateaus, which helps with convergence and avoiding local minima

        Returns:
            dict: A dictionary containing the optimizer and learning rate scheduler configuration
        """
        # Create the optimizer (Adam is commonly used for VAEs due to its adaptive learning rates)
        optimizer = torch.optim.Adam(self.parameters(), lr=self.learning_rate)

        # Configure the learning rate scheduler to reduce LR when validation performance plateaus
        scheduler = {
            "scheduler": torch.optim.lr_scheduler.ReduceLROnPlateau(
                optimizer, mode="min", patience=5, factor=0.5
            ),
            "monitor": "val_loss",  # Reduce LR when val_loss stops improving
            "interval": "epoch",  # Check after each epoch
            "frequency": 1,
        }

        return {"optimizer": optimizer, "lr_scheduler": scheduler}
