# %% Imports

# External libraries
import numpy as np
import matplotlib

# matplotlib.use("Qt5Agg")
import matplotlib.pyplot as plt
import torch
import pytorch_lightning as pl
from lightning.pytorch.loggers import WandbLogger
import torch.nn.functional as F

# Modules from this project
# v1: MNIST
# from datasets.mnist import MNISTDataModule
# v2: FashionMNIST
# from datasets.fashionmnist import FashionMNISTDataModule
# v3: BraTS
# from datasets.brats import BraTSDataModule
# v4: BraTS 2021
from datasets.brats_2021 import BraTSDataModule

# Import the different VAE models (e.g. StandardVAE)
import src.vae_models as models

# Import the different backbones (e.g. convolutional, MLP).
# The backbones don't have classes themselves, instead,
# the parameters for the backbones are imported. Those parameters are passed to
# the VAE model which then initializes the backbone.


import src.vae_backbones as backbones

# Assuming wandb has been initialized
# wandb_logger = WandbLogger(project="MNIST", log_model="all")

torch.set_float32_matmul_precision("medium")

# %% Create datamodule and loaders for MNIST dataset
# v1: MNIST
# datamodule = MNISTDataModule()
# v2: FashionMNIST
# datamodule = FashionMNISTDataModule()
# v3: BraTS
# datamodule = BraTSDataModule("data/BraTS/raw", batch_size=64, num_workers=7)
# v4: BraTS 2021
datamodule = BraTSDataModule(data_dir="/home/aruckert/local_scratch/github/brats_2021_data/", batch_size=64, num_workers=7)

datamodule.setup()
train_loader = datamodule.train_dataloader()
val_loader = datamodule.val_dataloader()
# Determine shape of input images, needed to initalize the model

x, _ = next(iter(train_loader))
input_shape = tuple(x[0].shape)
print(input_shape)

# %% Load model

# There is an AbstractVAE model which is a Pytorch Lightning module and contains
# most of the methods needed for a VAE except for the loss function.

# The StandardVAE is a concrete subclass of AbstractVAE that does contain a loss
# function consisting of a reconstructio loss and KL divergence term. No other VAE
# variants have been implemented at the moment.

# Under the hood, a VAE contains an encoder and a decoder. The encoder first maps
# an input sample to a feature vector. The feature vector is then mapped to a
# mean and (log) variance in latent space through fully-connected layers.
# The decoder takes a sampled latent vector, maps it to a feature vector with a
# fully-connected layer and then generates a sample using a procedure that is
# like the inverse of the feature extractor. The feature extractor and sample generator
# together are referred to as the "backbone" of the VAE.

# Use Convolutional backbone
backbone_params = backbones.ConvParams()
# Alternative: Use MLP backbone
# backbone_params = backbones.MLPParams()

# Set VAE options
kl_weight = 0
latent_dim = 20
learning_rate = 1e-3
recon_loss_function = F.mse_loss # F.binary_cross_entropy

# Initialize the VAE model
model = models.StandardVAE(
    input_shape,
    latent_dim,
    backbone_params,
    recon_loss_function,
    learning_rate,
    kl_weight,
)

# Test whether the model runs on a single batch in forward mode
x, y = next(iter(train_loader))
# print("Shape of input batch", x.shape)
# without skip connections:
# z_mean, z_logvar = model.encoder(x)
# with skip connections:
z_mean, z_logvar, feature_maps = model.encoder(x)
# print("Shape of mean value in latent space", z_mean.shape)
z = model.sample_latent_vec(z_mean, z_logvar)
# without skip connections:
# x_recon = model.decoder(z)
# with skip connections:
x_recon = model.decoder(z, feature_maps)
# print("Shape of reconstructed batch:", x_recon.shape)
assert x_recon.shape == x.shape

# %% Assemble trainer and train

max_epochs = 150
trainer = pl.Trainer(max_epochs=max_epochs) #, logger=wandb_logger)
trainer.fit(model, train_loader, val_loader)


def plot_samples_and_reconstructions_brats(x, x_recon, title="Samples and Reconstructions"):
    """Plot original and reconstructed images in a grid."""

    images = [x[i + j, 1].detach().cpu().numpy() for i in [0, 4, 8, 12] for j in [0, 1]]
    images = np.reshape(images, (4, 2, 240, 240))

    recons = [
        x_recon[i + j, 0].detach().cpu().numpy() for i in [0, 4, 8, 12] for j in [0, 1]
    ]
    recons = np.reshape(recons, (4, 2, 240, 240))
    fig, ax = plt.subplots(4, 4, figsize=(10, 10))
    fig.suptitle(title, fontsize=16)

    for i in range(4):
        for j in range(2):
            row = i
            col_image = 2 * j
            col_recon = 2 * j + 1
            ax[row, col_image].imshow(images[i, j, :, :], cmap="gray")
            ax[row, col_image].set_title("Original", fontsize=10)
            ax[row, col_image].axis("off")
            ax[row, col_recon].imshow(recons[i, j, :, :], cmap="gray")
            ax[row, col_recon].set_title("Reconstructed", fontsize=10)
            ax[row, col_recon].axis("off")

    plt.tight_layout(rect=[0, 0.03, 1, 0.95])  # Adjust layout to fit title
    plt.show()

model.eval()

# Get a batch of input samples and pass through the model
x, _ = next(iter(val_loader))
x_recon, _, _ = model.forward(x)

# BraTS:
plot_samples_and_reconstructions_brats(x, x_recon)


# %% Display some samples and reconstructions on a 4x4 grid


def plot_samples_and_reconstructions(x, x_recon, title=f"Samples and Reconstructions (N_epochs = {max_epochs})"):
    """Plot original and reconstructed images in a grid."""

    images = [
        x[i + j].squeeze(0).detach().cpu().numpy()
        for i in [0, 4, 8, 12]
        for j in [0, 1]
    ]
    images = np.reshape(images, (4, 2, 28, 28))

    recons = [
        x_recon[i + j].squeeze(0).detach().cpu().numpy()
        for i in [0, 4, 8, 12]
        for j in [0, 1]
    ]
    recons = np.reshape(recons, (4, 2, 28, 28))
    fig, ax = plt.subplots(4, 4, figsize=(10, 10))
    fig.suptitle(title, fontsize=16)

    for i in range(4):
        for j in range(2):
            row = i
            col_image = 2 * j
            col_recon = 2 * j + 1
            ax[row, col_image].imshow(images[i, j, :, :], cmap="gray")
            ax[row, col_image].set_title("Original", fontsize=10)
            ax[row, col_image].axis("off")
            ax[row, col_recon].imshow(recons[i, j, :, :], cmap="gray")
            ax[row, col_recon].set_title("Reconstructed", fontsize=10)
            ax[row, col_recon].axis("off")

    plt.tight_layout(rect=[0, 0.03, 1, 0.95])  # Adjust layout to fit title
    plt.show()


def plot_samples_and_reconstructions_brats(x, x_recon, title="Samples and Reconstructions"):
    """Plot original and reconstructed images in a grid."""

    images = [x[i + j, 1].detach().cpu().numpy() for i in [0, 4, 8, 12] for j in [0, 1]]
    images = np.reshape(images, (4, 2, 240, 240))

    recons = [
        x_recon[i + j, 0].detach().cpu().numpy() for i in [0, 4, 8, 12] for j in [0, 1]
    ]
    recons = np.reshape(recons, (4, 2, 240, 240))
    fig, ax = plt.subplots(4, 4, figsize=(10, 10))
    fig.suptitle(title, fontsize=16)

    for i in range(4):
        for j in range(2):
            row = i
            col_image = 2 * j
            col_recon = 2 * j + 1
            ax[row, col_image].imshow(images[i, j, :, :], cmap="gray")
            ax[row, col_image].set_title("Original", fontsize=10)
            ax[row, col_image].axis("off")
            ax[row, col_recon].imshow(recons[i, j, :, :], cmap="gray")
            ax[row, col_recon].set_title("Reconstructed", fontsize=10)
            ax[row, col_recon].axis("off")

    plt.tight_layout(rect=[0, 0.03, 1, 0.95])  # Adjust layout to fit title
    plt.show()

model.eval()

# Get a batch of input samples and pass through the model
x, _ = next(iter(val_loader))
x_recon, _, _ = model.forward(x)

# (Fashion)MNIST:
# plot_samples_and_reconstructions(x, x_recon)
# BraTS:
plot_samples_and_reconstructions_brats(x, x_recon)

# %% Interpolate between four digits
x, _ = next(iter(train_loader))
x1 = x[0].unsqueeze(0)
x2 = x[1].unsqueeze(0)
x3 = x[2].unsqueeze(0)
x4 = x[3].unsqueeze(0)

# Encode the samples
z_mean1, z_logvar1, _ = model.encoder(x1)
z_mean2, z_logvar2, _ = model.encoder(x2)
z_mean3, z_logvar3, _ = model.encoder(x3)
z_mean4, z_logvar4, _ = model.encoder(x4)

# Sample latent vectors
z1 = model.sample_latent_vec(z_mean1, z_logvar1)
z2 = model.sample_latent_vec(z_mean2, z_logvar2)
z3 = model.sample_latent_vec(z_mean3, z_logvar3)
z4 = model.sample_latent_vec(z_mean4, z_logvar4)

# Create a grid of interpolations
n_rows, n_cols = 8, 16
z_interpolated = torch.zeros((n_rows, n_cols, z1.size(1)))

for i in range(n_rows):
    for j in range(n_cols):
        # Interpolate between the four corners
        z_interpolated[i, j] = (
            z1 * (1 - i / (n_rows - 1)) * (1 - j / (n_cols - 1))
            + z2 * (1 - i / (n_rows - 1)) * (j / (n_cols - 1))
            + z3 * (i / (n_rows - 1)) * (1 - j / (n_cols - 1))
            + z4 * (i / (n_rows - 1)) * (j / (n_cols - 1))
        )

# Decode the interpolated latent vectors
x_interpolated = model.decoder(z_interpolated.view(-1, z1.size(1)))

fig, ax = plt.subplots(1, 1, figsize=(n_cols, n_rows))
fig.suptitle("Interpolated between four digits", fontsize=16)

# Concatenate the images into a single large array
large_image = np.concatenate(
    [
        np.concatenate(
            [
                x_interpolated[i * n_cols + j, 0, :, :].detach().numpy()
                for j in range(n_cols)
            ],
            axis=1,
        )
        for i in range(n_rows)
    ],
    axis=0,
)

ax.imshow(large_image, cmap="gray")
ax.axis("off")

plt.show()

# %% Add rectangles to samples

# Take a digit and add a strange rectangle somewhere to see how it deals with such unseen data
x, _ = next(iter(val_loader))

# Modify the images
x[:, 0, 10:25, 10:15] = 1

x_recon, _, _ = model.forward(x)

plot_samples_and_reconstructions(
    x, x_recon, title="Samples and Reconstructions with added rectangles"
)
# %%
