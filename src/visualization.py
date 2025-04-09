"""Visualization utilities for VAE models."""

import os

import matplotlib.pyplot as plt
import numpy as np
import torch


def plot_reconstructions(model, datamodule, output_dir: str):
    """Visualize original vs reconstructed samples."""
    model.eval()

    print("Plotting")

    # Get batch of validation data
    val_loader = datamodule.val_dataloader()
    x, _ = next(iter(val_loader))

    # Generate reconstructions
    with torch.no_grad():
        x_recon, _, _ = model(x)

    # Create directory if it doesn't exist
    os.makedirs(output_dir, exist_ok=True)

    # Plot original vs reconstructed samples
    fig, axes = plt.subplots(4, 4, figsize=(10, 10))
    fig.suptitle("Samples and Reconstructions", fontsize=16)

    for i in range(8):
        idx = i * 2
        # Original
        axes[i // 2, (i % 2) * 2].imshow(x[idx, 0].cpu().numpy(), cmap="gray")
        axes[i // 2, (i % 2) * 2].axis("off")
        axes[i // 2, (i % 2) * 2].set_title("Original")

        # Reconstruction
        axes[i // 2, (i % 2) * 2 + 1].imshow(x_recon[idx, 0].cpu().numpy(), cmap="gray")
        axes[i // 2, (i % 2) * 2 + 1].axis("off")
        axes[i // 2, (i % 2) * 2 + 1].set_title("Reconstructed")

    plt.tight_layout(rect=[0, 0.03, 1, 0.95])
    plt.savefig(os.path.join(output_dir, "reconstructions.png"))
    plt.show()
    print("Reconstructions saved to", os.path.join(output_dir, "reconstructions.png"))


def plot_latent_space_interpolations(model, datamodule, output_dir: str):
    """Visualize latent space interpolations."""
    model.eval()

    # Get batch of data
    val_loader = datamodule.val_dataloader()
    x, _ = next(iter(val_loader))

    # Get 4 samples for corners of interpolation grid
    x1, x2, x3, x4 = x[0:1], x[1:2], x[2:3], x[3:4]

    with torch.no_grad():
        # Encode samples to latent space
        z_mean1, z_logvar1, aux1 = model.encoder(x1)
        z_mean2, z_logvar2, aux2 = model.encoder(x2)
        z_mean3, z_logvar3, aux3 = model.encoder(x3)
        z_mean4, z_logvar4, aux4 = model.encoder(x4)

        # Sample latent vectors
        z1 = model.sample_latent_vec(z_mean1, z_logvar1)
        z2 = model.sample_latent_vec(z_mean2, z_logvar2)
        z3 = model.sample_latent_vec(z_mean3, z_logvar3)
        z4 = model.sample_latent_vec(z_mean4, z_logvar4)

    # Create grid of interpolated latent vectors
    n_rows, n_cols = 8, 16
    z_interp = torch.zeros((n_rows * n_cols, z1.size(1)))

    idx = 0
    for i in range(n_rows):
        for j in range(n_cols):
            # Bilinear interpolation between 4 corner vectors
            t1 = i / (n_rows - 1)
            t2 = j / (n_cols - 1)
            z = (
                (1 - t1) * (1 - t2) * z1
                + t1 * (1 - t2) * z3
                + (1 - t1) * t2 * z2
                + t1 * t2 * z4
            )
            z_interp[idx] = z
            idx += 1

    # Decode interpolated vectors
    with torch.no_grad():
        x_interp = model.decoder(z_interp, None)

    # Create a single large image showing the interpolations
    large_image = np.zeros((n_rows * 28, n_cols * 28))

    for i in range(n_rows):
        for j in range(n_cols):
            idx = i * n_cols + j
            img = x_interp[idx, 0].cpu().numpy()
            large_image[i * 28 : (i + 1) * 28, j * 28 : (j + 1) * 28] = img

    # Plot the interpolation grid
    plt.figure(figsize=(15, 8))
    plt.imshow(large_image, cmap="gray")
    plt.title("Latent Space Interpolation")
    plt.axis("off")
    plt.tight_layout()
    plt.show()
    plt.savefig(os.path.join(output_dir, "interpolations.png"))
