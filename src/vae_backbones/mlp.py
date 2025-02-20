import torch.nn as nn
from dataclasses import dataclass
import math


@dataclass
class MLPParams:
    """Holds parameters needed to instantiate MLP-based feature extractor
    and sample generator modules that can form the backbone of a VAE."""

    hidden_dims: tuple = (512, 256, 128, 64)
    activation: nn.Module = nn.SiLU()
    final_activation: nn.Module = nn.Sigmoid()


def _assemble_mlp_backbone(
    sample_shape: list,
    params: MLPParams,
):
    """Assemble a MLP-based feature extractor and sample generator that can form
    the backbone of a variational auto-encoder.

    The feature extractor maps an input sample to a feature vector, while the sample generator
    reconstructs a sample from a feature vector. The feature extractor is the backbone of the
    encoder within a variational auto-encoder, while the sample generator is the backbone of the
    decoder.

    Args:
        sample_shape (list): Shape of the input samples (excluding batch dimension).
        params (MLPParams): Parameters specific to the MLP-based networks.
    """
    feature_extractor = _assemble_mlp_feature_extractor(sample_shape, params)
    sample_generator = _assemble_mlp_sample_generator(sample_shape, params)

    return feature_extractor, sample_generator


def _assemble_mlp_feature_extractor(sample_shape, params):
    """Multi-layer perceptron based feature extractor that maps an input sample to a
    feature vector.

    The feature extractor is the backbone of the encoder within a
    variational auto-encoder. This variant flattens the input sample and then applies
    multiple layers, each following the sequence: fully-connected layer, activation.

    Afterwards the input may still have spatial dimensions and therefore
    it is flattened before being passed through a fully-connected layer
    that maps the features to latent space.

    Note that the length of the feature vector is dependent on the input shape
    while the latent space dimension is not.

    """
    p = params
    extractor_modules = []
    numel_sample = math.prod(sample_shape)

    # Build the MLP network:
    # Flatten the input, then apply a series of linear layers with activations
    for i, hidden_dim in enumerate(p.hidden_dims):
        if i == 0:
            extractor_modules.append(nn.Flatten())  # Remove spatial dimensions, if any
            extractor_modules.append(nn.Linear(numel_sample, hidden_dim))
        else:
            extractor_modules.append(params.activation)
            extractor_modules.append(nn.Linear(params.hidden_dims[i - 1], hidden_dim))

    return nn.Sequential(*extractor_modules)


def _assemble_mlp_sample_generator(sample_shape, params):
    """Convolutional generator that reconstructs a sample from a feature vector.

    The sample generator is the backbone of the decoder within a
    variational auto-encoder. This variant first passes the input feature vector
    through multiple layers, each following the sequence:
    fully-connected layer, activation.

    The final activation function may be different and is used map the values to a desired range.
    At the end the tensor is reshaped to the original sample shape.
    """
    p = params
    generator_modules = []
    numel_sample = math.prod(sample_shape)

    # Build the MLP network:
    # Flatten the input, then apply a series of linear layers with activations
    for i in range(len(p.hidden_dims)):
        if i < (len(p.hidden_dims) - 1):
            generator_modules.append(
                nn.Linear(p.hidden_dims[-(i + 1)], p.hidden_dims[-(i + 2)])
            )
            generator_modules.append(p.activation)
        else:
            generator_modules.append(nn.Linear(p.hidden_dims[0], numel_sample))
            generator_modules.append(p.final_activation)
            generator_modules.append(nn.Unflatten(1, sample_shape))

    return nn.Sequential(*generator_modules)
