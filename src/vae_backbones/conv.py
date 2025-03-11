import torch.nn as nn
from dataclasses import dataclass


@dataclass
class ConvParams:
    """Holds parameters needed to instantiate convolutional feature extractor
    and sample generator modules that can form the backbone of a VAE."""

    hidden_dims: list = (32, 64, 128, 256, 512)
    conv_layer: nn.Module = nn.Conv2d
    conv_transpose_layer: nn.Module = nn.ConvTranspose2d
    normalization: nn.Module = nn.BatchNorm2d
    activation: nn.Module = nn.LeakyReLU
    output_scaling: nn.Module = nn.Sigmoid
    kernel_size: int = 3
    stride: int = 2
    padding: int = 1


def _assemble_conv_backbone(
    sample_shape: list,
    params: ConvParams,
):
    """Assemble a convolutional feature extractor and sample generator that can form
    the backbone of a variational auto-encoder.

    Args:
        sample_shape (list): Shape of the input samples (excluding batch dimension).
        params (ConvParams): Parameters specific to the convolution-based networks.
    """
    feature_extractor = _assemble_conv_feature_extractor(sample_shape, params)
    sample_generator = _assemble_conv_sample_generator(sample_shape, params)

    return feature_extractor, sample_generator


def _assemble_conv_feature_extractor(sample_shape, params):
    """Convolutional feature extractor that maps an input sample to a
    feature vector.

    The feature extractor is the backbone of the encoder within a
    variational auto-encoder. This variant consists of multiple layers, each following
    the sequence: convolution, normalization, and activation.
    Afterwards the input may still have spatial dimensions and therefore
    it is flattened before being passed through a fully-connected layer
    that maps the features to latent space.

    Note that the length of the feature vector is dependent on the input shape
    while the latent space dimension is not.

    """
    input_chan_dim = sample_shape[0]
    p = params

    extractor_modules = []

    for i, hidden_dim in enumerate(p.hidden_dims):
        chan_dim_before_conv = input_chan_dim if i == 0 else p.hidden_dims[i - 1]
        chan_dim_after_conv = hidden_dim

        extractor_modules.append(
            nn.Sequential(
                p.conv_layer(
                    chan_dim_before_conv,
                    chan_dim_after_conv,
                    kernel_size=p.kernel_size,
                    stride=p.stride,
                    padding=p.padding,
                ),
                p.normalization(chan_dim_after_conv),
                p.activation(),
            )
        )

    # After the above layers, the input may have spatial dimensions
    # Before mapping to latent space, the feature values must be flattened
    extractor_modules.append(nn.Flatten())

    return nn.Sequential(*extractor_modules)


def _assemble_conv_sample_generator(sample_shape, params):
    """Convolutional generator that reconstructs a sample from a feature vector.

    The sample generator is the backbone of the decoder within a
    variational auto-encoder. This variant first reshapes the input feature vector
    to a tensor with (downsampled) spatial dimensions. Afterwards it passed
    the tensor through multiple layers, each following the sequence:
    transposed convolution, normalization, and activation.

    A final activation function is applied to map the values to a desired range.
    """
    p = params
    generator_modules = []

    # Calculate the shape of the input sample for each layer of the feature extractor
    # This information is needed to properly upsample back to the same spatial resolution
    shape_per_layer = _calculate_shape_per_layer(sample_shape, params)

    # Reshape the input feature vector to add back spatial dimensions
    generator_modules.append(nn.Unflatten(1, shape_per_layer[-1]))

    # Upsampling, normalization and activations
    for i in range(len(p.hidden_dims) - 1):
        chan_dim_before_conv, *spatial_dims_before_conv = shape_per_layer[-(i + 1)]
        chan_dim_after_conv, *spatial_dims_after_conv = shape_per_layer[-(i + 2)]

        output_padding = _calculate_output_padding(
            spatial_dims_before_conv,
            spatial_dims_after_conv,
            p.kernel_size,
            p.stride,
            p.padding,
        )

        generator_modules.append(
            nn.Sequential(
                p.conv_transpose_layer(
                    chan_dim_before_conv,
                    chan_dim_after_conv,
                    kernel_size=p.kernel_size,
                    stride=p.stride,
                    padding=p.padding,
                    output_padding=output_padding,
                ),
                p.normalization(chan_dim_after_conv),
                p.activation(),
            )
        )

    # Last upsampling layer
    chan_dim_before_conv, *spatial_dims_before_conv = shape_per_layer[1]
    chan_dim_after_conv, *spatial_dims_after_conv = shape_per_layer[0]
    output_padding = _calculate_output_padding(
        spatial_dims_before_conv,
        spatial_dims_after_conv,
        p.kernel_size,
        p.stride,
        p.padding,
    )

    # Why are the chan dims the same here
    generator_modules.append(
        nn.Sequential(
            p.conv_transpose_layer(
                p.hidden_dims[0],
                sample_shape[0],
                kernel_size=p.kernel_size,
                stride=p.stride,
                padding=p.padding,
                output_padding=output_padding,
            ),
        )
    )

    # # TODO: Check the purpose of this refinement layer
    # generator_modules.append(
    #     nn.Sequential(
    #         p.conv_layer(
    #             p.hidden_dims[0],
    #             sample_shape[0],
    #             kernel_size=p.kernel_size,
    #             stride=1,
    #             padding=p.padding,
    #         ),
    #     )
    # )

    # Scale output (for example to be between 0 and 1 if inputs have been scaled in such a way as well)
    # generator_modules.append(p.output_scaling())

    return nn.Sequential(*generator_modules)


def _calculate_shape_per_layer(initial_shape: tuple, params: ConvParams):
    """Calculates and stores the spatial shapes from the encoding process."""
    kernel_size = params.kernel_size
    stride = params.stride
    padding = params.padding
    hidden_dims = params.hidden_dims
    chan_dim, *spatial_dims = initial_shape
    shapes_per_layer = [initial_shape]

    for i in range(0, len(hidden_dims)):
        # Determine chan dim after convolution
        chan_dim = hidden_dims[i]
        # Determine spatial dims after convolution
        spatial_dims = _calculate_spatial_dims_after_conv_layer(
            spatial_dims, kernel_size, stride, padding
        )
        shapes_per_layer.append((chan_dim, *spatial_dims))

    return shapes_per_layer


def _calculate_spatial_dims_after_conv_layer(
    spatial_dims, kernel_size, stride, padding
):
    """
    Calculate new spatial dimensions after applying a convolutional layer.

    Args:
        spatial_dims (tuple): Current spatial dimensions.
        kernel_size (int): Kernel size of the convolutional layer.
        stride (int): Stride of the convolutional layer.
        padding (int): Padding of the convolutional layer.

    Returns:
        tuple: New spatial dimensions.
    """
    return tuple(
        (dim - kernel_size + 2 * padding) // stride + 1 for dim in spatial_dims
    )


def _calculate_output_padding(prev_dims, target_dims, kernel_size, stride, padding):
    """
    Calculate the output padding needed for transposed convolutional layers.

    The upsampling layers should return tensors that have the same shapes as
    during downsampling. To achieve this, the output_padding parameter of the
    transposed convolution needs to be adjusted.

    Args:
        prev_dims (tuple): Previous spatial dimensions.
        target_dims (tuple): Target spatial dimensions.
        kernel_size (int): Kernel size of the transposed convolutional layer.
        stride (int): Stride of the transposed convolutional layer.
        padding (int): Padding of the transposed convolutional layer.

    Returns:
        tuple: Output padding needed.
    """
    # return tuple(
    #     target - (prev * stride - 2 * padding + kernel_size)
    #     for prev, target in zip(prev_dims, target_dims)
    # )
    return tuple(
        max(0, target - ((prev - 1) * stride - 2 * padding + kernel_size))
        for prev, target in zip(prev_dims, target_dims)
    )
