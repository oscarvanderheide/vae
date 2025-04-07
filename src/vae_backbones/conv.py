import torch.nn as nn
from dataclasses import dataclass


@dataclass
class ConvParams:
    """Holds parameters needed to instantiate convolutional feature extractor
    and sample generator modules that can form the backbone of a VAE."""

    hidden_dims: list = (32, 64, 128)
    conv_layer: nn.Module = nn.Conv2d
    conv_transpose_layer: nn.Module = nn.ConvTranspose2d
    normalization: nn.Module = nn.BatchNorm2d
    activation: nn.Module = nn.ReLU
    output_scaling: nn.Module = nn.Sigmoid
    kernel_size: int = 3
    stride: int = 2
    padding: int = 1


def _assemble_conv_backbone(sample_shape, params):
    """Assemble a convolutional feature extractor and sample generator."""
    feature_extractor, feature_shapes = _assemble_conv_feature_extractor(sample_shape, params)
    sample_generator = _assemble_conv_sample_generator(sample_shape, params, feature_shapes)
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

    extractor_modules = nn.ModuleList()  # Use ModuleList to store layers
    feature_shapes = []  # Store shapes of intermediate feature maps

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

        # Calculate and store the shape of the feature map
        if i == 0:
            prev_spatial_dims = sample_shape[1:]  # Extract spatial dimensions
        else:
            prev_spatial_dims = feature_shapes[-1][1:]  # Extract spatial dimensions

        new_spatial_dims = _calculate_spatial_dims_after_conv_layer(
            prev_spatial_dims, p.kernel_size, p.stride, p.padding
        )
        feature_shapes.append((chan_dim_after_conv, *new_spatial_dims))  # Add channel dim

    # Debugging: Print feature shapes
    # print(f"Feature shapes in encoder (conv.py): {feature_shapes}")

    # Add a flattening layer at the end
    extractor_modules.append(nn.Flatten())

    return extractor_modules, feature_shapes


def _assemble_conv_sample_generator(sample_shape, params, feature_shapes):
    """Convolutional generator that reconstructs a sample from a feature vector.

    Args:
        sample_shape (tuple): Shape of the input samples (excluding batch dimension).
        params (ConvParams): Parameters specific to the convolution-based networks.
        feature_shapes (list): Shapes of intermediate feature maps from the encoder.
    """
    p = params
    generator_modules = nn.ModuleList()

    # Reshape the input feature vector to add back spatial dimensions
    generator_modules.append(nn.Unflatten(1, feature_shapes[-1]))

    # Debugging:
    # print(f"Expected unflatten shape: {feature_shapes[-1]}")
    # print(f"Feature shapes in encoder (i.e., '[torch.Size([32, 14, 14]), torch.Size([64, 7, 7]), torch.Size([128, 4, 4])]'): {feature_shapes}")
    # print(f"Expected unflatten shape (i.e., ' (128, 4, 4)'): {feature_shapes[-1]}")

    # Upsampling, normalization, and activations with skip connections
    # for i in range(len(p.hidden_dims) - 1):
        # chan_dim_before_conv = feature_shapes[-(i + 1)][0]
        # chan_dim_after_conv = feature_shapes[-(i + 2)][0]

        # generator_modules.append(
        #     nn.Sequential(
        #         p.conv_transpose_layer(
        #             chan_dim_before_conv,
        #             chan_dim_after_conv,
        #             kernel_size=p.kernel_size,
        #             stride=p.stride,
        #             padding=p.padding,
        #             output_padding=1,  # Add output_padding to fix dimension mismatch
        #         ),
        #         p.normalization(chan_dim_after_conv),
        #         p.activation(),
        #     )
        # )
    # # Final upsampling layer
    # generator_modules.append(
    #     nn.Sequential(
    #         p.conv_transpose_layer(
    #             p.hidden_dims[0],
    #             sample_shape[0],
    #             kernel_size=p.kernel_size,
    #             stride=p.stride,
    #             padding=p.padding,
    #         ),
    #         p.output_scaling(),
    #     )
    # )

    for i in range(len(p.hidden_dims) - 1):
        chan_dim_before_conv = feature_shapes[-(i + 1)][0]
        chan_dim_after_conv = feature_shapes[-(i + 2)][0]
        prev_dims = feature_shapes[-(i + 1)][1:]  # Spatial dimensions of the previous layer
        target_dims = feature_shapes[-(i + 2)][1:]  # Target spatial dimensions
        output_padding = _calculate_output_padding(prev_dims, target_dims, p.kernel_size, p.stride, p.padding)

        generator_modules.append(
            nn.Sequential(
                p.conv_transpose_layer(
                    chan_dim_before_conv,
                    chan_dim_after_conv,
                    kernel_size=p.kernel_size,
                    stride=p.stride,
                    padding=p.padding,
                    output_padding=output_padding,  # Dynamically calculated output_padding
                ),
                p.normalization(chan_dim_after_conv),
                p.activation(),
            )
        )

    # Final upsampling layer
    generator_modules.append(
        nn.Sequential(
            p.conv_transpose_layer(
                p.hidden_dims[0],
                sample_shape[0],
                kernel_size=p.kernel_size,
                stride=p.stride,
                padding=p.padding,
                output_padding=_calculate_output_padding(
                    feature_shapes[0][1:], sample_shape[1:], p.kernel_size, p.stride, p.padding
                ),
            ),
            p.output_scaling(),
        )
    )

    return generator_modules


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
