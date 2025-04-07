from dataclasses import dataclass

import torch
import torch.nn as nn


@dataclass
class ConvParams:
    """Holds parameters needed to instantiate a convolutional feature extractor
    and sample generator modules that can form the backbone of a VAE.

    Attributes:
    - hidden_dims: List of integers representing the number of channels in each convolutional layer.
    - conv_layer: Convolutional layer class (default: nn.Conv2d).
    - conv_transpose_layer: Transposed convolutional layer class (default: nn.ConvTranspose2d).
    - normalization: Normalization layer class (default: nn.BatchNorm2d).
    - activation: Activation function class (default: nn.ReLU).
    - output_scaling: Output scaling layer class (default: nn.Sigmoid).
    - kernel_size: Size of the convolutional kernel (default: 3).
    - stride: Stride of the convolutional layer (default: 2).
    - padding: Padding added to the input (default: 1).
    - use_skip_connections: Boolean indicating whether to use skip connections (default: True).
    """

    hidden_dims: list = (32, 64, 128)
    conv_layer: nn.Module = nn.Conv2d
    conv_transpose_layer: nn.Module = nn.ConvTranspose2d
    normalization: nn.Module = nn.BatchNorm2d
    activation: nn.Module = nn.ReLU
    output_scaling: nn.Module = nn.Sigmoid
    kernel_size: int = 3
    stride: int = 2
    padding: int = 1
    use_skip_connections: bool = True


class ConvBackbone(nn.Module):
    """Convolutional backbone for VAE, handling feature extraction and sample generation."""

    def __init__(self, sample_shape: list, params: ConvParams):
        super().__init__()
        self.params = params
        self.sample_shape = sample_shape
        self.shapes_per_layer = self._calculate_shape_per_layer()
        self._build_feature_extractor()
        self._build_sample_generator()

    def _build_feature_extractor(self):
        """Builds the convolutional feature extractor layers."""
        input_chan_dim = self.sample_shape[0]
        p = self.params
        self.feature_extractor_layers = nn.ModuleList()

        current_chan_dim = input_chan_dim
        for hidden_dim in p.hidden_dims:
            self.feature_extractor_layers.append(
                nn.Sequential(
                    p.conv_layer(
                        current_chan_dim,
                        hidden_dim,
                        kernel_size=p.kernel_size,
                        stride=p.stride,
                        padding=p.padding,
                    ),
                    p.normalization(hidden_dim),
                    p.activation(),
                )
            )
            current_chan_dim = hidden_dim

        self.flatten = nn.Flatten()

    def _build_sample_generator(self):
        """Builds the convolutional sample generator layers."""
        p = self.params
        self.sample_generator_layers = nn.ModuleList()

        # Unflatten layer
        self.unflatten = nn.Unflatten(1, self.shapes_per_layer[-1])

        # Upsampling layers
        for i in range(len(p.hidden_dims) - 1):
            chan_dim_before_conv, *spatial_dims_before_conv = self.shapes_per_layer[
                -(i + 1)
            ]
            chan_dim_after_conv, *spatial_dims_after_conv = self.shapes_per_layer[
                -(i + 2)
            ]

            in_channels = chan_dim_before_conv
            if p.use_skip_connections:
                in_channels *= 2

            output_padding = self._calculate_output_padding(
                spatial_dims_before_conv,
                spatial_dims_after_conv,
                p.kernel_size,
                p.stride,
                p.padding,
            )

            self.sample_generator_layers.append(
                nn.Sequential(
                    p.conv_transpose_layer(
                        in_channels,
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

        # Final upsampling layer to match input sample shape
        chan_dim_before_conv, *spatial_dims_before_conv = self.shapes_per_layer[1]
        chan_dim_after_conv, *spatial_dims_after_conv = self.shapes_per_layer[0]

        in_channels = chan_dim_before_conv
        if p.use_skip_connections:
            in_channels *= 2

        output_padding = self._calculate_output_padding(
            spatial_dims_before_conv,
            spatial_dims_after_conv,
            p.kernel_size,
            p.stride,
            p.padding,
        )

        self.sample_generator_layers.append(
            nn.Sequential(
                p.conv_transpose_layer(
                    in_channels,
                    chan_dim_after_conv,
                    kernel_size=p.kernel_size,
                    stride=p.stride,
                    padding=p.padding,
                    output_padding=output_padding,
                ),
            )
        )

        # Final output scaling
        self.output_scaling_layer = p.output_scaling()

    def extract_features(self, x: torch.Tensor) -> tuple[torch.Tensor, list | None]:
        """Forward pass through the feature extractor."""
        feature_maps = [] if self.params.use_skip_connections else None
        for layer in self.feature_extractor_layers:
            x = layer(x)
            if self.params.use_skip_connections:
                feature_maps.append(x)

        x = self.flatten(x)
        return x, feature_maps[::-1] if feature_maps else None

    def generate_sample(
        self, x: torch.Tensor, feature_maps: list | None
    ) -> torch.Tensor:
        """Forward pass through the sample generator."""
        x = self.unflatten(x)

        for i, layer in enumerate(self.sample_generator_layers):
            if self.params.use_skip_connections and feature_maps is not None and i > 0:
                skip_conn = feature_maps[i]
                x = torch.cat([x, skip_conn], dim=1)
            x = layer(x)

        x = self.output_scaling_layer(x)
        return x

    def _calculate_shape_per_layer(self):
        """Calculates and stores the spatial shapes from the encoding process."""
        kernel_size = self.params.kernel_size
        stride = self.params.stride
        padding = self.params.padding
        hidden_dims = self.params.hidden_dims
        chan_dim, *spatial_dims = self.sample_shape
        shapes_per_layer = [self.sample_shape]

        for i in range(0, len(hidden_dims)):
            chan_dim = hidden_dims[i]
            spatial_dims = self._calculate_spatial_dims_after_conv_layer(
                spatial_dims, kernel_size, stride, padding
            )
            shapes_per_layer.append((chan_dim, *spatial_dims))

        return shapes_per_layer

    def _calculate_spatial_dims_after_conv_layer(
        self, spatial_dims, kernel_size, stride, padding
    ):
        """
        Calculate new spatial dimensions after applying a convolutional layer.
        """
        return tuple(
            (dim - kernel_size + 2 * padding) // stride + 1 for dim in spatial_dims
        )

    def _calculate_output_padding(
        self, prev_dims, target_dims, kernel_size, stride, padding
    ):
        """
        Calculate the output padding needed for transposed convolutional layers.
        """
        return tuple(
            max(0, target - ((prev - 1) * stride - 2 * padding + kernel_size))
            for prev, target in zip(prev_dims, target_dims)
        )


def _assemble_conv_backbone(
    sample_shape: list,
    params: ConvParams,
) -> ConvBackbone:
    """Assemble a convolutional feature extractor and sample generator that can form
    the backbone of a variational auto-encoder.

    Args:
        sample_shape (list): Shape of the input samples (excluding batch dimension).
        params (ConvParams): Parameters specific to the convolution-based networks.

    Returns:
        ConvBackbone: An instance of the convolutional backbone module.
    """
    return ConvBackbone(sample_shape, params)
