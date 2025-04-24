from dataclasses import dataclass

import torch
import torch.nn as nn

from .abstract_backbone import AbstractBackbone


@dataclass
class ConvParams:
    """Configuration parameters for building a Convolutional Neural Network (CNN) backbone.

    This dataclass holds all the necessary parameters to define the architecture
    of both the feature extractor (encoder part) and the sample generator (decoder part)
    of a CNN-based VAE backbone. It allows for flexible configuration, including
    support for both 2D and 3D convolutions.

    Attributes:
        hidden_dims (list): A list of integers specifying the number of output channels
            for each convolutional layer in the feature extractor. The reverse sequence
            (excluding the last element) implicitly defines the input channels for the
            transposed convolutional layers in the sample generator.
            Example: `[32, 64, 128]` means 3 conv layers with 32, 64, 128 output channels.
        conv_layer (nn.Module): The PyTorch convolutional layer class to use (e.g., `nn.Conv2d`
            for 2D images, `nn.Conv3d` for 3D volumes like MRI). Default: `nn.Conv2d`.
        conv_transpose_layer (nn.Module): The PyTorch transposed convolutional layer class
            to use for upsampling in the generator (e.g., `nn.ConvTranspose2d`,
            `nn.ConvTranspose3d`). Must correspond to `conv_layer`. Default: `nn.ConvTranspose2d`.
        normalization (nn.Module): The PyTorch normalization layer class to use after
            convolution operations (e.g., `nn.BatchNorm2d`, `nn.BatchNorm3d`). Must
            correspond to the dimensionality of `conv_layer`. Default: `nn.BatchNorm2d`.
        activation (nn.Module): The PyTorch activation function class to use after
            normalization (e.g., `nn.ReLU`, `nn.SiLU`, `nn.LeakyReLU`). Default: `nn.ReLU`.
        output_scaling (nn.Module): The activation function applied to the final output
            of the sample generator to scale the reconstructed sample to the desired range
            (e.g., `nn.Sigmoid` for [0, 1], `nn.Tanh` for [-1, 1], or `nn.Identity` for no scaling).
            Default: `nn.Sigmoid`.
        kernel_size (int): The size of the convolutional kernel (filter). Assumed to be square/cubic.
            Default: 3.
        stride (int): The stride of the convolution operation, controlling the downsampling factor.
            Default: 2.
        padding (int): The amount of padding added to the input before convolution. Default: 1.
        use_skip_connections (bool): If True, enables U-Net style skip connections by
            concatenating corresponding feature maps from the extractor to the generator layers.
            This helps preserve finer details during reconstruction. Default: True.
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
    use_residual_blocks: bool = False


class ConvBlock(nn.Module):
    """
    Implements a Convolutional Block that can optionally function as a Residual Block.

    Args:
        in_channels (int): Number of input channels.
        out_channels (int): Number of output channels.
        params (ConvParams): Configuration parameters for the convolutional block.
        transpose (bool): If True, uses transposed convolution (upsampling).
        output_padding (tuple, optional): Output padding for transposed convolution to ensure
            the output shape matches the target shape. Default: None.
    """

    expansion = 1  # For basic blocks; ResNet Bottleneck blocks use expansion=4

    def __init__(
        self, in_channels, out_channels, params, transpose, output_padding=None
    ):
        super(ConvBlock, self).__init__()
        self.use_residual = params.use_residual_blocks

        conv_layer = params.conv_transpose_layer if transpose else params.conv_layer
        conv_kwargs = {
            "kernel_size": params.kernel_size,
            "stride": params.stride,
            "padding": params.padding,
            "bias": False,  # Typically False when using BatchNorm
        }
        if transpose:
            conv_kwargs["output_padding"] = (
                output_padding if output_padding is not None else 0
            )  # Add output_padding if transpose

        # --- Main Path (Always Present) ---
        self.conv = conv_layer(
            in_channels,
            out_channels,
            **conv_kwargs,  # Use keyword arguments
        )
        self.normalization = params.normalization(out_channels)
        self.activation = params.activation()  # First ReLU activation

        # --- Shortcut Path (Only if use_residual is True) ---
        self.downsample = None
        # Define downsample layer *only* if residual connection is used AND needed
        if self.use_residual and (
            params.stride != 1 or in_channels != out_channels * self.expansion
        ):
            # Adjust shortcut for transpose convolution if needed (might require different kernel/stride)
            shortcut_kwargs = {
                "kernel_size": 1,
                "stride": params.stride,
                "bias": False,
            }
            if transpose:
                # Transposed shortcut might need adjustment, e.g., matching output_padding
                # For kernel_size=1, output_padding might not be directly applicable in the same way.
                # This part might need careful review depending on the exact residual architecture desired with transpose.
                # For now, assume standard ConvTranspose for shortcut if needed.
                shortcut_kwargs["output_padding"] = (
                    output_padding if output_padding is not None else 0
                )

            self.downsample = nn.Sequential(
                conv_layer(  # Use the same layer type (conv or transpose)
                    in_channels, out_channels * self.expansion, **shortcut_kwargs
                ),
                params.normalization(out_channels * self.expansion),
            )

        self.final_relu = params.activation()

    def forward(self, x):
        # Store the input only if using residual connection
        identity = x if self.use_residual else None

        # --- Main Path ---
        out = self.conv(x)
        out = self.normalization(out)
        out = self.activation(out)  # Apply first ReLU here

        if self.use_residual:
            if self.downsample is not None:
                identity = self.downsample(identity)
            out = out + identity

        out = self.activation(out)

        return out


class ConvBackbone(AbstractBackbone):
    def __init__(self, sample_shape: tuple, params: ConvParams):
        """Initializes and builds the Convolutional Backbone.

        This constructor sets up the entire CNN backbone, including both the
        feature extraction (encoder) path and the sample generation (decoder) path,
        based on the provided sample shape and configuration parameters.

        Args:
            sample_shape (tuple): The shape of a single input sample, excluding the batch
                dimension. Expected format: (channels, *spatial_dims), e.g.,
                (C, H, W) for 2D or (C, D, H, W) for 3D.
            params (ConvParams): An instance of `ConvParams` containing the configuration
                for the convolutional layers, normalization, activation, etc.
        """
        super().__init__()
        self.params = params
        self.sample_shape = sample_shape
        # Pre-calculate the expected shape of feature maps at each layer/resolution level.
        # This is needed for setting up the generator (decoder) correctly, especially for skip connections.
        self.shapes_per_layer = self._calculate_shape_per_layer()

        # Build the network components
        self._build_feature_extractor()  # Encoder part
        self._build_sample_generator()  # Decoder part

    def _build_feature_extractor(self):
        """Builds the convolutional feature extractor (encoder) layers."""
        # Input channels are the channels of the sample (e.g., 1 for grayscale, 3 for RGB, 4 for MRI modalities)
        chan_dim_before_conv = self.sample_shape[0]
        params = self.params
        self.feature_extractor_layers = (
            nn.ModuleList()
        )  # Use ModuleList to store layers

        # Sequentially add convolutional blocks (Conv -> Norm -> Activation)
        for i in range(len(params.hidden_dims)):
            chan_dim_after_conv = params.hidden_dims[
                i
            ]  # Output channels for this layer

            # Create a block: Convolution -> Normalization -> Activation
            # with optional residual connections
            conv_block = ConvBlock(
                chan_dim_before_conv, chan_dim_after_conv, params, transpose=False
            )

            self.feature_extractor_layers.append(conv_block)

            # Update the input channel dimension for the next layer
            chan_dim_before_conv = chan_dim_after_conv

        # Add a final flattening layer to convert the spatial feature map to a 1D vector
        self.flatten = nn.Flatten()

    def _build_sample_generator(self):
        """Builds the convolutional sample generator (decoder) layers."""
        p = self.params
        self.sample_generator_layers = nn.ModuleList()

        # The input to the generator is a flat feature vector.
        # First, unflatten it back into a spatial shape corresponding to the
        # output of the last feature extractor layer.
        # `self.shapes_per_layer[-1]` holds the (channels, *spatial_dims) of the smallest feature map.
        self.unflatten = nn.Unflatten(dim=1, unflattened_size=self.shapes_per_layer[-1])

        # Upsampling layers
        # Sequentially add transposed convolutional blocks (ConvTranspose -> Norm -> Activation)
        # We iterate up to len(hidden_dims) - 1 because the final layer is handled separately.
        for i in range(len(p.hidden_dims) - 1):
            # Determine input and output channels for this transposed conv layer
            # We use the pre-calculated shapes from the encoder in reverse order.
            chan_dim_after_conv = self.shapes_per_layer[-(i + 2)][
                0
            ]  # Target channels (from encoder)
            chan_dim_before_conv = self.shapes_per_layer[-(i + 1)][
                0
            ]  # Input channels from previous generator layer

            # If using skip connections, adjust the input channels for the transposed conv.
            # We concatenate the output of the previous generator layer with the corresponding
            # feature map from the encoder.
            # Note: Skip connection logic might need adjustment depending on U-Net variant.
            # Here, we assume simple concatenation. We skip the first layer (i=0) as
            # there's no preceding feature map from the generator to concatenate with yet.
            if p.use_skip_connections and i > 0:
                # The skip connection comes from the *output* of the corresponding encoder layer,
                # so its channel dimension is `shapes_per_layer[-(i + 1)][0]`.
                skip_channels = self.shapes_per_layer[-(i + 1)][0]
                chan_dim_before_conv += skip_channels  # Concatenated channels

            # Get the spatial dimensions before and after this upsampling step
            # from the pre-calculated shapes list.
            _, *spatial_dims_before_conv = self.shapes_per_layer[
                -(i + 1)
            ]  # Current spatial dims
            _, *spatial_dims_after_conv = self.shapes_per_layer[
                -(i + 2)
            ]  # Target spatial dims

            # Calculate necessary output padding to ensure the output spatial dimensions
            # exactly match the corresponding feature map dimensions from the encoder.
            output_padding = self._calculate_output_padding(
                prev_dims=spatial_dims_before_conv,
                target_dims=spatial_dims_after_conv,
                kernel_size=p.kernel_size,
                stride=p.stride,
                padding=p.padding,
            )

            # Create a block: Transposed Convolution -> Normalization -> Activation
            # upsample_block = nn.Sequential(
            #     p.conv_transpose_layer(  # e.g., nn.ConvTranspose2d/3d
            #         in_channels=chan_dim_before_conv,
            #         out_channels=chan_dim_after_conv,
            #         kernel_size=p.kernel_size,
            #         stride=p.stride,
            #         padding=p.padding,
            #         output_padding=output_padding,
            #     ),
            #     p.normalization(chan_dim_after_conv),  # e.g., nn.BatchNorm2d/3d
            #     p.activation(),  # e.g., nn.ReLU()
            # )

            upsample_block = ConvBlock(
                chan_dim_before_conv,
                chan_dim_after_conv,
                p,
                transpose=True,
                output_padding=output_padding,
            )
            self.sample_generator_layers.append(upsample_block)

        # --- Final Upsampling Layer ---
        # This layer reconstructs the sample to its original shape.
        chan_dim_before_conv, *spatial_dims_before_conv = self.shapes_per_layer[1]
        # Input channels for this layer come from the output of the last loop iteration.
        chan_dim_before_conv, *spatial_dims_before_conv = self.shapes_per_layer[1]
        # Target channels and spatial dims are those of the original input sample.
        chan_dim_after_conv, *spatial_dims_after_conv = self.shapes_per_layer[
            0
        ]  # Original sample shape

        # Handle skip connection for the first feature map (if applicable)
        if p.use_skip_connections:
            # Concatenate with the first feature map from the encoder (shapes_per_layer[1])
            skip_channels = self.shapes_per_layer[1][0]
            chan_dim_before_conv += skip_channels

        output_padding = self._calculate_output_padding(
            spatial_dims_before_conv,
            spatial_dims_after_conv,
            p.kernel_size,
            p.stride,
            p.padding,
        )

        self.sample_generator_layers.append(
            # Note: No Norm/Activation typically applied after the final conv layer,
            # only the output scaling.
            p.conv_transpose_layer(
                in_channels=chan_dim_before_conv,
                out_channels=chan_dim_after_conv,  # Target sample channels
                kernel_size=p.kernel_size,
                stride=p.stride,
                padding=p.padding,
                output_padding=output_padding,
            )
        )

        # --- Final Output Scaling ---
        # Apply the final activation (e.g., Sigmoid) to scale outputs to the desired range.
        self.output_scaling_layer = p.output_scaling()

    def extract_features(self, x: torch.Tensor) -> tuple[torch.Tensor, list | None]:
        """Performs the forward pass through the feature extractor (encoder).

        Args:
            x (torch.Tensor): Input tensor (batch of samples).

        Returns:
            tuple[torch.Tensor, list | None]:
                - features (torch.Tensor): Flattened feature tensor (batch_size, feature_dim).
                - feature_maps (list | None): List of intermediate feature maps (for skip
                  connections), ordered from smallest spatial resolution to largest.
                  Returns None if `use_skip_connections` is False.
        """
        feature_maps = [] if self.params.use_skip_connections else None
        # Pass input through each convolutional block
        for layer in self.feature_extractor_layers:
            x = layer(x)
            # Store intermediate feature maps if using skip connections
            if self.params.use_skip_connections:
                feature_maps.append(x)

        # Flatten the final feature map
        features = self.flatten(x)

        # Return flattened features and the reversed list of feature maps (if used)
        # Reversing ensures maps are ordered from smallest to largest spatial resolution,
        # matching the order needed by the generator.
        return features, feature_maps[::-1] if feature_maps else None

    def generate_sample(
        self, features: torch.Tensor, feature_maps: list | None
    ) -> torch.Tensor:
        """Performs the forward pass through the sample generator (decoder).

        Args:
            features (torch.Tensor): Input feature tensor (batch_size, feature_dim).
            feature_maps (list | None): List of intermediate feature maps from the
              encoder for skip connections (ordered smallest to largest spatial size),
              or None if skip connections are not used.

        Returns:
            torch.Tensor: Reconstructed sample tensor.
        """
        x = self.unflatten(features)

        for i, layer in enumerate(self.sample_generator_layers):
            # Handle skip connections (concatenation) before passing through the layer.
            # We skip the first layer (i=0) because the input `x` is already the unflattened
            # feature vector corresponding to the smallest spatial size.
            # `feature_maps` are ordered smallest to largest spatial size.
            if self.params.use_skip_connections and feature_maps is not None and i > 0:
                skip_conn = feature_maps[i]  # Get corresponding feature map
                # Concatenate along the channel dimension (dim=1)
                x = torch.cat([x, skip_conn], dim=1)
            x = layer(x)

        # Apply final scaling activation
        recon_x = self.output_scaling_layer(x)
        return recon_x

    def _calculate_shape_per_layer(self):
        """Calculates the expected (channels, *spatial_dims) shape after each encoder layer."""
        kernel_size = self.params.kernel_size
        stride = self.params.stride
        padding = self.params.padding
        hidden_dims = self.params.hidden_dims
        chan_dim, *spatial_dims = self.sample_shape
        shapes_per_layer = [self.sample_shape]  # Start with original sample shape

        # Calculate shape after each convolutional layer
        for i in range(len(hidden_dims)):
            # Output channels for this layer
            chan_dim = hidden_dims[i]
            # Calculate spatial dimensions after this layer's convolution
            spatial_dims = self._calculate_spatial_dims_after_conv_layer(
                spatial_dims, kernel_size, stride, padding
            )
            # Store the shape (channels, *spatial_dims)
            shapes_per_layer.append((chan_dim, *spatial_dims))

        return shapes_per_layer  # List of shapes, including input shape at index 0

    def _calculate_spatial_dims_after_conv_layer(
        self, spatial_dims, kernel_size, stride, padding
    ):
        """Calculate spatial dimensions after a standard convolution operation.

        Uses the standard formula: output_dim = floor((input_dim - kernel + 2*padding) / stride) + 1

        Args:
            spatial_dims (tuple): Input spatial dimensions (e.g., (H, W) or (D, H, W)).
            kernel_size (int): Size of the convolution kernel.
            stride (int): Stride of the convolution.
            padding (int): Padding applied.

        Returns:
            tuple: Output spatial dimensions.
        """
        # Apply the formula for each spatial dimension
        return tuple(
            (dim + 2 * padding - kernel_size) // stride + 1 for dim in spatial_dims
        )

    def _calculate_output_padding(
        self, prev_dims, target_dims, kernel_size, stride, padding
    ):
        """Calculate the `output_padding` needed for a transposed convolution layer
        to achieve a target output spatial dimension.

        The formula for transposed convolution output size is:
        output_dim = (input_dim - 1) * stride - 2*padding + kernel_size + output_padding

        We rearrange this to solve for `output_padding` needed to hit `target_dims`.
        output_padding = target_dim - [(input_dim - 1) * stride - 2*padding + kernel_size]

        Args:
            prev_dims (tuple): Spatial dimensions of the input to the transposed conv layer.
            target_dims (tuple): Desired spatial dimensions of the output.
            kernel_size (int): Kernel size of the transposed conv layer.
            stride (int): Stride of the transposed conv layer.
            padding (int): Padding of the transposed conv layer.

        Returns:
            tuple: The calculated `output_padding` for each spatial dimension.
        """
        # Calculate required output padding for each dimension
        output_padding = tuple(
            target - ((prev - 1) * stride - 2 * padding + kernel_size)
            for prev, target in zip(prev_dims, target_dims)
        )
        # Ensure output padding is not negative (can happen with certain configurations)
        return tuple(max(0, op) for op in output_padding)
